"""End-to-end text-to-image generation (used by inference and train-time validation)."""

from pathlib import Path

import torch

from iris3b.flow.solver import FlowDPMSolver
from iris3b.models.dit import IrisDiT
from iris3b.models.step_cache import StepCache
from iris3b.text.base import TextEncoder, TextEncoding


def read_config(path: str | Path) -> dict:
    """The config dict alone, without reading any weights."""
    path = Path(path)
    if path.is_dir():
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path / "config.yaml"))
    return load_for_inference(path)[0]  # memory-mapped: the weights are not read


def load_for_inference(
    path: str | Path, dtype: torch.dtype | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    """``(config dict, weights)`` from a training checkpoint or an exported directory.

    A training ``.pth`` contributes its embedded config and its EMA weights,
    or its raw weights when it was trained without EMA. The checkpoint is
    memory-mapped, so its optimizer state is never read. An exported directory
    holds ``config.yaml`` and ``model.safetensors``.

    ``dtype`` converts tensor by tensor while reading, so loading the FP32
    release as BF16 peaks at the BF16 size (~6 GB) rather than FP32's ~12 GB.
    """
    path = Path(path)
    if path.is_dir():
        from omegaconf import OmegaConf
        from safetensors import safe_open

        raw = OmegaConf.to_container(OmegaConf.load(path / "config.yaml"))
        weights = {}
        with safe_open(str(path / "model.safetensors"), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118  (safe_open is not a mapping)
                tensor = f.get_tensor(key)
                weights[key] = tensor if dtype is None else tensor.to(dtype)
        return raw, weights
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if "config" not in payload:
        raise ValueError(f"{path} has no embedded config; it is not a training checkpoint")
    key = "state_dict_ema" if "state_dict_ema" in payload else "state_dict"
    weights = payload[key]
    if dtype is not None:
        weights = {k: v.to(dtype) for k, v in weights.items()}
    return payload["config"], weights


@torch.no_grad()
def load_model_streaming(
    model_cfg,
    checkpoint_dir: str | Path,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    quantize_keep=None,
    lora_path: str | Path | None = None,
    lora_strength: float = 1.0,
    flush_every_bytes: int = 1 << 29,
) -> tuple[IrisDiT, dict]:
    """Build an exported model on ``device`` one tensor at a time.

    Each checkpoint tensor is read straight onto the device, gets its LoRA delta
    (in fp32) if ``lora_path`` names an adapter, and is then either cast to
    ``dtype`` or, for Linears not accepted by ``quantize_keep``, quantized to
    int8. No full-precision copy of the model ever exists, and the device
    allocator's cache of freed staging tensors is flushed every
    ``flush_every_bytes``. ``quantize_keep=None`` disables quantization.
    Returns ``(model, adapter metadata or {})``.
    """
    from safetensors import safe_open
    from torch import nn

    from iris3b.devices import release_memory
    from iris3b.lora import read_metadata
    from iris3b.nn.modulation import rebind_shared_cores
    from iris3b.quant import Int8Linear

    device = torch.device(device)
    with torch.device("meta"):
        model = IrisDiT(model_cfg)
    if quantize_keep is not None:
        replaced: dict[int, nn.Module] = {}
        for name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                full = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and not quantize_keep(full):
                    shell = Int8Linear(child.in_features, child.out_features, child.bias is not None, device)
                    replaced[id(child)] = shell
                    setattr(module, child_name, shell)
        rebind_shared_cores(model, replaced)

    lora, meta, scale = {}, {}, 0.0
    if lora_path is not None:
        from safetensors.torch import load_file

        meta = read_metadata(lora_path)
        scale = lora_strength * meta["alpha"] / meta["rank"]
        lora = load_file(str(lora_path))

    since_flush = 0
    with safe_open(str(Path(checkpoint_dir) / "model.safetensors"), framework="pt", device=str(device)) as f:
        for key in f.keys():  # noqa: SIM118  (safe_open is not a mapping)
            owner_name, _, leaf = key.rpartition(".")  # "" for top-level parameters
            owner = model.get_submodule(owner_name)
            tensor = f.get_tensor(key)
            since_flush += tensor.numel() * tensor.element_size()
            if leaf == "weight" and f"{owner_name}.lora_down" in lora:
                down = lora.pop(f"{owner_name}.lora_down").to(device, torch.float32)
                up = lora.pop(f"{owner_name}.lora_up").to(device, torch.float32)
                tensor = tensor.float() + scale * (up @ down)
            if isinstance(owner, Int8Linear) and leaf == "weight":
                owner.load_weight(tensor)
            elif isinstance(owner, Int8Linear) and leaf == "bias":
                owner.bias = nn.Parameter(tensor.to(torch.bfloat16), requires_grad=False)
            elif leaf in owner._parameters:
                owner._parameters[leaf] = nn.Parameter(tensor.to(dtype), requires_grad=False)
            else:
                owner._buffers[leaf] = tensor
            del tensor
            if since_flush >= flush_every_bytes:
                release_memory(device)
                since_flush = 0
    if lora:
        raise KeyError(f"adapter has weights for layers the model lacks: {sorted(lora)[:4]}")
    if stale := [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]:
        raise RuntimeError(f"checkpoint left tensors unset: {stale[:5]}")
    release_memory(device)
    return model.eval(), meta


class PrecomputedText(TextEncoder):
    """Encodings computed up front, so the text encoder can be freed before the
    image model loads (the two do not fit together in 16 GB of unified memory)."""

    def __init__(self, rows: dict[str, TextEncoding], null: dict[str, TextEncoding], dim: int, max_length: int):
        self.dim, self.max_length = dim, max_length
        self.calls = self.overflow_rows = self.overflow_tokens = self.max_caption_tokens = 0
        self._rows, self._null = rows, null

    @classmethod
    def encode_with(
        cls, text_encoder: TextEncoder, prompts: list[str], negative_prompt: str = ""
    ) -> "PrecomputedText":
        rows = {}
        for prompt in dict.fromkeys(prompts):
            enc = text_encoder.encode([prompt])
            rows[prompt] = TextEncoding(enc.embeddings.cpu(), enc.mask.cpu())
        null = text_encoder.null(negative_prompt)
        nulls = {negative_prompt: TextEncoding(null.embeddings.cpu(), null.mask.cpu())}
        return cls(rows, nulls, text_encoder.dim, text_encoder.max_length)

    @classmethod
    def encode_in_subprocess(
        cls, text_cfg, prompts: list[str], negative_prompt: str, device: torch.device | str
    ) -> "PrecomputedText":
        """Encode in a child process, so every byte the encoder touched goes back
        to the OS when it exits; allocator caches (malloc, Metal) otherwise keep
        several GB of a freed encoder resident in this process."""
        import multiprocessing
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "encodings.pt"
            proc = multiprocessing.get_context("spawn").Process(
                target=_encode_worker, args=(text_cfg, prompts, negative_prompt, str(device), str(out))
            )
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"text encoder subprocess failed with exit code {proc.exitcode}")
            payload = torch.load(out, weights_only=False)
        return cls(payload["rows"], payload["null"], payload["dim"], payload["max_length"])

    def encode(self, prompts: list[str]) -> TextEncoding:
        rows = [self._rows[p] for p in prompts]
        return TextEncoding(torch.cat([r.embeddings for r in rows]), torch.cat([r.mask for r in rows]))

    def null(self, negative_prompt: str = "") -> TextEncoding:
        return self._null[negative_prompt]

    def to(self, device) -> "PrecomputedText":
        return self  # rows stay on the CPU; generate() moves each batch


def _encode_worker(text_cfg, prompts: list[str], negative_prompt: str, device: str, out: str) -> None:
    import iris3b.text  # noqa: F401  (registers encoders)
    from iris3b.registry import TEXT_ENCODERS

    encoder = TEXT_ENCODERS.build(text_cfg.name, text_cfg, device=device)
    pre = PrecomputedText.encode_with(encoder, prompts, negative_prompt)
    torch.save({"rows": pre._rows, "null": pre._null, "dim": pre.dim, "max_length": pre.max_length}, out)


@torch.no_grad()
def generate(
    model: IrisDiT,
    text_encoder: TextEncoder,
    prompts: list[str],
    height: int,
    width: int,
    steps: int = 100,
    order: int = 2,
    cfg_scale: float = 3.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    shift: float = 4.0,
    negative_prompt: str = "",
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
    noise: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
    prediction: str = "v",
    cache: StepCache | None = None,
) -> torch.Tensor:
    """Sample [B, C, H, W] images, clamped to [-1, 1].

    Integrator state stays in fp32; the network is evaluated in its own
    parameter dtype. The CFG unconditional is ``text_encoder.null`` of the
    negative prompt, which the encoder formats exactly like a positive prompt
    (an empty negative prompt is the dropout null used in training).

    ``cache`` carries cross-step reuse for this run; by default a fresh
    ``StepCache()`` reuses only the text adapter output, which is exact. Pass
    one with a trunk threshold to also skip trunk blocks (approximate).
    """
    model_dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=model_dtype)
    cond_mask = encoding.mask.to(device=device)
    uncond, cfg_mask = None, None
    if cfg_scale != 1.0:
        null = text_encoder.null(negative_prompt)
        uncond = null.embeddings.to(device=device, dtype=model_dtype).expand(
            len(prompts), *([-1] * (null.embeddings.ndim - 1))
        )
        null_mask = null.mask.to(device=device).expand(len(prompts), -1)
        # the solver's CFG batch is cat([uncond, cond]); the mask follows suit
        cfg_mask = torch.cat([null_mask, cond_mask], dim=0)

    if noise is None:
        noise = torch.randn(
            len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device
        )
    z = noise.to(device=device, dtype=torch.float32)
    if cache is None:
        cache = StepCache()

    def model_fn(x: torch.Tensor, t_model: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # the solver either passes `cond` itself or the doubled CFG batch
        out = model(x.to(model_dtype), t_model, y, y_mask=cond_mask if y is cond else cfg_mask, cache=cache)
        return out.x.float()

    solver = FlowDPMSolver(
        model_fn,
        num_timesteps=num_train_timesteps,
        cfg_scale=cfg_scale,
        cfg_interval=tuple(cfg_interval),
        prediction=prediction,
    )
    sample = solver.sample(z, cond, uncond, steps=steps, order=order, shift=shift)
    return sample.clamp(-1, 1)

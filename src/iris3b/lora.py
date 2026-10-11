"""Low-rank adapters on frozen Linear layers, with named adapter slots.

Several adapters share one frozen base model and ``use_adapter(model, name)``
selects which one is live (``None`` = the bare base model). Turbo distillation
keeps the student and the DMD fake-score critic as two adapters on a single
copy of the 3B weights, with the bare base acting as the teacher.

For inference an adapter file is folded into the plain weights
(``merge_lora_file``), so a distilled model runs at exactly the base cost.
"""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

# submodules of IrisDiT that receive adapters; the text adapter stays frozen
# because the prompt states it produces are shared by every timestep
DEFAULT_TARGETS = ("s_embedder", "t_embedder", "modulation_cores", "blocks", "pixel_blocks", "final_layer")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales: dict[str, float] = {}
        self.active: str | None = None

    def add_adapter(self, name: str, rank: int, alpha: float) -> None:
        down = nn.Linear(self.base.in_features, rank, bias=False)
        up = nn.Linear(rank, self.base.out_features, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)  # a fresh adapter is an exact no-op
        device = self.base.weight.device
        self.adapters[name] = nn.Sequential(down, up).to(device=device, dtype=torch.float32)
        self.scales[name] = alpha / rank

    def delta(self, name: str) -> torch.Tensor:
        down, up = self.adapters[name]
        return (up.weight @ down.weight) * self.scales[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active is None:
            return out
        return out + self.adapters[self.active](x) * self.scales[self.active]


def inject_lora(
    model: nn.Module, name: str, rank: int, alpha: float | None = None, targets=DEFAULT_TARGETS
) -> list[nn.Parameter]:
    """Add adapter ``name`` to every Linear under ``targets``; returns its parameters."""
    alpha = float(rank) if alpha is None else alpha
    params: list[nn.Parameter] = []

    def attach(layer: LoRALinear) -> None:
        if name not in layer.adapters:
            layer.add_adapter(name, rank, alpha)
            params.extend(layer.adapters[name].parameters())

    replaced: dict[int, nn.Module] = {}

    def walk(module: nn.Module) -> None:
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                attach(child)  # already wrapped: never descend into base/adapters
            elif isinstance(child, nn.Linear):
                wrapped = LoRALinear(child)
                replaced[id(child)] = wrapped
                setattr(module, child_name, wrapped)
                attach(wrapped)
            else:
                walk(child)

    for target in targets:
        root = model.get_submodule(target)
        if isinstance(root, nn.Linear | LoRALinear):
            raise ValueError(f"target '{target}' is itself a Linear; name its parent instead")
        walk(root)
    # shared adaLN cores are also held, unregistered, by every block
    from iris3b.nn.modulation import rebind_shared_cores

    rebind_shared_cores(model, replaced)
    return params


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


@contextmanager
def use_adapter(model: nn.Module, name: str | None):
    layers = lora_layers(model).values()
    previous = [layer.active for layer in layers]
    for layer in layers:
        layer.active = name
    try:
        yield
    finally:
        for layer, active in zip(layers, previous, strict=True):
            layer.active = active


def save_adapter(model: nn.Module, name: str, path: str | Path, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file

    tensors, rank, alpha = {}, None, None
    for prefix, layer in lora_layers(model).items():
        down, up = layer.adapters[name]
        tensors[f"{prefix}.lora_down"] = down.weight.detach().float().cpu().contiguous()
        tensors[f"{prefix}.lora_up"] = up.weight.detach().float().cpu().contiguous()
        rank, alpha = down.out_features, layer.scales[name] * down.out_features
    meta = {"rank": str(rank), "alpha": str(alpha), "extra": json.dumps(metadata or {})}
    save_file(tensors, str(path), metadata=meta)


def load_adapter(model: nn.Module, name: str, path: str | Path) -> None:
    """Load a saved adapter into slot ``name`` (injected beforehand with the same rank)."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    layers = lora_layers(model)
    for prefix, layer in layers.items():
        down, up = layer.adapters[name]
        down.weight.data.copy_(tensors[f"{prefix}.lora_down"])
        up.weight.data.copy_(tensors[f"{prefix}.lora_up"])
    expected = {f"{p}.lora_{s}" for p in layers for s in ("down", "up")}
    if missing := set(tensors) - expected:
        raise KeyError(f"adapter file has weights for layers the model lacks: {sorted(missing)[:5]}")


def resolve_adapter(spec: str | Path) -> Path:
    """Local path to an adapter file, given a path or a Hub id ``owner/repo[:name]``.

    A Hub repo carries a root ``config.json`` manifest (``{"default": name,
    "adapters": {name: {"file": ...}}}``), fetched first: it names the file and is
    also the file the Hub counts downloads by. ``name`` defaults to the
    manifest's ``default``.
    """
    path = Path(spec)
    if path.exists():
        return path
    repo, _, name = str(spec).partition(":")
    if repo.count("/") != 1 or repo.endswith(".safetensors"):
        raise FileNotFoundError(f"adapter {spec!r} is neither a file nor a Hub id 'owner/repo[:name]'")
    from huggingface_hub import hf_hub_download

    manifest = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    name = name or manifest["default"]
    if name not in manifest["adapters"]:
        raise KeyError(f"{repo} has no adapter {name!r} (available: {sorted(manifest['adapters'])})")
    return Path(hf_hub_download(repo, manifest["adapters"][name]["file"]))


def read_metadata(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return {"rank": int(meta["rank"]), "alpha": float(meta["alpha"]), **json.loads(meta.get("extra", "{}"))}


@torch.no_grad()
def merge_lora_file(model: nn.Module, path: str | Path, strength: float = 1.0) -> dict:
    """Fold an adapter file into a plain (uninjected) model's Linear weights."""
    from safetensors.torch import load_file

    meta = read_metadata(path)
    scale = strength * meta["alpha"] / meta["rank"]
    tensors = load_file(str(path))
    prefixes = {key.rsplit(".", 1)[0] for key in tensors}
    for prefix in prefixes:
        linear = model.get_submodule(prefix)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{prefix} is {type(linear).__name__}, expected Linear")
        down = tensors[f"{prefix}.lora_down"].to(linear.weight.device, torch.float32)
        up = tensors[f"{prefix}.lora_up"].to(linear.weight.device, torch.float32)
        merged = linear.weight.float() + scale * (up @ down)
        linear.weight.copy_(merged.to(linear.weight.dtype))
    return meta

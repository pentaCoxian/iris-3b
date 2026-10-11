"""Batch text-to-image inference.

Usage:
    python scripts/sample.py --checkpoint exported/iris-3b --prompt "a red fox" --prompt "a blue jay"
    python scripts/sample.py --checkpoint output/run/checkpoints/latest.pth --txt-file prompts.txt

``--checkpoint`` is either a directory written by ``scripts/export_checkpoint.py``
(``model.safetensors`` + ``config.yaml``) or a training ``.pth``, whose EMA
weights and embedded config are used. Prompts come from repeated --prompt flags
or a text file (one per line). Images are written as jpgs named after the
prompt plus a grid.png contact sheet.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iris3b.config import Config, inference_config
from iris3b.devices import default_device, inference_autocast, release_memory
from iris3b.distill import generate_turbo
from iris3b.lora import merge_lora_file, resolve_adapter
from iris3b.models.dit import IrisDiT
from iris3b.models.step_cache import StepCache
from iris3b.quant import quantize_linears
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import (
    PrecomputedText,
    generate,
    load_for_inference,
    load_model_streaming,
    read_config,
)


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def read_prompts(args: argparse.Namespace, cfg: Config) -> list[str]:
    if args.prompt:
        return list(args.prompt)
    if args.txt_file:
        lines = Path(args.txt_file).read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return list(cfg.train.validation_prompts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="exported directory or training .pth")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--prompt", action="append", default=[], help="repeatable prompt")
    parser.add_argument("--txt-file", default=None, help="prompt file, one per line")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None, help="default: sample.steps (100)")
    parser.add_argument("--order", type=int, default=None, help="DPM-Solver++ order, default: sample.order (2)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="default: sample.cfg_scale (3.0)")
    parser.add_argument(
        "--cfg-interval",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="apply CFG only for flow times strictly inside (LO, HI), t=1 is noise; "
        "outside it each step costs one forward instead of two (default: sample.cfg_interval)",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.0,
        help="reuse the trunk residual between steps while the first block's accumulated relative "
        "change stays below this (0 = off, exact; try 0.05-0.15)",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="adapter .safetensors or Hub id owner/repo[:name] to merge; a turbo adapter "
        "(scripts/distill_turbo.py) switches to its few-step sampler with guidance baked in, so "
        "--cfg-* and --cache-* do not apply",
    )
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--shift", type=float, default=None, help="flow shift (default: checkpoint flow.shift)")
    parser.add_argument("--outdir", default="output/samples")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    parser.add_argument(
        "--weights-dtype",
        choices=("auto", "bf16", "fp32"),
        default="auto",
        help="auto: bf16 on cuda/mps (half the memory of fp32 at the same speed under bf16 autocast), "
        "fp32 on cpu",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="images per batch (default: all prompts)")
    parser.add_argument(
        "--quantize",
        choices=("none", "int8"),
        default=None,
        help="int8: weight-only int8 for the image model's trunk and the text encoder - about half "
        "their memory, ~5%% slower (default: int8 under --preset mac16, otherwise none)",
    )
    parser.add_argument(
        "--preset",
        choices=("mac16",),
        default=None,
        help="mac16: fit a 16 GB Mac (~9 GB peak) - bf16 weights, prompts encoded and the text encoder "
        "freed before the image model loads, one image at a time. Pair it with a turbo --lora.",
    )
    args = parser.parse_args(argv)
    if args.lora:
        args.lora = str(resolve_adapter(args.lora))

    low_memory = args.preset == "mac16"
    if args.quantize is None:
        args.quantize = "int8" if low_memory else "none"
    cfg = inference_config(read_config(args.checkpoint), args.set)
    steps = args.steps if args.steps is not None else cfg.sample.steps
    order = args.order if args.order is not None else cfg.sample.order
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.sample.cfg_scale
    negative = args.negative_prompt if args.negative_prompt is not None else cfg.sample.negative_prompt
    shift = args.shift if args.shift is not None else cfg.flow.shift
    cfg_interval = tuple(args.cfg_interval) if args.cfg_interval is not None else tuple(cfg.sample.cfg_interval)
    device = torch.device(args.device) if args.device else default_device()
    if args.weights_dtype == "auto":
        bf16 = low_memory or device.type in ("cuda", "mps")
    else:
        bf16 = args.weights_dtype == "bf16"
    weights_dtype = torch.bfloat16 if bf16 else torch.float32

    prompts = read_prompts(args, cfg)
    if not prompts:
        parser.error("no prompts given (use --prompt or --txt-file)")
    batch_size = args.batch_size or (1 if low_memory else len(prompts))

    import iris3b.text  # noqa: F401  (registers encoders)

    if args.quantize == "int8":
        cfg.text_encoder.quantize = "int8"
    if low_memory:
        # Qwen and the image model never coexist: the prompts are encoded in a
        # child process, whose memory all returns to the OS when it exits
        text_encoder = PrecomputedText.encode_in_subprocess(cfg.text_encoder, prompts, negative, device)
    else:
        text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)

    # the pixel stage and the patch/timestep embedders are small; they stay bf16 under int8
    small = ("pixel_blocks", "pixel_embedder", "final_layer", "s_embedder", "t_embedder")
    keep = (lambda name: name.startswith(small)) if args.quantize == "int8" else None
    if Path(args.checkpoint).is_dir():
        # streamed onto the device tensor by tensor: no full fp32 or bf16 copy ever exists
        model, meta = load_model_streaming(
            cfg.model, args.checkpoint, device, weights_dtype, keep, args.lora, args.lora_strength
        )
    else:
        _, weights = load_for_inference(args.checkpoint, dtype=weights_dtype)
        # built on the meta device: the checkpoint tensors become the parameters
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        # merged on the CPU in fp32 per layer, before the move, so no device copy is doubled
        meta = merge_lora_file(model, args.lora, strength=args.lora_strength) if args.lora else {}
        if keep is not None:
            quantize_linears(model, keep=keep)
            release_memory("cpu")
        model = model.eval().to(device)  # BF16 autocast at inference either way
    turbo_steps = meta.get("turbo_steps")
    if turbo_steps is not None:
        steps = args.steps if args.steps is not None else turbo_steps
        shift = args.shift if args.shift is not None else meta["shift"]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    calls = skipped = 0
    batches = []
    start = time.perf_counter()
    for first in range(0, len(prompts), batch_size):
        chunk = prompts[first : first + batch_size]
        cache = StepCache(threshold=args.cache_threshold)  # one cache per sampling run
        with inference_autocast(device):
            batches.append(sample_batch(
                model, text_encoder, chunk, args, cfg, steps, order, cfg_scale, cfg_interval, shift,
                negative, generator, device, cache, turbo_steps,
            ).cpu())  # .cpu() waits for the device: CUDA and MPS run asynchronously
        calls, skipped = calls + cache.calls, skipped + cache.skipped
    images = torch.cat(batches)
    elapsed = time.perf_counter() - start
    skipped = f", trunk skipped on {skipped}/{calls} forwards" if args.cache_threshold > 0 else ""
    print(f"sampled {len(prompts)} image(s) in {elapsed:.1f}s on {device}{skipped}")

    from torchvision.utils import save_image

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        path = outdir / f"{i:03d}_{slugify(prompt)}.jpg"
        save_image(image, str(path), normalize=True, value_range=(-1, 1))
        print(f"wrote {path}")
    grid_path = outdir / "grid.png"
    save_image(
        images, str(grid_path), nrow=max(1, round(len(prompts) ** 0.5)), normalize=True, value_range=(-1, 1)
    )
    print(f"wrote {grid_path}")


def sample_batch(
    model, text_encoder, prompts, args, cfg, steps, order, cfg_scale, cfg_interval, shift, negative,
    generator, device, cache, turbo_steps,
) -> torch.Tensor:
    if turbo_steps is not None:
        return generate_turbo(
            model, text_encoder, prompts, height=args.height, width=args.width, steps=steps,
            shift=shift, generator=generator, device=device,
        )
    return generate(
        model,
        text_encoder,
        prompts,
        height=args.height,
        width=args.width,
        steps=steps,
        order=order,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        shift=shift,
        negative_prompt=negative,
        generator=generator,
        device=device,
        num_train_timesteps=cfg.flow.num_train_timesteps,
        prediction=cfg.flow.prediction,
        cache=cache,
    )


if __name__ == "__main__":
    main()

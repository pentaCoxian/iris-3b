"""Iris-3B generation speed demo: Fast (the 4-step turbo LoRA) vs. Balanced (training-free
30-step sampling with CFG interval and trunk cache) vs. the standard sampler.

    python demo/speed.py --lora output/turbo/latest/student.safetensors

Turbo runs on its own bf16 copy with the adapter merged, so it is timed at the
speed scripts/sample.py --lora runs at; the other modes run on the plain
weights (+6 GB in total). --live-lora keeps a single copy with the adapter
switched in per run instead, which saves the memory but costs ~10% per turbo
forward on MPS (~700 extra small kernels). Both models are warmed up at launch,
so the first click is not timed with one-off shader compilation. Each run
reports wall time, forward passes, time per forward and images per minute, and
the speed-up over the 100-step sampler, measured if it has run this session,
otherwise estimated from measured per-forward times.
"""

import argparse
import html
import time
from contextlib import nullcontext
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

import iris3b.text  # noqa: F401  (registers encoders)
from iris3b.config import inference_config
from iris3b.devices import default_device, inference_autocast
from iris3b.distill import STUDENT, generate_turbo
from iris3b.lora import inject_lora, load_adapter, read_metadata, use_adapter
from iris3b.models.step_cache import StepCache
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate, load_model_streaming, read_config

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/iris-3b"))
parser.add_argument("--lora", default=None, help="turbo adapter from scripts/distill_turbo.py")
parser.add_argument("--device", default=None)
parser.add_argument("--port", type=int, default=7861)
parser.add_argument("--share", action="store_true")
parser.add_argument(
    "--live-lora", action="store_true", help="one model copy, adapter switched per run (~10%% slower)"
)
args = parser.parse_args()

DEVICE = torch.device(args.device) if args.device else default_device()
cfg = inference_config(read_config(args.checkpoint), [])
model, _ = load_model_streaming(cfg.model, args.checkpoint, DEVICE, torch.bfloat16)
TURBO, turbo_model = None, None
if args.lora:
    meta = read_metadata(args.lora)
    TURBO = {"steps": meta["turbo_steps"], "shift": meta["shift"]}
    if args.live_lora:
        inject_lora(model, STUDENT, meta["rank"], meta["alpha"])
        load_adapter(model, STUDENT, args.lora)
        turbo_model = model
    else:
        turbo_model, _ = load_model_streaming(
            cfg.model, args.checkpoint, DEVICE, torch.bfloat16, lora_path=args.lora
        )
text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=DEVICE)


@torch.no_grad()
def warm_up() -> None:
    """One forward per model and batch shape at 1024²: Metal compiles its shaders and
    sizes its heaps on first use, which would otherwise land in the first timed run."""
    x = torch.randn(2, 3, 1024, 1024, device=DEVICE)
    t = torch.full((2,), 500.0, device=DEVICE)
    enc = text_encoder.encode(["warm up", "warm up"])
    y, mask = enc.embeddings.to(DEVICE, torch.bfloat16), enc.mask.to(DEVICE)
    with inference_autocast(DEVICE):
        for net, batch in [(model, 1), (model, 2), (turbo_model, 1)]:
            if net is None:
                continue
            ctx = use_adapter(net, STUDENT) if net is turbo_model and args.live_lora else nullcontext()
            with ctx:
                net(x[:batch], t[:batch], y[:batch], y_mask=mask[:batch])
    sync()


SIZES = {
    "1024 × 1024": (1024, 1024),
    "1152 × 896": (896, 1152),
    "896 × 1152": (1152, 896),
    "1344 × 768": (768, 1344),
    "768 × 1344": (1344, 768),
}
MODES = {
    "Fast": "4 steps, one forward per step",
    "Balanced": "30 steps, CFG on t < 0.9, trunk cache 0.1",
    "Standard": "100 DPM-Solver++ steps, CFG 3 on every step",
}
# seconds per forward, measured this session: {(pixels, batch rows): seconds}
FORWARD_S: dict[tuple[int, int], float] = {}
STANDARD_S: dict[int, float] = {}  # measured standard runs by pixel count


class ForwardTimer:
    """Counts model forwards and their batch rows (2 under CFG) through a hook."""

    def __init__(self, progress: gr.Progress, expected: int, label: str, net=None):
        self.rows, self.progress, self.expected, self.label = [], progress, expected, label
        self.net = model if net is None else net

    def __enter__(self):
        self.handle = self.net.register_forward_hook(self.hook)
        return self

    def hook(self, _module, inputs, _output):
        self.rows.append(inputs[0].shape[0])
        self.progress(
            (len(self.rows), self.expected), desc=f"{self.label}: forward {len(self.rows)}/{self.expected}"
        )

    def __exit__(self, *exc):
        self.handle.remove()


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def to_pil(image: torch.Tensor) -> Image.Image:
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(pixels))


def run_mode(mode: str, prompt: str, height: int, width: int, seed: int, progress: gr.Progress) -> dict:
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    sync()
    start = time.perf_counter()
    if mode == "Fast":
        if TURBO is None:
            raise gr.Error("Fast mode is not available: start the demo with --lora <adapter>.")
        with (
            ForwardTimer(progress, TURBO["steps"], mode, net=turbo_model) as timer,
            use_adapter(turbo_model, STUDENT) if args.live_lora else nullcontext(),
            inference_autocast(DEVICE),
        ):
            image = generate_turbo(
                turbo_model,
                text_encoder,
                [prompt],
                height,
                width,
                TURBO["steps"],
                TURBO["shift"],
                generator,
                DEVICE,
            )
        steps, cache = TURBO["steps"], None
    else:
        balanced = mode == "Balanced"
        steps = 30 if balanced else cfg.sample.steps
        cache = StepCache(threshold=0.1 if balanced else 0.0)
        with ForwardTimer(progress, steps, mode) as timer, inference_autocast(DEVICE):
            image = generate(
                model,
                text_encoder,
                [prompt],
                height,
                width,
                steps=steps,
                cfg_scale=cfg.sample.cfg_scale,
                cfg_interval=(0.0, 0.9) if balanced else tuple(cfg.sample.cfg_interval),
                shift=cfg.flow.shift,
                negative_prompt=cfg.sample.negative_prompt,
                generator=generator,
                device=DEVICE,
                num_train_timesteps=cfg.flow.num_train_timesteps,
                prediction=cfg.flow.prediction,
                cache=cache,
            )
    image = image.cpu()  # waits for the device
    seconds = time.perf_counter() - start
    rows = timer.rows
    pixels = height * width
    if cache is None or cache.skipped == 0:  # full forwards only: a clean per-forward cost
        for batch in set(rows):
            share = sum(r for r in rows if r == batch) / sum(rows)
            FORWARD_S[(pixels, batch)] = seconds * share / rows.count(batch)
    if mode == "Standard":
        STANDARD_S[pixels] = seconds
    return {
        "mode": mode,
        "image": to_pil(image[0]),
        "seconds": seconds,
        "steps": steps,
        "forwards": len(rows),
        "cfg_forwards": sum(1 for r in rows if r == 2),
        "skipped": cache.skipped if cache else 0,
        "pixels": pixels,
    }


def standard_estimate(pixels: int) -> tuple[float | None, str]:
    if pixels in STANDARD_S:
        return STANDARD_S[pixels], "measured"
    one, two = FORWARD_S.get((pixels, 1)), FORWARD_S.get((pixels, 2))
    if two is None and one is not None:
        two = 2 * one  # a CFG forward is a doubled batch
    if two is None:
        return None, ""
    # the standard sampler's first step (t = 1) sits outside the CFG interval
    one = one if one is not None else two / 2
    return one + (cfg.sample.steps - 1) * two, "estimated from measured forward times"


def stats_html(results: list[dict]) -> str:
    base, how = standard_estimate(results[0]["pixels"])
    rows = []
    for r in results:
        per_fwd = r["seconds"] / max(r["forwards"], 1) * 1000
        extra = f", trunk reused on {r['skipped']}" if r["skipped"] else ""
        speedup = f"{base / r['seconds']:.1f}×" if base else "–"
        rows.append(
            f"<tr><td><b>{r['mode']}</b><br><span class='sub'>{html.escape(MODES[r['mode']])}</span></td>"
            f"<td class='big'>{r['seconds']:.1f} s</td><td>{r['steps']} steps · {r['forwards']} forwards "
            f"({r['cfg_forwards']} with CFG{extra})</td><td>{per_fwd:.0f} ms / forward</td>"
            f"<td>{60 / r['seconds']:.1f} img/min</td><td class='big'>{speedup}</td></tr>"
        )
    note = (
        f"Speed-up vs. the 100-step sampler ({base:.0f} s, {how})."
        if base
        else "Run any mode once to calibrate the speed-up column."
    )
    device = f"{DEVICE.type.upper()} · " + (
        torch.cuda.get_device_name(DEVICE)
        if DEVICE.type == "cuda"
        else "Apple GPU"
        if DEVICE.type == "mps"
        else "CPU"
    )
    return (
        f"<table class='speed'><tr><th>Mode</th><th>Time</th><th>Work</th><th>Per forward</th><th>Throughput</th>"
        f"<th>Speed-up</th></tr>{''.join(rows)}</table><p class='sub'>{note} {device}, bf16 weights, "
        f"{results[0]['pixels'] / 1e6:.2f} MP, timing includes prompt encoding.</p>"
    )


def generate_one(prompt, mode, size, seed, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    height, width = SIZES[size]
    result = run_mode(mode, prompt, height, width, seed, progress)
    return [(result["image"], f"{result['mode']} · {result['seconds']:.1f} s")], stats_html([result])


def compare(prompt, size, seed, include_standard, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    height, width = SIZES[size]
    modes = (["Fast"] if TURBO else []) + ["Balanced"] + (["Standard"] if include_standard else [])
    results = [run_mode(m, prompt, height, width, seed, progress) for m in modes]
    gallery = [(r["image"], f"{r['mode']} · {r['seconds']:.1f} s") for r in results]
    return gallery, stats_html(results)


CSS = """
.speed { width: 100%; border-collapse: collapse; font-size: 14px; }
.speed th { text-align: left; font-weight: 600; opacity: .7; padding: 6px 8px; border-bottom: 1px solid var(--border-color-primary); }
.speed td { padding: 8px; border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.speed .big { font-size: 20px; font-weight: 700; white-space: nowrap; }
.sub { font-size: 12px; opacity: .65; }
"""
theme = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

with gr.Blocks(title="Iris-3B speed", theme=theme, css=CSS) as demo:
    gr.Markdown(
        "## Iris-3B · generation speed\nSame weights, prompt and seed in every mode; times are measured "
        "end to end on this machine."
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            prompt = gr.Textbox(
                label="Prompt", lines=3, value="a red fox sleeping in fresh snow, golden hour"
            )
            mode = gr.Radio(
                list(MODES),
                value="Fast" if TURBO else "Balanced",
                label="Mode",
                info=" · ".join(f"{k}: {v}" for k, v in MODES.items()),
            )
            with gr.Row():
                size = gr.Dropdown(list(SIZES), value="1024 × 1024", label="Size (width × height)")
                seed = gr.Number(value=0, precision=0, label="Seed")
            run_btn = gr.Button("Generate", variant="primary")
            with gr.Group():
                include_standard = gr.Checkbox(
                    value=False, label="Include the 100-step standard sampler (minutes per image on a laptop)"
                )
                compare_btn = gr.Button("Compare modes")
        with gr.Column(scale=6):
            gallery = gr.Gallery(label="Result", columns=3, height=560, object_fit="contain")
            stats = gr.HTML("<p class='sub'>Generate to see timings.</p>")
    run_btn.click(generate_one, [prompt, mode, size, seed], [gallery, stats])
    compare_btn.click(compare, [prompt, size, seed, include_standard], [gallery, stats])

if __name__ == "__main__":
    warm_up()
    demo.queue(default_concurrency_limit=1).launch(server_port=args.port, share=args.share)

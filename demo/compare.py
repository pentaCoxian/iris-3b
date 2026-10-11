"""Side-by-side comparison of few-step adapters on the same prompt and seed.

    python demo/compare.py --lora checkpoints/turbo-1500/student.safetensors --label "r1 · 1500" \\
                           --lora checkpoints/turbo-r2-1500/student.safetensors --label "r2 · 1500"

Each adapter is merged into its own bf16 copy of the model (~6 GB each), so every
column runs at full speed; the text encoder is shared. Models are warmed up at
launch, so the first click is timed like the rest.
"""

import argparse
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

import iris3b.text  # noqa: F401  (registers encoders)
from iris3b.config import inference_config
from iris3b.devices import default_device, inference_autocast
from iris3b.distill import generate_turbo
from iris3b.lora import resolve_adapter
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import load_model_streaming, read_config

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/iris-3b"))
parser.add_argument(
    "--lora", action="append", required=True, help="adapter file or Hub id owner/repo[:name] (repeat)"
)
parser.add_argument("--label", action="append", default=[], help="column label per --lora, in order")
parser.add_argument("--device", default=None)
parser.add_argument("--port", type=int, default=7862)
args = parser.parse_args()
labels = args.label + [
    str(p).rsplit(":", 1)[-1] if ":" in str(p) else Path(p).parent.name for p in args.lora[len(args.label) :]
]
args.lora = [str(resolve_adapter(p)) for p in args.lora]

DEVICE = torch.device(args.device) if args.device else default_device()
cfg = inference_config(read_config(args.checkpoint), [])
MODELS = []
for label, path in zip(labels, args.lora, strict=True):
    model, meta = load_model_streaming(cfg.model, args.checkpoint, DEVICE, torch.bfloat16, lora_path=path)
    if "turbo_steps" not in meta:
        raise SystemExit(f"{path} is not a few-step adapter (no turbo_steps in its metadata)")
    MODELS.append((label, model, meta))
text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=DEVICE)
SIZES = {
    "1024 × 1024": (1024, 1024),
    "1152 × 896": (896, 1152),
    "896 × 1152": (1152, 896),
    "1344 × 768": (768, 1344),
    "768 × 1344": (1344, 768),
}


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def to_pil(image: torch.Tensor) -> Image.Image:
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(pixels))


def run(prompt, size, seed, randomize, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    if randomize:
        seed = int(np.random.randint(0, 2**31 - 1))
    height, width = SIZES[size]
    images = []
    for i, (label, model, meta) in enumerate(MODELS):
        progress((i, len(MODELS)), desc=f"{label}")
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        sync()
        start = time.perf_counter()
        with inference_autocast(DEVICE):
            image = generate_turbo(
                model,
                text_encoder,
                [prompt],
                height,
                width,
                meta["turbo_steps"],
                meta["shift"],
                generator,
                DEVICE,
            ).cpu()
        images.append((to_pil(image[0]), f"{label} · {time.perf_counter() - start:.1f} s"))
    return images, seed


@torch.no_grad()
def warm_up() -> None:
    enc = text_encoder.encode(["warm up"])
    y, mask = enc.embeddings.to(DEVICE, torch.bfloat16), enc.mask.to(DEVICE)
    x, t = torch.randn(1, 3, 1024, 1024, device=DEVICE), torch.full((1,), 500.0, device=DEVICE)
    with inference_autocast(DEVICE):
        for _, model, _ in MODELS:
            model(x, t, y, y_mask=mask)
    sync()


theme = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
with gr.Blocks(title="Iris-3B adapter comparison", theme=theme) as demo:
    gr.Markdown(
        "## Iris-3B · side by side\nSame prompt, seed and size for every column: " + " · ".join(labels)
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt", lines=2, value="a red fox sleeping in fresh snow, golden hour", scale=6
        )
        with gr.Column(scale=2):
            size = gr.Dropdown(list(SIZES), value="1024 × 1024", label="Size (width × height)")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="Seed")
                randomize = gr.Checkbox(value=False, label="Random seed")
    run_btn = gr.Button("Generate", variant="primary")
    gallery = gr.Gallery(columns=len(MODELS), height=640, object_fit="contain", show_label=False)
    run_btn.click(run, [prompt, size, seed, randomize], [gallery, seed])
    prompt.submit(run, [prompt, size, seed, randomize], [gallery, seed])

if __name__ == "__main__":
    warm_up()
    demo.queue(default_concurrency_limit=1).launch(server_port=args.port)

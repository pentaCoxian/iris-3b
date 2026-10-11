"""Distill Iris-3B into a few-step turbo LoRA (DMD2, prompts only).

Usage (5090 trains, 4090 hosts the text encoder):
    python scripts/distill_turbo.py --checkpoint checkpoints/iris-3b --prompts prompts.txt \\
        --out output/turbo --device cuda:0 --text-device cuda:1 --iters 3000
    # continue a run
    python scripts/distill_turbo.py ... --resume output/turbo/latest

Train at about one megapixel (the 1024 default): the released checkpoint was
fine-tuned there and produces 16 px block artifacts at 512, which the student
would learn from the teacher.

The teacher, the student and the DMD critic share one frozen bf16 copy of the
base weights; only the two adapters train. ``--prompts`` is a text file with
one prompt per line (a few thousand diverse prompts is plenty). Every
``--save-every`` iterations both adapters are written to
``<out>/iter_<n>/`` and ``<out>/latest/``; ``scripts/sample.py --lora
<dir>/student.safetensors`` merges the student for one forward per step.
"""

import argparse
import dataclasses
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iris3b.config import inference_config
from iris3b.devices import default_device
from iris3b.distill import (
    FAKE,
    STUDENT,
    Conditioning,
    critic_update,
    student_sample,
    student_update,
    turbo_times,
)
from iris3b.flow.transport import RectifiedFlow
from iris3b.lora import inject_lora, load_adapter, save_adapter
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import load_for_inference

PREVIEW_PROMPTS = [
    "a red fox sleeping in fresh snow, golden hour",
    "Studio portrait of an elderly fisherman with a thick grey beard, overcast light, film photograph",
    "A glass sculpture of a heart filled with flowers, caustics and reflections, 3D render",
    "The ancient city of Petra with the Treasury carved into pink sandstone, morning light",
]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompts", required=True, help="text file, one prompt per line")
    p.add_argument("--out", default="output/turbo")
    p.add_argument("--device", default=None, help="training device (default: cuda, then mps, then cpu)")
    p.add_argument("--text-device", default=None, help="text encoder device (default: --device)")
    p.add_argument("--resolution", type=int, default=1024, help="train at ~1MP: the released model is blocky at 512")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--student-steps", type=int, default=4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-5, help="student adapter LR")
    p.add_argument("--critic-lr", type=float, default=5e-5)
    p.add_argument("--critic-updates", type=int, default=5, help="critic updates per student update (TTUR)")
    p.add_argument("--cfg-scale", type=float, default=3.0, help="teacher guidance baked into the student")
    p.add_argument("--t-range", type=float, nargs=2, default=(0.02, 0.98), help="DMD noise levels (pre-shift)")
    p.add_argument("--shift", type=float, default=None, help="default: checkpoint flow.shift")
    p.add_argument("--ema-decay", type=float, default=0.99, help="per student update; saved and previewed")
    p.add_argument(
        "--collapse-std",
        type=float,
        default=0.2,  # good previews ~0.23-0.44, the collapsed 1750 ones ~0.14
        help="stop when the previews' mean per-image pixel std (in [-1, 1]) falls below this",
    )
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--resume", default=None, help="directory holding student/fake .safetensors")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def encode(text_encoder, prompts, null, device, dtype) -> Conditioning:
    enc = text_encoder.encode(prompts)
    batch = len(prompts)
    return Conditioning(
        cond=enc.embeddings.to(device, dtype),
        cond_mask=enc.mask.to(device),
        null=null.embeddings.to(device, dtype).expand(batch, *null.embeddings.shape[1:]),
        null_mask=null.mask.to(device).expand(batch, -1),
    )


class EMA:
    """Exponential moving average of the student adapter.

    DMD can drift into a degenerate mode within a few hundred steps; the
    average lags such a drift, and it is what gets saved and previewed.
    """

    def __init__(self, params: list[torch.nn.Parameter], decay: float):
        self.params = params
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - self.decay)

    @contextmanager
    def applied(self):
        live = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, e in zip(self.params, self.shadow, strict=True):
                p.copy_(e)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, v in zip(self.params, live, strict=True):
                    p.copy_(v)


def save(model, ema: EMA, out: Path, tag: str, meta: dict) -> None:
    for folder in (out / tag, out / "latest"):
        folder.mkdir(parents=True, exist_ok=True)
        # student_raw resumes training; student (the EMA) is what sample.py merges
        save_adapter(model, STUDENT, folder / "student_raw.safetensors", metadata=meta)
        with ema.applied():
            save_adapter(model, STUDENT, folder / "student.safetensors", metadata=meta)
        save_adapter(model, FAKE, folder / "fake.safetensors", metadata=meta)


@torch.no_grad()
def preview(model, text_encoder, null, args, times, device, path: Path) -> float:
    """Write the preview grid; returns the mean per-image pixel std (collapse signal)."""
    from torchvision.utils import save_image

    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    c = encode(text_encoder, PREVIEW_PROMPTS, null, device, torch.bfloat16)
    noise = torch.randn(len(PREVIEW_PROMPTS), 3, args.resolution, args.resolution, generator=gen, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps")):
        images = student_sample(model, noise, c, times, generator=gen)
    images = images.clamp(-1, 1)
    save_image(images, str(path), nrow=len(PREVIEW_PROMPTS), normalize=True, value_range=(-1, 1))
    model.train()
    return images.float().std(dim=(1, 2, 3)).mean().item()


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    text_device = torch.device(args.text_device) if args.text_device else device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, weights = load_for_inference(args.checkpoint)
    cfg = inference_config(raw, [])
    shift = args.shift if args.shift is not None else cfg.flow.shift
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    # frozen base in bf16; the adapters are created in fp32 on top of it
    model = model.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    student_params = inject_lora(model, STUDENT, args.rank)
    critic_params = inject_lora(model, FAKE, args.rank)
    if args.resume:
        resume = Path(args.resume)
        raw = resume / "student_raw.safetensors"
        load_adapter(model, STUDENT, raw if raw.exists() else resume / "student.safetensors")
        load_adapter(model, FAKE, Path(args.resume) / "fake.safetensors")
    for param in student_params + critic_params:
        param.requires_grad_(True)
    model.activation_checkpointing = "full"
    model.train()
    n_params = sum(p.numel() for p in student_params)
    print(f"adapter rank {args.rank}: {n_params / 1e6:.1f}M params each (student, critic)", flush=True)

    ema = EMA(student_params, args.ema_decay)
    student_opt = torch.optim.AdamW(student_params, lr=args.lr, betas=(0.0, 0.999), weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic_params, lr=args.critic_lr, betas=(0.0, 0.999), weight_decay=0.0)

    import iris3b.text  # noqa: F401  (registers encoders)

    text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=text_device)
    null = text_encoder.null(cfg.sample.negative_prompt)
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")

    # the critic's flow loss uses the run's shift, never a resolution law
    flow = RectifiedFlow(dataclasses.replace(cfg.flow, shift=shift, shift_law="none"))
    times = turbo_times(args.student_steps, shift)
    meta = {"turbo_steps": args.student_steps, "shift": shift, "cfg_scale": args.cfg_scale,
            "resolution": args.resolution}
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (out / "log.jsonl").open("a")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))
    shape = (args.batch, 3, args.resolution, args.resolution)

    start = time.perf_counter()
    for it in range(1, args.iters + 1):
        c = encode(text_encoder, random.sample(prompts, args.batch), null, device, torch.bfloat16)
        noise = torch.randn(shape, device=device)
        grad_step = random.randrange(args.student_steps)  # backward simulation depth
        update_student = it % args.critic_updates == 0
        record = {"iter": it}
        with autocast:
            if update_student:
                student_opt.zero_grad(set_to_none=True)
                loss, x0 = student_update(
                    model, noise, c, times, grad_step, args.cfg_scale, tuple(args.t_range), shift
                )
                record["student_grad"] = torch.nn.utils.clip_grad_norm_(student_params, 1.0).item()
                student_opt.step()
                ema.update()
                record["dmd"] = loss.item()
            else:
                with torch.no_grad():
                    x0 = student_sample(model, noise, c, times, grad_step=grad_step)
            critic_opt.zero_grad(set_to_none=True)
            loss = critic_update(model, flow, x0, c)
            torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            critic_opt.step()
            record["critic"] = loss.item()
        record["sec"] = round(time.perf_counter() - start, 1)
        if it % args.save_every == 0 or it == args.iters:
            save(model, ema, out, f"iter_{it:06d}", meta)
        if it % args.preview_every == 0 or it == args.iters:
            with ema.applied():
                record["preview_std"] = round(
                    preview(model, text_encoder, null, args, times, device, out / f"preview_{it:06d}.png"), 4
                )
        log.write(json.dumps(record) + "\n")
        log.flush()
        if record.get("preview_std", 1.0) < args.collapse_std:
            print(json.dumps(record), flush=True)
            raise SystemExit(
                f"previews collapsed at iteration {it} (pixel std {record['preview_std']} < "
                f"{args.collapse_std}); resume from the last good iter_* directory"
            )
        if it % 10 == 0 or it == 1 or "preview_std" in record:
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

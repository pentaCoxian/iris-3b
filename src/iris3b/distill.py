"""Few-step ("turbo") distillation by distribution matching (DMD2, arXiv:2405.14867).

Three networks share one frozen base model:

- teacher: the bare base, sampled with classifier-free guidance;
- student: base + ``student`` adapter, a few-step generator with the
  guidance baked in (one forward per step, no CFG batch);
- critic:  base + ``fake`` adapter, a flow model kept fitted to the
  student's current output distribution.

The student's gradient is the difference between the critic's and the
teacher's clean-image predictions at a re-noised student sample: it pushes
the student's distribution toward the guided teacher's. The critic is updated
more often than the student (two time-scale update rule), which is what lets
DMD2 drop DMD's regression loss and therefore its precomputed teacher pairs:
the whole recipe needs prompts only.

Both update functions call ``backward`` themselves, inside the adapter
context that produced the graph: activation checkpointing recomputes blocks
during backward, and the recompute must see the same live adapter.

Student sampling (``turbo_times``) runs ``steps`` evaluations on the shifted
grid ``t_i = shift(1 - i / steps)``; between evaluations the predicted clean
image is re-noised to the next level with fresh noise. Training uses backward
simulation: a random number of no-grad student steps, then one step with grad.
"""

from dataclasses import dataclass

import torch

from iris3b.flow.schedule import shift_sigma
from iris3b.flow.transport import RectifiedFlow
from iris3b.lora import use_adapter
from iris3b.models.dit import IrisDiT

STUDENT, FAKE = "student", "fake"


def turbo_times(steps: int, shift: float) -> list[float]:
    """Student evaluation times, descending from pure noise (t = 1)."""
    grid = 1.0 - torch.arange(steps, dtype=torch.float64) / steps
    return shift_sigma(grid, shift).tolist()


@dataclass
class Conditioning:
    cond: torch.Tensor
    cond_mask: torch.Tensor
    null: torch.Tensor  # negative / empty prompt, batch-expanded
    null_mask: torch.Tensor


def predict_x0(model: IrisDiT, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """Clean-image prediction at flow times ``t`` [B] (1 = noise)."""
    out = model(x, t * 1000.0, y, y_mask=mask).x.float()
    return x.float() - t.view(-1, 1, 1, 1) * out


def teacher_x0(model: IrisDiT, x: torch.Tensor, t: torch.Tensor, c: Conditioning, cfg_scale: float):
    with use_adapter(model, None):
        both = predict_x0(
            model,
            torch.cat([x, x]),
            torch.cat([t, t]),
            torch.cat([c.null, c.cond]),
            torch.cat([c.null_mask, c.cond_mask]),
        )
    uncond, cond = both.chunk(2)
    return uncond + cfg_scale * (cond - uncond)


def student_sample(
    model: IrisDiT,
    noise: torch.Tensor,
    c: Conditioning,
    times: list[float],
    grad_step: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the student from ``noise``; with ``grad_step`` stop there with grad."""
    last = len(times) - 1 if grad_step is None else grad_step
    x = noise
    with use_adapter(model, STUDENT):
        for i in range(last + 1):
            t = torch.full((x.shape[0],), times[i], device=x.device)
            with torch.set_grad_enabled(grad_step is not None and i == last):
                x0 = predict_x0(model, x.to(noise.dtype), t, c.cond, c.cond_mask)
            if i == last:
                return x0
            fresh = torch.randn(x0.shape, device=x0.device, generator=generator, dtype=x0.dtype)
            x = (1.0 - times[i + 1]) * x0.detach() + times[i + 1] * fresh
    raise AssertionError("unreachable")


def dmd_loss(
    model: IrisDiT,
    x0_student: torch.Tensor,
    c: Conditioning,
    cfg_scale: float,
    t_range: tuple[float, float],
    shift: float,
) -> torch.Tensor:
    """Distribution-matching loss whose gradient w.r.t. ``x0_student`` is the DMD direction."""
    batch = x0_student.shape[0]
    u = torch.empty(batch, device=x0_student.device).uniform_(*t_range)
    t = shift_sigma(u, shift)
    x0 = x0_student.detach()
    xt = (1.0 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * torch.randn_like(x0)
    with torch.no_grad():
        real = teacher_x0(model, xt, t, c, cfg_scale)
        with use_adapter(model, FAKE):
            fake = predict_x0(model, xt, t, c.cond, c.cond_mask)
        # per-sample normalizer from DMD: keeps the step size independent of t
        weight = 1.0 / (x0 - real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        target = x0 - weight * (fake - real)
    return 0.5 * (x0_student.float() - target).pow(2).mean()


def student_update(
    model: IrisDiT,
    noise: torch.Tensor,
    c: Conditioning,
    times: list[float],
    grad_step: int,
    cfg_scale: float,
    t_range: tuple[float, float],
    shift: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate the DMD loss into the student adapter; returns (loss, student x0)."""
    with use_adapter(model, STUDENT):
        x0 = student_sample(model, noise, c, times, grad_step=grad_step)
        loss = dmd_loss(model, x0, c, cfg_scale, t_range, shift)
        loss.backward()
    return loss.detach(), x0.detach()


def critic_update(model: IrisDiT, flow: RectifiedFlow, x0_student: torch.Tensor, c: Conditioning) -> torch.Tensor:
    """Backpropagate the critic's flow-matching loss on the student's samples."""
    with use_adapter(model, FAKE):
        loss = flow.training_loss(
            model, x0_student.detach(), c.cond, model_kwargs={"y_mask": c.cond_mask}
        ).loss
        loss.backward()
    return loss.detach()


@torch.no_grad()
def generate_turbo(
    model: IrisDiT,
    text_encoder,
    prompts: list[str],
    height: int,
    width: int,
    steps: int,
    shift: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Sample [B, C, H, W] in [-1, 1] from a merged turbo model (no CFG)."""
    model_dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=model_dtype)
    mask = encoding.mask.to(device=device)
    times = turbo_times(steps, shift)
    x = torch.randn(len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device)
    for i, t_now in enumerate(times):
        t = torch.full((x.shape[0],), t_now, device=device)
        out = model(x.to(model_dtype), t * 1000.0, cond, y_mask=mask).x.float()
        x0 = x - t_now * out
        if i + 1 == len(times):
            return x0.clamp(-1, 1)
        fresh = torch.randn(x.shape, generator=generator, device=device)
        x = (1.0 - times[i + 1]) * x0 + times[i + 1] * fresh
    raise AssertionError("unreachable")

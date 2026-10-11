import dataclasses

import pytest
import torch

from iris3b.config import FlowConfig
from iris3b.distill import (
    FAKE,
    STUDENT,
    Conditioning,
    critic_update,
    generate_turbo,
    student_update,
    turbo_times,
)
from iris3b.flow.transport import RectifiedFlow
from iris3b.lora import LoRALinear, inject_lora, load_adapter, merge_lora_file, save_adapter, use_adapter
from iris3b.models import IrisDiT
from iris3b.text.base import TextEncoding
from tiny_config import tiny_model_config


def tiny_model() -> IrisDiT:
    torch.manual_seed(0)
    model = IrisDiT(tiny_model_config()).eval()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
    return model


def inputs(batch: int = 2):
    gen = torch.Generator().manual_seed(1)
    return (
        torch.randn(batch, 3, 32, 32, generator=gen),
        torch.full((batch,), 700.0),
        torch.randn(batch, 16, 32, generator=gen),
    )


def randomize(params, std=0.05):
    with torch.no_grad():
        for p in params:
            p.normal_(0, std)


def conditioning(batch: int = 2) -> Conditioning:
    gen = torch.Generator().manual_seed(2)
    mask = torch.ones(batch, 16, dtype=torch.int64)
    return Conditioning(torch.randn(batch, 16, 32, generator=gen), mask, torch.zeros(batch, 16, 32), mask)


@torch.no_grad()
def test_fresh_adapter_is_a_no_op_and_wraps_sequential_linears():
    model = tiny_model()
    expected = model(*inputs()).x
    inject_lora(model, STUDENT, rank=4)
    assert isinstance(model.t_embedder.mlp[0], LoRALinear)
    assert not any(isinstance(m, LoRALinear) for m in model.y_embedder.modules())
    with use_adapter(model, STUDENT):
        assert torch.equal(model(*inputs()).x, expected)


@torch.no_grad()
def test_adapters_are_independent_and_reinjection_does_not_rewrap():
    model = tiny_model()
    base = model(*inputs()).x
    randomize(inject_lora(model, STUDENT, rank=4))
    inject_lora(model, FAKE, rank=4)
    inject_lora(model, STUDENT, rank=4)  # existing slot: untouched
    assert not any(isinstance(m.base, LoRALinear) for m in model.modules() if isinstance(m, LoRALinear))
    with use_adapter(model, STUDENT):
        student = model(*inputs()).x
    with use_adapter(model, FAKE):
        assert torch.equal(model(*inputs()).x, base)
    assert not torch.allclose(student, base)
    assert torch.equal(model(*inputs()).x, base)  # context restored the bare base


@torch.no_grad()
def test_merge_and_roundtrip_match_the_live_adapter(tmp_path):
    model = tiny_model()
    randomize(inject_lora(model, STUDENT, rank=4, alpha=8.0))
    with use_adapter(model, STUDENT):
        live = model(*inputs()).x
    path = tmp_path / "a.safetensors"
    save_adapter(model, STUDENT, path, metadata={"turbo_steps": 4, "shift": 3.0})

    merged = tiny_model()
    meta = merge_lora_file(merged, path)
    assert meta["turbo_steps"] == 4 and meta["rank"] == 4
    torch.testing.assert_close(merged(*inputs()).x, live, rtol=1e-4, atol=1e-5)

    reloaded = tiny_model()
    inject_lora(reloaded, STUDENT, rank=4, alpha=8.0)
    load_adapter(reloaded, STUDENT, path)
    with use_adapter(reloaded, STUDENT):
        torch.testing.assert_close(reloaded(*inputs()).x, live)


def test_turbo_times():
    assert turbo_times(4, 1.0) == [1.0, 0.75, 0.5, 0.25]
    shifted = turbo_times(4, 3.0)
    assert shifted[0] == 1.0 and all(a > b for a, b in zip(shifted, shifted[1:], strict=False))


@pytest.mark.parametrize("checkpointing", ["none", "full"])
def test_student_step_trains_only_the_student_and_critic_step_only_the_critic(checkpointing):
    model = tiny_model().requires_grad_(False)
    student = inject_lora(model, STUDENT, rank=4)
    critic = inject_lora(model, FAKE, rank=4)
    randomize(critic, 0.01)  # a critic equal to the teacher gives a zero DMD gradient
    for p in student + critic:
        p.requires_grad_(True)
    # checkpointing recomputes blocks in backward, which must see the same adapter
    model.activation_checkpointing = checkpointing
    model.train()
    c = conditioning()
    noise = torch.randn(2, 3, 32, 32)
    times = turbo_times(4, 3.0)

    loss, x0 = student_update(model, noise, c, times, 2, cfg_scale=3.0, t_range=(0.1, 0.9), shift=3.0)
    assert loss.isfinite()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student)
    assert all(p.grad is None for p in critic)

    for p in student:
        p.grad = None
    flow = RectifiedFlow(dataclasses.replace(FlowConfig(), shift=3.0, shift_law="none"))
    critic_update(model, flow, x0, c)
    assert all(p.grad is None for p in student)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic)


@pytest.mark.parametrize("steps", [1, 4])
def test_generate_turbo_shape(steps):
    model = tiny_model()

    class Encoder:
        def encode(self, prompts):
            return TextEncoding(torch.randn(len(prompts), 16, 32), torch.ones(len(prompts), 16, dtype=torch.int64))

    out = generate_turbo(model, Encoder(), ["a", "b"], 32, 32, steps=steps, shift=3.0, device="cpu")
    assert out.shape == (2, 3, 32, 32) and out.abs().max() <= 1

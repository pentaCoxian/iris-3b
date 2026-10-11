import pytest
import torch

from iris3b.flow import FlowDPMSolver
from iris3b.models import IrisDiT
from iris3b.models.step_cache import StepCache
from tiny_config import tiny_model_config


def tiny_model(depth: int = 3) -> IrisDiT:
    torch.manual_seed(0)
    model = IrisDiT(tiny_model_config(depth=depth)).eval()
    # the head is zero-initialized; give it weights so outputs are comparable
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
    return model


def sample(model: IrisDiT, cache: StepCache | None, steps: int = 8) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(2, 3, 32, 32, generator=gen)
    cond = torch.randn(2, 16, 32, generator=gen)
    uncond = torch.zeros(1, 16, 32).expand(2, -1, -1)

    def model_fn(x, t, y):
        return model(x, t, y, cache=cache).x

    solver = FlowDPMSolver(model_fn, cfg_scale=3.0, cfg_interval=(0.0, 0.6))
    return solver.sample(z, cond, uncond, steps=steps, order=2, shift=3.0)


@torch.no_grad()
def test_zero_threshold_is_exact():
    model = tiny_model()
    assert torch.equal(sample(model, None), sample(model, StepCache(threshold=0.0)))


@torch.no_grad()
def test_text_adapter_runs_once_per_batch_shape():
    model = tiny_model()
    calls = []
    model.y_embedder.register_forward_hook(lambda m, i, o: calls.append(i[0].shape[0]))
    cache = StepCache()
    sample(model, cache, steps=8)
    # the CFG gate switches between the [2B] and [B] batches exactly once each
    assert sorted(calls) == [2, 4]


@torch.no_grad()
def test_unchanged_input_reuses_residual():
    model = tiny_model()
    x, t, y = torch.randn(1, 3, 32, 32), torch.tensor([500.0]), torch.randn(1, 16, 32)
    cache = StepCache(threshold=0.1)
    first = model(x, t, y, cache=cache).x
    second = model(x, t, y, cache=cache).x
    assert cache.skipped == 1
    torch.testing.assert_close(first, second)


@torch.no_grad()
def test_skips_are_bounded_and_reset_on_batch_change():
    model = tiny_model()
    x, t, y = torch.randn(1, 3, 32, 32), torch.tensor([500.0]), torch.randn(1, 16, 32)
    cache = StepCache(threshold=1.0, max_skips=2)
    for _ in range(4):
        model(x, t, y, cache=cache)
    assert cache.skipped == 2  # full, skip, skip, full (max_skips reached)
    model(x.expand(2, -1, -1, -1), t.expand(2), y.expand(2, -1, -1), cache=cache)
    assert cache.skipped == 2  # a new batch shape has no comparable residual


@torch.no_grad()
def test_large_threshold_skips_and_stays_close():
    model = tiny_model()
    exact = sample(model, None, steps=16)
    cache = StepCache(threshold=0.3)
    approx = sample(model, cache, steps=16)
    assert cache.skipped > 0
    assert approx.isfinite().all()
    assert (approx - exact).abs().mean() < 0.5 * exact.abs().mean()


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        StepCache(threshold=-1)
    with pytest.raises(ValueError):
        StepCache(probe_blocks=0)

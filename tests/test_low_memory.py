import torch
from safetensors.torch import save_file

from iris3b.sampling import PrecomputedText, load_for_inference, read_config
from iris3b.text.base import TextEncoding


class CountingEncoder:
    dim, max_length = 8, 4

    def __init__(self):
        self.calls = 0

    def encode(self, prompts):
        self.calls += 1
        rows = torch.stack([torch.full((4, 8), float(len(p))) for p in prompts])
        return TextEncoding(rows, torch.ones(len(prompts), 4, dtype=torch.int64))

    def null(self, negative_prompt=""):
        return TextEncoding(torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.int64))


def test_precomputed_text_matches_the_encoder_and_needs_it_no_more():
    encoder = CountingEncoder()
    pre = PrecomputedText.encode_with(encoder, ["a", "bbb", "a"], negative_prompt="")
    assert encoder.calls == 2  # duplicates encoded once
    out = pre.encode(["bbb", "a"])
    torch.testing.assert_close(out.embeddings, encoder.encode(["bbb", "a"]).embeddings)
    assert out.mask.shape == (2, 4)
    assert pre.null("").embeddings.shape == (1, 4, 8)


def test_load_for_inference_converts_dtype_while_reading(tmp_path):
    save_file({"w": torch.randn(3, 3), "b": torch.randn(3)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.yaml").write_text("model:\n  depth: 2\n")
    assert read_config(tmp_path) == {"model": {"depth": 2}}
    _, fp32 = load_for_inference(tmp_path)
    _, bf16 = load_for_inference(tmp_path, dtype=torch.bfloat16)
    assert fp32["w"].dtype == torch.float32 and bf16["w"].dtype == torch.bfloat16
    torch.testing.assert_close(bf16["w"].float(), fp32["w"], rtol=1e-2, atol=1e-2)


def shared_core_model():
    from iris3b.models import IrisDiT
    from tiny_config import tiny_model_config

    torch.manual_seed(0)
    model = IrisDiT(tiny_model_config(modulation="shared_bias")).eval()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
    return model


def tiny_inputs():
    gen = torch.Generator().manual_seed(1)
    return torch.randn(1, 3, 32, 32, generator=gen), torch.tensor([600.0]), torch.randn(1, 16, 32, generator=gen)


@torch.no_grad()
def test_int8_model_tracks_the_float_model_and_reaches_shared_cores():
    from iris3b.quant import Int8Linear, quantize_linears

    model = shared_core_model()
    reference = model(*tiny_inputs()).x
    n = quantize_linears(model)
    assert n > 0
    # every block must call the int8 core, not a stale float one
    cores = [m._core[0] for m in model.modules() if isinstance(getattr(m, "_core", None), tuple)]
    assert cores and all(isinstance(core, Int8Linear) for core in cores)
    out = model(*tiny_inputs()).x
    assert (out - reference).abs().mean() < 0.05 * reference.abs().mean()


@torch.no_grad()
def test_lora_on_a_shared_modulation_core_is_live():
    from iris3b.lora import inject_lora, use_adapter

    model = shared_core_model()
    base = model(*tiny_inputs()).x
    inject_lora(model, "student", rank=4, targets=("modulation_cores",))
    for layer in model.modulation_cores.values():
        torch.nn.init.normal_(layer.adapters["student"][1].weight, std=0.5)
    with use_adapter(model, "student"):
        assert not torch.allclose(model(*tiny_inputs()).x, base)


@torch.no_grad()
def test_streaming_load_matches_load_then_merge(tmp_path):
    from iris3b.lora import inject_lora, merge_lora_file, save_adapter
    from iris3b.sampling import load_model_streaming

    model = shared_core_model()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    with_adapter = shared_core_model()
    for p in inject_lora(with_adapter, "student", rank=4):
        p.normal_(0, 0.05)
    save_adapter(with_adapter, "student", tmp_path / "a.safetensors", metadata={"turbo_steps": 4})

    expected = shared_core_model()
    merge_lora_file(expected, tmp_path / "a.safetensors")
    reference = expected(*tiny_inputs()).x

    streamed, meta = load_model_streaming(
        expected.cfg, tmp_path, "cpu", torch.float32, lora_path=tmp_path / "a.safetensors"
    )
    assert meta["turbo_steps"] == 4
    torch.testing.assert_close(streamed(*tiny_inputs()).x, reference)

    quantized, _ = load_model_streaming(
        expected.cfg, tmp_path, "cpu", torch.float32, quantize_keep=lambda name: False,
        lora_path=tmp_path / "a.safetensors",
    )
    out = quantized(*tiny_inputs()).x
    assert (out - reference).abs().mean() < 0.05 * reference.abs().mean()

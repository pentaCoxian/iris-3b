"""Weight-only int8 Linear layers, for fitting inference into less memory.

Symmetric per-output-channel quantization: ``w ≈ scale[:, None] * q`` with
``q`` in [-127, 127]. Activations stay in the compute dtype, so this halves
BF16 weight memory without touching the activation path. Where the fused
``torch._weight_int8pack_mm`` kernel applies (CPU and MPS, both dims a
multiple of 32) a layer runs at BF16 matmul speed; elsewhere it dequantizes
one layer's weight transiently and runs a BF16 matmul (~7% slower on MPS).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight_q", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None
        self._fused = in_features % 32 == 0 and out_features % 32 == 0

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        layer.load_weight(linear.weight, linear.bias)
        return layer

    @torch.no_grad()
    def load_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        q, scale = quantize_weight(weight.to(self.weight_q.device))
        self.weight_q.copy_(q)
        self.scale.copy_(scale)
        if bias is not None:
            self.bias.data = bias.detach().to(self.weight_q.device, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the int8 op is outside autocast's op lists, so apply its dtype by hand
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            dtype = torch.bfloat16
        flat = x.reshape(-1, self.in_features).to(dtype)
        # not CUDA: its _weight_int8pack_mm is a GEMV-style kernel, ~125x slower than a
        # bf16 matmul at ~4k image tokens on an RTX 5090; the dequantizing path costs ~10%
        if self._fused and x.device.type in ("cpu", "mps"):
            out = torch._weight_int8pack_mm(flat, self.weight_q, self.scale.to(dtype))
        else:
            out = F.linear(flat, self.weight_q.to(dtype) * self.scale[:, None].to(dtype))
        if self.bias is not None:
            out = out + self.bias.to(dtype)
        return out.reshape(*x.shape[:-1], self.out_features)


def quantize_linears(module: nn.Module, keep: Callable[[str], bool] = lambda name: False) -> int:
    """Replace every ``nn.Linear`` under ``module`` (except names ``keep`` accepts) in place.

    Layers are converted one at a time, so peak memory grows by one layer at most.
    Returns the number of layers converted.
    """
    from iris3b.nn.modulation import rebind_shared_cores

    replaced: dict[int, nn.Module] = {}
    for name, child in list(module.named_modules()):
        for child_name, sub in list(child.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(sub, nn.Linear) and not keep(full):
                replaced[id(sub)] = Int8Linear.from_linear(sub)
                setattr(child, child_name, replaced[id(sub)])
    rebind_shared_cores(module, replaced)
    return len(replaced)

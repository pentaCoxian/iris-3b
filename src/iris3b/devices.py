"""Inference device selection: CUDA, then Apple MPS, then CPU."""

import torch


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference_autocast(device: torch.device | str) -> torch.autocast:
    """BF16 autocast on accelerators, over BF16 or FP32 weights.

    CPU stays in FP32: CPU BF16 matmuls are slower than FP32 on most hosts.
    """
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def release_memory(device: torch.device | str) -> None:
    """Hand cached allocator blocks back to the system (after freeing a model)."""
    import gc

    gc.collect()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

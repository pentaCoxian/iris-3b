"""Cross-step reuse for one sampling run (training-free, inference only).

Two things repeat between solver steps:

- The text adapter output depends only on the prompt, never on ``x`` or
  ``t``, so it is computed once per batch shape and reused exactly. A run sees
  at most two text batches -- ``cond`` [B] and the CFG batch [2B] -- which the
  batch size tells apart.
- The trunk's residual ``s_out - s_probe`` drifts slowly between adjacent
  steps (first-block caching, as in TeaCache / FBCache). The first
  ``probe_blocks`` blocks always run; their output change since the previous
  step, relative to its magnitude, is accumulated, and while the total stays
  under ``threshold`` the remaining blocks are skipped and the last residual
  is added instead. The pixel stage always runs on the current ``x``, so
  per-pixel detail is never stale.

``threshold=0`` disables trunk skipping, which leaves the output identical to
an uncached run. A cache belongs to exactly one sampling run: reusing it for
another prompt would replay the wrong text states.
"""

import torch


class StepCache:
    def __init__(self, threshold: float = 0.0, probe_blocks: int = 1, max_skips: int = 3):
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if probe_blocks < 1:
            raise ValueError(f"probe_blocks must be >= 1, got {probe_blocks}")
        self.threshold = threshold
        self.probe_blocks = probe_blocks
        self.max_skips = max_skips
        self._text: dict[int, torch.Tensor] = {}
        self._prev_probe: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._accum = 0.0
        self._run_of_skips = 0
        self.calls = 0
        self.skipped = 0

    def text(self, batch: int, compute) -> torch.Tensor:
        if batch not in self._text:
            self._text[batch] = compute()
        return self._text[batch]

    def should_skip(self, probe: torch.Tensor) -> bool:
        """Record this step's probe output and decide whether to reuse the residual."""
        self.calls += 1
        prev, self._prev_probe = self._prev_probe, probe
        if prev is None or prev.shape != probe.shape or self._residual is None:
            # first step, or the CFG gate changed the batch: nothing comparable
            self._residual = None
            return False
        if self.threshold <= 0 or self._run_of_skips >= self.max_skips:
            return False
        change = (probe - prev).abs().mean() / prev.abs().mean().clamp_min(1e-8)
        self._accum += change.item()
        if self._accum >= self.threshold:
            return False
        self._run_of_skips += 1
        self.skipped += 1
        return True

    def residual(self) -> torch.Tensor:
        return self._residual

    def store(self, probe: torch.Tensor, out: torch.Tensor) -> None:
        self._residual = out - probe
        self._accum = 0.0
        self._run_of_skips = 0

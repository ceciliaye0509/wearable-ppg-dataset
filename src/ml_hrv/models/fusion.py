"""Explicitly gated, uncertainty-aware fusion of direct and beat estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FusionDecision:
    approved: bool
    reason: str
    direct_mae_ms: float
    beat_mae_ms: float
    fused_mae_ms: float


class UncertaintyFusion:
    """Precision-weighted fusion; only callable after complementarity approval."""

    def __init__(self, decision: FusionDecision | None = None) -> None:
        self.decision = decision

    def __call__(
        self,
        direct_ms: torch.Tensor,
        direct_std_ms: torch.Tensor,
        beat_ms: torch.Tensor,
        beat_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decision is None or not self.decision.approved:
            raise RuntimeError("Fusion is disabled until held-out complementarity is demonstrated")
        direct_precision = direct_std_ms.clamp_min(1.0).square().reciprocal()
        beat_std = (80.0 * (1.0 - beat_confidence.unsqueeze(-1)) + 5.0).clamp_min(1.0)
        beat_precision = beat_std.square().reciprocal()
        disagreement = (direct_ms - beat_ms).abs().mean(-1, keepdim=True)
        beat_precision = beat_precision * torch.exp(-disagreement / 50.0)
        total = direct_precision + beat_precision
        fused = (direct_precision * direct_ms + beat_precision * beat_ms) / total.clamp_min(1e-8)
        confidence = torch.exp(-total.rsqrt().mean(-1) / 50.0)
        return fused, confidence


def assess_complementarity(
    direct_abs_error: torch.Tensor,
    beat_abs_error: torch.Tensor,
    fused_abs_error: torch.Tensor,
    min_gain_ms: float = 0.5,
) -> FusionDecision:
    direct = float(direct_abs_error.mean())
    beat = float(beat_abs_error.mean())
    fused = float(fused_abs_error.mean())
    best = min(direct, beat)
    gain = best - fused
    anti_correlated = False
    if direct_abs_error.numel() > 2:
        stacked = torch.stack((direct_abs_error.flatten(), beat_abs_error.flatten()))
        anti_correlated = bool(torch.corrcoef(stacked)[0, 1] < 0.85)
    approved = gain >= min_gain_ms and anti_correlated
    reason = f"gain={gain:.3f}ms, error_correlation_below_0.85={anti_correlated}"
    return FusionDecision(approved, reason, direct, beat, fused)

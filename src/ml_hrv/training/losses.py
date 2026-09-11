"""Uncertainty, multi-view, and delay-aware beat objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ml_hrv.config import ExperimentConfig

from .targets import build_ecg_beat_targets, shift_candidates_with_zeros, shift_with_zeros


@dataclass
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, target_ms: np.ndarray) -> "TargetScaler":
        transformed = np.log(np.clip(np.asarray(target_ms, dtype=float), 1e-3, None))
        return cls(transformed.mean(0), transformed.std(0).clip(1e-6))

    def encode(self, target_ms: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=target_ms.dtype, device=target_ms.device)
        std = torch.as_tensor(self.std, dtype=target_ms.dtype, device=target_ms.device)
        return (torch.log(target_ms.clamp_min(1e-3)) - mean) / std

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=normalized.dtype, device=normalized.device)
        std = torch.as_tensor(self.std, dtype=normalized.dtype, device=normalized.device)
        return torch.exp(normalized * std + mean)

    def std_ms(self, normalized_loc: torch.Tensor, normalized_logvar: torch.Tensor) -> torch.Tensor:
        """Delta-method standard deviation in milliseconds."""
        prediction = self.decode(normalized_loc)
        scale = torch.as_tensor(self.std, dtype=prediction.dtype, device=prediction.device)
        return prediction * scale * torch.exp(0.5 * normalized_logvar)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "transform": ["log", "log"]}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "TargetScaler":
        return cls(np.asarray(state["mean"], dtype=float), np.asarray(state["std"], dtype=float))


def heteroscedastic_loss(loc: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (0.5 * (torch.exp(-logvar) * (target - loc).square() + logvar)).mean()


def multiview_consistency_loss(loc: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group in torch.unique(group_id):
        selected = loc[group_id == group]
        if len(selected) > 1:
            losses.append((selected - selected.mean(0, keepdim=True)).square().mean())
    return torch.stack(losses).mean() if losses else loc.sum() * 0.0


class DelayAwareBeatLoss:
    """Align ECG events to later PPG pulses by latent per-window delay search."""

    def __init__(self, fs_hz: float, delay_min_ms: float, delay_max_ms: float, delay_step_ms: float = 20.0):
        self.fs_hz = fs_hz
        self.min_shift = round(delay_min_ms * fs_hz / 1000.0)
        self.max_shift = round(delay_max_ms * fs_hz / 1000.0)
        self.step = max(1, round(delay_step_ms * fs_hz / 1000.0))

    def __call__(
        self,
        logits: torch.Tensor,
        offsets_ms: torch.Tensor,
        predicted_delay_ms: torch.Tensor,
        rpeaks_ms: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = logits.shape[0]
        flat_logits = logits.reshape(batch, -1)
        flat_offsets = offsets_ms.reshape(batch, -1)
        heatmap, offset_target = build_ecg_beat_targets(
            rpeaks_ms, flat_logits.shape[-1], self.fs_hz, device=flat_logits.device
        )
        candidates = torch.arange(self.min_shift, self.max_shift + 1, self.step, device=flat_logits.device)
        with torch.no_grad():
            probabilities = flat_logits.sigmoid()
            candidate_targets = shift_candidates_with_zeros(heatmap, candidates)
            positive = (probabilities[:, None, :] * candidate_targets).sum(-1)
            positive = positive / candidate_targets.sum(-1).clamp_min(1.0)
            score_matrix = positive - 0.05 * probabilities.mean(-1, keepdim=True)
            selected_shift = candidates[score_matrix.argmax(1)]
        aligned_heatmap = shift_with_zeros(heatmap, selected_shift)
        aligned_offsets = shift_with_zeros(offset_target, selected_shift)
        weights = 1.0 + 12.0 * aligned_heatmap
        heatmap_loss = (F.binary_cross_entropy_with_logits(flat_logits, aligned_heatmap, reduction="none") * weights).mean()
        center = aligned_heatmap >= 0.95
        offset_loss = F.smooth_l1_loss(flat_offsets[center], aligned_offsets[center]) if center.any() else flat_offsets.sum() * 0.0
        selected_delay_ms = selected_shift.to(logits.dtype) * 1000.0 / self.fs_hz
        delay_loss = F.smooth_l1_loss(predicted_delay_ms, selected_delay_ms)
        return heatmap_loss + 0.02 * delay_loss, offset_loss, selected_delay_ms


class PipelineLoss:
    def __init__(self, config: ExperimentConfig, scaler: TargetScaler) -> None:
        self.config = config
        self.scaler = scaler
        self.beat = DelayAwareBeatLoss(
            config.data.fs_hz, config.model.delay_min_ms, config.model.delay_max_ms
        )

    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, object]) -> dict[str, torch.Tensor]:
        if self.config.train.stage == "beat_pretrain":
            # ECG event supervision only: the short crop must never inherit or
            # optimize a 5-minute RMSSD/SDNN scalar label.
            direct = outputs["direct_loc"].sum() * 0.0
            consistency = direct
            total = direct
        else:
            target_ms = batch["target_ms"]
            assert isinstance(target_ms, torch.Tensor)
            target = self.scaler.encode(target_ms)
            direct = heteroscedastic_loss(outputs["direct_loc"], outputs["direct_logvar"], target)
            consistency = multiview_consistency_loss(outputs["direct_loc"], batch["group_id"])
            total = direct + self.config.train.consistency_weight * consistency
        losses: dict[str, torch.Tensor] = {"direct": direct, "consistency": consistency}
        if self.config.train.stage in {"beat_pretrain", "beat", "joint"}:
            beat_loss, offset_loss, delay_target = self.beat(
                outputs["beat_heatmap_logits"],
                outputs["beat_offset_ms"],
                outputs["pulse_delay_ms"],
                batch["rpeaks_ms"],
            )
            sqi_target = (outputs["segment_coverage"].detach() >= 0.90).to(outputs["beat_sqi_logits"].dtype)
            sqi_loss = F.binary_cross_entropy_with_logits(outputs["beat_sqi_logits"], sqi_target)
            if self.config.train.stage == "beat":
                total = total * 0.05
            total = (
                total
                + self.config.train.beat_weight * beat_loss
                + self.config.train.offset_weight * offset_loss
                + self.config.train.sqi_weight * sqi_loss
            )
            losses.update(beat=beat_loss, offset=offset_loss, sqi=sqi_loss, delay_target_ms=delay_target.mean())
        losses["total"] = total
        return losses

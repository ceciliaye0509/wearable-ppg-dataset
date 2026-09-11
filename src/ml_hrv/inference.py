"""Stateful causal 5-minute / 30-second-update deployment wrapper."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ml_hrv import DEVICE_NAMES
from ml_hrv.config import ExperimentConfig
from ml_hrv.models.fusion import UncertaintyFusion
from ml_hrv.physiology import physiological_hrv
from ml_hrv.training.losses import TargetScaler


@dataclass(frozen=True)
class HRVOutput:
    rmssd_ms: float
    sdnn_ms: float
    confidence: float
    reject_reason: str
    source: str
    ready: bool


def relative_timestamp_jitter(
    raw_timestamp_ms: np.ndarray,
    grid_timestamp_ms: np.ndarray,
    mask: np.ndarray,
    fs_hz: float,
) -> np.ndarray:
    delta = (np.asarray(raw_timestamp_ms) - np.asarray(grid_timestamp_ms)) / (1000.0 / fs_hz)
    return np.where(mask & np.isfinite(delta), np.clip(delta, -2.0, 2.0), 0.0).astype(np.float32)


class CausalHRVMonitor:
    """Consume exactly one update block and emit only after causal warm-up."""

    def __init__(
        self,
        model: torch.nn.Module,
        scaler: TargetScaler,
        config: ExperimentConfig,
        device_name: str,
        torch_device: torch.device | str = "cpu",
        beat_trained: bool = False,
        fusion: UncertaintyFusion | None = None,
    ) -> None:
        self.model = model.to(torch_device).eval()
        self.scaler = scaler
        self.config = config
        self.device_id = DEVICE_NAMES.index(device_name)
        self.torch_device = torch.device(torch_device)
        self.beat_trained = beat_trained
        self.fusion = fusion
        self.samples_per_segment = round(config.data.fs_hz * config.data.segment_seconds)
        self.segments_per_update = config.data.update_seconds // config.data.segment_seconds
        self.required_segments = config.data.window_seconds // config.data.segment_seconds
        if config.data.update_seconds % config.data.segment_seconds:
            raise ValueError("update_seconds must contain whole encoder segments")
        self.values: deque[np.ndarray] = deque(maxlen=self.required_segments)
        self.masks: deque[np.ndarray] = deque(maxlen=self.required_segments)
        self.jitters: deque[np.ndarray] = deque(maxlen=self.required_segments)

    def update(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        jitter: np.ndarray,
        accel_motion_mean_mag: float,
    ) -> HRVOutput:
        expected = (self.segments_per_update, 2, self.samples_per_segment)
        if values.shape != expected or mask.shape != expected or jitter.shape != expected:
            raise ValueError(f"Each update must have shape {expected}")
        for index in range(self.segments_per_update):
            self.values.append(np.asarray(values[index], dtype=np.float32))
            self.masks.append(np.asarray(mask[index], dtype=np.bool_))
            self.jitters.append(np.asarray(jitter[index], dtype=np.float32))
        if len(self.values) < self.required_segments:
            return HRVOutput(float("nan"), float("nan"), 0.0, "causal_warmup", "none", False)

        values_t = torch.from_numpy(np.stack(self.values))[None].to(self.torch_device)
        mask_t = torch.from_numpy(np.stack(self.masks))[None].to(self.torch_device)
        jitter_t = torch.from_numpy(np.stack(self.jitters))[None].to(self.torch_device)
        device_t = torch.tensor([self.device_id], device=self.torch_device)
        accel_t = torch.tensor([[accel_motion_mean_mag]], dtype=torch.float32, device=self.torch_device)
        with torch.no_grad():
            output = self.model(values_t, mask_t, jitter_t, device_t, accel_t, compute_beat=self.beat_trained)
            direct = self.scaler.decode(output["direct_loc"])[0]
            direct_std = self.scaler.std_ms(output["direct_loc"], output["direct_logvar"])[0]
            confidence = float(torch.exp(-direct_std.mean() / 50.0))
            prediction = direct
            source = "direct"
            beat = None
            if self.beat_trained:
                beat = physiological_hrv(
                    output["beat_heatmap_logits"].sigmoid().cpu().numpy(),
                    output["beat_offset_ms"].cpu().numpy(),
                    output["beat_sqi_logits"].sigmoid().cpu().numpy(),
                    output["segment_coverage"].cpu().numpy(),
                    self.config.data.fs_hz,
                )
            if (
                beat is not None
                and not beat.reject_reason
                and self.fusion is not None
                and np.isfinite([beat.rmssd_ms, beat.sdnn_ms]).all()
            ):
                beat_tensor = torch.tensor([[beat.rmssd_ms, beat.sdnn_ms]], device=self.torch_device)
                prediction, fused_confidence = self.fusion(
                    direct[None], direct_std[None], beat_tensor, torch.tensor([beat.confidence], device=self.torch_device)
                )
                prediction = prediction[0]
                confidence = float(fused_confidence[0])
                source = "direct_beat_fusion"

        coverage = float(np.mean(np.stack(self.masks)))
        reasons: list[str] = []
        if coverage < 0.90:
            reasons.append("low_ppg_coverage")
        if confidence < self.config.evaluation.reject_confidence_below:
            reasons.append("high_uncertainty")
        return HRVOutput(
            float(prediction[0]),
            float(prediction[1]),
            confidence,
            "|".join(dict.fromkeys(reasons)),
            source,
            True,
        )

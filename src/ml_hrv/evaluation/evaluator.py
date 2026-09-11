"""Convert both model paths to auditable per-window predictions."""

from __future__ import annotations

import math

import numpy as np
import torch

from ml_hrv.config import ExperimentConfig
from ml_hrv.physiology import physiological_hrv
from ml_hrv.training.losses import TargetScaler

from .beat_events import match_beat_events


def _move(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate_model(
    model: torch.nn.Module,
    loader,
    scaler: TargetScaler,
    config: ExperimentConfig,
    device: torch.device,
    fold_name: str,
    beat_trained: bool,
    max_batches: int = 0,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = _move(raw_batch, device)
            outputs = model(
                batch["values"], batch["mask"], batch["jitter"], batch["device_id"], batch["accel"],
                compute_beat=beat_trained,
            )
            direct_ms = scaler.decode(outputs["direct_loc"])
            direct_std_ms = scaler.std_ms(outputs["direct_loc"], outputs["direct_logvar"])
            direct_confidence = torch.exp(-direct_std_ms.mean(-1) / 50.0)
            heatmap = offset = sqi = None
            if beat_trained:
                heatmap = outputs["beat_heatmap_logits"].sigmoid().reshape(len(direct_ms), -1).cpu().numpy()
                offset = outputs["beat_offset_ms"].reshape(len(direct_ms), -1).cpu().numpy()
                sqi = outputs["beat_sqi_logits"].sigmoid().cpu().numpy()
            segment_coverage = outputs["segment_coverage"].cpu().numpy()
            for index in range(len(direct_ms)):
                beat = None
                if beat_trained:
                    assert heatmap is not None and offset is not None and sqi is not None
                    beat = physiological_hrv(
                        heatmap[index], offset[index], sqi[index], segment_coverage[index], config.data.fs_hz
                    )
                pulse_delay_ms = float(outputs["pulse_delay_ms"][index]) if beat_trained else math.nan
                event = None
                if beat is not None:
                    reference_peaks = np.asarray(raw_batch["rpeaks_ms"][index], dtype=float)
                    event = match_beat_events(beat.beat_times_ms, reference_peaks, pulse_delay_ms)
                confidence = float(direct_confidence[index])
                reasons: list[str] = []
                coverage = float(raw_batch["coverage"][index])
                # float32 representation of an exact 0.90 ratio can be
                # 0.899999976; do not turn that into a spurious rejection.
                if coverage < 0.90 - 1e-6:
                    reasons.append("low_ppg_coverage")
                if confidence < config.evaluation.reject_confidence_below:
                    reasons.append("high_direct_uncertainty")
                target = raw_batch["target_ms"][index]
                row = {
                    "fold": fold_name,
                    "participant": raw_batch["participant"][index],
                    "window_index": int(raw_batch["window_index"][index]),
                    "window_start_ms": float(raw_batch["window_start_ms"][index]),
                    "device": raw_batch["device"][index],
                    "label_source": raw_batch["label_source"][index],
                    "coverage": coverage,
                    "accel_motion_mean_mag": float(raw_batch["motion_scalar"][index, 0]),
                    "target_rmssd_ms": float(target[0]),
                    "target_sdnn_ms": float(target[1]),
                    "direct_rmssd_ms": float(direct_ms[index, 0]),
                    "direct_sdnn_ms": float(direct_ms[index, 1]),
                    "direct_rmssd_std_ms": float(direct_std_ms[index, 0]),
                    "direct_sdnn_std_ms": float(direct_std_ms[index, 1]),
                    "beat_rmssd_ms": beat.rmssd_ms if beat else math.nan,
                    "beat_sdnn_ms": beat.sdnn_ms if beat else math.nan,
                    "beat_confidence": beat.confidence if beat else math.nan,
                    "beat_reject_reason": beat.reject_reason if beat else "not_evaluated",
                    "beat_count": len(beat.beat_times_ms) if beat else 0,
                    "beat_event_tp": event.tp if event else 0,
                    "beat_event_fp": event.fp if event else 0,
                    "beat_event_fn": event.fn if event else 0,
                    "beat_event_precision": event.precision if event else math.nan,
                    "beat_event_recall": event.recall if event else math.nan,
                    "beat_event_f1": event.f1 if event else math.nan,
                    "beat_timing_mae_ms": event.timing_mae_ms if event else math.nan,
                    "beat_count_error": event.count_error if event else 0,
                    "rr_correction_ratio": beat.corrected_rr.correction_ratio if beat else math.nan,
                    "pulse_delay_ms": pulse_delay_ms,
                    # Fusion is intentionally not selected in phase one.
                    "prediction_rmssd_ms": float(direct_ms[index, 0]),
                    "prediction_sdnn_ms": float(direct_ms[index, 1]),
                    "confidence": confidence,
                    "reject_reason": "|".join(dict.fromkeys(reasons)),
                    "accepted": not reasons,
                    "prediction_source": "direct",
                }
                rows.append(row)
            if max_batches and batch_index + 1 >= max_batches:
                break
    return rows

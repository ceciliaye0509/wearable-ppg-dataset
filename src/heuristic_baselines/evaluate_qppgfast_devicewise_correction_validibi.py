"""Full-cohort qppgfast devicewise ablation: correction and valid-IBI gate only."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rawslots_baseline_ablation import _dataset_npz_files
from evaluate_rawslots_detector_fiducials import DATASET_DIR, DetectorPipeline, _load_channel_metrics_for_path


DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
CONDITIONS = (
    ("correction_off_no_gate", False, None),
    ("correction_on_no_gate", True, None),
    ("correction_on_valid_ibi_ge085", True, 0.85),
)


def _valid(group: pd.DataFrame, minimum_valid_ibi: float | None) -> pd.Series:
    valid = (
        np.isfinite(group["ppg_rmssd_ms"].astype(float))
        & np.isfinite(group["ppg_sdnn_ms"].astype(float))
        & np.isfinite(group["ecg_rmssd_ms"].astype(float))
        & np.isfinite(group["ecg_sdnn_ms"].astype(float))
    )
    if minimum_valid_ibi is not None:
        valid &= group["ppg_valid_ibi_ratio"].astype(float) >= minimum_valid_ibi
    return valid


def _summarize(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for condition, correction, gate in CONDITIONS:
        source = selected[selected["ibi_correction"].eq(correction)]
        for (device, channel), group in source.groupby(["device", "channel"], sort=False):
            valid = _valid(group, gate)
            value = group[valid]
            strict = group["ppg_max_raw_gap_ms"].astype(float) <= 100.0
            rmssd = value["ppg_rmssd_ms"].astype(float)
            ecg_rmssd = value["ecg_rmssd_ms"].astype(float)
            sdnn = value["ppg_sdnn_ms"].astype(float)
            ecg_sdnn = value["ecg_sdnn_ms"].astype(float)
            r = float(np.corrcoef(rmssd, ecg_rmssd)[0, 1]) if len(value) >= 3 and rmssd.std() > 0 and ecg_rmssd.std() > 0 else np.nan
            rows.append({
                "condition": condition,
                "ibi_correction": correction,
                "valid_ibi_ratio_gate": gate if gate is not None else "none",
                "device": device,
                "channel": channel,
                "n_total": len(group),
                "n_strict_input": int(strict.sum()),
                "strict_input_coverage_pct": 100.0 * float(strict.mean()),
                "n_valid": int(valid.sum()),
                "computable_hrv_coverage_pct": 100.0 * float(valid.mean()),
                "RMSSD_MAE_ms": float(np.abs(rmssd - ecg_rmssd).mean()) if len(value) else np.nan,
                "SDNN_MAE_ms": float(np.abs(sdnn - ecg_sdnn).mean()) if len(value) else np.nan,
                "RMSSD_R": r,
            })
    channel = pd.DataFrame(rows)
    device = channel.groupby(["condition", "ibi_correction", "valid_ibi_ratio_gate", "device"], as_index=False).agg(
        channels=("channel", "size"),
        strict_input_coverage_pct=("strict_input_coverage_pct", "mean"),
        computable_hrv_coverage_pct=("computable_hrv_coverage_pct", "mean"),
        RMSSD_MAE_ms=("RMSSD_MAE_ms", "mean"),
        SDNN_MAE_ms=("SDNN_MAE_ms", "mean"),
        RMSSD_R=("RMSSD_R", "mean"),
    )
    overall = device.groupby(["condition", "ibi_correction", "valid_ibi_ratio_gate"], as_index=False).agg(
        devices=("device", "size"),
        strict_input_coverage_pct=("strict_input_coverage_pct", "mean"),
        min_strict_input_coverage_pct=("strict_input_coverage_pct", "min"),
        computable_hrv_coverage_pct=("computable_hrv_coverage_pct", "mean"),
        min_computable_hrv_coverage_pct=("computable_hrv_coverage_pct", "min"),
        RMSSD_MAE_ms=("RMSSD_MAE_ms", "mean"),
        SDNN_MAE_ms=("SDNN_MAE_ms", "mean"),
        RMSSD_R=("RMSSD_R", "mean"),
    )
    return channel, device, overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    frames: list[pd.DataFrame] = []
    for path in _dataset_npz_files(args.dataset_dir):
        with np.load(path, allow_pickle=True) as z:
            n_windows = int(np.asarray(z["ecg_rmssd_corrected_ms"]).shape[0])
        print(f"[qppgfast-correction] {path.stem} windows={n_windows}", flush=True)
        raw = _load_channel_metrics_for_path(
            path, np.arange(n_windows), (DetectorPipeline("qppgfast", (0.7, 3.5)),),
            input_mode="strict_interp100", max_gap_ms=100.0, correction_states=(False, True), polarity_mode="peak_train",
        )
        expected = raw["device"].map(DEVICE_FIDUCIALS)
        frames.append(raw[raw["fiducial"].eq(expected)].copy())
        del raw
        gc.collect()
    selected = pd.concat(frames, ignore_index=True)
    channel, device, overall = _summarize(selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "window_metrics.csv", index=False)
    channel.to_csv(args.out_dir / "channel_summary.csv", index=False)
    device.to_csv(args.out_dir / "device_summary.csv", index=False)
    overall.to_csv(args.out_dir / "overall_summary.csv", index=False)


if __name__ == "__main__":
    main()

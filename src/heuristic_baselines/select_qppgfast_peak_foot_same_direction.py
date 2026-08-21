"""Evaluate a qppgfast peak/foot selector after peak_train fixes polarity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["participant", "window_index", "device", "channel"]


def _corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.channel_metrics)
    df = df[(df.detector == "qppgfast") & (df.polarity_mode == "peak_train") & (df.common_bandpass == "bp070_350") & (df.ibi_correction == True)].copy()
    ranked = df.sort_values(KEYS + ["fiducial_ibi_score", "fiducial"], ascending=[True, True, True, True, False, False], kind="stable")
    selected = ranked.groupby(KEYS, as_index=False, sort=False).head(1).copy()
    selected["method"] = "peak_train_same_direction_peak_foot_selector"
    selected["selected_fiducial"] = selected["fiducial"]
    rows = []
    for keys, group in selected.groupby(["participant", "device", "channel"], sort=False):
        valid = np.isfinite(group.ppg_rmssd_ms) & np.isfinite(group.ppg_sdnn_ms) & np.isfinite(group.ecg_rmssd_ms) & np.isfinite(group.ecg_sdnn_ms)
        value = group.loc[valid]
        rows.append({"participant": keys[0], "device": keys[1], "channel": keys[2], "n_total": len(group), "n_valid": int(valid.sum()), "coverage_pct": 100 * float(valid.mean()), "peak_selected_pct": 100 * float((group.selected_fiducial == "peak").mean()), "foot_selected_pct": 100 * float((group.selected_fiducial == "foot").mean()), "MAE_ms": float(np.abs(value.ppg_rmssd_ms - value.ecg_rmssd_ms).mean()) if len(value) else np.nan, "R": _corr(value.ppg_rmssd_ms, value.ecg_rmssd_ms)})
    channel = pd.DataFrame(rows)
    device = channel.groupby("device", as_index=False).agg(participant_channels=("MAE_ms", "size"), coverage_pct=("coverage_pct", "mean"), peak_selected_pct=("peak_selected_pct", "mean"), foot_selected_pct=("foot_selected_pct", "mean"), MAE_ms=("MAE_ms", "mean"), R=("R", "mean"))
    overall = pd.DataFrame([{
        "participant_channels": int(len(channel)),
        "coverage_pct": float(channel.coverage_pct.mean()),
        "peak_selected_pct": float(channel.peak_selected_pct.mean()),
        "foot_selected_pct": float(channel.foot_selected_pct.mean()),
        "MAE_ms": float(channel.MAE_ms.mean()),
        "R": float(channel.R.mean()),
    }])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "window_metrics.csv", index=False)
    channel.to_csv(args.out_dir / "participant_channel_summary.csv", index=False)
    device.to_csv(args.out_dir / "device_summary.csv", index=False)
    overall.to_csv(args.out_dir / "overall_summary.csv", index=False)


if __name__ == "__main__":
    main()

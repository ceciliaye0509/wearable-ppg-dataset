"""Build compact report tables from the frozen qppgfast-devicewise full16 run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = {"qppgfast_devicewise_v1": 0, "scipy_peak_v1": 1}
DEVICE_ORDER = {"Earring": 0, "Ring": 1, "Watch": 2}
CHANNEL_ORDER = {"ppg_green": 0, "ppg_ir": 1}


def _stats(group: pd.DataFrame) -> dict[str, float | int]:
    ppg = group["ppg_rmssd_ms"].to_numpy(float)
    ecg = group["ecg_rmssd_ms"].to_numpy(float)
    sdnn = group["ppg_sdnn_ms"].to_numpy(float)
    valid = np.isfinite(ppg) & np.isfinite(ecg) & np.isfinite(sdnn)
    ppg, ecg = ppg[valid], ecg[valid]
    n_total = int(group.shape[0])
    n_valid = int(ppg.size)
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "coverage_pct": 100.0 * n_valid / n_total if n_total else np.nan,
        "MAE_ms": float(np.mean(np.abs(ppg - ecg))) if n_valid else np.nan,
        "R": float(np.corrcoef(ppg, ecg)[0, 1]) if n_valid >= 3 and np.std(ppg) > 0 and np.std(ecg) > 0 else np.nan,
    }


def _by_channel(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(["baseline_method", "device", "channel"], dropna=False):
        rows.append({"baseline_method": keys[0], "device": keys[1], "channel": keys[2], **_stats(group)})
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["baseline_method", "device", "channel"],
        key=lambda s: s.map(METHOD_ORDER if s.name == "baseline_method" else DEVICE_ORDER if s.name == "device" else CHANNEL_ORDER),
    ).reset_index(drop=True)


def _macro(by_channel: pd.DataFrame) -> pd.DataFrame:
    return (
        by_channel.groupby("baseline_method", dropna=False)
        .agg(
            channels=("channel", "size"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
            mean_MAE_ms=("MAE_ms", "mean"),
            mean_R=("R", "mean"),
            n_total=("n_total", "sum"),
            n_valid=("n_valid", "sum"),
        )
        .reset_index()
        .sort_values("mean_MAE_ms")
        .reset_index(drop=True)
    )


def _common_valid(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "window_index", "device", "channel"]
    scipy = metrics[metrics["baseline_method"] == "scipy_peak_v1"].copy()
    qppg = metrics[metrics["baseline_method"] == "qppgfast_devicewise_v1"].copy()
    cols = keys + ["ppg_rmssd_ms", "ppg_sdnn_ms", "ecg_rmssd_ms"]
    joined = qppg[cols].merge(scipy[cols], on=keys, suffixes=("_qppgfast", "_scipy"), validate="one_to_one")
    valid = (
        np.isfinite(joined["ppg_rmssd_ms_qppgfast"].to_numpy(float))
        & np.isfinite(joined["ppg_sdnn_ms_qppgfast"].to_numpy(float))
        & np.isfinite(joined["ppg_rmssd_ms_scipy"].to_numpy(float))
        & np.isfinite(joined["ppg_sdnn_ms_scipy"].to_numpy(float))
        & np.isfinite(joined["ecg_rmssd_ms_qppgfast"].to_numpy(float))
    )
    joined["common_valid"] = valid
    rows: list[dict[str, object]] = []
    for keys_value, group in joined.groupby(["device", "channel"], dropna=False):
        common = group[group["common_valid"]]
        ref = common["ecg_rmssd_ms_qppgfast"].to_numpy(float)
        qppg_pred = common["ppg_rmssd_ms_qppgfast"].to_numpy(float)
        scipy_pred = common["ppg_rmssd_ms_scipy"].to_numpy(float)
        n_common = int(common.shape[0])
        qppg_mae = float(np.mean(np.abs(qppg_pred - ref))) if n_common else np.nan
        scipy_mae = float(np.mean(np.abs(scipy_pred - ref))) if n_common else np.nan
        qppg_r = float(np.corrcoef(qppg_pred, ref)[0, 1]) if n_common >= 3 and np.std(qppg_pred) > 0 and np.std(ref) > 0 else np.nan
        scipy_r = float(np.corrcoef(scipy_pred, ref)[0, 1]) if n_common >= 3 and np.std(scipy_pred) > 0 and np.std(ref) > 0 else np.nan
        rows.append({
            "device": keys_value[0],
            "channel": keys_value[1],
            "n_total": int(group.shape[0]),
            "n_common_valid": n_common,
            "common_coverage_pct": 100.0 * n_common / int(group.shape[0]),
            "qppgfast_MAE_ms": qppg_mae,
            "scipy_MAE_ms": scipy_mae,
            "qppgfast_minus_scipy_MAE_ms": qppg_mae - scipy_mae,
            "qppgfast_R": qppg_r,
            "scipy_R": scipy_r,
            "qppgfast_minus_scipy_R": qppg_r - scipy_r,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["device", "channel"], key=lambda s: s.map(DEVICE_ORDER if s.name == "device" else CHANNEL_ORDER)
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report tables from frozen qppgfast full16 results.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_channel = _by_channel(metrics)
    macro = _macro(by_channel)
    common = _common_valid(metrics)
    by_channel.to_csv(args.out_dir / "rawslots_qppgfast_devicewise_v1_full16_by_channel.csv", index=False)
    macro.to_csv(args.out_dir / "rawslots_qppgfast_devicewise_v1_full16_macro.csv", index=False)
    common.to_csv(args.out_dir / "rawslots_qppgfast_devicewise_v1_full16_common_valid_vs_scipy_by_channel.csv", index=False)


if __name__ == "__main__":
    main()

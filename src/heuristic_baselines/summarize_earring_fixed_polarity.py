"""Summarize the full-cohort Earring fixed-polarity validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["participant", "window_index", "device", "channel"]
METHODS = ("fixed_positive", "fixed_negative", "peak_train")


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    return (
        np.isfinite(df["ppg_rmssd_ms"].astype(float))
        & np.isfinite(df["ppg_sdnn_ms"].astype(float))
        & np.isfinite(df["ecg_rmssd_ms"].astype(float))
        & np.isfinite(df["ecg_sdnn_ms"].astype(float))
    )


def _correlation(pred: pd.Series, ref: pd.Series) -> float:
    if len(pred) < 3 or float(pred.std()) == 0.0 or float(ref.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, ref)[0, 1])


def _load(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["device"] == "Earring") & (df["detector"] == "qppgfast") & (df["fiducial"] == "peak")].copy()
    if method == "peak_train":
        if "baseline_method" in df:
            df = df[df["baseline_method"] == "qppgfast_devicewise_v1"].copy()
        if "strict_input_pass" in df:
            df["strict_input_pass"] = df["strict_input_pass"].astype(bool)
        else:
            df["strict_input_pass"] = df["polarity"].ne("invalid")
    else:
        df["strict_input_pass"] = df["polarity"].ne("invalid")
    df["metric_valid"] = _valid_mask(df)
    if df.empty:
        raise RuntimeError(f"no Earring qppgfast-peak rows in {path}")
    return df


def _participant_channel_summary(df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (participant, channel), group in df.groupby(["participant", "channel"], dropna=False):
        valid = group[group["metric_valid"]]
        rows.append(
            {
                "method": method,
                "participant": participant,
                "channel": channel,
                "n_total": int(len(group)),
                "n_strict_input": int(group["strict_input_pass"].sum()),
                "n_valid": int(len(valid)),
                "strict_input_coverage_over_total_pct": 100.0 * float(group["strict_input_pass"].mean()),
                "coverage_over_total_pct": 100.0 * float(group["metric_valid"].mean()),
                "coverage_over_strict_input_pct": 100.0 * len(valid) / int(group["strict_input_pass"].sum()) if int(group["strict_input_pass"].sum()) else float("nan"),
                "MAE_ms": float(np.mean(np.abs(valid["ppg_rmssd_ms"] - valid["ecg_rmssd_ms"]))) if len(valid) else float("nan"),
                "R": _correlation(valid["ppg_rmssd_ms"], valid["ecg_rmssd_ms"]),
            }
        )
    return pd.DataFrame(rows)


def _primary_summary(raw: dict[str, pd.DataFrame], by_channel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, df in raw.items():
        channel = by_channel[by_channel["method"] == method]
        valid = df["metric_valid"]
        strict = df["strict_input_pass"]
        rows.append(
            {
                "method": method,
                "n_total": int(len(df)),
                "n_strict_input": int(strict.sum()),
                "n_valid": int(valid.sum()),
                "strict_input_coverage_over_total_pct": 100.0 * float(strict.mean()),
                "coverage_over_total_pct": 100.0 * float(valid.mean()),
                "coverage_over_strict_input_pct": 100.0 * float(valid.sum()) / int(strict.sum()),
                "mean_participant_channel_coverage_over_total_pct": float(channel["coverage_over_total_pct"].mean()),
                "min_participant_channel_coverage_over_total_pct": float(channel["coverage_over_total_pct"].min()),
                "mean_participant_channel_MAE_ms": float(channel["MAE_ms"].mean()),
                "mean_participant_channel_R": float(channel["R"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _common_valid_summary(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for method, df in raw.items():
        keep = df[KEYS + ["metric_valid", "ppg_rmssd_ms", "ecg_rmssd_ms"]].copy()
        keep = keep.rename(
            columns={
                "metric_valid": f"metric_valid_{method}",
                "ppg_rmssd_ms": f"ppg_rmssd_ms_{method}",
                "ecg_rmssd_ms": f"ecg_rmssd_ms_{method}",
            }
        )
        merged = keep if merged is None else merged.merge(keep, on=KEYS, how="inner", validate="one_to_one")
    assert merged is not None
    valid_columns = [f"metric_valid_{method}" for method in METHODS]
    common = merged[valid_columns].all(axis=1)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        per_channel: list[tuple[float, float]] = []
        for _, group in merged[common].groupby(["participant", "channel"], dropna=False):
            pred = group[f"ppg_rmssd_ms_{method}"]
            ref = group[f"ecg_rmssd_ms_{method}"]
            per_channel.append((float(np.mean(np.abs(pred - ref))), _correlation(pred, ref)))
        rows.append(
            {
                "method": method,
                "n_common_valid": int(common.sum()),
                "common_valid_coverage_over_total_pct": 100.0 * float(common.mean()),
                "mean_participant_channel_MAE_ms": float(np.mean([x[0] for x in per_channel])),
                "mean_participant_channel_R": float(np.mean([x[1] for x in per_channel])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Earring fixed-polarity validation outputs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rawslots_earring_fixed_polarity_full16_summary_v2"))
    args = parser.parse_args()
    sources = {
        "fixed_positive": args.outputs_root / "rawslots_earring_fixed_polarity_positive_full16_v1/rawslots_detector_fiducial_channel_metrics.csv",
        "fixed_negative": args.outputs_root / "rawslots_earring_fixed_polarity_negative_full16_v1/rawslots_detector_fiducial_channel_metrics.csv",
        "peak_train": args.outputs_root / "rawslots_earring_peak_train_polarity_full16_v2/rawslots_detector_fiducial_channel_metrics.csv",
    }
    raw = {method: _load(path, method) for method, path in sources.items()}
    by_channel = pd.concat([_participant_channel_summary(df, method) for method, df in raw.items()], ignore_index=True)
    primary = _primary_summary(raw, by_channel)
    common = _common_valid_summary(raw)
    polarity = (
        raw["peak_train"]
        .loc[raw["peak_train"]["strict_input_pass"], "polarity"]
        .value_counts()
        .rename_axis("polarity")
        .reset_index(name="n_windows")
    )
    polarity["fraction_of_strict_input_pct"] = 100.0 * polarity["n_windows"] / polarity["n_windows"].sum()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.out_dir / "primary_summary.csv", index=False)
    by_channel.to_csv(args.out_dir / "participant_channel_summary.csv", index=False)
    common.to_csv(args.out_dir / "common_valid_summary.csv", index=False)
    polarity.to_csv(args.out_dir / "peak_train_polarity_distribution.csv", index=False)
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()

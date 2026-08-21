"""Sweep pre-registered peak/foot/reject thresholds from cached qppgfast candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_qppgfast_adaptive_selector import _device_summary, _participant_channel_summary


DEFAULT_SCORES = (0.0, 1.0, 1.5, 2.0, 2.25, 2.5)
DEFAULT_MARGINS = (0.0, 0.01, 0.025, 0.05, 0.10)
def _parse_values(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _choose(top_cache: pd.DataFrame, min_score: float, min_margin: float) -> pd.DataFrame:
    """Thresholds only accept/reject the already-fixed highest-score candidate."""
    selected = top_cache.copy()
    strict_invalid = selected["selector_status"].eq("invalid_strict_input")
    accepted = (~strict_invalid) & (selected["best_score"] >= min_score) & (selected["score_margin"] >= min_margin)
    selected["selector_status"] = np.where(
        strict_invalid,
        "invalid_strict_input",
        np.where(selected["best_score"] < min_score, "reject_low_score", np.where(selected["score_margin"] < min_margin, "reject_ambiguous", "selected")),
    )
    selected["selected_polarity"] = np.where(accepted, selected["top_polarity"], "invalid")
    selected["selected_fiducial"] = np.where(accepted, selected["top_fiducial"], "reject")
    for metric in ("ppg_rmssd_ms", "ppg_sdnn_ms"):
        selected[metric] = np.where(accepted, selected[f"top_{metric}"], np.nan)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep cached qppgfast adaptive-selector candidates without rerunning detectors.")
    parser.add_argument("--top-candidate-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores", type=str, default=",".join(map(str, DEFAULT_SCORES)))
    parser.add_argument("--margins", type=str, default=",".join(map(str, DEFAULT_MARGINS)))
    args = parser.parse_args()
    top_cache = pd.read_csv(args.top_candidate_cache)
    scores, margins = _parse_values(args.scores), _parse_values(args.margins)
    overall_rows: list[pd.DataFrame] = []
    device_rows: list[pd.DataFrame] = []
    channel_rows: list[pd.DataFrame] = []
    for min_score in scores:
        for min_margin in margins:
            selected = _choose(top_cache, min_score, min_margin)
            channel = _participant_channel_summary(selected, "adaptive_peak_foot_reject")
            device = _device_summary(channel)
            overall = (
                channel.groupby("method", dropna=False)
                .agg(
                    participant_channels=("MAE_ms", "size"),
                    coverage_pct=("coverage_over_total_pct", "mean"),
                    peak_selected_pct=("peak_selected_pct", "mean"),
                    foot_selected_pct=("foot_selected_pct", "mean"),
                    reject_pct=("reject_pct", "mean"),
                    MAE_ms=("MAE_ms", "mean"),
                    R=("R", "mean"),
                ).reset_index()
            )
            for frame in (channel, device, overall):
                frame.insert(0, "min_margin", min_margin)
                frame.insert(0, "min_score", min_score)
            channel_rows.append(channel)
            device_rows.append(device)
            overall_rows.append(overall)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(overall_rows, ignore_index=True).to_csv(args.out_dir / "overall_grid.csv", index=False)
    pd.concat(device_rows, ignore_index=True).to_csv(args.out_dir / "device_grid.csv", index=False)
    pd.concat(channel_rows, ignore_index=True).to_csv(args.out_dir / "participant_channel_grid.csv", index=False)
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()

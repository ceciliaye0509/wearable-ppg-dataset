"""Create a reproducible raw-ECG signal-quality audit for participant P5.

The audit works on non-overlapping 20-s raw-ECG clips.  It intentionally
keeps morphology-detector outcomes separate from amplitude/noise indicators:
the latter describe signal quality, while the former identifies clips needing
manual review.  The resulting CSV is the sampling frame for a blinded human
assessment; it is not a clinical judgement about the participant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _independent_raw_r_candidates


CLIP_MS = 20_000.0
MIN_IBI_MS = 300.0
MAX_IBI_MS = 2_000.0


def _mad(x: np.ndarray) -> float:
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def _clip_metrics(t: np.ndarray, x: np.ndarray, clip_id: int, run_id: int) -> dict[str, object]:
    fs = float(1000.0 / np.median(np.diff(t)))
    q01, q99 = np.quantile(x, [0.01, 0.99])
    amplitude_span = float(q99 - q01)
    block = max(1, int(round(2.0 * fs)))
    block_medians = np.asarray([np.median(x[i : i + block]) for i in range(0, x.size, block)])
    baseline_wander = float(np.quantile(block_medians, 0.95) - np.quantile(block_medians, 0.05))
    derivative_noise_ratio = float(_mad(np.diff(x)) / max(_mad(x), 1e-6))
    severe = _has_severe_amplitude_artifact(x)
    peak_indices = np.empty(0, dtype=int) if severe else _independent_raw_r_candidates(x, t)
    peaks = t[peak_indices]
    ibi = np.diff(peaks)
    valid_ibi = (ibi >= MIN_IBI_MS) & (ibi <= MAX_IBI_MS)
    valid_ibi_ratio = float(valid_ibi.mean()) if ibi.size else 0.0
    valid_values = ibi[valid_ibi]
    mean_ibi = float(np.mean(valid_values)) if valid_values.size else float("nan")
    candidate_hr = float(60_000.0 / mean_ibi) if np.isfinite(mean_ibi) and mean_ibi > 0 else float("nan")
    ibi_cv = float(np.std(valid_values, ddof=1) / mean_ibi) if valid_values.size >= 3 else float("nan")
    reasons: list[str] = []
    if severe:
        reasons.append("severe_amplitude_artifact")
    if peak_indices.size < 6:
        reasons.append("too_few_morphology_candidates")
    if ibi.size and valid_ibi_ratio < 0.95:
        reasons.append("invalid_ibi")
    if np.isfinite(candidate_hr) and not 40.0 <= candidate_hr <= 220.0:
        reasons.append("candidate_hr_out_of_range")
    if np.isfinite(ibi_cv) and ibi_cv > 0.20:
        reasons.append("high_ibi_variability")
    if severe:
        tier = "artifact"
    elif reasons:
        tier = "uncertain"
    else:
        tier = "provisionally_assessable"
    return {
        "clip_id": clip_id,
        "run_id": run_id,
        "start_unix_ms": float(t[0]),
        "end_unix_ms": float(t[-1]),
        "duration_seconds": float((t[-1] - t[0]) / 1000.0),
        "sample_count": int(x.size),
        "sampling_rate_hz": fs,
        "amplitude_p01_p99_uv": amplitude_span,
        "raw_mad_uv": _mad(x),
        "baseline_wander_p05_p95_uv": baseline_wander,
        "derivative_noise_ratio": derivative_noise_ratio,
        "severe_amplitude_artifact": severe,
        "candidate_count": int(peak_indices.size),
        "candidate_hr_bpm": candidate_hr,
        "valid_ibi_ratio": valid_ibi_ratio,
        "ibi_cv": ibi_cv,
        "auto_quality_tier": tier,
        "auto_qc_reasons": "|".join(reasons),
    }


def _select_manual_sample(frame: pd.DataFrame, per_tier: int, seed: int) -> pd.DataFrame:
    """Seeded, time-stratified draw from each automatic tier."""
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for tier, group in frame.groupby("auto_quality_tier", sort=True):
        target = min(per_tier, len(group))
        if target == 0:
            continue
        # Split chronological order into bins before sampling so one long run
        # cannot dominate the blinded human sample.
        ordered = group.sort_values("start_unix_ms").reset_index(drop=True)
        bins = np.array_split(np.arange(len(ordered)), target)
        positions = [int(rng.choice(chunk)) for chunk in bins if len(chunk)]
        draw = ordered.iloc[positions].copy()
        draw["manual_sampling_stratum"] = tier
        selected.append(draw)
    result = pd.concat(selected, ignore_index=True).sort_values("start_unix_ms").reset_index(drop=True)
    result.insert(0, "manual_review_id", [f"P5_SQI_{i:03d}" for i in range(1, len(result) + 1)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("Multisite-PPG/raw_data/P5/P5_polar_ecg_raw.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/p5_ecg_signal_quality_v1"))
    parser.add_argument("--manual-per-tier", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.manual_per_tier < 1:
        raise ValueError("--manual-per-tier must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.raw, allow_pickle=True) as raw:
        signal = np.asarray(raw["ecg_uv"], dtype=float)
        times = np.asarray(raw["time_ms"], dtype=float)
    breaks = np.flatnonzero(np.diff(times) > 250.0) + 1
    bounds = np.r_[0, breaks, len(times)]
    rows: list[dict[str, object]] = []
    clip_id = 0
    for run_id, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True), start=1):
        run_t, run_x = times[lo:hi], signal[lo:hi]
        for start in np.arange(float(run_t[0]), float(run_t[-1]) - CLIP_MS + 1.0, CLIP_MS):
            left = int(np.searchsorted(run_t, start, side="left"))
            right = int(np.searchsorted(run_t, start + CLIP_MS, side="left"))
            if right - left < 30:
                continue
            clip_id += 1
            rows.append(_clip_metrics(run_t[left:right], run_x[left:right], clip_id, run_id))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "p5_20s_raw_ecg_qc.csv", index=False)
    manual = _select_manual_sample(metrics, args.manual_per_tier, args.seed)
    manual.to_csv(args.out_dir / "p5_manual_blinded_sampling_manifest.csv", index=False)
    summary = {
        "version": "p5_raw_ecg_signal_quality_v1",
        "participant": "P5",
        "raw_source": str(args.raw.resolve()),
        "clip_seconds": CLIP_MS / 1000.0,
        "clips": int(len(metrics)),
        "time_gaps": int(len(breaks)),
        "manual_sampling_seed": args.seed,
        "manual_clips": int(len(manual)),
        "tier_counts": {str(k): int(v) for k, v in metrics["auto_quality_tier"].value_counts().sort_index().items()},
        "median_metrics": {
            key: float(metrics[key].median())
            for key in ("amplitude_p01_p99_uv", "raw_mad_uv", "baseline_wander_p05_p95_uv", "derivative_noise_ratio", "valid_ibi_ratio")
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

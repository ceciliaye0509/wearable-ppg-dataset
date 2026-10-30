"""Evaluate ECG R-peak candidates against dual blinded full-window annotations.

This script deliberately *evaluates* candidate methods; it does not select a
winner by assumption.  Its main selection metric is corrected five-minute
RMSSD error against the manual consensus, alongside peak timing and event
errors.  Run it only after two independent CSVs exported from
``build_full_window_blinded_ecg_review.py`` are available.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


CONSENSUS_TOLERANCE_MS = 40.0
EVENT_TOLERANCE_MS = 150.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0
IBI_CORRECTION_THRESHOLD = 0.20


def _infer_fs(times_ms: np.ndarray) -> float:
    delta = np.diff(np.asarray(times_ms, dtype=float))
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if not delta.size:
        raise ValueError("cannot infer ECG sample rate from timestamps")
    return float(1000.0 / np.median(delta))


def _one_to_one_match(a: np.ndarray, b: np.ndarray, tolerance_ms: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Chronological one-to-one matching for sorted timestamp arrays."""
    i = j = 0
    matched: list[tuple[int, int]] = []
    only_a: list[int] = []
    only_b: list[int] = []
    while i < len(a) and j < len(b):
        delta = float(a[i] - b[j])
        if abs(delta) <= tolerance_ms:
            matched.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            only_a.append(i)
            i += 1
        else:
            only_b.append(j)
            j += 1
    only_a.extend(range(i, len(a)))
    only_b.extend(range(j, len(b)))
    return matched, only_a, only_b


def _corrected_rmssd(times_ms: np.ndarray) -> tuple[float, float, int, int]:
    """Match the release's corrected-RR rule and report its intervention count."""
    rr = np.diff(np.asarray(times_ms, dtype=float))
    nn = rr[(rr >= IBI_MIN_MS) & (rr <= IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan"), float("nan"), int(nn.size), 0
    corrected = nn.copy()
    if corrected.size >= 5:
        local_median = median_filter(corrected, size=11, mode="reflect")
        replace = np.abs(corrected - local_median) > IBI_CORRECTION_THRESHOLD * local_median
        corrected[replace] = local_median[replace]
        corrected_count = int(replace.sum())
    else:
        corrected_count = 0
    rmssd = float(np.sqrt(np.mean(np.diff(corrected) ** 2)))
    return rmssd, float(corrected_count / corrected.size), int(nn.size), corrected_count


def _pan_tompkins(segment: np.ndarray, fs: float) -> np.ndarray:
    clean = nk.ecg_clean(segment, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def _neurokit(segment: np.ndarray, fs: float) -> np.ndarray:
    clean = nk.ecg_clean(segment, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def _refine_to_raw_apex(signal: np.ndarray, indices: np.ndarray, fs: float) -> np.ndarray:
    """Polarity-aware raw-apex refinement around detector candidates.

    This method is intentionally only a candidate in the comparison.  It uses
    a global dominant-QRS polarity and a +/-150 ms local search, then relies on
    the manual full-window validation set to establish whether that rule is
    appropriate for this lead and recording population.
    """
    x = np.asarray(signal, dtype=float)
    if not x.size:
        return np.empty(0, dtype=int)
    baseline = float(np.median(x))
    positive = float(np.quantile(x, 0.995) - baseline)
    negative = float(baseline - np.quantile(x, 0.005))
    polarity = 1.0 if positive >= negative else -1.0
    radius = max(1, int(round(0.150 * fs)))
    refined: list[int] = []
    for peak in np.asarray(indices, dtype=int):
        lo, hi = max(0, peak - radius), min(len(x), peak + radius + 1)
        if lo < hi:
            refined.append(lo + int(np.argmax(polarity * x[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


def _candidate_peak_times(signal: np.ndarray, times: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    pan = _pan_tompkins(signal, fs)
    candidates["pantompkins1985"] = times[pan[(pan >= 0) & (pan < len(times))]]
    refined = _refine_to_raw_apex(signal, pan, fs)
    candidates["pantompkins1985_raw_apex_refined"] = times[refined]
    try:
        neurokit = _neurokit(signal, fs)
        candidates["neurokit"] = times[neurokit[(neurokit >= 0) & (neurokit < len(times))]]
    except Exception:
        # A failed candidate is reported as unavailable rather than silently
        # allowing a different method to stand in for it.
        candidates["neurokit"] = np.empty(0, dtype=float)
    return candidates


def main() -> None:
    root = Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/full_window_blinded_ecg_review_v2_training_cohort")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "full_window_review_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "reconciled_results")
    args = parser.parse_args()

    reviewer_a = pd.read_csv(args.reviewer_a)
    reviewer_b = pd.read_csv(args.reviewer_b)
    manifest = pd.read_csv(args.manifest)
    required = {"review_id", "participant", "window_start_unix_ms", "manual_rpeak_abs_unix_ms"}
    flags_by_reviewer: dict[str, dict[str, int]] = {}
    cleaned_reviewers: dict[str, pd.DataFrame] = {}
    for name, table in {"reviewer_a": reviewer_a, "reviewer_b": reviewer_b}.items():
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        notes = table["notes"].fillna("").astype(str) if "notes" in table.columns else pd.Series("", index=table.index)
        flagged = notes.eq("unresolvable_raw_ecg_chunk")
        flags_by_reviewer[name] = table.loc[flagged].groupby("review_id").size().astype(int).to_dict()
        table = table.loc[~flagged].copy()
        table["manual_rpeak_abs_unix_ms"] = pd.to_numeric(table["manual_rpeak_abs_unix_ms"], errors="coerce")
        cleaned_reviewers[name] = table.loc[table["manual_rpeak_abs_unix_ms"].notna()].copy()
    reviewer_a, reviewer_b = cleaned_reviewers["reviewer_a"], cleaned_reviewers["reviewer_b"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    consensus_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for _, window in manifest.iterrows():
        review_id = str(window.review_id)
        start = float(window.window_start_unix_ms)
        end = float(window.window_end_unix_ms)
        raw_path = str(window.raw_ecg_path)
        if raw_path not in raw_cache:
            with np.load(raw_path, allow_pickle=True) as raw_file:
                raw = np.asarray(raw_file["ecg_uv"], dtype=float)
                raw_times = np.asarray(raw_file["time_ms"], dtype=float)
            raw_cache[raw_path] = raw, raw_times, _infer_fs(raw_times)
        raw, raw_times, fs = raw_cache[raw_path]
        lo = int(np.searchsorted(raw_times, start, side="left"))
        hi = int(np.searchsorted(raw_times, end, side="right"))
        signal, times = raw[lo:hi], raw_times[lo:hi]

        a = np.sort(reviewer_a.loc[reviewer_a.review_id.astype(str).eq(review_id), "manual_rpeak_abs_unix_ms"].to_numpy(float))
        b = np.sort(reviewer_b.loc[reviewer_b.review_id.astype(str).eq(review_id), "manual_rpeak_abs_unix_ms"].to_numpy(float))
        reviewer_a_unresolvable_chunks = int(flags_by_reviewer["reviewer_a"].get(review_id, 0))
        reviewer_b_unresolvable_chunks = int(flags_by_reviewer["reviewer_b"].get(review_id, 0))
        manual_window_eligible = reviewer_a_unresolvable_chunks == 0 and reviewer_b_unresolvable_chunks == 0
        agreement, only_a, only_b = _one_to_one_match(a, b, CONSENSUS_TOLERANCE_MS)
        manual = np.asarray([(a[i] + b[j]) / 2.0 for i, j in agreement], dtype=float)
        for number, ((i, j), apex) in enumerate(zip(agreement, manual, strict=True), start=1):
            consensus_rows.append({
                "review_id": review_id,
                "participant": window.participant,
                "manual_peak_number": number,
                "reviewer_a_unix_ms": a[i],
                "reviewer_b_unix_ms": b[j],
                "consensus_unix_ms": apex,
                "reviewer_a_minus_b_ms": a[i] - b[j],
            })
        manual_rmssd, manual_correction_ratio, manual_valid_ibi, manual_corrected_ibi = _corrected_rmssd(manual)
        for method, detector in _candidate_peak_times(signal, times, fs).items():
            matched, manual_only, detector_only = _one_to_one_match(manual, detector, EVENT_TOLERANCE_MS)
            deltas = np.asarray([detector[j] - manual[i] for i, j in matched], dtype=float)
            detector_rmssd, detector_correction_ratio, detector_valid_ibi, detector_corrected_ibi = _corrected_rmssd(detector)
            precision = len(matched) / len(detector) if len(detector) else float("nan")
            recall = len(matched) / len(manual) if len(manual) else float("nan")
            summary_rows.append({
                "review_id": review_id,
                "participant": window.participant,
                "candidate_method": method,
                "reviewer_a_peaks": len(a),
                "reviewer_b_peaks": len(b),
                "reviewer_a_unresolvable_chunks": reviewer_a_unresolvable_chunks,
                "reviewer_b_unresolvable_chunks": reviewer_b_unresolvable_chunks,
                "manual_window_eligible": manual_window_eligible,
                "manual_consensus_peaks": len(manual),
                "reviewer_agreement_pct_of_mean_count": 100.0 * len(agreement) / max(1.0, (len(a) + len(b)) / 2.0),
                "reviewer_a_only": len(only_a),
                "reviewer_b_only": len(only_b),
                "manual_corrected_rmssd_ms": manual_rmssd,
                "manual_correction_ratio": manual_correction_ratio,
                "manual_valid_ibi": manual_valid_ibi,
                "manual_corrected_ibi": manual_corrected_ibi,
                "candidate_peaks": len(detector),
                "event_matched": len(matched),
                "event_missed": len(manual_only),
                "event_extra": len(detector_only),
                "precision_pct": 100.0 * precision,
                "recall_pct": 100.0 * recall,
                "f1_pct": 200.0 * precision * recall / (precision + recall) if precision + recall else float("nan"),
                "candidate_minus_manual_timing_median_ms": float(np.median(deltas)) if deltas.size else float("nan"),
                "candidate_minus_manual_timing_p95_abs_ms": float(np.quantile(np.abs(deltas), 0.95)) if deltas.size else float("nan"),
                "candidate_corrected_rmssd_ms": detector_rmssd,
                "candidate_correction_ratio": detector_correction_ratio,
                "candidate_valid_ibi": detector_valid_ibi,
                "candidate_corrected_ibi": detector_corrected_ibi,
                "candidate_minus_manual_corrected_rmssd_ms": detector_rmssd - manual_rmssd,
                "abs_corrected_rmssd_error_ms": abs(detector_rmssd - manual_rmssd),
            })

    consensus = pd.DataFrame(consensus_rows)
    results = pd.DataFrame(summary_rows)
    consensus.to_csv(args.out_dir / "manual_consensus_rpeaks.csv", index=False)
    results.to_csv(args.out_dir / "candidate_full_window_metrics.csv", index=False)
    eligible_results = results.loc[results["manual_window_eligible"]].copy()
    if eligible_results.empty:
        raise ValueError("no manually eligible full windows: resolve or exclude every unresolvable chunk before validation")
    aggregate = eligible_results.groupby("candidate_method", as_index=False).agg(
        windows=("review_id", "size"),
        median_abs_corrected_rmssd_error_ms=("abs_corrected_rmssd_error_ms", "median"),
        mean_abs_corrected_rmssd_error_ms=("abs_corrected_rmssd_error_ms", "mean"),
        median_signed_corrected_rmssd_error_ms=("candidate_minus_manual_corrected_rmssd_ms", "median"),
        median_precision_pct=("precision_pct", "median"),
        median_recall_pct=("recall_pct", "median"),
        median_f1_pct=("f1_pct", "median"),
        median_timing_error_ms=("candidate_minus_manual_timing_median_ms", "median"),
        p95_timing_abs_ms=("candidate_minus_manual_timing_p95_abs_ms", "median"),
    )
    aggregate.to_csv(args.out_dir / "candidate_full_window_summary.csv", index=False)
    report = {
        "scope": "full five-minute windows annotated independently by two reviewers",
        "reviewer_consensus_tolerance_ms": CONSENSUS_TOLERANCE_MS,
        "event_match_tolerance_ms": EVENT_TOLERANCE_MS,
        "ibi_rule": "300-2000ms valid IBI; 11-point local median; replace >20% deviation",
        "selection_rule": "choose only after inspecting full-window corrected-RMSSD error, event errors, and timing error",
        "manual_windows_total": int(results["review_id"].nunique()),
        "manual_windows_eligible": int(eligible_results["review_id"].nunique()),
        "manual_windows_excluded_for_unresolvable_raw_ecg": int(results.loc[~results["manual_window_eligible"], "review_id"].nunique()),
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

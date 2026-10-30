"""Compare raw-ECG R-peak candidate strategies on the existing blind consensus.

This is a *development* comparison only: the consensus excerpts have already
been inspected while developing raw morphology proposals, so no method chosen
here may be reported as independently validated.  It writes reproducible
peak-level, timing, and corrected-RMSSD metrics for a later held-out study.
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

from build_blinded_ecg_rpeak_review import _independent_raw_r_candidates


EVENT_TOLERANCE_MS = 150.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0
IBI_CORRECTION_THRESHOLD = 0.20
DETAIL_EXCERPTS_SECONDS = ((65.0, 85.0), (215.0, 235.0))


def _infer_fs(times_ms: np.ndarray) -> float:
    step = np.diff(np.asarray(times_ms, dtype=float))
    step = step[np.isfinite(step) & (step > 0)]
    if not step.size:
        raise ValueError("cannot infer sampling rate")
    return float(1000.0 / np.median(step))


def _one_to_one_match(a: np.ndarray, b: np.ndarray, tolerance_ms: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    i = j = 0
    matches: list[tuple[int, int]] = []
    only_a: list[int] = []
    only_b: list[int] = []
    while i < len(a) and j < len(b):
        delta = float(a[i] - b[j])
        if abs(delta) <= tolerance_ms:
            matches.append((i, j)); i += 1; j += 1
        elif delta < 0:
            only_a.append(i); i += 1
        else:
            only_b.append(j); j += 1
    only_a.extend(range(i, len(a))); only_b.extend(range(j, len(b)))
    return matches, only_a, only_b


def _corrected_rmssd(times_ms: np.ndarray) -> float:
    rr = np.diff(np.asarray(times_ms, dtype=float))
    nn = rr[(rr >= IBI_MIN_MS) & (rr <= IBI_MAX_MS)]
    if nn.size < 3:
        return float("nan")
    corrected = nn.copy()
    if corrected.size >= 5:
        local = median_filter(corrected, size=11, mode="reflect")
        corrected[np.abs(corrected - local) > IBI_CORRECTION_THRESHOLD * local] = local[np.abs(corrected - local) > IBI_CORRECTION_THRESHOLD * local]
    return float(np.sqrt(np.mean(np.diff(corrected) ** 2)))


def _pan_tompkins(signal: np.ndarray, fs: float) -> np.ndarray:
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def _neurokit(signal: np.ndarray, fs: float) -> np.ndarray:
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def _refine_to_raw_apex(signal: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if not x.size:
        return np.empty(0, dtype=int)
    baseline = float(np.median(x))
    polarity = 1.0 if np.quantile(x, .995) - baseline >= baseline - np.quantile(x, .005) else -1.0
    radius = max(1, int(round(.150 * fs)))
    refined: list[int] = []
    for peak in np.asarray(peaks, dtype=int):
        left, right = max(0, peak - radius), min(len(x), peak + radius + 1)
        if left < right:
            refined.append(left + int(np.argmax(polarity * x[left:right])))
    return np.unique(np.asarray(refined, dtype=int))


def _raw_first_union(raw: np.ndarray, pan_refined: np.ndarray) -> np.ndarray:
    """Retain morphology positions and fill only raw-morphology gaps from Pan."""
    selected = list(np.asarray(raw, dtype=float))
    for peak in np.asarray(pan_refined, dtype=float):
        if not selected or min(abs(peak - candidate) for candidate in selected) > EVENT_TOLERANCE_MS:
            selected.append(float(peak))
    return np.asarray(sorted(selected), dtype=float)


def _raw_pan_agreement(raw: np.ndarray, pan_refined: np.ndarray) -> np.ndarray:
    """High-confidence intersection: retain raw positions only when Pan agrees."""
    raw = np.asarray(raw, dtype=float)
    pan_refined = np.asarray(pan_refined, dtype=float)
    if not raw.size or not pan_refined.size:
        return np.empty(0, dtype=float)
    return raw[np.asarray([np.min(np.abs(pan_refined - peak)) <= EVENT_TOLERANCE_MS for peak in raw])]


def _candidates(signal: np.ndarray, times: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    raw_idx = _independent_raw_r_candidates(signal, times)
    raw = times[raw_idx]
    try:
        pan_idx = _pan_tompkins(signal, fs)
    except Exception:
        pan_idx = np.empty(0, dtype=int)
    pan_idx = pan_idx[(pan_idx >= 0) & (pan_idx < len(times))]
    pan = times[pan_idx]
    pan_refined = times[_refine_to_raw_apex(signal, pan_idx, fs)]
    try:
        neurokit_idx = _neurokit(signal, fs)
    except Exception:
        neurokit_idx = np.empty(0, dtype=int)
    neurokit_idx = neurokit_idx[(neurokit_idx >= 0) & (neurokit_idx < len(times))]
    neurokit_refined = times[_refine_to_raw_apex(signal, neurokit_idx, fs)]
    return {
        "raw_morphology": raw,
        "pantompkins1985": pan,
        "pantompkins1985_raw_apex_refined": pan_refined,
        "raw_first_pan_fill": _raw_first_union(raw, pan_refined),
        "raw_pan_agreement": _raw_pan_agreement(raw, pan_refined),
        "neurokit_raw_apex_refined": neurokit_refined,
        "raw_first_neurokit_fill": _raw_first_union(raw, neurokit_refined),
    }


def _aggregate(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, table in results.groupby(group_columns, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        manual = int(table.manual_peaks.sum())
        proposed = int(table.candidate_peaks.sum())
        matched = int(table.matched_peaks.sum())
        precision = matched / proposed if proposed else float("nan")
        recall = matched / manual if manual else float("nan")
        timing = table.loc[table.timing_median_ms.notna(), "timing_median_ms"]
        rmssd = table.loc[table.abs_corrected_rmssd_error_ms.notna(), "abs_corrected_rmssd_error_ms"]
        row = dict(zip(group_columns, key, strict=True))
        row.update({
            "excerpts": int(len(table)), "manual_peaks": manual, "candidate_peaks": proposed,
            "matched_peaks": matched, "missed_peaks": manual - matched, "extra_peaks": proposed - matched,
            "precision_pct": 100 * precision, "recall_pct": 100 * recall,
            "f1_pct": 200 * precision * recall / (precision + recall) if precision + recall else float("nan"),
            "median_timing_error_ms": float(timing.median()) if len(timing) else float("nan"),
            "rmssd_evaluable_excerpts": int(len(rmssd)),
            "median_abs_corrected_rmssd_error_ms": float(rmssd.median()) if len(rmssd) else float("nan"),
            "mean_abs_corrected_rmssd_error_ms": float(rmssd.mean()) if len(rmssd) else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/blinded_ecg_rpeak_review_v2_training_cohort")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "review_manifest.csv")
    parser.add_argument("--manual-consensus", type=Path, default=root / "reconciled_results/manual_consensus_rpeaks.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "candidate_development_v1")
    parser.add_argument("--detail-seconds", type=float, default=20.0)
    parser.add_argument("--detail-centers-sec", default="75,225")
    args = parser.parse_args()
    detail_centers = tuple(float(value) for value in args.detail_centers_sec.split(",") if value.strip())
    if args.detail_seconds <= 0 or not detail_centers:
        raise ValueError("detail duration and centres must be positive/non-empty")
    manifest = pd.read_csv(args.manifest)
    manual = pd.read_csv(args.manual_consensus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    rows: list[dict[str, object]] = []
    for _, window in manifest.iterrows():
        review_id, participant = str(window.review_id), str(window.participant)
        raw_path = str(window.raw_ecg_path)
        if raw_path not in raw_cache:
            with np.load(raw_path, allow_pickle=True) as source:
                x, t = np.asarray(source["ecg_uv"], dtype=float), np.asarray(source["time_ms"], dtype=float)
            raw_cache[raw_path] = x, t, _infer_fs(t)
        x, t, fs = raw_cache[raw_path]
        start = float(window.window_start_unix_ms)
        for clip_number, center in enumerate(detail_centers, start=1):
            left_sec, right_sec = center - args.detail_seconds / 2.0, center + args.detail_seconds / 2.0
            clip_id = f"{review_id}_C{clip_number}"
            left, right = start + left_sec * 1000.0, start + right_sec * 1000.0
            lo, hi = np.searchsorted(t, left, side="left"), np.searchsorted(t, right, side="right")
            signal, times = x[lo:hi], t[lo:hi]
            reference = np.sort(manual.loc[manual.clip_id.astype(str).eq(clip_id), "consensus_unix_ms"].to_numpy(float))
            reference_rmssd = _corrected_rmssd(reference)
            for method, proposal in _candidates(signal, times, fs).items():
                proposed = np.sort(proposal[(proposal >= left) & (proposal <= right)])
                pairs, missed, extra = _one_to_one_match(reference, proposed, EVENT_TOLERANCE_MS)
                deltas = np.asarray([proposed[j] - reference[i] for i, j in pairs], dtype=float)
                candidate_rmssd = _corrected_rmssd(proposed)
                rows.append({
                    "clip_id": clip_id, "review_id": review_id, "participant": participant,
                    "candidate_method": method, "manual_peaks": len(reference), "candidate_peaks": len(proposed),
                    "matched_peaks": len(pairs), "missed_peaks": len(missed), "extra_peaks": len(extra),
                    "timing_median_ms": float(np.median(deltas)) if deltas.size else float("nan"),
                    "timing_p95_abs_ms": float(np.quantile(np.abs(deltas), .95)) if deltas.size else float("nan"),
                    "manual_corrected_rmssd_ms": reference_rmssd, "candidate_corrected_rmssd_ms": candidate_rmssd,
                    "signed_corrected_rmssd_error_ms": candidate_rmssd - reference_rmssd,
                    "abs_corrected_rmssd_error_ms": abs(candidate_rmssd - reference_rmssd),
                })
    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "excerpt_candidate_metrics.csv", index=False)
    summary = _aggregate(results, ["candidate_method"])
    summary.to_csv(args.out_dir / "candidate_summary.csv", index=False)
    by_participant = _aggregate(results, ["participant", "candidate_method"])
    by_participant.to_csv(args.out_dir / "candidate_by_participant.csv", index=False)
    report = {
        "scope": f"development-only; dual-blind {args.detail_seconds:g}-second consensus excerpts",
        "do_not_use_as_independent_validation": True,
        "event_match_tolerance_ms": EVENT_TOLERANCE_MS,
        "rmssd_rule": "valid IBI 300-2000ms; 11-point median replacement for >20% deviations",
        "candidate_methods": list(summary.candidate_method),
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Re-score Pan--Tompkins against the existing blinded manual-apex consensus.

This is a reporting-only reanalysis of the 32 annotated 20-s excerpts.  It
separates broad event detection (150 ms) from R-apex localization (10/20/40
ms).  The optional global-delay row subtracts the median detector-minus-manual
delay estimated on this *same* set, so it is explicitly descriptive/post-hoc,
not an independent validation result.
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


TOLERANCES_MS = (10.0, 20.0, 40.0, 150.0)


def _infer_fs(times_ms: np.ndarray) -> float:
    dt = np.diff(np.asarray(times_ms, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        raise ValueError("cannot infer ECG sampling rate")
    return float(1000.0 / np.median(dt))


def _match(manual: np.ndarray, detector: np.ndarray, tolerance_ms: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Chronological one-to-one matching, identical to the original audit."""
    i = j = 0
    pairs: list[tuple[int, int]] = []
    only_manual: list[int] = []
    only_detector: list[int] = []
    while i < len(manual) and j < len(detector):
        difference = float(manual[i] - detector[j])
        if abs(difference) <= tolerance_ms:
            pairs.append((i, j))
            i += 1
            j += 1
        elif difference < 0:
            only_manual.append(i)
            i += 1
        else:
            only_detector.append(j)
            j += 1
    only_manual.extend(range(i, len(manual)))
    only_detector.extend(range(j, len(detector)))
    return pairs, only_manual, only_detector


def _parse_excerpts(value: str) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in str(value).split(";"):
        left, right = item.strip().split("-", maxsplit=1)
        spans.append((float(left), float(right)))
    return spans


def _detector_times(raw: np.ndarray, raw_times: np.ndarray, start_ms: float, end_ms: float, fs: float) -> np.ndarray:
    lo = int(np.searchsorted(raw_times, start_ms, side="left"))
    hi = int(np.searchsorted(raw_times, end_ms, side="right"))
    signal, times = raw[lo:hi], raw_times[lo:hi]
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return times[indices[(indices >= 0) & (indices < len(times))]]


def _score(clips: list[dict[str, object]], tolerance_ms: float, delay_ms: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    reference = candidate = matched = missed = extra = 0
    residuals: list[float] = []
    rows: list[dict[str, object]] = []
    for clip in clips:
        manual = np.asarray(clip["manual"], dtype=float)
        detector = np.asarray(clip["detector"], dtype=float) - delay_ms
        pairs, manual_only, detector_only = _match(manual, detector, tolerance_ms)
        delta = np.asarray([detector[j] - manual[i] for i, j in pairs], dtype=float)
        reference += len(manual)
        candidate += len(detector)
        matched += len(pairs)
        missed += len(manual_only)
        extra += len(detector_only)
        residuals.extend(delta.tolist())
        rows.append({"clip_id": clip["clip_id"], "participant": clip["participant"], "tolerance_ms": tolerance_ms,
                     "delay_subtracted_ms": delay_ms, "reference_peaks": len(manual), "candidate_peaks": len(detector),
                     "matched_peaks": len(pairs), "missed_peaks": len(manual_only), "extra_peaks": len(detector_only),
                     "precision_pct": 100 * len(pairs) / len(detector) if len(detector) else np.nan,
                     "recall_pct": 100 * len(pairs) / len(manual) if len(manual) else np.nan,
                     "f1_pct": (200 * len(pairs) / (len(detector) + len(manual))) if len(detector) + len(manual) else np.nan})
    precision = matched / candidate if candidate else float("nan")
    recall = matched / reference if reference else float("nan")
    return {
        "tolerance_ms": tolerance_ms, "delay_subtracted_ms": delay_ms, "reference_peaks": reference,
        "candidate_peaks": candidate, "matched_peaks": matched, "missed_peaks": missed, "extra_peaks": extra,
        "precision_pct": 100 * precision, "recall_pct": 100 * recall,
        "f1_pct": 200 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "matched_residual_median_ms": float(np.median(residuals)) if residuals else float("nan"),
        "matched_residual_p95_abs_ms": float(np.quantile(np.abs(residuals), .95)) if residuals else float("nan"),
    }, rows


def main() -> None:
    default_root = Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/blinded_ecg_rpeak_review_v2_training_cohort")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_root / "review_manifest.csv")
    parser.add_argument("--consensus", type=Path, default=default_root / "reconciled_results/manual_consensus_rpeaks.csv")
    parser.add_argument("--out-dir", type=Path, default=default_root / "reconciled_results/pan_tompkins_tolerance_sweep")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    consensus = pd.read_csv(args.consensus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    clips: list[dict[str, object]] = []
    for _, window in manifest.iterrows():
        path = str(window.raw_ecg_path)
        if path not in raw_cache:
            with np.load(path, allow_pickle=True) as source:
                raw, raw_times = np.asarray(source["ecg_uv"], dtype=float), np.asarray(source["time_ms"], dtype=float)
            raw_cache[path] = raw, raw_times, _infer_fs(raw_times)
        raw, raw_times, fs = raw_cache[path]
        start = float(window.window_start_unix_ms)
        # Match the original blinded-audit detector call: run it over exactly
        # five minutes, then take the two displayed excerpts.
        detector_all = _detector_times(raw, raw_times, start, start + 300_000.0, fs)
        for number, (left_s, right_s) in enumerate(_parse_excerpts(window.detail_excerpt_seconds), start=1):
            clip_id = f"{window.review_id}_C{number}"
            manual = np.sort(consensus.loc[consensus.clip_id.astype(str).eq(clip_id), "consensus_unix_ms"].to_numpy(float))
            clip_start, clip_end = start + left_s * 1000.0, start + right_s * 1000.0
            detector = detector_all[(detector_all >= clip_start) & (detector_all <= clip_end)]
            clips.append({"clip_id": clip_id, "participant": str(window.participant), "manual": manual, "detector": detector})

    raw_150, _ = _score(clips, 150.0, 0.0)
    global_lag_ms = float(raw_150["matched_residual_median_ms"])
    tables: list[dict[str, object]] = []
    clip_tables: list[dict[str, object]] = []
    for label, delay in (("raw_pan_tompkins", 0.0), ("posthoc_global_lag_corrected", global_lag_ms)):
        for tolerance in TOLERANCES_MS:
            total, by_clip = _score(clips, tolerance, delay)
            tables.append({"alignment": label, **total})
            for item in by_clip:
                clip_tables.append({"alignment": label, **item})
    sweep = pd.DataFrame(tables)
    sweep.to_csv(args.out_dir / "pan_tompkins_precision_recall_f1_by_tolerance.csv", index=False)
    pd.DataFrame(clip_tables).to_csv(args.out_dir / "pan_tompkins_precision_recall_f1_by_clip.csv", index=False)
    report = {
        "scope": f"{len(clips)} existing blinded 20-s excerpts; no label was changed",
        "manual_consensus_peaks": int(sum(len(c["manual"]) for c in clips)),
        "raw_pan_tompkins_peaks": int(sum(len(c["detector"]) for c in clips)),
        "global_detector_minus_manual_lag_ms": global_lag_ms,
        "posthoc_correction": "candidate time minus global lag; lag estimated on this same set, not held out",
        "tolerances_ms": list(TOLERANCES_MS),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# Pan--Tompkins event versus localization reanalysis\n\n"
        "`raw_pan_tompkins` evaluates detector timestamps as reported. `posthoc_global_lag_corrected` subtracts the median delay estimated from the same manual set; it is descriptive only, not a held-out performance estimate. "
        "Every row reports candidate-level precision, manual-reference recall, and F1 under chronological one-to-one matching.\n",
        encoding="utf-8",
    )
    print(sweep.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

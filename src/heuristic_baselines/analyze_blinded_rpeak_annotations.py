"""Reconcile two blinded R-apex annotation CSVs and audit Pan--Tompkins peaks.

The inputs contain independent manual clicks from the blinded raw-ECG review
app. Results distinguish (a) reviewer agreement, (b) detector placement and
missed/extra beats, and (c) RMSSD in the *annotated 20-s excerpts only*.
They must not be interpreted as a full five-minute manual-RMSSD evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


CONSENSUS_TOLERANCE_MS = 40.0
DETECTOR_MATCH_TOLERANCE_MS = 150.0


def _infer_fs(t: np.ndarray) -> float:
    diff = np.diff(np.asarray(t, dtype=float))
    diff = diff[np.isfinite(diff) & (diff > 0)]
    return float(1000.0 / np.median(diff))


def _one_to_one_match(a: np.ndarray, b: np.ndarray, tolerance_ms: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Chronological, one-to-one nearest matching for sorted beat times."""
    i = j = 0
    matches: list[tuple[int, int]] = []
    only_a: list[int] = []
    only_b: list[int] = []
    while i < len(a) and j < len(b):
        delta = float(a[i] - b[j])
        if abs(delta) <= tolerance_ms:
            matches.append((i, j))
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
    return matches, only_a, only_b


def _rmssd(rpeak_times_ms: np.ndarray) -> float:
    rr = np.diff(np.asarray(rpeak_times_ms, dtype=float))
    return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size >= 2 else float("nan")


def _detector_peaks(raw: np.ndarray, raw_t: np.ndarray, start_ms: float, end_ms: float, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = int(np.searchsorted(raw_t, start_ms, side="left"))
    hi = int(np.searchsorted(raw_t, end_ms, side="right"))
    signal, times = raw[lo:hi], raw_t[lo:hi]
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    indices = indices[(indices >= 0) & (indices < len(times))]
    return signal, times, times[indices]


def _plot_clip(
    signal: np.ndarray,
    times: np.ndarray,
    clip_start: float,
    clip_end: float,
    reviewer_a: np.ndarray,
    reviewer_b: np.ndarray,
    consensus: np.ndarray,
    detector: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    keep = (times >= clip_start) & (times <= clip_end)
    x, y = times[keep], signal[keep]
    fig, ax = plt.subplots(figsize=(15, 4.5), constrained_layout=True)
    ax.plot((x - clip_start) / 1000.0, y, color="#343a40", linewidth=0.9, label="raw ECG")
    def draw_marks(values: np.ndarray, color: str, marker: str, label: str) -> None:
        values = values[(values >= clip_start) & (values <= clip_end)]
        if values.size:
            amplitude = np.interp(values, x, y)
            ax.scatter((values - clip_start) / 1000.0, amplitude, s=38, color=color, marker=marker, label=label, zorder=4)
    draw_marks(reviewer_a, "#9467bd", "|", "reviewer 1")
    draw_marks(reviewer_b, "#ff7f0e", "|", "reviewer 2")
    draw_marks(consensus, "#2ca02c", "o", "manual consensus")
    draw_marks(detector, "#d62728", "v", "Pan--Tompkins")
    ax.set(title=title, xlabel="seconds in blinded excerpt", ylabel="ECG (µV)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncols=5)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/blinded_ecg_rpeak_review_v2_training_cohort")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "review_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "reconciled_results")
    parser.add_argument("--detail-seconds", type=float, default=20.0)
    parser.add_argument("--detail-centers-sec", default="75,225")
    args = parser.parse_args()
    detail_centers = tuple(float(value) for value in args.detail_centers_sec.split(",") if value.strip())
    if args.detail_seconds <= 0 or not detail_centers:
        raise ValueError("detail duration and centres must be positive/non-empty")

    a, b, manifest = pd.read_csv(args.reviewer_a), pd.read_csv(args.reviewer_b), pd.read_csv(args.manifest)
    required = {"review_id", "participant", "window_start_unix_ms", "manual_rpeak_abs_unix_ms"}
    for name, table in {"reviewer_a": a, "reviewer_b": b}.items():
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    def auto_excluded_clips(table: pd.DataFrame) -> set[str]:
        notes = table.get("notes", pd.Series("", index=table.index)).fillna("").astype(str)
        prefix = "unresolvable_raw_ecg_chunk:"
        return {note[len(prefix):] for note in notes if note.startswith(prefix)}

    excluded_by_a, excluded_by_b = auto_excluded_clips(a), auto_excluded_clips(b)
    auto_excluded = excluded_by_a | excluded_by_b
    for table in (a, b):
        table["manual_rpeak_abs_unix_ms"] = pd.to_numeric(table["manual_rpeak_abs_unix_ms"], errors="coerce")
        table.dropna(subset=["manual_rpeak_abs_unix_ms"], inplace=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "adjudication_plots"
    plot_dir.mkdir(exist_ok=True)

    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    consensus_rows: list[dict[str, object]] = []
    detector_match_rows: list[dict[str, object]] = []
    clip_rows: list[dict[str, object]] = []
    for _, window in manifest.iterrows():
        review_id = str(window.review_id)
        start = float(window.window_start_unix_ms)
        raw_path = str(window.raw_ecg_path)
        if raw_path not in raw_cache:
            with np.load(raw_path, allow_pickle=True) as z:
                raw, raw_t = np.asarray(z["ecg_uv"], dtype=float), np.asarray(z["time_ms"], dtype=float)
            raw_cache[raw_path] = raw, raw_t, _infer_fs(raw_t)
        raw, raw_t, fs = raw_cache[raw_path]
        signal, times, detector_all = _detector_peaks(raw, raw_t, start, start + 300_000.0, fs)
        a_window = np.sort(a.loc[a.review_id.astype(str).eq(review_id), "manual_rpeak_abs_unix_ms"].to_numpy(float))
        b_window = np.sort(b.loc[b.review_id.astype(str).eq(review_id), "manual_rpeak_abs_unix_ms"].to_numpy(float))
        for clip_number, center in enumerate(detail_centers, start=1):
            left, right = center - args.detail_seconds / 2.0, center + args.detail_seconds / 2.0
            clip_id = f"{review_id}_C{clip_number}"
            if clip_id in auto_excluded:
                continue
            clip_start, clip_end = start + left * 1000.0, start + right * 1000.0
            a_clip = a_window[(a_window >= clip_start) & (a_window <= clip_end)]
            b_clip = b_window[(b_window >= clip_start) & (b_window <= clip_end)]
            agreement, only_a, only_b = _one_to_one_match(a_clip, b_clip, CONSENSUS_TOLERANCE_MS)
            consensus = np.asarray([(a_clip[i] + b_clip[j]) / 2.0 for i, j in agreement], dtype=float)
            for peak_no, ((i, j), value) in enumerate(zip(agreement, consensus, strict=True), start=1):
                consensus_rows.append({"clip_id": clip_id, "review_id": review_id, "participant": window.participant, "manual_peak_number": peak_no, "reviewer_a_unix_ms": a_clip[i], "reviewer_b_unix_ms": b_clip[j], "consensus_unix_ms": value, "reviewer_a_minus_b_ms": a_clip[i] - b_clip[j]})
            detector = detector_all[(detector_all >= clip_start) & (detector_all <= clip_end)]
            matched, manual_only, detector_only = _one_to_one_match(consensus, detector, DETECTOR_MATCH_TOLERANCE_MS)
            deltas = np.asarray([detector[j] - consensus[i] for i, j in matched], dtype=float)
            for manual_index, detector_index in matched:
                detector_match_rows.append({
                    "clip_id": clip_id, "review_id": review_id, "participant": window.participant,
                    "manual_consensus_unix_ms": consensus[manual_index], "detector_unix_ms": detector[detector_index],
                    "detector_minus_manual_ms": detector[detector_index] - consensus[manual_index],
                })
            manual_rmssd, detector_rmssd = _rmssd(consensus), _rmssd(detector)
            plot_path = plot_dir / f"{clip_id}_adjudication.png"
            _plot_clip(signal, times, clip_start, clip_end, a_clip, b_clip, consensus, detector, f"{clip_id}: manual consensus vs Pan--Tompkins", plot_path)
            clip_rows.append({
                "clip_id": clip_id, "review_id": review_id, "participant": window.participant,
                "reviewer_a_peaks": len(a_clip), "reviewer_b_peaks": len(b_clip), "reviewer_agreed_peaks": len(agreement),
                "reviewer_a_only": len(only_a), "reviewer_b_only": len(only_b),
                "reviewer_match_pct_of_mean_count": 100 * len(agreement) / max(1.0, (len(a_clip) + len(b_clip)) / 2),
                "reviewer_median_abs_disagreement_ms": float(np.median([abs(a_clip[i] - b_clip[j]) for i, j in agreement])) if agreement else np.nan,
                "detector_peaks": len(detector), "detector_manual_matched": len(matched), "detector_manual_missed": len(manual_only), "detector_extra": len(detector_only),
                "detector_minus_manual_median_ms": float(np.median(deltas)) if deltas.size else np.nan,
                "detector_minus_manual_p95_abs_ms": float(np.quantile(np.abs(deltas), .95)) if deltas.size else np.nan,
                "manual_excerpt_rmssd_uncorrected_ms": manual_rmssd, "detector_excerpt_rmssd_uncorrected_ms": detector_rmssd,
                "detector_minus_manual_excerpt_rmssd_uncorrected_ms": detector_rmssd - manual_rmssd if np.isfinite([manual_rmssd, detector_rmssd]).all() else np.nan,
                "adjudication_plot_path": str(plot_path.resolve()),
            })

    consensus_table, detector_matches, clips = pd.DataFrame(consensus_rows), pd.DataFrame(detector_match_rows), pd.DataFrame(clip_rows)
    consensus_table.to_csv(args.out_dir / "manual_consensus_rpeaks.csv", index=False)
    detector_matches.to_csv(args.out_dir / "detector_manual_matched_rpeaks.csv", index=False)
    clips.to_csv(args.out_dir / "agreement_detector_excerpt_metrics.csv", index=False)
    pd.DataFrame({"clip_id": sorted(auto_excluded)}).to_csv(args.out_dir / "auto_excluded_clips.csv", index=False)
    reviewer_a_total = int(len(a))
    reviewer_b_total = int(len(b))
    detector_total = int(clips.detector_peaks.sum())
    detector_matched = int(clips.detector_manual_matched.sum())
    manual_total = int(len(consensus_table))
    precision = detector_matched / detector_total if detector_total else float("nan")
    recall = detector_matched / manual_total if manual_total else float("nan")
    summary = {
        "scope": f"{len(clips)} blinded {args.detail_seconds:g}-second excerpts only; not a full 5-minute manual-RMSSD study",
        "auto_excluded_severe_raw_ecg_clips": len(auto_excluded),
        "reviewer_consensus_tolerance_ms": CONSENSUS_TOLERANCE_MS,
        "detector_match_tolerance_ms": DETECTOR_MATCH_TOLERANCE_MS,
        "n_excerpts": int(len(clips)),
        "reviewer_a_peaks": reviewer_a_total,
        "reviewer_b_peaks": reviewer_b_total,
        "reviewer_matched_peaks": manual_total,
        "reviewer_agreement_pct_of_mean_count": 100 * manual_total / ((reviewer_a_total + reviewer_b_total) / 2),
        "reviewer_median_abs_disagreement_ms": float(np.median(np.abs(consensus_table.reviewer_a_minus_b_ms))) if len(consensus_table) else None,
        "detector_peaks": detector_total,
        "detector_matched_manual_peaks": detector_matched,
        "detector_manual_missed": int(clips.detector_manual_missed.sum()),
        "detector_extra": int(clips.detector_extra.sum()),
        "detector_minus_manual_median_ms": float(np.median(detector_matches.detector_minus_manual_ms)) if len(detector_matches) else None,
        "detector_minus_manual_p95_abs_ms": float(np.quantile(np.abs(detector_matches.detector_minus_manual_ms), .95)) if len(detector_matches) else None,
        "detector_precision_pct": 100 * precision,
        "detector_recall_pct": 100 * recall,
        "detector_f1_pct": 200 * precision * recall / (precision + recall) if precision + recall else None,
        "median_abs_excerpt_rmssd_difference_uncorrected_ms": float(np.median(np.abs(clips.detector_minus_manual_excerpt_rmssd_uncorrected_ms.dropna()))) if clips.detector_minus_manual_excerpt_rmssd_uncorrected_ms.notna().any() else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

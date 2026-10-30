"""Build a versioned hybrid ECG-label candidate without overwriting legacy labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _independent_raw_r_candidates
from evaluate_assisted_human_confirmed_rpeaks import infer_fs, pan_tompkins_indices, raw_first_pan_fill, refine_to_dominant_raw_extreme
from rebuild_ecg_ground_truth_raw_morphology import _correct_rr, _object_array, _stats


PARTICIPANTS = ("P1", "P3", "P4", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20")
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}
WINDOW_MS = 300_000.0
RAW_CHUNK_MS = 5_000.0
FALLBACK_GAP_MS = 150.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0


def _raw_morphology_window(signal: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, int]:
    proposals: list[np.ndarray] = []
    severe_chunks = 0
    for start in np.arange(float(times[0]), float(times[-1]), RAW_CHUNK_MS):
        end = start + RAW_CHUNK_MS
        lo, hi = np.searchsorted(times, start, side="left"), np.searchsorted(times, end, side="left")
        chunk_signal, chunk_times = signal[lo:hi], times[lo:hi]
        if len(chunk_times) < 30:
            continue
        if _has_severe_amplitude_artifact(chunk_signal):
            severe_chunks += 1
            continue
        indices = _independent_raw_r_candidates(chunk_signal, chunk_times)
        if len(indices):
            proposals.append(chunk_times[indices])
    return (np.unique(np.concatenate(proposals)) if proposals else np.empty(0, dtype=float)), severe_chunks


def _global_raw_morphology(signal: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run 5-s morphology proposals once per raw recording, not per overlap."""
    proposals: list[np.ndarray] = []
    severe_starts: list[float] = []
    severe_ends: list[float] = []
    breaks = np.flatnonzero(np.diff(times) > 250.0) + 1
    bounds = np.r_[0, breaks, len(times)]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        run_signal, run_times = signal[lo:hi], times[lo:hi]
        for start in np.arange(float(run_times[0]), float(run_times[-1]), RAW_CHUNK_MS):
            end = start + RAW_CHUNK_MS
            left, right = np.searchsorted(run_times, start, side="left"), np.searchsorted(run_times, end, side="left")
            chunk_signal, chunk_times = run_signal[left:right], run_times[left:right]
            if len(chunk_times) < 30:
                continue
            if _has_severe_amplitude_artifact(chunk_signal):
                severe_starts.append(start); severe_ends.append(end)
                continue
            indices = _independent_raw_r_candidates(chunk_signal, chunk_times)
            if len(indices):
                proposals.append(chunk_times[indices])
    candidates = np.unique(np.concatenate(proposals)) if proposals else np.empty(0, dtype=float)
    return candidates, np.asarray(severe_starts, dtype=float), np.asarray(severe_ends, dtype=float)


def _hybrid_peaks(signal: np.ndarray, times: np.ndarray, fs: float, raw: np.ndarray, severe_chunks: int) -> tuple[np.ndarray, int, int, int, str]:
    try:
        pan_indices = pan_tompkins_indices(signal, fs)
        pan_refined = refine_to_dominant_raw_extreme(signal, times, pan_indices, fs)
        hybrid = raw_first_pan_fill(raw, pan_refined)
        fallback_count = int(len(hybrid) - len(raw))
        return hybrid, int(len(raw)), fallback_count, severe_chunks, ""
    except Exception as error:
        return raw, int(len(raw)), 0, severe_chunks, f"pan_fallback_failed:{type(error).__name__}"


def _rebuild_participant(raw_path: Path, windows_path: Path, out_path: Path, participant: str) -> dict[str, object]:
    with np.load(raw_path, allow_pickle=True) as source:
        raw_signal = np.asarray(source["ecg_uv"], dtype=float)
        raw_times = np.asarray(source["time_ms"], dtype=float)
    valid = np.isfinite(raw_signal) & np.isfinite(raw_times)
    raw_signal, raw_times = raw_signal[valid], raw_times[valid]
    with np.load(windows_path, allow_pickle=True) as source:
        starts = np.asarray(source["ppg_window_t0_ms"], dtype=float)
        ends = np.asarray(source["ppg_window_t1_ms"], dtype=float)
    if starts.shape != ends.shape:
        raise ValueError(f"{participant}: source window bounds mismatch")
    fs = infer_fs(raw_times)
    global_raw, severe_starts, severe_ends = _global_raw_morphology(raw_signal, raw_times)
    peaks_out: list[np.ndarray] = []
    rr_uncorrected: list[np.ndarray] = []
    rr_corrected: list[np.ndarray] = []
    rmssd_raw: list[float] = []
    rmssd_corrected: list[float] = []
    sdnn_corrected: list[float] = []
    mean_ibi: list[float] = []
    hr: list[float] = []
    correction_ratio: list[float] = []
    valid_ratio: list[float] = []
    raw_counts: list[int] = []
    fallback_counts: list[int] = []
    fallback_ratio: list[float] = []
    severe_counts: list[int] = []
    qc_pass: list[bool] = []
    review_required: list[bool] = []
    reasons: list[str] = []
    for start, end in zip(starts, ends, strict=True):
        lo, hi = np.searchsorted(raw_times, start, side="left"), np.searchsorted(raw_times, end, side="right")
        signal, times = raw_signal[lo:hi], raw_times[lo:hi]
        raw_left, raw_right = np.searchsorted(global_raw, start, side="left"), np.searchsorted(global_raw, end, side="right")
        raw_for_window = global_raw[raw_left:raw_right]
        severe_chunk_count = int(np.sum((severe_starts < end) & (severe_ends > start)))
        peaks, raw_count, fallback_count, severe_chunk_count, detector_error = _hybrid_peaks(signal, times, fs, raw_for_window, severe_chunk_count)
        rr_all = np.diff(peaks)
        valid_ibi = (rr_all >= IBI_MIN_MS) & (rr_all <= IBI_MAX_MS)
        ratio_valid = float(valid_ibi.mean()) if len(rr_all) else 0.0
        rr_valid = rr_all[valid_ibi]
        corrected, corrected_ratio = _correct_rr(rr_valid)
        raw_rmssd, _, _, _ = _stats(rr_valid)
        corrected_rmssd, one_sdnn, one_mean, one_hr = _stats(corrected)
        issues: list[str] = []
        if severe_chunk_count:
            issues.append("severe_raw_ecg_artifact_chunk")
        if len(peaks) < 3:
            issues.append("too_few_hybrid_peaks")
        if ratio_valid < 1.0:
            issues.append("invalid_ibi")
        if not np.isfinite(corrected_rmssd) or corrected_rmssd > 200.0:
            issues.append("invalid_or_high_rmssd")
        if corrected_ratio > 0.20:
            issues.append("high_ibi_correction_ratio")
        if np.isfinite(one_hr) and not (40.0 <= one_hr <= 220.0):
            issues.append("hr_out_of_range")
        if detector_error:
            issues.append(detector_error)
        passed = not issues
        peaks_out.append((peaks - start).astype(np.float32))
        rr_uncorrected.append(rr_all.astype(np.float32))
        rr_corrected.append(corrected.astype(np.float32))
        rmssd_raw.append(raw_rmssd)
        rmssd_corrected.append(corrected_rmssd)
        sdnn_corrected.append(one_sdnn)
        mean_ibi.append(one_mean)
        hr.append(one_hr)
        correction_ratio.append(corrected_ratio)
        valid_ratio.append(ratio_valid)
        raw_counts.append(raw_count)
        fallback_counts.append(fallback_count)
        fallback_ratio.append(fallback_count / len(peaks) if len(peaks) else float("nan"))
        severe_counts.append(severe_chunk_count)
        qc_pass.append(passed)
        review_required.append(not passed)
        reasons.append("|".join(issues))
    config = {
        "label_version": "hybrid_raw_morphology_pan_raw_apex_candidate_v3",
        "raw_morphology_chunk_seconds": RAW_CHUNK_MS / 1000.0,
        "fallback": "window_pantompkins1985_then_dominant_local_raw_apex_if_no_raw_morphology_candidate_within_150ms",
        "pan_tompkins_used_as_fallback_only": True,
        "source_windows": str(windows_path.resolve()),
        "source_raw_ecg": str(raw_path.resolve()),
        "status": "candidate_pending_frozen_holdout_review",
    }
    np.savez_compressed(
        out_path,
        config_json=json.dumps(config, ensure_ascii=False), participant=np.array(participant),
        source_window_index=np.arange(len(starts), dtype=np.int32),
        ppg_window_t0_ms=starts.astype(np.float64), ppg_window_t1_ms=ends.astype(np.float64),
        ecg_r_peak_times_rel_ms=_object_array(peaks_out, np.float32),
        ecg_rr_intervals_uncorrected_ms=_object_array(rr_uncorrected, np.float32),
        ecg_rr_intervals_corrected_ms=_object_array(rr_corrected, np.float32),
        ecg_rmssd_raw_ms=np.asarray(rmssd_raw, dtype=np.float32),
        ecg_rmssd_corrected_ms=np.asarray(rmssd_corrected, dtype=np.float32),
        ecg_sdnn_corrected_ms=np.asarray(sdnn_corrected, dtype=np.float32),
        ecg_mean_ibi_corrected_ms=np.asarray(mean_ibi, dtype=np.float32),
        ecg_hr_bpm_corrected=np.asarray(hr, dtype=np.float32),
        ecg_valid_ibi_ratio=np.asarray(valid_ratio, dtype=np.float32),
        ecg_ibi_correction_ratio=np.asarray(correction_ratio, dtype=np.float32),
        ecg_raw_morphology_peak_count=np.asarray(raw_counts, dtype=np.int16),
        ecg_pan_fallback_peak_count=np.asarray(fallback_counts, dtype=np.int16),
        ecg_pan_fallback_ratio=np.asarray(fallback_ratio, dtype=np.float32),
        ecg_severe_artifact_chunk_count=np.asarray(severe_counts, dtype=np.int16),
        ecg_label_qc_pass=np.asarray(qc_pass, dtype=bool), ecg_review_required=np.asarray(review_required, dtype=bool),
        ecg_label_qc_reason=np.asarray(reasons, dtype=object),
    )
    return {"participant": participant, "windows": len(starts), "qc_pass_windows": int(np.sum(qc_pass)), "review_required_windows": int(np.sum(review_required)), "qc_pass_pct": float(100 * np.mean(qc_pass)), "median_fallback_ratio": float(np.nanmedian(fallback_ratio)), "output": str(out_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--windows-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots"))
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/ecg_ground_truth_hybrid_candidate_v3"))
    parser.add_argument("--participants", nargs="+", default=list(PARTICIPANTS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    invalid = set(args.participants) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants cannot enter reconstruction: {sorted(invalid)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for participant in args.participants:
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        windows_path = args.windows_dir / f"synced_3device_rawaligned_training_v1_stride30_rawslots_{participant}.npz"
        if not raw_path.exists() or not windows_path.exists():
            raise FileNotFoundError(f"{participant}: missing source data")
        jobs.append((raw_path, windows_path, args.out_dir / f"ecg_ground_truth_hybrid_candidate_v3_{participant}.npz", participant))
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(jobs))) as pool:
        futures = {pool.submit(_rebuild_participant, *job): job[-1] for job in jobs}
        for future in as_completed(futures):
            result = future.result(); rows.append(result); print(json.dumps(result), flush=True)
    summary = pd.DataFrame(sorted(rows, key=lambda row: str(row["participant"])))
    summary.to_csv(args.out_dir / "rebuild_summary.csv", index=False)
    (args.out_dir / "README.md").write_text("# Hybrid ECG ground-truth candidate v3\n\nPending frozen hold-out review. This sidecar does not overwrite legacy labels.\n", encoding="utf-8")


if __name__ == "__main__":
    main()

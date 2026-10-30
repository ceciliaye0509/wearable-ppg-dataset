"""Rebuild ECG labels from raw-morphology candidates on existing rawslot windows.

The legacy Pan--Tompkins labels are never overwritten.  This writes a versioned
sidecar dataset keyed by the same participant/window index and PPG-window
timestamps.  Candidate detection runs once per non-overlapping 20-s raw ECG
chunk, then is sliced into the existing five-minute, 30-s-stride windows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _independent_raw_r_candidates


PARTICIPANTS = ("P1", "P3", "P4", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20")
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}
WINDOW_MS = 300_000.0
CHUNK_MS = 20_000.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0


def _infer_fs(times_ms: np.ndarray) -> float:
    step = np.diff(np.asarray(times_ms, dtype=float))
    step = step[np.isfinite(step) & (step > 0)]
    if not step.size:
        raise ValueError("cannot infer ECG sample rate")
    return float(1000.0 / np.median(step))


def _object_array(items: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = np.asarray(item, dtype=dtype)
    return out


def _correct_rr(rr: np.ndarray) -> tuple[np.ndarray, float]:
    corrected = np.asarray(rr, dtype=float).copy()
    if corrected.size < 5:
        return corrected, 0.0
    local_median = median_filter(corrected, size=11, mode="reflect")
    replace = np.abs(corrected - local_median) > 0.20 * local_median
    corrected[replace] = local_median[replace]
    return corrected, float(replace.mean())


def _stats(rr: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(rr, dtype=float)
    if values.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = np.diff(values)
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    sdnn = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    mean = float(np.mean(values))
    hr = float(60_000.0 / mean) if mean > 0 else float("nan")
    return rmssd, sdnn, mean, hr


def _continuous_chunk_candidates(signal: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate times and intervals suppressed for severe artefact."""
    x, t = np.asarray(signal, dtype=float), np.asarray(times, dtype=float)
    # A few raw files contain isolated non-finite timestamps.  Do not let them
    # create an unbounded chunk; they are a missing-data gap, not a beat.
    finite = np.isfinite(x) & np.isfinite(t)
    x, t = x[finite], t[finite]
    if t.size < 30:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)
    breaks = np.flatnonzero(np.diff(t) > 250.0) + 1
    bounds = np.r_[0, breaks, len(t)]
    all_candidates: list[np.ndarray] = []
    severe_starts: list[float] = []
    severe_ends: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        run_t, run_x = t[lo:hi], x[lo:hi]
        for start in np.arange(float(run_t[0]), float(run_t[-1]), CHUNK_MS):
            end = start + CHUNK_MS
            # ``run_t`` is sorted.  Binary slicing is equivalent to the old
            # boolean mask but avoids rescanning a multi-hour recording for
            # every 20-s chunk.
            left = int(np.searchsorted(run_t, start, side="left"))
            right = int(np.searchsorted(run_t, end, side="left"))
            chunk_t, chunk_x = run_t[left:right], run_x[left:right]
            if chunk_t.size < 30:
                continue
            if _has_severe_amplitude_artifact(chunk_x):
                severe_starts.append(start)
                severe_ends.append(end)
                continue
            indices = _independent_raw_r_candidates(chunk_x, chunk_t)
            if indices.size:
                all_candidates.append(chunk_t[indices])
    candidates = np.unique(np.concatenate(all_candidates)) if all_candidates else np.empty(0, dtype=float)
    return candidates, np.asarray(severe_starts, dtype=float), np.asarray(severe_ends, dtype=float)


def _rebuild_participant(raw_path: Path, windows_path: Path, out_path: Path, participant: str) -> dict[str, object]:
    with np.load(raw_path, allow_pickle=True) as source:
        raw_x = np.asarray(source["ecg_uv"], dtype=float)
        raw_t = np.asarray(source["time_ms"], dtype=float)
    with np.load(windows_path, allow_pickle=True) as windows:
        starts = np.asarray(windows["ppg_window_t0_ms"], dtype=float)
        ends = np.asarray(windows["ppg_window_t1_ms"], dtype=float)
    if starts.shape != ends.shape:
        raise ValueError(f"{participant}: window starts/ends mismatch")
    fs = _infer_fs(raw_t)
    candidates, severe_starts, severe_ends = _continuous_chunk_candidates(raw_x, raw_t)

    peak_times: list[np.ndarray] = []
    rr_raw: list[np.ndarray] = []
    rr_corrected: list[np.ndarray] = []
    rmssd: list[float] = []
    sdnn: list[float] = []
    mean_ibi: list[float] = []
    hr: list[float] = []
    correction_ratio: list[float] = []
    valid_ibi_ratio: list[float] = []
    qc_pass: list[bool] = []
    review_required: list[bool] = []
    reasons: list[str] = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        left, right = np.searchsorted(candidates, start, side="left"), np.searchsorted(candidates, end, side="right")
        peaks = candidates[left:right]
        rr_all = np.diff(peaks)
        valid = (rr_all >= IBI_MIN_MS) & (rr_all <= IBI_MAX_MS)
        valid_ratio = float(valid.mean()) if rr_all.size else 0.0
        # Do not bridge an invalid interval for RMSSD.  The conservative QC
        # below requires all intervals valid before accepting a label.
        rr = rr_all[valid]
        corrected, ratio = _correct_rr(rr)
        one_rmssd, one_sdnn, one_mean, one_hr = _stats(corrected)
        severe = bool(np.any((severe_starts < end) & (severe_ends > start)))
        issues: list[str] = []
        if severe:
            issues.append("severe_raw_ecg_artifact_chunk")
        if peaks.size < 3:
            issues.append("too_few_raw_morphology_peaks")
        if valid_ratio < 1.0:
            issues.append("invalid_ibi")
        if not np.isfinite(one_rmssd) or one_rmssd > 200.0:
            issues.append("invalid_or_high_rmssd")
        if ratio > 0.20:
            issues.append("high_ibi_correction_ratio")
        if np.isfinite(one_hr) and not (40.0 <= one_hr <= 220.0):
            issues.append("hr_out_of_range")
        passed = not issues
        peak_times.append((peaks - start).astype(np.float32))
        rr_raw.append(rr_all.astype(np.float32))
        rr_corrected.append(corrected.astype(np.float32))
        rmssd.append(one_rmssd)
        sdnn.append(one_sdnn)
        mean_ibi.append(one_mean)
        hr.append(one_hr)
        correction_ratio.append(ratio)
        valid_ibi_ratio.append(valid_ratio)
        qc_pass.append(passed)
        review_required.append(not passed)
        reasons.append("|".join(issues))

    config = {
        "label_version": "raw_morphology_ecg_ground_truth_candidate_v2",
        "candidate_method": "independent_raw_morphology_5to25Hz_polarity_aware_raw_extremum",
        "candidate_chunk_seconds": CHUNK_MS / 1000.0,
        "pan_tompkins_used": False,
        "qc_policy": "reject severe raw-artifact chunks, invalid IBI, RMSSD>200ms, correction>20%, or HR outside 40-220 bpm",
        "source_windows": str(windows_path.resolve()),
        "source_raw_ecg": str(raw_path.resolve()),
    }
    np.savez_compressed(
        out_path,
        config_json=json.dumps(config, ensure_ascii=False),
        participant=np.array(participant),
        source_window_index=np.arange(len(starts), dtype=np.int32),
        ppg_window_t0_ms=starts.astype(np.float64),
        ppg_window_t1_ms=ends.astype(np.float64),
        ecg_r_peak_times_rel_ms=_object_array(peak_times, np.float32),
        ecg_rr_intervals_uncorrected_ms=_object_array(rr_raw, np.float32),
        ecg_rr_intervals_corrected_ms=_object_array(rr_corrected, np.float32),
        ecg_rmssd_corrected_ms=np.asarray(rmssd, dtype=np.float32),
        ecg_sdnn_corrected_ms=np.asarray(sdnn, dtype=np.float32),
        ecg_mean_ibi_corrected_ms=np.asarray(mean_ibi, dtype=np.float32),
        ecg_hr_bpm_corrected=np.asarray(hr, dtype=np.float32),
        ecg_valid_ibi_ratio=np.asarray(valid_ibi_ratio, dtype=np.float32),
        ecg_ibi_correction_ratio=np.asarray(correction_ratio, dtype=np.float32),
        ecg_label_qc_pass=np.asarray(qc_pass, dtype=bool),
        ecg_review_required=np.asarray(review_required, dtype=bool),
        ecg_label_qc_reason=np.asarray(reasons, dtype=object),
    )
    return {
        "participant": participant,
        "windows": int(len(starts)),
        "qc_pass_windows": int(np.sum(qc_pass)),
        "review_required_windows": int(np.sum(review_required)),
        "qc_pass_pct": float(100.0 * np.mean(qc_pass)),
        "median_rmssd_ms_accepted": float(np.nanmedian(np.asarray(rmssd)[np.asarray(qc_pass)])) if any(qc_pass) else float("nan"),
        "raw_ecg_fs_hz": fs,
        "output": str(out_path.resolve()),
    }


def _rebuild_job(job: tuple[Path, Path, Path, str]) -> dict[str, object]:
    """Pickle-friendly per-participant worker; each output path is distinct."""
    return _rebuild_participant(*job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--windows-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_stride30_rawslots"))
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/ecg_ground_truth_raw_morphology_v2"))
    parser.add_argument("--participants", nargs="+", default=list(PARTICIPANTS))
    parser.add_argument("--workers", type=int, default=4, help="independent CPU workers (GPU is not used)")
    args = parser.parse_args()
    invalid = set(args.participants) & EXCLUDED_PARTICIPANTS
    if invalid:
        raise ValueError(f"excluded participants cannot enter the reconstruction: {sorted(invalid)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path, Path, str]] = []
    for participant in args.participants:
        raw_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        windows_path = args.windows_dir / f"synced_3device_rawaligned_training_v1_stride30_rawslots_{participant}.npz"
        if not raw_path.exists() or not windows_path.exists():
            raise FileNotFoundError(f"{participant}: missing raw ECG or source window file")
        out_path = args.out_dir / f"ecg_ground_truth_raw_morphology_v2_{participant}.npz"
        jobs.append((raw_path, windows_path, out_path, participant))
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(args.workers), len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_rebuild_job, job): job[-1] for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: str(row["participant"]))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "rebuild_summary.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "# Raw-morphology ECG ground-truth candidate v2\n\n"
        "This is a versioned candidate sidecar keyed to the released rawslot windows. "
        "It does not overwrite legacy Pan--Tompkins labels. Windows with failed QC require assisted review and are not accepted as ground truth.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

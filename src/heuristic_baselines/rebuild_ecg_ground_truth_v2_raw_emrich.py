"""Build the selected conservative Emrich local-QRS ECG-label candidate.

The sidecar is aligned to rawslot windows selected only by PPG boundary and
coverage QC (never by legacy Pan--Tompkins ECG QC), and never overwrites the
legacy reference. Each Emrich2023 event is moved to the dominant
raw QRS extreme within +/-50 ms. Raw morphology remains an audit diagnostic,
not a peak-fill source, because it created excessive short-RR events in a
complete-five-minute smoke test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact
from evaluate_assisted_human_confirmed_rpeaks import (
    detector_indices,
    infer_fs,
    refine_to_local_qrs_extreme,
)
from rebuild_ecg_ground_truth_raw_morphology import _correct_rr, _object_array, _stats


PARTICIPANTS = ("P1", "P3", "P4", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20")
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}
RAW_CHUNK_MS = 5_000.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0
MIN_ECG_VALID_SAMPLE_RATIO = 1.0
MIN_VALID_IBI_RATIO = 1.0
MIN_HR_BPM = 30.0
MAX_HR_BPM = 200.0
MAX_PEAKS_PER_WINDOW = 1_200
MAX_RMSSD_MS = 200.0
MAX_IBI_CORRECTION_RATIO = 0.20


def severe_chunk_count_5s(signal: np.ndarray, times: np.ndarray) -> int:
    """Count severe artefact chunks without running raw-morphology detection."""
    count = 0
    for start in np.arange(float(times[0]), float(times[-1]), RAW_CHUNK_MS):
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, start + RAW_CHUNK_MS, side="left")
        if hi - lo >= 30 and _has_severe_amplitude_artifact(signal[lo:hi]):
            count += 1
    return count


def emrich_local_qrs(signal: np.ndarray, times: np.ndarray, fs: float) -> tuple[np.ndarray, str]:
    try:
        indices = detector_indices(signal, fs, "emrich2023")
        return refine_to_local_qrs_extreme(signal, times, indices, fs), ""
    except Exception as error:
        return np.empty(0, dtype=float), f"emrich2023_failed:{type(error).__name__}"


def rebuild_participant(raw_path: Path, windows_path: Path, out_path: Path, participant: str) -> dict[str, object]:
    with np.load(raw_path, allow_pickle=True) as source:
        raw_signal_all = np.asarray(source["ecg_uv"], dtype=float)
        raw_times_all = np.asarray(source["time_ms"], dtype=float)
    # Keep non-finite ECG samples until per-window QC is computed.  Dropping
    # them first would incorrectly turn an incomplete window into 100% cover.
    finite_time = np.isfinite(raw_times_all)
    raw_signal = raw_signal_all[finite_time]
    raw_times = raw_times_all[finite_time]
    with np.load(windows_path, allow_pickle=True) as source:
        starts = np.asarray(source["ppg_window_t0_ms"], dtype=float)
        ends = np.asarray(source["ppg_window_t1_ms"], dtype=float)
    if starts.shape != ends.shape:
        raise ValueError(f"{participant}: source window bounds mismatch")
    fs = infer_fs(raw_times)

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
    sample_ratio: list[float] = []
    emrich_counts: list[int] = []
    severe_counts: list[int] = []
    qc_pass: list[bool] = []
    review_required: list[bool] = []
    reasons: list[str] = []

    for start, end in zip(starts, ends, strict=True):
        lo = np.searchsorted(raw_times, start, side="left")
        hi = np.searchsorted(raw_times, end, side="right")
        signal, times = raw_signal[lo:hi], raw_times[lo:hi]
        expected_samples = max(1, int(round((end - start) * fs / 1_000.0)))
        one_sample_ratio = float(np.isfinite(signal).sum() / expected_samples) if signal.size else 0.0
        complete_signal = bool(signal.size >= 50 and np.isfinite(signal).all())
        if complete_signal:
            severe_chunk_count = severe_chunk_count_5s(signal, times)
            emrich, detector_error = emrich_local_qrs(signal, times, fs)
        else:
            # The coverage gate below records the root cause.  Do not run a
            # detector or amplitude-artifact heuristic on missing samples.
            severe_chunk_count = 0
            emrich, detector_error = np.empty(0, dtype=float), ""
        peaks = emrich

        rr_all = np.diff(peaks)
        valid_ibi = (rr_all >= IBI_MIN_MS) & (rr_all <= IBI_MAX_MS)
        ratio_valid = float(valid_ibi.mean()) if len(rr_all) else 0.0
        rr_valid = rr_all[valid_ibi]
        corrected, corrected_ratio = _correct_rr(rr_valid)
        raw_rmssd, _, _, _ = _stats(rr_valid)
        corrected_rmssd, one_sdnn, one_mean, one_hr = _stats(corrected)

        min_peak_count = max(3, int(np.floor(((end - start) / 1_000.0) * MIN_HR_BPM / 60.0)))
        issues: list[str] = []
        if one_sample_ratio < MIN_ECG_VALID_SAMPLE_RATIO:
            issues.append("low_ecg_sample_ratio")
        if ratio_valid < MIN_VALID_IBI_RATIO:
            issues.append("low_ecg_valid_ibi_ratio")
        if len(peaks) < min_peak_count:
            issues.append("too_few_ecg_peaks")
        if len(peaks) > MAX_PEAKS_PER_WINDOW:
            issues.append("too_many_ecg_peaks")
        if not np.isfinite(corrected_rmssd):
            issues.append("invalid_ecg_hrv")
        if corrected_ratio > MAX_IBI_CORRECTION_RATIO:
            issues.append("high_ecg_ibi_correction_ratio")
        if np.isfinite(one_hr) and not (MIN_HR_BPM <= one_hr <= MAX_HR_BPM):
            issues.append("ecg_hr_out_of_range")
        if np.isfinite(corrected_rmssd) and corrected_rmssd > MAX_RMSSD_MS:
            issues.append("ecg_rmssd_too_high")
        if severe_chunk_count:
            issues.append("severe_raw_ecg_artifact_chunk")
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
        sample_ratio.append(one_sample_ratio)
        emrich_counts.append(len(emrich))
        severe_counts.append(severe_chunk_count)
        qc_pass.append(passed)
        # A modest correction remains legacy-QC compatible, but it is not a
        # fully automatic reference label: preserve it for review/QC evidence.
        review_required.append(bool(issues) or corrected_ratio > 0.0)
        reasons.append("ok" if passed else "|".join(issues))

    config = {
        "label_version": "ecg_ground_truth_v3_candidate_emrich2023_local_qrs_pan_ungated",
        "candidate_method": "emrich2023_detector + per_event_dominant_raw_qrs_extreme_within_50ms",
        "raw_morphology_role": "not used for peak fill; retained only as a separately validated diagnostic candidate",
        "emrich_local_qrs_radius_ms": 50.0,
        "ibi_correction": "replace RR differing >20% from 11-interval local median; never bridge out-of-range RR",
        "ecg_v3_qc": {
            "valid_sample_ratio_min": MIN_ECG_VALID_SAMPLE_RATIO,
            "valid_ibi_ratio_min": MIN_VALID_IBI_RATIO,
            "min_hr_bpm": MIN_HR_BPM,
            "max_hr_bpm": MAX_HR_BPM,
            "max_peak_count": MAX_PEAKS_PER_WINDOW,
            "max_rmssd_ms": MAX_RMSSD_MS,
            "max_ibi_correction_ratio": MAX_IBI_CORRECTION_RATIO,
            "extra_failures": ["severe_raw_ecg_artifact_chunk", "emrich2023_failed:*"],
            "ibi_correction_policy": "any_nonzero_correction_requires_review",
        },
        "status": "candidate_pending_full_window_holdout_rmssd_validation",
        "excluded_participants": sorted(EXCLUDED_PARTICIPANTS),
        "source_windows": str(windows_path.resolve()),
        "source_raw_ecg": str(raw_path.resolve()),
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
        ecg_valid_sample_ratio=np.asarray(sample_ratio, dtype=np.float32),
        ecg_valid_ibi_ratio=np.asarray(valid_ratio, dtype=np.float32),
        ecg_ibi_correction_ratio=np.asarray(correction_ratio, dtype=np.float32),
        ecg_emrich2023_local_qrs_peak_count=np.asarray(emrich_counts, dtype=np.int16),
        ecg_severe_artifact_chunk_count=np.asarray(severe_counts, dtype=np.int16),
        ecg_v3_qc_pass=np.asarray(qc_pass, dtype=bool),
        ecg_v3_qc_reason=np.asarray(reasons, dtype=object),
        # Compatibility aliases for downstream code expecting the generic name.
        ecg_label_qc_pass=np.asarray(qc_pass, dtype=bool),
        ecg_review_required=np.asarray(review_required, dtype=bool),
        ecg_label_qc_reason=np.asarray(reasons, dtype=object),
    )
    return {
        "participant": participant, "windows": int(len(starts)),
        "qc_pass_windows": int(np.sum(qc_pass)),
        "review_required_windows": int(np.sum(review_required)),
        "qc_pass_pct": float(100 * np.mean(qc_pass)),
        "median_rmssd_ms_qc_pass": float(np.nanmedian(np.asarray(rmssd_corrected)[np.asarray(qc_pass)])) if any(qc_pass) else float("nan"),
        "output": str(out_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--windows-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v2_stride30_rawslots_pan_ungated"))
    parser.add_argument("--out-dir", type=Path, default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/ecg_ground_truth_v3_candidate_emrich_local_qrs_pan_ungated"))
    parser.add_argument("--windows-prefix", default="synced_3device_rawaligned_training_v2_stride30_rawslots_pan_ungated")
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
        windows_path = args.windows_dir / f"{args.windows_prefix}_{participant}.npz"
        if not raw_path.exists() or not windows_path.exists():
            raise FileNotFoundError(f"{participant}: missing raw ECG or source windows")
        jobs.append((raw_path, windows_path, args.out_dir / f"ecg_ground_truth_v3_candidate_emrich_local_qrs_pan_ungated_{participant}.npz", participant))
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(jobs))) as pool:
        futures = {pool.submit(rebuild_participant, *job): job[-1] for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = pd.DataFrame(sorted(rows, key=lambda row: str(row["participant"])))
    summary.to_csv(args.out_dir / "rebuild_summary.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "# ECG ground-truth v3 candidate\n\n"
        "This is a candidate sidecar. It does not overwrite legacy Pan--Tompkins labels. "
        "It remains pending a full-window RMSSD holdout validation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

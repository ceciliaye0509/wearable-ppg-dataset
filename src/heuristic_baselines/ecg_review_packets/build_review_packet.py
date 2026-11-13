"""Build a corrected-waveform, algorithm-assisted blinded review packet.

The output intentionally contains only a small, non-overlapping sample from
each participant. Raw ECG stays on the user's host; the generated packet is
local data and must not be committed to source control or uploaded without
explicit data-handling approval.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent
WORKSPACE_ROOT = BASELINE_DIR.parents[2]
sys.path.insert(0, str(BASELINE_DIR))

from build_blinded_ecg_rpeak_review import (  # noqa: E402
    DETAIL_CENTERS_SEC,
    DETAIL_SEC,
    WINDOW_SEC,
    _candidate_starts,
    _has_severe_amplitude_artifact,
)
from correct_ecg_polarity import apply_manifest_polarity, read_correction_intervals  # noqa: E402
from ecg_review_packets.quality_flags import attach_quality_flags  # noqa: E402
from evaluate_assisted_human_confirmed_rpeaks import (  # noqa: E402
    detector_indices,
    infer_fs,
    refine_to_local_qrs_extreme,
)


DEFAULT_RAW_ROOT = WORKSPACE_ROOT / "Multisite-PPG" / "raw_data"
DEFAULT_CORRECTIONS = BASELINE_DIR / "outputs" / "ecg_polarity_correction_all20_v1" / "correction_intervals.csv"
DEFAULT_OUT_DIR = BASELINE_DIR / "outputs" / "ecg_rpeak_blind_review_remote_v1"
DEFAULT_PRIOR_REVIEW_DIR = BASELINE_DIR / "outputs" / "assisted_ecg_rpeak_final_validation_v4"
DEFAULT_PRIOR_ANNOTATIONS = DEFAULT_PRIOR_REVIEW_DIR / "rpeak_annotations_1_p6_w01_c1_c2_unreadable.csv"
WINDOW_MS = WINDOW_SEC * 1000.0
CLIP_CENTERS_SEC = tuple(DETAIL_CENTERS_SEC)


def _prior_manifests(output_dir: Path) -> list[Path]:
    paths = sorted(output_dir.glob("*/*review_manifest.csv"))
    return [path for path in paths if path.parent.resolve() != DEFAULT_OUT_DIR.resolve()]


def _read_forbidden_starts(paths: list[Path]) -> dict[str, list[float]]:
    forbidden: dict[str, list[float]] = {}
    for path in paths:
        frame = pd.read_csv(path, dtype={"participant": str})
        required = {"participant", "window_start_unix_ms"}
        if not required.issubset(frame.columns):
            continue
        for participant, start in zip(frame.participant, frame.window_start_unix_ms, strict=True):
            if pd.notna(start):
                forbidden.setdefault(str(participant), []).append(float(start))
    return forbidden


def _nearest_index(times_ms: np.ndarray, timestamp_ms: float) -> int:
    right = int(np.searchsorted(times_ms, timestamp_ms, side="left"))
    if right <= 0:
        return 0
    if right >= len(times_ms):
        return len(times_ms) - 1
    return right - 1 if timestamp_ms - times_ms[right - 1] <= times_ms[right] - timestamp_ms else right


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _build_prior_manual_clips(
    *, raw_root: Path, correction_rows: list[dict[str, object]], prior_review_dir: Path,
    annotation_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Rehydrate the previous v4 manual review pages on corrected ECG samples."""
    manifest_path = prior_review_dir / "review_manifest.csv"
    excluded_path = prior_review_dir / "auto_excluded_severe_artifact_clips.csv"
    if not manifest_path.is_file() or not annotation_path.is_file() or not excluded_path.is_file():
        raise FileNotFoundError(
            "Previous manual review source files are missing; expected review_manifest.csv, "
            "auto_excluded_severe_artifact_clips.csv, and the populated rpeak annotation CSV."
        )
    manifests = _read_csv_rows(manifest_path)
    annotations = _read_csv_rows(annotation_path)
    excluded = {row["clip_id"] for row in _read_csv_rows(excluded_path) if row.get("clip_id")}
    required = {"review_id", "participant", "window_start_unix_ms", "manual_rpeak_abs_unix_ms"}
    if not annotations or not required.issubset(annotations[0]):
        raise ValueError(f"Prior annotation file must include: {', '.join(sorted(required))}")

    mark_records_by_review: dict[str, list[tuple[float, str]]] = {}
    unreadable_by_review: dict[str, dict[int, str]] = {}
    for row in annotations:
        review_id = row["review_id"]
        peak_value = (row.get("manual_rpeak_abs_unix_ms") or "").strip()
        if peak_value:
            confidence = (row.get("confidence") or "high").strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "high"
            mark_records_by_review.setdefault(review_id, []).append((float(peak_value), confidence))
        note = (row.get("notes") or "").strip()
        match = re.search(r"unresolvable_raw_ecg_chunk:.*_C(\d+)$", note)
        if match:
            unreadable_by_review.setdefault(review_id, {})[int(match.group(1))] = note

    clips: list[dict[str, object]] = []
    legacy_manifest_rows: list[dict[str, object]] = []
    assigned_marks = 0
    covered_review_ids: set[str] = set()
    unmatched_marks: list[float] = []

    for manifest in manifests:
        review_id = manifest["review_id"]
        participant = manifest["participant"]
        window_start = float(manifest["window_start_unix_ms"])
        excerpt_ranges: list[tuple[float, float]] = []
        for excerpt in manifest["detail_excerpt_seconds"].split(";"):
            start_text, end_text = excerpt.split("-", maxsplit=1)
            start_sec, end_sec = float(start_text), float(end_text)
            if end_sec <= start_sec:
                raise ValueError(f"{review_id}: invalid excerpt range {excerpt!r}")
            excerpt_ranges.append((start_sec, end_sec))

        source_path = raw_root / participant / f"{participant}_polar_ecg_raw.npz"
        if not source_path.is_file():
            raise FileNotFoundError(f"{participant}: missing raw ECG source {source_path}")
        with np.load(source_path, allow_pickle=True) as source:
            times = np.asarray(source["time_ms"], dtype=np.float64)
            signal = np.asarray(source["ecg_uv"], dtype=np.float64)
        valid = np.isfinite(times) & np.isfinite(signal)
        times, signal = times[valid], signal[valid]
        corrected, _ = apply_manifest_polarity(participant, signal, times, correction_rows)
        unique = np.r_[True, np.diff(times) > 0]
        times, corrected = times[unique], corrected[unique]
        if times.size < 2 or np.any(np.diff(times) < 0):
            raise ValueError(f"{participant}: prior ECG source timestamps are invalid")

        review_marks = mark_records_by_review.get(review_id, [])
        mark_groups: list[list[tuple[float, str]]] = [[] for _ in excerpt_ranges]
        for timestamp, confidence in review_marks:
            matching = [
                index for index, (start_sec, end_sec) in enumerate(excerpt_ranges)
                if window_start + start_sec * 1000.0 <= timestamp <= window_start + end_sec * 1000.0
            ]
            if not matching:
                unmatched_marks.append(timestamp)
                continue
            chosen = min(
                matching,
                key=lambda index: abs(
                    timestamp - window_start - sum(excerpt_ranges[index]) * 500.0
                ),
            )
            mark_groups[chosen].append((timestamp, confidence))

        for clip_rank, ((start_sec, end_sec), clip_marks) in enumerate(zip(excerpt_ranges, mark_groups, strict=True), start=1):
            legacy_id = f"prior_v4_{review_id}_C{clip_rank}"
            if f"{review_id}_C{clip_rank}" in excluded:
                continue
            left_ms = window_start + start_sec * 1000.0
            right_ms = window_start + end_sec * 1000.0
            left = int(np.searchsorted(times, left_ms, side="left"))
            right = int(np.searchsorted(times, right_ms, side="right"))
            clip_times, clip_signal = times[left:right], corrected[left:right]
            if len(clip_times) < 50:
                if clip_marks:
                    raise ValueError(f"{legacy_id}: prior peaks exist but the ECG excerpt is incomplete")
                continue
            peak_indices: list[int] = []
            for timestamp, _ in clip_marks:
                index = _nearest_index(clip_times, timestamp)
                if abs(float(clip_times[index]) - timestamp) > 50.0:
                    raise ValueError(f"{legacy_id}: prior peak cannot be aligned within 50 ms")
                peak_indices.append(index)
            peak_indices = sorted(set(peak_indices))
            confidence_counts: dict[str, int] = {}
            for _, confidence in clip_marks:
                confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            prior_confidence = max(confidence_counts, key=confidence_counts.get) if confidence_counts else "high"
            note = unreadable_by_review.get(review_id, {}).get(clip_rank, "")
            if not peak_indices and not note and not review_marks:
                continue
            # Existing outputs represent whole review windows. Include every non-excluded
            # excerpt for a window with annotations, including empty excerpts, so they can
            # be corrected in the same workspace.
            clip_start = float(clip_times[0])
            clips.append({
                "id": legacy_id,
                "reviewId": review_id,
                "reviewSet": "assisted_ecg_rpeak_final_validation_v4",
                "sourceLabel": "Previous manual review (v4)",
                "isPrevious": True,
                "hasPriorMarks": bool(peak_indices),
                "priorUnreadable": bool(note),
                "priorNote": "Previously marked as unreadable." if note else "",
                "priorConfidence": prior_confidence,
                "participant": participant,
                "windowStart": round(window_start, 3),
                "clipStart": round(clip_start, 3),
                "startSec": round((clip_start - window_start) / 1000.0, 3),
                "endSec": round((float(clip_times[-1]) - window_start) / 1000.0, 3),
                "t": np.round((clip_times - clip_start) / 1000.0, 4).tolist(),
                "x": np.round(clip_signal, 2).tolist(),
                "suggested": [],
                "restorePeaks": peak_indices,
                "aidAvailable": False,
            })
            legacy_manifest_rows.append({
                "clip_id": legacy_id,
                "review_id": review_id,
                "participant": participant,
                "window_start_unix_ms": round(window_start, 3),
                "segment_start_sec": start_sec,
                "segment_end_sec": end_sec,
                "previous_manual_peak_count": len(peak_indices),
                "previous_status": "unresolvable" if note else "annotated" if peak_indices else "no_saved_peaks",
                "waveform_polarity_corrected": True,
            })
            assigned_marks += len(peak_indices)
            covered_review_ids.add(review_id)

    if unmatched_marks:
        raise ValueError(f"{len(unmatched_marks)} prior manual peaks did not map to any source segment")
    expected_marks = sum(len(values) for values in mark_records_by_review.values())
    if assigned_marks != expected_marks:
        raise ValueError(f"Prior annotation import lost marks: mapped {assigned_marks} of {expected_marks}")
    if not clips:
        raise RuntimeError("No previous manually annotated ECG segments could be rebuilt")
    return clips, legacy_manifest_rows, {
        "segment_count": len(clips),
        "review_window_count": len(covered_review_ids),
        "manual_peak_count": assigned_marks,
        "unreadable_segment_count": sum(bool(item["priorUnreadable"]) for item in clips),
    }


def _choose_nonoverlapping_starts(candidates: np.ndarray, requested: int) -> np.ndarray:
    """Pick up to the requested count with no mutual 5-minute overlap."""
    values = np.asarray(candidates, dtype=float)
    if not values.size or requested < 1:
        return np.empty(0, dtype=float)
    max_count = 1
    previous = float(values[0])
    for value in values[1:]:
        if float(value) - previous >= WINDOW_MS:
            max_count += 1
            previous = float(value)
    count = min(requested, max_count)
    targets = values[np.linspace(0, len(values) - 1, count).round().astype(int)]
    chosen: list[float] = []
    for index, target in enumerate(targets):
        lower = chosen[-1] + WINDOW_MS if chosen else float(values[0])
        remaining = count - index - 1
        upper = float(values[-1]) - remaining * WINDOW_MS
        left = int(np.searchsorted(values, lower, side="left"))
        right = int(np.searchsorted(values, upper, side="right"))
        options = values[left:right]
        if not options.size:
            break
        chosen.append(float(options[int(np.argmin(np.abs(options - target)))]))
    return np.asarray(chosen, dtype=float)


def build_packet(
    *, raw_root: Path, correction_path: Path, output_dir: Path, windows_per_participant: int,
    prior_review_dir: Path = DEFAULT_PRIOR_REVIEW_DIR,
    prior_annotation_path: Path = DEFAULT_PRIOR_ANNOTATIONS,
) -> dict[str, object]:
    raw_files = sorted(raw_root.glob("P*/P*_polar_ecg_raw.npz"), key=lambda p: int(p.parent.name[1:]))
    if not raw_files:
        raise FileNotFoundError(f"No Polar ECG files found under {raw_root}")
    correction_rows = read_correction_intervals(correction_path)
    manifests = _prior_manifests(BASELINE_DIR / "outputs")
    forbidden = _read_forbidden_starts(manifests)
    clips: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []

    for raw_path in raw_files:
        participant = raw_path.parent.name
        with np.load(raw_path, allow_pickle=True) as source:
            times = np.asarray(source["time_ms"], dtype=np.float64)
            signal = np.asarray(source["ecg_uv"], dtype=np.float64)
        valid = np.isfinite(times) & np.isfinite(signal)
        times, signal = times[valid], signal[valid]
        if times.size < 50 or np.any(np.diff(times) < 0):
            raise ValueError(f"{participant}: raw ECG time values must be finite and non-decreasing")
        duplicate_times = times[:-1][np.diff(times) == 0]
        corrected, flip_mask = apply_manifest_polarity(participant, signal, times, correction_rows)
        keep = np.r_[True, np.diff(times) > 0]
        removed_duplicate_times = int(len(times) - np.count_nonzero(keep))
        times, corrected, flip_mask = times[keep], corrected[keep], flip_mask[keep]
        starts_available = _candidate_starts(times)
        blocked = np.asarray(forbidden.get(participant, []), dtype=float)
        if blocked.size:
            eligible = starts_available[
                np.all(np.abs(starts_available[:, None] - blocked[None, :]) >= WINDOW_MS, axis=1)
            ]
        else:
            eligible = starts_available
        if duplicate_times.size and eligible.size:
            duplicate_positions = np.searchsorted(duplicate_times, eligible, side="left")
            in_range = duplicate_positions < len(duplicate_times)
            has_duplicate = np.zeros(len(eligible), dtype=bool)
            has_duplicate[in_range] = (
                duplicate_times[duplicate_positions[in_range]] <= eligible[in_range] + WINDOW_MS
            )
            eligible = eligible[~has_duplicate]
        count = min(windows_per_participant, len(eligible))
        chosen_starts = _choose_nonoverlapping_starts(eligible, count)
        count = len(chosen_starts)
        if count < 1:
            participant_rows.append({
                "participant": participant,
                "window_count": 0,
                "clip_count": 0,
                "removed_duplicate_timestamp_samples": removed_duplicate_times,
                "reason": "no non-overlapping complete review window remains",
            })
            continue
        participant_clip_count = 0
        for rank, window_start in enumerate(chosen_starts, start=1):
            window_left = int(np.searchsorted(times, window_start, side="left"))
            window_right = int(np.searchsorted(times, window_start + WINDOW_MS, side="right"))
            wt, wx = times[window_left:window_right], corrected[window_left:window_right]
            if len(wt) < 50 or np.any(np.diff(wt) > 250.0):
                continue
            window_id = f"{participant}_W{rank:02d}"
            fs = infer_fs(wt)
            try:
                detector_events = detector_indices(wx, fs, "neurokit")
                candidate_times = refine_to_local_qrs_extreme(wx, wt, detector_events, fs)
            except Exception:
                candidate_times = np.empty(0, dtype=float)
            manifest_rows.append({
                "review_id": window_id,
                "participant": participant,
                "window_start_unix_ms": round(float(window_start), 3),
                "window_end_unix_ms": round(float(window_start + WINDOW_MS), 3),
                "sampling_rate_hz": round(float(fs), 4),
                "corrected_samples_in_window": int(np.count_nonzero(flip_mask[window_left:window_right])),
                "duplicate_timestamp_samples_in_window": 0,
            })

            for clip_rank, center in enumerate(CLIP_CENTERS_SEC, start=1):
                clip_start = float(window_start + (center - DETAIL_SEC / 2.0) * 1000.0)
                clip_end = float(window_start + (center + DETAIL_SEC / 2.0) * 1000.0)
                left = int(np.searchsorted(wt, clip_start, side="left"))
                right = int(np.searchsorted(wt, clip_end, side="right"))
                ct, cx = wt[left:right], wx[left:right]
                if len(ct) < 50:
                    continue
                severe = _has_severe_amplitude_artifact(cx)
                suggested: list[int] = []
                if not severe and candidate_times.size:
                    selected = candidate_times[(candidate_times >= ct[0]) & (candidate_times <= ct[-1])]
                    suggested = sorted({_nearest_index(ct, float(value)) for value in selected})
                clip_id = f"{window_id}_C{clip_rank}"
                clips.append({
                    "id": clip_id,
                    "reviewId": window_id,
                    "reviewSet": "ecg_rpeak_blind_review_remote_v2",
                    "sourceLabel": "New algorithm-assisted review",
                    "isPrevious": False,
                    "hasPriorMarks": False,
                    "priorUnreadable": False,
                    "participant": participant,
                    "windowStart": round(float(window_start), 3),
                    "clipStart": round(float(ct[0]), 3),
                    "startSec": round(float((ct[0] - window_start) / 1000.0), 3),
                    "endSec": round(float((ct[-1] - window_start) / 1000.0), 3),
                    "t": np.round((ct - ct[0]) / 1000.0, 4).tolist(),
                    "x": np.round(cx, 2).tolist(),
                    "suggested": suggested,
                    "restorePeaks": suggested,
                    "aidAvailable": bool(not severe and candidate_times.size),
                })
                participant_clip_count += 1

        participant_rows.append({
            "participant": participant,
            "window_count": sum(row["participant"] == participant for row in manifest_rows),
            "clip_count": participant_clip_count,
            "correction_applied_samples_in_source": int(np.count_nonzero(flip_mask)),
            "removed_duplicate_timestamp_samples": removed_duplicate_times,
            "reason": "",
        })

    prior_clips, prior_manifest_rows, prior_summary = _build_prior_manual_clips(
        raw_root=raw_root,
        correction_rows=correction_rows,
        prior_review_dir=prior_review_dir,
        annotation_path=prior_annotation_path,
    )
    prior_counts: dict[str, int] = {}
    for clip in prior_clips:
        participant = str(clip["participant"])
        prior_counts[participant] = prior_counts.get(participant, 0) + 1
    participant_row_by_id = {str(row["participant"]): row for row in participant_rows}
    for participant, count in prior_counts.items():
        if participant not in participant_row_by_id:
            participant_rows.append({
                "participant": participant,
                "window_count": 0,
                "clip_count": 0,
                "new_clip_count": 0,
                "correction_applied_samples_in_source": 0,
                "removed_duplicate_timestamp_samples": 0,
                "reason": "no new review clips; prior manual review segments available",
            })
            participant_row_by_id[participant] = participant_rows[-1]
    for row in participant_rows:
        participant = str(row["participant"])
        new_count = int(row.get("new_clip_count", row.get("clip_count", 0)))
        prior_count = prior_counts.get(participant, 0)
        row["new_clip_count"] = new_count
        row["prior_manual_clip_count"] = prior_count
        row["clip_count"] = new_count + prior_count

    clips = prior_clips + clips
    if not clips:
        raise RuntimeError("No review clips could be generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "participants": participant_rows, "clips": clips}
    screening_summary = attach_quality_flags(payload, raw_root=raw_root, correction_dir=correction_path.parent)
    payload_path = output_dir / "review_data.json.gz"
    with gzip.open(payload_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)

    pd.DataFrame(manifest_rows).to_csv(output_dir / "review_manifest.csv", index=False)
    pd.DataFrame(prior_manifest_rows).to_csv(output_dir / "prior_manual_review_manifest.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(output_dir / "participant_manifest.csv", index=False)
    protocol = {
        "version": "ecg_rpeak_blind_review_remote_v2",
        "purpose": "web review of previous manual marks plus new algorithm-assisted human verification",
        "polarity_correction_manifest": str(correction_path.resolve()),
        "raw_data_root": str(raw_root.resolve()),
        "sampled_review_windows_per_participant_max": int(windows_per_participant),
        "window_seconds": WINDOW_SEC,
        "clips_per_window": len(CLIP_CENTERS_SEC),
        "clip_seconds": DETAIL_SEC,
        "participants": [row["participant"] for row in participant_rows],
        "candidate_method_for_audit_only": "NeuroKit detector + per-candidate dominant raw-QRS extremum within +/-50 ms on polarity-corrected ECG",
        "candidate_identity_hidden_from_reviewer_ui": True,
        "previous_manual_annotations": str(prior_annotation_path.resolve()),
        "previous_review_manifest": str((prior_review_dir / "review_manifest.csv").resolve()),
        "previous_manual_review_summary": prior_summary,
        "quality_screening_summary": screening_summary,
        "quality_screening_protocol": payload["qualityScreeningProtocol"],
        "previous_manual_waveforms": "rebuilt from the original raw NPZ files, polarity-corrected, and shown as a separate prior-review source; Restore returns to imported human marks",
        "prior_review_manifests_avoided": [str(path.resolve()) for path in manifests],
        "data_handling": "review_data.json.gz contains coded participant IDs and sampled ECG only; keep it on an approved user-controlled host",
        "timestamp_cleanup": "all 5-minute windows containing duplicate timestamps are excluded; samples outside selected windows are reduced to the first row per timestamp for index construction only; original NPZ is unchanged, with source counts in participant_manifest.csv",
        "warning": "Because candidate peaks are visible, this is algorithm-assisted verification, not an independent blind detector-accuracy estimate.",
    }
    (output_dir / "review_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "participant_count": len(participant_rows),
        "participants_with_clips": sum(int(row["clip_count"]) > 0 for row in participant_rows),
        "window_count": len(manifest_rows),
        "clip_count": len(clips),
        "new_clip_count": len(clips) - len(prior_clips),
        "prior_manual_review": prior_summary,
        "quality_screening": screening_summary,
        "data_path": str(payload_path),
        "packet_bytes": payload_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--correction-intervals", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--windows-per-participant", type=int, default=4)
    parser.add_argument("--prior-review-dir", type=Path, default=DEFAULT_PRIOR_REVIEW_DIR)
    parser.add_argument("--prior-annotations", type=Path, default=DEFAULT_PRIOR_ANNOTATIONS)
    args = parser.parse_args()
    if args.windows_per_participant < 1:
        raise ValueError("--windows-per-participant must be positive")
    result = build_packet(
        raw_root=args.raw_root,
        correction_path=args.correction_intervals,
        output_dir=args.out_dir,
        windows_per_participant=args.windows_per_participant,
        prior_review_dir=args.prior_review_dir,
        prior_annotation_path=args.prior_annotations,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

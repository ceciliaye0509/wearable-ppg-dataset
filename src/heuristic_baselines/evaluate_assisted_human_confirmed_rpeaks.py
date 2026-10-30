"""Audit ECG peak candidates against assisted, human-confirmed QRS timings.

This evaluator is deliberately limited to excerpt-level peak detection and
timing.  Five-second excerpts are not used to estimate a five-minute RMSSD.
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

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _independent_raw_r_candidates


EVENT_TOLERANCE_MS = 150.0
DEPLOYMENT_CHUNK_MS = 20_000.0
EXCLUDED_PARTICIPANTS = {"P2", "P5", "P14", "P16", "P17"}
UNRESOLVABLE_PREFIX = "unresolvable_raw_ecg_chunk:"


def infer_fs(times_ms: np.ndarray) -> float:
    steps = np.diff(np.asarray(times_ms, dtype=float))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if not len(steps):
        raise ValueError("cannot infer ECG sampling frequency")
    return float(1000.0 / np.median(steps))


def pan_tompkins_indices(signal: np.ndarray, fs: float) -> np.ndarray:
    """Replicate the old label generator: detect within its full five-minute window."""
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    idx = idx[(idx >= 0) & (idx < len(signal))]
    return idx


def detector_indices(signal: np.ndarray, fs: float, method: str) -> np.ndarray:
    """Run a NeuroKit detector with its intended cleaning convention."""
    # Only FastNVG has an algorithm-specific cleaner in this NeuroKit build.
    # Khamis and ProMAC are peak detectors, not accepted ecg_clean methods.
    cleaning_method = "emrich2023" if method == "emrich2023" else "neurokit"
    clean = nk.ecg_clean(signal, sampling_rate=fs, method=cleaning_method)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
    idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return idx[(idx >= 0) & (idx < len(signal))]


def refine_to_dominant_raw_extreme(signal: np.ndarray, times_ms: np.ndarray, indices: np.ndarray, fs: float) -> np.ndarray:
    """Snap detector events to the locally dominant raw-QRS extreme."""
    baseline = float(np.median(signal))
    polarity = 1.0 if np.quantile(signal, 0.995) - baseline >= baseline - np.quantile(signal, 0.005) else -1.0
    radius = max(1, int(round(0.150 * fs)))
    refined: list[int] = []
    for index in np.asarray(indices, dtype=int):
        left, right = max(0, index - radius), min(len(signal), index + radius + 1)
        if left < right:
            refined.append(left + int(np.argmax(polarity * signal[left:right])))
    return times_ms[np.unique(np.asarray(refined, dtype=int))] if refined else np.empty(0, dtype=float)


def refine_to_local_qrs_extreme(signal: np.ndarray, times_ms: np.ndarray, indices: np.ndarray, fs: float) -> np.ndarray:
    """Snap each detector event to its local dominant QRS extreme within ±50 ms.

    Unlike the legacy global-polarity refinement, polarity is chosen per event.
    This avoids moving an upright R apex to its following S trough when another
    recording segment happens to have larger negative excursions.
    """
    x = np.asarray(signal, dtype=float)
    radius = max(1, int(round(0.050 * fs)))
    refined: list[int] = []
    for index in np.asarray(indices, dtype=int):
        left, right = max(0, index - radius), min(len(x), index + radius + 1)
        segment = x[left:right]
        if not len(segment):
            continue
        baseline = float(np.median(segment))
        upward = left + int(np.argmax(segment))
        downward = left + int(np.argmin(segment))
        refined.append(upward if abs(x[upward] - baseline) >= abs(x[downward] - baseline) else downward)
    return times_ms[np.unique(np.asarray(refined, dtype=int))] if refined else np.empty(0, dtype=float)


def raw_first_pan_fill(raw_morphology: np.ndarray, pan_refined: np.ndarray) -> np.ndarray:
    selected = list(np.asarray(raw_morphology, dtype=float))
    for peak in np.asarray(pan_refined, dtype=float):
        if not selected or min(abs(peak - current) for current in selected) > EVENT_TOLERANCE_MS:
            selected.append(float(peak))
    return np.asarray(sorted(selected), dtype=float)


def raw_emrich_with_short_rr_prune(raw_morphology: np.ndarray, emrich_local: np.ndarray) -> tuple[np.ndarray, int]:
    """Merge raw-first and Emrich candidates, removing impossible duplicates.

    A retained pair <300 ms apart cannot represent two physiologic QRS events
    under the project HR inclusion range. Preserve a consensus event first,
    then an Emrich-only event, then a raw-only event. Ties use the neighboring
    RR interval that is closest to the local valid-RR median.
    """
    raw = np.unique(np.asarray(raw_morphology, dtype=float))
    emrich = np.unique(np.asarray(emrich_local, dtype=float))
    events: list[tuple[float, int]] = []
    for point in raw:
        support = 2 if len(emrich) and np.min(np.abs(emrich - point)) <= EVENT_TOLERANCE_MS else 0
        events.append((float(point), support))
    for point in emrich:
        if not len(raw) or np.min(np.abs(raw - point)) > EVENT_TOLERANCE_MS:
            events.append((float(point), 1))
    events.sort()
    pruned = 0
    while len(events) > 1:
        peaks = np.asarray([event[0] for event in events], dtype=float)
        short = np.flatnonzero(np.diff(peaks) < 300.0)
        if not len(short):
            break
        left_index = int(short[0])
        right_index = left_index + 1
        left_support, right_support = events[left_index][1], events[right_index][1]
        if left_support != right_support:
            remove = left_index if left_support < right_support else right_index
        else:
            valid_rr = np.diff(peaks)
            valid_rr = valid_rr[(valid_rr >= 300.0) & (valid_rr <= 2_000.0)]
            target = float(np.median(valid_rr)) if len(valid_rr) else 800.0
            def neighbor_cost(remove_index: int) -> float:
                before, after = remove_index - 1, remove_index + 1
                if before < 0 or after >= len(peaks):
                    return 0.0
                return abs((peaks[after] - peaks[before]) - target)
            remove = left_index if neighbor_cost(left_index) <= neighbor_cost(right_index) else right_index
        events.pop(remove)
        pruned += 1
    return np.asarray([event[0] for event in events], dtype=float), pruned


def emrich_promac_with_v3_long_gap_rescue(
    emrich_promac: np.ndarray,
    v3: np.ndarray,
    *,
    long_gap_ratio: float = 1.5,
) -> np.ndarray:
    """Add a v3-only event only when it plausibly fills one missing beat.

    This is deliberately *not* the unrestricted union of two detectors.  A
    v3-only proposal is admitted only if it splits an Emrich+ProMAC interval
    that is at least ``long_gap_ratio`` times the local median RR, and both
    resulting subintervals remain physiologically plausible relative to that
    local rhythm.  It is a candidate policy to be frozen before use on the
    fresh adjudication set, not a post-hoc correction of human labels.
    """
    base = np.unique(np.sort(np.asarray(emrich_promac, dtype=float)))
    backup = np.unique(np.sort(np.asarray(v3, dtype=float)))
    if base.size < 3 or not backup.size:
        return base
    rr = np.diff(base)
    backup_rr = np.diff(backup)
    def _physiologic_median(values: np.ndarray) -> float:
        values = values[(values >= 300.0) & (values <= 2_000.0)]
        return float(np.median(values)) if len(values) else float("nan")
    # If the primary detector skips every second beat in a short clip, its own
    # median RR is inflated and would prevent a rescue.  The smaller valid
    # rhythm estimate is conservative for testing whether a primary gap is
    # compatible with one missing beat; the subsequent split constraints still
    # reject arbitrary v3-only extras.
    local_median = float(np.nanmin([_physiologic_median(rr), _physiologic_median(backup_rr)]))
    if not np.isfinite(local_median) or local_median <= 0:
        return base
    chosen = list(base)
    for candidate in backup:
        if np.min(np.abs(base - candidate)) <= EVENT_TOLERANCE_MS:
            continue
        position = int(np.searchsorted(base, candidate))
        if position == 0 or position == len(base):
            continue
        previous, following = float(base[position - 1]), float(base[position])
        gap = following - previous
        left_rr, right_rr = candidate - previous, following - candidate
        if (
            gap >= long_gap_ratio * local_median
            and left_rr >= 0.35 * local_median
            and right_rr >= 0.35 * local_median
            and left_rr <= 1.35 * local_median
            and right_rr <= 1.35 * local_median
        ):
            chosen.append(float(candidate))
    return np.unique(np.sort(np.asarray(chosen, dtype=float)))


def one_to_one_match(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance_ms: float = EVENT_TOLERANCE_MS,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    i = j = 0
    matched: list[tuple[int, int]] = []
    missed: list[int] = []
    extra: list[int] = []
    while i < len(reference) and j < len(candidate):
        delta = float(candidate[j] - reference[i])
        if abs(delta) <= tolerance_ms:
            matched.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            extra.append(j)
            j += 1
        else:
            missed.append(i)
            i += 1
    missed.extend(range(i, len(reference)))
    extra.extend(range(j, len(candidate)))
    return matched, missed, extra


def aggregate(metrics: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, table in metrics.groupby(groups, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        reference = int(table.reference_peaks.sum())
        candidate = int(table.candidate_peaks.sum())
        matched = int(table.matched_peaks.sum())
        precision = matched / candidate if candidate else float("nan")
        recall = matched / reference if reference else float("nan")
        deltas = table.loc[table.timing_median_error_ms.notna(), "timing_median_error_ms"]
        p95 = table.loc[table.timing_p95_abs_error_ms.notna(), "timing_p95_abs_error_ms"]
        record = dict(zip(groups, key, strict=True))
        record.update({
            "clips": len(table),
            "reference_peaks": reference,
            "candidate_peaks": candidate,
            "matched_peaks": matched,
            "missed_peaks": reference - matched,
            "extra_peaks": candidate - matched,
            "precision_pct": 100.0 * precision,
            "recall_pct": 100.0 * recall,
            "f1_pct": 200.0 * precision * recall / (precision + recall) if precision + recall else float("nan"),
            "median_clip_timing_error_ms": float(deltas.median()) if len(deltas) else float("nan"),
            "median_clip_p95_abs_timing_error_ms": float(p95.median()) if len(p95) else float("nan"),
        })
        records.append(record)
    return pd.DataFrame(records)


def parse_clip_ranges(detail_excerpt_seconds: str) -> list[tuple[float, float]]:
    ranges = []
    for pair in str(detail_excerpt_seconds).split(";"):
        left, right = pair.split("-")
        ranges.append((float(left), float(right)))
    return ranges


def deployment_candidates_for_clip(
    signal: np.ndarray, times: np.ndarray, left_ms: float, right_ms: float, *, chunk_ms: float
) -> np.ndarray:
    """Reproduce only deployment chunks touching one review clip.

    This is equivalent to the batch generator for continuous recordings, while
    avoiding an unnecessary full-recording detector pass during validation.
    """
    origin = float(times[0])
    first = int(np.floor((left_ms - origin) / chunk_ms))
    last = int(np.floor((right_ms - origin) / chunk_ms))
    proposals: list[np.ndarray] = []
    for chunk_index in range(max(0, first), last + 1):
        start = origin + chunk_index * chunk_ms
        end = start + chunk_ms
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, end, side="left")
        chunk_signal, chunk_times = signal[lo:hi], times[lo:hi]
        if len(chunk_times) < 30 or _has_severe_amplitude_artifact(chunk_signal):
            continue
        indices = _independent_raw_r_candidates(chunk_signal, chunk_times)
        if len(indices):
            proposals.append(chunk_times[indices])
    all_candidates = np.unique(np.concatenate(proposals)) if proposals else np.empty(0, dtype=float)
    return all_candidates[(all_candidates >= left_ms) & (all_candidates <= right_ms)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--auto-excluded-clips",
        type=Path,
        help="CSV audit list of severe-artifact clips that were not reviewable.",
    )
    parser.add_argument(
        "--exclude-clips",
        nargs="*",
        default=(),
        help="Additional known-unreadable clip IDs, retained outside an annotation CSV.",
    )
    parser.add_argument(
        "--reference-label",
        default="raw-morphology-assisted, human-confirmed QRS timing",
        help="provenance label written verbatim to the validation report",
    )
    parser.add_argument(
        "--event-tolerance-ms",
        type=float,
        default=EVENT_TOLERANCE_MS,
        help="one-to-one peak-match tolerance; candidate construction remains unchanged",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--include-confidence",
        nargs="+",
        default=None,
        help="Optional confidence values to retain from the adjudication CSV (for example: high).",
    )
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations, dtype={"review_id": str, "participant": str, "notes": str})
    manifest = pd.read_csv(args.manifest, dtype={"review_id": str, "participant": str})
    required = {"review_id", "participant", "manual_rpeak_abs_unix_ms", "notes"}
    if missing := required - set(annotations.columns):
        raise ValueError(f"annotations missing required columns: {sorted(missing)}")

    annotations["notes"] = annotations["notes"].fillna("")
    unresolvable = {
        note[len(UNRESOLVABLE_PREFIX):]
        for note in annotations.notes
        if note.startswith(UNRESOLVABLE_PREFIX)
    }
    severe_artifact: set[str] = set()
    if args.auto_excluded_clips:
        severe = pd.read_csv(args.auto_excluded_clips, dtype={"clip_id": str})
        if "clip_id" not in severe:
            raise ValueError("auto-excluded CSV must contain clip_id")
        severe_artifact = set(severe.clip_id.dropna().astype(str))
    explicit_exclusions = set(args.exclude_clips)
    excluded_clips = unresolvable | severe_artifact | explicit_exclusions
    annotations["manual_rpeak_abs_unix_ms"] = pd.to_numeric(
        annotations["manual_rpeak_abs_unix_ms"], errors="coerce"
    )
    if args.include_confidence is not None:
        if "confidence" not in annotations.columns:
            raise ValueError("--include-confidence requires a confidence column in annotations")
        annotations = annotations[annotations["confidence"].isin(args.include_confidence)].copy()
    annotations = annotations[
        annotations.participant.notna()
        & ~annotations.participant.isin(EXCLUDED_PARTICIPANTS)
        & annotations.manual_rpeak_abs_unix_ms.notna()
    ].copy()
    manifest = manifest[~manifest.participant.isin(EXCLUDED_PARTICIPANTS)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    metric_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    evaluated_clips: list[str] = []
    detector_errors: dict[str, list[str]] = {method: [] for method in ("emrich2023", "khamis2016", "promac")}

    for _, window in manifest.iterrows():
        review_id = str(window.review_id)
        participant = str(window.participant)
        raw_path = str(window.raw_ecg_path)
        if raw_path not in cache:
            with np.load(raw_path, allow_pickle=True) as source:
                signal = np.asarray(source["ecg_uv"], dtype=float)
                times = np.asarray(source["time_ms"], dtype=float)
            valid = np.isfinite(signal) & np.isfinite(times)
            cache[raw_path] = signal[valid], times[valid], infer_fs(times[valid])
        raw, raw_times, fs = cache[raw_path]
        start, end = float(window.window_start_unix_ms), float(window.window_end_unix_ms)
        lo, hi = np.searchsorted(raw_times, start, side="left"), np.searchsorted(raw_times, end, side="right")
        full_signal, full_times = raw[lo:hi], raw_times[lo:hi]
        try:
            full_pan_indices = pan_tompkins_indices(full_signal, fs)
            full_pan = full_times[full_pan_indices]
            full_pan_refined = refine_to_dominant_raw_extreme(full_signal, full_times, full_pan_indices, fs)
        except Exception:
            full_pan = np.empty(0, dtype=float)
            full_pan_refined = np.empty(0, dtype=float)
        secondary_refined: dict[str, np.ndarray] = {}
        secondary_events: dict[str, np.ndarray] = {}
        secondary_local: dict[str, np.ndarray] = {}
        for method in detector_errors:
            try:
                indices = detector_indices(full_signal, fs, method)
                secondary_events[method] = full_times[indices]
                secondary_refined[method] = refine_to_dominant_raw_extreme(full_signal, full_times, indices, fs)
                secondary_local[method] = refine_to_local_qrs_extreme(full_signal, full_times, indices, fs)
            except Exception as error:
                detector_errors[method].append(f"{review_id}: {type(error).__name__}: {error}")

        for clip_number, (left_s, right_s) in enumerate(parse_clip_ranges(window.detail_excerpt_seconds), start=1):
            clip_id = f"{review_id}_C{clip_number}"
            if clip_id in excluded_clips:
                continue
            left, right = start + 1000.0 * left_s, start + 1000.0 * right_s
            clip_lo = np.searchsorted(raw_times, left, side="left")
            clip_hi = np.searchsorted(raw_times, right, side="right")
            clip_signal, clip_times = raw[clip_lo:clip_hi], raw_times[clip_lo:clip_hi]
            reference = np.sort(
                annotations.loc[
                    annotations.review_id.eq(review_id)
                    & annotations.manual_rpeak_abs_unix_ms.between(left, right),
                    "manual_rpeak_abs_unix_ms",
                ].to_numpy(float)
            )
            raw_morphology = clip_times[_independent_raw_r_candidates(clip_signal, clip_times)]
            deployment_raw_5s = deployment_candidates_for_clip(raw, raw_times, left, right, chunk_ms=5_000.0)
            pan_refined = full_pan_refined[(full_pan_refined >= left) & (full_pan_refined <= right)]
            methods = {
                "raw_morphology_app_candidates": raw_morphology,
                "raw_morphology_deployment_20s_chunks": deployment_candidates_for_clip(raw, raw_times, left, right, chunk_ms=DEPLOYMENT_CHUNK_MS),
                "raw_morphology_deployment_5s_chunks": deployment_raw_5s,
                "legacy_window_pantompkins1985": full_pan[(full_pan >= left) & (full_pan <= right)],
                "pantompkins1985_raw_apex_refined": pan_refined,
                "raw_morphology_plus_pan_raw_apex_fill": raw_first_pan_fill(raw_morphology, pan_refined),
                "deployment_5s_raw_morphology_plus_pan_raw_apex_fill": raw_first_pan_fill(deployment_raw_5s, pan_refined),
            }
            for method, refined in secondary_refined.items():
                detector_for_clip = refined[(refined >= left) & (refined <= right)]
                detector_events_for_clip = secondary_events[method][
                    (secondary_events[method] >= left) & (secondary_events[method] <= right)
                ]
                methods[f"{method}_detector_event"] = detector_events_for_clip
                methods[f"raw_morphology_plus_{method}_detector_event_fill"] = raw_first_pan_fill(
                    raw_morphology, detector_events_for_clip
                )
                methods[f"{method}_detector_event_plus_raw_morphology_fill"] = raw_first_pan_fill(
                    detector_events_for_clip, raw_morphology
                )
                methods[f"{method}_raw_apex_refined"] = detector_for_clip
                methods[f"raw_morphology_plus_{method}_raw_apex_fill"] = raw_first_pan_fill(
                    raw_morphology, detector_for_clip
                )
                methods[f"{method}_plus_raw_morphology_fill"] = raw_first_pan_fill(
                    detector_for_clip, raw_morphology
                )
            emrich = secondary_refined.get("emrich2023", np.empty(0, dtype=float))
            promac = secondary_refined.get("promac", np.empty(0, dtype=float))
            emrich_events = secondary_events.get("emrich2023", np.empty(0, dtype=float))
            promac_events = secondary_events.get("promac", np.empty(0, dtype=float))
            emrich_local = secondary_local.get("emrich2023", np.empty(0, dtype=float))
            promac_local = secondary_local.get("promac", np.empty(0, dtype=float))
            emrich_for_clip = emrich[(emrich >= left) & (emrich <= right)]
            promac_for_clip = promac[(promac >= left) & (promac <= right)]
            emrich_events_for_clip = emrich_events[(emrich_events >= left) & (emrich_events <= right)]
            promac_events_for_clip = promac_events[(promac_events >= left) & (promac_events <= right)]
            emrich_local_for_clip = emrich_local[(emrich_local >= left) & (emrich_local <= right)]
            promac_local_for_clip = promac_local[(promac_local >= left) & (promac_local <= right)]
            methods["emrich2023_local_qrs_extreme"] = emrich_local_for_clip
            methods["raw_morphology_plus_emrich2023_local_qrs_extreme_fill"] = raw_first_pan_fill(
                raw_morphology, emrich_local_for_clip
            )
            methods["raw_morphology_plus_emrich2023_local_qrs_extreme_fill_short_rr_pruned"], _ = raw_emrich_with_short_rr_prune(
                raw_morphology, emrich_local_for_clip
            )
            methods["emrich2023_local_qrs_extreme_plus_promac_local_qrs_extreme_fill"] = raw_first_pan_fill(
                emrich_local_for_clip, promac_local_for_clip
            )
            methods["promac_local_qrs_extreme_plus_emrich2023_local_qrs_extreme_fill"] = raw_first_pan_fill(
                promac_local_for_clip, emrich_local_for_clip
            )
            methods["emrich2023_promac_local_qrs_extreme_plus_v3_long_gap_rescue"] = emrich_promac_with_v3_long_gap_rescue(
                methods["emrich2023_local_qrs_extreme_plus_promac_local_qrs_extreme_fill"],
                methods["deployment_5s_raw_morphology_plus_pan_raw_apex_fill"],
            )
            methods["emrich2023_detector_event_plus_promac_detector_event_fill"] = raw_first_pan_fill(
                emrich_events_for_clip, promac_events_for_clip
            )
            methods["promac_detector_event_plus_emrich2023_detector_event_fill"] = raw_first_pan_fill(
                promac_events_for_clip, emrich_events_for_clip
            )
            methods["emrich2023_plus_promac_raw_apex_fill"] = raw_first_pan_fill(
                emrich_for_clip, promac_for_clip
            )
            methods["promac_plus_emrich2023_raw_apex_fill"] = raw_first_pan_fill(
                promac_for_clip, emrich_for_clip
            )
            for method, candidate in methods.items():
                candidate = np.sort(np.asarray(candidate, dtype=float))
                matched, missed, extra = one_to_one_match(
                    reference, candidate, tolerance_ms=float(args.event_tolerance_ms)
                )
                deltas = np.asarray([candidate[j] - reference[i] for i, j in matched], dtype=float)
                metric_rows.append({
                    "clip_id": clip_id,
                    "review_id": review_id,
                    "participant": participant,
                    "candidate_method": method,
                    "reference_peaks": len(reference),
                    "candidate_peaks": len(candidate),
                    "matched_peaks": len(matched),
                    "missed_peaks": len(missed),
                    "extra_peaks": len(extra),
                    "timing_median_error_ms": float(np.median(deltas)) if len(deltas) else float("nan"),
                    "timing_p95_abs_error_ms": float(np.quantile(np.abs(deltas), .95)) if len(deltas) else float("nan"),
                })
                for i, j in matched:
                    match_rows.append({
                        "clip_id": clip_id, "participant": participant, "candidate_method": method,
                        "reference_unix_ms": reference[i], "candidate_unix_ms": candidate[j],
                        "candidate_minus_reference_ms": candidate[j] - reference[i], "match_status": "matched",
                    })
                for i in missed:
                    match_rows.append({
                        "clip_id": clip_id, "participant": participant, "candidate_method": method,
                        "reference_unix_ms": reference[i], "candidate_unix_ms": np.nan,
                        "candidate_minus_reference_ms": np.nan, "match_status": "missed_reference",
                    })
                for j in extra:
                    match_rows.append({
                        "clip_id": clip_id, "participant": participant, "candidate_method": method,
                        "reference_unix_ms": np.nan, "candidate_unix_ms": candidate[j],
                        "candidate_minus_reference_ms": np.nan, "match_status": "extra_candidate",
                    })
            evaluated_clips.append(clip_id)

    metrics = pd.DataFrame(metric_rows)
    matches = pd.DataFrame(match_rows)
    summary = aggregate(metrics, ["candidate_method"])
    by_participant = aggregate(metrics, ["participant", "candidate_method"])
    metrics.to_csv(args.out_dir / "clip_metrics.csv", index=False)
    matches.to_csv(args.out_dir / "peak_matches.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    by_participant.to_csv(args.out_dir / "by_participant.csv", index=False)
    report = {
        "reference": args.reference_label,
        "scope": "five-second excerpts; peak detection and timing only, not five-minute RMSSD",
        "excluded_participants": sorted(EXCLUDED_PARTICIPANTS),
        "excluded_unresolvable_clips": sorted(unresolvable),
        "excluded_severe_artifact_clips": sorted(severe_artifact),
        "excluded_explicit_clips": sorted(explicit_exclusions),
        "evaluated_clips": len(evaluated_clips),
        "event_match_tolerance_ms": float(args.event_tolerance_ms),
        "detector_errors": {method: messages for method, messages in detector_errors.items() if messages},
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

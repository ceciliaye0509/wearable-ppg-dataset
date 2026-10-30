"""Build a small, targeted assisted ECG-v3 review packet.

This intentionally does *not* turn every QC/review flag into manual work.
It selects a diverse set of windows with detector/IBI anomalies, retains a
five-minute overview image for each, and exposes only 20-second clips around
the anomalous IBI locations for interactive correction.  Severe-artifact and
minor-correction-only windows remain in the machine-readable review queue but
are not added to this human review packet.
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
import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _has_severe_amplitude_artifact, _write_annotation_app
from rebuild_ecg_ground_truth_raw_morphology import _correct_rr


WINDOW_MS = 300_000.0
CLIP_MS = 20_000.0
IBI_MIN_MS = 300.0
IBI_MAX_MS = 2_000.0
MANUAL_REASONS = {
    "low_ecg_valid_ibi_ratio",
    "high_ecg_ibi_correction_ratio",
    "too_few_ecg_peaks",
    "invalid_ecg_hrv",
    "emrich2023_failed:IndexError",
    "ecg_rmssd_too_high",
}
AUTO_EXCLUDE_REASONS = {"severe_raw_ecg_artifact_chunk", "low_ecg_sample_ratio"}


def _reasons(value: object) -> set[str]:
    return {item for item in str(value).split("|") if item}


def _priority(reasons: set[str]) -> int:
    return (
        200 * int("emrich2023_failed:IndexError" in reasons)
        + 150 * int("low_ecg_valid_ibi_ratio" in reasons)
        + 120 * int("too_few_ecg_peaks" in reasons)
        + 80 * int("high_ecg_ibi_correction_ratio" in reasons)
        + 50 * int("invalid_ecg_hrv" in reasons)
        + 25 * int("ecg_rmssd_too_high" in reasons)
    )


def _select_windows(queue: pd.DataFrame, limit: int) -> pd.DataFrame:
    queue = queue.copy()
    queue["_reasons"] = queue.qc_reason.map(_reasons)
    queue = queue[
        queue._reasons.map(lambda values: bool(values & MANUAL_REASONS) and not bool(values & AUTO_EXCLUDE_REASONS))
    ].copy()
    if queue.empty:
        raise ValueError("no non-artifact detector/RR anomalies are available for targeted review")
    queue["priority"] = queue._reasons.map(_priority)
    queue = queue.sort_values(["priority", "participant", "ppg_window_t0_ms"], ascending=[False, True, True])

    selected: list[int] = []
    selected_by_participant: dict[str, list[float]] = {}
    # Round-robin across participants first, then fill remaining slots. Do not
    # make a reviewer correct two heavily overlapping 5-min windows.
    while len(selected) < limit:
        made_progress = False
        for participant, group in queue.groupby("participant", sort=True):
            if len(selected) >= limit:
                break
            previous = selected_by_participant.setdefault(str(participant), [])
            if len(previous) >= 2:
                continue
            for row in group.itertuples():
                start = float(row.ppg_window_t0_ms)
                if row.Index in selected or any(abs(start - old) < WINDOW_MS for old in previous):
                    continue
                selected.append(int(row.Index))
                previous.append(start)
                made_progress = True
                break
        if not made_progress:
            break
    return queue.loc[selected].sort_values(["priority", "participant"], ascending=[False, True]).reset_index(drop=True)


def _candidate_event_times(peaks_abs: np.ndarray) -> np.ndarray:
    """Return peak times adjacent to invalid or corrected IBIs."""
    peaks = np.asarray(peaks_abs, dtype=float)
    rr_all = np.diff(peaks)
    if not rr_all.size:
        return np.empty(0, dtype=float)
    valid = (rr_all >= IBI_MIN_MS) & (rr_all <= IBI_MAX_MS)
    events = list(peaks[1:][~valid])
    valid_indices = np.flatnonzero(valid)
    corrected, _ = _correct_rr(rr_all[valid])
    changed = np.abs(corrected - rr_all[valid]) > 1e-3
    events.extend(peaks[1:][valid_indices[changed]])
    return np.unique(np.asarray(events, dtype=float))


def _clip_centers(events: np.ndarray, start: float, end: float) -> np.ndarray:
    values = [float(value) for value in events if start <= value <= end]
    if not values:
        return np.asarray([(start + end) / 2.0])
    kept: list[float] = []
    for value in values:
        if not kept or value - kept[-1] >= CLIP_MS:
            kept.append(value)
        if len(kept) == 2:
            break
    return np.asarray(kept, dtype=float)


def _closest_indices(times: np.ndarray, values: np.ndarray) -> list[int]:
    result: list[int] = []
    for value in values:
        right = int(np.searchsorted(times, value, side="left"))
        choices = [index for index in (right - 1, right) if 0 <= index < len(times)]
        if choices:
            result.append(min(choices, key=lambda index: abs(times[index] - value)))
    return sorted(set(result))


def _write_overview(
    path: Path, times: np.ndarray, signal: np.ndarray, start: float, end: float,
    peaks_abs: np.ndarray, centers: np.ndarray, title: str,
) -> None:
    lo, hi = np.searchsorted(times, start, side="left"), np.searchsorted(times, end, side="right")
    t, x = times[lo:hi], signal[lo:hi]
    stride = max(1, len(t) // 10_000)
    fig, axis = plt.subplots(figsize=(15, 3.8), constrained_layout=True)
    axis.plot((t[::stride] - start) / 1000.0, x[::stride], color="#273746", linewidth=0.35)
    in_window = peaks_abs[(peaks_abs >= start) & (peaks_abs <= end)]
    marker_idx = _closest_indices(t, in_window)
    if marker_idx:
        axis.scatter((t[marker_idx] - start) / 1000.0, x[marker_idx], s=7, color="#c0392b", zorder=3)
    for center in centers:
        left, right = max(start, center - CLIP_MS / 2), min(end, center + CLIP_MS / 2)
        axis.axvspan((left - start) / 1000.0, (right - start) / 1000.0, color="#f5b041", alpha=0.25)
    axis.set(title=title, xlabel="seconds in five-minute window", ylabel="ECG (µV)", xlim=(0, (end - start) / 1000.0))
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--sidecar-prefix", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("Multisite-PPG/raw_data"))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("wearable-ppg-dataset/src/heuristic_baselines/outputs/ecg_v3_targeted_assisted_review_v1"),
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    selected = _select_windows(pd.read_csv(args.queue), args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overview_dir = args.out_dir / "five_minute_overviews"
    overview_dir.mkdir(exist_ok=True)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sidecar_cache: dict[str, dict[str, np.ndarray]] = {}
    manifest: list[dict[str, object]] = []
    clips: list[dict[str, object]] = []

    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        participant = str(row.participant)
        start, end = float(row.ppg_window_t0_ms), float(row.ppg_window_t1_ms)
        if participant not in raw_cache:
            with np.load(args.raw_root / participant / f"{participant}_polar_ecg_raw.npz", allow_pickle=True) as source:
                raw_cache[participant] = (np.asarray(source["ecg_uv"], dtype=float), np.asarray(source["time_ms"], dtype=float))
        if participant not in sidecar_cache:
            with np.load(args.sidecar_dir / f"{args.sidecar_prefix}_{participant}.npz", allow_pickle=True) as source:
                sidecar_cache[participant] = {
                    "starts": np.asarray(source["ppg_window_t0_ms"], dtype=float),
                    "ends": np.asarray(source["ppg_window_t1_ms"], dtype=float),
                    "peaks": np.asarray(source["ecg_r_peak_times_rel_ms"], dtype=object),
                }
        signal, times = raw_cache[participant]
        sidecar = sidecar_cache[participant]
        anchor = int(row.source_window_index)
        peaks_abs = sidecar["starts"][anchor] + np.asarray(sidecar["peaks"][anchor], dtype=float)
        centers = _clip_centers(_candidate_event_times(peaks_abs), start, end)
        review_id = f"T{rank:02d}_{participant}_W{anchor:05d}"
        overview_path = overview_dir / f"{review_id}_overview.png"
        _write_overview(
            overview_path, times, signal, start, end, peaks_abs, centers,
            f"{review_id} | {participant} | {row.qc_reason}",
        )
        manifest.append({
            "review_id": review_id, "review_rank": rank, "participant": participant,
            "window_start_unix_ms": start, "window_end_unix_ms": end,
            "qc_reasons": row.qc_reason, "priority": int(row.priority),
            "source_window_index": anchor, "overview_png": str(overview_path.resolve()),
            "interactive_clip_count": len(centers),
        })
        for clip_number, center in enumerate(centers, start=1):
            clip_start, clip_end = max(start, center - CLIP_MS / 2), min(end, center + CLIP_MS / 2)
            lo, hi = np.searchsorted(times, clip_start, side="left"), np.searchsorted(times, clip_end, side="right")
            clip_t, clip_x = times[lo:hi], signal[lo:hi]
            severe = _has_severe_amplitude_artifact(clip_x)
            candidate_times = peaks_abs[(peaks_abs >= clip_start) & (peaks_abs <= clip_end)]
            clips.append({
                "id": f"{review_id}_C{clip_number}", "reviewId": review_id, "participant": participant,
                "windowStart": start, "startSec": (clip_start - start) / 1000.0, "endSec": (clip_end - start) / 1000.0,
                "t": (clip_t - start).round(3).tolist(), "x": clip_x.round(2).tolist(),
                "aidStatus": "suppressed_severe_artifact" if severe else "shown", "autoSkip": bool(severe),
                "suggested": [] if severe else _closest_indices(clip_t, candidate_times),
            })

    pd.DataFrame(manifest).to_csv(args.out_dir / "review_manifest.csv", index=False)
    selected.drop(columns="_reasons").to_csv(args.out_dir / "selected_source_windows.csv", index=False)
    pd.DataFrame(columns=["reviewer_id", "review_id", "participant", "window_start_unix_ms", "manual_rpeak_abs_unix_ms", "confidence", "review_status", "notes"]).to_csv(args.out_dir / "rpeak_annotations_template.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "# Targeted ECG-v3 assisted review\n\n"
        "This packet contains a small, diverse subset of detector/RR anomalies. Each review unit has a five-minute overview PNG with yellow interactive-clip regions. Only those 20-second clips appear in the annotation app. Severe-artifact and minor-correction-only windows remain in the QC queue but are not manual work. This is assisted correction, not blinded validation.\n",
        encoding="utf-8",
    )
    (args.out_dir / "review_protocol.json").write_text(json.dumps({
        "version": "ecg_v3_targeted_assisted_review_v1", "review_units": len(manifest), "interactive_clips": len(clips),
        "selection": "non-artifact detector_or_rr_anomaly; participant-balanced; non-overlapping five-minute windows",
        "auto_excluded": sorted(AUTO_EXCLUDE_REASONS), "not_manually_expanded": ["ibi_correction_applied"],
    }, indent=2) + "\n", encoding="utf-8")
    _write_annotation_app(args.out_dir, manifest, clips, assisted=True, aid_method="emrich2023", storage_key="ecg-v3-targeted-assisted-review-v1")
    print(f"Wrote {len(manifest)} review units, {len(clips)} interactive clips, and {len(manifest)} five-minute overviews to {args.out_dir}")


if __name__ == "__main__":
    main()

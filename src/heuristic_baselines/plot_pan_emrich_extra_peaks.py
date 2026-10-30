"""Visualize all extra hybrid candidates shared by Pan and Emrich fallbacks."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _independent_raw_r_candidates
from evaluate_assisted_human_confirmed_rpeaks import (
    detector_indices,
    infer_fs,
    pan_tompkins_indices,
    raw_first_pan_fill,
    refine_to_dominant_raw_extreme,
)


ROOT = Path("wearable-ppg-dataset/src/heuristic_baselines")
RAW_ROOT = Path("Multisite-PPG/raw_data")
MANIFEST = ROOT / "outputs/assisted_ecg_rpeak_final_validation_v4/review_manifest.csv"
ANNOTATIONS = Path("/Users/jiqiyu/Downloads/rpeak_annotations_1 (1).csv")
MATCHES = ROOT / "outputs/assisted_human_confirmed_validation_v3_new_csv/peak_matches.csv"
OUT_DIR = ROOT / "outputs/pan_emrich_extra_peak_audit_v1"
CLIP_RANGES = ((70.0, 75.0), (75.0, 80.0), (220.0, 225.0), (225.0, 230.0))


def clip_bounds(clip_id: str, manifest: pd.DataFrame) -> tuple[float, float, float, str]:
    review_id = clip_id.rsplit("_C", 1)[0]
    number = int(clip_id.rsplit("_C", 1)[1])
    row = manifest.loc[manifest.review_id.eq(review_id)].iloc[0]
    window_start = float(row.window_start_unix_ms)
    left_s, right_s = CLIP_RANGES[number - 1]
    return window_start + left_s * 1000.0, window_start + right_s * 1000.0, window_start, str(row.participant)


def nearest_origin(peak: float, raw: np.ndarray, pan_fallback: np.ndarray, emrich_fallback: np.ndarray) -> str:
    if len(raw) and np.min(np.abs(raw - peak)) < 0.5:
        return "raw-morphology"
    if len(pan_fallback) and np.min(np.abs(pan_fallback - peak)) < 0.5:
        return "Pan fallback"
    if len(emrich_fallback) and np.min(np.abs(emrich_fallback - peak)) < 0.5:
        return "Emrich fallback"
    return "unresolved"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype={"review_id": str, "participant": str})
    annotations = pd.read_csv(ANNOTATIONS, dtype={"review_id": str})
    annotations["manual_rpeak_abs_unix_ms"] = pd.to_numeric(annotations["manual_rpeak_abs_unix_ms"], errors="coerce")
    annotations = annotations.dropna(subset=["manual_rpeak_abs_unix_ms"])
    matches = pd.read_csv(MATCHES)
    extra = matches[(matches.match_status.eq("extra_candidate")) & matches.candidate_method.eq("raw_morphology_plus_pan_raw_apex_fill")]
    figure, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    figure.suptitle("Pan and Emrich hybrid extra candidates: all six cases\nBlack rings: human reference; red ×: extra candidate", fontsize=15, fontweight="bold")
    audit_rows: list[dict[str, object]] = []

    for axis, row in zip(axes.flat, extra.itertuples(index=False), strict=True):
        clip_id = str(row.clip_id)
        left, right, window_start, participant = clip_bounds(clip_id, manifest)
        review_id = clip_id.rsplit("_C", 1)[0]
        with np.load(RAW_ROOT / participant / f"{participant}_polar_ecg_raw.npz", allow_pickle=True) as source:
            raw_signal, times = np.asarray(source["ecg_uv"], dtype=float), np.asarray(source["time_ms"], dtype=float)
        valid = np.isfinite(raw_signal) & np.isfinite(times)
        raw_signal, times = raw_signal[valid], times[valid]
        full_lo, full_hi = np.searchsorted(times, window_start, side="left"), np.searchsorted(times, window_start + 300_000.0, side="right")
        signal, window_times = raw_signal[full_lo:full_hi], times[full_lo:full_hi]
        fs = infer_fs(times)
        pan_idx = pan_tompkins_indices(signal, fs)
        pan = refine_to_dominant_raw_extreme(signal, window_times, pan_idx, fs)
        emrich_idx = detector_indices(signal, fs, "emrich2023")
        emrich = refine_to_dominant_raw_extreme(signal, window_times, emrich_idx, fs)
        lo, hi = np.searchsorted(times, left, side="left"), np.searchsorted(times, right, side="right")
        x, t = raw_signal[lo:hi], times[lo:hi]
        raw_candidates = t[_independent_raw_r_candidates(x, t)]
        pan_clip = pan[(pan >= left) & (pan <= right)]
        emrich_clip = emrich[(emrich >= left) & (emrich <= right)]
        pan_fallback = np.asarray([peak for peak in pan_clip if not len(raw_candidates) or np.min(np.abs(raw_candidates - peak)) > 150.0])
        emrich_fallback = np.asarray([peak for peak in emrich_clip if not len(raw_candidates) or np.min(np.abs(raw_candidates - peak)) > 150.0])
        pan_hybrid = raw_first_pan_fill(raw_candidates, pan_clip)
        emrich_hybrid = raw_first_pan_fill(raw_candidates, emrich_clip)
        reference = np.sort(annotations.loc[
            annotations.review_id.eq(review_id) & annotations.manual_rpeak_abs_unix_ms.between(left, right),
            "manual_rpeak_abs_unix_ms",
        ].to_numpy(float))
        extra_peak = float(row.candidate_unix_ms)
        origin = nearest_origin(extra_peak, raw_candidates, pan_fallback, emrich_fallback)

        axis.plot((t - window_start) / 1000.0, x, color="#334155", linewidth=1.0)
        for values, color, marker, label in (
            (reference, "black", "o", "human reference"),
            (raw_candidates, "#1677ff", "o", "raw morphology"),
            (pan_fallback, "#e87321", "s", "Pan fallback"),
            (emrich_fallback, "#18a957", "^", "Emrich fallback"),
        ):
            if not len(values):
                continue
            y = np.interp(values, t, x)
            if label == "human reference":
                axis.scatter((values - window_start) / 1000.0, y, s=68, facecolors="none", edgecolors=color, linewidths=1.2, zorder=5, label=label)
            else:
                axis.scatter((values - window_start) / 1000.0, y, s=38, color=color, marker=marker, zorder=4, label=label)
        extra_y = float(np.interp(extra_peak, t, x))
        axis.scatter((extra_peak - window_start) / 1000.0, extra_y, s=140, color="#dc2626", marker="x", linewidths=2.2, zorder=6, label="extra candidate")
        same_hybrid = len(pan_hybrid) == len(emrich_hybrid) and np.allclose(pan_hybrid, emrich_hybrid, atol=0.5)
        axis.set(title=f"{clip_id} · origin={origin} · Pan/Emrich hybrid identical={same_hybrid}", ylabel="ECG (µV)")
        axis.grid(alpha=0.25)
        audit_rows.append({"clip_id": clip_id, "participant": participant, "extra_unix_ms": f"{extra_peak:.3f}", "origin": origin, "pan_emrich_hybrid_identical": same_hybrid})

    axes[-1, 0].set_xlabel("seconds within five-minute window")
    axes[-1, 1].set_xlabel("seconds within five-minute window")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.985))
    figure.savefig(OUT_DIR / "all_pan_emrich_extra_candidates.png", dpi=180)
    with (OUT_DIR / "extra_candidate_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)
    print(OUT_DIR)


if __name__ == "__main__":
    main()

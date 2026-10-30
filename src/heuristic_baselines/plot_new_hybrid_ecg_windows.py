"""Create raw-ECG peak-location inspection plots for unseen source windows."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

from build_blinded_ecg_rpeak_review import _independent_raw_r_candidates
from evaluate_assisted_human_confirmed_rpeaks import refine_to_dominant_raw_extreme


ROOT = Path("wearable-ppg-dataset/src/heuristic_baselines")
RAW_ROOT = Path("Multisite-PPG/raw_data")
WINDOWS_DIR = ROOT / "outputs/synced_3device_rawaligned_training_v1_stride30_rawslots"
VALIDATION_MANIFEST = ROOT / "outputs/assisted_ecg_rpeak_final_validation_v4/review_manifest.csv"
OUT_DIR = ROOT / "outputs/new_hybrid_ecg_window_plots_v1"
PARTICIPANTS = ("P1", "P3", "P4", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15")
CLIP_STARTS_S = (55.0, 125.0, 195.0, 265.0)
CLIP_SECONDS = 5.0
FALLBACK_GAP_MS = 150.0


def infer_fs(times_ms: np.ndarray) -> float:
    delta = np.diff(times_ms)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    return float(1000.0 / np.median(delta))


def pan_refined(signal: np.ndarray, times_ms: np.ndarray, fs: float) -> np.ndarray:
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    indices = indices[(indices >= 0) & (indices < len(signal))]
    return refine_to_dominant_raw_extreme(signal, times_ms, indices, fs)


def choose_unseen_window(starts: np.ndarray, previous_starts: np.ndarray) -> tuple[int, float]:
    viable = [
        index for index, start in enumerate(starts)
        if not np.any(np.abs(previous_starts - start) < 300_000.0)
    ]
    if not viable:
        raise RuntimeError("no non-overlapping five-minute window available")
    # Deterministic middle-third selection, avoiding a cherry-picked easy case.
    index = viable[len(viable) // 2]
    return index, float(starts[index])


def fallback_peaks(raw_candidates: np.ndarray, refined_pan: np.ndarray) -> np.ndarray:
    return np.asarray(
        [peak for peak in refined_pan if not len(raw_candidates) or np.min(np.abs(raw_candidates - peak)) > FALLBACK_GAP_MS],
        dtype=float,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior_by_participant: dict[str, list[float]] = {}
    with VALIDATION_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prior_by_participant.setdefault(row["participant"], []).append(float(row["window_start_unix_ms"]))

    selections: list[dict[str, object]] = []
    for participant in PARTICIPANTS:
        windows_path = WINDOWS_DIR / f"synced_3device_rawaligned_training_v1_stride30_rawslots_{participant}.npz"
        raw_path = RAW_ROOT / participant / f"{participant}_polar_ecg_raw.npz"
        with np.load(windows_path, allow_pickle=True) as windows:
            starts = np.asarray(windows["ppg_window_t0_ms"], dtype=float)
        window_index, start = choose_unseen_window(starts, np.asarray(prior_by_participant.get(participant, []), dtype=float))
        end = start + 300_000.0
        with np.load(raw_path, allow_pickle=True) as raw_source:
            raw = np.asarray(raw_source["ecg_uv"], dtype=float)
            times = np.asarray(raw_source["time_ms"], dtype=float)
        valid = np.isfinite(raw) & np.isfinite(times)
        raw, times = raw[valid], times[valid]
        lo, hi = np.searchsorted(times, start, side="left"), np.searchsorted(times, end, side="right")
        signal, window_times = raw[lo:hi], times[lo:hi]
        fs = infer_fs(window_times)
        refined_pan = pan_refined(signal, window_times, fs)

        figure, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
        figure.suptitle(
            f"{participant} · unseen 5-min window index {window_index} · hybrid QRS timing audit\n"
            "Blue: raw-morphology proposal; orange: Pan–Tompkins→raw-apex fallback",
            fontsize=14,
            fontweight="bold",
        )
        for axis, clip_start_s in zip(axes, CLIP_STARTS_S, strict=True):
            left, right = start + clip_start_s * 1000.0, start + (clip_start_s + CLIP_SECONDS) * 1000.0
            clip_lo, clip_hi = np.searchsorted(times, left, side="left"), np.searchsorted(times, right, side="right")
            clip_signal, clip_times = raw[clip_lo:clip_hi], times[clip_lo:clip_hi]
            candidates = clip_times[_independent_raw_r_candidates(clip_signal, clip_times)]
            pan_for_clip = refined_pan[(refined_pan >= left) & (refined_pan <= right)]
            fallback = fallback_peaks(candidates, pan_for_clip)
            axis.plot((clip_times - start) / 1000.0, clip_signal, color="#25364a", linewidth=1.0, label="raw ECG")
            if len(candidates):
                candidate_y = np.interp(candidates, clip_times, clip_signal)
                axis.scatter((candidates - start) / 1000.0, candidate_y, s=42, color="#1677ff", zorder=3, label="raw morphology")
            if len(fallback):
                fallback_y = np.interp(fallback, clip_times, clip_signal)
                axis.scatter((fallback - start) / 1000.0, fallback_y, s=58, marker="s", color="#e87321", zorder=4, label="Pan fallback")
            axis.set(xlim=(clip_start_s, clip_start_s + CLIP_SECONDS), ylabel="ECG (µV)")
            axis.grid(alpha=0.25)
            axis.set_title(f"{clip_start_s:.0f}–{clip_start_s + CLIP_SECONDS:.0f} s  ·  raw={len(candidates)}  fallback={len(fallback)}", loc="left", fontsize=10)
        axes[-1].set_xlabel("seconds within 5-minute window")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.985))
        plot_path = OUT_DIR / f"{participant}_window_{window_index:04d}_hybrid_audit.png"
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
        selections.append({
            "participant": participant,
            "source_window_index": window_index,
            "window_start_unix_ms": int(start),
            "plot": plot_path.name,
        })

    with (OUT_DIR / "selected_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selections[0].keys())
        writer.writeheader()
        writer.writerows(selections)
    (OUT_DIR / "README.md").write_text(
        "# Unseen hybrid ECG audit plots\n\n"
        "Twelve five-minute source windows were selected deterministically, excluding all windows used by the assisted-validation packet. "
        "Each image shows four 5-second raw-ECG excerpts. Blue marks are raw-morphology proposals. Orange squares are only the Pan–Tompkins detections that were needed to fill a raw-morphology gap, after snapping to the local raw QRS extreme. "
        "No human reference labels are overlaid because these are fresh inspection windows.\n",
        encoding="utf-8",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()

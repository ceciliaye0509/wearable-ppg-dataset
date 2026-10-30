"""Plot representative detector successes and failures against human-confirmed peaks."""
from __future__ import annotations

import csv
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_blinded_ecg_rpeak_review import _independent_raw_r_candidates
from evaluate_assisted_human_confirmed_rpeaks import (
    detector_indices,
    infer_fs,
    one_to_one_match,
    pan_tompkins_indices,
    raw_first_pan_fill,
    refine_to_dominant_raw_extreme,
)


ROOT = Path("wearable-ppg-dataset/src/heuristic_baselines")
RAW_ROOT = Path("Multisite-PPG/raw_data")
MANIFEST = ROOT / "outputs/assisted_ecg_rpeak_final_validation_v4/review_manifest.csv"
ANNOTATIONS = Path("/Users/jiqiyu/Downloads/rpeak_annotations_1 (1).csv")
OUT_DIR = ROOT / "outputs/detector_comparison_typical_cases_v2_new_reference"
CLIP_RANGES = ((70.0, 75.0), (75.0, 80.0), (220.0, 225.0), (225.0, 230.0))
CASES = (
    ("P7_W01_C2", "clean agreement"),
    ("P6_W02_C2", "raw morphology failure; all fallbacks rescue"),
    ("P1_W01_C4", "Pan miss; raw morphology rescues"),
    ("P4_W03_C4", "Pan / Emrich / ProMAC disagreement"),
    ("P4_W02_C1", "extra-candidate comparison"),
)


def clip_bounds(review_id: str, clip_id: str, manifest: pd.DataFrame) -> tuple[float, float, str]:
    row = manifest.loc[manifest.review_id.eq(review_id)].iloc[0]
    number = int(clip_id.rsplit("_C", 1)[1])
    left, right = CLIP_RANGES[number - 1]
    start = float(row.window_start_unix_ms)
    return start + 1000.0 * left, start + 1000.0 * right, str(row.participant)


def markers(ax: plt.Axes, candidates: np.ndarray, times: np.ndarray, signal: np.ndarray, start: float, *, color: str, marker: str, label: str) -> None:
    if not len(candidates):
        ax.text(0.99, 0.84, "no candidates", transform=ax.transAxes, ha="right", va="top", color=color, fontsize=9)
        return
    y = np.interp(candidates, times, signal)
    ax.scatter((candidates - start) / 1000.0, y, s=40, marker=marker, color=color, zorder=3, label=label)


def match_label(reference: np.ndarray, candidates: np.ndarray) -> str:
    matched, missed, extra = one_to_one_match(reference, candidates)
    return f"candidate={len(candidates)} · match={len(matched)} · miss={len(missed)} · extra={len(extra)}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype={"review_id": str, "participant": str})
    annotations = pd.read_csv(ANNOTATIONS, dtype={"review_id": str})
    annotations["manual_rpeak_abs_unix_ms"] = pd.to_numeric(annotations["manual_rpeak_abs_unix_ms"], errors="coerce")
    annotations = annotations.dropna(subset=["manual_rpeak_abs_unix_ms"])
    index_rows: list[dict[str, str]] = []

    for clip_id, scenario in CASES:
        review_id = clip_id.rsplit("_C", 1)[0]
        left, right, participant = clip_bounds(review_id, clip_id, manifest)
        with np.load(RAW_ROOT / participant / f"{participant}_polar_ecg_raw.npz", allow_pickle=True) as source:
            raw, times = np.asarray(source["ecg_uv"], dtype=float), np.asarray(source["time_ms"], dtype=float)
        valid = np.isfinite(raw) & np.isfinite(times)
        raw, times = raw[valid], times[valid]
        window_start = float(manifest.loc[manifest.review_id.eq(review_id), "window_start_unix_ms"].iloc[0])
        window_end = window_start + 300_000.0
        full_lo, full_hi = np.searchsorted(times, window_start, side="left"), np.searchsorted(times, window_end, side="right")
        full_signal, full_times = raw[full_lo:full_hi], times[full_lo:full_hi]
        # Match the evaluator exactly: sampling frequency is inferred from the
        # full raw recording, not a possibly irregular five-minute subsection.
        fs = infer_fs(times)
        pan_indices = pan_tompkins_indices(full_signal, fs)
        pan = refine_to_dominant_raw_extreme(full_signal, full_times, pan_indices, fs)
        detector_peaks: dict[str, np.ndarray] = {"Pan–Tompkins → raw apex": pan}
        for method, label in (("emrich2023", "Emrich2023 → raw apex"), ("promac", "ProMAC → raw apex"), ("khamis2016", "Khamis2016 → raw apex")):
            with redirect_stdout(io.StringIO()):
                try:
                    indices = detector_indices(full_signal, fs, method)
                except Exception:
                    indices = np.empty(0, dtype=int)
            detector_peaks[label] = refine_to_dominant_raw_extreme(full_signal, full_times, indices, fs)
        clip_lo, clip_hi = np.searchsorted(times, left, side="left"), np.searchsorted(times, right, side="right")
        clip_signal, clip_times = raw[clip_lo:clip_hi], times[clip_lo:clip_hi]
        raw_morphology = clip_times[_independent_raw_r_candidates(clip_signal, clip_times)]
        reference = np.sort(annotations.loc[
            annotations.review_id.eq(review_id) & annotations.manual_rpeak_abs_unix_ms.between(left, right),
            "manual_rpeak_abs_unix_ms",
        ].to_numpy(float))
        per_clip = {name: values[(values >= left) & (values <= right)] for name, values in detector_peaks.items()}
        pan_hybrid = raw_first_pan_fill(raw_morphology, per_clip["Pan–Tompkins → raw apex"])
        emrich_hybrid = raw_first_pan_fill(raw_morphology, per_clip["Emrich2023 → raw apex"])
        promac_hybrid = raw_first_pan_fill(raw_morphology, per_clip["ProMAC → raw apex"])

        rows = [
            ("raw-morphology", raw_morphology, "#1677ff", "o"),
            ("hybrid: raw morphology + Pan raw-apex fallback", pan_hybrid, "#e87321", "s"),
            ("hybrid: raw morphology + Emrich2023 raw-apex fallback", emrich_hybrid, "#18a957", "^"),
            ("hybrid: raw morphology + ProMAC raw-apex fallback", promac_hybrid, "#8b5cf6", "D"),
        ]
        figure, axes = plt.subplots(len(rows), 1, figsize=(14, 14), sharex=True, constrained_layout=True)
        figure.suptitle(f"{clip_id} · {participant} · {scenario}\nBlack rings: human-confirmed QRS timing", fontsize=15, fontweight="bold")
        for axis, (name, candidate, color, marker) in zip(axes, rows, strict=True):
            axis.plot((clip_times - window_start) / 1000.0, clip_signal, color="#334155", linewidth=1.0)
            if len(reference):
                ref_y = np.interp(reference, clip_times, clip_signal)
                axis.scatter((reference - window_start) / 1000.0, ref_y, s=72, facecolors="none", edgecolors="black", linewidths=1.2, zorder=4, label="human reference")
            markers(axis, candidate, clip_times, clip_signal, window_start, color=color, marker=marker, label=name)
            axis.set(ylabel="µV", title=f"{name}  ·  {match_label(reference, candidate)}")
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("seconds within the five-minute window")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.985))
        filename = f"{clip_id}_detector_comparison.png"
        figure.savefig(OUT_DIR / filename, dpi=180)
        plt.close(figure)
        index_rows.append({"clip_id": clip_id, "scenario": scenario, "plot": filename})

    with (OUT_DIR / "case_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "scenario", "plot"))
        writer.writeheader()
        writer.writerows(index_rows)
    print(OUT_DIR)


if __name__ == "__main__":
    main()

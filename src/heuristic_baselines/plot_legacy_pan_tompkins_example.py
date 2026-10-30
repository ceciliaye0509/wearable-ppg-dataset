"""Plot the unmodified legacy Pan--Tompkins output on one raw-ECG excerpt.

This intentionally does *not* snap events to raw extrema or apply the v3
raw-morphology rules.  It is a visual record of the positions used by the
original ECG label generator.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


ROOT = Path("wearable-ppg-dataset/src/heuristic_baselines")
MANIFEST = ROOT / "outputs/assisted_ecg_rpeak_review_v1/review_manifest.csv"
RAW_PATH = Path("Multisite-PPG/raw_data/P13/P13_polar_ecg_raw.npz")
OUT_PATH = ROOT / "outputs/legacy_pan_tompkins_example_p13_w03_c1.png"
REVIEW_ID = "P13_W03"
LEFT_SEC, RIGHT_SEC = 65.0, 85.0


def infer_fs(times_ms: np.ndarray) -> float:
    dt = np.diff(times_ms)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1000.0 / np.median(dt))


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype={"review_id": str})
    row = manifest.loc[manifest.review_id.eq(REVIEW_ID)].iloc[0]
    window_start = float(row.window_start_unix_ms)
    with np.load(RAW_PATH, allow_pickle=True) as source:
        raw = np.asarray(source["ecg_uv"], dtype=float)
        absolute_time = np.asarray(source["time_ms"], dtype=float)
    keep = np.isfinite(raw) & np.isfinite(absolute_time)
    raw, absolute_time = raw[keep], absolute_time[keep]

    # This is the old generator exactly: clean + Pan--Tompkins peak event.
    whole_window = (absolute_time >= window_start) & (absolute_time < window_start + 300_000.0)
    signal, times = raw[whole_window], absolute_time[whole_window]
    fs = infer_fs(times)
    clean = nk.ecg_clean(signal, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
    pan_indices = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    pan_indices = pan_indices[(pan_indices >= 0) & (pan_indices < len(signal))]

    rel_sec = (times - window_start) / 1000.0
    excerpt = (rel_sec >= LEFT_SEC) & (rel_sec <= RIGHT_SEC)
    peaks_in_excerpt = pan_indices[(rel_sec[pan_indices] >= LEFT_SEC) & (rel_sec[pan_indices] <= RIGHT_SEC)]

    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.plot(rel_sec[excerpt], signal[excerpt], color="#334155", linewidth=0.9, label="raw ECG")
    axis.vlines(
        rel_sec[peaks_in_excerpt],
        np.nanpercentile(signal[excerpt], 2),
        np.nanpercentile(signal[excerpt], 98),
        colors="#dc2626",
        alpha=0.35,
        linewidth=0.8,
    )
    axis.scatter(
        rel_sec[peaks_in_excerpt],
        signal[peaks_in_excerpt],
        color="#dc2626",
        marker="v",
        s=52,
        zorder=4,
        label="legacy Pan–Tompkins output (unrefined)",
    )
    axis.set(
        title=(
            "Legacy Pan–Tompkins peak positions on raw ECG — P13_W03, 65–85 s\n"
            "No raw-extremum snapping, morphology rules, or IBI correction"
        ),
        xlabel="seconds within the five-minute window",
        ylabel="ECG (µV)",
        xlim=(LEFT_SEC, RIGHT_SEC),
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="upper right")
    fig.savefig(OUT_PATH, dpi=200)
    print(f"fs={fs:.2f} Hz; all-window Pan events={len(pan_indices)}; excerpt events={len(peaks_in_excerpt)}")
    print(OUT_PATH)


if __name__ == "__main__":
    main()

"""Plot high-RMSSD-error 5-minute windows for manual spot checks."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/daily_hrv_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from algorithms.hrv import _correct_ibi_artifacts_with_ratio  # noqa: E402
from evaluate_rawslots_detector_fiducials import (  # noqa: E402
    DATASET_DIR,
    DetectorPipeline,
    _choose_polarity_detector,
    _strict_interp100_channel,
)


DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
DEFAULT_METRICS_CSV = (
    _HERE
    / "outputs"
    / "rawslots_qppgfast_devicewise_correction_validibi_full16_v1"
    / "window_metrics.csv"
)
DEFAULT_OUT_DIR = _HERE / "outputs" / "spot_check_high_rmssd_windows"


def _finite_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _select_high_error_rows(metrics_csv: Path, top_n: int, valid_ibi_gate: float) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    selected = df[
        df["detector"].eq("qppgfast")
        & df["ibi_correction"].astype(str).str.lower().eq("true")
        & df["device"].isin(DEVICE_FIDUCIALS)
    ].copy()
    selected = selected[selected["fiducial"].eq(selected["device"].map(DEVICE_FIDUCIALS))].copy()

    for col in ("ppg_rmssd_ms", "ecg_rmssd_ms", "ppg_sdnn_ms", "ecg_sdnn_ms", "ppg_valid_ibi_ratio"):
        selected[col] = _finite_float(selected[col])
    selected = selected[
        np.isfinite(selected["ppg_rmssd_ms"])
        & np.isfinite(selected["ecg_rmssd_ms"])
        & (selected["ppg_valid_ibi_ratio"] >= valid_ibi_gate)
    ].copy()
    selected["rmssd_abs_error_ms"] = (selected["ppg_rmssd_ms"] - selected["ecg_rmssd_ms"]).abs()
    selected["rmssd_signed_error_ms"] = selected["ppg_rmssd_ms"] - selected["ecg_rmssd_ms"]
    return selected.sort_values("rmssd_abs_error_ms", ascending=False).head(top_n).reset_index(drop=True)


def _participant_npz(dataset_dir: Path, participant: str) -> Path:
    path = dataset_dir / f"synced_3device_rawaligned_training_v1_stride30_rawslots_{participant}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _ibi_from_times_ms(times_ms: np.ndarray, do_correction: bool) -> tuple[np.ndarray, np.ndarray, float]:
    if times_ms.size < 2:
        return np.empty(0), np.empty(0), 0.0
    ibi = np.diff(times_ms)
    valid = ibi[(ibi >= 300.0) & (ibi <= 2000.0)]
    if do_correction:
        corrected, ratio = _correct_ibi_artifacts_with_ratio(valid, threshold=0.20)
        return valid, corrected, ratio
    return valid, valid.copy(), 0.0


def _rmssd(ibi_ms: np.ndarray) -> float:
    if ibi_ms.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))


def _plot_row(row: pd.Series, dataset_dir: Path, out_dir: Path, rank: int) -> dict[str, object]:
    participant = str(row["participant"])
    window_index = int(row["window_index"])
    device = str(row["device"])
    channel = str(row["channel"])
    fiducial = str(row["fiducial"])

    with np.load(_participant_npz(dataset_dir, participant), allow_pickle=True) as z:
        devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
        channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
        di = devices.index(device)
        ci = channels.index(channel)

        arrays = {
            "ppg_grid_timestamp_ms": np.asarray(z["ppg_grid_timestamp_ms"], dtype=np.float64),
            "ppg_rawslot_values": np.asarray(z["ppg_rawslot_values"], dtype=np.float32),
            "ppg_rawslot_mask": np.asarray(z["ppg_rawslot_mask"], dtype=bool),
            "ppg_rawslot_timestamp_ms": np.asarray(z["ppg_rawslot_timestamp_ms"], dtype=np.float64),
            "ppg_rawslot_valid_sample_ratio": np.asarray(z["ppg_rawslot_valid_sample_ratio"], dtype=np.float32),
        }
        signal, times_ms, fs, _sample_ratio, _max_gap_ms = _strict_interp100_channel(
            arrays, window_index, di, ci, max_gap_ms=100.0
        )
        fiducials, oriented, polarity, _score, _votes, _consensus = _choose_polarity_detector(
            "qppgfast",
            signal,
            times_ms,
            fs,
            common_band=(0.7, 3.5),
            polarity_mode="peak_train",
        )
        fid = np.asarray(fiducials[fiducial], dtype=np.int64)
        fid_times = times_ms[fid]
        ppg_ibi_raw, ppg_ibi_used, corr_ratio = _ibi_from_times_ms(fid_times, do_correction=True)
        ppg_ibi_t = (fid_times[1 : 1 + ppg_ibi_used.size] - times_ms[0]) / 1000.0

        ecg_rr = np.asarray(z["ecg_rr_intervals_corrected_ms"][window_index], dtype=np.float64)
        ecg_rr = ecg_rr[np.isfinite(ecg_rr)]
        ecg_peak_t = np.asarray(z["ecg_r_peak_times_rel_ms"][window_index], dtype=np.float64)
        ecg_ibi_t = ecg_peak_t[1 : 1 + ecg_rr.size] / 1000.0 if ecg_peak_t.size > 1 else np.arange(ecg_rr.size)

    t_sec = (times_ms - times_ms[0]) / 1000.0
    title = (
        f"#{rank} {participant} W{window_index} {device} {channel} {fiducial} | "
        f"RMSSD err={row['rmssd_abs_error_ms']:.1f} ms"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(t_sec, oriented, color="#1f77b4", linewidth=0.8)
    if fid.size:
        axes[0].scatter(t_sec[fid], oriented[fid], s=12, color="#d62728", label=f"{fiducial} fiducials")
    axes[0].set_title(title)
    axes[0].set_ylabel("PPG filtered")
    axes[0].legend(loc="upper right")

    axes[1].plot(ecg_ibi_t, ecg_rr, color="#2ca02c", linewidth=1.0, label=f"ECG IBI, RMSSD={row['ecg_rmssd_ms']:.1f} ms")
    axes[1].plot(ppg_ibi_t, ppg_ibi_used, color="#ff7f0e", linewidth=1.0, alpha=0.9, label=f"PPG IBI, RMSSD={row['ppg_rmssd_ms']:.1f} ms")
    axes[1].set_ylabel("IBI (ms)")
    axes[1].legend(loc="upper right")

    if ppg_ibi_raw.size:
        axes[2].plot(ppg_ibi_t[: ppg_ibi_raw.size], ppg_ibi_raw, color="#9467bd", linewidth=0.9, label="PPG raw valid IBI")
    axes[2].plot(ppg_ibi_t, ppg_ibi_used, color="#8c564b", linewidth=1.0, label=f"PPG corrected IBI, correction ratio={corr_ratio:.2f}")
    axes[2].axhline(300, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].axhline(2000, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Seconds in 5-min window")
    axes[2].set_ylabel("PPG IBI (ms)")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 300)

    out_name = f"rank{rank:02d}_{participant}_w{window_index:04d}_{device}_{channel}_{fiducial}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "rank": rank,
        "participant": participant,
        "window_index": window_index,
        "device": device,
        "channel": channel,
        "fiducial": fiducial,
        "polarity": polarity,
        "ecg_rmssd_ms": float(row["ecg_rmssd_ms"]),
        "ppg_rmssd_ms": float(row["ppg_rmssd_ms"]),
        "rmssd_abs_error_ms": float(row["rmssd_abs_error_ms"]),
        "ppg_valid_ibi_ratio": float(row["ppg_valid_ibi_ratio"]),
        "n_fiducials": int(fid.size),
        "recomputed_ppg_rmssd_ms": _rmssd(ppg_ibi_used),
        "recomputed_ppg_ibi_correction_ratio": corr_ratio,
        "plot_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--valid-ibi-gate", type=float, default=0.85)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _select_high_error_rows(args.metrics_csv, args.top_n, args.valid_ibi_gate)
    if rows.empty:
        raise SystemExit("No finite high-error rows matched the selection criteria.")

    records = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        records.append(_plot_row(row, args.dataset_dir, args.out_dir, rank))
    index = pd.DataFrame(records)
    index.to_csv(args.out_dir / "high_rmssd_spot_check_index.csv", index=False)
    rows.to_csv(args.out_dir / "high_rmssd_spot_check_source_rows.csv", index=False)
    print(f"Wrote {len(index)} plots to {args.out_dir}")
    print(index[["rank", "participant", "window_index", "device", "channel", "rmssd_abs_error_ms", "plot_path"]].to_string(index=False))


if __name__ == "__main__":
    main()

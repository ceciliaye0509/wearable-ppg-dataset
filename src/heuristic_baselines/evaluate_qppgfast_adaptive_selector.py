"""Evaluate a pre-defined qppgfast peak/foot/reject adaptive selector."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rawslots_baseline_ablation import DATASET_NAME, _dataset_npz_files, _fast_ppg_sqi, _fiducial_metrics, _score_ibi_train
from evaluate_rawslots_detector_fiducials import DetectorMethod, _bandpass_or_original, _detect_oriented, _participant_from_path, _strict_interp100_channel


DEVICE_FIDUCIALS = {"Earring": "peak", "Ring": "foot", "Watch": "foot"}
DEFAULT_MIN_SCORE = 2.50
DEFAULT_MIN_MARGIN = 0.05
COMMON_BAND = (0.7, 3.5)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    return (
        np.isfinite(df["ppg_rmssd_ms"].astype(float))
        & np.isfinite(df["ppg_sdnn_ms"].astype(float))
        & np.isfinite(df["ecg_rmssd_ms"].astype(float))
        & np.isfinite(df["ecg_sdnn_ms"].astype(float))
    )


def _correlation(pred: pd.Series, ref: pd.Series) -> float:
    if len(pred) < 3 or float(pred.std()) == 0.0 or float(ref.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, ref)[0, 1])


def _arrays(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "ppg_grid_timestamp_ms": np.asarray(z["ppg_grid_timestamp_ms"], dtype=np.float64),
        "ppg_rawslot_values": np.asarray(z["ppg_rawslot_values"], dtype=np.float32),
        "ppg_rawslot_mask": np.asarray(z["ppg_rawslot_mask"], dtype=bool),
        "ppg_rawslot_timestamp_ms": np.asarray(z["ppg_rawslot_timestamp_ms"], dtype=np.float64),
        "ppg_rawslot_valid_sample_ratio": np.asarray(z["ppg_rawslot_valid_sample_ratio"], dtype=np.float32),
        "ecg_rmssd_corrected_ms": np.asarray(z["ecg_rmssd_corrected_ms"], dtype=np.float32),
        "ecg_sdnn_corrected_ms": np.asarray(z["ecg_sdnn_corrected_ms"], dtype=np.float32),
    }


def _candidates(signal: np.ndarray, times_ms: np.ndarray, fs: float) -> list[tuple[float, str, str, np.ndarray, np.ndarray]]:
    """Return the four detector/fiducial candidates without selecting one."""
    filt = _bandpass_or_original(signal, fs, COMMON_BAND)
    candidates: list[tuple[float, str, str, np.ndarray, np.ndarray]] = []
    for sign, polarity in ((1, "positive"), (-1, "negative")):
        oriented = filt if sign > 0 else -filt
        fiducials = _detect_oriented("qppgfast", oriented, times_ms, fs)
        for fiducial in ("peak", "foot"):
            candidates.append((_score_ibi_train(fiducials[fiducial], times_ms), polarity, fiducial, fiducials[fiducial], oriented))
    candidates.sort(key=lambda item: (item[0], item[1] == "positive", item[2] == "peak"), reverse=True)
    return candidates


def _select(
    candidates: list[tuple[float, str, str, np.ndarray, np.ndarray]],
    *,
    min_score: float,
    min_margin: float,
) -> tuple[str, str, np.ndarray, np.ndarray, float, float, str]:
    best_score, polarity, fiducial, selected, oriented = candidates[0]
    second_score = candidates[1][0]
    if best_score < min_score:
        return polarity, fiducial, np.empty(0, dtype=np.int64), oriented, best_score, second_score, "reject_low_score"
    if best_score - second_score < min_margin:
        return polarity, fiducial, np.empty(0, dtype=np.int64), oriented, best_score, second_score, "reject_ambiguous"
    return polarity, fiducial, selected, oriented, best_score, second_score, "selected"


def _evaluate(
    dataset_dir: Path,
    participants: tuple[str, ...] | None,
    *,
    min_score: float,
    min_margin: float,
) -> pd.DataFrame:
    paths = _dataset_npz_files(dataset_dir)
    if participants is not None:
        wanted = set(participants)
        paths = [path for path in paths if _participant_from_path(path) in wanted]
        missing = wanted - {_participant_from_path(path) for path in paths}
        if missing:
            raise ValueError(f"participants not found: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            participant = str(np.asarray(z["participant"]).item()) if "participant" in z.files else _participant_from_path(path)
            devices = [str(value) for value in np.asarray(z["devices"]).tolist()]
            channels = [str(value) for value in np.asarray(z["channels"]).tolist()]
            arrays = _arrays(z)
            n_windows = int(arrays["ecg_rmssd_corrected_ms"].shape[0])
            print(f"[adaptive-selector] {participant} windows={n_windows}")
            for wi in range(n_windows):
                for di, device in enumerate(devices):
                    for ci, channel in enumerate(channels):
                        signal, times_ms, fs, sample_ratio, max_gap = _strict_interp100_channel(arrays, wi, di, ci, max_gap_ms=100.0)
                        base = {
                            "dataset": DATASET_NAME,
                            "participant": participant,
                            "window_index": wi,
                            "device": device,
                            "channel": channel,
                            "ppg_valid_sample_ratio": sample_ratio,
                            "ppg_max_raw_gap_ms": max_gap,
                            "ecg_rmssd_ms": float(arrays["ecg_rmssd_corrected_ms"][wi]),
                            "ecg_sdnn_ms": float(arrays["ecg_sdnn_corrected_ms"][wi]),
                        }
                        if signal.size < 50 or not np.isfinite(signal).all():
                            method = DetectorMethod("qppgfast", "peak", common_band=COMMON_BAND, ibi_correction=True)
                            rows.append({
                                **base,
                                "selected_polarity": "invalid",
                                "selected_fiducial": "reject",
                                "best_score": np.nan,
                                "second_score": np.nan,
                                "score_margin": np.nan,
                                "selector_status": "invalid_strict_input",
                                **_fiducial_metrics(np.empty(0, dtype=np.int64), signal, times_ms, method),
                            })
                            continue
                        candidates = _candidates(signal, times_ms, fs)
                        top_score, top_polarity, top_fiducial, top_indices, top_signal = candidates[0]
                        top_method = DetectorMethod("qppgfast", top_fiducial, common_band=COMMON_BAND, ibi_correction=True)
                        top_sqi = _fast_ppg_sqi(signal, times_ms, top_indices)
                        top_metrics = _fiducial_metrics(top_indices, top_signal, times_ms, top_method, sqi=top_sqi)
                        polarity, fiducial, selected, oriented, best_score, second_score, status = _select(
                            candidates,
                            min_score=min_score,
                            min_margin=min_margin,
                        )
                        selected_metrics = top_metrics if status == "selected" else _fiducial_metrics(
                            np.empty(0, dtype=np.int64),
                            top_signal,
                            times_ms,
                            top_method,
                        )
                        rows.append({
                            **base,
                            "selected_polarity": polarity,
                            "selected_fiducial": fiducial if status == "selected" else "reject",
                            "top_polarity": top_polarity,
                            "top_fiducial": top_fiducial,
                            "top_score": top_score,
                            "best_score": best_score,
                            "second_score": second_score,
                            "score_margin": best_score - second_score,
                            "selector_status": status,
                            **{f"top_{key}": value for key, value in top_metrics.items()},
                            **selected_metrics,
                        })
    return pd.DataFrame(rows)


def _participant_channel_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(["participant", "device", "channel"], dropna=False):
        valid = _valid_mask(group)
        accepted = group["selector_status"].eq("selected") if "selector_status" in group else pd.Series(True, index=group.index)
        value = group[valid]
        rows.append({
            "method": label,
            "participant": keys[0],
            "device": keys[1],
            "channel": keys[2],
            "n_total": int(len(group)),
            "n_valid": int(valid.sum()),
            "coverage_over_total_pct": 100.0 * float(valid.mean()),
            "peak_selected_pct": 100.0 * float((group.get("selected_fiducial", pd.Series("", index=group.index)) == "peak").mean()),
            "foot_selected_pct": 100.0 * float((group.get("selected_fiducial", pd.Series("", index=group.index)) == "foot").mean()),
            "reject_pct": 100.0 * float((~accepted).mean()),
            "MAE_ms": float(np.mean(np.abs(value["ppg_rmssd_ms"] - value["ecg_rmssd_ms"]))) if len(value) else float("nan"),
            "R": _correlation(value["ppg_rmssd_ms"], value["ecg_rmssd_ms"]),
        })
    return pd.DataFrame(rows)


def _device_summary(channel: pd.DataFrame) -> pd.DataFrame:
    return (
        channel.groupby(["method", "device"], dropna=False)
        .agg(
            participant_channels=("MAE_ms", "size"),
            coverage_pct=("coverage_over_total_pct", "mean"),
            peak_selected_pct=("peak_selected_pct", "mean"),
            foot_selected_pct=("foot_selected_pct", "mean"),
            reject_pct=("reject_pct", "mean"),
            MAE_ms=("MAE_ms", "mean"),
            R=("R", "mean"),
        )
        .reset_index()
    )


def _current_comparator(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["detector"] == "qppgfast") & (df["polarity_mode"] == "peak_train")].copy()
    expected = df["device"].map(DEVICE_FIDUCIALS)
    return df[df["fiducial"] == expected].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed adaptive qppgfast peak/foot/reject selector.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--participants", type=str, required=True)
    parser.add_argument("--current-comparator", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    args = parser.parse_args()
    participants = tuple(value.strip().upper() for value in args.participants.split(",") if value.strip())
    selected = _evaluate(
        args.dataset_dir,
        participants,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )
    current = _current_comparator(args.current_comparator)
    selected_channel = _participant_channel_summary(selected, "adaptive_peak_foot_reject_v1")
    current_channel = _participant_channel_summary(current, "peak_train_devicewise")
    channel = pd.concat([selected_channel, current_channel], ignore_index=True)
    device = _device_summary(channel)
    overall = (
        channel.groupby("method", dropna=False)
        .agg(
            participant_channels=("MAE_ms", "size"),
            coverage_pct=("coverage_over_total_pct", "mean"),
            peak_selected_pct=("peak_selected_pct", "mean"),
            foot_selected_pct=("foot_selected_pct", "mean"),
            reject_pct=("reject_pct", "mean"),
            MAE_ms=("MAE_ms", "mean"),
            R=("R", "mean"),
        )
        .reset_index()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "adaptive_selector_window_metrics.csv", index=False)
    channel.to_csv(args.out_dir / "participant_channel_summary.csv", index=False)
    device.to_csv(args.out_dir / "device_summary.csv", index=False)
    overall.to_csv(args.out_dir / "overall_summary.csv", index=False)
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()

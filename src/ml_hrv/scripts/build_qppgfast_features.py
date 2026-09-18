"""Build frozen qPPGFast features for one raw-slot HRV development fold.

This intentionally uses only NumPy/SciPy plus the repository's qPPGFast
implementation so it runs in the existing Python 3.8 ``water-legacy``
environment.  The output is PPG-only and contains no ECG target fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ml_hrv.config import ExperimentConfig


FROZEN_QPPGFAST = {
    "detector": "qppgfast",
    "fiducial": "peak",
    "input": "strict interpolation to 100 Hz; max raw support gap 100 ms; no boundary extrapolation",
    "common_bandpass_hz": [0.7, 3.5],
    "ibi_correction": True,
    "ibi_correction_threshold": 0.20,
    "polarity_mode": "peak_train",
    "device": "Earring",
    "channel": "green",
}
KEYS = ("participant", "window_index", "device", "channel")
FIELDS = (
    *KEYS, "split", "qppg_rmssd_ms", "qppg_sdnn_ms", "qppg_mean_ibi_ms",
    "qppg_hr_bpm", "qppg_peak_count", "qppg_valid_ibi_ratio", "qppg_ibi_cv",
    "qppg_ibi_correction_ratio", "qppg_sqi", "qppg_valid_sample_ratio",
    "qppg_max_raw_gap_ms", "accel_motion_mean_mag", "qppg_polarity", "qppg_valid",
)


def _heuristic_modules():
    root = Path(__file__).resolve().parents[2] / "heuristic_baselines"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms import hrv, qppgfast
    from preprocess import bandpass_filter
    return hrv, qppgfast, bandpass_filter


def _raw_paths(source_dir: Path, participants: set[str]) -> list[Path]:
    paths = sorted(source_dir.glob("*.npz"))
    found = {path.stem.rsplit("_", 1)[-1].upper() for path in paths}
    missing = sorted(participants - found)
    if missing:
        raise FileNotFoundError("raw-slot NPZ files missing for: %s" % missing)
    return [path for path in paths if path.stem.rsplit("_", 1)[-1].upper() in participants]


def _strict_interp100(values, timestamps_ms, mask, grid_ms, raw_ratio):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(timestamps_ms)
    if int(valid.sum()) < 2 or len(grid_ms) < 50:
        return np.empty(0), np.empty(0), float("inf")
    times = np.asarray(timestamps_ms[valid], dtype=np.float64)
    signal = np.asarray(values[valid], dtype=np.float64)
    order = np.argsort(times)
    times, signal = times[order], signal[order]
    unique = np.r_[True, np.diff(times) > 0]
    times, signal = times[unique], signal[unique]
    if len(times) < 2:
        return np.empty(0), np.empty(0), float("inf")
    leading = max(0.0, float(times[0] - grid_ms[0]))
    trailing = max(0.0, float(grid_ms[-1] - times[-1]))
    max_gap = float(max(np.max(np.diff(times)), leading, trailing))
    if not np.isfinite(max_gap) or max_gap > 100.0:
        return np.empty(0), np.empty(0), max_gap
    inside = (grid_ms >= times[0]) & (grid_ms <= times[-1])
    out_time = np.asarray(grid_ms[inside], dtype=np.float64)
    if len(out_time) < 50:
        return np.empty(0), np.empty(0), max_gap
    return np.interp(out_time, times, signal), out_time, max_gap


def _score_ibi_train(fiducials, times_ms, hrv):
    fiducials = np.asarray(fiducials, dtype=np.int64)
    if len(fiducials) < 3:
        return -1.0
    ibi = np.diff(np.asarray(times_ms, dtype=np.float64)[fiducials])
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ratio = float(np.mean(valid)) if len(ibi) else 0.0
    nn = ibi[valid]
    if len(nn) < 3:
        return valid_ratio
    mean_nn = float(np.mean(nn))
    hr_score = 1.0 if 35.0 <= 60000.0 / mean_nn <= 180.0 else 0.0
    cv = float(np.std(nn, ddof=1) / mean_nn) if mean_nn > 0 and len(nn) > 1 else 1.0
    return valid_ratio + hr_score + float(np.clip(1.0 - cv / 0.40, 0.0, 1.0))


def _sqi(signal, times_ms, peaks, hrv):
    if len(peaks) < 5 or len(signal) < 50:
        return 0.0
    frequency = 1000.0 / float(np.median(np.diff(times_ms)))
    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(power.sum())
    freqs = np.fft.rfftfreq(len(signal), 1.0 / frequency)
    spectral = 0.0 if total < 1e-12 else float(np.clip(power[(freqs >= .7) & (freqs <= 3.5)].sum() / total, 0.0, 1.0))
    ibi = np.diff(times_ms[peaks])
    nn = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if len(nn) < 3:
        regularity = 0.0
    else:
        regularity = float(np.clip(1.0 - np.std(nn, ddof=1) / np.mean(nn) / .35, 0.0, 1.0))
    return float(np.clip(.55 * spectral + .45 * regularity, 0.0, 1.0))


def _metrics(signal, times_ms, peaks, hrv):
    peaks = np.asarray(peaks, dtype=np.int64)
    result = {
        "qppg_rmssd_ms": float("nan"), "qppg_sdnn_ms": float("nan"),
        "qppg_mean_ibi_ms": float("nan"), "qppg_hr_bpm": float("nan"),
        "qppg_peak_count": int(len(peaks)), "qppg_valid_ibi_ratio": 0.0,
        "qppg_ibi_cv": float("nan"), "qppg_ibi_correction_ratio": float("nan"),
        "qppg_sqi": _sqi(signal, times_ms, peaks, hrv),
    }
    if len(peaks) < 3:
        return result
    ibi = np.diff(times_ms[peaks])
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    result["qppg_valid_ibi_ratio"] = float(np.mean(valid)) if len(ibi) else 0.0
    nn = ibi[valid]
    if len(nn) < 3:
        return result
    corrected, ratio = hrv._correct_ibi_artifacts_with_ratio(nn, threshold=.20)
    diffs = np.diff(corrected)
    result.update({
        "qppg_rmssd_ms": float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else float("nan"),
        "qppg_sdnn_ms": float(np.std(corrected, ddof=1)) if len(corrected) > 1 else float("nan"),
        "qppg_mean_ibi_ms": float(np.mean(corrected)),
        "qppg_hr_bpm": float(60000.0 / np.mean(corrected)) if np.mean(corrected) > 0 else float("nan"),
        "qppg_ibi_cv": float(np.std(corrected, ddof=1) / np.mean(corrected)) if len(corrected) > 1 and np.mean(corrected) > 0 else float("nan"),
        "qppg_ibi_correction_ratio": float(ratio),
    })
    return result


def _participant_rows(path, split, hrv, qppgfast, bandpass_filter):
    rows = []
    with np.load(path, allow_pickle=True) as data:
        participant = str(np.asarray(data["participant"]).item()) if "participant" in data.files else path.stem.rsplit("_", 1)[-1]
        devices = [str(x) for x in np.asarray(data["devices"]).tolist()]
        channels = [str(x) for x in np.asarray(data["channels"]).tolist()]
        device_index = devices.index("Earring")
        # The raw release names channels ``ppg_green``/``ppg_ir``; the ML
        # cache's public contract intentionally shortens these to green/ir.
        channel_index = channels.index("green") if "green" in channels else channels.index("ppg_green")
        count = int(np.asarray(data["ecg_rmssd_corrected_ms"]).shape[0])
        for index in range(count):
            values = np.asarray(data["ppg_rawslot_values"][index, device_index, channel_index], dtype=np.float64)
            times = np.asarray(data["ppg_rawslot_timestamp_ms"][index, device_index, channel_index], dtype=np.float64)
            mask = np.asarray(data["ppg_rawslot_mask"][index, device_index, channel_index], dtype=bool)
            grid = np.asarray(data["ppg_grid_timestamp_ms"][index], dtype=np.float64)
            raw_ratio = float(data["ppg_rawslot_valid_sample_ratio"][index, device_index, channel_index])
            signal, grid_times, max_gap = _strict_interp100(values, times, mask, grid, raw_ratio)
            row = {
                "participant": participant, "window_index": index, "device": "Earring", "channel": "green", "split": split,
                "qppg_valid_sample_ratio": raw_ratio, "qppg_max_raw_gap_ms": max_gap,
                "accel_motion_mean_mag": float(data["accel_motion_mean_mag"][index, device_index]), "qppg_polarity": "invalid",
            }
            if len(signal) >= 50:
                try:
                    filtered = bandpass_filter(signal, .7, 3.5, 100.0)
                    positive, _ = qppgfast._qppgfast_beat_detector(np.array(filtered, copy=True), 100.0)
                    negative, _ = qppgfast._qppgfast_beat_detector(np.array(-filtered, copy=True), 100.0)
                    if _score_ibi_train(negative, grid_times, hrv) > _score_ibi_train(positive, grid_times, hrv):
                        peaks, oriented, polarity = negative, -filtered, "negative"
                    else:
                        peaks, oriented, polarity = positive, filtered, "positive"
                    row["qppg_polarity"] = polarity
                    row.update(_metrics(oriented, grid_times, peaks, hrv))
                except Exception:
                    row.update(_metrics(signal, grid_times, np.empty(0, dtype=np.int64), hrv))
            else:
                row.update(_metrics(signal, grid_times, np.empty(0, dtype=np.int64), hrv))
            row["qppg_valid"] = bool(np.isfinite(row["qppg_rmssd_ms"]) and np.isfinite(row["qppg_sdnn_ms"]))
            rows.append(row)
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows):
    keys = {(row["participant"], row["window_index"], row["device"], row["channel"]) for row in rows}
    count = sum(bool(row["qppg_valid"]) for row in rows)
    return {"rows": len(rows), "qppg_valid_rows": count, "qppg_valid_fraction": count / len(rows) if rows else float("nan"), "unique_keys": len(keys)}


def main():
    parser = argparse.ArgumentParser(description="Build frozen qPPGFast features for one fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold-name", default="fold_0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if tuple(config.data.devices) != ("Earring",) or tuple(config.data.ppg_channels) != ("green",):
        raise ValueError("This frozen adapter is limited to the Earring green comparison contract")
    raw_folds = json.loads((Path(args.run_dir) / "folds.json").read_text(encoding="utf-8"))
    folds = {str(fold["name"]): fold for fold in raw_folds}
    fold = folds.get(args.fold_name)
    if fold is None:
        raise KeyError("No %s in %s/folds.json" % (args.fold_name, args.run_dir))
    split_by_participant = {str(participant): "train" for participant in fold["train"]}
    split_by_participant.update({str(participant): "val" for participant in fold["val"]})
    hrv, qppgfast, bandpass_filter = _heuristic_modules()
    rows = []
    for path in _raw_paths(Path(config.data.source_dir), set(split_by_participant)):
        participant = path.stem.rsplit("_", 1)[-1].upper()
        participant_rows = _participant_rows(path, split_by_participant[participant], hrv, qppgfast, bandpass_filter)
        rows.extend(participant_rows)
        print("[qppgfast-features] %s rows=%d valid=%d" % (participant, len(participant_rows), sum(x["qppg_valid"] for x in participant_rows)), flush=True)
    if len({tuple(row[key] for key in KEYS) for row in rows}) != len(rows):
        raise RuntimeError("qPPGFast output has duplicate window keys")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "qppgfast_features.csv", rows)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    _write_csv(output / "qppgfast_features_train.csv", train)
    _write_csv(output / "qppgfast_features_val.csv", val)
    audit = {
        "contract": {"name": "comparison_rawslot100_v1", **FROZEN_QPPGFAST}, "fold": fold,
        "all": _summary(rows), "train": _summary(train), "val": _summary(val),
        "source_dir": str(Path(config.data.source_dir)),
        "label_leakage": "none: output fields are computed from PPG, timestamps, and accelerometer metadata only",
    }
    (output / "qppgfast_feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

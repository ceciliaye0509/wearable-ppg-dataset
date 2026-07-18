"""
基于重新计算 PPG 峰值的 device-wise v1.1 启发式 baseline。

本脚本从 rawaligned NPZ 里的 `ppg_resampled` 出发，重新计算 PPG 峰值、
IBI、SQI、QC 和 HRV。本脚本不使用已保存的 PPG 派生元数据，不训练模型，
也不做跨设备融合。

v1.1 在 v1 的基础上扩展一个很小的候选集合：
  - SciPy prominence：0.25 / 0.30
  - IBI correction threshold：0.20 / 0.30
  - 当前主力设置的 peak refinement 变体（默认不跑，需显式打开）

策略选择只在重新计算后的 training set 上重新冻结，然后应用到 strict_reference。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from algorithms import hrv  # noqa: E402
from preprocess import bandpass_filter  # noqa: E402


METRICS = ("RMSSD", "SDNN")
CHANNELS = ("ppg_green", "ppg_ir")
DEVICES = ("Earring", "Ring", "Watch")
PARTICIPANTS = ("P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _agreement_stats(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    ppg = np.asarray(ppg, dtype=np.float64)
    ecg = np.asarray(ecg, dtype=np.float64)
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p = ppg[mask]
    e = ecg[mask]
    if p.size == 0:
        return {
            "n_valid": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R": float("nan"),
            "bias": float("nan"),
            "LoA_lower": float("nan"),
            "LoA_upper": float("nan"),
        }
    diff = p - e
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "R": _safe_corr(p, e),
        "bias": bias,
        "LoA_lower": bias - 1.96 * sd if np.isfinite(sd) else float("nan"),
        "LoA_upper": bias + 1.96 * sd if np.isfinite(sd) else float("nan"),
    }


@dataclass(frozen=True)
class PeakMethod:
    name: str
    detector: str
    band_low_hz: float
    band_high_hz: float
    prominence_std: float
    correction_threshold: float
    refine_peaks: bool = False


PEAK_METHODS = (
    PeakMethod("scipy_bp07_35_prom025_corr02", "scipy", 0.7, 3.5, 0.25, 0.20),
    PeakMethod("scipy_bp05_40_prom025_corr02", "scipy", 0.5, 4.0, 0.25, 0.20),
    PeakMethod("scipy_bp07_35_prom025_corr03", "scipy", 0.7, 3.5, 0.25, 0.30),
    PeakMethod("scipy_bp05_40_prom025_corr03", "scipy", 0.5, 4.0, 0.25, 0.30),
    PeakMethod("scipy_bp07_35_prom030_corr02", "scipy", 0.7, 3.5, 0.30, 0.20),
    PeakMethod("scipy_bp05_40_prom030_corr02", "scipy", 0.5, 4.0, 0.30, 0.20),
    PeakMethod("scipy_bp07_35_prom030_corr03", "scipy", 0.7, 3.5, 0.30, 0.30),
    PeakMethod("scipy_bp05_40_prom030_corr03", "scipy", 0.5, 4.0, 0.30, 0.30),
    PeakMethod("scipy_bp07_35_prom025_corr02_refine", "scipy", 0.7, 3.5, 0.25, 0.20, True),
    PeakMethod("scipy_bp05_40_prom025_corr02_refine", "scipy", 0.5, 4.0, 0.25, 0.20, True),
    PeakMethod("scipy_bp07_35_prom025_corr03_refine", "scipy", 0.7, 3.5, 0.25, 0.30, True),
    PeakMethod("scipy_bp05_40_prom025_corr03_refine", "scipy", 0.5, 4.0, 0.25, 0.30, True),
)

NEUROKIT_METHODS = (
    PeakMethod("nk_elgendi_raw_cleanonly_corr02", "neurokit_raw", 0.0, 0.0, 0.0, 0.20),
    PeakMethod("nk_elgendi_bp07_35_doubleclean_corr02", "neurokit_bandpass", 0.7, 3.5, 0.0, 0.20),
)


@dataclass(frozen=True)
class Gate:
    name: str
    min_sqi: float
    min_valid_ibi: float
    max_correction: float
    max_ibi_cv: float
    max_rmssd_ms: float


GATES = (
    Gate("gate_sqi04_ibi08_corr02_cv25_rmssd200", 0.4, 0.80, 0.20, 0.25, 200.0),
    Gate("gate_sqi04_ibi08_corr03_cv30_rmssd200", 0.4, 0.80, 0.30, 0.30, 200.0),
    Gate("gate_sqi05_ibi08_corr02_cv25_rmssd200", 0.5, 0.80, 0.20, 0.25, 200.0),
    Gate("gate_sqi04_ibi09_corr02_cv25_rmssd200", 0.4, 0.90, 0.20, 0.25, 200.0),
    Gate("gate_sqi035_ibi08_corr03_cv35_rmssd200", 0.35, 0.80, 0.30, 0.35, 200.0),
)


def _dataset_npz_files(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.glob("*.npz") if not p.name.startswith(".") and "_summary" not in p.name)


def _fill_missing(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if np.isfinite(y).all():
        return y
    idx = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() < 50:
        return np.full_like(y, np.nan)
    return np.interp(idx, idx[good], y[good])


def _detect_scipy(signal: np.ndarray, fs: float, prominence_std: float, polarity: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if polarity < 0:
        x = -x
    x = x - np.nanmean(x)
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < 1e-12:
        return np.empty(0, dtype=np.int64)
    min_dist = max(1, int(round(0.30 * fs)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence_std * std)
    return peaks.astype(np.int64)


def _detect_neurokit_elgendi(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if hrv.nk is None:
        return np.empty(0, dtype=np.int64), np.asarray(signal, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = hrv.nk.ppg_clean(np.asarray(signal, dtype=np.float64), sampling_rate=fs, method="elgendi")
            _signals, info = hrv.nk.ppg_peaks(clean, sampling_rate=fs, method="elgendi", correct_artifacts=True)
        peaks = np.asarray(info.get("PPG_Peaks", []), dtype=np.int64).ravel()
        peaks = peaks[(peaks >= 0) & (peaks < clean.size)]
        return peaks, np.asarray(clean, dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.int64), np.asarray(signal, dtype=np.float64)


def _fast_ppg_sqi(signal: np.ndarray, fs: float, peaks: np.ndarray) -> float:
    if peaks.size < 5 or signal.size < 50:
        return 0.0
    x = np.asarray(signal, dtype=np.float64)
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    power = fft_vals ** 2
    total = float(np.sum(power))
    if total < 1e-12:
        spectral = 0.0
    else:
        cardiac = (freqs >= 0.7) & (freqs <= 3.5)
        spectral = float(np.clip(np.sum(power[cardiac]) / total, 0.0, 1.0))
    ibi = np.diff(peaks.astype(np.float64)) / fs * 1000.0
    valid = ibi[(ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)]
    if valid.size < 3:
        regularity = 0.0
    else:
        mean_ibi = float(np.mean(valid))
        cv = float(np.std(valid, ddof=1) / mean_ibi) if mean_ibi > 0 else 1.0
        regularity = float(np.clip(1.0 - cv / 0.35, 0.0, 1.0))
    return float(np.clip(0.55 * spectral + 0.45 * regularity, 0.0, 1.0))


def _ibi_metrics(peaks: np.ndarray, signal: np.ndarray, fs: float, correction_threshold: float, refine_peaks: bool = False) -> dict[str, float]:
    peaks_int = np.asarray(peaks, dtype=np.int64)
    if peaks_int.size < 3:
        return {
            "n_peaks": float(peaks_int.size),
            "ppg_rmssd_ms": np.nan,
            "ppg_sdnn_ms": np.nan,
            "ppg_mean_ibi_ms": np.nan,
            "ppg_hr_bpm": np.nan,
            "ppg_valid_ibi_ratio": 0.0,
            "ppg_ibi_cv": np.nan,
            "ppg_ibi_correction_ratio": np.nan,
            "ppg_sqi": 0.0,
        }

    # v1.1 将 peak refinement 作为显式候选变量；SQI 仍使用整数峰值保证可比。
    if refine_peaks:
        peaks_for_ibi = hrv._refine_peaks_parabolic(np.asarray(signal, dtype=np.float64), peaks_int)
    else:
        peaks_for_ibi = peaks_int.astype(np.float64)
    ibi = np.diff(peaks_for_ibi) / float(fs) * 1000.0
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ibi_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    if nn.size < 3:
        rmssd = sdnn = mean_ibi = hr_bpm = ibi_cv = np.nan
        corr_ratio = np.nan
    else:
        corrected, corr_ratio = hrv._correct_ibi_artifacts_with_ratio(nn, threshold=correction_threshold)
        diff = np.diff(corrected)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
        sdnn = float(np.std(corrected, ddof=1)) if corrected.size > 1 else np.nan
        mean_ibi = float(np.mean(corrected)) if corrected.size else np.nan
        hr_bpm = float(60000.0 / mean_ibi) if np.isfinite(mean_ibi) and mean_ibi > 0 else np.nan
        ibi_cv = float(np.std(corrected, ddof=1) / mean_ibi) if corrected.size > 1 and mean_ibi > 0 else np.nan
    sqi = _fast_ppg_sqi(signal, fs, peaks_int)
    return {
        "n_peaks": float(peaks_int.size),
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_mean_ibi_ms": mean_ibi,
        "ppg_hr_bpm": hr_bpm,
        "ppg_valid_ibi_ratio": valid_ibi_ratio,
        "ppg_ibi_cv": ibi_cv,
        "ppg_ibi_correction_ratio": corr_ratio,
        "ppg_sqi": sqi,
    }


def _score_peaks(peaks: np.ndarray, signal: np.ndarray, fs: float, correction_threshold: float) -> float:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 3:
        return -1.0
    ibi = np.diff(peaks.astype(np.float64)) / float(fs) * 1000.0
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    mean_ibi = float(np.mean(nn)) if nn.size else np.nan
    hr = float(60000.0 / mean_ibi) if np.isfinite(mean_ibi) and mean_ibi > 0 else np.nan
    corr_penalty = 1.0
    if nn.size >= 5:
        _corrected, corr_penalty = hrv._correct_ibi_artifacts_with_ratio(nn, threshold=correction_threshold)
    hr_score = 1.0 if np.isfinite(hr) and 35.0 <= hr <= 180.0 else 0.0
    count_score = min(1.0, peaks.size / max(1.0, 5.0 * fs * 300.0 / 60.0))
    return valid_ratio + hr_score + count_score - float(corr_penalty)


def _recompute_one_channel(raw: np.ndarray, fs: float, method: PeakMethod) -> dict[str, float | str]:
    x = _fill_missing(raw)
    if not np.isfinite(x).all():
        return {
            "peak_method": method.name,
            "detector": method.detector,
            "polarity": "invalid",
            **_ibi_metrics(np.empty(0, dtype=np.int64), np.zeros(0), fs, method.correction_threshold, method.refine_peaks),
        }
    if method.detector == "neurokit_raw":
        peaks, clean = _detect_neurokit_elgendi(x, fs)
        return {
            "peak_method": method.name,
            "detector": method.detector,
            "polarity": "neurokit",
            **_ibi_metrics(peaks, clean, fs, method.correction_threshold, method.refine_peaks),
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filt = bandpass_filter(x - np.mean(x), method.band_low_hz, method.band_high_hz, fs)
    except Exception:
        filt = x - np.mean(x)

    if method.detector == "neurokit_bandpass":
        peaks, clean = _detect_neurokit_elgendi(filt, fs)
        return {
            "peak_method": method.name,
            "detector": method.detector,
            "polarity": "neurokit",
            **_ibi_metrics(peaks, clean, fs, method.correction_threshold, method.refine_peaks),
        }

    pos = _detect_scipy(filt, fs, method.prominence_std, polarity=1)
    neg = _detect_scipy(filt, fs, method.prominence_std, polarity=-1)
    pos_score = _score_peaks(pos, filt, fs, method.correction_threshold)
    neg_score = _score_peaks(neg, -filt, fs, method.correction_threshold)
    if neg_score > pos_score:
        peaks = neg
        signal_for_metrics = -filt
        polarity = "negative"
    else:
        peaks = pos
        signal_for_metrics = filt
        polarity = "positive"
    return {
        "peak_method": method.name,
        "detector": method.detector,
        "polarity": polarity,
        **_ibi_metrics(peaks, signal_for_metrics, fs, method.correction_threshold, method.refine_peaks),
    }


def _load_recomputed_rows(
    dataset_dir: Path,
    dataset_name: str,
    role: str,
    participants: tuple[str, ...],
    peak_methods: tuple[PeakMethod, ...],
    max_windows_per_participant: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    files_by_participant = {p.stem.rsplit("_", 1)[-1]: p for p in _dataset_npz_files(dataset_dir)}
    tasks = [
        (
            files_by_participant.get(participant),
            dataset_name,
            role,
            participant,
            peak_methods,
            max_windows_per_participant,
        )
        for participant in participants
    ]

    if workers <= 1:
        rows: list[dict[str, object]] = []
        for task in tasks:
            rows.extend(_load_one_participant_rows(task))
        return pd.DataFrame(rows)

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_load_one_participant_rows, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
    return pd.DataFrame(rows)


def _load_one_participant_rows(
    task: tuple[Path | None, str, str, str, tuple[PeakMethod, ...], int | None],
) -> list[dict[str, object]]:
    path, dataset_name, role, participant, peak_methods, max_windows_per_participant = task
    rows: list[dict[str, object]] = []
    if path is None:
        print(f"[WARN] missing {role} {participant}")
        return rows
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        cfg = json.loads(str(np.asarray(z["config_json"]).item())) if "config_json" in z.files else {}
        fs = float(cfg.get("target_fs", 100.0))
        devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
        channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
        ppg = np.asarray(z["ppg_resampled"], dtype=np.float32)
        valid_ratio = np.asarray(z["ppg_valid_sample_ratio"], dtype=np.float64)
        accel_motion = np.asarray(z["accel_motion_mean_mag"], dtype=np.float64) if "accel_motion_mean_mag" in z.files else None
        ecg_rmssd = np.asarray(z["ecg_rmssd_ms"], dtype=np.float64)
        ecg_sdnn = np.asarray(z["ecg_sdnn_ms"], dtype=np.float64)
        n_windows = int(ppg.shape[0])
        if max_windows_per_participant is not None and n_windows > max_windows_per_participant:
            window_indices = np.unique(np.linspace(0, n_windows - 1, max_windows_per_participant, dtype=int))
        else:
            window_indices = np.arange(n_windows, dtype=int)
        print(f"[{role}] {participant} windows={n_windows} eval_windows={len(window_indices)}")
        for wi in window_indices:
            for di, device in enumerate(devices):
                motion_value = float(accel_motion[wi, di]) if accel_motion is not None else np.nan
                for ci, channel in enumerate(channels):
                    raw = ppg[wi, di, ci]
                    base = {
                        "dataset": dataset_name,
                        "role": role,
                        "participant": participant,
                        "window_index": wi,
                        "device": device,
                        "channel": channel,
                        "target_fs": fs,
                        "ppg_valid_sample_ratio": float(valid_ratio[wi, di, ci]),
                        "accel_motion_mean_mag": motion_value,
                        "ecg_rmssd_ms": float(ecg_rmssd[wi]),
                        "ecg_sdnn_ms": float(ecg_sdnn[wi]),
                    }
                    for method in peak_methods:
                        metrics = _recompute_one_channel(raw, fs, method)
                        rows.append({**base, **metrics})
    return rows


def _gate_mask(df: pd.DataFrame, gate: Gate) -> pd.Series:
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    ibi_cv = df["ppg_ibi_cv"].astype(float)
    return (
        (df["ppg_valid_sample_ratio"].astype(float) >= 0.90)
        & (df["ppg_sqi"].astype(float) >= gate.min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= gate.min_valid_ibi)
        & (np.isfinite(corr) & (corr <= gate.max_correction))
        & (np.isfinite(ibi_cv) & (ibi_cv <= gate.max_ibi_cv))
        & (df["ppg_rmssd_ms"].astype(float) <= gate.max_rmssd_ms)
        & np.isfinite(df["ppg_rmssd_ms"].astype(float))
        & np.isfinite(df["ppg_sdnn_ms"].astype(float))
    )


def _build_strategy_predictions(channel_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gate in GATES:
        gated = channel_df[_gate_mask(channel_df, gate)].copy()
        if gated.empty:
            continue
        fixed = gated.copy()
        fixed["strategy"] = "fixed_" + fixed["channel"].astype(str)
        fixed["gate"] = gate.name
        fixed["method"] = fixed["peak_method"].astype(str) + "__" + fixed["strategy"].astype(str) + "__" + fixed["gate"].astype(str)
        frames.append(fixed)

        sort_cols = [
            "dataset", "role", "participant", "window_index", "device", "peak_method",
            "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio",
        ]
        best = gated.sort_values(sort_cols, ascending=[True, True, True, True, True, True, False, False, True])
        best = best.groupby(["dataset", "role", "participant", "window_index", "device", "peak_method"], dropna=False).head(1).copy()
        best["strategy"] = "device_best_sqi"
        best["gate"] = gate.name
        best["method"] = best["peak_method"].astype(str) + "__device_best_sqi__" + best["gate"].astype(str)
        frames.append(best)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize(predictions: pd.DataFrame, denominators: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "role", "device", "method", "peak_method", "strategy", "gate"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        n_total = denominators[(str(base["dataset"]), str(base["role"]), str(base["device"]))]
        for metric in METRICS:
            ppg_col = f"ppg_{metric.lower()}_ms"
            ecg_col = f"ecg_{metric.lower()}_ms"
            finite = np.isfinite(group[ppg_col].to_numpy(float)) & np.isfinite(group[ecg_col].to_numpy(float))
            stats = _agreement_stats(group.loc[finite, ppg_col].to_numpy(float), group.loc[finite, ecg_col].to_numpy(float))
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": int(n_total),
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
            })
    return pd.DataFrame(rows)


def _choose_frozen_strategies(summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    train = summary[
        (summary["role"] == "training_stride30")
        & (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if train.empty:
        return train
    return (
        train.sort_values(["device", "MAE", "coverage_pct"], ascending=[True, True, False])
        .groupby("device", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def _apply_frozen(summary: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    if frozen.empty:
        return frozen
    keys = frozen[["device", "method"]].drop_duplicates()
    return summary.merge(keys, on=["device", "method"], how="inner").sort_values(["dataset", "device", "hrv_metric"])


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _write_readme(out_dir: Path, channel_df: pd.DataFrame, summary: pd.DataFrame, frozen: pd.DataFrame, frozen_eval: pd.DataFrame) -> None:
    lines = [
        "# 设备级 Baseline v1.1：重新计算 PPG 峰值",
        "",
        "本报告从 `ppg_resampled` 重新计算 PPG 峰值、IBI、SQI、QC 和 HRV。v1.1 在 v1 基础上扩展 prominence、IBI correction threshold 和 peak refinement 候选。",
        "",
        "## 重要原则",
        "",
        "- 不训练模型。",
        "- 不做跨设备 fusion。",
        "- 不使用 ECG label 逐窗口选通道。",
        "- 通道选择、QC 覆盖率和最佳策略全部基于 v1.1 重新计算结果。",
        "- 策略只在 `training_stride30` 上选择并冻结，再报告 `strict_reference`。",
        "",
        "## 评估规模",
        "",
        "| 数据集角色 | 参与者数 | 通道级行数 |",
        "|---|---:|---:|",
    ]
    for role, group in channel_df.groupby("role", dropna=False):
        lines.append(f"| {role} | {group['participant'].nunique()} | {len(group)} |")

    lines.extend([
        "",
        "## 冻结的 v1.1 策略",
        "",
        "| 设备 | 冻结方法 | 训练集 RMSSD MAE | 训练集 R | 训练集覆盖率 |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in frozen.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['method']}` | {_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
        )

    lines.extend([
        "",
        "## 冻结 v1.1 在 strict_reference / training 上的结果",
        "",
        "| 数据集 | 设备 | 指标 | 有效数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['device']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    top = summary[
        (summary["role"] == "training_stride30")
        & (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= 20.0)
    ].copy()
    top = top.sort_values(["device", "MAE", "coverage_pct"], ascending=[True, True, False]).groupby("device", dropna=False).head(5)
    lines.extend([
        "",
        "## Training 上每设备排名靠前的 RMSSD 候选",
        "",
        "| 设备 | 方法 | 有效数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['method']}` | {int(row['n_valid'])} | {_fmt(row['coverage_pct'])}% | "
            f"{_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `v1_1_recomputed_channel_metrics.csv` | 重新计算的通道级 PPG HRV / SQI / IBI 指标 |",
        "| `v1_1_strategy_summary.csv` | 每个候选策略的 MAE/RMSE/R/Bias/覆盖率 |",
        "| `v1_1_frozen_training_strategies.csv` | 在 training_stride30 上重新冻结的每设备策略 |",
        "| `v1_1_frozen_strategy_eval.csv` | 冻结策略在 training 和 strict_reference 上的结果 |",
        "| `summary.json` | 机器可读运行配置 |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估重新计算 PPG 峰值后的 v1.1 device-wise 启发式 baseline。")
    parser.add_argument(
        "--strict-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100"),
    )
    parser.add_argument(
        "--training-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_1_recomputed_peaks"),
    )
    parser.add_argument("--participants", default=",".join(PARTICIPANTS))
    parser.add_argument("--min-freeze-coverage-pct", type=float, default=20.0)
    parser.add_argument("--reuse-channel-metrics", action="store_true")
    parser.add_argument(
        "--include-neurokit",
        action="store_true",
        help="同时评估 NeuroKit Elgendi 变体：raw->NeuroKit clean 以及 bandpass->NeuroKit clean。",
    )
    parser.add_argument(
        "--include-refinement",
        action="store_true",
        help="同时评估 peak refinement 候选；该选项较慢，建议先用于抽样检查。",
    )
    parser.add_argument(
        "--base-channel-metrics-csv",
        default="",
        help="可选：追加已有 v1 channel metrics，避免重复计算 v1 原始候选。",
    )
    parser.add_argument(
        "--max-windows-per-participant",
        type=int,
        default=None,
        help="对每个数据集角色、每个参与者进行确定性子采样，最多保留这么多个窗口。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="并行处理参与者的线程数。",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_csv = out_dir / "v1_1_recomputed_channel_metrics.csv"
    participants = tuple(p.strip() for p in args.participants.split(",") if p.strip())
    peak_methods = tuple(m for m in PEAK_METHODS if args.include_refinement or not m.refine_peaks)
    base_df = pd.DataFrame()
    if args.base_channel_metrics_csv:
        base_df = pd.read_csv(Path(args.base_channel_metrics_csv).resolve())
        existing_names = set(str(x) for x in base_df["peak_method"].dropna().unique())
        peak_methods = tuple(m for m in peak_methods if m.name not in existing_names)
    if args.include_neurokit:
        peak_methods = peak_methods + NEUROKIT_METHODS

    if args.reuse_channel_metrics and channel_csv.exists():
        channel_df = pd.read_csv(channel_csv)
    else:
        strict_df = _load_recomputed_rows(
            Path(args.strict_dir).resolve(),
            Path(args.strict_dir).resolve().name,
            "strict_reference",
            participants,
            peak_methods,
            args.max_windows_per_participant,
            args.workers,
        )
        training_df = _load_recomputed_rows(
            Path(args.training_dir).resolve(),
            Path(args.training_dir).resolve().name,
            "training_stride30",
            participants,
            peak_methods,
            args.max_windows_per_participant,
            args.workers,
        )
        channel_df = pd.concat([strict_df, training_df], ignore_index=True)
        if args.base_channel_metrics_csv and not base_df.empty:
            if set(base_df["role"].dropna().unique()) == {"strict_reference", "training_stride30"}:
                channel_df = pd.concat([base_df, channel_df], ignore_index=True)
        channel_df.to_csv(channel_csv, index=False)

    denominators = {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in channel_df.groupby(["dataset", "role", "device"], dropna=False)
    }
    predictions = _build_strategy_predictions(channel_df)
    summary = _summarize(predictions, denominators)
    frozen = _choose_frozen_strategies(summary, min_coverage_pct=args.min_freeze_coverage_pct)
    frozen_eval = _apply_frozen(summary, frozen)

    summary.to_csv(out_dir / "v1_1_strategy_summary.csv", index=False)
    frozen.to_csv(out_dir / "v1_1_frozen_training_strategies.csv", index=False)
    frozen_eval.to_csv(out_dir / "v1_1_frozen_strategy_eval.csv", index=False)
    run_summary = {
        "strict_dir": str(Path(args.strict_dir).resolve()),
        "training_dir": str(Path(args.training_dir).resolve()),
        "out_dir": str(out_dir),
        "participants": list(participants),
        "peak_methods": [m.__dict__ for m in peak_methods],
        "gates": [g.__dict__ for g in GATES],
        "min_freeze_coverage_pct": args.min_freeze_coverage_pct,
        "cross_device_fusion": False,
        "uses_saved_ppg_metadata": False,
        "include_neurokit": bool(args.include_neurokit),
        "include_refinement": bool(args.include_refinement),
        "base_channel_metrics_csv": str(Path(args.base_channel_metrics_csv).resolve()) if args.base_channel_metrics_csv else "",
        "max_windows_per_participant": args.max_windows_per_participant,
        "workers": args.workers,
    }
    (out_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, channel_df, summary, frozen, frozen_eval)

    print(f"[saved] {out_dir}")
    print(frozen_eval.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Evaluate current v1.2 frozen baseline on training_stride30 with:

- no QC gate;
- no baseline-level fill_missing;
- no IBI artifact correction;
- no peak interpolation/refinement.

This ablation keeps the frozen v1.2 device x channel peak-method choices and
their bandpass / SciPy detector parameters, then recomputes PPG-derived RMSSD
and SDNN directly from valid physiological IBI values.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import config  # noqa: E402
from algorithms import hrv  # noqa: E402
from preprocess import bandpass_filter  # noqa: E402


METRICS = ("RMSSD", "SDNN")
POLICY = "v1_2_frozen_peak_methods_no_qc_no_fill_no_ibi_correction_training_stride30"
PARTICIPANTS = ("P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20")


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _agreement(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p = ppg[mask]
    e = ecg[mask]
    if p.size == 0:
        return {"n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R": np.nan, "bias": np.nan}
    diff = p - e
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "R": _safe_corr(p, e),
        "bias": float(np.mean(diff)),
    }


def _parse_peak_method(name: str) -> dict[str, float]:
    match = re.fullmatch(r"scipy_bp(\d{2})_(\d{2})_prom(\d{3})_corr\d{2}", name)
    if match is None:
        raise ValueError(f"Unsupported frozen peak method for this ablation: {name}")
    low = float(match.group(1)) / 10.0
    high = float(match.group(2)) / 10.0
    prominence_std = float(match.group(3)) / 100.0
    return {"band_low_hz": low, "band_high_hz": high, "prominence_std": prominence_std}


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


def _score_peaks_no_correction(peaks: np.ndarray, fs: float) -> float:
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 3:
        return -1.0
    ibi = np.diff(peaks.astype(np.float64)) / float(fs) * 1000.0
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    mean_ibi = float(np.mean(nn)) if nn.size else np.nan
    hr = float(60000.0 / mean_ibi) if np.isfinite(mean_ibi) and mean_ibi > 0 else np.nan
    hr_score = 1.0 if np.isfinite(hr) and 35.0 <= hr <= 180.0 else 0.0
    count_score = min(1.0, peaks.size / max(1.0, 5.0 * fs * 300.0 / 60.0))
    return valid_ratio + hr_score + count_score


def _ibi_metrics_no_correction(peaks: np.ndarray, fs: float) -> dict[str, float]:
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
            "ppg_ibi_correction_ratio": 0.0,
        }
    ibi = np.diff(peaks_int.astype(np.float64)) / float(fs) * 1000.0
    valid = (ibi >= hrv.IBI_MIN_MS) & (ibi <= hrv.IBI_MAX_MS)
    valid_ibi_ratio = float(np.mean(valid)) if ibi.size else 0.0
    nn = ibi[valid]
    if nn.size < 3:
        rmssd = sdnn = mean_ibi = hr_bpm = ibi_cv = np.nan
    else:
        diff = np.diff(nn)
        rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
        sdnn = float(np.std(nn, ddof=1)) if nn.size > 1 else np.nan
        mean_ibi = float(np.mean(nn))
        hr_bpm = float(60000.0 / mean_ibi) if mean_ibi > 0 else np.nan
        ibi_cv = float(np.std(nn, ddof=1) / mean_ibi) if nn.size > 1 and mean_ibi > 0 else np.nan
    return {
        "n_peaks": float(peaks_int.size),
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_mean_ibi_ms": mean_ibi,
        "ppg_hr_bpm": hr_bpm,
        "ppg_valid_ibi_ratio": valid_ibi_ratio,
        "ppg_ibi_cv": ibi_cv,
        "ppg_ibi_correction_ratio": 0.0,
    }


def _recompute_one_channel_no_fill_no_correction(raw: np.ndarray, fs: float, peak_method: str) -> dict[str, float | str]:
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all():
        return {
            "peak_method": peak_method,
            "detector": "scipy",
            "polarity": "invalid_nonfinite_no_fill",
            **_ibi_metrics_no_correction(np.empty(0, dtype=np.int64), fs),
        }
    parsed = _parse_peak_method(peak_method)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filt = bandpass_filter(x - np.mean(x), parsed["band_low_hz"], parsed["band_high_hz"], fs)
    except Exception:
        filt = x - np.mean(x)

    pos = _detect_scipy(filt, fs, parsed["prominence_std"], polarity=1)
    neg = _detect_scipy(filt, fs, parsed["prominence_std"], polarity=-1)
    pos_score = _score_peaks_no_correction(pos, fs)
    neg_score = _score_peaks_no_correction(neg, fs)
    if neg_score > pos_score:
        peaks = neg
        polarity = "negative"
    else:
        peaks = pos
        polarity = "positive"
    return {
        "peak_method": peak_method,
        "detector": "scipy",
        "polarity": polarity,
        **_ibi_metrics_no_correction(peaks, fs),
    }


def _load_predictions(dataset_dir: Path, dataset_name: str, choices: pd.DataFrame) -> pd.DataFrame:
    files_by_participant = {p.stem.rsplit("_", 1)[-1]: p for p in dataset_dir.glob("*.npz") if "_summary" not in p.name}
    rows: list[dict[str, object]] = []
    choices = choices.copy()
    for participant in PARTICIPANTS:
        path = files_by_participant.get(participant)
        if path is None:
            print(f"[WARN] missing training_stride30 {participant}")
            continue
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
            print(f"[training_stride30] {participant} windows={n_windows}")
            for _, choice in choices.iterrows():
                device = str(choice["device"])
                channel = str(choice["channel"])
                peak_method = str(choice["peak_method"])
                di = devices.index(device)
                ci = channels.index(channel)
                for wi in range(n_windows):
                    metrics = _recompute_one_channel_no_fill_no_correction(ppg[wi, di, ci], fs, peak_method)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "role": "training_stride30",
                            "participant": participant,
                            "window_index": wi,
                            "device": device,
                            "channel": channel,
                            "target_fs": fs,
                            "ppg_valid_sample_ratio": float(valid_ratio[wi, di, ci]),
                            "accel_motion_mean_mag": float(accel_motion[wi, di]) if accel_motion is not None else np.nan,
                            "ecg_rmssd_ms": float(ecg_rmssd[wi]),
                            "ecg_sdnn_ms": float(ecg_sdnn[wi]),
                            "ignored_gate": str(choice["gate"]),
                            "policy": POLICY,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def _summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    for (device, channel), group in pred.groupby(["device", "channel"], dropna=False):
        n_total = int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        base = group.iloc[0]
        for metric in METRICS:
            stats = _agreement(
                group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            rows.append(
                {
                    "dataset": base["dataset"],
                    "role": "training_stride30",
                    "device": device,
                    "channel": channel,
                    "peak_method": base["peak_method"],
                    "ignored_gate": base["ignored_gate"],
                    "policy": POLICY,
                    "hrv_metric": metric,
                    "n_total": n_total,
                    **stats,
                    "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
                }
            )

    for (participant, device, channel), group in pred.groupby(["participant", "device", "channel"], dropna=False):
        n_total = int(group["window_index"].nunique())
        base = group.iloc[0]
        for metric in METRICS:
            stats = _agreement(
                group[f"ppg_{metric.lower()}_ms"].to_numpy(float),
                group[f"ecg_{metric.lower()}_ms"].to_numpy(float),
            )
            participant_rows.append(
                {
                    "dataset": base["dataset"],
                    "role": "training_stride30",
                    "participant": participant,
                    "device": device,
                    "channel": channel,
                    "peak_method": base["peak_method"],
                    "ignored_gate": base["ignored_gate"],
                    "policy": POLICY,
                    "hrv_metric": metric,
                    "n_total": n_total,
                    **stats,
                    "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(participant_rows)


def _join_rmssd_sdnn(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rmssd = summary[summary["hrv_metric"] == "RMSSD"].copy()
    sdnn = summary[summary["hrv_metric"] == "SDNN"][["device", "channel", "MAE", "R"]].rename(
        columns={"MAE": f"{prefix}_SDNN_MAE", "R": f"{prefix}_SDNN_R"}
    )
    rmssd = rmssd.rename(
        columns={
            "n_valid": f"{prefix}_n_valid",
            "coverage_pct": f"{prefix}_coverage_pct",
            "MAE": f"{prefix}_RMSSD_MAE",
            "RMSE": f"{prefix}_RMSSD_RMSE",
            "R": f"{prefix}_RMSSD_R",
            "bias": f"{prefix}_RMSSD_bias",
        }
    )
    return rmssd.merge(sdnn, on=["device", "channel"], how="left")


def _write_readme(
    out_dir: Path,
    summary: pd.DataFrame,
    participant: pd.DataFrame,
    standard_noqc: pd.DataFrame,
    standard_noqc_compare: pd.DataFrame,
    dataset_dir: Path,
    choices_csv: Path,
) -> pd.DataFrame:
    ablation = _join_rmssd_sdnn(summary, "noQC_noFill_noCorr")
    standard = _join_rmssd_sdnn(standard_noqc, "noQC")
    compare = ablation.merge(
        standard[
            [
                "device",
                "channel",
                "noQC_n_valid",
                "noQC_coverage_pct",
                "noQC_RMSSD_MAE",
                "noQC_RMSSD_R",
                "noQC_SDNN_MAE",
                "noQC_SDNN_R",
            ]
        ],
        on=["device", "channel"],
        how="left",
    )
    compare["coverage_change_vs_noQC_pctpt"] = compare["noQC_noFill_noCorr_coverage_pct"] - compare["noQC_coverage_pct"]
    compare["rmssd_mae_change_vs_noQC_ms"] = compare["noQC_noFill_noCorr_RMSSD_MAE"] - compare["noQC_RMSSD_MAE"]
    compare["rmssd_r_change_vs_noQC"] = compare["noQC_noFill_noCorr_RMSSD_R"] - compare["noQC_RMSSD_R"]
    compare["sdnn_mae_change_vs_noQC_ms"] = compare["noQC_noFill_noCorr_SDNN_MAE"] - compare["noQC_SDNN_MAE"]
    compare["sdnn_r_change_vs_noQC"] = compare["noQC_noFill_noCorr_SDNN_R"] - compare["noQC_SDNN_R"]
    compare = compare.sort_values("noQC_noFill_noCorr_RMSSD_MAE", na_position="last").reset_index(drop=True)
    compare.to_csv(out_dir / "v1_2_noQC_noFill_noCorr_vs_noQC_training_stride30_comparison.csv", index=False)

    rmssd_part = participant[participant["hrv_metric"] == "RMSSD"].copy()
    variability = (
        rmssd_part.groupby(["device", "channel"], dropna=False)
        .agg(
            participant_R_median=("R", "median"),
            participant_R_q25=("R", lambda x: x.quantile(0.25)),
            participant_R_q75=("R", lambda x: x.quantile(0.75)),
            participant_coverage_median=("coverage_pct", "median"),
            participant_coverage_min=("coverage_pct", "min"),
            participant_coverage_max=("coverage_pct", "max"),
        )
        .reset_index()
        .sort_values(["device", "channel"])
    )

    lines = [
        "# v1.2 noQC no-fill no-IBI-correction Training-Stride30 Ablation",
        "",
        "本目录汇总当前 v1.2 双通道 frozen baseline 在 `training_stride30` 上关闭所有 QC gate、关闭 baseline-level `fill_missing`、关闭 IBI artifact correction 后的结果。",
        "",
        "## Definition",
        "",
        "保留：",
        "",
        "- `ppg_resampled` 输入，也就是 dataset 生成阶段已经完成的 raw PPG 线性重采样。",
        "- v1.2 frozen `device x channel` peak method。",
        "- frozen peak method 对应的 bandpass 参数。",
        "- SciPy `find_peaks` detector 和 prominence/distance 设置。",
        "- 生理 IBI 范围过滤：`300-2000 ms`，用于从 detected peaks 得到可计算 HRV 的 NN interval。",
        "",
        "关闭：",
        "",
        "- 所有 frozen QC gate：SQI、valid IBI ratio、IBI correction ratio、IBI CV、RMSSD 上限等。",
        "- baseline-level `_fill_missing`：如果输入 window 含 NaN/inf，本次不做线性补点。",
        "- IBI artifact correction：不再用局部中位数替换异常 IBI。",
        "- peak interpolation/refinement：当前 frozen choices 本来也不含 `_refine`。",
        "",
        "因此本 ablation 的处理链是：",
        "",
        "```text",
        "raw PPG -> dataset-level ppg_resampled -> bandpass -> SciPy peak detector -> physiological IBI filter -> RMSSD/SDNN",
        "```",
        "",
        "注意：`ppg_resampled` 中由 dataset 生成阶段写入的 invalid zero-filled 点仍保留为 0；本实验只是关闭 baseline-level `_fill_missing`，不是从原始 timestamp PPG 重新截窗。",
        "",
        "## Inputs",
        "",
        f"- Dataset directory: `{dataset_dir}`",
        f"- Frozen choices: `{choices_csv}`",
        "- Role: `training_stride30`",
        "",
        "## Aggregate Results",
        "",
        "| Rank | Device | Channel | Valid / total | Coverage | RMSSD MAE | RMSSD R | RMSSD bias | SDNN MAE | SDNN R | Peak method |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, (_, row) in enumerate(ablation.sort_values("noQC_noFill_noCorr_RMSSD_MAE", na_position="last").iterrows(), start=1):
        lines.append(
            f"| {i} | `{row['device']}` | `{row['channel']}` | "
            f"{int(row['noQC_noFill_noCorr_n_valid'])} / {int(row['n_total'])} | "
            f"{_fmt(row['noQC_noFill_noCorr_coverage_pct'])}% | "
            f"{_fmt(row['noQC_noFill_noCorr_RMSSD_MAE'])} ms | {_fmt(row['noQC_noFill_noCorr_RMSSD_R'], 3)} | "
            f"{_fmt(row['noQC_noFill_noCorr_RMSSD_bias'])} ms | "
            f"{_fmt(row['noQC_noFill_noCorr_SDNN_MAE'])} ms | {_fmt(row['noQC_noFill_noCorr_SDNN_R'], 3)} | "
            f"`{row['peak_method']}` |"
        )

    lines.extend([
        "",
        "## Participant Variability",
        "",
        "| Device | Channel | Participant RMSSD R median [IQR] | Participant coverage median [min, max] |",
        "|---|---|---:|---:|",
    ])
    for _, row in variability.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | "
            f"{_fmt(row['participant_R_median'], 3)} [{_fmt(row['participant_R_q25'], 3)}, {_fmt(row['participant_R_q75'], 3)}] | "
            f"{_fmt(row['participant_coverage_median'])}% [{_fmt(row['participant_coverage_min'])}, {_fmt(row['participant_coverage_max'])}] |"
        )

    lines.extend([
        "",
        "## Comparison With v1.2 noQC Training-Stride30 Ablation",
        "",
        "对比对象是现有 `v1.2 noQC Training-Stride30 Ablation`，即保留 `_fill_missing` 和 IBI correction、但关闭 QC gate 的结果。",
        "",
        "| Device | Channel | This coverage | noQC coverage | Coverage change | This RMSSD MAE | noQC RMSSD MAE | MAE change | This RMSSD R | noQC RMSSD R | R change | This SDNN MAE | noQC SDNN MAE | SDNN MAE change |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in compare.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['channel']}` | "
            f"{_fmt(row['noQC_noFill_noCorr_coverage_pct'])}% | {_fmt(row['noQC_coverage_pct'])}% | {_fmt(row['coverage_change_vs_noQC_pctpt'])} pp | "
            f"{_fmt(row['noQC_noFill_noCorr_RMSSD_MAE'])} ms | {_fmt(row['noQC_RMSSD_MAE'])} ms | {_fmt(row['rmssd_mae_change_vs_noQC_ms'])} ms | "
            f"{_fmt(row['noQC_noFill_noCorr_RMSSD_R'], 3)} | {_fmt(row['noQC_RMSSD_R'], 3)} | {_fmt(row['rmssd_r_change_vs_noQC'], 3)} | "
            f"{_fmt(row['noQC_noFill_noCorr_SDNN_MAE'])} ms | {_fmt(row['noQC_SDNN_MAE'])} ms | {_fmt(row['sdnn_mae_change_vs_noQC_ms'])} ms |"
        )

    lines.extend([
        "",
        "### Summary",
        "",
    ])
    for _, row in compare.iterrows():
        lines.append(
            f"- `{row['device']} {row['channel']}`: RMSSD MAE change `{_fmt(row['rmssd_mae_change_vs_noQC_ms'])} ms`, "
            f"R change `{_fmt(row['rmssd_r_change_vs_noQC'], 3)}`, coverage change `{_fmt(row['coverage_change_vs_noQC_pctpt'])} pp`."
        )

    lines.extend([
        "",
        "整体解释：如果本 ablation 相比现有 noQC 明显变差，说明 IBI correction 在关闭 QC gate 时仍然承担了重要的异常 IBI 缓冲作用；如果 coverage 基本不变，说明当前 `training_stride30` 的 `ppg_resampled` 输入大多已经是 finite，baseline-level `_fill_missing` 对覆盖率影响有限。",
        "",
        "## Output Files",
        "",
        "- `v1_2_noQC_noFill_noCorr_training_stride30_eval.csv`",
        "- `v1_2_noQC_noFill_noCorr_training_stride30_participant_eval.csv`",
        "- `v1_2_noQC_noFill_noCorr_training_stride30_predictions.csv`",
        "- `v1_2_noQC_noFill_noCorr_vs_noQC_training_stride30_comparison.csv`",
        "- `summary.json`",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return compare


def main() -> None:
    out_dir = _THIS_DIR / "no_fill_no_ibi_correction"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30"
    dataset_dir = config.HEURISTIC_RESULT_ROOT / dataset_name
    choices_csv = (
        config.HEURISTIC_RESULT_ROOT
        / "formal_v1_2_report_both_channels"
        / "v1_2_both_channels_frozen_choices.csv"
    )
    standard_noqc_csv = _THIS_DIR / "v1_2_noQC_training_stride30_eval.csv"
    standard_noqc_compare_csv = _THIS_DIR / "v1_2_noQC_vs_QC_training_stride30_comparison.csv"

    choices = pd.read_csv(choices_csv)
    standard_noqc = pd.read_csv(standard_noqc_csv)
    standard_noqc_compare = pd.read_csv(standard_noqc_compare_csv)
    pred = _load_predictions(dataset_dir, dataset_name, choices)
    summary, participant = _summarize(pred)

    pred.to_csv(out_dir / "v1_2_noQC_noFill_noCorr_training_stride30_predictions.csv", index=False)
    summary.to_csv(out_dir / "v1_2_noQC_noFill_noCorr_training_stride30_eval.csv", index=False)
    participant.to_csv(out_dir / "v1_2_noQC_noFill_noCorr_training_stride30_participant_eval.csv", index=False)
    compare = _write_readme(out_dir, summary, participant, standard_noqc, standard_noqc_compare, dataset_dir, choices_csv)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "policy": POLICY,
                "role": "training_stride30",
                "dataset_dir": str(dataset_dir.resolve()),
                "frozen_choices_csv": str(choices_csv.resolve()),
                "standard_noqc_csv": str(standard_noqc_csv.resolve()),
                "quality_checks_applied": False,
                "baseline_fill_missing_applied": False,
                "ibi_artifact_correction_applied": False,
                "peak_interpolation_refinement_applied": False,
                "dataset_level_ppg_resampling_applied": True,
                "reports_green_and_ir": True,
                "validity_requirement": "finite PPG and ECG HRV metric values",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {out_dir}")
    print(summary.to_string(index=False))
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()

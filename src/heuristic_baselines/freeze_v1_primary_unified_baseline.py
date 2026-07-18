"""
基于重新计算的峰值指标，冻结 v1-primary 统一启发式 baseline。

本脚本不重新计算峰值，而是读取 `evaluate_devicewise_recomputed_peaks_v1.py`
生成的 `v1_recomputed_channel_metrics.csv`。

规则：
  - 所有设备共用同一套 peak detector / bandpass / QC gate；
  - 不做跨设备融合；
  - 可以为每个设备单独选择通道，但必须处在同一套共享处理规则内；
  - 只在 training_stride30 上冻结规则，然后报告 strict_reference 结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402


METRICS = ("RMSSD", "SDNN")


GATES = {
    "gate_sqi04_ibi08_corr02_cv25_rmssd200": {
        "min_sqi": 0.4,
        "min_valid_ibi": 0.80,
        "max_correction": 0.20,
        "max_ibi_cv": 0.25,
        "max_rmssd_ms": 200.0,
    },
    "gate_sqi04_ibi08_corr03_cv30_rmssd200": {
        "min_sqi": 0.4,
        "min_valid_ibi": 0.80,
        "max_correction": 0.30,
        "max_ibi_cv": 0.30,
        "max_rmssd_ms": 200.0,
    },
    "gate_sqi05_ibi08_corr02_cv25_rmssd200": {
        "min_sqi": 0.5,
        "min_valid_ibi": 0.80,
        "max_correction": 0.20,
        "max_ibi_cv": 0.25,
        "max_rmssd_ms": 200.0,
    },
    "gate_sqi04_ibi09_corr02_cv25_rmssd200": {
        "min_sqi": 0.4,
        "min_valid_ibi": 0.90,
        "max_correction": 0.20,
        "max_ibi_cv": 0.25,
        "max_rmssd_ms": 200.0,
    },
}


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
            "MAE": np.nan,
            "RMSE": np.nan,
            "R": np.nan,
            "bias": np.nan,
            "LoA_lower": np.nan,
            "LoA_upper": np.nan,
        }
    diff = p - e
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "R": _safe_corr(p, e),
        "bias": bias,
        "LoA_lower": bias - 1.96 * sd if np.isfinite(sd) else np.nan,
        "LoA_upper": bias + 1.96 * sd if np.isfinite(sd) else np.nan,
    }


def _gate_mask(df: pd.DataFrame, gate_name: str) -> pd.Series:
    gate = GATES[gate_name]
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    ibi_cv = df["ppg_ibi_cv"].astype(float)
    return (
        (df["ppg_valid_sample_ratio"].astype(float) >= 0.90)
        & (df["ppg_sqi"].astype(float) >= gate["min_sqi"])
        & (df["ppg_valid_ibi_ratio"].astype(float) >= gate["min_valid_ibi"])
        & np.isfinite(corr)
        & (corr <= gate["max_correction"])
        & np.isfinite(ibi_cv)
        & (ibi_cv <= gate["max_ibi_cv"])
        & np.isfinite(df["ppg_rmssd_ms"].astype(float))
        & np.isfinite(df["ppg_sdnn_ms"].astype(float))
        & (df["ppg_rmssd_ms"].astype(float) <= gate["max_rmssd_ms"])
    )


def _summarize(pred: pd.DataFrame, group_cols: list[str], denominators: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom = denominators[(str(base["dataset"]), str(base["role"]), str(base["device"]))]
        for metric in METRICS:
            ppg_col = f"ppg_{metric.lower()}_ms"
            ecg_col = f"ecg_{metric.lower()}_ms"
            finite = np.isfinite(group[ppg_col].to_numpy(float)) & np.isfinite(group[ecg_col].to_numpy(float))
            stats = _agreement_stats(group.loc[finite, ppg_col].to_numpy(float), group.loc[finite, ecg_col].to_numpy(float))
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": int(denom),
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / denom) if denom else np.nan,
            })
    return pd.DataFrame(rows)


def _fixed_best_channel_predictions(df: pd.DataFrame, peak_method: str, gate_name: str, min_coverage_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gated = df[(df["peak_method"] == peak_method) & _gate_mask(df, gate_name)].copy()
    if gated.empty:
        return pd.DataFrame(), pd.DataFrame()
    denominators = {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in df.groupby(["dataset", "role", "device"], dropna=False)
    }
    train = gated[gated["role"] == "training_stride30"].copy()
    chan_summary = _summarize(
        train,
        ["dataset", "role", "device", "channel", "peak_method"],
        denominators,
    )
    rmssd = chan_summary[
        (chan_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(chan_summary["MAE"].to_numpy(float))
        & (chan_summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if rmssd.empty:
        return pd.DataFrame(), pd.DataFrame()
    chosen = (
        rmssd.sort_values(["device", "MAE", "coverage_pct"], ascending=[True, True, False])
        .groupby("device", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    if chosen["device"].nunique() < df["device"].nunique():
        return pd.DataFrame(), chosen
    selected = gated.merge(chosen[["device", "channel"]], on=["device", "channel"], how="inner").copy()
    selected["unified_strategy"] = "fixed_best_channel_per_device"
    selected["unified_method"] = f"{peak_method}__fixed_best_channel_per_device__{gate_name}"
    selected["gate"] = gate_name
    return selected, chosen


def _device_best_sqi_predictions(df: pd.DataFrame, peak_method: str, gate_name: str) -> pd.DataFrame:
    gated = df[(df["peak_method"] == peak_method) & _gate_mask(df, gate_name)].copy()
    if gated.empty:
        return gated
    sort_cols = ["dataset", "role", "participant", "window_index", "device", "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio"]
    best = gated.sort_values(sort_cols, ascending=[True, True, True, True, True, False, False, True])
    best = best.groupby(["dataset", "role", "participant", "window_index", "device"], dropna=False).head(1).copy()
    best["unified_strategy"] = "device_internal_sqi_best"
    best["unified_method"] = f"{peak_method}__device_internal_sqi_best__{gate_name}"
    best["gate"] = gate_name
    return best


def _candidate_predictions(df: pd.DataFrame, min_coverage_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    channel_choices: list[pd.DataFrame] = []
    peak_methods = sorted(str(x) for x in df["peak_method"].dropna().unique())
    for peak_method in peak_methods:
        for gate_name in GATES:
            fixed, chosen = _fixed_best_channel_predictions(df, peak_method, gate_name, min_coverage_pct)
            if not fixed.empty:
                frames.append(fixed)
            if not chosen.empty:
                cc = chosen.copy()
                cc["unified_strategy"] = "fixed_best_channel_per_device"
                cc["unified_method"] = f"{peak_method}__fixed_best_channel_per_device__{gate_name}"
                cc["gate"] = gate_name
                channel_choices.append(cc)
            sqi_best = _device_best_sqi_predictions(df, peak_method, gate_name)
            if not sqi_best.empty:
                frames.append(sqi_best)
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    choices = pd.concat(channel_choices, ignore_index=True) if channel_choices else pd.DataFrame()
    return pred, choices


def _freeze_unified(summary: pd.DataFrame, min_coverage_pct: float) -> pd.DataFrame:
    train = summary[
        (summary["role"] == "training_stride30")
        & (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= min_coverage_pct)
    ].copy()
    if train.empty:
        return train
    agg = (
        train.groupby(["unified_method", "unified_strategy", "peak_method", "gate"], dropna=False)
        .agg(
            n_devices=("device", "nunique"),
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
            total_valid=("n_valid", "sum"),
        )
        .reset_index()
    )
    required_devices = train["device"].nunique()
    agg = agg[agg["n_devices"] == required_devices].copy()
    if agg.empty:
        return agg
    return agg.sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False]).head(1).reset_index(drop=True)


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _write_readme(out_dir: Path, frozen: pd.DataFrame, frozen_eval: pd.DataFrame, channel_choices: pd.DataFrame, top: pd.DataFrame) -> None:
    lines = [
        "# v1-primary 统一启发式 Baseline",
        "",
        "本报告重新冻结 primary baseline：所有设备使用同一套 peak detector / bandpass / IBI correction / QC gate。",
        "允许每个设备在自己的 `ppg_green` / `ppg_ir` 中选择固定最佳通道，但不允许每个设备使用不同 bandpass 或不同 QC。",
        "",
        "## 冻结的统一规则",
        "",
    ]
    if frozen.empty:
        lines.append("没有找到满足覆盖率约束的统一规则。")
    else:
        row = frozen.iloc[0]
        lines.extend([
            f"- 统一方法：`{row['unified_method']}`",
            f"- 策略：`{row['unified_strategy']}`",
            f"- 峰值检测/bandpass 方法：`{row['peak_method']}`",
            f"- Gate: `{row['gate']}`",
            f"- 训练集跨设备平均 RMSSD MAE：`{_fmt(row['mean_MAE'])} ms`",
            f"- 训练集跨设备平均 R：`{_fmt(row['mean_R'], 3)}`",
            f"- 训练集平均覆盖率：`{_fmt(row['mean_coverage_pct'])}%`",
        ])

    if not channel_choices.empty and not frozen.empty:
        method = str(frozen.iloc[0]["unified_method"])
        chosen = channel_choices[channel_choices["unified_method"] == method].copy()
        if not chosen.empty:
            lines.extend([
                "",
                "## 冻结通道",
                "",
                "| 设备 | 通道 | 训练集 RMSSD MAE | 训练集 R | 训练集覆盖率 |",
                "|---|---|---:|---:|---:|",
            ])
            for _, row in chosen.sort_values("device").iterrows():
                lines.append(
                    f"| `{row['device']}` | `{row['channel']}` | {_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
                )

    lines.extend([
        "",
        "## 冻结规则评估",
        "",
        "| 数据集角色 | 设备 | 指标 | 有效数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.sort_values(["role", "device", "hrv_metric"]).iterrows():
        lines.append(
            f"| {row['role']} | `{row['device']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## Training 上排名靠前的统一候选",
        "",
        "| 方法 | 平均 MAE | 平均 R | 平均覆盖率 | 最低设备覆盖率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['unified_method']}` | {_fmt(row['mean_MAE'])} ms | {_fmt(row['mean_R'], 3)} | "
            f"{_fmt(row['mean_coverage_pct'])}% | {_fmt(row['min_coverage_pct'])}% |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `v1_primary_unified_predictions.csv` | 所有统一候选规则的窗口级预测 |",
        "| `v1_primary_unified_summary.csv` | 每个统一规则在每个设备上的指标 |",
        "| `v1_primary_unified_frozen_rule.csv` | training 上冻结出的唯一 primary rule |",
        "| `v1_primary_unified_frozen_eval.csv` | 冻结规则在 training / strict_reference 上的结果 |",
        "| `v1_primary_unified_channel_choices.csv` | fixed-best-channel 策略在 training 上选出的通道 |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 v1-primary 统一启发式 baseline。")
    parser.add_argument(
        "--channel-metrics-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_recomputed_peaks" / "v1_recomputed_channel_metrics.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_v1_primary_unified_baseline"),
    )
    parser.add_argument("--min-device-coverage-pct", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_df = pd.read_csv(args.channel_metrics_csv)
    denominators = {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in channel_df.groupby(["dataset", "role", "device"], dropna=False)
    }
    predictions, channel_choices = _candidate_predictions(channel_df, args.min_device_coverage_pct)
    summary = _summarize(
        predictions,
        ["dataset", "role", "device", "unified_method", "unified_strategy", "peak_method", "gate"],
        denominators,
    )
    frozen = _freeze_unified(summary, args.min_device_coverage_pct)
    frozen_eval = pd.DataFrame()
    if not frozen.empty:
        frozen_method = str(frozen.iloc[0]["unified_method"])
        frozen_eval = summary[summary["unified_method"] == frozen_method].copy()

    train_rmssd = summary[
        (summary["role"] == "training_stride30")
        & (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= args.min_device_coverage_pct)
    ].copy()
    top = (
        train_rmssd.groupby(["unified_method", "unified_strategy", "peak_method", "gate"], dropna=False)
        .agg(
            n_devices=("device", "nunique"),
            mean_MAE=("MAE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
    )
    required_devices = channel_df["device"].nunique()
    top = top[top["n_devices"] == required_devices].sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False]).head(10)

    predictions.to_csv(out_dir / "v1_primary_unified_predictions.csv", index=False)
    summary.to_csv(out_dir / "v1_primary_unified_summary.csv", index=False)
    frozen.to_csv(out_dir / "v1_primary_unified_frozen_rule.csv", index=False)
    frozen_eval.to_csv(out_dir / "v1_primary_unified_frozen_eval.csv", index=False)
    channel_choices.to_csv(out_dir / "v1_primary_unified_channel_choices.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "channel_metrics_csv": str(Path(args.channel_metrics_csv).resolve()),
        "out_dir": str(out_dir),
        "min_device_coverage_pct": args.min_device_coverage_pct,
        "shared_processing_required": True,
        "cross_device_fusion": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, frozen, frozen_eval, channel_choices, top)
    print(f"[saved] {out_dir}")
    print(frozen_eval.to_string(index=False))


if __name__ == "__main__":
    main()

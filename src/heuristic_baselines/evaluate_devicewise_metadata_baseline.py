"""
Evaluate device-wise metadata-level heuristic baselines.

This script uses the existing window-level pairs produced by
`evaluate_selected_existing_baseline.py`. It does not recompute PPG peaks, does
not train a model, and does not fuse information across devices.
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
from evaluate_rawaligned_hrv_baseline import _agreement_stats  # noqa: E402


METRICS = ("RMSSD", "SDNN")


def _summary(
    predictions: pd.DataFrame,
    denominators: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "role", "device", "method"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom_key = (str(base["dataset"]), str(base["role"]), str(base["device"]))
        n_total = int(denominators[denom_key])
        for metric in METRICS:
            ppg_col = f"ppg_{metric.lower()}_ms"
            ecg_col = f"ecg_{metric.lower()}_ms"
            finite = np.isfinite(group[ppg_col].to_numpy(float)) & np.isfinite(group[ecg_col].to_numpy(float))
            stats = _agreement_stats(
                group.loc[finite, ppg_col].to_numpy(float),
                group.loc[finite, ecg_col].to_numpy(float),
            )
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": n_total,
                "denominator_unit": "device_windows",
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
            })
    return pd.DataFrame(rows)


def _fixed_channels(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs[pairs["ppg_quality_flag"].astype(bool)].copy()
    df["method"] = df["channel"].astype(str) + "__ppg_quality_flag"
    return df


def _device_best_sqi(
    pairs: pd.DataFrame,
    *,
    min_sqi: float,
    min_valid_ibi: float,
    max_correction: float,
    max_motion: float,
) -> pd.DataFrame:
    df = pairs.copy()
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    mask = (
        df["ppg_quality_flag"].astype(bool)
        & (df["ppg_sqi"].astype(float) >= min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= min_valid_ibi)
        & (df["motion_fraction"].astype(float) < max_motion)
        & ((~np.isfinite(corr)) | (corr <= max_correction))
    )
    df = df[mask].copy()
    if df.empty:
        return df

    sort_cols = [
        "dataset",
        "role",
        "participant",
        "window_index",
        "device",
        "ppg_sqi",
        "ppg_valid_ibi_ratio",
        "motion_fraction",
        "ppg_ibi_correction_ratio",
    ]
    df = df.sort_values(
        sort_cols,
        ascending=[True, True, True, True, True, False, False, True, True],
    )
    best = df.groupby(["dataset", "role", "participant", "window_index", "device"], dropna=False).head(1).copy()
    best["method"] = (
        f"device_best_sqi__sqi{min_sqi:g}_ibi{min_valid_ibi:g}_corr{max_correction:g}_motion{max_motion:g}"
    )
    return best


def _build_predictions(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = [_fixed_channels(pairs)]
    sweep_frames: list[pd.DataFrame] = []
    for min_sqi in (0.0, 0.4, 0.5, 0.6, 0.7):
        for min_valid_ibi in (0.8, 0.9, 0.95):
            for max_correction in (0.2, 0.3, 0.5):
                for max_motion in (0.2, 0.5, 1.01):
                    pred = _device_best_sqi(
                        pairs,
                        min_sqi=min_sqi,
                        min_valid_ibi=min_valid_ibi,
                        max_correction=max_correction,
                        max_motion=max_motion,
                    )
                    if not pred.empty:
                        sweep_frames.append(pred)
    sweep = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame()
    if not sweep.empty:
        frames.append(sweep)
    return pd.concat(frames, ignore_index=True), sweep


def _best_by_device(summary: pd.DataFrame) -> pd.DataFrame:
    rmssd = summary[
        (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= 20.0)
    ].copy()
    if rmssd.empty:
        return rmssd
    for col in ("MAE", "RMSE", "R", "bias", "coverage_pct"):
        rmssd[f"{col}_rounded"] = rmssd[col].astype(float).round(6)
    rmssd = rmssd.drop_duplicates(
        subset=[
            "dataset",
            "role",
            "device",
            "MAE_rounded",
            "RMSE_rounded",
            "R_rounded",
            "bias_rounded",
            "coverage_pct_rounded",
        ],
        keep="first",
    )
    return (
        rmssd.sort_values(["dataset", "device", "MAE", "coverage_pct"], ascending=[True, True, True, False])
        .groupby(["dataset", "role", "device"], dropna=False)
        .head(5)
        .reset_index(drop=True)
    )


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _write_readme(out_dir: Path, summary: pd.DataFrame, best: pd.DataFrame) -> None:
    lines = [
        "# 每设备独立 Metadata-level Baseline 评估",
        "",
        "本报告只评估 Earring、Ring、Watch 各自独立的 baseline。所有方法都只使用同一个设备内部的 green/IR 信息，不做跨设备 median fusion，也不做跨设备最佳选择。",
        "",
        "## 方法",
        "",
        "- 固定通道：`ppg_green__ppg_quality_flag`、`ppg_ir__ppg_quality_flag`。",
        "- 设备内部质量择优：`device_best_sqi`，每个 window、每个 device 只在该 device 的 green/IR 中选择 SQI 更高且通过阈值的一路。",
        "- sweep 条件包括 SQI、IBI valid ratio、IBI correction ratio 和 motion fraction。",
        "- coverage 分母是该 dataset 中对应 device 的全部窗口数，不是设备-通道行数。",
        "- 本脚本不重新检测 PPG peaks，不训练模型，不使用 ECG label 选通道。",
        "",
        "## 每设备 Top 方法（RMSSD，coverage >= 20%）",
        "",
        "| 数据集 | 设备 | 方法 | 有效配对数 | 覆盖率 | RMSSD MAE | RMSSD RMSE | RMSSD R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['device']}` | `{row['method']}` | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## 固定通道对照",
        "",
        "| 数据集 | 设备 | 方法 | 指标 | 有效配对数 | 覆盖率 | MAE | RMSE | R |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    fixed = summary[summary["method"].isin({"ppg_green__ppg_quality_flag", "ppg_ir__ppg_quality_flag"})]
    fixed = fixed.sort_values(["dataset", "device", "method", "hrv_metric"])
    for _, row in fixed.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['device']}` | `{row['method']}` | {row['hrv_metric']} | "
            f"{int(row['n_valid'])} | {_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | "
            f"{_fmt(row['RMSE'])} ms | {_fmt(row['R'], 3)} |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `devicewise_method_predictions.csv` | 每设备独立方法的 window-level 预测结果 |",
        "| `devicewise_method_summary.csv` | 每设备/方法的 MAE/RMSE/R/Bias/coverage 汇总 |",
        "| `devicewise_best_rmssd_methods.csv` | 每个 dataset/device 的 RMSSD top 方法 |",
        "| `summary.json` | 机器可读运行配置 |",
        "",
        "## 解释",
        "",
        "这个评估更适合作为你的 primary heuristic baseline，因为它回答的是：在不借助其他设备的情况下，每个 wearable site 自己能做到什么程度。",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate device-wise metadata-level baselines.")
    parser.add_argument(
        "--pairs-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "selected_p1_p3_p5_existing_baseline_eval" / "selected_window_level_hrv_pairs.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "selected_p1_p3_p5_devicewise_metadata_baseline"),
    )
    args = parser.parse_args()

    pairs_csv = Path(args.pairs_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_csv)
    predictions, sweep = _build_predictions(pairs)
    denominators = {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in pairs.groupby(["dataset", "role", "device"], dropna=False)
    }
    summary = _summary(predictions, denominators)
    best = _best_by_device(summary)

    predictions.to_csv(out_dir / "devicewise_method_predictions.csv", index=False)
    summary.to_csv(out_dir / "devicewise_method_summary.csv", index=False)
    best.to_csv(out_dir / "devicewise_best_rmssd_methods.csv", index=False)
    run_summary = {
        "pairs_csv": str(pairs_csv),
        "out_dir": str(out_dir),
        "n_input_rows": int(len(pairs)),
        "n_prediction_rows": int(len(predictions)),
        "n_sweep_prediction_rows": int(len(sweep)) if not sweep.empty else 0,
        "denominator_unit": "device_windows",
        "denominators": {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in denominators.items()},
        "cross_device_fusion": False,
        "selection_features": ["ppg_quality_flag", "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio", "motion_fraction"],
    }
    (out_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, summary, best)

    print(f"[saved] {out_dir}")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()

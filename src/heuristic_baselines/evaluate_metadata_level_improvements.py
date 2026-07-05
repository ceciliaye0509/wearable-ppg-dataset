"""
Evaluate metadata-level heuristic improvements using existing NPZ baseline fields.

This script assumes `evaluate_selected_existing_baseline.py` has already
produced `selected_window_level_hrv_pairs.csv`. It does not recompute PPG peaks
and does not train any model.
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


def _finite_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    return np.isfinite(df[f"ppg_{metric.lower()}_ms"].to_numpy(float)) & np.isfinite(
        df[f"ecg_{metric.lower()}_ms"].to_numpy(float)
    )


def _summarize_prediction_table(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    window_denominators: dict[tuple[str, str], int],
    row_denominators: dict[tuple[str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        dataset_role = (str(base["dataset"]), str(base["role"]))
        method = str(base.get("method", ""))
        if method.startswith("pooled_all_channels"):
            n_total = int(row_denominators[dataset_role])
            denominator_unit = "device_channel_rows"
        else:
            n_total = int(window_denominators[dataset_role])
            denominator_unit = "windows"
        for metric in METRICS:
            ppg_col = f"ppg_{metric.lower()}_ms"
            ecg_col = f"ecg_{metric.lower()}_ms"
            finite = np.isfinite(group[ppg_col].to_numpy(float)) & np.isfinite(group[ecg_col].to_numpy(float))
            stats = _agreement_stats(group.loc[finite, ppg_col].to_numpy(float), group.loc[finite, ecg_col].to_numpy(float))
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": n_total,
                "denominator_unit": denominator_unit,
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / n_total) if n_total else np.nan,
            })
    return pd.DataFrame(rows)


def _pooled_all_channels(pairs: pd.DataFrame, qc_mode: str) -> pd.DataFrame:
    df = pairs.copy()
    if qc_mode == "ppg_quality_flag":
        df = df[df["ppg_quality_flag"].astype(bool)].copy()
    df["method"] = f"pooled_all_channels__{qc_mode}"
    return df


def _fixed_channel(pairs: pd.DataFrame, qc_mode: str) -> pd.DataFrame:
    df = pairs.copy()
    if qc_mode == "ppg_quality_flag":
        df = df[df["ppg_quality_flag"].astype(bool)].copy()
    df["method"] = "fixed_channel"
    return df


def _quality_best_sqi(pairs: pd.DataFrame, *, min_sqi: float, min_valid_ibi: float, max_correction: float, max_motion: float) -> pd.DataFrame:
    df = pairs.copy()
    mask = (
        df["ppg_quality_flag"].astype(bool)
        & (df["ppg_sqi"].astype(float) >= min_sqi)
        & (df["ppg_valid_ibi_ratio"].astype(float) >= min_valid_ibi)
        & (df["motion_fraction"].astype(float) < max_motion)
    )
    corr = df["ppg_ibi_correction_ratio"].astype(float)
    mask &= (~np.isfinite(corr)) | (corr <= max_correction)
    df = df[mask].copy()
    if df.empty:
        return df

    sort_cols = ["dataset", "role", "participant", "window_index", "ppg_sqi", "ppg_valid_ibi_ratio", "motion_fraction"]
    df = df.sort_values(sort_cols, ascending=[True, True, True, True, False, False, True])
    best = df.groupby(["dataset", "role", "participant", "window_index"], dropna=False).head(1).copy()
    best["method"] = (
        f"quality_best_sqi__sqi{min_sqi:g}_ibi{min_valid_ibi:g}_corr{max_correction:g}_motion{max_motion:g}"
    )
    return best


def _median_fusion(pairs: pd.DataFrame, *, min_sqi: float, min_valid_ibi: float, max_correction: float, max_motion: float) -> pd.DataFrame:
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

    group_cols = ["dataset", "role", "participant", "window_index"]
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["device"] = "FUSION"
        row["channel"] = "median_valid_channels"
        row["method"] = (
            f"median_fusion__sqi{min_sqi:g}_ibi{min_valid_ibi:g}_corr{max_correction:g}_motion{max_motion:g}"
        )
        row["n_channels_used"] = int(len(group))
        row["ppg_sqi"] = float(np.nanmedian(group["ppg_sqi"].to_numpy(float)))
        row["motion_fraction"] = float(np.nanmedian(group["motion_fraction"].to_numpy(float)))
        row["ppg_valid_ibi_ratio"] = float(np.nanmedian(group["ppg_valid_ibi_ratio"].to_numpy(float)))
        row["ppg_ibi_correction_ratio"] = float(np.nanmedian(group["ppg_ibi_correction_ratio"].to_numpy(float)))
        for metric in METRICS:
            row[f"ecg_{metric.lower()}_ms"] = float(group[f"ecg_{metric.lower()}_ms"].iloc[0])
            row[f"ppg_{metric.lower()}_ms"] = float(np.nanmedian(group[f"ppg_{metric.lower()}_ms"].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_method_predictions(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_frames: list[pd.DataFrame] = [
        _pooled_all_channels(pairs, "finite_only"),
        _pooled_all_channels(pairs, "ppg_quality_flag"),
    ]

    fixed = _fixed_channel(pairs, "ppg_quality_flag")
    fixed["method"] = fixed["device"].astype(str) + "_" + fixed["channel"].astype(str) + "__ppg_quality_flag"
    method_frames.append(fixed)

    sweep_rows: list[pd.DataFrame] = []
    for min_sqi in (0.0, 0.4, 0.5, 0.6, 0.7):
        for min_valid_ibi in (0.8, 0.9, 0.95):
            for max_correction in (0.2, 0.3, 0.5):
                for max_motion in (0.2, 0.5, 1.01):
                    best = _quality_best_sqi(
                        pairs,
                        min_sqi=min_sqi,
                        min_valid_ibi=min_valid_ibi,
                        max_correction=max_correction,
                        max_motion=max_motion,
                    )
                    fusion = _median_fusion(
                        pairs,
                        min_sqi=min_sqi,
                        min_valid_ibi=min_valid_ibi,
                        max_correction=max_correction,
                        max_motion=max_motion,
                    )
                    if not best.empty:
                        sweep_rows.append(best)
                    if not fusion.empty:
                        sweep_rows.append(fusion)

    sweep_predictions = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    if not sweep_predictions.empty:
        method_frames.append(sweep_predictions)
    return pd.concat(method_frames, ignore_index=True), sweep_predictions


def _best_methods(summary: pd.DataFrame) -> pd.DataFrame:
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
            "MAE_rounded",
            "RMSE_rounded",
            "R_rounded",
            "bias_rounded",
            "coverage_pct_rounded",
        ],
        keep="first",
    )
    return (
        rmssd.sort_values(["dataset", "MAE", "coverage_pct"], ascending=[True, True, False])
        .groupby(["dataset", "role"], dropna=False)
        .head(10)
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
        "# Metadata-level Baseline 改进评估",
        "",
        "本报告只使用现有 rawaligned NPZ 中已经保存的 PPG-derived HRV metadata，不重新检测 PPG peaks，也不训练任何模型。",
        "",
        "## 评估方法",
        "",
        "- `pooled_all_channels__finite_only`：所有设备/通道全部合并，只要求 PPG/ECG HRV 为有限值。",
        "- `pooled_all_channels__ppg_quality_flag`：所有设备/通道合并，但只保留 `ppg_quality_flag=True`。",
        "- 固定设备/通道：例如 `Earring_ppg_ir__ppg_quality_flag`。",
        "- `quality_best_sqi`：每个 window 从通过 QC 的设备/通道中选择 SQI 最高的一路。",
        "- `median_fusion`：每个 window 对所有通过 QC 的设备/通道 HRV 做 median fusion。",
        "- sweep 条件包括 SQI、IBI valid ratio、IBI correction ratio 和 motion fraction。",
        "- coverage 的分母：`pooled_all_channels` 使用设备-通道行数；固定通道、selection、fusion 使用原始窗口数。",
        "",
        "## Top 方法（RMSSD，coverage >= 20%）",
        "",
        "| 数据集 | 方法 | 分母 | 有效配对数 | 覆盖率 | RMSSD MAE | RMSSD RMSE | RMSSD R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['method']}` | {row['denominator_unit']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## Baseline 对照",
        "",
        "| 数据集 | 方法 | 指标 | 分母 | 有效配对数 | 覆盖率 | MAE | RMSE | R |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    base_methods = {"pooled_all_channels__finite_only", "pooled_all_channels__ppg_quality_flag"}
    base = summary[summary["method"].isin(base_methods)].sort_values(["dataset", "method", "hrv_metric"])
    for _, row in base.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['method']}` | {row['hrv_metric']} | {row['denominator_unit']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | {_fmt(row['R'], 3)} |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `metadata_method_predictions.csv` | 每种 metadata-level 方法的 window-level 预测结果 |",
        "| `metadata_method_summary.csv` | 每种方法的 MAE/RMSE/R/Bias/coverage 汇总 |",
        "| `metadata_best_rmssd_methods.csv` | 每个数据集 RMSSD top 方法，要求 coverage >= 20% |",
        "| `summary.json` | 机器可读的运行配置 |",
        "",
        "## 注意事项",
        "",
        "- 这是 P1/P3/P5 的快速开发评估，不是最终全 cohort 结果。",
        "- `quality_best_sqi` 和 `median_fusion` 只使用 PPG 自身 metadata 做选择，不使用 ECG label 选通道。",
        "- 如果用这些 sweep 结果决定最终阈值，应当在 development set 冻结规则，再到 strict reference 或 held-out subjects 上评估。",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metadata-level heuristic improvements.")
    parser.add_argument(
        "--pairs-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "selected_p1_p3_p5_existing_baseline_eval" / "selected_window_level_hrv_pairs.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "selected_p1_p3_p5_metadata_improvements"),
    )
    args = parser.parse_args()

    pairs_csv = Path(args.pairs_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_csv)
    predictions, sweep_predictions = _build_method_predictions(pairs)
    window_denominators = {
        (str(dataset), str(role)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role), group in pairs.groupby(["dataset", "role"], dropna=False)
    }
    row_denominators = {
        (str(dataset), str(role)): int(len(group))
        for (dataset, role), group in pairs.groupby(["dataset", "role"], dropna=False)
    }
    summary = _summarize_prediction_table(
        predictions,
        ["dataset", "role", "method"],
        window_denominators=window_denominators,
        row_denominators=row_denominators,
    )
    best = _best_methods(summary)

    predictions.to_csv(out_dir / "metadata_method_predictions.csv", index=False)
    summary.to_csv(out_dir / "metadata_method_summary.csv", index=False)
    best.to_csv(out_dir / "metadata_best_rmssd_methods.csv", index=False)
    run_summary = {
        "pairs_csv": str(pairs_csv),
        "out_dir": str(out_dir),
        "n_input_rows": int(len(pairs)),
        "n_prediction_rows": int(len(predictions)),
        "n_sweep_prediction_rows": int(len(sweep_predictions)) if not sweep_predictions.empty else 0,
        "window_denominators": {f"{k[0]}::{k[1]}": v for k, v in window_denominators.items()},
        "row_denominators": {f"{k[0]}::{k[1]}": v for k, v in row_denominators.items()},
        "selection_features": ["ppg_quality_flag", "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio", "motion_fraction"],
    }
    (out_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, summary, best)

    print(f"[saved] {out_dir}")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()

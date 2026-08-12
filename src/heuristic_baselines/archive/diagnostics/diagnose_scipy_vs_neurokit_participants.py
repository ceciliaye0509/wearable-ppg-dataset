"""
比较 primary SciPy baseline 与最佳 NeuroKit unified candidate 的参与者级误差。

本脚本只读取已经生成的 channel metrics 和 formal freeze 输出，不重新计算峰值。
用途是回答：NeuroKit 加入候选池后，错误和覆盖率在 participant/device 层面如何变化。
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
import formal_freeze_v1_1_unified_baseline as formal_v11  # noqa: E402


METRICS = ("RMSSD", "SDNN")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _agreement_stats(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ppg) & np.isfinite(ecg)
    p = np.asarray(ppg, dtype=np.float64)[mask]
    e = np.asarray(ecg, dtype=np.float64)[mask]
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


def _best_neurokit_unified(candidate_summary: pd.DataFrame, min_device_coverage_pct: float) -> pd.Series:
    rmssd = candidate_summary[
        (candidate_summary["role"] == "training_stride30")
        & (candidate_summary["hrv_metric"] == "RMSSD")
        & candidate_summary["peak_method"].astype(str).str.startswith("nk_")
        & np.isfinite(candidate_summary["MAE"].to_numpy(float))
        & (candidate_summary["coverage_pct"].astype(float) >= min_device_coverage_pct)
    ].copy()
    agg = (
        rmssd.groupby(["unified_method", "unified_strategy", "peak_method", "gate"], dropna=False)
        .agg(
            n_devices=("device", "nunique"),
            mean_MAE=("MAE", "mean"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .reset_index()
    )
    required_devices = candidate_summary["device"].nunique()
    agg = agg[agg["n_devices"] == required_devices].copy()
    if agg.empty:
        raise RuntimeError("没有找到满足覆盖率约束的 NeuroKit unified candidate。")
    return agg.sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False]).iloc[0]


def _select_predictions(
    channel_df: pd.DataFrame,
    selected: pd.Series,
    channel_choices: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    peak_method = str(selected["peak_method"])
    gate = str(selected["gate"])
    strategy = str(selected["unified_strategy"])
    method = str(selected["unified_method"])
    gated = channel_df[(channel_df["peak_method"] == peak_method) & formal_v11._gate_mask(channel_df, gate)].copy()
    if strategy == "fixed_best_channel_per_device":
        chosen = channel_choices[channel_choices["unified_method"] == method][["device", "channel"]].drop_duplicates()
        if chosen.empty:
            raise RuntimeError(f"找不到 fixed channel choices: {method}")
        pred = gated.merge(chosen, on=["device", "channel"], how="inner").copy()
    elif strategy == "device_internal_sqi_best":
        sort_cols = ["dataset", "role", "participant", "window_index", "device", "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio"]
        pred = gated.sort_values(sort_cols, ascending=[True, True, True, True, True, False, False, True])
        pred = pred.groupby(["dataset", "role", "participant", "window_index", "device"], dropna=False).head(1).copy()
    else:
        raise RuntimeError(f"未知 strategy: {strategy}")
    pred["comparison_label"] = label
    pred["comparison_method"] = method
    pred["comparison_peak_method"] = peak_method
    pred["comparison_gate"] = gate
    pred["comparison_strategy"] = strategy
    return pred


def _participant_metrics(pred: pd.DataFrame, channel_df: pd.DataFrame) -> pd.DataFrame:
    denominators = {
        (str(dataset), str(role), str(participant), str(device)): int(group[["window_index"]].drop_duplicates().shape[0])
        for (dataset, role, participant, device), group in channel_df.groupby(["dataset", "role", "participant", "device"], dropna=False)
    }
    rows: list[dict[str, object]] = []
    group_cols = ["comparison_label", "comparison_method", "dataset", "role", "participant", "device"]
    for keys, group in pred.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        denom = denominators[(str(base["dataset"]), str(base["role"]), str(base["participant"]), str(base["device"]))]
        for metric in METRICS:
            ppg_col = f"ppg_{metric.lower()}_ms"
            ecg_col = f"ecg_{metric.lower()}_ms"
            stats = _agreement_stats(group[ppg_col].to_numpy(float), group[ecg_col].to_numpy(float))
            rows.append({
                **base,
                "hrv_metric": metric,
                "n_total": denom,
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / denom) if denom else np.nan,
            })
    return pd.DataFrame(rows)


def _qc_stage_summary(
    channel_df: pd.DataFrame,
    selected: pd.Series,
    channel_choices: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    peak_method = str(selected["peak_method"])
    gate_name = str(selected["gate"])
    method = str(selected["unified_method"])
    strategy = str(selected["unified_strategy"])
    gate = formal_v11.freeze_base.GATES[gate_name]
    rows_df = channel_df[channel_df["peak_method"] == peak_method].copy()
    if strategy == "fixed_best_channel_per_device":
        chosen = channel_choices[channel_choices["unified_method"] == method][["device", "channel"]].drop_duplicates()
        rows_df = rows_df.merge(chosen, on=["device", "channel"], how="inner").copy()

    rows: list[dict[str, object]] = []
    for keys, group in rows_df.groupby(["dataset", "role", "participant", "device"], dropna=False):
        dataset, role, participant, device = keys
        total = int(group[["window_index"]].drop_duplicates().shape[0])
        corr = group["ppg_ibi_correction_ratio"].astype(float)
        ibi_cv = group["ppg_ibi_cv"].astype(float)
        masks = {
            "valid_sample_ratio": group["ppg_valid_sample_ratio"].astype(float) >= 0.90,
            "sqi": group["ppg_sqi"].astype(float) >= gate["min_sqi"],
            "valid_ibi": group["ppg_valid_ibi_ratio"].astype(float) >= gate["min_valid_ibi"],
            "correction_ratio": np.isfinite(corr) & (corr <= gate["max_correction"]),
            "ibi_cv": np.isfinite(ibi_cv) & (ibi_cv <= gate["max_ibi_cv"]),
            "finite_hrv": np.isfinite(group["ppg_rmssd_ms"].astype(float)) & np.isfinite(group["ppg_sdnn_ms"].astype(float)),
            "rmssd_limit": group["ppg_rmssd_ms"].astype(float) <= gate["max_rmssd_ms"],
        }
        all_pass = np.logical_and.reduce([m.to_numpy(bool) if hasattr(m, "to_numpy") else np.asarray(m, dtype=bool) for m in masks.values()])
        row: dict[str, object] = {
            "comparison_label": label,
            "comparison_method": method,
            "dataset": dataset,
            "role": role,
            "participant": participant,
            "device": device,
            "n_total": total,
            "n_pass_all": int(np.sum(all_pass)),
            "pass_all_pct": float(100.0 * np.sum(all_pass) / total) if total else np.nan,
        }
        for name, mask in masks.items():
            arr = mask.to_numpy(bool) if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool)
            row[f"fail_{name}_pct"] = float(100.0 * np.sum(~arr) / total) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _write_readme(out_dir: Path, selected: pd.DataFrame, participant_metrics: pd.DataFrame, qc: pd.DataFrame) -> None:
    strict_rmssd = participant_metrics[
        (participant_metrics["role"] == "strict_reference")
        & (participant_metrics["hrv_metric"] == "RMSSD")
    ].copy()
    high_level = (
        strict_rmssd.groupby(["comparison_label", "device"], dropna=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_R=("R", "mean"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .reset_index()
    )
    worst = strict_rmssd.sort_values("MAE", ascending=False).groupby(["comparison_label", "device"], dropna=False).head(3)

    lines = [
        "# SciPy vs NeuroKit 参与者级诊断",
        "",
        "本报告比较当前 primary SciPy unified baseline 与 training 上最佳 NeuroKit unified candidate。",
        "",
        "## 比较对象",
        "",
        "| 标签 | unified method | strategy | peak method | gate |",
        "|---|---|---|---|---|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| `{row['comparison_label']}` | `{row['unified_method']}` | `{row['unified_strategy']}` | "
            f"`{row['peak_method']}` | `{row['gate']}` |"
        )

    lines.extend([
        "",
        "## strict_reference 参与者级均值",
        "",
        "| 标签 | 设备 | mean MAE | median MAE | mean R | mean coverage |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in high_level.sort_values(["device", "comparison_label"]).iterrows():
        lines.append(
            f"| `{row['comparison_label']}` | `{row['device']}` | {_fmt(row['mean_MAE'])} ms | "
            f"{_fmt(row['median_MAE'])} ms | {_fmt(row['mean_R'], 3)} | {_fmt(row['mean_coverage_pct'])}% |"
        )

    lines.extend([
        "",
        "## strict_reference 每设备 MAE 最高的 participant",
        "",
        "| 标签 | 设备 | participant | MAE | R | coverage |",
        "|---|---|---|---:|---:|---:|",
    ])
    for _, row in worst.sort_values(["comparison_label", "device", "MAE"], ascending=[True, True, False]).iterrows():
        lines.append(
            f"| `{row['comparison_label']}` | `{row['device']}` | `{row['participant']}` | "
            f"{_fmt(row['MAE'])} ms | {_fmt(row['R'], 3)} | {_fmt(row['coverage_pct'])}% |"
        )

    qc_strict = qc[qc["role"] == "strict_reference"].copy()
    qc_device = (
        qc_strict.groupby(["comparison_label", "device"], dropna=False)
        .agg(
            pass_all_pct=("pass_all_pct", "mean"),
            fail_sqi_pct=("fail_sqi_pct", "mean"),
            fail_valid_ibi_pct=("fail_valid_ibi_pct", "mean"),
            fail_correction_ratio_pct=("fail_correction_ratio_pct", "mean"),
            fail_ibi_cv_pct=("fail_ibi_cv_pct", "mean"),
        )
        .reset_index()
    )
    lines.extend([
        "",
        "## strict_reference 平均 QC 失败原因",
        "",
        "| 标签 | 设备 | pass all | fail SQI | fail valid IBI | fail correction | fail IBI CV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in qc_device.sort_values(["device", "comparison_label"]).iterrows():
        lines.append(
            f"| `{row['comparison_label']}` | `{row['device']}` | {_fmt(row['pass_all_pct'])}% | "
            f"{_fmt(row['fail_sqi_pct'])}% | {_fmt(row['fail_valid_ibi_pct'])}% | "
            f"{_fmt(row['fail_correction_ratio_pct'])}% | {_fmt(row['fail_ibi_cv_pct'])}% |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `selected_methods.csv` | 本次比较选中的 SciPy / NeuroKit 方法 |",
        "| `participant_device_metric_summary.csv` | 每个 participant/device 的 RMSSD 和 SDNN 误差 |",
        "| `participant_device_qc_failure_summary.csv` | 每个 participant/device 的 QC 失败比例 |",
        "| `summary.json` | 机器可读配置 |",
        "",
        "## 解读提醒",
        "",
        "- 这里的 NeuroKit candidate 是 training 上最佳的 unified NeuroKit candidate，不是用 strict_reference 选出来的。",
        "- 如果 NeuroKit 在某些设备上 coverage 更好但 MAE/R 更差，不能直接替换 primary baseline；需要看论文对 baseline 的目标是低误差、覆盖率，还是二者折中。",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SciPy vs NeuroKit participant-level error diagnosis。")
    parser.add_argument(
        "--channel-metrics-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_1_plus_neurokit" / "v1_1_plus_neurokit_channel_metrics.csv"),
    )
    parser.add_argument(
        "--formal-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "formal_v1_1_plus_neurokit_unified_baseline"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "diagnosis_scipy_vs_neurokit_participants"),
    )
    parser.add_argument("--min-device-coverage-pct", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formal_dir = Path(args.formal_dir).resolve()

    channel_df = pd.read_csv(Path(args.channel_metrics_csv).resolve())
    frozen = pd.read_csv(formal_dir / "formal_frozen_rule.csv")
    candidate_summary = pd.read_csv(formal_dir / "formal_training_candidate_summary.csv")
    channel_choices = pd.read_csv(formal_dir / "formal_development_channel_choices.csv")

    primary = frozen.iloc[0].copy()
    best_neurokit = _best_neurokit_unified(candidate_summary, args.min_device_coverage_pct)
    selected = pd.DataFrame([
        {"comparison_label": "primary_scipy", **primary.to_dict()},
        {"comparison_label": "best_neurokit_unified", **best_neurokit.to_dict()},
    ])

    predictions = pd.concat([
        _select_predictions(channel_df, primary, channel_choices, "primary_scipy"),
        _select_predictions(channel_df, best_neurokit, channel_choices, "best_neurokit_unified"),
    ], ignore_index=True)
    metrics = _participant_metrics(predictions, channel_df)
    qc = pd.concat([
        _qc_stage_summary(channel_df, primary, channel_choices, "primary_scipy"),
        _qc_stage_summary(channel_df, best_neurokit, channel_choices, "best_neurokit_unified"),
    ], ignore_index=True)

    selected.to_csv(out_dir / "selected_methods.csv", index=False)
    metrics.to_csv(out_dir / "participant_device_metric_summary.csv", index=False)
    qc.to_csv(out_dir / "participant_device_qc_failure_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "channel_metrics_csv": str(Path(args.channel_metrics_csv).resolve()),
        "formal_dir": str(formal_dir),
        "out_dir": str(out_dir),
        "min_device_coverage_pct": args.min_device_coverage_pct,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, selected, metrics, qc)
    print(f"[saved] {out_dir}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()

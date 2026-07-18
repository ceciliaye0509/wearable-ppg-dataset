"""
正式冻结 v1.1 统一规则的 device-wise heuristic baseline。

与探索性 freeze 脚本不同，本脚本只在 training_stride30 上选择规则；
strict_reference 只在规则冻结后用于最终评估。本脚本不会写出 strict_reference
上的候选扫描，避免把正式评估集再次用于调参。

v1.1 相比 v1 扩展了 prominence、IBI correction threshold、peak refinement
和一个 coverage-friendly QC gate。v1 已冻结结果不会被覆盖。
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
import freeze_v1_primary_unified_baseline as freeze_base  # noqa: E402


freeze_base.GATES = {
    **freeze_base.GATES,
    "gate_sqi035_ibi08_corr03_cv35_rmssd200": {
        "min_sqi": 0.35,
        "min_valid_ibi": 0.80,
        "max_correction": 0.30,
        "max_ibi_cv": 0.35,
        "max_rmssd_ms": 200.0,
    },
}

_candidate_predictions = freeze_base._candidate_predictions
_fmt = freeze_base._fmt
_freeze_unified = freeze_base._freeze_unified
_gate_mask = freeze_base._gate_mask
_summarize = freeze_base._summarize


def _denominators(df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    return {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in df.groupby(["dataset", "role", "device"], dropna=False)
    }


def _strict_frozen_predictions(
    strict_df: pd.DataFrame,
    frozen_row: pd.Series,
    channel_choices: pd.DataFrame,
) -> pd.DataFrame:
    peak_method = str(frozen_row["peak_method"])
    gate = str(frozen_row["gate"])
    strategy = str(frozen_row["unified_strategy"])
    unified_method = str(frozen_row["unified_method"])

    gated = strict_df[(strict_df["peak_method"] == peak_method) & _gate_mask(strict_df, gate)].copy()
    if gated.empty:
        return gated

    if strategy == "fixed_best_channel_per_device":
        chosen = channel_choices[channel_choices["unified_method"] == unified_method][["device", "channel"]].drop_duplicates()
        selected = gated.merge(chosen, on=["device", "channel"], how="inner").copy()
    elif strategy == "device_internal_sqi_best":
        sort_cols = ["dataset", "role", "participant", "window_index", "device", "ppg_sqi", "ppg_valid_ibi_ratio", "ppg_ibi_correction_ratio"]
        selected = gated.sort_values(sort_cols, ascending=[True, True, True, True, True, False, False, True])
        selected = selected.groupby(["dataset", "role", "participant", "window_index", "device"], dropna=False).head(1).copy()
    else:
        raise ValueError(f"Unknown frozen strategy: {strategy}")

    selected["unified_strategy"] = strategy
    selected["unified_method"] = unified_method
    selected["gate"] = gate
    return selected


def _write_readme(
    out_dir: Path,
    frozen: pd.DataFrame,
    channel_choices: pd.DataFrame,
    training_eval: pd.DataFrame,
    strict_eval: pd.DataFrame,
    top_training: pd.DataFrame,
) -> None:
    lines = [
        "# 正式 v1.1 统一 Heuristic Baseline",
        "",
        "这是实验记录，不是论文正文。",
        "",
        "## 实验边界",
        "",
        "- 参数选择只使用 `training_stride30`。",
        "- `strict_reference` 只在规则冻结后使用。",
        "- 正式输出不写出 `strict_reference` 候选扫描。",
        "- 不做跨设备 fusion，也不做跨设备 best selection。",
        "- 允许每个设备选择自己的固定通道，但必须在同一套共享处理规则之内。",
        "- v1.1 不覆盖 v1；它是新增候选集合下的重新冻结结果。",
        "",
        "## 冻结规则",
        "",
    ]
    if frozen.empty:
        lines.append("没有候选规则满足覆盖率约束。")
    else:
        row = frozen.iloc[0]
        lines.extend([
            f"- 统一方法：`{row['unified_method']}`",
            f"- 策略：`{row['unified_strategy']}`",
            f"- Peak 方法：`{row['peak_method']}`",
            f"- Gate：`{row['gate']}`",
            f"- 训练集平均 RMSSD MAE：`{_fmt(row['mean_MAE'])} ms`",
            f"- 训练集平均 R：`{_fmt(row['mean_R'], 3)}`",
            f"- 训练集平均覆盖率：`{_fmt(row['mean_coverage_pct'])}%`",
        ])

    if not frozen.empty and not channel_choices.empty:
        method = str(frozen.iloc[0]["unified_method"])
        chosen = channel_choices[channel_choices["unified_method"] == method].copy()
        lines.extend([
            "",
            "## 开发集上选择的通道",
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
        "| 数据角色 | 设备 | 指标 | 有效数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    eval_df = pd.concat([training_eval, strict_eval], ignore_index=True)
    for _, row in eval_df.sort_values(["role", "device", "hrv_metric"]).iterrows():
        lines.append(
            f"| {row['role']} | `{row['device']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{_fmt(row['coverage_pct'])}% | {_fmt(row['MAE'])} ms | {_fmt(row['RMSE'])} ms | "
            f"{_fmt(row['R'], 3)} | {_fmt(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## 开发集上的 Top 候选",
        "",
        "| 统一方法 | 平均 MAE | 平均 R | 平均覆盖率 | 最低设备覆盖率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in top_training.iterrows():
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
        "| `formal_training_candidate_summary.csv` | 只基于开发数据的候选汇总 |",
        "| `formal_frozen_rule.csv` | 在开发数据上选出的冻结规则 |",
        "| `formal_development_channel_choices.csv` | 在开发数据上选出的设备通道 |",
        "| `formal_frozen_training_eval.csv` | 冻结规则在开发数据上的结果 |",
        "| `formal_frozen_strict_eval.csv` | 冻结规则在 strict_reference 上的结果 |",
        "| `summary.json` | 机器可读运行配置 |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="正式冻结 v1.1 统一 baseline，不写出 strict_reference 候选扫描。")
    parser.add_argument(
        "--channel-metrics-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_1_recomputed_peaks" / "v1_1_recomputed_channel_metrics.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "formal_v1_1_unified_baseline"),
    )
    parser.add_argument("--min-device-coverage-pct", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_df = pd.read_csv(args.channel_metrics_csv)
    training_df = channel_df[channel_df["role"] == "training_stride30"].copy()
    strict_df = channel_df[channel_df["role"] == "strict_reference"].copy()

    training_predictions, channel_choices = _candidate_predictions(training_df, args.min_device_coverage_pct)
    training_summary = _summarize(
        training_predictions,
        ["dataset", "role", "device", "unified_method", "unified_strategy", "peak_method", "gate"],
        _denominators(training_df),
    )
    frozen = _freeze_unified(training_summary, args.min_device_coverage_pct)
    if frozen.empty:
        frozen_method = ""
        training_eval = pd.DataFrame()
        strict_eval = pd.DataFrame()
    else:
        frozen_method = str(frozen.iloc[0]["unified_method"])
        training_eval = training_summary[training_summary["unified_method"] == frozen_method].copy()
        strict_predictions = _strict_frozen_predictions(strict_df, frozen.iloc[0], channel_choices)
        strict_eval = _summarize(
            strict_predictions,
            ["dataset", "role", "device", "unified_method", "unified_strategy", "peak_method", "gate"],
            _denominators(strict_df),
        )

    train_rmssd = training_summary[
        (training_summary["hrv_metric"] == "RMSSD")
        & np.isfinite(training_summary["MAE"].to_numpy(float))
        & (training_summary["coverage_pct"].astype(float) >= args.min_device_coverage_pct)
    ].copy()
    top_training = (
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
    required_devices = training_df["device"].nunique()
    top_training = top_training[top_training["n_devices"] == required_devices]
    top_training = top_training.sort_values(["mean_MAE", "min_coverage_pct"], ascending=[True, False]).head(10)

    training_summary.to_csv(out_dir / "formal_training_candidate_summary.csv", index=False)
    frozen.to_csv(out_dir / "formal_frozen_rule.csv", index=False)
    channel_choices.to_csv(out_dir / "formal_development_channel_choices.csv", index=False)
    training_eval.to_csv(out_dir / "formal_frozen_training_eval.csv", index=False)
    strict_eval.to_csv(out_dir / "formal_frozen_strict_eval.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "channel_metrics_csv": str(Path(args.channel_metrics_csv).resolve()),
        "out_dir": str(out_dir),
        "min_device_coverage_pct": args.min_device_coverage_pct,
        "selection_role": "training_stride30",
        "final_evaluation_role": "strict_reference",
        "strict_candidate_sweeps_written": False,
        "frozen_method": frozen_method,
        "version": "v1.1",
        "expanded_gate_names": list(freeze_base.GATES.keys()),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, frozen, channel_choices, training_eval, strict_eval, top_training)

    print(f"[saved] {out_dir}")
    print(strict_eval.to_string(index=False))


if __name__ == "__main__":
    main()

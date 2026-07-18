"""
补算 full-cohort NeuroKit candidates，并与已有 v1.1 SciPy 指标合并。

这个脚本的目的不是替代 v1/v1.1 正式结果，而是把 NeuroKit Elgendi 放进
同一套候选池，供后续 formal freeze comparison 和 per-participant 诊断使用。

为避免长任务中断后全部重跑，NeuroKit 结果会按 dataset role + participant
写入缓存 CSV；已经存在的缓存会直接复用。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
import evaluate_devicewise_recomputed_peaks_v1_1 as eval_v11  # noqa: E402


def _participants(text: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in text.split(",") if p.strip())


def _cached_role_participant(
    *,
    role: str,
    dataset_dir: Path,
    participant: str,
    cache_dir: Path,
    max_windows_per_participant: int | None,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{role}_{participant}_neurokit.csv"
    if cache_path.exists():
        print(f"[cache] {role} {participant} -> {cache_path.name}")
        return pd.read_csv(cache_path)

    df = eval_v11._load_recomputed_rows(
        dataset_dir=dataset_dir,
        dataset_name=dataset_dir.name,
        role=role,
        participants=(participant,),
        peak_methods=eval_v11.NEUROKIT_METHODS,
        max_windows_per_participant=max_windows_per_participant,
        workers=1,
    )
    df.to_csv(cache_path, index=False)
    print(f"[saved-cache] {role} {participant} rows={len(df)}")
    return df


def _write_readme(
    out_dir: Path,
    channel_df: pd.DataFrame,
    summary: pd.DataFrame,
    frozen: pd.DataFrame,
    frozen_eval: pd.DataFrame,
) -> None:
    lines = [
        "# v1.1 + NeuroKit full-cohort candidate",
        "",
        "本目录把 NeuroKit Elgendi full-cohort candidates 补算进 v1.1 候选池。",
        "",
        "## 实验边界",
        "",
        "- 不训练模型。",
        "- 不做跨设备 fusion。",
        "- 不使用 ECG label 逐窗口选通道。",
        "- NeuroKit 结果按 participant 缓存，便于中断后继续。",
        "- 本目录是 detector candidate comparison，不自动替代 `formal_v1_unified_baseline`。",
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
        "## 每设备独立冻结结果",
        "",
        "| 设备 | 冻结方法 | 训练集 RMSSD MAE | 训练集 R | 训练集覆盖率 |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in frozen.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['method']}` | {eval_v11._fmt(row['MAE'])} ms | "
            f"{eval_v11._fmt(row['R'], 3)} | {eval_v11._fmt(row['coverage_pct'])}% |"
        )

    lines.extend([
        "",
        "## 冻结策略在 training / strict_reference 上的结果",
        "",
        "| 数据集角色 | 设备 | 指标 | 有效数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in frozen_eval.sort_values(["role", "device", "hrv_metric"]).iterrows():
        lines.append(
            f"| {row['role']} | `{row['device']}` | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{eval_v11._fmt(row['coverage_pct'])}% | {eval_v11._fmt(row['MAE'])} ms | "
            f"{eval_v11._fmt(row['RMSE'])} ms | {eval_v11._fmt(row['R'], 3)} | "
            f"{eval_v11._fmt(row['bias'])} ms |"
        )

    top = summary[
        (summary["role"] == "training_stride30")
        & (summary["hrv_metric"] == "RMSSD")
        & np.isfinite(summary["MAE"].to_numpy(float))
        & (summary["coverage_pct"].astype(float) >= 20.0)
    ].copy()
    top = top.sort_values(["device", "MAE", "coverage_pct"], ascending=[True, True, False]).groupby("device", dropna=False).head(8)
    lines.extend([
        "",
        "## Training 上每设备 RMSSD Top 候选",
        "",
        "| 设备 | 方法 | 有效数 | 覆盖率 | MAE | R |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['device']}` | `{row['method']}` | {int(row['n_valid'])} | "
            f"{eval_v11._fmt(row['coverage_pct'])}% | {eval_v11._fmt(row['MAE'])} ms | "
            f"{eval_v11._fmt(row['R'], 3)} |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `v1_1_plus_neurokit_channel_metrics.csv` | v1.1 SciPy 指标 + full-cohort NeuroKit 指标 |",
        "| `v1_1_plus_neurokit_strategy_summary.csv` | 所有候选策略汇总 |",
        "| `v1_1_plus_neurokit_frozen_training_strategies.csv` | 每设备独立冻结策略 |",
        "| `v1_1_plus_neurokit_frozen_strategy_eval.csv` | 每设备独立冻结策略在 training/strict 上的结果 |",
        "| `.cache/neurokit_rows/` | 按 role + participant 缓存的 NeuroKit rows |",
        "| `summary.json` | 机器可读运行配置 |",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="补算 full-cohort NeuroKit candidates，并合并已有 v1.1 指标。")
    parser.add_argument(
        "--strict-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100"),
    )
    parser.add_argument(
        "--training-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30"),
    )
    parser.add_argument(
        "--base-channel-metrics-csv",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_1_recomputed_peaks" / "v1_1_recomputed_channel_metrics.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "all_participants_devicewise_baseline_v1_1_plus_neurokit"),
    )
    parser.add_argument("--participants", default=",".join(eval_v11.PARTICIPANTS))
    parser.add_argument("--max-windows-per-participant", type=int, default=None)
    parser.add_argument("--min-freeze-coverage-pct", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1, help="并行补算 role+participant 缓存的线程数。")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache" / "neurokit_rows"
    cache_dir.mkdir(parents=True, exist_ok=True)

    strict_dir = Path(args.strict_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    participants = _participants(args.participants)

    tasks = [
        {
            "role": role,
            "dataset_dir": dataset_dir,
            "participant": participant,
            "cache_dir": cache_dir,
            "max_windows_per_participant": args.max_windows_per_participant,
        }
        for role, dataset_dir in (("strict_reference", strict_dir), ("training_stride30", training_dir))
        for participant in participants
    ]
    frames: list[pd.DataFrame] = []
    if args.workers <= 1:
        for task in tasks:
            frames.append(_cached_role_participant(**task))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_cached_role_participant, **task) for task in tasks]
            for future in as_completed(futures):
                frames.append(future.result())

    neurokit_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    base_df = pd.read_csv(Path(args.base_channel_metrics_csv).resolve())
    channel_df = pd.concat([base_df, neurokit_df], ignore_index=True)
    channel_csv = out_dir / "v1_1_plus_neurokit_channel_metrics.csv"
    channel_df.to_csv(channel_csv, index=False)

    denominators = {
        (str(dataset), str(role), str(device)): int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        for (dataset, role, device), group in channel_df.groupby(["dataset", "role", "device"], dropna=False)
    }
    predictions = eval_v11._build_strategy_predictions(channel_df)
    summary = eval_v11._summarize(predictions, denominators)
    frozen = eval_v11._choose_frozen_strategies(summary, min_coverage_pct=args.min_freeze_coverage_pct)
    frozen_eval = eval_v11._apply_frozen(summary, frozen)

    summary.to_csv(out_dir / "v1_1_plus_neurokit_strategy_summary.csv", index=False)
    frozen.to_csv(out_dir / "v1_1_plus_neurokit_frozen_training_strategies.csv", index=False)
    frozen_eval.to_csv(out_dir / "v1_1_plus_neurokit_frozen_strategy_eval.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "strict_dir": str(strict_dir),
        "training_dir": str(training_dir),
        "base_channel_metrics_csv": str(Path(args.base_channel_metrics_csv).resolve()),
        "out_dir": str(out_dir),
        "participants": list(participants),
        "neurokit_methods": [m.__dict__ for m in eval_v11.NEUROKIT_METHODS],
        "min_freeze_coverage_pct": args.min_freeze_coverage_pct,
        "workers": args.workers,
        "cross_device_fusion": False,
        "uses_saved_ppg_metadata": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, channel_df, summary, frozen, frozen_eval)

    print(f"[saved] {out_dir}")
    print(frozen_eval.to_string(index=False))


if __name__ == "__main__":
    main()

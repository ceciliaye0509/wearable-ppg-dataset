"""
Evaluate existing PPG-derived HRV baseline metadata for selected participants.

This script does not recompute PPG peaks. It reads the already-generated
rawaligned NPZ files and compares existing `ppg_rmssd_ms` / `ppg_sdnn_ms`
against ECG-derived labels for a small participant subset.
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
QC_MODES = ("finite_only", "ppg_quality_flag")


def _participant_sort_key(pid: str) -> int:
    return int(pid[1:]) if pid.startswith("P") and pid[1:].isdigit() else 999999


def _npz_for_participant(dataset_dir: Path, participant: str) -> Path | None:
    matches = sorted(dataset_dir.glob(f"*_{participant}.npz"))
    return matches[0] if matches else None


def _load_pairs(dataset_dir: Path, dataset_name: str, role: str, participants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant in participants:
        path = _npz_for_participant(dataset_dir, participant)
        if path is None:
            print(f"[WARN] missing {dataset_name} {participant}")
            continue

        with np.load(path, allow_pickle=True) as z:
            devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
            channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
            n_windows = int(np.asarray(z["ecg_rmssd_ms"]).shape[0])
            motion = np.asarray(z["motion_fraction"], dtype=np.float64)
            ppg_quality_flag = np.asarray(z["ppg_quality_flag"])

            for wi in range(n_windows):
                for di, device in enumerate(devices):
                    for ci, channel in enumerate(channels):
                        rows.append({
                            "dataset": dataset_name,
                            "role": role,
                            "participant": participant,
                            "window_index": wi,
                            "device": device,
                            "channel": channel,
                            "t0_ms": float(np.asarray(z["t0_ms"], dtype=np.float64)[wi]) if "t0_ms" in z.files else np.nan,
                            "ppg_quality_flag": bool(ppg_quality_flag[wi, di, ci]),
                            "ppg_quality_reason": str(np.asarray(z["ppg_quality_reason"], dtype=object)[wi, di, ci])
                                if "ppg_quality_reason" in z.files else "",
                            "ppg_sqi": float(np.asarray(z["ppg_sqi"], dtype=np.float64)[wi, di, ci]),
                            "motion_fraction": float(motion[wi, di]),
                            "ppg_valid_ibi_ratio": float(np.asarray(z["ppg_valid_ibi_ratio"], dtype=np.float64)[wi, di, ci])
                                if "ppg_valid_ibi_ratio" in z.files else np.nan,
                            "ppg_ibi_correction_ratio": float(np.asarray(z["ppg_ibi_correction_ratio"], dtype=np.float64)[wi, di, ci])
                                if "ppg_ibi_correction_ratio" in z.files else np.nan,
                            "ecg_rmssd_ms": float(np.asarray(z["ecg_rmssd_ms"], dtype=np.float64)[wi]),
                            "ecg_sdnn_ms": float(np.asarray(z["ecg_sdnn_ms"], dtype=np.float64)[wi]),
                            "ppg_rmssd_ms": float(np.asarray(z["ppg_rmssd_ms"], dtype=np.float64)[wi, di, ci]),
                            "ppg_sdnn_ms": float(np.asarray(z["ppg_sdnn_ms"], dtype=np.float64)[wi, di, ci]),
                        })
    return pd.DataFrame(rows)


def _metric_table(df: pd.DataFrame, group_cols: list[str], qc_mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        qc_mask = group["ppg_quality_flag"].astype(bool) if qc_mode == "ppg_quality_flag" else pd.Series(True, index=group.index)
        sub = group[qc_mask]
        for metric in METRICS:
            ecg_col = f"ecg_{metric.lower()}_ms"
            ppg_col = f"ppg_{metric.lower()}_ms"
            finite = np.isfinite(sub[ecg_col].to_numpy(float)) & np.isfinite(sub[ppg_col].to_numpy(float))
            stats = _agreement_stats(sub.loc[finite, ppg_col].to_numpy(float), sub.loc[finite, ecg_col].to_numpy(float))
            n_windows = int(len(group))
            rows.append({
                **base,
                "hrv_metric": metric,
                "qc_mode": qc_mode,
                "n_windows": n_windows,
                "n_qc_pass": int(qc_mask.sum()),
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / n_windows) if n_windows else np.nan,
            })
    return pd.DataFrame(rows)


def _dataset_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_cols = ["dataset", "role", "participant", "device", "channel"]
    for qc_mode in QC_MODES:
        rows.append(_metric_table(pairs, group_cols, qc_mode))
    return pd.concat(rows, ignore_index=True)


def _high_level_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for qc_mode in QC_MODES:
        rows.append(_metric_table(pairs, ["dataset", "role"], qc_mode))
    return pd.concat(rows, ignore_index=True)


def _format_float(value: object, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _write_readme(out_dir: Path, pairs: pd.DataFrame, by_device: pd.DataFrame, high: pd.DataFrame, participants: list[str]) -> None:
    lines = [
        "# 选定参与者现有 Baseline 评估",
        "",
        "本报告评估 rawaligned NPZ 文件中已经保存好的 PPG-derived HRV baseline metadata。",
        "本评估不重新检测 PPG peaks，也不训练任何模型。",
        "",
        "## 评估范围",
        "",
        f"- 参与者：{', '.join(participants)}",
        "- 选择理由：P1 = 相对干净/正常样本；P3 = 窗口数最多样本；P5 = 困难/低窗口数样本。",
        "- Baseline 字段：`ppg_rmssd_ms`, `ppg_sdnn_ms`。",
        "- ECG reference 字段：`ecg_rmssd_ms`, `ecg_sdnn_ms`。",
        "- 主 QC 模式：`ppg_quality_flag`；补充模式：`finite_only`。",
        "",
        "## 数据集规模",
        "",
        "| 数据集 | 角色 | 参与者数 | 唯一窗口数 | 设备-通道行数 |",
        "|---|---|---:|---:|---:|",
    ]
    for (dataset, role), group in pairs.groupby(["dataset", "role"], dropna=False):
        n_windows = int(group[["participant", "window_index"]].drop_duplicates().shape[0])
        lines.append(
            f"| `{dataset}` | {role} | {group['participant'].nunique()} | {n_windows} | {len(group)} |"
        )

    lines.extend([
        "",
        "## 高层汇总结果",
        "",
        "| 数据集 | QC | 指标 | 有效配对数 | 覆盖率 | MAE | RMSE | R | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    high_sorted = high.sort_values(["dataset", "qc_mode", "hrv_metric"])
    for _, row in high_sorted.iterrows():
        lines.append(
            f"| `{row['dataset']}` | {row['qc_mode']} | {row['hrv_metric']} | "
            f"{int(row['n_valid'])} | {_format_float(row['coverage_pct'])}% | "
            f"{_format_float(row['MAE'])} ms | {_format_float(row['RMSE'])} ms | "
            f"{_format_float(row['R'], 3)} | {_format_float(row['bias'])} ms |"
        )

    lines.extend([
        "",
        "## 每个数据集/参与者的最佳设备通道",
        "",
        "下表只在本次选定子集内，根据 `ppg_quality_flag` 模式下最低 RMSSD MAE 选择最佳设备/通道；这不是最终公平比较中的全局选择规则。",
        "",
        "| 数据集 | 参与者 | 设备 | 通道 | RMSSD MAE | RMSSD 覆盖率 | RMSSD R |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    rmssd = by_device[
        (by_device["qc_mode"] == "ppg_quality_flag")
        & (by_device["hrv_metric"] == "RMSSD")
        & np.isfinite(by_device["MAE"].to_numpy(float))
    ].copy()
    if not rmssd.empty:
        for (dataset, participant), group in rmssd.groupby(["dataset", "participant"], dropna=False):
            best = group.sort_values(["MAE", "coverage_pct"], ascending=[True, False]).iloc[0]
            lines.append(
                f"| `{dataset}` | {participant} | {best['device']} | {best['channel']} | "
                f"{_format_float(best['MAE'])} ms | {_format_float(best['coverage_pct'])}% | "
                f"{_format_float(best['R'], 3)} |"
            )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `selected_window_level_hrv_pairs.csv` | 窗口/设备/通道级别的 ECG-PPG HRV 配对数据 |",
        "| `selected_hrv_accuracy_by_device_channel.csv` | 按数据集/参与者/设备/通道/QC/HRV 指标分组的评估结果 |",
        "| `selected_hrv_accuracy_high_level_summary.csv` | 按数据集/QC/HRV 指标聚合的高层汇总结果 |",
        "| `summary.json` | 机器可读的评估范围和配置 |",
        "",
        "## 注意事项",
        "",
        "- 这是选定参与者的快速评估，不是最终全 cohort 结果。",
        "- Training stride30 窗口高度重叠，适合开发/debug，不应当当作独立测试证据。",
        "- Strict reference 使用不重叠的 5 分钟窗口，更接近最终评估场景。",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing baseline metadata for selected participants.")
    parser.add_argument("--participants", default="P1,P3,P5")
    parser.add_argument(
        "--strict-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90"),
    )
    parser.add_argument(
        "--training-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_stride30"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.HEURISTIC_RESULT_ROOT / "selected_p1_p3_p5_existing_baseline_eval"),
    )
    args = parser.parse_args()

    participants = sorted([p.strip() for p in args.participants.split(",") if p.strip()], key=_participant_sort_key)
    specs = [
        ("synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90", "strict_reference", Path(args.strict_dir)),
        ("synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_stride30", "training_stride30", Path(args.training_dir)),
    ]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_tables = []
    for dataset_name, role, dataset_dir in specs:
        pair_tables.append(_load_pairs(dataset_dir.resolve(), dataset_name, role, participants))
    pairs = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    if pairs.empty:
        raise SystemExit("No window pairs loaded.")

    by_device = _dataset_summary(pairs)
    high = _high_level_summary(pairs)

    pairs.to_csv(out_dir / "selected_window_level_hrv_pairs.csv", index=False)
    by_device.to_csv(out_dir / "selected_hrv_accuracy_by_device_channel.csv", index=False)
    high.to_csv(out_dir / "selected_hrv_accuracy_high_level_summary.csv", index=False)
    summary = {
        "participants": participants,
        "datasets": [
            {"dataset": name, "role": role, "dir": str(path.resolve())}
            for name, role, path in specs
        ],
        "out_dir": str(out_dir),
        "baseline_fields": ["ppg_rmssd_ms", "ppg_sdnn_ms"],
        "reference_fields": ["ecg_rmssd_ms", "ecg_sdnn_ms"],
        "qc_modes": list(QC_MODES),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, pairs, by_device, high, participants)

    print(f"[saved] {out_dir}")
    print(high.sort_values(["dataset", "qc_mode", "hrv_metric"]).to_string(index=False))


if __name__ == "__main__":
    main()

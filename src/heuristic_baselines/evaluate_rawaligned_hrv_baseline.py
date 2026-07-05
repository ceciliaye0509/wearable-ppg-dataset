"""
评估 raw-aligned PPG-derived HRV 与 ECG-derived HRV 标签的一致性。

本脚本是 generate_rawaligned_4device_dataset.py 生成的数据集的可复现评估器。
读取每个参与者的 NPZ 文件，将每个窗口/设备/通道展开为窗口级表格，输出：

  - *_window_level_hrv_pairs.csv        # 窗口级 ECG-PPG HRV 配对数据
  - *_hrv_accuracy_by_device_channel.csv # 按设备/通道分组的准确性指标
  - *_hrv_accuracy_stratified.csv        # 按运动/SQI 分层的准确性指标
  - combined_hrv_accuracy_main_qc_summary.csv  # 主 QC 模式下的设备/通道汇总
  - combined_hrv_accuracy_high_level_summary.csv  # 高层汇总
  - summary.json                         # 机器可读的范围和注意事项
  - README.md                            # 人类可读的评估报告

运行示例：

  python src/heuristic_baselines/evaluate_rawaligned_hrv_baseline.py \\
    --strict-dir src/heuristic_baselines/outputs/synced_4device_rawaligned_strict_reference \\
    --training-dir src/heuristic_baselines/outputs/synced_4device_rawaligned_training_v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 包路径设置
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import config  # noqa: E402
from generate_rawaligned_4device_dataset import (  # noqa: E402
    _preprocess_for_peaks,
    _ppg_peaks_and_qc,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
METRICS = ("RMSSD", "SDNN")                  # 评估的 HRV 指标
QC_MODES = ("ppg_quality_flag", "finite_only")  # 质控模式：PPG 质量标记 / 仅有限值
# 加速度平均幅值分层区间
ACCEL_BINS = [
    ("<0.1", -math.inf, 0.1),
    ("0.1-0.3", 0.1, 0.3),
    (">=0.3", 0.3, math.inf),
]
# PPG SQI 分层区间
SQI_BINS = [
    ("<0.4", -math.inf, 0.4),
    ("0.4-0.7", 0.4, 0.7),
    (">=0.7", 0.7, math.inf),
]


# ---------------------------------------------------------------------------
# 辅助函数：安全计算 Pearson 相关系数（忽略 NaN/Inf）
# ---------------------------------------------------------------------------
def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# ---------------------------------------------------------------------------
# 辅助函数：计算 PPG vs ECG 的一致性统计量
# 包括 MAE、RMSE、Pearson R、偏差（bias）、95% Bland-Altman LoA
# ---------------------------------------------------------------------------
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
    diff = p - e                         # PPG - ECG 差值
    bias = float(np.mean(diff))          # 平均偏差
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")  # 差值标准差
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),                             # 平均绝对误差
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),                      # 均方根误差
        "R": _safe_corr(p, e),                                           # Pearson 相关系数
        "bias": bias,                                                     # 偏差
        "LoA_lower": bias - 1.96 * sd if np.isfinite(sd) else float("nan"),  # Bland-Altman 下界
        "LoA_upper": bias + 1.96 * sd if np.isfinite(sd) else float("nan"),  # Bland-Altman 上界
    }


# ---------------------------------------------------------------------------
# 辅助函数：获取数据集目录下的所有 NPZ 文件
# ---------------------------------------------------------------------------
def _dataset_npz_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        p for p in dataset_dir.glob("*.npz")
        if not p.name.startswith(".") and "_summary" not in p.name
    )


# ---------------------------------------------------------------------------
# 核心函数：从 NPZ 文件中提取窗口级 ECG-PPG HRV 配对
# 对每个窗口×设备×通道，提取 ECG 和 PPG 的 RMSSD/SDNN 及质控信息
# ---------------------------------------------------------------------------
def _load_window_pairs(dataset_dir: Path, dataset_name: str, note: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = _dataset_npz_files(dataset_dir)
    if not files:
        return pd.DataFrame()

    for path in files:
        with np.load(path, allow_pickle=True) as z:
            participant = str(np.asarray(z["participant"]).item())
            devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
            channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
            n_windows = int(np.asarray(z["ecg_rmssd_ms"]).shape[0])
            target_fs = float(np.asarray(json.loads(str(np.asarray(z["config_json"]).item())).get("target_fs", 100.0)))

            # 读取 PPG 原始波形（用于实时计算 PPG HRV）
            ppg_resampled = np.asarray(z["ppg_resampled"])

            for wi in range(n_windows):
                for di, device in enumerate(devices):
                    for ci, channel in enumerate(channels):
                        # 实时计算 PPG 波峰、IBI、HRV、SQI
                        raw_sig = ppg_resampled[wi, di, ci].astype(np.float64)
                        valid_ratio = float(np.asarray(z["ppg_valid_sample_ratio"], dtype=np.float64)[wi, di, ci])
                        peak_signal = _preprocess_for_peaks(raw_sig, target_fs, "bandpass")
                        peak_info = _ppg_peaks_and_qc(raw_sig, peak_signal, fs=target_fs, valid_sample_ratio=valid_ratio)

                        # 实时计算 PPG 质量标记
                        _sqi = peak_info["ppg_sqi"]
                        _corr = peak_info["ppg_ibi_correction_ratio"]
                        _rmssd = peak_info["ppg_rmssd_ms"]
                        _qf = bool(
                            (not np.isfinite(_corr) or _corr <= 0.5)
                            and (not np.isfinite(_rmssd) or _rmssd <= 200.0)
                            and (np.isfinite(_sqi) and _sqi >= 0.4)
                            and np.isfinite(_rmssd)
                        )

                        rows.append({
                            "dataset": dataset_name,
                            "participant": participant,
                            "window_index": wi,
                            "device": device,
                            "channel": channel,
                            "note": note,
                            "ppg_quality_flag": _qf,
                            "ppg_sqi": _sqi,
                            "accel_mean_mag": float(np.asarray(z["accel_mean_mag"], dtype=np.float64)[wi, di])
                                if "accel_mean_mag" in z.files
                                else (float(np.asarray(z["motion_fraction"], dtype=np.float64)[wi, di])
                                      if "motion_fraction" in z.files else float("nan")),
                            "ppg_valid_ibi_ratio": peak_info["ppg_valid_ibi_ratio"],
                            "ppg_ibi_correction_ratio": _corr,
                            # ECG ground truth
                            "ecg_rmssd_ms": float(np.asarray(z["ecg_rmssd_ms"], dtype=np.float64)[wi]),
                            "ecg_sdnn_ms": float(np.asarray(z["ecg_sdnn_ms"], dtype=np.float64)[wi]),
                            # PPG 估计值（实时计算）
                            "ppg_rmssd_ms": _rmssd,
                            "ppg_sdnn_ms": peak_info["ppg_sdnn_ms"],
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 核心函数：按分组计算 HRV 一致性指标表
# 支持两种 QC 模式：ppg_quality_flag（仅质量达标窗口）/ finite_only（所有有限值）
# ---------------------------------------------------------------------------
def _metric_table(df: pd.DataFrame, group_cols: list[str], qc_mode: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        # 根据 QC 模式选择窗口子集
        if qc_mode == "ppg_quality_flag":
            qc_mask = group["ppg_quality_flag"].astype(bool)     # 仅保留质量达标的窗口
        elif qc_mode == "finite_only":
            qc_mask = pd.Series(True, index=group.index)         # 保留所有窗口
        else:
            raise ValueError(f"unknown qc_mode: {qc_mode}")

        for metric in METRICS:
            ecg_col = f"ecg_{metric.lower()}_ms"
            ppg_col = f"ppg_{metric.lower()}_ms"
            sub = group[qc_mask]
            finite = np.isfinite(sub[ecg_col].to_numpy(float)) & np.isfinite(sub[ppg_col].to_numpy(float))
            stats = _agreement_stats(sub.loc[finite, ppg_col].to_numpy(float), sub.loc[finite, ecg_col].to_numpy(float))
            n_windows = int(len(group))
            row = {
                **base,
                "hrv_metric": metric,
                "qc_mode": qc_mode,
                "n_windows": n_windows,                          # 该分组的总窗口数
                "n_qc_pass": int(qc_mask.sum()),                 # 通过 QC 的窗口数
                **stats,
                "coverage_pct": float(100.0 * stats["n_valid"] / n_windows) if n_windows else float("nan"),  # 有效覆盖率
            }
            out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# 核心函数：按运动占比和 SQI 分层计算一致性指标
# ---------------------------------------------------------------------------
def _stratified_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "participant", "device", "channel", "note"]
    for strat_name, bins, value_col in (
        ("accel_bin", ACCEL_BINS, "accel_mean_mag"),    # 按加速度平均幅值分层
        ("sqi_bin", SQI_BINS, "ppg_sqi"),                  # 按 PPG SQI 分层
    ):
        for label, low, high in bins:
            if np.isneginf(low):
                mask = df[value_col] < high
            elif np.isposinf(high):
                mask = df[value_col] >= low
            else:
                mask = (df[value_col] >= low) & (df[value_col] < high)
            sub = df[mask].copy()
            if sub.empty:
                continue
            table = _metric_table(sub, group_cols, "ppg_quality_flag")
            if table.empty:
                continue
            table["stratification"] = strat_name
            table["bin"] = label
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 辅助函数：生成人类可读的评估报告 README.md
# ---------------------------------------------------------------------------
def _write_readme(out_dir: Path, high_level: pd.DataFrame, dataset_notes: list[dict[str, object]]) -> None:
    def fmt(x: object, digits: int = 2) -> str:
        try:
            f = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"

    lines = [
        "# rawaligned_hrv_baseline_evaluation",
        "",
        "本报告由 `evaluate_rawaligned_hrv_baseline.py` 自动生成，用于评估传统 PPG-derived HRV baseline 相对 ECG-derived HRV reference 的准确性。",
        "",
        "## Dataset Scope",
        "",
        "| Dataset | Role | Windowing/Note | Participants | Windows |",
        "|---|---|---|---:|---:|",
    ]
    for item in dataset_notes:
        lines.append(
            f"| `{item['dataset']}` | {item['role']} | {item['note']} | "
            f"{int(item['participants'])} | {int(item['windows'])} |"
        )

    lines.extend([
        "",
        "## Evaluation Definition",
        "",
        "| Item | Definition |",
        "|---|---|",
        "| ECG reference | `ecg_rmssd_ms`, `ecg_sdnn_ms` |",
        "| PPG baseline | 每个设备/通道的 `ppg_rmssd_ms`, `ppg_sdnn_ms` |",
        "| Main QC mode | `ppg_quality_flag=True` 且 ECG/PPG HRV 为 finite |",
        "| Metrics | MAE, RMSE, Pearson R, bias, 95% limits of agreement, coverage |",
        "| Stratification | device, channel, motion bins, SQI bins |",
        "",
        "## High-Level Results",
        "",
        "| Dataset | HRV Metric | Valid Pairs | Coverage | MAE | RMSE | R | Bias | LoA Lower | LoA Upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in high_level.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['hrv_metric']} | {int(row['n_valid'])} | "
            f"{fmt(row['coverage_pct'])}% | {fmt(row['MAE'])} ms | {fmt(row['RMSE'])} ms | "
            f"{fmt(row['R'], 3)} | {fmt(row['bias'])} ms | {fmt(row['LoA_lower'])} ms | {fmt(row['LoA_upper'])} ms |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
        "| File | Meaning |",
        "|---|---|",
        "| `strict_reference_hrv_accuracy_by_device_channel.csv` | strict-reference 按参与者/设备/通道/指标分组的准确性 |",
        "| `strict_reference_hrv_accuracy_stratified.csv` | strict-reference 按运动和 SQI 分层的准确性 |",
        "| `strict_reference_window_level_hrv_pairs.csv` | strict-reference 窗口级 ECG/PPG HRV 配对 |",
        "| `training_v1_supplement_hrv_accuracy_by_device_channel.csv` | training-v1 补充准确性指标（如有本地 NPZ 文件） |",
        "| `training_v1_supplement_hrv_accuracy_stratified.csv` | training-v1 按运动/SQI 分层的补充准确性 |",
        "| `training_v1_supplement_window_level_hrv_pairs.csv` | training-v1 窗口级配对 |",
        "| `combined_hrv_accuracy_main_qc_summary.csv` | 主 QC 模式下的设备/通道汇总 |",
        "| `combined_hrv_accuracy_high_level_summary.csv` | 高层汇总 |",
        "| `summary.json` | 机器可读的范围和注意事项 |",
        "",
        "## Notes",
        "",
        "- `coverage_pct` 的分母是该 dataset 中对应 device/channel 的全部 windows；主 QC 下分子为 `ppg_quality_flag=True` 且 ECG/PPG HRV finite 的窗口数。",
        "- `training_v1` 是 rolling windows，窗口重叠强，适合作补充分析，不应按完全独立样本解释。",
        "- Activity stratification 当前未包含，因为 aligned NPZ 未直接保存 activity labels；后续可用 `activity_log.txt` 按时间戳 join。",
        "",
    ])
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 核心函数：评估单个数据集
# 生成窗口级配对 → 按设备/通道分组统计 → 分层统计
# ---------------------------------------------------------------------------
def _evaluate_dataset(dataset_dir: Path, dataset_name: str, prefix: str, note: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 加载窗口级 ECG-PPG HRV 配对
    pairs = _load_window_pairs(dataset_dir, dataset_name, note)
    if pairs.empty:
        return pairs, pd.DataFrame(), pd.DataFrame()
    pairs.to_csv(out_dir / f"{prefix}_window_level_hrv_pairs.csv", index=False)

    # 按设备/通道计算一致性指标（两种 QC 模式）
    by = _metric_table(pairs, ["dataset", "participant", "device", "channel", "note"], "ppg_quality_flag")
    by_finite = _metric_table(pairs, ["dataset", "participant", "device", "channel", "note"], "finite_only")
    by_all = pd.concat([by, by_finite], ignore_index=True)
    by_all.to_csv(out_dir / f"{prefix}_hrv_accuracy_by_device_channel.csv", index=False)

    # 按运动占比和 SQI 分层统计
    strat = _stratified_table(pairs)
    strat.to_csv(out_dir / f"{prefix}_hrv_accuracy_stratified.csv", index=False)
    return pairs, by_all, strat


# ===========================================================================
# ★ 主入口
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="评估 raw-aligned HRV baseline 数据集")
    parser.add_argument("--strict-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_rawaligned_strict_reference"))
    parser.add_argument("--training-dir", default=str(config.HEURISTIC_RESULT_ROOT / "synced_4device_rawaligned_training_v1"))
    parser.add_argument("--out-dir", default=str(config.HEURISTIC_RESULT_ROOT / "rawaligned_hrv_baseline_evaluation"))
    parser.add_argument("--strict-name", default="synced_4device_rawaligned_strict_reference")
    parser.add_argument("--training-name", default="synced_4device_rawaligned_training_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 定义要评估的数据集：strict-reference（主报告）和 training-v1（补充报告）
    datasets = [
        {
            "dataset": args.strict_name,
            "prefix": "strict_reference",
            "role": "主报告",
            "note": "5-min non-overlap, stride_sec=300",
            "dir": Path(args.strict_dir).resolve(),
        },
        {
            "dataset": args.training_name,
            "prefix": "training_v1_supplement",
            "role": "补充报告",
            "note": "5-min rolling, stride_sec=60",
            "dir": Path(args.training_dir).resolve(),
        },
    ]

    # --- 逐数据集评估 ---
    all_pairs: list[pd.DataFrame] = []
    all_by: list[pd.DataFrame] = []
    dataset_notes: list[dict[str, object]] = []
    for spec in datasets:
        pairs, by, _strat = _evaluate_dataset(spec["dir"], spec["dataset"], spec["prefix"], spec["note"], out_dir)
        if pairs.empty:
            dataset_notes.append({
                "dataset": spec["dataset"],
                "role": spec["role"],
                "note": f"{spec['note']} (未找到本地 NPZ 文件)",
                "participants": 0,
                "windows": 0,
            })
            continue
        all_pairs.append(pairs)
        all_by.append(by)
        dataset_notes.append({
            "dataset": spec["dataset"],
            "role": spec["role"],
            "note": spec["note"],
            "participants": int(pairs["participant"].nunique()),
            "windows": int(pairs[["participant", "window_index"]].drop_duplicates().shape[0]),
        })

    # --- 合并所有数据集的设备/通道级汇总（主 QC 模式） ---
    combined_by = pd.concat(all_by, ignore_index=True) if all_by else pd.DataFrame()
    if not combined_by.empty:
        main_qc = combined_by[combined_by["qc_mode"] == "ppg_quality_flag"].copy()
        # 按数据集/指标/设备/通道聚合
        device_summary = (
            main_qc
            .groupby(["dataset", "hrv_metric", "device", "channel"], dropna=False)
            .agg(
                n_windows=("n_windows", "sum"),
                n_valid=("n_valid", "sum"),
                MAE=("MAE", "mean"),
                RMSE=("RMSE", "mean"),
                R=("R", "mean"),
                bias=("bias", "mean"),
                LoA_lower=("LoA_lower", "mean"),
                LoA_upper=("LoA_upper", "mean"),
            )
            .reset_index()
        )
        device_summary["coverage_pct"] = 100.0 * device_summary["n_valid"] / device_summary["n_windows"]
        device_summary.to_csv(out_dir / "combined_hrv_accuracy_main_qc_summary.csv", index=False)

    # --- 高层汇总：每个数据集×每个 HRV 指标的整体一致性 ---
    combined_pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    high_rows: list[dict[str, object]] = []
    if not combined_pairs.empty:
        for dataset, dataset_df in combined_pairs.groupby("dataset", dropna=False):
            for metric in METRICS:
                ecg_col = f"ecg_{metric.lower()}_ms"
                ppg_col = f"ppg_{metric.lower()}_ms"
                # 仅使用通过 PPG 质量标记的窗口
                sub = dataset_df[dataset_df["ppg_quality_flag"].astype(bool)]
                finite = np.isfinite(sub[ecg_col].to_numpy(float)) & np.isfinite(sub[ppg_col].to_numpy(float))
                stats = _agreement_stats(sub.loc[finite, ppg_col].to_numpy(float), sub.loc[finite, ecg_col].to_numpy(float))
                high_rows.append({
                    "dataset": dataset,
                    "hrv_metric": metric,
                    **stats,
                    "coverage_pct": 100.0 * stats["n_valid"] / len(dataset_df) if len(dataset_df) else float("nan"),
                })
    high_level = pd.DataFrame(high_rows)
    high_level.to_csv(out_dir / "combined_hrv_accuracy_high_level_summary.csv", index=False)

    # --- 保存机器可读的汇总 JSON ---
    summary = {
        "script": str(Path(__file__).resolve()),
        "out_dir": str(out_dir),
        "datasets": dataset_notes,
        "main_qc_mode": "ppg_quality_flag",
        "metrics": ["MAE", "RMSE", "R", "bias", "LoA_lower", "LoA_upper", "coverage_pct"],
        "stratification": ["device", "channel", "accel_bin", "sqi_bin"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- 生成 README 评估报告 ---
    _write_readme(out_dir, high_level, dataset_notes)

    print(f"[已保存] {out_dir}")
    for item in dataset_notes:
        print(f"  {item['dataset']}: 参与者={item['participants']} 窗口={item['windows']}")


if __name__ == "__main__":
    main()

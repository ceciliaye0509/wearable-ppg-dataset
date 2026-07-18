"""
历史探索脚本：生成旧版 current-best no-motion baseline 的参与者/设备报告。

本脚本已归档，不属于正式 v1-primary unified heuristic baseline 主流程。
保留它只是为了回看或复跑旧 motion threshold 实验。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# 归档脚本位于 archive/legacy_current_best_baseline/ 下；这里向上推导
# heuristic_baselines 根目录和 src 根目录，保证从任意工作目录运行时都能导入项目模块。
ARCHIVE_ROOT = Path(__file__).resolve().parent
HEURISTIC_ROOT = ARCHIVE_ROOT.parents[1]
SRC_ROOT = HEURISTIC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 旧版 no-motion 报告读取 heuristic_baselines/outputs 下的 raw-aligned NPZ，
# 并默认把报告写回当前归档目录。
OUTPUT_ROOT = HEURISTIC_ROOT / "outputs"

# 默认 no-motion 阈值。脚本入口允许用 --motion-threshold 覆盖，
# 用于复跑 <0.1、<0.2 等旧版 sensitivity analysis。
NO_MOTION_THRESHOLD = 0.1

# 两个输入数据集代表两种旧分析视角：strict_reference 用于更接近最终评估，
# training_v1_stride30 用于观察重叠窗口下的开发/训练表现。
DATASETS = [
    {
        "label": "strict_reference",
        "name": "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100",
        "interpretation": "不重叠 5 分钟窗口；按设备筛选 no-motion 窗口",
    },
    {
        "label": "training_v1_stride30",
        "name": "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "dir": OUTPUT_ROOT / "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30",
        "interpretation": "30 秒 stride 滚动窗口；按设备筛选 no-motion 窗口",
    },
]

METRICS = ("RMSSD", "SDNN")


def _threshold_token(threshold: float) -> str:
    # 文件名不能直接保留小数点和负号语义，这里把 0.2 转成 0p2，便于稳定命名。
    return str(threshold).replace(".", "p").replace("-", "neg")


def _report_paths(threshold: float, output_dir: Path | None = None) -> tuple[Path, Path]:
    # 根据 motion threshold 决定输出文件名；<0.1 保留历史文件名以兼容旧结果。
    root = output_dir or ARCHIVE_ROOT
    if np.isclose(threshold, 0.1):
        stem = "current_best_baseline_no_motion_strict_vs_training_stride30_by_participant_device"
    else:
        stem = f"current_best_baseline_motion_lt_{_threshold_token(threshold)}_strict_vs_training_stride30_by_participant_device"
    return root / f"{stem}.md", root / f"{stem}.csv"


def _participant_sort_key(pid: object) -> int:
    # 按 P1, P2, ... 的数字顺序排序；无法解析的 id 放到最后。
    s = str(pid)
    return int(s[1:]) if s.startswith("P") and s[1:].isdigit() else 999999


def _safe_corr(x: list[float], y: list[float]) -> float:
    # Pearson R 至少需要 3 个有效配对点；否则返回 NaN，避免报告虚假的相关性。
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def _agreement(ppg: list[float], ecg: list[float]) -> dict[str, float]:
    # 将 PPG-derived HRV 与 ECG reference 对齐后计算误差、相关性和系统偏差。
    p = np.asarray(ppg, dtype=float)
    e = np.asarray(ecg, dtype=float)
    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]
    if p.size == 0:
        return {"n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R": np.nan, "bias": np.nan}
    diff = p - e
    return {
        "n_valid": int(p.size),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff * diff))),
        "R": _safe_corr(p.tolist(), e.tolist()),
        "bias": float(np.mean(diff)),
    }


def _process_npz(task: tuple[str, str, str]) -> list[dict[str, object]]:
    # 单个 participant NPZ 的核心评估逻辑：先按设备筛 no-motion 窗口，
    # 再逐窗口、逐通道重新从 PPG 算 HRV。
    dataset_label, dataset_name, path_str = task
    from heuristic_baselines.algorithms.hrv import hrv_from_ppg

    rows: list[dict[str, object]] = []
    path = Path(path_str)
    with np.load(path, allow_pickle=True) as z:
        participant = str(np.asarray(z["participant"]).item())
        devices = [str(x) for x in np.asarray(z["devices"]).tolist()]
        channels = [str(x) for x in np.asarray(z["channels"]).tolist()]
        cfg = json.loads(str(np.asarray(z["config_json"]).item()))
        fs = float(cfg.get("target_fs", 100.0))
        ppg = z["ppg_resampled"]
        ecg_rmssd = np.asarray(z["ecg_rmssd_ms"], dtype=float)
        ecg_sdnn = np.asarray(z["ecg_sdnn_ms"], dtype=float)
        accel_raw = np.asarray(z["accel_mean_mag"], dtype=float)
        accel_motion = np.asarray(
            z["accel_motion_mean_mag"] if "accel_motion_mean_mag" in z.files else z["accel_mean_mag"],
            dtype=float,
        )
        total_windows = int(ppg.shape[0])

        # 每个设备独立判断 no-motion。这样同一个 window 对 Earring 可能被保留，
        # 但对 Ring/Watch 可能因为 motion 更高而被剔除。
        for di, device in enumerate(devices):
            no_motion_mask = np.isfinite(accel_motion[:, di]) & (accel_motion[:, di] < NO_MOTION_THRESHOLD)
            no_motion_idx = np.where(no_motion_mask)[0]
            n_no_motion = int(no_motion_idx.size)
            pred = {metric: [] for metric in METRICS}
            ref = {metric: [] for metric in METRICS}
            selected_counts: Counter[str] = Counter()
            reason_counts: Counter[str] = Counter()
            sqi_vals: list[float] = []
            valid_ibi_vals: list[float] = []
            corr_vals: list[float] = []
            n_selected = 0

            # 只遍历当前设备通过 no-motion 阈值的窗口；旧版 no-motion 报告的分母就是这些窗口。
            for wi in no_motion_idx:
                candidates: list[dict[str, float | str]] = []
                # 对该设备的 green/IR 两个通道分别运行同一个 hrv_from_ppg 旧版管线。
                for ci, channel in enumerate(channels):
                    out = hrv_from_ppg(ppg[wi, di, ci], fs, freq=False, nonlinear=False)
                    rmssd = float(out.get("HRV_RMSSD", np.nan))
                    sdnn = float(out.get("HRV_SDNN", np.nan))
                    sqi = float(out.get("sqi", np.nan))
                    valid_ibi = float(out.get("valid_ibi_ratio", np.nan))
                    corr = float(out.get("ibi_correction_ratio", np.nan))
                    reason = str(out.get("ppg_qc_reason", ""))
                    if np.isfinite(rmssd) and np.isfinite(sdnn):
                        candidates.append(
                            {
                                "channel": channel,
                                "rmssd": rmssd,
                                "sdnn": sdnn,
                                "sqi": sqi,
                                "valid_ibi": valid_ibi,
                                "corr": corr,
                                "reason": reason,
                            }
                        )
                    else:
                        reason_counts[reason or "invalid"] += 1

                if not candidates:
                    continue

                def sort_key(c: dict[str, float | str]) -> tuple[float, float, float, str]:
                    # 旧版 best-SQI 规则：优先 SQI 高，其次 valid IBI ratio 高，
                    # 再其次 IBI correction ratio 低；最后用通道名稳定打破平局。
                    corr = float(c["corr"]) if np.isfinite(float(c["corr"])) else 999.0
                    sqi = float(c["sqi"]) if np.isfinite(float(c["sqi"])) else -1.0
                    valid_ibi = float(c["valid_ibi"]) if np.isfinite(float(c["valid_ibi"])) else -1.0
                    return (-sqi, -valid_ibi, corr, str(c["channel"]))

                # 只把 no-motion 子集中选中的通道写入预测序列，并与同一窗口 ECG HRV 比较。
                best = sorted(candidates, key=sort_key)[0]
                n_selected += 1
                selected_counts[str(best["channel"])] += 1
                reason_counts[str(best["reason"]) or "ok"] += 1
                if np.isfinite(float(best["sqi"])):
                    sqi_vals.append(float(best["sqi"]))
                if np.isfinite(float(best["valid_ibi"])):
                    valid_ibi_vals.append(float(best["valid_ibi"]))
                if np.isfinite(float(best["corr"])):
                    corr_vals.append(float(best["corr"]))
                pred["RMSSD"].append(float(best["rmssd"]))
                ref["RMSSD"].append(float(ecg_rmssd[wi]))
                pred["SDNN"].append(float(best["sdnn"]))
                ref["SDNN"].append(float(ecg_sdnn[wi]))

            # base 保存 participant-device 级别的共同统计；后面分别附上 RMSSD/SDNN 的误差。
            base = {
                "dataset": dataset_name,
                "dataset_label": dataset_label,
                "participant": participant,
                "device": device,
                "method": "no_motion_device_best_sqi_green_or_ir__hrv_from_ppg",
                "no_motion_definition": f"accel_motion_mean_mag < {NO_MOTION_THRESHOLD}",
                "total_windows": total_windows,
                "n_no_motion_windows": n_no_motion,
                "no_motion_pct": 100.0 * n_no_motion / total_windows if total_windows else np.nan,
                "n_selected_windows": n_selected,
                "coverage_pct_within_no_motion": 100.0 * n_selected / n_no_motion if n_no_motion else np.nan,
                "coverage_pct_of_all_windows": 100.0 * n_selected / total_windows if total_windows else np.nan,
                "mean_accel_mean_mag_all_windows": float(np.nanmean(accel_raw[:, di])) if total_windows else np.nan,
                "mean_accel_motion_mean_mag_all_windows": float(np.nanmean(accel_motion[:, di])) if total_windows else np.nan,
                "mean_accel_mean_mag_no_motion": float(np.nanmean(accel_raw[no_motion_idx, di])) if n_no_motion else np.nan,
                "mean_accel_motion_mean_mag_no_motion": float(np.nanmean(accel_motion[no_motion_idx, di])) if n_no_motion else np.nan,
                "selected_ppg_green": int(selected_counts.get("ppg_green", 0)),
                "selected_ppg_ir": int(selected_counts.get("ppg_ir", 0)),
                "mean_selected_sqi": float(np.nanmean(sqi_vals)) if sqi_vals else np.nan,
                "mean_selected_valid_ibi_ratio": float(np.nanmean(valid_ibi_vals)) if valid_ibi_vals else np.nan,
                "mean_selected_ibi_correction_ratio": float(np.nanmean(corr_vals)) if corr_vals else np.nan,
                "top_qc_reasons": "; ".join(f"{k}:{v}" for k, v in reason_counts.most_common(4)),
            }
            for metric in METRICS:
                rows.append({**base, "hrv_metric": metric, **_agreement(pred[metric], ref[metric])})
    return rows


def _fmt(value: object, digits: int = 2) -> str:
    # Markdown 表格专用数值格式化；非有限值留空，避免报告里出现 nan/inf。
    try:
        f = float(value)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    # 将 DataFrame 的指定列渲染成 Markdown 表格，并按列语义控制小数位。
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            v = row[col]
            if col in {
                "MAE",
                "RMSE",
                "bias",
                "mean_accel_mean_mag_no_motion",
                "mean_accel_motion_mean_mag_no_motion",
                "RMSSD_MAE_mean_by_participant",
                "mae_full",
                "mae_candidate",
                "mean_rmssd_mae",
            }:
                vals.append(_fmt(v, 2))
            elif col in {"R", "RMSSD_R_median_by_participant", "r_full_median", "r_candidate_median", "median_rmssd_r"}:
                vals.append(_fmt(v, 3))
            elif col.endswith("pct") or col.startswith("coverage_pct") or col == "no_motion_pct" or col.endswith("_pct"):
                vals.append(_fmt(v, 1))
            elif col.startswith("n_") or col.startswith("selected_") or col == "total_windows" or col in {"paired_mae_rows", "candidate_lower_mae", "paired_r_rows", "candidate_higher_r", "coverage_rows", "candidate_higher_coverage", "subset_windows", "valid_predictions"}:
                vals.append(str(int(v)) if pd.notna(v) else "")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _sort_participant_frame(df: pd.DataFrame) -> pd.DataFrame:
    # 报告中的 participant-device 明细按 dataset、participant 数字、device 排序，方便人工查阅。
    return (
        df.assign(_pid_sort=df["participant"].map(_participant_sort_key))
        .sort_values(["dataset_label", "_pid_sort", "device"])
        .drop(columns=["_pid_sort"])
    )


def _comparison_candidates(current_threshold: float, current_csv: Path, output_dir: Path | None = None) -> dict[str, Path]:
    # 收集可用于 sensitivity comparison 的旧报告 CSV：full、<0.1、<0.2 和当前阈值。
    # 如果 <0.2 只存在于早期临时目录，也允许读取它以复现旧分析。
    root = output_dir or current_csv.parent
    candidates = {
        "full": root / "current_best_baseline_strict_vs_training_stride30_by_participant_device.csv",
        "<0.1": _report_paths(0.1, root)[1],
        "<0.2": _report_paths(0.2, root)[1],
    }
    temp_02 = Path("/private/tmp/hrv_no_motion_threshold_0p2.csv")
    if not candidates["<0.2"].is_file() and temp_02.is_file():
        candidates["<0.2"] = temp_02
    label = f"<{current_threshold:g}"
    candidates[label] = current_csv
    return candidates


def _coverage_col(df: pd.DataFrame) -> str:
    # full 报告和 no-motion 报告的 coverage 列名不同，这里统一取当前表能用的列。
    if "coverage_pct_within_no_motion" in df.columns:
        return "coverage_pct_within_no_motion"
    return "coverage_pct"


def _selected_denominator_col(df: pd.DataFrame) -> str:
    # no-motion 报告的有效预测覆盖率分母是 n_no_motion_windows；
    # full 报告的分母是 n_windows。
    if "n_no_motion_windows" in df.columns:
        return "n_no_motion_windows"
    return "n_windows"


def _paired_comparison(reference: pd.DataFrame, candidate: pd.DataFrame, reference_label: str, candidate_label: str) -> pd.DataFrame:
    # 在 dataset/participant/device/metric 完全相同的行之间做配对比较，
    # 统计候选阈值相对 reference 是否降低 MAE、提高 R、提高 coverage。
    keys = ["dataset_label", "participant", "device", "hrv_metric"]
    ref_cov = _coverage_col(reference)
    cand_cov = _coverage_col(candidate)
    merged = reference[keys + ["MAE", "R", ref_cov]].merge(
        candidate[keys + ["MAE", "R", cand_cov]],
        on=keys,
        suffixes=("_ref", "_cand"),
    )
    ref_cov_col = f"{ref_cov}_ref" if f"{ref_cov}_ref" in merged.columns else ref_cov
    cand_cov_col = f"{cand_cov}_cand" if f"{cand_cov}_cand" in merged.columns else cand_cov
    rows = []
    for metric in METRICS:
        m = merged[merged["hrv_metric"] == metric]
        mae = m[np.isfinite(m["MAE_ref"]) & np.isfinite(m["MAE_cand"])]
        r = m[np.isfinite(m["R_ref"]) & np.isfinite(m["R_cand"])]
        rows.append(
            {
                "comparison": f"{candidate_label} vs {reference_label}",
                "hrv_metric": metric,
                "paired_mae_rows": len(mae),
                "candidate_lower_mae": int((mae["MAE_cand"] < mae["MAE_ref"]).sum()),
                "paired_r_rows": len(r),
                "candidate_higher_r": int((r["R_cand"] > r["R_ref"]).sum()),
                "coverage_rows": len(m),
                "candidate_higher_coverage": int((m[cand_cov_col] > m[ref_cov_col]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_against_full(full: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    # 将当前 no-motion 阈值和 full report 按 dataset x device 汇总比较，
    # 用来观察限制低运动窗口后 MAE/R/coverage 的整体变化。
    rows = []
    for metric in METRICS:
        full_metric = full[full["hrv_metric"] == metric]
        cand_metric = candidate[candidate["hrv_metric"] == metric]
        for dataset in ("strict_reference", "training_v1_stride30"):
            for device in ("Earring", "Ring", "Watch"):
                f = full_metric[(full_metric["dataset_label"] == dataset) & (full_metric["device"] == device)]
                c = cand_metric[(cand_metric["dataset_label"] == dataset) & (cand_metric["device"] == device)]
                if f.empty or c.empty:
                    continue
                full_denom = f["n_windows"].sum()
                cand_denom = c[_selected_denominator_col(c)].sum()
                rows.append(
                    {
                        "hrv_metric": metric,
                        "dataset_label": dataset,
                        "device": device,
                        "mae_full": float(np.nanmean(f["MAE"])),
                        "mae_candidate": float(np.nanmean(c["MAE"])),
                        "r_full_median": float(np.nanmedian(f["R"])),
                        "r_candidate_median": float(np.nanmedian(c["R"])),
                        "coverage_full_pct": 100.0 * f["n_selected_windows"].sum() / full_denom if full_denom else np.nan,
                        "coverage_candidate_pct": 100.0 * c["n_selected_windows"].sum() / cand_denom if cand_denom else np.nan,
                        "candidate_subset_pct": 100.0 * cand_denom / c["total_windows"].sum() if "total_windows" in c else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _threshold_summary(candidates: dict[str, Path]) -> pd.DataFrame:
    # 对所有已存在的 no-motion threshold CSV 做 RMSSD 概览，
    # 方便判断阈值变宽时覆盖率和误差如何移动。
    rows = []
    for label, path in candidates.items():
        if label == "full" or not path.is_file():
            continue
        df = pd.read_csv(path)
        rmssd = df[df["hrv_metric"] == "RMSSD"]
        for dataset in ("strict_reference", "training_v1_stride30"):
            one = rmssd[rmssd["dataset_label"] == dataset]
            if one.empty:
                continue
            denom = one[_selected_denominator_col(one)].sum()
            rows.append(
                {
                    "threshold": label,
                    "dataset_label": dataset,
                    "subset_windows": int(denom),
                    "valid_predictions": int(one["n_selected_windows"].sum()),
                    "coverage_pct": 100.0 * one["n_selected_windows"].sum() / denom if denom else np.nan,
                    "mean_rmssd_mae": float(np.nanmean(one["MAE"])),
                    "median_rmssd_r": float(np.nanmedian(one["R"])),
                }
            )
    return pd.DataFrame(rows)


def _comparison_section(df: pd.DataFrame, current_threshold: float, current_csv: Path, output_dir: Path | None = None) -> list[str]:
    # 生成 Markdown 中的比较分析章节。如果 full 报告不存在，就只保留主报告结果。
    candidates = _comparison_candidates(current_threshold, current_csv, output_dir)
    full_path = candidates["full"]
    if not full_path.is_file():
        return ["", "## 比较分析", "", "- 未找到完整窗口报告 CSV，因此跳过 full/no-motion 比较。"]

    full = pd.read_csv(full_path)
    current_label = f"<{current_threshold:g}"
    sections = ["", "## 与 full / <0.1 / <0.2 的比较", ""]
    sections.append(
        f"当前报告的 motion threshold 是 `{current_label}`。这里把它和 full report、严格 no-motion `<0.1`、以及已计算的 `<0.2` 结果做 sensitivity comparison。"
    )

    paired_frames = [_paired_comparison(full, df, "full", current_label)]
    for label in ("<0.1", "<0.2"):
        path = candidates.get(label)
        if path and path.is_file() and path != current_csv:
            paired_frames.append(_paired_comparison(pd.read_csv(path), df, label, current_label))
    paired = pd.concat(paired_frames, ignore_index=True)
    sections.extend(
        [
            "",
            "### 有效配对行统计",
            "",
            _md_table(
                paired,
                [
                    "comparison",
                    "hrv_metric",
                    "paired_mae_rows",
                    "candidate_lower_mae",
                    "paired_r_rows",
                    "candidate_higher_r",
                    "coverage_rows",
                    "candidate_higher_coverage",
                ],
                [
                    "Comparison",
                    "Metric",
                    "Paired MAE rows",
                    "Candidate lower MAE",
                    "Paired R rows",
                    "Candidate higher R",
                    "Coverage rows",
                    "Candidate higher coverage",
                ],
            ),
        ]
    )

    threshold_overview = _threshold_summary(candidates)
    if not threshold_overview.empty:
        sections.extend(
            [
                "",
                "### Threshold 总览（RMSSD）",
                "",
                _md_table(
                    threshold_overview.sort_values(["dataset_label", "threshold"]),
                    ["threshold", "dataset_label", "subset_windows", "valid_predictions", "coverage_pct", "mean_rmssd_mae", "median_rmssd_r"],
                    ["Threshold", "Dataset", "Subset windows", "Valid preds", "Coverage %", "Mean RMSSD MAE", "Median RMSSD R"],
                ),
            ]
        )

    agg = _aggregate_against_full(full, df)
    if not agg.empty:
        for metric in METRICS:
            metric_agg = agg[agg["hrv_metric"] == metric]
            sections.extend(
                [
                    "",
                    f"### Dataset x Device 汇总比较（{metric}, {current_label} vs full）",
                    "",
                    _md_table(
                        metric_agg,
                        [
                            "dataset_label",
                            "device",
                            "mae_full",
                            "mae_candidate",
                            "r_full_median",
                            "r_candidate_median",
                            "coverage_full_pct",
                            "coverage_candidate_pct",
                            "candidate_subset_pct",
                        ],
                        [
                            "Dataset",
                            "Device",
                            "MAE full",
                            f"MAE {current_label}",
                            "R full",
                            f"R {current_label}",
                            "Coverage full",
                            f"Coverage {current_label}",
                            f"Subset % {current_label}",
                        ],
                    ),
                ]
            )

    sections.extend(
        [
            "",
            "解释：threshold 越宽，subset windows 通常越多；但它逐渐从 strict no-motion 变成 low-motion / low-to-moderate-motion sensitivity analysis。Coverage 的分母是 threshold 子集内部窗口数，不能直接等同于完整报告的 overall coverage。",
        ]
    )
    return sections


def main() -> None:
    # 主流程：解析阈值参数，选择重新计算或复用 CSV，然后生成 CSV 和 Markdown。
    global NO_MOTION_THRESHOLD

    parser = argparse.ArgumentParser(description="评估旧版 current-best baseline 在不同 motion threshold 下的表现。")
    parser.add_argument("--motion-threshold", type=float, default=NO_MOTION_THRESHOLD)
    parser.add_argument("--reuse-csv", action="store_true", help="跳过 PPG 重新计算，使用已有 CSV 重新生成 Markdown 报告。")
    parser.add_argument("--reuse-csv-source", default=None, help="可选：先复制/使用这个 CSV，再重新生成 Markdown 报告。")
    parser.add_argument("--output-dir", default=str(ARCHIVE_ROOT), help="生成 MD/CSV 报告的目录。")
    args = parser.parse_args()
    NO_MOTION_THRESHOLD = float(args.motion_threshold)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md, report_csv = _report_paths(NO_MOTION_THRESHOLD, output_dir)

    start = time.time()
    if args.reuse_csv:
        # 复用模式只重写 Markdown，适合已经有 CSV、只想改报告解释或比较章节时使用。
        if args.reuse_csv_source:
            source = Path(args.reuse_csv_source).resolve()
            if not source.is_file():
                raise SystemExit(f"--reuse-csv-source not found: {source}")
            df = pd.read_csv(source)
            df.to_csv(report_csv, index=False)
            print(f"[reuse] copied {source} -> {report_csv}")
        if not report_csv.is_file():
            raise SystemExit(f"--reuse-csv requested but not found: {report_csv}")
        print(f"[reuse] loading {report_csv}")
        df = pd.read_csv(report_csv)
    else:
        # 重新计算模式会扫描两个数据集的全部 participant NPZ，并重新运行旧版 PPG HRV 管线。
        tasks: list[tuple[str, str, str]] = []
        for spec in DATASETS:
            # 每个 NPZ 对应一个 participant；这里统一按 participant 编号排序，保证输出可复现。
            files = sorted(
                spec["dir"].glob(f"{spec['name']}_P*.npz"),
                key=lambda p: _participant_sort_key(p.stem.split("_")[-1]),
            )
            if not files:
                raise SystemExit(f"No NPZ files found for {spec['dir']}")
            tasks.extend((spec["label"], spec["name"], str(path)) for path in files)

        print(f"[eval no-motion] tasks={len(tasks)}")
        all_rows: list[dict[str, object]] = []
        max_workers = min(4, os.cpu_count() or 1)
        # 单个 participant 的计算互相独立，用少量线程并行加速旧版复算。
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_npz, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                all_rows.extend(rows)
                sample = rows[0] if rows else {}
                print(
                    f"[eval no-motion] done {i}/{len(tasks)} "
                    f"{sample.get('dataset_label', '')} {sample.get('participant', '')} rows={len(rows)}",
                    flush=True,
                )

        df = pd.DataFrame(all_rows)
        df = _sort_participant_frame(df).reset_index(drop=True)
        df.to_csv(report_csv, index=False)

    # 分离 RMSSD/SDNN，后续主汇总以 RMSSD 为主，SDNN 作为补充明细。
    rmssd = df[df["hrv_metric"] == "RMSSD"].copy()
    sdnn = df[df["hrv_metric"] == "SDNN"].copy()
    # dataset x device 汇总先在 participant 层算 MAE/R，再做均值/中位数，
    # 避免窗口数多的 participant 过度主导结果。
    dataset_device = (
        rmssd.groupby(["dataset_label", "device"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_participants": g["participant"].nunique(),
                    "total_windows": g["total_windows"].sum(),
                    "n_no_motion_windows": g["n_no_motion_windows"].sum(),
                    "n_selected_windows": g["n_selected_windows"].sum(),
                    "no_motion_pct": 100.0 * g["n_no_motion_windows"].sum() / g["total_windows"].sum(),
                    "coverage_pct_within_no_motion": 100.0 * g["n_selected_windows"].sum() / g["n_no_motion_windows"].sum()
                    if g["n_no_motion_windows"].sum()
                    else np.nan,
                    "RMSSD_MAE_mean_by_participant": g["MAE"].mean(),
                    "RMSSD_R_median_by_participant": g["R"].median(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values(["dataset_label", "device"])
    )

    # 下面开始组装 Markdown 报告正文：先说明 motion threshold 和方法，
    # 再给汇总、明细、阈值比较和解释。
    lines = [
        "# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30",
        "",
        "本报告只在指定 motion threshold 子集上重新计算当前最佳 raw-aligned baseline。",
        "",
        "## Motion Threshold 定义",
        "",
        f"- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < {NO_MOTION_THRESHOLD}`。",
        "- `accel_motion_mean_mag` 是数据集内保存的去重力 motion magnitude 均值。",
        "- 过滤发生在 device 层面：同一个 window 对 Earring 可能是 no-motion，对 Ring/Watch 不一定是 no-motion。",
        "",
        "## Baseline 定义",
        "",
        "- 方法：`no_motion_device_best_sqi_green_or_ir__hrv_from_ppg`。",
        "- 对 no-motion 子集中的每个 window、每个 device 独立评估 `ppg_green` 与 `ppg_ir`。",
        "- 使用 `heuristic_baselines.algorithms.hrv.hrv_from_ppg` 从 PPG 重新计算 PRV/HRV。",
        "- 若 green 和 IR 都有效，则仅用 PPG 自身信息选择 SQI 更高的一路；不使用 ECG label 选通道。",
        "",
        "## 数据集",
        "",
        "| Dataset | Total windows | No-motion windows | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for spec in DATASETS:
        one = rmssd[rmssd["dataset_label"] == spec["label"]]
        total_windows = int(one[["participant", "device", "total_windows"]].drop_duplicates()["total_windows"].sum())
        no_motion_windows = int(one[["participant", "device", "n_no_motion_windows"]].drop_duplicates()["n_no_motion_windows"].sum())
        lines.append(f"| `{spec['label']}` | {total_windows} | {no_motion_windows} | {spec['interpretation']} |")

    lines.extend(
        [
            "",
            "## Dataset x Device 汇总（RMSSD）",
            "",
            _md_table(
                dataset_device,
                [
                    "dataset_label",
                    "device",
                    "n_participants",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "RMSSD_MAE_mean_by_participant",
                    "RMSSD_R_median_by_participant",
                ],
                [
                    "Dataset",
                    "Device",
                    "Participants",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "Mean participant RMSSD MAE",
                    "Median participant R",
                ],
            ),
            "",
            "## 每个参与者 x 每个设备结果（RMSSD）",
            "",
            _md_table(
                _sort_participant_frame(rmssd),
                [
                    "dataset_label",
                    "participant",
                    "device",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "MAE",
                    "RMSE",
                    "R",
                    "bias",
                    "mean_accel_mean_mag_no_motion",
                    "mean_accel_motion_mean_mag_no_motion",
                    "selected_ppg_green",
                    "selected_ppg_ir",
                ],
                [
                    "Dataset",
                    "Participant",
                    "Device",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "MAE ms",
                    "RMSE ms",
                    "R",
                    "Bias ms",
                    "Raw accel no-motion",
                    "Motion accel no-motion",
                    "Green selected",
                    "IR selected",
                ],
            ),
            "",
            "## 每个参与者 x 每个设备结果（SDNN）",
            "",
            _md_table(
                _sort_participant_frame(sdnn),
                [
                    "dataset_label",
                    "participant",
                    "device",
                    "total_windows",
                    "n_no_motion_windows",
                    "n_selected_windows",
                    "no_motion_pct",
                    "coverage_pct_within_no_motion",
                    "MAE",
                    "RMSE",
                    "R",
                    "bias",
                ],
                [
                    "Dataset",
                    "Participant",
                    "Device",
                    "Total windows",
                    "No-motion windows",
                    "Valid preds",
                    "No-motion %",
                    "Coverage within no-motion %",
                    "MAE ms",
                    "RMSE ms",
                    "R",
                    "Bias ms",
                ],
            ),
            *_comparison_section(df, NO_MOTION_THRESHOLD, report_csv, output_dir),
            "",
            "## 重要解析",
            "",
            "- 这个报告回答的是“在数据集内低运动窗口中，当前 PPG heuristic baseline 表现如何”。",
            "- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。",
            "- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。",
            "- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。",
            "- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。",
            "- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。",
            "",
            "## 输出文件",
            "",
            f"- Markdown report: `{report_md.name}`",
            f"- Machine-readable table: `{report_csv.name}`",
            "",
            f"Generated in {time.time() - start:.1f} seconds.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {report_md}")
    print(f"[saved] {report_csv}")


if __name__ == "__main__":
    main()

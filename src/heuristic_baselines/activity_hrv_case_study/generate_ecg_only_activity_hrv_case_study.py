#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hrv_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 让脚本可以复用 heuristic_baselines 目录和当前目录中的辅助模块。
THIS_DIR = Path(__file__).resolve().parent
HEURISTIC_DIR = THIS_DIR.parent
sys.path.insert(0, str(HEURISTIC_DIR))
sys.path.insert(0, str(THIS_DIR))

# 复用共同窗口脚本里的 activity 解析/报告工具，以及数据集生成脚本里的 ECG label 逻辑。
import generate_activity_hrv_case_study as activity_case
import generate_rawaligned_4device_dataset as rawaligned


# 默认从临时 raw 下载目录读取 Polar ECG，从本目录写出 case-study 结果。
DEFAULT_RAW_ROOT = Path("/private/tmp/hrv_raw_gap_diagnosis/raw_data")
DEFAULT_LOG_DIR = THIS_DIR / "source_logs" / "raw_data"
DEFAULT_OUTPUT_DIR = THIS_DIR

# ECG-only 分析默认使用 5 分钟窗口、30 秒 stride，并沿用严格 ECG reference 条件。
DEFAULT_WINDOW_SEC = 300
DEFAULT_STRIDE_SEC = 30
DEFAULT_TARGET_FS = 100.0
DEFAULT_MIN_ECG_VALID_IBI_RATIO = 1.0
DEFAULT_MIN_ECG_VALID_SAMPLE_RATIO = 1.0
DEFAULT_MIN_HR_BPM = 30.0
DEFAULT_MAX_HR_BPM = 220.0
DEFAULT_PLOT_BREAK_GAP_MIN = 5.0


# 统一调用共同脚本中的显示时间转换，保证 Local Time 一致。
def display_dt(ms: float, offset: timedelta) -> datetime:
    return activity_case.display_dt(ms, offset)


# 从 raw Polar ECG 直接切 5 分钟窗口，并只保留 ECG-QC 通过的窗口。
def make_ecg_only_windows(
    ecg_raw: dict[str, np.ndarray],
    participant: str,
    day: str,
    display_offset: timedelta,
    *,
    window_sec: int,
    stride_sec: int,
    target_fs: float,
    min_valid_ibi_ratio: float,
    min_ecg_valid_sample_ratio: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    # 使用 activity log 的 Local Time 偏移来定义当天边界。
    day_start = datetime.strptime(day, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    day_start_ms = activity_case.local_day_epoch_ms(day_start, display_offset)
    day_end_ms = activity_case.local_day_epoch_ms(day_end, display_offset)

    # 候选窗口必须同时落在目标日期和 raw ECG 实际记录范围内。
    raw_t = np.asarray(ecg_raw["time_ms"], dtype=np.float64)
    ecg_start_ms = float(raw_t[0])
    ecg_end_ms = float(raw_t[-1])
    window_ms = float(window_sec * 1000)
    stride_ms = float(stride_sec * 1000)
    first_t0 = math.ceil(max(day_start_ms, ecg_start_ms) / stride_ms) * stride_ms
    last_t0 = min(day_end_ms, ecg_end_ms) - window_ms
    if last_t0 < first_t0:
        raise RuntimeError(f"No ECG-only candidate windows for {participant} on {day}")

    # 以固定 stride 枚举候选窗口，再调用现有 ECG label/QC 函数。
    t0_all = np.arange(first_t0, last_t0 + 0.5 * stride_ms, stride_ms, dtype=np.float64)
    t1_all = t0_all + window_ms
    ecg_fs = rawaligned._ecg_fs_from_raw(ecg_raw)

    rows: list[dict[str, object]] = []
    fail_reasons: dict[str, int] = {}
    for wi, (t0, t1) in enumerate(zip(t0_all, t1_all)):
        label = rawaligned._ecg_label_for_window(
            ecg_raw,
            ecg_fs,
            t0=float(t0),
            t1=float(t1),
            target_fs=target_fs,
            min_valid_ibi_ratio=min_valid_ibi_ratio,
            min_hr_bpm=min_hr_bpm,
            max_hr_bpm=max_hr_bpm,
            min_ecg_valid_sample_ratio=min_ecg_valid_sample_ratio,
            detector_method="window_pantompkins1985",
        )
        # QC 不通过的窗口不进入 HRV 曲线，但保留失败原因计数用于报告。
        if not bool(label["ecg_label_qc_pass"]):
            reason = str(label["ecg_label_qc_reason"])
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            continue
        center = (float(t0) + float(t1)) / 2.0
        # 通过 QC 的窗口写成和共同窗口脚本兼容的表结构。
        rows.append(
            {
                "participant": participant,
                "window_idx": wi,
                "t0_ms": int(t0),
                "t1_ms": int(t1),
                "center_ms": int(center),
                "center_time": display_dt(center, display_offset).strftime("%Y-%m-%d %H:%M:%S"),
                "ecg_rmssd_ms": float(label["ecg_rmssd_ms"]),
                "ecg_sdnn_ms": float(label["ecg_sdnn_ms"]),
                "ecg_hr_bpm": float(label["ecg_hr_bpm"]),
                "ecg_label_qc_pass": bool(label["ecg_label_qc_pass"]),
                "ecg_valid_sample_ratio": float(label["ecg_valid_sample_ratio"]),
                "ecg_valid_ibi_ratio": float(label["ecg_valid_ibi_ratio"]),
                "ecg_ibi_correction_ratio": float(label["ecg_ibi_correction_ratio"]),
                "ecg_qrs_sqi": float(label["ecg_qrs_sqi"]),
                "ecg_peak_count": int(label["ecg_peak_count"]),
                "ecg_label_qc_reason": str(label["ecg_label_qc_reason"]),
                "accel_motion_mean_mag": np.nan,
            }
        )

    if not rows:
        raise RuntimeError(f"No ECG-QC pass windows for {participant} on {day}")

    # 记录 ECG-only 生成过程的覆盖率和失败原因，方便解释空白区。
    summary = {
        "participant": participant,
        "day": day,
        "source": "raw Polar ECG only",
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "target_fs": target_fs,
        "ecg_fs": ecg_fs,
        "n_ecg_candidate_windows": int(t0_all.size),
        "n_ecg_qc_pass": len(rows),
        "ecg_qc_pass_pct": 100.0 * len(rows) / int(t0_all.size),
        "n_ecg_qc_fail": int(t0_all.size) - len(rows),
        "ecg_qc_fail_reasons_json": json.dumps(fail_reasons, ensure_ascii=False, sort_keys=True),
    }
    return pd.DataFrame(rows), summary


# 相邻 HRV 点间隔过大时插入断点，避免缺失数据被画成长直线。
def line_series_with_breaks(x: pd.Series, y: pd.Series, break_gap_min: float) -> pd.Series:
    gap_breaks = x.diff().dt.total_seconds().fillna(0) > break_gap_min * 60.0
    return y.mask(gap_breaks)


# 绘制 ECG-only 版本图像：只包含 ECG-derived RMSSD、SDNN、HR，不画设备 motion。
def make_ecg_only_plot(
    df: pd.DataFrame,
    intervals: list[activity_case.Interval],
    point_events: list[activity_case.Event],
    out_path: Path,
    participant: str,
    day: str,
    display_offset: timedelta,
    *,
    break_gap_min: float,
) -> None:
    # ECG-only 窗口已经在生成阶段通过 QC，因此这里直接画所有行。
    x = pd.to_datetime(df["center_time"])
    rmssd = line_series_with_breaks(x, df["ecg_rmssd_ms"], break_gap_min)
    sdnn = line_series_with_breaks(x, df["ecg_sdnn_ms"], break_gap_min)
    hr = line_series_with_breaks(x, df["ecg_hr_bpm"], break_gap_min)
    rmssd_smooth = line_series_with_breaks(x, df["ecg_rmssd_ms"].rolling(15, min_periods=3, center=True).median(), break_gap_min)

    # 与共同窗口图保持一致的 activity 背景颜色。
    colors = {
        "walking": "#4E79A7",
        "exercise": "#E15759",
        "transport": "#9C755F",
        "eating_drinking": "#F28E2B",
        "stress": "#B07AA1",
        "work_study": "#59A14F",
        "rest_sitting": "#76B7B2",
        "social_entertainment": "#EDC948",
        "personal_care": "#BAB0AC",
        "sleep": "#1F2937",
        "other": "#A0CBE8",
    }

    # 背景色表示 activity interval，虚线表示 point/unpaired activity event。
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for ax in axes:
        for interval in intervals:
            start = display_dt(interval.start_ms, display_offset)
            end = display_dt(interval.end_ms, display_offset)
            ax.axvspan(start, end, color=colors.get(interval.category, "#D3D3D3"), alpha=0.16, linewidth=0)

    # 三个子图分别展示 RMSSD、SDNN 和 ECG heart rate。
    axes[0].plot(x, rmssd, color="#1f77b4", linewidth=1.0, alpha=0.65, label="RMSSD")
    axes[0].plot(x, rmssd_smooth, color="#0b3d91", linewidth=2.0, label="RMSSD rolling median")
    axes[0].set_ylabel("RMSSD (ms)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, sdnn, color="#2ca02c", linewidth=1.1)
    axes[1].set_ylabel("SDNN (ms)")

    axes[2].plot(x, hr, color="#d62728", linewidth=1.1)
    axes[2].set_ylabel("ECG HR (bpm)")
    axes[2].set_xlabel("Local time")

    # 横轴只覆盖 ECG-QC 合格窗口范围，避免展示不可解释的全天空白。
    if len(x):
        x_min = x.min() - pd.Timedelta(minutes=15)
        x_max = x.max() + pd.Timedelta(minutes=15)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    # 单点/未配对 activity 事件用灰色虚线标出。
    for event in point_events:
        dt = display_dt(event.ts_ms, display_offset)
        for ax in axes:
            ax.axvline(dt, color="#333333", alpha=0.25, linewidth=0.8, linestyle="--")

    # 只显示实际出现过的 activity 类别。
    legend_handles = []
    for cat, color in colors.items():
        if any(i.category == cat for i in intervals):
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.25, label=cat))
    if legend_handles:
        axes[0].legend(handles=[*axes[0].get_legend_handles_labels()[0], *legend_handles], loc="upper left", ncol=4, fontsize=8)

    fig.suptitle(f"{participant} ECG-only HRV aligned with activity log ({day})", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# 写出 ECG-only 版本中文 Markdown 报告。
def write_ecg_only_report(
    out_path: Path,
    participant: str,
    day: str,
    df: pd.DataFrame,
    activity_summary: pd.DataFrame,
    intervals: list[activity_case.Interval],
    point_events: list[activity_case.Event],
    plot_name: str,
    csv_name: str,
    display_offset: timedelta,
    generation_summary: dict[str, int | float | str],
) -> None:
    # ECG-only 表中已经只保留 QC 通过窗口，这里直接计算日内 HRV 统计。
    rmssd = df["ecg_rmssd_ms"].dropna()
    sdnn = df["ecg_sdnn_ms"].dropna()
    hr = df["ecg_hr_bpm"].dropna()
    rmssd_diff = rmssd.diff().abs().dropna()
    rmssd_30m = rmssd.diff(60).abs().dropna() if len(rmssd) > 60 else pd.Series(dtype=float)
    unlabeled_pct = 100.0 * float((df["activity_category"] == "unlabeled").mean()) if len(df) else math.nan
    first_center = str(df["center_time"].iloc[0]) if len(df) else "n/a"
    last_center = str(df["center_time"].iloc[-1]) if len(df) else "n/a"
    blocks = activity_case.unlabeled_blocks(df)
    min_t0 = int(df["t0_ms"].min()) if len(df) else None
    max_t1 = int(df["t1_ms"].max()) if len(df) else None
    # 只展示落在 ECG-only HRV 覆盖范围内的 point/unpaired event。
    relevant_point_events = [event for event in point_events if min_t0 is not None and max_t1 is not None and min_t0 <= event.ts_ms < max_t1]

    # 组装中文报告正文，结构与共同窗口版本保持一致。
    lines = [
        f"# {participant} ECG-only HRV 与活动日志对齐案例分析（{day}）",
        "",
        "## 目的",
        "",
        "使用 raw Polar ECG 单独计算 5 分钟 HRV 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本版本只要求 ECG label QC 通过，不要求 Earring/Ring/Watch 三设备 raw boundary aligned，也不要求 PPG sample coverage。",
        "",
        "## 输入数据",
        "",
        "- HRV 来源：`Multisite-PPG/raw_data` 中的 raw Polar ECG",
        f"- 活动日志来源：`Multisite-PPG/raw_data/{participant}/{participant}_activity_log.txt`",
        "- 窗口设置：5 min window，stride 30 s",
        "- ECG 保留条件：`ecg_label_qc_pass == True`",
        "",
        "## 输出文件",
        "",
        f"- window-level sidecar CSV：`{csv_name}`",
        f"- 可视化图：`{plot_name}`",
        "",
        "## 数据质量",
        "",
        f"- 当天 ECG-only 候选窗口数：{generation_summary['n_ecg_candidate_windows']}",
        f"- ECG-QC 合格窗口数：{generation_summary['n_ecg_qc_pass']} ({generation_summary['ecg_qc_pass_pct']:.1f}%)",
        f"- ECG-QC 不合格窗口数：{generation_summary['n_ecg_qc_fail']}",
        f"- ECG-QC 合格窗口中心时间范围：{first_center} 到 {last_center}",
        f"- 当天解析出的 activity interval 数：{len(intervals)}",
        f"- HRV 覆盖时间内的 point/unpaired activity event 数：{len(relevant_point_events)}",
        f"- 未匹配到 activity 的窗口比例：{unlabeled_pct:.1f}%",
        "",
        "这个 ECG-only 版本用于回答全天 ECG-derived HRV 与 activity log 的关系；它不适合直接作为 PPG model training dataset，因为没有要求三设备 PPG 同时可用。",
        "",
        "## 单日 HRV 波动范围",
        "",
        f"- RMSSD 中位数/IQR：{rmssd.median():.2f} ms / {(rmssd.quantile(0.75) - rmssd.quantile(0.25)):.2f} ms" if not rmssd.empty else "- RMSSD 中位数/IQR：n/a",
        f"- RMSSD 最小值/最大值：{rmssd.min():.2f} ms / {rmssd.max():.2f} ms" if not rmssd.empty else "- RMSSD 最小值/最大值：n/a",
        f"- 相邻窗口之间最大的 RMSSD 变化：{rmssd_diff.max():.2f} ms" if not rmssd_diff.empty else "- 相邻窗口之间最大的 RMSSD 变化：n/a",
        f"- 约 30 分钟内最大的 RMSSD 变化：{rmssd_30m.max():.2f} ms" if not rmssd_30m.empty else "- 约 30 分钟内最大的 RMSSD 变化：n/a",
        f"- SDNN 中位数/IQR：{sdnn.median():.2f} ms / {(sdnn.quantile(0.75) - sdnn.quantile(0.25)):.2f} ms" if not sdnn.empty else "- SDNN 中位数/IQR：n/a",
        f"- ECG HR 中位数/IQR：{hr.median():.2f} bpm / {(hr.quantile(0.75) - hr.quantile(0.25)):.2f} bpm" if not hr.empty else "- ECG HR 中位数/IQR：n/a",
        "",
        "## 按活动类别汇总",
        "",
        activity_case.md_table(activity_summary),
        "",
        "## 活动粗分类与具体事件对应表",
        "",
        "下表列出 ECG-only HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。",
        "",
        activity_case.activity_mapping_table(intervals, point_events, min_t0, max_t1, display_offset),
        "",
        "## 未匹配 Activity 的窗口",
        "",
        activity_case.unlabeled_block_table(blocks),
        "",
        "## Point/Unpaired Activity Events",
        "",
        activity_case.point_event_table(relevant_point_events),
        "",
        "## 图像",
        "",
        f"图中相邻 HRV 点间隔超过 {DEFAULT_PLOT_BREAK_GAP_MIN:.0f} 分钟时会自动断线，避免把缺失数据误画成长直线。",
        "",
        "断线主要来自相邻 ECG-QC 合格窗口间隔过大，常见原因包括 raw ECG sample 不完整、R peak/IBI 质量不足，或 HRV 计算未通过 QC。",
        "",
        f"![{participant} ECG-only HRV activity plot]({plot_name})",
        "",
        "## 解读注意事项",
        "",
        "- 这个版本只回答 ECG-derived HRV 的日内变化和 activity log 的对应关系。",
        "- 与 raw-aligned training dataset 相比，它不会因为 Earring/Ring/Watch PPG 边界不齐而丢掉 ECG 可用窗口。",
        "- activity category 来自自由文本规则归类，适合探索候选关系，不作为因果检验。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# 生成单个 participant/day 的 ECG-only CSV、summary、图像和 Markdown。
def generate_one(args: argparse.Namespace) -> None:
    participant = args.participant
    activity_log = args.activity_log or DEFAULT_LOG_DIR / participant / f"{participant}_activity_log.txt"
    ecg_path = args.raw_root / participant / f"{participant}_polar_ecg_raw.npz"
    if not activity_log.is_file():
        raise FileNotFoundError(activity_log)
    if not ecg_path.is_file():
        raise FileNotFoundError(ecg_path)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取并解析 activity log，得到当天 activity interval 和 point/unpaired event。
    events = activity_case.read_activity_log(activity_log)
    display_offset = activity_case.infer_activity_local_offset(events)
    day_start = datetime.strptime(args.day, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    day_start_ms = activity_case.local_day_epoch_ms(day_start, display_offset)
    day_end_ms = activity_case.local_day_epoch_ms(day_end, display_offset)
    intervals, point_events = activity_case.events_to_intervals(events, day_start_ms, day_end_ms)

    # 读取 raw Polar ECG，并计算 ECG-only HRV 窗口。
    z = np.load(ecg_path, allow_pickle=True)
    ecg_raw = {k: z[k] for k in z.files}
    windows, generation_summary = make_ecg_only_windows(
        ecg_raw,
        participant,
        args.day,
        display_offset,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        target_fs=args.target_fs,
        min_valid_ibi_ratio=args.ecg_min_valid_ibi_ratio,
        min_ecg_valid_sample_ratio=args.min_ecg_valid_sample_ratio,
        min_hr_bpm=args.min_hr_bpm,
        max_hr_bpm=args.max_hr_bpm,
    )

    # 将 ECG-only HRV 窗口和 activity interval 对齐，并按活动类别汇总。
    aligned = activity_case.annotate_windows(windows, intervals, point_events)
    activity_summary = activity_case.summarize_by_activity(aligned)

    stem = f"{participant}_{args.day}_activity_hrv_ecg_only_stride{args.stride_sec}"
    csv_path = out_dir / f"{stem}_window_aligned.csv"
    plot_path = out_dir / f"{stem}_plot.png"
    md_path = out_dir / f"{stem}_case_study.md"

    # 写出窗口级 sidecar、图像和 Markdown；生成过程/activity 汇总只保留在 MD 中。
    aligned.to_csv(csv_path, index=False)
    make_ecg_only_plot(
        aligned,
        intervals,
        point_events,
        plot_path,
        participant,
        args.day,
        display_offset,
        break_gap_min=args.plot_break_gap_min,
    )
    write_ecg_only_report(
        md_path,
        participant,
        args.day,
        aligned,
        activity_summary,
        intervals,
        point_events,
        plot_path.name,
        csv_path.name,
        display_offset,
        generation_summary,
    )
    print(f"[saved] {csv_path}")
    print(f"[saved] {plot_path}")
    print(f"[saved] {md_path}")


# 命令行入口：配置 participant/day、ECG QC 阈值和断线阈值。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--activity-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--stride-sec", type=int, default=DEFAULT_STRIDE_SEC)
    parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS)
    parser.add_argument("--ecg-min-valid-ibi-ratio", type=float, default=DEFAULT_MIN_ECG_VALID_IBI_RATIO)
    parser.add_argument("--min-ecg-valid-sample-ratio", type=float, default=DEFAULT_MIN_ECG_VALID_SAMPLE_RATIO)
    parser.add_argument("--min-hr-bpm", type=float, default=DEFAULT_MIN_HR_BPM)
    parser.add_argument("--max-hr-bpm", type=float, default=DEFAULT_MAX_HR_BPM)
    parser.add_argument("--plot-break-gap-min", type=float, default=DEFAULT_PLOT_BREAK_GAP_MIN)
    args = parser.parse_args()
    generate_one(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hrv_matplotlib_cache")

import numpy as np
import pandas as pd

# 绘图依赖是可选的：已有图片时可以只更新 CSV/MD，不强制安装 matplotlib。
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


# 默认读取已经生成好的三设备共同窗口 stride30 数据集。
DEFAULT_DATASET_DIR = Path(
    "/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/heuristic_baselines/outputs/"
    "synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30"
)
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "source_logs" / "raw_data"
DEFAULT_PLOT_BREAK_GAP_MIN = 5.0


# 原始 activity log 逐行解析后的事件结构。
@dataclass
class Event:
    ts_ms: int
    local_time: str
    text: str
    category: str
    kind: str


# 可与 HRV 窗口重叠计算的 start-stop 活动区间。
@dataclass
class Interval:
    start_ms: int
    end_ms: int
    category: str
    raw_text: str


# Unix epoch ms 与本地显示时间之间的基础转换。
def local_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0)


def display_dt(ms: float, offset: timedelta) -> datetime:
    return local_dt(ms) + offset


def epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def local_day_epoch_ms(dt: datetime, offset: timedelta) -> int:
    return epoch_ms(dt - offset)


# 统一 activity 文本格式，方便做规则匹配。
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# 将自由文本 activity log 粗分成少数可汇总类别。
def categorize_activity(text: str) -> str:
    t = normalize_text(text)
    if any(w in t for w in ["sleep", "nap", "napping"]):
        return "sleep"
    if any(w in t for w in ["yoga", "breathing"]):
        return "rest_sitting"
    if "incline walk" in t:
        return "walking"
    if any(w in t for w in ["run", "treadmill", "exercise"]):
        return "exercise"
    if any(w in t for w in ["walk", "walking"]):
        return "walking"
    if any(w in t for w in ["drive", "driving", "drivign", "light rail", "bus"]):
        return "transport"
    if any(w in t for w in ["eat", "dinner", "ramen", "dessert", "coffee", "water", "restaurant", "tangerine", "cooking", "warming up", "lunch", "food"]):
        return "eating_drinking"
    if any(w in t for w in ["stress", "stressed"]):
        return "stress"
    if any(w in t for w in ["homework", "study", "stduy", "working", "work", "laptop", "writing", "class", "meeting", "zoom"]):
        return "work_study"
    if any(w in t for w in ["sitting", "sit", "laying", "bed", "couch"]):
        return "rest_sitting"
    if any(w in t for w in ["chat", "talk", "music", "mysic", "youtube", "youtueb", "watching", "drama", "phone", "relax", "play"]):
        return "social_entertainment"
    if any(w in t for w in ["hair", "pajama", "changing", "peeling"]):
        return "personal_care"
    return "other"


# 判断一条 activity log 是开始、结束，还是单点/未配对事件。
def event_kind(text: str) -> str:
    t = normalize_text(text)
    stop_prefixes = ("stop", "stopped", "end", "ended", "exit", "exited", "completed")
    start_prefixes = ("start", "started", "begin", "began", "enter", "entered")
    if t.startswith(stop_prefixes) or "has ended" in t:
        return "stop"
    if t.startswith(start_prefixes) or t.startswith("seated") or t.startswith("mood is"):
        return "start"
    if t.startswith(("sitting", "laying", "listening")):
        return "start"
    return "point"


# 读取原始 activity log，并补上粗分类和事件类型。
def read_activity_log(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts_ms = int(row[0].strip())
            except ValueError:
                continue
            text = ",".join(row[2:]).strip()
            events.append(
                Event(
                    ts_ms=ts_ms,
                    local_time=row[1].strip(),
                    text=text,
                    category=categorize_activity(text),
                    kind=event_kind(text),
                )
            )
    return sorted(events, key=lambda e: e.ts_ms)


# 通过 log 里写出的 Local Time 反推显示时区偏移，保证图和日志时间一致。
def infer_activity_local_offset(events: list[Event]) -> timedelta:
    offsets: list[float] = []
    for event in events:
        try:
            log_dt = datetime.strptime(event.local_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        offsets.append((log_dt - local_dt(event.ts_ms)).total_seconds())
    if not offsets:
        return timedelta(0)
    return timedelta(seconds=int(round(float(np.median(offsets)))))


# 把 start/stop 事件配对成区间；无法配对的记录保留为 point/unpaired event。
def events_to_intervals(events: list[Event], day_start_ms: int, day_end_ms: int) -> tuple[list[Interval], list[Event]]:
    intervals: list[Interval] = []
    point_events: list[Event] = []
    open_by_category: dict[str, Event] = {}

    for event in events:
        if event.ts_ms < day_start_ms - 12 * 3600_000 or event.ts_ms > day_end_ms + 12 * 3600_000:
            continue
        if event.kind == "start":
            if event.category in open_by_category:
                prev = open_by_category[event.category]
                if event.ts_ms > prev.ts_ms:
                    intervals.append(Interval(prev.ts_ms, event.ts_ms, prev.category, prev.text))
            open_by_category[event.category] = event
        elif event.kind == "stop":
            start_event = open_by_category.pop(event.category, None)
            if start_event is not None and event.ts_ms > start_event.ts_ms:
                intervals.append(
                    Interval(
                        start_ms=start_event.ts_ms,
                        end_ms=event.ts_ms,
                        category=start_event.category,
                        raw_text=f"{start_event.text} -> {event.text}",
                    )
                )
            else:
                point_events.append(event)
        else:
            point_events.append(event)

    for start_event in open_by_category.values():
        capped_end = min(day_end_ms, start_event.ts_ms + 4 * 3600_000)
        if capped_end > start_event.ts_ms:
            intervals.append(Interval(start_event.ts_ms, capped_end, start_event.category, start_event.text + " -> [unclosed]"))

    intervals = [
        Interval(max(i.start_ms, day_start_ms), min(i.end_ms, day_end_ms), i.category, i.raw_text)
        for i in intervals
        if min(i.end_ms, day_end_ms) > max(i.start_ms, day_start_ms)
    ]
    point_events = [e for e in point_events if day_start_ms <= e.ts_ms < day_end_ms]
    return intervals, point_events


# 计算 HRV 窗口和 activity interval 的重叠秒数。
def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / 1000.0


# 从 training_v1_stride30 NPZ 中抽取指定参与者/日期的共同窗口 HRV 表。
def build_window_table(npz_path: Path, participant: str, day: str, display_offset: timedelta) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    t0 = z["t0_ms"].astype(float)
    t1 = z["t1_ms"].astype(float)
    center = (t0 + t1) / 2.0
    local_times = [display_dt(x, display_offset) for x in center]
    target_day = datetime.strptime(day, "%Y-%m-%d").date()
    mask = np.array([dt.date() == target_day for dt in local_times])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise RuntimeError(f"No windows found for {participant} on {day}")

    # 兼容旧版/新版 accel_motion_mean_mag 的形状：按窗口得到一个 motion 数值。
    accel_motion = np.asarray(z["accel_motion_mean_mag"], dtype=float)
    if accel_motion.ndim == 2:
        accel_motion_mean = np.nanmean(accel_motion, axis=1)
    else:
        accel_motion_mean = accel_motion

    return pd.DataFrame(
        {
            "participant": participant,
            "window_idx": idx,
            "t0_ms": t0[idx].astype(np.int64),
            "t1_ms": t1[idx].astype(np.int64),
            "center_ms": center[idx].astype(np.int64),
            "center_time": [local_times[i].strftime("%Y-%m-%d %H:%M:%S") for i in idx],
            "ecg_rmssd_ms": np.asarray(z["ecg_rmssd_ms"], dtype=float)[idx],
            "ecg_sdnn_ms": np.asarray(z["ecg_sdnn_ms"], dtype=float)[idx],
            "ecg_hr_bpm": np.asarray(z["ecg_hr_bpm"], dtype=float)[idx],
            "ecg_label_qc_pass": np.asarray(z["ecg_label_qc_pass"], dtype=bool)[idx],
            "ecg_valid_ibi_ratio": np.asarray(z["ecg_valid_ibi_ratio"], dtype=float)[idx],
            "ecg_ibi_correction_ratio": np.asarray(z["ecg_ibi_correction_ratio"], dtype=float)[idx],
            "ecg_qrs_sqi": np.asarray(z["ecg_qrs_sqi"], dtype=float)[idx],
            "ecg_peak_count": np.asarray(z["ecg_peak_count"], dtype=int)[idx],
            "accel_motion_mean_mag": accel_motion_mean[idx],
        }
    )


# 给每个 HRV 窗口标注重叠最多的 activity 类别，并保留原始重叠细节。
def annotate_windows(df: pd.DataFrame, intervals: list[Interval], point_events: list[Event]) -> pd.DataFrame:
    categories = []
    raw_labels = []
    overlaps = []
    point_labels = []
    for row in df.itertuples(index=False):
        overlap_by_cat: dict[str, float] = defaultdict(float)
        raw_parts: list[str] = []
        for interval in intervals:
            sec = overlap_seconds(row.t0_ms, row.t1_ms, interval.start_ms, interval.end_ms)
            if sec > 0:
                overlap_by_cat[interval.category] += sec
                raw_parts.append(f"{interval.category}:{sec:.0f}s:{interval.raw_text}")
        if overlap_by_cat:
            cat, sec = max(overlap_by_cat.items(), key=lambda kv: kv[1])
        else:
            cat, sec = "unlabeled", 0.0
        pts = [e.text for e in point_events if row.t0_ms <= e.ts_ms < row.t1_ms]
        categories.append(cat)
        raw_labels.append(" | ".join(raw_parts))
        overlaps.append(sec)
        point_labels.append(" | ".join(pts))
    out = df.copy()
    out["activity_category"] = categories
    out["activity_overlap_sec"] = overlaps
    out["activity_raw"] = raw_labels
    out["activity_point_events"] = point_labels
    return out


# 只用 ECG-QC 合格且有 activity 标签的窗口，按活动类别汇总 HRV/motion。
def summarize_by_activity(df: pd.DataFrame) -> pd.DataFrame:
    good = df[
        (df["ecg_label_qc_pass"])
        & (df["ecg_valid_ibi_ratio"] >= 1.0)
        & (df["ecg_ibi_correction_ratio"] <= 0.2)
        & (df["activity_category"] != "unlabeled")
    ].copy()
    if good.empty:
        return pd.DataFrame()
    return (
        good.groupby("activity_category", dropna=False)
        .agg(
            n_windows=("window_idx", "count"),
            total_overlap_min=("activity_overlap_sec", lambda x: float(np.nansum(x)) / 60.0),
            rmssd_median_ms=("ecg_rmssd_ms", "median"),
            rmssd_iqr_ms=("ecg_rmssd_ms", lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))),
            sdnn_median_ms=("ecg_sdnn_ms", "median"),
            hr_median_bpm=("ecg_hr_bpm", "median"),
            motion_median=("accel_motion_mean_mag", "median"),
        )
        .reset_index()
        .sort_values(["n_windows", "activity_category"], ascending=[False, True])
    )


# 绘制共同窗口版本的 HRV/activity 图；长时间缺口会断线，避免误导。
def make_plot(
    df: pd.DataFrame,
    intervals: list[Interval],
    point_events: list[Event],
    out_path: Path,
    participant: str,
    day: str,
    display_offset: timedelta,
    break_gap_min: float = DEFAULT_PLOT_BREAK_GAP_MIN,
) -> None:
    if not HAS_MATPLOTLIB:
        if out_path.is_file():
            return
        raise RuntimeError("matplotlib is required to create a new plot.")

    # ECG-QC 不合格窗口不画 HRV 值。
    x = pd.to_datetime(df["center_time"])
    good = (
        (df["ecg_label_qc_pass"])
        & (df["ecg_valid_ibi_ratio"] >= 1.0)
        & (df["ecg_ibi_correction_ratio"] <= 0.2)
    )
    rmssd = df["ecg_rmssd_ms"].where(good)
    sdnn = df["ecg_sdnn_ms"].where(good)
    hr = df["ecg_hr_bpm"].where(good)
    rmssd_smooth = rmssd.rolling(15, min_periods=3, center=True).median()
    # 相邻窗口间隔过大时插入断点，防止缺失段被画成直线。
    gap_breaks = x.diff().dt.total_seconds().fillna(0) > break_gap_min * 60.0
    rmssd = rmssd.mask(gap_breaks)
    sdnn = sdnn.mask(gap_breaks)
    hr = hr.mask(gap_breaks)
    rmssd_smooth = rmssd_smooth.mask(gap_breaks)
    motion = df["accel_motion_mean_mag"].mask(gap_breaks)

    # 每个 activity 粗分类使用固定颜色，便于跨图比较。
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

    # 用半透明背景区间表示 activity interval。
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for ax in axes:
        for interval in intervals:
            start = display_dt(interval.start_ms, display_offset)
            end = display_dt(interval.end_ms, display_offset)
            color = colors.get(interval.category, "#D3D3D3")
            ax.axvspan(start, end, color=color, alpha=0.16, linewidth=0)

    # 上三行画 ECG-derived HRV/HR，最后一行画共同窗口 motion magnitude。
    axes[0].plot(x, rmssd, color="#1f77b4", linewidth=1.0, alpha=0.65, label="RMSSD")
    axes[0].plot(x, rmssd_smooth, color="#0b3d91", linewidth=2.0, label="RMSSD rolling median")
    axes[0].set_ylabel("RMSSD (ms)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, sdnn, color="#2ca02c", linewidth=1.1)
    axes[1].set_ylabel("SDNN (ms)")

    axes[2].plot(x, hr, color="#d62728", linewidth=1.1)
    axes[2].set_ylabel("ECG HR (bpm)")

    axes[3].plot(x, motion, color="#555555", linewidth=1.0)
    axes[3].set_ylabel("Motion mag")
    axes[3].set_xlabel("Local time")
    # 坐标轴只覆盖 HRV 有数据的范围，避免显示无法分析的全天空白区。
    if len(x):
        x_min = x.min() - pd.Timedelta(minutes=15)
        x_max = x.max() + pd.Timedelta(minutes=15)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    # 单点/未配对 activity 事件用虚线标记。
    for event in point_events:
        dt = display_dt(event.ts_ms, display_offset)
        for ax in axes:
            ax.axvline(dt, color="#333333", alpha=0.25, linewidth=0.8, linestyle="--")

    # 只把实际出现过的 activity 类别放进图例。
    legend_handles = []
    for cat, color in colors.items():
        if any(i.category == cat for i in intervals):
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.25, label=cat))
    if legend_handles:
        axes[0].legend(handles=[*axes[0].get_legend_handles_labels()[0], *legend_handles], loc="upper left", ncol=4, fontsize=8)
    fig.suptitle(f"{participant} ECG-derived HRV aligned with activity log ({day})", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# 把 pandas 表转成 Markdown 表，并统一常用列名的中文显示。
def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_没有可显示的行。_"
    show = df.head(max_rows).copy()
    rename = {
        "activity_category": "活动类别",
        "n_windows": "窗口数",
        "total_overlap_min": "重叠分钟数",
        "rmssd_median_ms": "RMSSD中位数_ms",
        "rmssd_iqr_ms": "RMSSD_IQR_ms",
        "sdnn_median_ms": "SDNN中位数_ms",
        "hr_median_bpm": "心率中位数_bpm",
        "motion_median": "motion中位数",
    }
    cols = list(show.columns)
    header = [rename.get(col, col) for col in cols]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# 找出连续未匹配 activity 的 HRV 窗口段，方便在报告里解释缺失覆盖。
def unlabeled_blocks(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    blocks: list[tuple[str, str, int]] = []
    start = None
    prev = None
    count = 0
    for row in df.itertuples(index=False):
        if row.activity_category == "unlabeled":
            if start is None:
                start = row.center_time
                count = 1
            else:
                count += 1
            prev = row.center_time
        elif start is not None:
            blocks.append((str(start), str(prev), count))
            start = None
            prev = None
            count = 0
    if start is not None:
        blocks.append((str(start), str(prev), count))
    return blocks


# 生成 point/unpaired activity event 的 Markdown 表。
def point_event_table(point_events: list[Event]) -> str:
    if not point_events:
        return "_没有 point/unpaired activity event。_"
    lines = [
        "| 时间 | 类型 | 类别 | 原始文本 | 主要原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for event in point_events:
        if event.kind == "stop":
            reason = "只有 stop 或 stop 无法和同类别 start 稳定配对"
        else:
            reason = "单点状态/感受记录，不是明确 start-stop 活动区间"
        lines.append(f"| {event.local_time} | {event.kind} | {event.category} | {event.text} | {reason} |")
    return "\n".join(lines)


# 生成连续 unlabeled 区间的 Markdown 表。
def unlabeled_block_table(blocks: list[tuple[str, str, int]], max_rows: int = 10) -> str:
    if not blocks:
        return "_没有未匹配 activity 的连续区间。_"
    lines = [
        "| 开始中心时间 | 结束中心时间 | 窗口数 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for start, end, count in blocks[:max_rows]:
        lines.append(f"| {start} | {end} | {count} | 该时间段没有可靠 activity interval 与 HRV window 重叠 |")
    if len(blocks) > max_rows:
        lines.append(f"| ... | ... | ... | 还有 {len(blocks) - max_rows} 段未显示 |")
    return "\n".join(lines)


# 转义 Markdown 表格单元格里的特殊字符。
def escape_md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


# 汇总粗分类和原始 activity 文本的对应关系，帮助人工检查分类规则。
def activity_mapping_table(
    intervals: list[Interval],
    point_events: list[Event],
    min_t0: int | None,
    max_t1: int | None,
    display_offset: timedelta,
) -> str:
    if min_t0 is None or max_t1 is None:
        return "_没有 HRV 覆盖范围内的 activity。_"

    # 先汇总可配对成 interval 的 activity。
    interval_rows: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for interval in intervals:
        overlap = overlap_seconds(min_t0, max_t1, interval.start_ms, interval.end_ms)
        if overlap <= 0:
            continue
        key = (interval.category, interval.raw_text)
        if key not in interval_rows:
            interval_rows[key] = {
                "category": interval.category,
                "kind": "interval",
                "text": interval.raw_text,
                "count": 0,
                "minutes": 0.0,
                "examples": [],
            }
        interval_rows[key]["count"] = int(interval_rows[key]["count"]) + 1
        interval_rows[key]["minutes"] = float(interval_rows[key]["minutes"]) + overlap / 60.0
        examples = interval_rows[key]["examples"]
        if isinstance(examples, list) and len(examples) < 2:
            examples.append(
                f"{display_dt(interval.start_ms, display_offset).strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"{display_dt(interval.end_ms, display_offset).strftime('%Y-%m-%d %H:%M:%S')}"
            )

    # 再汇总无法配成区间的单点/未配对 activity。
    point_rows: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for event in point_events:
        if not (min_t0 <= event.ts_ms < max_t1):
            continue
        key = (event.category, event.text)
        if key not in point_rows:
            point_rows[key] = {
                "category": event.category,
                "kind": "point/unpaired",
                "text": event.text,
                "count": 0,
                "minutes": 0.0,
                "examples": [],
            }
        point_rows[key]["count"] = int(point_rows[key]["count"]) + 1
        examples = point_rows[key]["examples"]
        if isinstance(examples, list) and len(examples) < 2:
            examples.append(event.local_time)

    # 合并、排序，并输出为 Markdown。
    rows = list(interval_rows.values()) + list(point_rows.values())
    if not rows:
        return "_没有 HRV 覆盖范围内的 activity。_"

    rows.sort(key=lambda r: (str(r["category"]), str(r["kind"]), str(r["text"])))

    lines = [
        "| 粗分类 | 类型 | 具体事件原始文本 | 出现次数 | 覆盖分钟数 | 示例时间 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        examples = row["examples"]
        example_text = "; ".join(examples) if isinstance(examples, list) else ""
        minutes = float(row["minutes"])
        minutes_text = f"{minutes:.1f}" if minutes > 0 else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md_cell(str(row["category"])),
                    escape_md_cell(str(row["kind"])),
                    escape_md_cell(str(row["text"])),
                    str(int(row["count"])),
                    minutes_text,
                    escape_md_cell(example_text),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# 写出共同窗口版本的中文 Markdown 报告。
def write_report(
    out_path: Path,
    participant: str,
    day: str,
    df: pd.DataFrame,
    activity_summary: pd.DataFrame,
    intervals: list[Interval],
    point_events: list[Event],
    plot_name: str,
    csv_name: str,
    display_offset: timedelta,
) -> None:
    # 报告中的主统计只基于 ECG-QC 合格窗口。
    good = df[
        (df["ecg_label_qc_pass"])
        & (df["ecg_valid_ibi_ratio"] >= 1.0)
        & (df["ecg_ibi_correction_ratio"] <= 0.2)
    ].copy()
    rmssd = good["ecg_rmssd_ms"].dropna()
    sdnn = good["ecg_sdnn_ms"].dropna()
    hr = good["ecg_hr_bpm"].dropna()
    rmssd_diff = rmssd.diff().abs().dropna()
    rmssd_30m = rmssd.diff(60).abs().dropna() if len(rmssd) > 60 else pd.Series(dtype=float)
    unlabeled_pct = 100.0 * float((df["activity_category"] == "unlabeled").mean()) if len(df) else math.nan
    first_center = str(df["center_time"].iloc[0]) if len(df) else "n/a"
    last_center = str(df["center_time"].iloc[-1]) if len(df) else "n/a"
    blocks = unlabeled_blocks(df)
    # 只报告落在 HRV 覆盖范围内的 point/unpaired event。
    if len(df):
        min_t0 = int(df["t0_ms"].min())
        max_t1 = int(df["t1_ms"].max())
        relevant_point_events = [event for event in point_events if min_t0 <= event.ts_ms < max_t1]
    else:
        min_t0 = None
        max_t1 = None
        relevant_point_events = []

    # 组装中文 Markdown 正文。
    lines = [
        f"# {participant} ECG-HRV 与活动日志对齐案例分析（{day}）",
        "",
        "## 目的",
        "",
        "使用 stride30 数据集中的 ECG-derived HRV 作为 reference 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本分析用于探索单日 HRV 波动和活动记录之间的对应关系，不作为因果检验。",
        "",
        "## 输入数据",
        "",
        "- HRV 来源：`training_v1_stride30` NPZ",
        f"- 活动日志来源：`Multisite-PPG/raw_data/{participant}/{participant}_activity_log.txt`",
        "- 窗口设置：5 min window，stride 30 s",
        "- 活动对齐方式：计算每个 HRV 窗口 `[t0_ms, t1_ms]` 与解析出的 activity interval 的时间重叠",
        "",
        "## 输出文件",
        "",
        f"- window-level sidecar CSV：`{csv_name}`",
        f"- 可视化图：`{plot_name}`",
        "",
        "## 数据质量",
        "",
        f"- 当天 HRV 窗口数：{len(df)}",
        f"- HRV 窗口中心时间范围：{first_center} 到 {last_center}",
        f"- 用于汇总的 ECG-QC 合格窗口数：{len(good)}",
        f"- 当天解析出的 activity interval 数：{len(intervals)}",
        f"- HRV 覆盖时间内的 point/unpaired activity event 数：{len(relevant_point_events)}",
        f"- 未匹配到 activity 的窗口比例：{unlabeled_pct:.1f}%",
        "",
        "这里的 ECG-QC 合格定义为：`ecg_label_qc_pass == True`、`ecg_valid_ibi_ratio >= 1.0`、且 `ecg_ibi_correction_ratio <= 0.2`。",
        "",
        "未匹配 activity 的窗口主要表示：这些 HRV window 没有和任何可解析出的 activity interval 发生时间重叠。它不是 ECG-QC 或 motion 筛选造成的；本 case study 没有用 motion threshold 筛窗口。",
        "",
        "## 单日 HRV 波动范围",
        "",
    ]
    if not rmssd.empty:
        lines.extend(
            [
                f"- RMSSD 中位数/IQR：{rmssd.median():.2f} ms / {(rmssd.quantile(0.75) - rmssd.quantile(0.25)):.2f} ms",
                f"- RMSSD 最小值/最大值：{rmssd.min():.2f} ms / {rmssd.max():.2f} ms",
                f"- 相邻窗口之间最大的 RMSSD 变化：{rmssd_diff.max():.2f} ms" if not rmssd_diff.empty else "- 相邻窗口之间最大的 RMSSD 变化：n/a",
                f"- 约 30 分钟内最大的 RMSSD 变化：{rmssd_30m.max():.2f} ms" if not rmssd_30m.empty else "- 约 30 分钟内最大的 RMSSD 变化：n/a",
                f"- SDNN 中位数/IQR：{sdnn.median():.2f} ms / {(sdnn.quantile(0.75) - sdnn.quantile(0.25)):.2f} ms" if not sdnn.empty else "- SDNN 中位数/IQR：n/a",
                f"- ECG HR 中位数/IQR：{hr.median():.2f} bpm / {(hr.quantile(0.75) - hr.quantile(0.25)):.2f} bpm" if not hr.empty else "- ECG HR 中位数/IQR：n/a",
            ]
        )
    else:
        lines.append("- 没有 ECG-QC 合格窗口。")
    lines.extend(
        [
            "",
            "## 按活动类别汇总",
            "",
            md_table(activity_summary),
            "",
            "## 活动粗分类与具体事件对应表",
            "",
            "下表列出 HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。`interval` 表示可配对成 start-stop 时间段的事件，`point/unpaired` 表示只有单点记录或没有稳定配对的事件。",
            "",
            activity_mapping_table(intervals, point_events, min_t0, max_t1, display_offset),
            "",
            "## 未匹配 Activity 的窗口",
            "",
            "下面列出连续的 `unlabeled` 区间。原因通常是原始 activity log 在对应时间没有可靠 start-stop 区间，或者只有单点记录，脚本没有强行延长成活动段。",
            "",
            unlabeled_block_table(blocks),
            "",
            "## Point/Unpaired Activity Events",
            "",
            "这些事件来自原始 activity log，但没有被配成完整 activity interval。常见原因包括重复 stop、拼写错误导致类别无法稳定配对、或本身只是单点状态记录。",
            "",
            point_event_table(relevant_point_events),
            "",
            "## 图像",
            "",
            "下图把 ECG-derived RMSSD、SDNN、ECG HR 和 motion magnitude 画在同一时间轴上，并用背景色标出解析出的 activity interval。灰色虚线表示 point/unpaired event。",
            "",
            f"图中相邻 HRV 点间隔超过 {DEFAULT_PLOT_BREAK_GAP_MIN:.0f} 分钟时会自动断线，避免把缺失数据误画成长直线。",
            "",
            "断线主要来自相邻可用窗口间隔过大，常见原因包括 ECG-QC 未通过、共同窗口数据缺失，或三设备边界/sample coverage 不满足要求。",
            "",
            "需要注意的是，图像只显示 HRV 有可用 window 的时间范围；如果原始 activity log 晚上还有记录但 training stride30 HRV 数据在那些时段没有可用窗口，晚上 activity log 不会显示在图中。这样做是为了避免把没有 HRV 数据的 activity 区间误读成可分析区间。",
            "",
            f"![{participant} HRV activity plot]({plot_name})",
            "",
            "## 解读注意事项",
            "",
            "- activity log 是人工输入的自由文本，因此 activity category 只是近似归类。",
            "- 每个 5 分钟 HRV 窗口的 `activity_category` 由时间重叠最多的 activity interval 决定。",
            "- 同一时间可能存在重叠活动；CSV 的 `activity_raw` 字段保留了原始重叠细节。",
            "- 这张图适合用来发现 HRV 与 activity 的候选对应关系；如果要做一般性结论，需要在更多 participant/day 上验证。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# 命令行入口：读取输入、完成对齐、生成 CSV/PNG/MD。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="P18")
    parser.add_argument("--day", default="2026-03-31")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--activity-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--plot-break-gap-min", type=float, default=DEFAULT_PLOT_BREAK_GAP_MIN)
    args = parser.parse_args()

    participant = args.participant
    npz_path = args.dataset_dir / f"synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30_{participant}.npz"
    activity_log = args.activity_log or DEFAULT_LOG_DIR / participant / f"{participant}_activity_log.txt"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not activity_log.is_file():
        raise FileNotFoundError(activity_log)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # activity log 使用自己的 Local Time 偏移来定义目标日期边界。
    events = read_activity_log(activity_log)
    display_offset = infer_activity_local_offset(events)
    day_start = datetime.strptime(args.day, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    day_start_ms = local_day_epoch_ms(day_start, display_offset)
    day_end_ms = local_day_epoch_ms(day_end, display_offset)

    # 解析 activity、抽取 HRV 窗口、完成窗口级 activity 标注。
    intervals, point_events = events_to_intervals(events, day_start_ms, day_end_ms)
    windows = build_window_table(npz_path, participant, args.day, display_offset)
    aligned = annotate_windows(windows, intervals, point_events)
    activity_summary = summarize_by_activity(aligned)

    stem = f"{participant}_{args.day}_activity_hrv_stride30"
    csv_path = out_dir / f"{stem}_window_aligned.csv"
    plot_path = out_dir / f"{stem}_plot.png"
    md_path = out_dir / f"{stem}_case_study.md"

    # 依次写出窗口级 sidecar CSV、图像和 Markdown 报告；activity 汇总只保留在 MD 中。
    aligned.to_csv(csv_path, index=False)
    make_plot(aligned, intervals, point_events, plot_path, participant, args.day, display_offset, args.plot_break_gap_min)
    write_report(
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
    )
    print(f"[saved] {csv_path}")
    print(f"[saved] {plot_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()

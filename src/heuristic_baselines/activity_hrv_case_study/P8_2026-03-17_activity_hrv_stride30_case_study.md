# P8 ECG-HRV 与活动日志对齐案例分析（2026-03-17）

## 目的

使用 stride30 数据集中的 ECG-derived HRV 作为 reference 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本分析用于探索单日 HRV 波动和活动记录之间的对应关系，不作为因果检验。

## 输入数据

- HRV 来源：`training_v1_stride30` NPZ
- 活动日志来源：`Multisite-PPG/raw_data/P8/P8_activity_log.txt`
- 窗口设置：5 min window，stride 30 s
- 活动对齐方式：计算每个 HRV 窗口 `[t0_ms, t1_ms]` 与解析出的 activity interval 的时间重叠

## 输出文件

- window-level sidecar CSV：`P8_2026-03-17_activity_hrv_stride30_window_aligned.csv`
- 可视化图：`P8_2026-03-17_activity_hrv_stride30_plot.png`

## 数据质量

- 当天 HRV 窗口数：976
- HRV 窗口中心时间范围：2026-03-17 05:03:29 到 2026-03-17 19:20:59
- 用于汇总的 ECG-QC 合格窗口数：976
- 当天解析出的 activity interval 数：10
- HRV 覆盖时间内的 point/unpaired activity event 数：3
- 未匹配到 activity 的窗口比例：52.4%

这里的 ECG-QC 合格定义为：`ecg_label_qc_pass == True`、`ecg_valid_ibi_ratio >= 1.0`、且 `ecg_ibi_correction_ratio <= 0.2`。

未匹配 activity 的窗口主要表示：这些 HRV window 没有和任何可解析出的 activity interval 发生时间重叠。它不是 ECG-QC 或 motion 筛选造成的；本 case study 没有用 motion threshold 筛窗口。

## 单日 HRV 波动范围

- RMSSD 中位数/IQR：41.44 ms / 9.20 ms
- RMSSD 最小值/最大值：24.17 ms / 59.00 ms
- 相邻窗口之间最大的 RMSSD 变化：16.06 ms
- 约 30 分钟内最大的 RMSSD 变化：22.35 ms
- SDNN 中位数/IQR：51.84 ms / 15.54 ms
- ECG HR 中位数/IQR：98.97 bpm / 12.54 bpm

## 按活动类别汇总

| 活动类别 | 窗口数 | 重叠分钟数 | RMSSD中位数_ms | RMSSD_IQR_ms | SDNN中位数_ms | 心率中位数_bpm | motion中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| work_study | 219 | 1076.94 | 40.78 | 8.75 | 52.11 | 101.85 | 0.48 |
| rest_sitting | 145 | 705.45 | 38.36 | 6.13 | 45.31 | 100.71 | 0.57 |
| walking | 101 | 505.00 | 39.60 | 6.66 | 43.91 | 102.42 | 0.47 |

## 活动粗分类与具体事件对应表

下表列出 HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。`interval` 表示可配对成 start-stop 时间段的事件，`point/unpaired` 表示只有单点记录或没有稳定配对的事件。

| 粗分类 | 类型 | 具体事件原始文本 | 出现次数 | 覆盖分钟数 | 示例时间 |
| --- | --- | --- | --- | --- | --- |
| eating_drinking | interval | start eating -> [unclosed] | 1 | 99.6 | 2026-03-17 17:43:53 -> 2026-03-17 21:43:53 |
| eating_drinking | interval | start eating -> end eating | 1 | 23.3 | 2026-03-17 13:09:59 -> 2026-03-17 13:33:16 |
| rest_sitting | interval | start playing on the phone -> [unclosed] | 1 | 235.8 | 2026-03-17 15:27:39 -> 2026-03-17 19:27:39 |
| social_entertainment | point/unpaired | end play | 1 |  | 2026-03-17 15:34:48 |
| walking | interval | start walking -> end walking | 4 | 100.6 | 2026-03-17 11:45:04 -> 2026-03-17 11:45:19; 2026-03-17 11:58:20 -> 2026-03-17 12:01:41 |
| work_study | interval | start study -> end stduy | 1 | 67.8 | 2026-03-17 16:35:58 -> 2026-03-17 17:43:45 |
| work_study | interval | start study -> end study | 1 | 109.7 | 2026-03-17 13:34:14 -> 2026-03-17 15:23:58 |
| work_study | point/unpaired | end study | 1 |  | 2026-03-17 13:09:50 |
| work_study | point/unpaired | study | 1 |  | 2026-03-17 12:40:31 |

## 未匹配 Activity 的窗口

下面列出连续的 `unlabeled` 区间。原因通常是原始 activity log 在对应时间没有可靠 start-stop 区间，或者只有单点记录，脚本没有强行延长成活动段。

| 开始中心时间 | 结束中心时间 | 窗口数 | 说明 |
| --- | --- | --- | --- |
| 2026-03-17 05:03:29 | 2026-03-17 10:33:29 | 511 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |

## Point/Unpaired Activity Events

这些事件来自原始 activity log，但没有被配成完整 activity interval。常见原因包括重复 stop、拼写错误导致类别无法稳定配对、或本身只是单点状态记录。

| 时间 | 类型 | 类别 | 原始文本 | 主要原因 |
| --- | --- | --- | --- | --- |
| 2026-03-17 12:40:31 | point | work_study | study | 单点状态/感受记录，不是明确 start-stop 活动区间 |
| 2026-03-17 13:09:50 | stop | work_study | end study | 只有 stop 或 stop 无法和同类别 start 稳定配对 |
| 2026-03-17 15:34:48 | stop | social_entertainment | end play | 只有 stop 或 stop 无法和同类别 start 稳定配对 |

## 图像

下图把 ECG-derived RMSSD、SDNN、ECG HR 和 motion magnitude 画在同一时间轴上，并用背景色标出解析出的 activity interval。灰色虚线表示 point/unpaired event。

图中相邻 HRV 点间隔超过 5 分钟时会自动断线，避免把缺失数据误画成长直线。

断线主要来自相邻可用窗口间隔过大，常见原因包括 ECG-QC 未通过、共同窗口数据缺失，或三设备边界/sample coverage 不满足要求。

需要注意的是，图像只显示 HRV 有可用 window 的时间范围；如果原始 activity log 晚上还有记录但 training stride30 HRV 数据在那些时段没有可用窗口，晚上 activity log 不会显示在图中。这样做是为了避免把没有 HRV 数据的 activity 区间误读成可分析区间。

![P8 HRV activity plot](P8_2026-03-17_activity_hrv_stride30_plot.png)

## 解读注意事项

- activity log 是人工输入的自由文本，因此 activity category 只是近似归类。
- 每个 5 分钟 HRV 窗口的 `activity_category` 由时间重叠最多的 activity interval 决定。
- 同一时间可能存在重叠活动；CSV 的 `activity_raw` 字段保留了原始重叠细节。
- 这张图适合用来发现 HRV 与 activity 的候选对应关系；如果要做一般性结论，需要在更多 participant/day 上验证。

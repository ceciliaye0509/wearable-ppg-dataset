# P18 ECG-only HRV 与活动日志对齐案例分析（2026-04-01）

## 目的

使用 raw Polar ECG 单独计算 5 分钟 HRV 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本版本只要求 ECG label QC 通过，不要求 Earring/Ring/Watch 三设备 raw boundary aligned，也不要求 PPG sample coverage。

## 输入数据

- HRV 来源：`Multisite-PPG/raw_data` 中的 raw Polar ECG
- 活动日志来源：`Multisite-PPG/raw_data/P18/P18_activity_log.txt`
- 窗口设置：5 min window，stride 30 s
- ECG 保留条件：`ecg_label_qc_pass == True`

## 输出文件

- window-level sidecar CSV：`P18_2026-04-01_activity_hrv_ecg_only_stride30_window_aligned.csv`
- 可视化图：`P18_2026-04-01_activity_hrv_ecg_only_stride30_plot.png`

## 数据质量

- 当天 ECG-only 候选窗口数：2871
- ECG-QC 合格窗口数：1008 (35.1%)
- ECG-QC 不合格窗口数：1863
- ECG-QC 合格窗口中心时间范围：2026-04-01 04:57:00 到 2026-04-01 17:43:30
- 当天解析出的 activity interval 数：41
- HRV 覆盖时间内的 point/unpaired activity event 数：1
- 未匹配到 activity 的窗口比例：57.4%

这个 ECG-only 版本用于回答全天 ECG-derived HRV 与 activity log 的关系；它不适合直接作为 PPG model training dataset，因为没有要求三设备 PPG 同时可用。

## 单日 HRV 波动范围

- RMSSD 中位数/IQR：69.24 ms / 14.24 ms
- RMSSD 最小值/最大值：40.28 ms / 98.97 ms
- 相邻窗口之间最大的 RMSSD 变化：17.86 ms
- 约 30 分钟内最大的 RMSSD 变化：36.24 ms
- SDNN 中位数/IQR：84.40 ms / 32.58 ms
- ECG HR 中位数/IQR：65.47 bpm / 17.56 bpm

## 按活动类别汇总

| 活动类别 | 窗口数 | 重叠分钟数 | RMSSD中位数_ms | RMSSD_IQR_ms | SDNN中位数_ms | 心率中位数_bpm | motion中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| social_entertainment | 127 | 634.44 | 72.70 | 14.24 | 101.86 | 59.06 |  |
| rest_sitting | 103 | 490.97 | 76.44 | 10.07 | 86.58 | 54.79 |  |
| work_study | 67 | 335.00 | 75.18 | 11.81 | 79.25 | 51.75 |  |
| transport | 64 | 294.95 | 67.05 | 7.93 | 84.39 | 58.86 |  |
| walking | 42 | 208.03 | 70.01 | 9.60 | 122.33 | 58.01 |  |
| eating_drinking | 26 | 130.00 | 80.87 | 8.58 | 104.78 | 54.01 |  |

## 活动粗分类与具体事件对应表

下表列出 ECG-only HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。

| 粗分类 | 类型 | 具体事件原始文本 | 出现次数 | 覆盖分钟数 | 示例时间 |
| --- | --- | --- | --- | --- | --- |
| eating_drinking | interval | start drinking water -> stop drinking water | 3 | 1.4 | 2026-04-01 13:01:34 -> 2026-04-01 13:01:57; 2026-04-01 17:00:17 -> 2026-04-01 17:00:48 |
| eating_drinking | interval | start eating Mac and cheese and stir fried vegetables -> stop eating | 1 | 17.6 | 2026-04-01 17:03:36 -> 2026-04-01 17:21:10 |
| eating_drinking | interval | start warming up lunch -> stop warming up food | 1 | 12.4 | 2026-04-01 16:47:43 -> 2026-04-01 17:00:05 |
| rest_sitting | interval | start sitting on couch -> stop sitting on couch | 2 | 107.3 | 2026-04-01 15:45:39 -> 2026-04-01 16:47:10; 2026-04-01 17:00:11 -> 2026-04-01 19:16:34 |
| social_entertainment | interval | start listening to music -> stop listening to mysic | 1 | 34.4 | 2026-04-01 11:56:35 -> 2026-04-01 12:30:58 |
| social_entertainment | interval | start talking on facetime -> stop talking on phone | 1 | 19.8 | 2026-04-01 16:27:18 -> 2026-04-01 16:47:03 |
| social_entertainment | interval | start watching youtube -> stop watching youtueb | 1 | 51.7 | 2026-04-01 12:40:24 -> 2026-04-01 13:32:06 |
| transport | interval | start driving -> stop driving | 1 | 31.8 | 2026-04-01 11:54:45 -> 2026-04-01 12:26:35 |
| transport | point/unpaired | stop driving | 1 |  | 2026-04-01 12:30:52 |
| walking | interval | start incline walk on treadmill -> stop incline walk | 1 | 61.9 | 2026-04-01 12:30:16 -> 2026-04-01 13:32:12 |
| walking | interval | start walking -> stop walking | 2 | 15.1 | 2026-04-01 12:27:37 -> 2026-04-01 12:30:05; 2026-04-01 16:47:14 -> 2026-04-01 16:59:53 |
| work_study | interval | start joining Zoom call on laptop -> stop zoom meeting | 1 | 37.8 | 2026-04-01 17:03:19 -> 2026-04-01 17:41:05 |
| work_study | interval | start working on laptop | 1 | 77.6 | 2026-04-01 15:45:46 -> 2026-04-01 17:03:19 |
| work_study | interval | start working on laptop -> stop working on laptop | 1 | 4.8 | 2026-04-01 17:41:12 -> 2026-04-01 19:16:48 |

## 未匹配 Activity 的窗口

| 开始中心时间 | 结束中心时间 | 窗口数 | 说明 |
| --- | --- | --- | --- |
| 2026-04-01 04:57:00 | 2026-04-01 11:52:00 | 499 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-04-01 15:03:30 | 2026-04-01 15:43:00 | 80 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |

## Point/Unpaired Activity Events

| 时间 | 类型 | 类别 | 原始文本 | 主要原因 |
| --- | --- | --- | --- | --- |
| 2026-04-01 12:30:52 | stop | transport | stop driving | 只有 stop 或 stop 无法和同类别 start 稳定配对 |

## 图像

图中相邻 HRV 点间隔超过 5 分钟时会自动断线，避免把缺失数据误画成长直线。

断线主要来自相邻 ECG-QC 合格窗口间隔过大，常见原因包括 raw ECG sample 不完整、R peak/IBI 质量不足，或 HRV 计算未通过 QC。

![P18 ECG-only HRV activity plot](P18_2026-04-01_activity_hrv_ecg_only_stride30_plot.png)

## 解读注意事项

- 这个版本只回答 ECG-derived HRV 的日内变化和 activity log 的对应关系。
- 与 raw-aligned training dataset 相比，它不会因为 Earring/Ring/Watch PPG 边界不齐而丢掉 ECG 可用窗口。
- activity category 来自自由文本规则归类，适合探索候选关系，不作为因果检验。

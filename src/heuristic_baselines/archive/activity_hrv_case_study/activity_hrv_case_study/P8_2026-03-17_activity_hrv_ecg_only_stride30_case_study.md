# P8 ECG-only HRV 与活动日志对齐案例分析（2026-03-17）

## 目的

使用 raw Polar ECG 单独计算 5 分钟 HRV 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本版本只要求 ECG label QC 通过，不要求 Earring/Ring/Watch 三设备 raw boundary aligned，也不要求 PPG sample coverage。

## 输入数据

- HRV 来源：`Multisite-PPG/raw_data` 中的 raw Polar ECG
- 活动日志来源：`Multisite-PPG/raw_data/P8/P8_activity_log.txt`
- 窗口设置：5 min window，stride 30 s
- ECG 保留条件：`ecg_label_qc_pass == True`

## 输出文件

- window-level sidecar CSV：`P8_2026-03-17_activity_hrv_ecg_only_stride30_window_aligned.csv`
- 可视化图：`P8_2026-03-17_activity_hrv_ecg_only_stride30_plot.png`

## 数据质量

- 当天 ECG-only 候选窗口数：2354
- ECG-QC 合格窗口数：1467 (62.3%)
- ECG-QC 不合格窗口数：887
- ECG-QC 合格窗口中心时间范围：2026-03-17 04:20:59 到 2026-03-17 19:20:59
- 当天解析出的 activity interval 数：10
- HRV 覆盖时间内的 point/unpaired activity event 数：3
- 未匹配到 activity 的窗口比例：53.6%

这个 ECG-only 版本用于回答全天 ECG-derived HRV 与 activity log 的关系；它不适合直接作为 PPG model training dataset，因为没有要求三设备 PPG 同时可用。

## 单日 HRV 波动范围

- RMSSD 中位数/IQR：41.57 ms / 8.88 ms
- RMSSD 最小值/最大值：24.17 ms / 59.00 ms
- 相邻窗口之间最大的 RMSSD 变化：16.06 ms
- 约 30 分钟内最大的 RMSSD 变化：27.47 ms
- SDNN 中位数/IQR：51.63 ms / 17.07 ms
- ECG HR 中位数/IQR：97.76 bpm / 13.20 bpm

## 按活动类别汇总

| 活动类别 | 窗口数 | 重叠分钟数 | RMSSD中位数_ms | RMSSD_IQR_ms | SDNN中位数_ms | 心率中位数_bpm | motion中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| work_study | 295 | 1456.94 | 41.44 | 8.09 | 52.65 | 102.41 |  |
| walking | 208 | 942.96 | 38.67 | 6.55 | 42.86 | 99.41 |  |
| rest_sitting | 177 | 865.45 | 38.53 | 6.26 | 45.59 | 100.49 |  |

## 活动粗分类与具体事件对应表

下表列出 ECG-only HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。

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

| 开始中心时间 | 结束中心时间 | 窗口数 | 说明 |
| --- | --- | --- | --- |
| 2026-03-17 04:20:59 | 2026-03-17 11:42:29 | 732 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-03-17 11:48:59 | 2026-03-17 11:55:29 | 13 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-03-17 12:40:59 | 2026-03-17 13:01:29 | 42 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |

## Point/Unpaired Activity Events

| 时间 | 类型 | 类别 | 原始文本 | 主要原因 |
| --- | --- | --- | --- | --- |
| 2026-03-17 12:40:31 | point | work_study | study | 单点状态/感受记录，不是明确 start-stop 活动区间 |
| 2026-03-17 13:09:50 | stop | work_study | end study | 只有 stop 或 stop 无法和同类别 start 稳定配对 |
| 2026-03-17 15:34:48 | stop | social_entertainment | end play | 只有 stop 或 stop 无法和同类别 start 稳定配对 |

## 图像

图中相邻 HRV 点间隔超过 5 分钟时会自动断线，避免把缺失数据误画成长直线。

断线主要来自相邻 ECG-QC 合格窗口间隔过大，常见原因包括 raw ECG sample 不完整、R peak/IBI 质量不足，或 HRV 计算未通过 QC。

![P8 ECG-only HRV activity plot](P8_2026-03-17_activity_hrv_ecg_only_stride30_plot.png)

## 解读注意事项

- 这个版本只回答 ECG-derived HRV 的日内变化和 activity log 的对应关系。
- 与 raw-aligned training dataset 相比，它不会因为 Earring/Ring/Watch PPG 边界不齐而丢掉 ECG 可用窗口。
- activity category 来自自由文本规则归类，适合探索候选关系，不作为因果检验。

# P2 ECG-only HRV 与活动日志对齐案例分析（2026-02-25）

## 目的

使用 raw Polar ECG 单独计算 5 分钟 HRV 时间序列，再把 raw activity log 作为 sidecar annotation 对齐到每个窗口。本版本只要求 ECG label QC 通过，不要求 Earring/Ring/Watch 三设备 raw boundary aligned，也不要求 PPG sample coverage。

## 输入数据

- HRV 来源：`Multisite-PPG/raw_data` 中的 raw Polar ECG
- 活动日志来源：`Multisite-PPG/raw_data/P2/P2_activity_log.txt`
- 窗口设置：5 min window，stride 30 s
- ECG 保留条件：`ecg_label_qc_pass == True`

## 输出文件

- window-level sidecar CSV：`P2_2026-02-25_activity_hrv_ecg_only_stride30_window_aligned.csv`
- 可视化图：`P2_2026-02-25_activity_hrv_ecg_only_stride30_plot.png`

## 数据质量

- 当天 ECG-only 候选窗口数：2581
- ECG-QC 合格窗口数：1071 (41.5%)
- ECG-QC 不合格窗口数：1510
- ECG-QC 合格窗口中心时间范围：2026-02-25 02:34:30 到 2026-02-25 11:54:00
- 当天解析出的 activity interval 数：14
- HRV 覆盖时间内的 point/unpaired activity event 数：0
- 未匹配到 activity 的窗口比例：89.1%

这个 ECG-only 版本用于回答全天 ECG-derived HRV 与 activity log 的关系；它不适合直接作为 PPG model training dataset，因为没有要求三设备 PPG 同时可用。

## 单日 HRV 波动范围

- RMSSD 中位数/IQR：59.34 ms / 6.02 ms
- RMSSD 最小值/最大值：43.54 ms / 73.11 ms
- 相邻窗口之间最大的 RMSSD 变化：9.09 ms
- 约 30 分钟内最大的 RMSSD 变化：18.55 ms
- SDNN 中位数/IQR：61.10 ms / 20.08 ms
- ECG HR 中位数/IQR：68.16 bpm / 9.91 bpm

## 按活动类别汇总

| 活动类别 | 窗口数 | 重叠分钟数 | RMSSD中位数_ms | RMSSD_IQR_ms | SDNN中位数_ms | 心率中位数_bpm | motion中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eating_drinking | 86 | 303.66 | 60.76 | 4.77 | 66.99 | 70.55 |  |
| work_study | 31 | 104.33 | 65.23 | 2.57 | 77.41 | 73.13 |  |

## 活动粗分类与具体事件对应表

下表列出 ECG-only HRV 覆盖时间范围内，脚本解析到的每个粗分类及其对应的原始 activity 文本。

| 粗分类 | 类型 | 具体事件原始文本 | 出现次数 | 覆盖分钟数 | 示例时间 |
| --- | --- | --- | --- | --- | --- |
| eating_drinking | interval | start eating -> end eating | 3 | 32.6 | 2026-02-25 10:41:48 -> 2026-02-25 10:41:57; 2026-02-25 11:17:24 -> 2026-02-25 11:40:37 |
| work_study | interval | start typing | 1 | 0.4 | 2026-02-25 10:54:10 -> 2026-02-25 10:54:32 |
| work_study | interval | start typing -> stop typing | 1 | 10.1 | 2026-02-25 10:54:32 -> 2026-02-25 11:04:36 |

## 未匹配 Activity 的窗口

| 开始中心时间 | 结束中心时间 | 窗口数 | 说明 |
| --- | --- | --- | --- |
| 2026-02-25 02:34:30 | 2026-02-25 10:39:00 | 922 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-02-25 10:44:30 | 2026-02-25 10:51:30 | 14 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-02-25 11:07:30 | 2026-02-25 11:14:30 | 15 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |
| 2026-02-25 11:43:30 | 2026-02-25 11:44:30 | 3 | 该时间段没有可靠 activity interval 与 HRV window 重叠 |

## Point/Unpaired Activity Events

_没有 point/unpaired activity event。_

## 图像

图中相邻 HRV 点间隔超过 5 分钟时会自动断线，避免把缺失数据误画成长直线。

断线主要来自相邻 ECG-QC 合格窗口间隔过大，常见原因包括 raw ECG sample 不完整、R peak/IBI 质量不足，或 HRV 计算未通过 QC。

![P2 ECG-only HRV activity plot](P2_2026-02-25_activity_hrv_ecg_only_stride30_plot.png)

## 解读注意事项

- 这个版本只回答 ECG-derived HRV 的日内变化和 activity log 的对应关系。
- 与 raw-aligned training dataset 相比，它不会因为 Earring/Ring/Watch PPG 边界不齐而丢掉 ECG 可用窗口。
- activity category 来自自由文本规则归类，适合探索候选关系，不作为因果检验。

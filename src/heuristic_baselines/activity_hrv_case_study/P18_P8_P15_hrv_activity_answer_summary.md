# P18、P8、P15 ECG-derived HRV 与 Activity Log 结果总结

## 直接回答

### 日内波动幅度与速度

结合 P18、P8、P15 三个 case study，ECG-derived HRV 在一天内有明显日内波动。以 RMSSD 为例，三位参与者在单日内的 RMSSD 范围大约覆盖 **24.17-98.97 ms**；单个参与者一天内的 RMSSD 跨度约 **34.83-46.68 ms**。即使用 5 分钟窗口计算，HRV 也不是稳定不变的。

变化速度上，三位参与者在相邻 stride30 窗口之间的最大 RMSSD 变化约 **16.06-20.34 ms**；在约 30 分钟尺度内，最大 RMSSD 变化约 **22.35-35.94 ms**。这说明 HRV 可以在几十分钟内出现明显变化。

### 与 Activity 的对应关系

最清楚的模式来自 P18：低强度活动，如 `rest_sitting`、`eating_drinking`、`work_study`、`social_entertainment`，通常对应较高 RMSSD 和较低 ECG HR；移动/交通相关活动，如 `transport` 和部分 `walking`，RMSSD 相对更低一些。

P8 的 HR 整体较高、RMSSD 整体较低，activity 类别之间分离不如 P18 明显。P15 显示 `social_entertainment` 的 RMSSD 明显高于 `transport`，但 P15 未匹配窗口和断线较多，因此更适合作为数据覆盖限制的例子，而不是最强主结论。

总体结论是：**HRV 与 activity 状态有时间上的对应关系，尤其是低强度状态常对应较高 HRV/较低 HR，而交通、移动或较高身体负荷状态常对应较低 HRV或更高 HR。但这些结果应解释为探索性关联，不应写成因果证明。**

## 结果文件链接

| 参与者 | 日期 | 版本 | 测试结果 |
| --- | --- | --- | --- |
| P18 | 2026-04-01 | 共同窗口 stride30 | [P18_2026-04-01_activity_hrv_stride30_case_study.md](P18_2026-04-01_activity_hrv_stride30_case_study.md) |
| P18 | 2026-04-01 | ECG-only stride30 | [P18_2026-04-01_activity_hrv_ecg_only_stride30_case_study.md](P18_2026-04-01_activity_hrv_ecg_only_stride30_case_study.md) |
| P8 | 2026-03-17 | 共同窗口 stride30 | [P8_2026-03-17_activity_hrv_stride30_case_study.md](P8_2026-03-17_activity_hrv_stride30_case_study.md) |
| P8 | 2026-03-17 | ECG-only stride30 | [P8_2026-03-17_activity_hrv_ecg_only_stride30_case_study.md](P8_2026-03-17_activity_hrv_ecg_only_stride30_case_study.md) |
| P15 | 2026-03-29 | 共同窗口 stride30 | [P15_2026-03-29_activity_hrv_stride30_case_study.md](P15_2026-03-29_activity_hrv_stride30_case_study.md) |
| P15 | 2026-03-29 | ECG-only stride30 | [P15_2026-03-29_activity_hrv_ecg_only_stride30_case_study.md](P15_2026-03-29_activity_hrv_ecg_only_stride30_case_study.md) |

## 单日 HRV 波动幅度

下表同时列出 ECG-only 和共同窗口 stride30 结果。**ECG-only 更适合回答生理问题**，也就是“一个人的 ECG-derived HRV 当天如何变化”；共同窗口版本更接近后续模型/数据集使用的窗口，可作为一致性检查。

| 参与者 | 版本 | RMSSD 中位数/IQR ms | RMSSD 最小-最大 ms | 单日 RMSSD 跨度 ms | 相邻窗口最大变化 ms | 约 30 分钟最大变化 ms | ECG HR 中位数/IQR bpm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P18 | 共同窗口 | 70.32 / 13.84 | 52.29-98.97 | 46.68 | 17.63 | 35.94 | 64.19 / 16.70 |
| P18 | ECG-only | 69.24 / 14.24 | 40.28-98.97 | 58.69 | 17.86 | 36.24 | 65.47 / 17.56 |
| P8 | 共同窗口 | 41.44 / 9.20 | 24.17-59.00 | 34.83 | 16.06 | 22.35 | 98.97 / 12.54 |
| P8 | ECG-only | 41.57 / 8.88 | 24.17-59.00 | 34.83 | 16.06 | 27.47 | 97.76 / 13.20 |
| P15 | 共同窗口 | 51.26 / 12.87 | 31.63-73.95 | 42.32 | 20.34 | 31.92 | 85.85 / 8.00 |
| P15 | ECG-only | 51.16 / 13.24 | 31.63-73.95 | 42.32 | 14.50 | 31.92 | 85.69 / 7.84 |

## 按参与者解读

### P18：最适合作为主例子

P18 的 activity log 比较丰富，且 activity 类别之间的 HRV/HR 差异较容易解释。ECG-only 结果中，RMSSD 范围为 **40.28-98.97 ms**，单日跨度 **58.69 ms**；相邻窗口最大变化 **17.86 ms**，约 30 分钟内最大变化 **36.24 ms**。这说明 P18 一天内 HRV 波动明显，而且可以在几十分钟内快速变化。

![P18 ECG-only HRV activity plot](P18_2026-04-01_activity_hrv_ecg_only_stride30_plot.png)

P18 的 ECG-only activity 汇总中：

| 活动类别 | RMSSD 中位数 ms | SDNN 中位数 ms | ECG HR 中位数 bpm | 解释 |
| --- | --- | --- | --- | --- |
| eating_drinking | 80.87 | 104.78 | 54.01 | HR 较低、HRV 较高，符合低强度状态 |
| rest_sitting | 76.44 | 86.58 | 54.79 | HR 较低、HRV 较高 |
| work_study | 75.18 | 79.25 | 51.75 | 静态工作/电脑状态，HR 较低 |
| social_entertainment | 72.70 | 101.86 | 59.06 | 低强度娱乐/通话/听音乐 |
| walking | 70.01 | 122.33 | 58.01 | 移动相关，但 HR 未明显升高 |
| transport | 67.05 | 84.39 | 58.86 | RMSSD 相对更低 |

P18 的 ECG-only 结果支持：低强度活动下 HRV 较高、HR 较低；交通/移动相关活动下 HRV 相对更低。共同窗口版本得到的排序基本一致，因此 P18 是三个例子中最适合用来回答 activity-HRV 对应关系的 participant。

### P8：HR 整体偏高、HRV 整体偏低

P8 的 RMSSD 整体低于 P18，ECG HR 整体明显更高。ECG-only 结果中，RMSSD 范围为 **24.17-59.00 ms**，单日跨度 **34.83 ms**；相邻窗口最大变化 **16.06 ms**，约 30 分钟内最大变化 **27.47 ms**。

![P8 ECG-only HRV activity plot](P8_2026-03-17_activity_hrv_ecg_only_stride30_plot.png)

P8 的 ECG-only activity 汇总中：

| 活动类别 | RMSSD 中位数 ms | SDNN 中位数 ms | ECG HR 中位数 bpm | 解释 |
| --- | --- | --- | --- | --- |
| work_study | 41.44 | 52.65 | 102.41 | HR 高、HRV 低 |
| walking | 38.67 | 42.86 | 99.41 | HR 仍较高 |
| rest_sitting | 38.53 | 45.59 | 100.49 | 即使休息/玩手机，HR 仍高 |

P8 说明不同个体之间 baseline 可以差很多。P8 的 activity 类别之间没有 P18 那么清楚的分离，但它支持另一个常识性现象：当 HR 整体偏高时，RMSSD 往往整体偏低。共同窗口版本也显示相似模式。

### P15：有一定 activity 关联，但数据覆盖限制更明显

P15 的 RMSSD 范围和变化速度都比较明显，但未匹配窗口和断线较多，所以更适合作为补充例子。ECG-only 结果中，RMSSD 范围为 **31.63-73.95 ms**，单日跨度 **42.32 ms**；相邻窗口最大变化 **14.50 ms**，约 30 分钟内最大变化 **31.92 ms**。

![P15 ECG-only HRV activity plot](P15_2026-03-29_activity_hrv_ecg_only_stride30_plot.png)

P15 的 ECG-only activity 汇总中：

| 活动类别 | RMSSD 中位数 ms | SDNN 中位数 ms | ECG HR 中位数 bpm | 解释 |
| --- | --- | --- | --- | --- |
| social_entertainment | 63.74 | 73.96 | 85.74 | HRV 相对较高 |
| rest_sitting | 53.09 | 57.45 | 86.52 | 中间水平 |
| transport | 51.71 | 55.41 | 88.06 | RMSSD 较低、HR 较高 |

P15 也显示 `transport` 相比 `social_entertainment` 更低 HRV、更高 HR；共同窗口版本得到相同方向。但因为 activity log 在早晨覆盖不足，且有较多 point/unpaired 记录，所以不能把 P15 作为最强证据。

## 综合结论

1. **HRV 日内变化幅度明显。**  
   ECG-only 结果中，P18、P8、P15 的 RMSSD 单日跨度约 **34.83-58.69 ms**；共同窗口版本中跨度约 **34.83-46.68 ms**。这说明同一个人一天内 HRV 可以有很大变化。

2. **HRV 可以在几十分钟内快速变化。**  
   ECG-only 结果中，约 30 分钟内最大 RMSSD 变化为 **27.47-36.24 ms**；共同窗口版本中约为 **22.35-35.94 ms**。因此，HRV 不只是“天与天之间”变化，也会在一天内部随状态变化。

3. **活动关联最清楚的是 P18。**  
   P18 中休息、吃饭、工作/学习、社交娱乐等低强度活动对应较高 RMSSD 和较低 HR；交通/移动相关活动对应相对较低 RMSSD。

4. **P8 和 P15 提供补充信息。**  
   P8 显示个体 baseline 差异很大：HR 整体高、RMSSD 整体低，各 activity 之间差异弱。P15 显示 `social_entertainment` 高于 `transport`，但数据覆盖限制明显。

5. **结论应写成探索性关联。**  
   activity log 是自由文本且覆盖不完整，部分 activity 只有单点或未配对记录；图中断线也说明并非所有时间都有连续可靠 HRV。因此这些结果支持“HRV 与 activity 状态存在时间对应关系”，但不能写成“某活动导致 HRV 改变”的因果结论。

## 可放入报告的一句话

基于 P18、P8、P15 的 ECG-derived HRV 与 activity log 对齐分析，单个参与者的 RMSSD 在一天内可变化约 **35-59 ms**，在约 30 分钟内可变化约 **27-36 ms**；活动关联方面，低强度状态通常对应较高 HRV 和较低 HR，而交通、移动或较高身体负荷状态通常对应较低 HRV 或较高 HR，但由于 activity log 覆盖不完整且为自由文本记录，该结论应视为探索性关联而非因果证明。

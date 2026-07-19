# v1.2 Motion Threshold Sensitivity on training_stride30

本汇总使用当前 v1.2 最佳 baseline：每个 `device x channel` 使用冻结参数，并且 green/IR 都汇报。

## Threshold Impact Comparison

下表比较 motion threshold 从严格到宽松时，对当前 baseline 的影响。Coverage 使用 motion 子集内部窗口作分母。

| Device | Channel | Motion windows `<0.1 -> <1.0` | Coverage `<0.1 -> <1.0` | RMSSD MAE `<0.1 -> <1.0` | RMSSD R `<0.1 -> <1.0` | 主要影响 |
|---|---|---:|---:|---:|---:|---|
| `Earring` | `ppg_green` | 4070 -> 15318 | 93.88% -> 85.59% | 17.66 -> 17.50 ms | 0.375 -> 0.416 | 阈值放宽后窗口数增加很多，MAE/R 基本稳定，coverage 小幅下降。 |
| `Earring` | `ppg_ir` | 4070 -> 15318 | 88.60% -> 67.42% | 14.83 -> 15.49 ms | 0.403 -> 0.427 | IR 保持 Earring 的 RMSSD 最佳通道；放宽阈值主要降低 coverage，误差只小幅变差。 |
| `Ring` | `ppg_green` | 372 -> 13229 | 79.57% -> 42.86% | 10.48 -> 11.06 ms | 0.806 -> 0.548 | 严格 motion gate 明显提高 R 并降低 MAE，但代价是只保留很少窗口。 |
| `Ring` | `ppg_ir` | 372 -> 13229 | 69.35% -> 25.63% | 12.52 -> 14.22 ms | 0.722 -> 0.472 | 与 Ring green 同方向，但始终弱于 green。 |
| `Watch` | `ppg_green` | 1295 -> 14820 | 49.88% -> 21.84% | 11.74 -> 14.80 ms | 0.599 -> 0.423 | motion filtering 对 Watch green 帮助最大：严格阈值显著改善 RMSSD MAE/R，但窗口数大幅减少。 |
| `Watch` | `ppg_ir` | 1295 -> 14820 | 39.92% -> 17.65% | 66.79 -> 76.90 ms | -0.115 -> -0.051 | 即使在低 motion 子集，IR 仍失败；motion filtering 不能修复 Watch IR。 |

核心结论：

- `<0.1` 是最严格的 low-motion 子集，Ring green 和 Watch green 的 RMSSD R 最高，但 Ring/Watch 的窗口数少，代表性有限。
- `<0.2` 是更平衡的 low-motion sensitivity：窗口数明显多于 `<0.1`，同时 Ring green 和 Watch green 仍保留较好的 R。
- `<0.5` 和 `<1.0` 更接近 low-to-moderate motion/full training 行为，窗口数多，但 Ring/Watch 的 R 和 coverage within motion 明显下降。
- 对当前 v1.2 baseline，motion threshold 的主要收益集中在 Ring green 和 Watch green；Earring 相对稳，Watch IR 始终不适合作为 HRV baseline。

## Aggregate RMSSD/SDNN Results

| Threshold | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| <0.1 | `Ring` | `ppg_green` | 372 | 296 | 79.57% | 10.48 ms | 0.806 | 10.18 ms | 0.812 |
| <0.1 | `Watch` | `ppg_green` | 1295 | 646 | 49.88% | 11.74 ms | 0.599 | 21.74 ms | 0.521 |
| <0.1 | `Ring` | `ppg_ir` | 372 | 258 | 69.35% | 12.52 ms | 0.722 | 23.80 ms | 0.556 |
| <0.1 | `Earring` | `ppg_ir` | 4070 | 3606 | 88.60% | 14.83 ms | 0.403 | 7.46 ms | 0.783 |
| <0.1 | `Earring` | `ppg_green` | 4070 | 3821 | 93.88% | 17.66 ms | 0.375 | 6.19 ms | 0.896 |
| <0.1 | `Watch` | `ppg_ir` | 1295 | 517 | 39.92% | 66.79 ms | -0.115 | 82.71 ms | -0.092 |
| <0.2 | `Ring` | `ppg_green` | 2092 | 1485 | 70.98% | 10.89 ms | 0.677 | 10.20 ms | 0.775 |
| <0.2 | `Ring` | `ppg_ir` | 2092 | 1143 | 54.64% | 13.76 ms | 0.572 | 28.27 ms | 0.517 |
| <0.2 | `Earring` | `ppg_ir` | 9301 | 7421 | 79.79% | 14.24 ms | 0.456 | 7.36 ms | 0.848 |
| <0.2 | `Watch` | `ppg_green` | 5567 | 2020 | 36.29% | 14.37 ms | 0.513 | 21.04 ms | 0.609 |
| <0.2 | `Earring` | `ppg_green` | 9301 | 8522 | 91.62% | 16.30 ms | 0.454 | 5.84 ms | 0.929 |
| <0.2 | `Watch` | `ppg_ir` | 5567 | 1428 | 25.65% | 73.65 ms | -0.035 | 91.11 ms | 0.126 |
| <0.5 | `Ring` | `ppg_green` | 8292 | 4540 | 54.75% | 10.83 ms | 0.580 | 11.06 ms | 0.779 |
| <0.5 | `Ring` | `ppg_ir` | 8292 | 3053 | 36.82% | 14.14 ms | 0.476 | 27.13 ms | 0.597 |
| <0.5 | `Watch` | `ppg_green` | 12539 | 3177 | 25.34% | 14.78 ms | 0.424 | 21.31 ms | 0.598 |
| <0.5 | `Earring` | `ppg_ir` | 14411 | 9824 | 68.17% | 15.50 ms | 0.421 | 7.49 ms | 0.856 |
| <0.5 | `Earring` | `ppg_green` | 14411 | 12388 | 85.96% | 17.47 ms | 0.415 | 6.08 ms | 0.931 |
| <0.5 | `Watch` | `ppg_ir` | 12539 | 2341 | 18.67% | 77.19 ms | -0.051 | 96.16 ms | 0.151 |
| <1.0 | `Ring` | `ppg_green` | 13229 | 5670 | 42.86% | 11.06 ms | 0.548 | 10.98 ms | 0.796 |
| <1.0 | `Ring` | `ppg_ir` | 13229 | 3390 | 25.63% | 14.22 ms | 0.472 | 27.01 ms | 0.599 |
| <1.0 | `Watch` | `ppg_green` | 14820 | 3236 | 21.84% | 14.80 ms | 0.423 | 21.29 ms | 0.604 |
| <1.0 | `Earring` | `ppg_ir` | 15318 | 10327 | 67.42% | 15.49 ms | 0.427 | 7.50 ms | 0.855 |
| <1.0 | `Earring` | `ppg_green` | 15318 | 13110 | 85.59% | 17.50 ms | 0.416 | 6.06 ms | 0.930 |
| <1.0 | `Watch` | `ppg_ir` | 14820 | 2615 | 17.65% | 76.90 ms | -0.051 | 96.49 ms | 0.124 |

## Best Channel per Device at Each Motion Threshold

这里只用于解释 motion sensitivity；正式报告仍保留两个通道。

| Threshold | Device | Best channel by RMSSD MAE | RMSSD MAE | RMSSD R | Coverage within motion |
|---:|---|---|---:|---:|---:|
| <0.1 | `Earring` | `ppg_ir` | 14.83 ms | 0.403 | 88.60% |
| <0.1 | `Ring` | `ppg_green` | 10.48 ms | 0.806 | 79.57% |
| <0.1 | `Watch` | `ppg_green` | 11.74 ms | 0.599 | 49.88% |
| <0.2 | `Earring` | `ppg_ir` | 14.24 ms | 0.456 | 79.79% |
| <0.2 | `Ring` | `ppg_green` | 10.89 ms | 0.677 | 70.98% |
| <0.2 | `Watch` | `ppg_green` | 14.37 ms | 0.513 | 36.29% |
| <0.5 | `Earring` | `ppg_ir` | 15.50 ms | 0.421 | 68.17% |
| <0.5 | `Ring` | `ppg_green` | 10.83 ms | 0.580 | 54.75% |
| <0.5 | `Watch` | `ppg_green` | 14.78 ms | 0.424 | 25.34% |
| <1.0 | `Earring` | `ppg_ir` | 15.49 ms | 0.427 | 67.42% |
| <1.0 | `Ring` | `ppg_green` | 11.06 ms | 0.548 | 42.86% |
| <1.0 | `Watch` | `ppg_green` | 14.80 ms | 0.423 | 21.84% |

## Interpretation

- 更严格的 motion threshold 通常提升 Ring/Watch 的 RMSSD R、降低 MAE，但会牺牲大量 motion windows。
- Earring green/IR 在低 motion 子集下保留较多窗口，说明 Earring 的 motion 分布相对更温和；但 RMSSD agreement 仍通常是 IR 更好，SDNN 通常是 green 更好。
- Ring green 在所有阈值下都是 Ring 的最佳通道；Ring IR 保留汇报，但 MAE/R/coverage 都弱于 green。
- Watch green 是最受 motion threshold 影响的可用通道：`<0.1` 下表现明显改善，但这更多是 low-motion subset analysis，不代表全训练集表现。
- Watch IR 在所有阈值下仍明显失败；motion filtering 不能把 Watch IR 修成可用 HRV baseline。
- Coverage within motion 使用 motion 子集作为分母，不能直接和完整 training_stride30 overall coverage 混为一谈。

## Files

- `v1_2_motion_threshold_sensitivity_training_stride30.csv`
- `v1_2_motion_threshold_sensitivity_participant_training_stride30.csv`
- `v1_2_motion_lt_0p1_training_stride30.md`
- `v1_2_motion_lt_0p2_training_stride30.md`
- `v1_2_motion_lt_0p5_training_stride30.md`
- `v1_2_motion_lt_1p0_training_stride30.md`

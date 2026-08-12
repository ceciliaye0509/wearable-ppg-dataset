# v1.2 Motion Threshold Sensitivity on training_stride30

本汇总使用当前 v1.2 最佳 baseline：每个 `device x channel` 使用冻结参数，并且 green/IR 都汇报。

Motion threshold 横向比较使用同一批窗口交集：每个 threshold 先取所有汇报的 `device x channel` 都满足 `accel_motion_mean_mag < threshold` 的 `participant + window_index`，再在这批 common-motion windows 内计算各通道 QC-valid coverage、MAE 和 R。

## Threshold Impact Comparison

下表比较 motion threshold 从严格到宽松时，对当前 baseline 的影响。Coverage 使用同一批 common-motion windows 作分母。

| Device | Channel | Motion windows `<0.1 -> <1.0` | Coverage `<0.1 -> <1.0` | RMSSD MAE `<0.1 -> <1.0` | RMSSD R `<0.1 -> <1.0` | 主要影响 |
|---|---|---:|---:|---:|---:|---|
| `Earring` | `ppg_green` | 118 -> 13201 | 97.46% -> 86.14% | 9.17 -> 17.08 ms | 0.762 -> 0.436 | common low-motion 子集下误差最低；阈值放宽后样本更充分但 MAE 上升。 |
| `Earring` | `ppg_ir` | 118 -> 13201 | 96.61% -> 70.96% | 9.34 -> 15.54 ms | 0.660 -> 0.442 | common low-motion 子集下接近 green；阈值放宽后仍是 Earring 的主要 RMSSD 通道。 |
| `Ring` | `ppg_green` | 118 -> 13201 | 64.41% -> 42.95% | 12.25 -> 11.06 ms | 0.733 -> 0.548 | 在 common-motion 横向比较中保持 Ring 最佳；严格阈值样本很少。 |
| `Ring` | `ppg_ir` | 118 -> 13201 | 56.78% -> 25.64% | 13.94 -> 14.18 ms | 0.270 -> 0.474 | coverage 和 RMSSD agreement 均弱于 Ring green；严格阈值下 R 不稳定。 |
| `Watch` | `ppg_green` | 118 -> 13201 | 41.53% -> 24.10% | 12.26 -> 14.70 ms | 0.403 -> 0.431 | Watch 的可用通道；common-motion 下 coverage 仍明显低于 Earring/Ring green。 |
| `Watch` | `ppg_ir` | 118 -> 13201 | 61.02% -> 18.47% | 47.42 -> 77.23 ms | -0.515 -> -0.039 | 即使使用 common low-motion 子集，IR 仍明显失败。 |

核心结论：

- `<0.1` 是最严格的 common low-motion 子集，窗口数很少，代表性有限。
- `<0.2` 是更平衡的 common low-motion sensitivity：窗口数明显多于 `<0.1`。
- `<0.5` 和 `<1.0` 更接近 common low-to-moderate motion/full training 行为，窗口数多，但 Ring/Watch 的 coverage within motion 仍明显低于 Earring。
- 对当前 v1.2 baseline，common low-motion 子集会改变各通道 RMSSD 排序；Watch IR 始终不适合作为 HRV baseline。

## Aggregate RMSSD/SDNN Results

| Threshold | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| <0.1 | `Earring` | `ppg_green` | 118 | 115 | 97.46% | 9.17 ms | 0.762 | 7.73 ms | 0.706 |
| <0.1 | `Earring` | `ppg_ir` | 118 | 114 | 96.61% | 9.34 ms | 0.660 | 9.89 ms | 0.562 |
| <0.1 | `Ring` | `ppg_green` | 118 | 76 | 64.41% | 12.25 ms | 0.733 | 14.34 ms | 0.562 |
| <0.1 | `Watch` | `ppg_green` | 118 | 49 | 41.53% | 12.26 ms | 0.403 | 26.89 ms | 0.085 |
| <0.1 | `Ring` | `ppg_ir` | 118 | 67 | 56.78% | 13.94 ms | 0.270 | 26.74 ms | 0.061 |
| <0.1 | `Watch` | `ppg_ir` | 118 | 72 | 61.02% | 47.42 ms | -0.515 | 52.74 ms | -0.360 |
| <0.2 | `Ring` | `ppg_green` | 1367 | 995 | 72.79% | 10.78 ms | 0.731 | 10.73 ms | 0.792 |
| <0.2 | `Watch` | `ppg_green` | 1367 | 576 | 42.14% | 13.67 ms | 0.616 | 21.38 ms | 0.479 |
| <0.2 | `Ring` | `ppg_ir` | 1367 | 720 | 52.67% | 14.33 ms | 0.611 | 30.20 ms | 0.522 |
| <0.2 | `Earring` | `ppg_ir` | 1367 | 1235 | 90.34% | 14.42 ms | 0.642 | 7.55 ms | 0.872 |
| <0.2 | `Earring` | `ppg_green` | 1367 | 1311 | 95.90% | 14.93 ms | 0.632 | 6.16 ms | 0.914 |
| <0.2 | `Watch` | `ppg_ir` | 1367 | 560 | 40.97% | 69.04 ms | -0.063 | 82.88 ms | -0.065 |
| <0.5 | `Ring` | `ppg_green` | 7996 | 4395 | 54.96% | 10.79 ms | 0.590 | 11.15 ms | 0.780 |
| <0.5 | `Ring` | `ppg_ir` | 7996 | 2919 | 36.51% | 14.23 ms | 0.492 | 27.17 ms | 0.607 |
| <0.5 | `Watch` | `ppg_green` | 7996 | 2642 | 33.04% | 14.39 ms | 0.443 | 20.69 ms | 0.596 |
| <0.5 | `Earring` | `ppg_ir` | 7996 | 6321 | 79.05% | 14.84 ms | 0.489 | 7.71 ms | 0.838 |
| <0.5 | `Earring` | `ppg_green` | 7996 | 7251 | 90.68% | 16.35 ms | 0.489 | 5.96 ms | 0.922 |
| <0.5 | `Watch` | `ppg_ir` | 7996 | 1925 | 24.07% | 74.78 ms | -0.039 | 93.11 ms | 0.158 |
| <1.0 | `Ring` | `ppg_green` | 13201 | 5670 | 42.95% | 11.06 ms | 0.548 | 10.98 ms | 0.796 |
| <1.0 | `Ring` | `ppg_ir` | 13201 | 3385 | 25.64% | 14.18 ms | 0.474 | 26.99 ms | 0.600 |
| <1.0 | `Watch` | `ppg_green` | 13201 | 3181 | 24.10% | 14.70 ms | 0.431 | 21.22 ms | 0.604 |
| <1.0 | `Earring` | `ppg_ir` | 13201 | 9368 | 70.96% | 15.54 ms | 0.442 | 7.53 ms | 0.853 |
| <1.0 | `Earring` | `ppg_green` | 13201 | 11372 | 86.14% | 17.08 ms | 0.436 | 6.02 ms | 0.928 |
| <1.0 | `Watch` | `ppg_ir` | 13201 | 2438 | 18.47% | 77.23 ms | -0.039 | 96.34 ms | 0.136 |

## Best Channel per Device at Each Motion Threshold

这里只用于解释 motion sensitivity；正式报告仍保留两个通道。

| Threshold | Device | Best channel by RMSSD MAE | RMSSD MAE | RMSSD R | Coverage within motion |
|---:|---|---|---:|---:|---:|
| <0.1 | `Earring` | `ppg_green` | 9.17 ms | 0.762 | 97.46% |
| <0.1 | `Ring` | `ppg_green` | 12.25 ms | 0.733 | 64.41% |
| <0.1 | `Watch` | `ppg_green` | 12.26 ms | 0.403 | 41.53% |
| <0.2 | `Earring` | `ppg_ir` | 14.42 ms | 0.642 | 90.34% |
| <0.2 | `Ring` | `ppg_green` | 10.78 ms | 0.731 | 72.79% |
| <0.2 | `Watch` | `ppg_green` | 13.67 ms | 0.616 | 42.14% |
| <0.5 | `Earring` | `ppg_ir` | 14.84 ms | 0.489 | 79.05% |
| <0.5 | `Ring` | `ppg_green` | 10.79 ms | 0.590 | 54.96% |
| <0.5 | `Watch` | `ppg_green` | 14.39 ms | 0.443 | 33.04% |
| <1.0 | `Earring` | `ppg_ir` | 15.54 ms | 0.442 | 70.96% |
| <1.0 | `Ring` | `ppg_green` | 11.06 ms | 0.548 | 42.95% |
| <1.0 | `Watch` | `ppg_green` | 14.70 ms | 0.431 | 24.10% |

## Interpretation

- 更严格的 common motion threshold 通常提升部分通道的 RMSSD R、降低 MAE，但会牺牲大量 motion windows。
- Earring green/IR 在低 motion 子集下保留较多窗口，说明 Earring 的 motion 分布相对更温和；但 RMSSD agreement 仍通常是 IR 更好，SDNN 通常是 green 更好。
- Ring green 在所有阈值下都是 Ring 的最佳通道；Ring IR 保留汇报，但 MAE/R/coverage 都弱于 green。
- Watch green 是最受 motion threshold 影响的可用通道之一；严格阈值下样本量很小，因此不能单独代表全训练集表现。
- Watch IR 在所有阈值下仍明显失败；motion filtering 不能把 Watch IR 修成可用 HRV baseline。
- Coverage within motion 使用 common-motion 子集作为分母，不能直接和完整 training_stride30 overall coverage 混为一谈。

## Files

- `v1_2_motion_threshold_sensitivity_training_stride30.csv`
- `v1_2_motion_threshold_sensitivity_participant_training_stride30.csv`
- `v1_2_motion_lt_0p1_training_stride30.md`
- `v1_2_motion_lt_0p2_training_stride30.md`
- `v1_2_motion_lt_0p5_training_stride30.md`
- `v1_2_motion_lt_1p0_training_stride30.md`

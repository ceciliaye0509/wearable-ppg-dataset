# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30

本报告只在 no-motion 窗口上重新计算当前最佳 raw-aligned baseline。

## No-Motion 定义

- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < 0.1`。
- `accel_motion_mean_mag` 是数据集内保存的去重力 motion magnitude 均值。
- 过滤发生在 device 层面：同一个 window 对 Earring 可能是 no-motion，对 Ring/Watch 不一定是 no-motion。

## Baseline 定义

- 方法：`no_motion_device_best_sqi_green_or_ir__hrv_from_ppg`。
- 对 no-motion 子集中的每个 window、每个 device 独立评估 `ppg_green` 与 `ppg_ir`。
- 使用 `heuristic_baselines.algorithms.hrv.hrv_from_ppg` 从 PPG 重新计算 PRV/HRV。
- 若 green 和 IR 都有效，则仅用 PPG 自身信息选择 SQI 更高的一路；不使用 ECG label 选通道。

## 数据集

| Dataset | Total windows | No-motion windows | Interpretation |
|---|---:|---:|---|
| `strict_reference` | 5118 | 567 | non-overlapping 5 min windows, filtered to per-device no-motion windows |
| `training_v1_stride30` | 51192 | 5737 | 30 s stride rolling windows, filtered to per-device no-motion windows |

## Dataset x Device 汇总（RMSSD）

| Dataset | Device | Participants | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | Mean participant RMSSD MAE | Median participant R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 16 | 1706 | 406 | 394 | 23.8 | 97.0 | 11.97 | 0.356 |
| strict_reference | Ring | 16 | 1706 | 36 | 33 | 2.1 | 91.7 | 12.09 | 0.736 |
| strict_reference | Watch | 16 | 1706 | 125 | 52 | 7.3 | 41.6 | 18.90 | 0.378 |
| training_v1_stride30 | Earring | 16 | 17064 | 4070 | 3948 | 23.9 | 97.0 | 13.99 | 0.287 |
| training_v1_stride30 | Ring | 16 | 17064 | 372 | 325 | 2.2 | 87.4 | 14.24 | 0.428 |
| training_v1_stride30 | Watch | 16 | 17064 | 1295 | 498 | 7.6 | 38.5 | 18.49 | 0.356 |

## 每个参与者 x 每个设备结果（RMSSD）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms | Raw accel no-motion | Motion accel no-motion | Green selected | IR selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 1 | 1 | 0.9 | 100.0 | 4.14 | 4.14 |  | -4.14 | 1.63 | 0.10 | 0 | 1 |
| strict_reference | P1 | Ring | 107 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P1 | Watch | 107 | 1 | 0 | 0.9 | 0.0 |  |  |  |  | 2.25 | 0.06 | 0 | 0 |
| strict_reference | P3 | Earring | 361 | 48 | 48 | 13.3 | 100.0 | 22.23 | 23.89 | 0.318 | -21.74 | 9.76 | 0.07 | 10 | 38 |
| strict_reference | P3 | Ring | 361 | 1 | 0 | 0.3 | 0.0 |  |  |  |  | 9.81 | 0.09 | 0 | 0 |
| strict_reference | P3 | Watch | 361 | 3 | 2 | 0.8 | 66.7 | 5.11 | 6.06 |  | 3.26 | 9.84 | 0.09 | 1 | 1 |
| strict_reference | P4 | Earring | 129 | 35 | 30 | 27.1 | 85.7 | 9.00 | 14.21 | 0.626 | -5.30 | 9.79 | 0.07 | 13 | 17 |
| strict_reference | P4 | Ring | 129 | 7 | 7 | 5.4 | 100.0 | 8.51 | 9.96 | 0.736 | 2.65 | 9.83 | 0.07 | 7 | 0 |
| strict_reference | P4 | Watch | 129 | 22 | 10 | 17.1 | 45.5 | 15.35 | 18.56 | 0.237 | 15.35 | 9.82 | 0.05 | 9 | 1 |
| strict_reference | P5 | Earring | 14 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P5 | Ring | 14 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P5 | Watch | 14 | 1 | 0 | 7.1 | 0.0 |  |  |  |  | 9.78 | 0.07 | 0 | 0 |
| strict_reference | P6 | Earring | 119 | 28 | 26 | 23.5 | 92.9 | 9.65 | 14.52 | 0.329 | 3.18 | 9.73 | 0.08 | 1 | 25 |
| strict_reference | P6 | Ring | 119 | 14 | 12 | 11.8 | 85.7 | 7.80 | 9.93 | 0.035 | 5.97 | 9.78 | 0.08 | 7 | 5 |
| strict_reference | P6 | Watch | 119 | 2 | 1 | 1.7 | 50.0 | 23.33 | 23.33 |  | 23.33 | 9.83 | 0.07 | 1 | 0 |
| strict_reference | P7 | Earring | 62 | 22 | 22 | 35.5 | 100.0 | 4.46 | 6.39 | 0.748 | 2.81 | 9.88 | 0.08 | 0 | 22 |
| strict_reference | P7 | Ring | 62 | 1 | 1 | 1.6 | 100.0 | 21.81 | 21.81 |  | 21.81 | 9.81 | 0.07 | 1 | 0 |
| strict_reference | P7 | Watch | 62 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P8 | Earring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P8 | Ring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P8 | Watch | 136 | 2 | 1 | 1.5 | 50.0 | 31.08 | 31.08 |  | 31.08 | 9.82 | 0.09 | 1 | 0 |
| strict_reference | P9 | Earring | 136 | 87 | 86 | 64.0 | 98.9 | 7.52 | 9.74 | 0.745 | -0.99 | 9.85 | 0.07 | 23 | 63 |
| strict_reference | P9 | Ring | 136 | 8 | 8 | 5.9 | 100.0 | 8.36 | 11.30 | 0.891 | 2.90 | 9.85 | 0.07 | 5 | 3 |
| strict_reference | P9 | Watch | 136 | 13 | 5 | 9.6 | 38.5 | 11.08 | 15.63 | 0.945 | 8.64 | 9.79 | 0.07 | 5 | 0 |
| strict_reference | P10 | Earring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P10 | Ring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P10 | Watch | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P11 | Earring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P11 | Ring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P11 | Watch | 37 | 4 | 4 | 10.8 | 100.0 | 16.56 | 19.47 | -0.144 | 16.56 | 9.84 | 0.04 | 4 | 0 |
| strict_reference | P12 | Earring | 78 | 17 | 17 | 21.8 | 100.0 | 7.54 | 9.89 | 0.383 | 1.42 | 9.87 | 0.08 | 13 | 4 |
| strict_reference | P12 | Ring | 78 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P12 | Watch | 78 | 16 | 11 | 20.5 | 68.8 | 10.81 | 12.91 | 0.539 | 10.81 | 9.81 | 0.08 | 11 | 0 |
| strict_reference | P13 | Earring | 42 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P13 | Ring | 42 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P13 | Watch | 42 | 3 | 0 | 7.1 | 0.0 |  |  |  |  | 9.85 | 0.08 | 0 | 0 |
| strict_reference | P15 | Earring | 100 | 41 | 41 | 41.0 | 100.0 | 12.62 | 16.78 | -0.271 | 2.50 | 9.83 | 0.08 | 17 | 24 |
| strict_reference | P15 | Ring | 100 | 2 | 2 | 2.0 | 100.0 | 9.28 | 10.37 |  | 4.63 | 9.87 | 0.09 | 2 | 0 |
| strict_reference | P15 | Watch | 100 | 18 | 5 | 18.0 | 27.8 | 30.64 | 32.71 | -0.643 | 30.64 | 9.80 | 0.07 | 5 | 0 |
| strict_reference | P18 | Earring | 187 | 66 | 63 | 35.3 | 95.5 | 10.89 | 13.88 | 0.542 | -8.73 | 9.84 | 0.07 | 1 | 62 |
| strict_reference | P18 | Ring | 187 | 2 | 2 | 1.1 | 100.0 | 7.98 | 8.22 |  | -7.98 | 9.79 | 0.06 | 2 | 0 |
| strict_reference | P18 | Watch | 187 | 28 | 11 | 15.0 | 39.3 | 29.27 | 34.14 | 0.519 | 28.79 | 9.83 | 0.07 | 9 | 2 |
| strict_reference | P19 | Earring | 87 | 56 | 55 | 64.4 | 98.2 | 22.28 | 24.76 | 0.235 | -16.43 | 9.81 | 0.06 | 15 | 40 |
| strict_reference | P19 | Ring | 87 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P19 | Watch | 87 | 12 | 2 | 13.8 | 16.7 | 15.77 | 15.89 |  | 15.77 | 9.84 | 0.08 | 2 | 0 |
| strict_reference | P20 | Earring | 45 | 5 | 5 | 11.1 | 100.0 | 21.39 | 22.73 | 0.240 | -21.39 | 9.76 | 0.08 | 4 | 1 |
| strict_reference | P20 | Ring | 45 | 1 | 1 | 2.2 | 100.0 | 20.92 | 20.92 |  | -20.92 | 9.79 | 0.04 | 0 | 1 |
| strict_reference | P20 | Watch | 45 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P1 | Earring | 1087 | 26 | 26 | 2.4 | 100.0 | 9.55 | 12.04 | 0.169 | -9.40 | 1.62 | 0.09 | 1 | 25 |
| training_v1_stride30 | P1 | Ring | 1087 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P1 | Watch | 1087 | 16 | 0 | 1.5 | 0.0 |  |  |  |  | 2.23 | 0.06 | 0 | 0 |
| training_v1_stride30 | P3 | Earring | 3589 | 489 | 488 | 13.6 | 99.8 | 22.34 | 24.08 | 0.336 | -21.99 | 9.76 | 0.07 | 90 | 398 |
| training_v1_stride30 | P3 | Ring | 3589 | 17 | 10 | 0.5 | 58.8 | 29.08 | 29.86 | 0.031 | -29.08 | 9.78 | 0.09 | 10 | 0 |
| training_v1_stride30 | P3 | Watch | 3589 | 27 | 15 | 0.8 | 55.6 | 9.40 | 10.80 | -0.733 | 7.62 | 9.85 | 0.09 | 5 | 10 |
| training_v1_stride30 | P4 | Earring | 1297 | 332 | 292 | 25.6 | 88.0 | 9.00 | 14.07 | 0.639 | -4.88 | 9.79 | 0.07 | 139 | 153 |
| training_v1_stride30 | P4 | Ring | 1297 | 72 | 63 | 5.6 | 87.5 | 8.37 | 9.78 | 0.697 | 3.04 | 9.82 | 0.06 | 63 | 0 |
| training_v1_stride30 | P4 | Watch | 1297 | 190 | 87 | 14.6 | 45.8 | 15.23 | 17.78 | 0.465 | 14.84 | 9.81 | 0.05 | 75 | 12 |
| training_v1_stride30 | P5 | Earring | 144 | 3 | 0 | 2.1 | 0.0 |  |  |  |  | 9.90 | 0.10 | 0 | 0 |
| training_v1_stride30 | P5 | Ring | 144 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P5 | Watch | 144 | 17 | 0 | 11.8 | 0.0 |  |  |  |  | 9.78 | 0.08 | 0 | 0 |
| training_v1_stride30 | P6 | Earring | 1132 | 256 | 238 | 22.6 | 93.0 | 9.50 | 14.22 | 0.269 | 1.56 | 9.73 | 0.08 | 1 | 237 |
| training_v1_stride30 | P6 | Ring | 1132 | 134 | 115 | 11.8 | 85.8 | 9.79 | 12.06 | 0.306 | 7.93 | 9.79 | 0.08 | 84 | 31 |
| training_v1_stride30 | P6 | Watch | 1132 | 22 | 10 | 1.9 | 45.5 | 21.41 | 21.83 | 0.582 | 21.41 | 9.81 | 0.08 | 10 | 0 |
| training_v1_stride30 | P7 | Earring | 610 | 187 | 187 | 30.7 | 100.0 | 4.43 | 6.41 | 0.684 | 2.47 | 9.88 | 0.08 | 1 | 186 |
| training_v1_stride30 | P7 | Ring | 610 | 7 | 7 | 1.1 | 100.0 | 19.28 | 20.41 | 0.991 | 18.38 | 9.82 | 0.08 | 7 | 0 |
| training_v1_stride30 | P7 | Watch | 610 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P8 | Earring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P8 | Ring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P8 | Watch | 1381 | 10 | 7 | 0.7 | 70.0 | 26.04 | 26.37 | 0.934 | 26.04 | 9.84 | 0.08 | 7 | 0 |
| training_v1_stride30 | P9 | Earring | 1383 | 906 | 894 | 65.5 | 98.7 | 7.15 | 9.28 | 0.754 | -1.62 | 9.85 | 0.07 | 252 | 642 |
| training_v1_stride30 | P9 | Ring | 1383 | 85 | 78 | 6.1 | 91.8 | 7.60 | 9.22 | 0.929 | 4.08 | 9.84 | 0.07 | 53 | 25 |
| training_v1_stride30 | P9 | Watch | 1383 | 171 | 66 | 12.4 | 38.6 | 13.53 | 16.63 | 0.882 | 12.03 | 9.80 | 0.07 | 63 | 3 |
| training_v1_stride30 | P10 | Earring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P10 | Ring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P10 | Watch | 645 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P11 | Earring | 350 | 3 | 3 | 0.9 | 100.0 | 33.29 | 33.52 | -0.722 | -33.29 | 9.89 | 0.09 | 0 | 3 |
| training_v1_stride30 | P11 | Ring | 350 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P11 | Watch | 350 | 54 | 46 | 15.4 | 85.2 | 17.14 | 18.83 | -0.287 | 17.14 | 9.84 | 0.04 | 46 | 0 |
| training_v1_stride30 | P12 | Earring | 805 | 180 | 178 | 22.4 | 98.9 | 7.62 | 10.34 | 0.231 | 2.05 | 9.87 | 0.09 | 140 | 38 |
| training_v1_stride30 | P12 | Ring | 805 | 2 | 0 | 0.2 | 0.0 |  |  |  |  | 9.84 | 0.08 | 0 | 0 |
| training_v1_stride30 | P12 | Watch | 805 | 162 | 106 | 20.1 | 65.4 | 12.05 | 14.19 | 0.254 | 10.95 | 9.81 | 0.08 | 106 | 0 |
| training_v1_stride30 | P13 | Earring | 431 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P13 | Ring | 431 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P13 | Watch | 431 | 14 | 0 | 3.2 | 0.0 |  |  |  |  | 9.86 | 0.08 | 0 | 0 |
| training_v1_stride30 | P15 | Earring | 1009 | 430 | 423 | 42.6 | 98.4 | 11.27 | 14.93 | -0.188 | 1.65 | 9.83 | 0.08 | 169 | 254 |
| training_v1_stride30 | P15 | Ring | 1009 | 21 | 21 | 2.1 | 100.0 | 14.16 | 16.45 | 0.636 | 9.49 | 9.85 | 0.10 | 21 | 0 |
| training_v1_stride30 | P15 | Watch | 1009 | 181 | 49 | 17.9 | 27.1 | 24.55 | 27.57 | -0.142 | 24.55 | 9.79 | 0.08 | 49 | 0 |
| training_v1_stride30 | P18 | Earring | 1849 | 663 | 634 | 35.9 | 95.6 | 11.36 | 14.23 | 0.590 | -8.18 | 9.84 | 0.07 | 9 | 625 |
| training_v1_stride30 | P18 | Ring | 1849 | 19 | 19 | 1.0 | 100.0 | 13.79 | 19.13 | -0.366 | -0.06 | 9.81 | 0.08 | 19 | 0 |
| training_v1_stride30 | P18 | Watch | 1849 | 301 | 94 | 16.3 | 31.2 | 30.90 | 36.82 | 0.368 | 30.51 | 9.83 | 0.07 | 83 | 11 |
| training_v1_stride30 | P19 | Earring | 904 | 542 | 532 | 60.0 | 98.2 | 22.04 | 24.61 | 0.234 | -16.16 | 9.81 | 0.06 | 140 | 392 |
| training_v1_stride30 | P19 | Ring | 904 | 8 | 5 | 0.9 | 62.5 | 7.64 | 7.69 | 0.428 | 7.64 | 9.75 | 0.09 | 5 | 0 |
| training_v1_stride30 | P19 | Watch | 904 | 126 | 18 | 13.9 | 14.3 | 14.66 | 15.54 | 0.344 | 14.66 | 9.83 | 0.08 | 18 | 0 |
| training_v1_stride30 | P20 | Earring | 448 | 53 | 53 | 11.8 | 100.0 | 20.27 | 21.80 | 0.304 | -20.27 | 9.76 | 0.08 | 43 | 10 |
| training_v1_stride30 | P20 | Ring | 448 | 7 | 7 | 1.6 | 100.0 | 18.44 | 18.82 | -0.437 | -18.44 | 9.81 | 0.06 | 4 | 3 |
| training_v1_stride30 | P20 | Watch | 448 | 4 | 0 | 0.9 | 0.0 |  |  |  |  | 9.77 | 0.09 | 0 | 0 |

## 每个参与者 x 每个设备结果（SDNN）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 1 | 1 | 0.9 | 100.0 | 2.60 | 2.60 |  | 2.60 |
| strict_reference | P1 | Ring | 107 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P1 | Watch | 107 | 1 | 0 | 0.9 | 0.0 |  |  |  |  |
| strict_reference | P3 | Earring | 361 | 48 | 48 | 13.3 | 100.0 | 7.05 | 7.80 | 0.949 | -6.60 |
| strict_reference | P3 | Ring | 361 | 1 | 0 | 0.3 | 0.0 |  |  |  |  |
| strict_reference | P3 | Watch | 361 | 3 | 2 | 0.8 | 66.7 | 6.76 | 6.78 |  | 0.42 |
| strict_reference | P4 | Earring | 129 | 35 | 30 | 27.1 | 85.7 | 6.18 | 11.28 | 0.649 | -2.85 |
| strict_reference | P4 | Ring | 129 | 7 | 7 | 5.4 | 100.0 | 13.45 | 19.63 | -0.121 | -6.69 |
| strict_reference | P4 | Watch | 129 | 22 | 10 | 17.1 | 45.5 | 17.61 | 20.17 | -0.282 | 8.33 |
| strict_reference | P5 | Earring | 14 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P5 | Ring | 14 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P5 | Watch | 14 | 1 | 0 | 7.1 | 0.0 |  |  |  |  |
| strict_reference | P6 | Earring | 119 | 28 | 26 | 23.5 | 92.9 | 3.99 | 6.09 | 0.828 | 1.43 |
| strict_reference | P6 | Ring | 119 | 14 | 12 | 11.8 | 85.7 | 5.22 | 7.28 | 0.714 | 4.80 |
| strict_reference | P6 | Watch | 119 | 2 | 1 | 1.7 | 50.0 | 11.06 | 11.06 |  | 11.06 |
| strict_reference | P7 | Earring | 62 | 22 | 22 | 35.5 | 100.0 | 2.51 | 3.28 | 0.981 | 2.44 |
| strict_reference | P7 | Ring | 62 | 1 | 1 | 1.6 | 100.0 | 11.53 | 11.53 |  | 11.53 |
| strict_reference | P7 | Watch | 62 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P8 | Earring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P8 | Ring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P8 | Watch | 136 | 2 | 1 | 1.5 | 50.0 | 27.73 | 27.73 |  | 27.73 |
| strict_reference | P9 | Earring | 136 | 87 | 86 | 64.0 | 98.9 | 3.08 | 5.12 | 0.940 | 1.22 |
| strict_reference | P9 | Ring | 136 | 8 | 8 | 5.9 | 100.0 | 7.80 | 10.10 | 0.836 | 5.08 |
| strict_reference | P9 | Watch | 136 | 13 | 5 | 9.6 | 38.5 | 9.32 | 12.12 | 0.968 | 2.22 |
| strict_reference | P10 | Earring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P10 | Ring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P10 | Watch | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P11 | Earring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P11 | Ring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P11 | Watch | 37 | 4 | 4 | 10.8 | 100.0 | 8.63 | 9.85 | 0.843 | 8.63 |
| strict_reference | P12 | Earring | 78 | 17 | 17 | 21.8 | 100.0 | 3.04 | 4.44 | 0.960 | 1.81 |
| strict_reference | P12 | Ring | 78 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P12 | Watch | 78 | 16 | 11 | 20.5 | 68.8 | 5.45 | 7.05 | 0.763 | 4.20 |
| strict_reference | P13 | Earring | 42 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P13 | Ring | 42 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P13 | Watch | 42 | 3 | 0 | 7.1 | 0.0 |  |  |  |  |
| strict_reference | P15 | Earring | 100 | 41 | 41 | 41.0 | 100.0 | 6.99 | 9.91 | 0.669 | 3.47 |
| strict_reference | P15 | Ring | 100 | 2 | 2 | 2.0 | 100.0 | 5.99 | 7.60 |  | 5.99 |
| strict_reference | P15 | Watch | 100 | 18 | 5 | 18.0 | 27.8 | 17.53 | 20.22 | 0.074 | 17.53 |
| strict_reference | P18 | Earring | 187 | 66 | 63 | 35.3 | 95.5 | 4.72 | 6.91 | 0.921 | -2.03 |
| strict_reference | P18 | Ring | 187 | 2 | 2 | 1.1 | 100.0 | 3.27 | 3.96 |  | -3.27 |
| strict_reference | P18 | Watch | 187 | 28 | 11 | 15.0 | 39.3 | 33.12 | 48.42 | 0.176 | 33.12 |
| strict_reference | P19 | Earring | 87 | 56 | 55 | 64.4 | 98.2 | 11.20 | 12.70 | 0.671 | -2.83 |
| strict_reference | P19 | Ring | 87 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P19 | Watch | 87 | 12 | 2 | 13.8 | 16.7 | 19.87 | 20.03 |  | 19.87 |
| strict_reference | P20 | Earring | 45 | 5 | 5 | 11.1 | 100.0 | 5.77 | 6.30 | 0.976 | -5.77 |
| strict_reference | P20 | Ring | 45 | 1 | 1 | 2.2 | 100.0 | 6.36 | 6.36 |  | -6.36 |
| strict_reference | P20 | Watch | 45 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P1 | Earring | 1087 | 26 | 26 | 2.4 | 100.0 | 3.56 | 4.25 | 0.958 | -1.18 |
| training_v1_stride30 | P1 | Ring | 1087 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P1 | Watch | 1087 | 16 | 0 | 1.5 | 0.0 |  |  |  |  |
| training_v1_stride30 | P3 | Earring | 3589 | 489 | 488 | 13.6 | 99.8 | 7.14 | 7.85 | 0.942 | -6.79 |
| training_v1_stride30 | P3 | Ring | 3589 | 17 | 10 | 0.5 | 58.8 | 8.21 | 8.36 | 0.982 | -8.21 |
| training_v1_stride30 | P3 | Watch | 3589 | 27 | 15 | 0.8 | 55.6 | 7.96 | 8.13 | -0.069 | 3.39 |
| training_v1_stride30 | P4 | Earring | 1297 | 332 | 292 | 25.6 | 88.0 | 5.62 | 11.07 | 0.666 | -1.99 |
| training_v1_stride30 | P4 | Ring | 1297 | 72 | 63 | 5.6 | 87.5 | 12.51 | 19.86 | -0.020 | -3.31 |
| training_v1_stride30 | P4 | Watch | 1297 | 190 | 87 | 14.6 | 45.8 | 17.29 | 21.95 | 0.060 | 11.24 |
| training_v1_stride30 | P5 | Earring | 144 | 3 | 0 | 2.1 | 0.0 |  |  |  |  |
| training_v1_stride30 | P5 | Ring | 144 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P5 | Watch | 144 | 17 | 0 | 11.8 | 0.0 |  |  |  |  |
| training_v1_stride30 | P6 | Earring | 1132 | 256 | 238 | 22.6 | 93.0 | 4.01 | 6.39 | 0.812 | 1.12 |
| training_v1_stride30 | P6 | Ring | 1132 | 134 | 115 | 11.8 | 85.8 | 7.90 | 12.92 | 0.507 | 7.40 |
| training_v1_stride30 | P6 | Watch | 1132 | 22 | 10 | 1.9 | 45.5 | 12.43 | 13.32 | 0.248 | 12.43 |
| training_v1_stride30 | P7 | Earring | 610 | 187 | 187 | 30.7 | 100.0 | 2.64 | 3.62 | 0.968 | 2.32 |
| training_v1_stride30 | P7 | Ring | 610 | 7 | 7 | 1.1 | 100.0 | 13.91 | 15.31 | 0.936 | 13.91 |
| training_v1_stride30 | P7 | Watch | 610 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P8 | Earring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P8 | Ring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P8 | Watch | 1381 | 10 | 7 | 0.7 | 70.0 | 12.61 | 18.34 | -0.958 | 12.12 |
| training_v1_stride30 | P9 | Earring | 1383 | 906 | 894 | 65.5 | 98.7 | 2.78 | 5.13 | 0.938 | 0.82 |
| training_v1_stride30 | P9 | Ring | 1383 | 85 | 78 | 6.1 | 91.8 | 9.06 | 13.24 | 0.859 | 7.13 |
| training_v1_stride30 | P9 | Watch | 1383 | 171 | 66 | 12.4 | 38.6 | 12.81 | 16.41 | 0.672 | 6.18 |
| training_v1_stride30 | P10 | Earring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P10 | Ring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P10 | Watch | 645 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P11 | Earring | 350 | 3 | 3 | 0.9 | 100.0 | 11.80 | 11.96 | 0.742 | -11.80 |
| training_v1_stride30 | P11 | Ring | 350 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P11 | Watch | 350 | 54 | 46 | 15.4 | 85.2 | 10.09 | 11.13 | 0.675 | 10.09 |
| training_v1_stride30 | P12 | Earring | 805 | 180 | 178 | 22.4 | 98.9 | 3.46 | 6.10 | 0.943 | 0.98 |
| training_v1_stride30 | P12 | Ring | 805 | 2 | 0 | 0.2 | 0.0 |  |  |  |  |
| training_v1_stride30 | P12 | Watch | 805 | 162 | 106 | 20.1 | 65.4 | 6.93 | 9.55 | 0.716 | 5.26 |
| training_v1_stride30 | P13 | Earring | 431 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P13 | Ring | 431 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P13 | Watch | 431 | 14 | 0 | 3.2 | 0.0 |  |  |  |  |
| training_v1_stride30 | P15 | Earring | 1009 | 430 | 423 | 42.6 | 98.4 | 6.83 | 9.82 | 0.709 | 3.50 |
| training_v1_stride30 | P15 | Ring | 1009 | 21 | 21 | 2.1 | 100.0 | 6.94 | 9.80 | 0.772 | 6.23 |
| training_v1_stride30 | P15 | Watch | 1009 | 181 | 49 | 17.9 | 27.1 | 16.72 | 20.55 | 0.477 | 16.72 |
| training_v1_stride30 | P18 | Earring | 1849 | 663 | 634 | 35.9 | 95.6 | 5.35 | 8.68 | 0.889 | -1.60 |
| training_v1_stride30 | P18 | Ring | 1849 | 19 | 19 | 1.0 | 100.0 | 8.34 | 13.77 | 0.685 | 2.45 |
| training_v1_stride30 | P18 | Watch | 1849 | 301 | 94 | 16.3 | 31.2 | 29.64 | 37.91 | 0.302 | 29.59 |
| training_v1_stride30 | P19 | Earring | 904 | 542 | 532 | 60.0 | 98.2 | 11.38 | 13.41 | 0.648 | -2.75 |
| training_v1_stride30 | P19 | Ring | 904 | 8 | 5 | 0.9 | 62.5 | 10.26 | 10.99 | -0.525 | 10.26 |
| training_v1_stride30 | P19 | Watch | 904 | 126 | 18 | 13.9 | 14.3 | 22.22 | 25.66 | 0.403 | 22.22 |
| training_v1_stride30 | P20 | Earring | 448 | 53 | 53 | 11.8 | 100.0 | 6.35 | 7.68 | 0.797 | -4.33 |
| training_v1_stride30 | P20 | Ring | 448 | 7 | 7 | 1.6 | 100.0 | 5.22 | 5.48 | 0.424 | -4.14 |
| training_v1_stride30 | P20 | Watch | 448 | 4 | 0 | 0.9 | 0.0 |  |  |  |  |

## 与完整窗口报告的比较

总体结论：no-motion 子集的 MAE 通常更好，且在 no-motion 子集内部的有效预测率更高；但相关系数 `R` 没有稳定改善，很多 participant/device 因 no-motion 窗口数较少而变得更不稳定。

### 有效配对行统计

| Metric | Paired MAE rows | No-motion MAE better | Paired R rows | No-motion R better | Coverage rows | No-motion coverage higher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSSD | 59 | 36 | 50 | 18 | 96 | 57 |
| SDNN | 59 | 40 | 50 | 16 | 96 | 57 |

### Dataset x Device 汇总比较（RMSSD）

| Dataset | Device | MAE full | MAE no-motion | R full | R no-motion | Coverage full | Coverage no-motion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_reference | Earring | 14.99 | 11.97 | 0.362 | 0.356 | 80.9% | 97.0% |
| strict_reference | Ring | 11.84 | 12.09 | 0.403 | 0.736 | 32.5% | 91.7% |
| strict_reference | Watch | 17.25 | 18.90 | 0.421 | 0.378 | 10.6% | 41.6% |
| training_v1_stride30 | Earring | 15.13 | 13.99 | 0.394 | 0.287 | 80.9% | 97.0% |
| training_v1_stride30 | Ring | 11.92 | 14.24 | 0.429 | 0.428 | 33.1% | 87.4% |
| training_v1_stride30 | Watch | 16.18 | 18.49 | 0.282 | 0.356 | 10.7% | 38.5% |

### Dataset x Device 汇总比较（SDNN）

| Dataset | Device | MAE full | MAE no-motion | R full | R no-motion | Coverage full | Coverage no-motion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_reference | Earring | 6.42 | 5.19 | 0.866 | 0.931 | 80.9% | 97.0% |
| strict_reference | Ring | 9.87 | 7.66 | 0.636 | 0.714 | 32.5% | 91.7% |
| strict_reference | Watch | 15.25 | 15.71 | 0.572 | 0.469 | 10.6% | 41.6% |
| training_v1_stride30 | Earring | 6.26 | 5.91 | 0.892 | 0.850 | 80.9% | 97.0% |
| training_v1_stride30 | Ring | 10.46 | 9.15 | 0.675 | 0.685 | 33.1% | 87.4% |
| training_v1_stride30 | Watch | 15.49 | 14.87 | 0.495 | 0.353 | 10.7% | 38.5% |

解释上要注意：这里的 no-motion coverage 是 `coverage within no-motion windows`，分母只包含已经被判定为 no-motion 的窗口。因此它不能直接等同于完整报告的整体 coverage。更准确的表述是：在接近静止的窗口中，PPG baseline 更容易给出有效预测；但可用窗口总数明显减少，尤其 Ring 和 Watch 的 no-motion 比例较低。

## 重要解析

- 这个报告回答的是“在数据集内接近静止的窗口中，当前 PPG heuristic baseline 表现如何”。
- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。
- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。
- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。
- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。
- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。

## 输出文件

- Markdown report: `current_best_baseline_no_motion_strict_vs_training_stride30_by_participant_device.md`
- Machine-readable table: `current_best_baseline_no_motion_strict_vs_training_stride30_by_participant_device.csv`

Generated in 347.0 seconds.

# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30

本报告只在指定 motion threshold 子集上重新计算当前最佳 raw-aligned baseline。

## Motion Threshold 定义

- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < 0.2`。
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
| `strict_reference` | 5118 | 1708 | non-overlapping 5 min windows, filtered to per-device no-motion windows |
| `training_v1_stride30` | 51192 | 16960 | 30 s stride rolling windows, filtered to per-device no-motion windows |

## Dataset x Device 汇总（RMSSD）

| Dataset | Device | Participants | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | Mean participant RMSSD MAE | Median participant R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 16 | 1706 | 940 | 874 | 55.1 | 93.0 | 13.20 | 0.546 |
| strict_reference | Ring | 16 | 1706 | 219 | 141 | 12.8 | 64.4 | 12.73 | 0.602 |
| strict_reference | Watch | 16 | 1706 | 549 | 116 | 32.2 | 21.1 | 19.55 | 0.265 |
| training_v1_stride30 | Earring | 16 | 17064 | 9301 | 8658 | 54.5 | 93.1 | 13.62 | 0.425 |
| training_v1_stride30 | Ring | 16 | 17064 | 2092 | 1406 | 12.3 | 67.2 | 12.35 | 0.230 |
| training_v1_stride30 | Watch | 16 | 17064 | 5567 | 1154 | 32.6 | 20.7 | 16.79 | 0.278 |

## 每个参与者 x 每个设备结果（RMSSD）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms | Raw accel no-motion | Motion accel no-motion | Green selected | IR selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 20 | 16 | 18.7 | 80.0 | 9.00 | 10.70 | -0.038 | -2.63 | 1.61 | 0.14 | 2 | 14 |
| strict_reference | P1 | Ring | 107 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P1 | Watch | 107 | 16 | 2 | 15.0 | 12.5 | 22.58 | 22.82 |  | 22.58 | 2.34 | 0.15 | 0 | 2 |
| strict_reference | P3 | Earring | 361 | 118 | 118 | 32.7 | 100.0 | 20.26 | 22.99 | 0.047 | -18.94 | 9.82 | 0.12 | 41 | 77 |
| strict_reference | P3 | Ring | 361 | 20 | 9 | 5.5 | 45.0 | 14.50 | 15.63 | 0.842 | -12.15 | 9.79 | 0.15 | 8 | 1 |
| strict_reference | P3 | Watch | 361 | 42 | 11 | 11.6 | 26.2 | 10.80 | 12.39 | -0.511 | 9.59 | 9.86 | 0.15 | 6 | 5 |
| strict_reference | P4 | Earring | 129 | 54 | 44 | 41.9 | 81.5 | 8.43 | 13.05 | 0.651 | -3.14 | 9.81 | 0.10 | 23 | 21 |
| strict_reference | P4 | Ring | 129 | 10 | 8 | 7.8 | 80.0 | 7.87 | 9.39 | 0.749 | 2.74 | 9.79 | 0.09 | 8 | 0 |
| strict_reference | P4 | Watch | 129 | 50 | 11 | 38.8 | 22.0 | 15.54 | 18.45 | 0.364 | 15.54 | 9.82 | 0.11 | 10 | 1 |
| strict_reference | P5 | Earring | 14 | 10 | 5 | 71.4 | 50.0 | 19.07 | 22.61 | -0.078 | -18.61 | 9.93 | 0.13 | 1 | 4 |
| strict_reference | P5 | Ring | 14 | 4 | 1 | 28.6 | 25.0 | 13.83 | 13.83 |  | -13.83 | 9.92 | 0.17 | 1 | 0 |
| strict_reference | P5 | Watch | 14 | 4 | 0 | 28.6 | 0.0 |  |  |  |  | 9.88 | 0.14 | 0 | 0 |
| strict_reference | P6 | Earring | 119 | 112 | 107 | 94.1 | 95.5 | 13.65 | 17.34 | 0.157 | -4.37 | 9.69 | 0.13 | 3 | 104 |
| strict_reference | P6 | Ring | 119 | 74 | 48 | 62.2 | 64.9 | 12.44 | 15.17 | 0.397 | 9.76 | 9.80 | 0.13 | 38 | 10 |
| strict_reference | P6 | Watch | 119 | 29 | 17 | 24.4 | 58.6 | 27.86 | 29.04 | 0.480 | 27.86 | 9.77 | 0.15 | 17 | 0 |
| strict_reference | P7 | Earring | 62 | 61 | 60 | 98.4 | 98.4 | 5.16 | 7.34 | 0.843 | 3.03 | 9.91 | 0.11 | 6 | 54 |
| strict_reference | P7 | Ring | 62 | 2 | 1 | 3.2 | 50.0 | 21.81 | 21.81 |  | 21.81 | 9.87 | 0.11 | 1 | 0 |
| strict_reference | P7 | Watch | 62 | 20 | 7 | 32.3 | 35.0 | 19.39 | 20.17 | 0.198 | 19.39 | 9.72 | 0.18 | 7 | 0 |
| strict_reference | P8 | Earring | 136 | 28 | 18 | 20.6 | 64.3 | 12.40 | 14.57 | 0.291 | 12.40 | 9.64 | 0.18 | 14 | 4 |
| strict_reference | P8 | Ring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P8 | Watch | 136 | 30 | 5 | 22.1 | 16.7 | 24.43 | 25.03 | 0.331 | 24.43 | 9.80 | 0.16 | 5 | 0 |
| strict_reference | P9 | Earring | 136 | 119 | 117 | 87.5 | 98.3 | 7.57 | 9.73 | 0.821 | -1.49 | 9.86 | 0.09 | 37 | 80 |
| strict_reference | P9 | Ring | 136 | 49 | 38 | 36.0 | 77.6 | 9.51 | 12.24 | 0.924 | 7.57 | 9.86 | 0.14 | 35 | 3 |
| strict_reference | P9 | Watch | 136 | 69 | 9 | 50.7 | 13.0 | 13.66 | 17.79 | 0.892 | 12.31 | 9.86 | 0.14 | 9 | 0 |
| strict_reference | P10 | Earring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P10 | Ring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P10 | Watch | 66 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P11 | Earring | 37 | 17 | 15 | 45.9 | 88.2 | 14.91 | 18.90 | 0.577 | -14.91 | 9.94 | 0.15 | 5 | 10 |
| strict_reference | P11 | Ring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| strict_reference | P11 | Watch | 37 | 10 | 4 | 27.0 | 40.0 | 16.56 | 19.47 | -0.144 | 16.56 | 9.86 | 0.11 | 4 | 0 |
| strict_reference | P12 | Earring | 78 | 64 | 58 | 82.1 | 90.6 | 7.63 | 9.87 | 0.546 | 2.62 | 9.89 | 0.12 | 45 | 13 |
| strict_reference | P12 | Ring | 78 | 5 | 0 | 6.4 | 0.0 |  |  |  |  | 9.81 | 0.16 | 0 | 0 |
| strict_reference | P12 | Watch | 78 | 45 | 23 | 57.7 | 51.1 | 12.49 | 14.87 | 0.185 | 11.45 | 9.84 | 0.13 | 23 | 0 |
| strict_reference | P13 | Earring | 42 | 3 | 3 | 7.1 | 100.0 | 18.85 | 19.01 | 0.949 | -18.85 | 9.97 | 0.17 | 1 | 2 |
| strict_reference | P13 | Ring | 42 | 1 | 1 | 2.4 | 100.0 | 3.51 | 3.51 |  | -3.51 | 9.91 | 0.19 | 0 | 1 |
| strict_reference | P13 | Watch | 42 | 4 | 0 | 9.5 | 0.0 |  |  |  |  | 9.82 | 0.10 | 0 | 0 |
| strict_reference | P15 | Earring | 100 | 81 | 70 | 81.0 | 86.4 | 12.94 | 16.33 | 0.043 | 3.30 | 9.82 | 0.11 | 32 | 38 |
| strict_reference | P15 | Ring | 100 | 18 | 16 | 18.0 | 88.9 | 10.55 | 13.25 | 0.305 | 7.72 | 9.83 | 0.14 | 13 | 3 |
| strict_reference | P15 | Watch | 100 | 52 | 10 | 52.0 | 19.2 | 32.72 | 34.18 | -0.123 | 32.72 | 9.82 | 0.12 | 10 | 0 |
| strict_reference | P18 | Earring | 187 | 143 | 136 | 76.5 | 95.1 | 11.78 | 15.24 | 0.584 | -8.03 | 9.88 | 0.11 | 8 | 128 |
| strict_reference | P18 | Ring | 187 | 15 | 10 | 8.0 | 66.7 | 9.27 | 10.32 | 0.602 | -2.33 | 9.86 | 0.16 | 9 | 1 |
| strict_reference | P18 | Watch | 187 | 101 | 15 | 54.0 | 14.9 | 22.82 | 29.40 | 0.371 | 20.87 | 9.83 | 0.12 | 13 | 2 |
| strict_reference | P19 | Earring | 87 | 84 | 81 | 96.6 | 96.4 | 22.29 | 24.37 | 0.403 | -17.88 | 9.85 | 0.09 | 19 | 62 |
| strict_reference | P19 | Ring | 87 | 19 | 8 | 21.8 | 42.1 | 15.82 | 18.27 | -0.375 | -1.71 | 9.74 | 0.15 | 8 | 0 |
| strict_reference | P19 | Watch | 87 | 65 | 2 | 74.7 | 3.1 | 15.77 | 15.89 |  | 15.77 | 9.86 | 0.14 | 2 | 0 |
| strict_reference | P20 | Earring | 45 | 26 | 26 | 57.8 | 100.0 | 14.02 | 16.52 | 0.672 | -13.31 | 9.67 | 0.15 | 19 | 7 |
| strict_reference | P20 | Ring | 45 | 2 | 1 | 4.4 | 50.0 | 20.92 | 20.92 |  | -20.92 | 9.85 | 0.11 | 0 | 1 |
| strict_reference | P20 | Watch | 45 | 12 | 0 | 26.7 | 0.0 |  |  |  |  | 9.81 | 0.16 | 0 | 0 |
| training_v1_stride30 | P1 | Earring | 1087 | 223 | 187 | 20.5 | 83.9 | 10.33 | 12.30 | -0.061 | -3.95 | 1.60 | 0.15 | 42 | 145 |
| training_v1_stride30 | P1 | Ring | 1087 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P1 | Watch | 1087 | 171 | 13 | 15.7 | 7.6 | 21.26 | 21.80 | -0.017 | 21.26 | 2.33 | 0.15 | 0 | 13 |
| training_v1_stride30 | P3 | Earring | 3589 | 1158 | 1153 | 32.3 | 99.6 | 20.32 | 23.08 | 0.039 | -19.09 | 9.82 | 0.12 | 396 | 757 |
| training_v1_stride30 | P3 | Ring | 3589 | 206 | 108 | 5.7 | 52.4 | 14.18 | 16.32 | 0.704 | -10.17 | 9.81 | 0.15 | 102 | 6 |
| training_v1_stride30 | P3 | Watch | 3589 | 429 | 112 | 12.0 | 26.1 | 10.38 | 12.12 | -0.010 | 9.34 | 9.86 | 0.15 | 73 | 39 |
| training_v1_stride30 | P4 | Earring | 1297 | 528 | 449 | 40.7 | 85.0 | 9.50 | 14.70 | 0.591 | -2.90 | 9.80 | 0.10 | 242 | 207 |
| training_v1_stride30 | P4 | Ring | 1297 | 113 | 79 | 8.7 | 69.9 | 7.21 | 8.95 | 0.733 | 2.81 | 9.79 | 0.10 | 79 | 0 |
| training_v1_stride30 | P4 | Watch | 1297 | 530 | 113 | 40.9 | 21.3 | 16.27 | 18.65 | 0.566 | 15.82 | 9.83 | 0.11 | 101 | 12 |
| training_v1_stride30 | P5 | Earring | 144 | 102 | 53 | 70.8 | 52.0 | 21.05 | 23.40 | -0.101 | -20.95 | 9.93 | 0.13 | 5 | 48 |
| training_v1_stride30 | P5 | Ring | 144 | 32 | 10 | 22.2 | 31.2 | 17.10 | 18.16 | -0.203 | -17.10 | 9.91 | 0.17 | 6 | 4 |
| training_v1_stride30 | P5 | Watch | 144 | 41 | 1 | 28.5 | 2.4 | 2.79 | 2.79 |  | 2.79 | 9.89 | 0.13 | 0 | 1 |
| training_v1_stride30 | P6 | Earring | 1132 | 1071 | 1013 | 94.6 | 94.6 | 14.15 | 17.97 | 0.125 | -4.06 | 9.69 | 0.13 | 14 | 999 |
| training_v1_stride30 | P6 | Ring | 1132 | 681 | 463 | 60.2 | 68.0 | 12.19 | 14.88 | 0.426 | 9.63 | 9.80 | 0.13 | 391 | 72 |
| training_v1_stride30 | P6 | Watch | 1132 | 287 | 150 | 25.4 | 52.3 | 27.90 | 29.08 | 0.470 | 27.90 | 9.78 | 0.15 | 150 | 0 |
| training_v1_stride30 | P7 | Earring | 610 | 593 | 587 | 97.2 | 99.0 | 4.93 | 7.09 | 0.847 | 2.82 | 9.91 | 0.12 | 71 | 516 |
| training_v1_stride30 | P7 | Ring | 610 | 20 | 14 | 3.3 | 70.0 | 17.84 | 20.62 | 0.401 | 17.39 | 9.88 | 0.13 | 14 | 0 |
| training_v1_stride30 | P7 | Watch | 610 | 197 | 66 | 32.3 | 33.5 | 19.37 | 20.09 | 0.174 | 19.37 | 9.72 | 0.18 | 66 | 0 |
| training_v1_stride30 | P8 | Earring | 1381 | 276 | 189 | 20.0 | 68.5 | 13.24 | 15.88 | 0.112 | 13.19 | 9.65 | 0.18 | 154 | 35 |
| training_v1_stride30 | P8 | Ring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P8 | Watch | 1381 | 309 | 58 | 22.4 | 18.8 | 21.62 | 22.96 | -0.018 | 21.62 | 9.81 | 0.16 | 58 | 0 |
| training_v1_stride30 | P9 | Earring | 1383 | 1232 | 1200 | 89.1 | 97.4 | 7.33 | 9.48 | 0.848 | -1.65 | 9.86 | 0.09 | 374 | 826 |
| training_v1_stride30 | P9 | Ring | 1383 | 460 | 349 | 33.3 | 75.9 | 9.61 | 12.53 | 0.912 | 7.35 | 9.86 | 0.14 | 319 | 30 |
| training_v1_stride30 | P9 | Watch | 1383 | 703 | 87 | 50.8 | 12.4 | 13.01 | 16.10 | 0.866 | 11.88 | 9.86 | 0.14 | 84 | 3 |
| training_v1_stride30 | P10 | Earring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P10 | Ring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P10 | Watch | 645 | 4 | 0 | 0.6 | 0.0 |  |  |  |  | 9.81 | 0.18 | 0 | 0 |
| training_v1_stride30 | P11 | Earring | 350 | 158 | 139 | 45.1 | 88.0 | 16.15 | 19.95 | 0.397 | -13.82 | 9.94 | 0.16 | 45 | 94 |
| training_v1_stride30 | P11 | Ring | 350 | 0 | 0 | 0.0 |  |  |  |  |  |  |  | 0 | 0 |
| training_v1_stride30 | P11 | Watch | 350 | 92 | 46 | 26.3 | 50.0 | 17.14 | 18.83 | -0.287 | 17.14 | 9.86 | 0.09 | 46 | 0 |
| training_v1_stride30 | P12 | Earring | 805 | 630 | 590 | 78.3 | 93.7 | 7.43 | 9.82 | 0.488 | 2.49 | 9.89 | 0.11 | 479 | 111 |
| training_v1_stride30 | P12 | Ring | 805 | 48 | 6 | 6.0 | 12.5 | 8.26 | 8.83 | -0.143 | 6.98 | 9.82 | 0.17 | 6 | 0 |
| training_v1_stride30 | P12 | Watch | 805 | 442 | 221 | 54.9 | 50.0 | 13.78 | 16.21 | 0.278 | 12.42 | 9.84 | 0.12 | 221 | 0 |
| training_v1_stride30 | P13 | Earring | 431 | 41 | 39 | 9.5 | 95.1 | 19.86 | 20.09 | 0.610 | -19.86 | 9.97 | 0.17 | 28 | 11 |
| training_v1_stride30 | P13 | Ring | 431 | 2 | 2 | 0.5 | 100.0 | 9.52 | 11.25 |  | -9.52 | 9.95 | 0.20 | 1 | 1 |
| training_v1_stride30 | P13 | Watch | 431 | 62 | 3 | 14.4 | 4.8 | 2.27 | 2.38 | 0.973 | -0.21 | 9.85 | 0.14 | 3 | 0 |
| training_v1_stride30 | P15 | Earring | 1009 | 796 | 675 | 78.9 | 84.8 | 12.07 | 15.38 | 0.026 | 2.36 | 9.82 | 0.11 | 309 | 366 |
| training_v1_stride30 | P15 | Ring | 1009 | 176 | 151 | 17.4 | 85.8 | 12.23 | 15.05 | 0.179 | 8.42 | 9.84 | 0.14 | 143 | 8 |
| training_v1_stride30 | P15 | Watch | 1009 | 525 | 113 | 52.0 | 21.5 | 29.11 | 31.36 | -0.018 | 29.11 | 9.82 | 0.12 | 113 | 0 |
| training_v1_stride30 | P18 | Earring | 1849 | 1397 | 1324 | 75.6 | 94.8 | 11.81 | 15.09 | 0.546 | -7.46 | 9.88 | 0.11 | 52 | 1272 |
| training_v1_stride30 | P18 | Ring | 1849 | 136 | 114 | 7.4 | 83.8 | 11.03 | 15.60 | 0.230 | -0.57 | 9.86 | 0.15 | 100 | 14 |
| training_v1_stride30 | P18 | Watch | 1849 | 1006 | 152 | 54.4 | 15.1 | 24.92 | 31.67 | 0.282 | 23.16 | 9.83 | 0.12 | 139 | 13 |
| training_v1_stride30 | P19 | Earring | 904 | 867 | 831 | 95.9 | 95.8 | 22.27 | 24.50 | 0.425 | -18.21 | 9.85 | 0.09 | 200 | 631 |
| training_v1_stride30 | P19 | Ring | 904 | 204 | 99 | 22.6 | 48.5 | 15.56 | 18.31 | -0.157 | -1.27 | 9.73 | 0.14 | 98 | 1 |
| training_v1_stride30 | P19 | Watch | 904 | 658 | 19 | 72.8 | 2.9 | 15.23 | 16.23 | 0.286 | 15.23 | 9.86 | 0.14 | 19 | 0 |
| training_v1_stride30 | P20 | Earring | 448 | 229 | 229 | 51.1 | 100.0 | 13.93 | 16.44 | 0.632 | -12.94 | 9.68 | 0.15 | 174 | 55 |
| training_v1_stride30 | P20 | Ring | 448 | 14 | 11 | 3.1 | 78.6 | 13.46 | 15.32 | 0.012 | -12.25 | 9.85 | 0.12 | 8 | 3 |
| training_v1_stride30 | P20 | Watch | 448 | 111 | 0 | 24.8 | 0.0 |  |  |  |  | 9.81 | 0.16 | 0 | 0 |

## 每个参与者 x 每个设备结果（SDNN）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 20 | 16 | 18.7 | 80.0 | 4.99 | 6.24 | 0.755 | 1.08 |
| strict_reference | P1 | Ring | 107 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P1 | Watch | 107 | 16 | 2 | 15.0 | 12.5 | 19.18 | 20.15 |  | 19.18 |
| strict_reference | P3 | Earring | 361 | 118 | 118 | 32.7 | 100.0 | 6.46 | 7.75 | 0.903 | -4.95 |
| strict_reference | P3 | Ring | 361 | 20 | 9 | 5.5 | 45.0 | 7.28 | 11.15 | 0.765 | 3.31 |
| strict_reference | P3 | Watch | 361 | 42 | 11 | 11.6 | 26.2 | 9.18 | 10.41 | 0.580 | 7.84 |
| strict_reference | P4 | Earring | 129 | 54 | 44 | 41.9 | 81.5 | 5.68 | 10.02 | 0.769 | -1.50 |
| strict_reference | P4 | Ring | 129 | 10 | 8 | 7.8 | 80.0 | 12.16 | 18.39 | -0.127 | -5.46 |
| strict_reference | P4 | Watch | 129 | 50 | 11 | 38.8 | 22.0 | 17.12 | 19.58 | -0.189 | 8.68 |
| strict_reference | P5 | Earring | 14 | 10 | 5 | 71.4 | 50.0 | 11.05 | 13.79 | 0.687 | -0.57 |
| strict_reference | P5 | Ring | 14 | 4 | 1 | 28.6 | 25.0 | 1.00 | 1.00 |  | -1.00 |
| strict_reference | P5 | Watch | 14 | 4 | 0 | 28.6 | 0.0 |  |  |  |  |
| strict_reference | P6 | Earring | 119 | 112 | 107 | 94.1 | 95.5 | 5.44 | 7.98 | 0.788 | 0.10 |
| strict_reference | P6 | Ring | 119 | 74 | 48 | 62.2 | 64.9 | 9.90 | 13.50 | 0.565 | 9.03 |
| strict_reference | P6 | Watch | 119 | 29 | 17 | 24.4 | 58.6 | 19.87 | 21.85 | 0.695 | 19.87 |
| strict_reference | P7 | Earring | 62 | 61 | 60 | 98.4 | 98.4 | 2.94 | 3.98 | 0.973 | 2.64 |
| strict_reference | P7 | Ring | 62 | 2 | 1 | 3.2 | 50.0 | 11.53 | 11.53 |  | 11.53 |
| strict_reference | P7 | Watch | 62 | 20 | 7 | 32.3 | 35.0 | 12.12 | 13.10 | 0.394 | 12.12 |
| strict_reference | P8 | Earring | 136 | 28 | 18 | 20.6 | 64.3 | 5.17 | 6.69 | 0.849 | 5.15 |
| strict_reference | P8 | Ring | 136 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P8 | Watch | 136 | 30 | 5 | 22.1 | 16.7 | 21.64 | 23.28 | 0.364 | 21.64 |
| strict_reference | P9 | Earring | 136 | 119 | 117 | 87.5 | 98.3 | 3.49 | 6.61 | 0.940 | 1.33 |
| strict_reference | P9 | Ring | 136 | 49 | 38 | 36.0 | 77.6 | 7.82 | 10.80 | 0.941 | 7.09 |
| strict_reference | P9 | Watch | 136 | 69 | 9 | 50.7 | 13.0 | 9.96 | 13.85 | 0.660 | 6.01 |
| strict_reference | P10 | Earring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P10 | Ring | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P10 | Watch | 66 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P11 | Earring | 37 | 17 | 15 | 45.9 | 88.2 | 7.57 | 9.34 | 0.825 | -0.24 |
| strict_reference | P11 | Ring | 37 | 0 | 0 | 0.0 |  |  |  |  |  |
| strict_reference | P11 | Watch | 37 | 10 | 4 | 27.0 | 40.0 | 8.63 | 9.85 | 0.843 | 8.63 |
| strict_reference | P12 | Earring | 78 | 64 | 58 | 82.1 | 90.6 | 5.34 | 8.05 | 0.955 | 3.64 |
| strict_reference | P12 | Ring | 78 | 5 | 0 | 6.4 | 0.0 |  |  |  |  |
| strict_reference | P12 | Watch | 78 | 45 | 23 | 57.7 | 51.1 | 8.49 | 11.47 | 0.683 | 5.22 |
| strict_reference | P13 | Earring | 42 | 3 | 3 | 7.1 | 100.0 | 4.68 | 4.95 | 0.776 | -4.68 |
| strict_reference | P13 | Ring | 42 | 1 | 1 | 2.4 | 100.0 | 21.51 | 21.51 |  | 21.51 |
| strict_reference | P13 | Watch | 42 | 4 | 0 | 9.5 | 0.0 |  |  |  |  |
| strict_reference | P15 | Earring | 100 | 81 | 70 | 81.0 | 86.4 | 7.55 | 10.10 | 0.726 | 3.83 |
| strict_reference | P15 | Ring | 100 | 18 | 16 | 18.0 | 88.9 | 10.61 | 13.74 | 0.732 | 9.68 |
| strict_reference | P15 | Watch | 100 | 52 | 10 | 52.0 | 19.2 | 18.98 | 20.90 | 0.487 | 18.98 |
| strict_reference | P18 | Earring | 187 | 143 | 136 | 76.5 | 95.1 | 5.42 | 8.15 | 0.937 | -1.07 |
| strict_reference | P18 | Ring | 187 | 15 | 10 | 8.0 | 66.7 | 3.97 | 4.88 | 0.973 | 0.15 |
| strict_reference | P18 | Watch | 187 | 101 | 15 | 54.0 | 14.9 | 31.44 | 44.96 | 0.246 | 29.83 |
| strict_reference | P19 | Earring | 87 | 84 | 81 | 96.6 | 96.4 | 10.16 | 11.60 | 0.766 | -3.91 |
| strict_reference | P19 | Ring | 87 | 19 | 8 | 21.8 | 42.1 | 12.92 | 14.96 | 0.028 | 7.39 |
| strict_reference | P19 | Watch | 87 | 65 | 2 | 74.7 | 3.1 | 19.87 | 20.03 |  | 19.87 |
| strict_reference | P20 | Earring | 45 | 26 | 26 | 57.8 | 100.0 | 4.72 | 5.96 | 0.923 | -2.07 |
| strict_reference | P20 | Ring | 45 | 2 | 1 | 4.4 | 50.0 | 6.36 | 6.36 |  | -6.36 |
| strict_reference | P20 | Watch | 45 | 12 | 0 | 26.7 | 0.0 |  |  |  |  |
| training_v1_stride30 | P1 | Earring | 1087 | 223 | 187 | 20.5 | 83.9 | 4.39 | 5.49 | 0.821 | 0.68 |
| training_v1_stride30 | P1 | Ring | 1087 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P1 | Watch | 1087 | 171 | 13 | 15.7 | 7.6 | 24.26 | 25.40 | -0.380 | 24.26 |
| training_v1_stride30 | P3 | Earring | 3589 | 1158 | 1153 | 32.3 | 99.6 | 6.47 | 7.73 | 0.909 | -5.12 |
| training_v1_stride30 | P3 | Ring | 3589 | 206 | 108 | 5.7 | 52.4 | 7.23 | 11.46 | 0.718 | 3.89 |
| training_v1_stride30 | P3 | Watch | 3589 | 429 | 112 | 12.0 | 26.1 | 10.76 | 12.26 | 0.442 | 9.17 |
| training_v1_stride30 | P4 | Earring | 1297 | 528 | 449 | 40.7 | 85.0 | 5.35 | 9.94 | 0.867 | -1.23 |
| training_v1_stride30 | P4 | Ring | 1297 | 113 | 79 | 8.7 | 69.9 | 11.11 | 18.15 | 0.090 | -2.20 |
| training_v1_stride30 | P4 | Watch | 1297 | 530 | 113 | 40.9 | 21.3 | 17.19 | 21.55 | 0.508 | 11.48 |
| training_v1_stride30 | P5 | Earring | 144 | 102 | 53 | 70.8 | 52.0 | 7.33 | 8.86 | 0.521 | -4.90 |
| training_v1_stride30 | P5 | Ring | 144 | 32 | 10 | 22.2 | 31.2 | 4.20 | 5.93 | 0.132 | -2.35 |
| training_v1_stride30 | P5 | Watch | 144 | 41 | 1 | 28.5 | 2.4 | 6.48 | 6.48 |  | 6.48 |
| training_v1_stride30 | P6 | Earring | 1132 | 1071 | 1013 | 94.6 | 94.6 | 5.97 | 8.94 | 0.754 | 0.63 |
| training_v1_stride30 | P6 | Ring | 1132 | 681 | 463 | 60.2 | 68.0 | 10.74 | 15.34 | 0.581 | 9.89 |
| training_v1_stride30 | P6 | Watch | 1132 | 287 | 150 | 25.4 | 52.3 | 19.05 | 21.48 | 0.616 | 19.05 |
| training_v1_stride30 | P7 | Earring | 610 | 593 | 587 | 97.2 | 99.0 | 2.77 | 3.82 | 0.972 | 2.31 |
| training_v1_stride30 | P7 | Ring | 610 | 20 | 14 | 3.3 | 70.0 | 13.46 | 15.40 | 0.780 | 13.46 |
| training_v1_stride30 | P7 | Watch | 610 | 197 | 66 | 32.3 | 33.5 | 14.31 | 15.98 | 0.410 | 14.31 |
| training_v1_stride30 | P8 | Earring | 1381 | 276 | 189 | 20.0 | 68.5 | 5.61 | 7.63 | 0.823 | 5.60 |
| training_v1_stride30 | P8 | Ring | 1381 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P8 | Watch | 1381 | 309 | 58 | 22.4 | 18.8 | 14.92 | 18.23 | 0.291 | 14.86 |
| training_v1_stride30 | P9 | Earring | 1383 | 1232 | 1200 | 89.1 | 97.4 | 3.33 | 6.06 | 0.952 | 0.80 |
| training_v1_stride30 | P9 | Ring | 1383 | 460 | 349 | 33.3 | 75.9 | 8.33 | 12.70 | 0.897 | 7.22 |
| training_v1_stride30 | P9 | Watch | 1383 | 703 | 87 | 50.8 | 12.4 | 11.25 | 14.94 | 0.673 | 6.22 |
| training_v1_stride30 | P10 | Earring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P10 | Ring | 645 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P10 | Watch | 645 | 4 | 0 | 0.6 | 0.0 |  |  |  |  |
| training_v1_stride30 | P11 | Earring | 350 | 158 | 139 | 45.1 | 88.0 | 7.98 | 9.93 | 0.794 | -1.10 |
| training_v1_stride30 | P11 | Ring | 350 | 0 | 0 | 0.0 |  |  |  |  |  |
| training_v1_stride30 | P11 | Watch | 350 | 92 | 46 | 26.3 | 50.0 | 10.09 | 11.13 | 0.675 | 10.09 |
| training_v1_stride30 | P12 | Earring | 805 | 630 | 590 | 78.3 | 93.7 | 4.87 | 7.75 | 0.941 | 2.97 |
| training_v1_stride30 | P12 | Ring | 805 | 48 | 6 | 6.0 | 12.5 | 12.06 | 13.31 | 0.480 | 11.87 |
| training_v1_stride30 | P12 | Watch | 805 | 442 | 221 | 54.9 | 50.0 | 10.10 | 14.17 | 0.729 | 7.30 |
| training_v1_stride30 | P13 | Earring | 431 | 41 | 39 | 9.5 | 95.1 | 4.69 | 5.13 | 0.887 | -4.06 |
| training_v1_stride30 | P13 | Ring | 431 | 2 | 2 | 0.5 | 100.0 | 14.06 | 15.91 |  | 14.06 |
| training_v1_stride30 | P13 | Watch | 431 | 62 | 3 | 14.4 | 4.8 | 11.96 | 13.57 | -0.803 | 11.96 |
| training_v1_stride30 | P15 | Earring | 1009 | 796 | 675 | 78.9 | 84.8 | 7.13 | 9.89 | 0.751 | 4.04 |
| training_v1_stride30 | P15 | Ring | 1009 | 176 | 151 | 17.4 | 85.8 | 9.26 | 12.99 | 0.685 | 8.29 |
| training_v1_stride30 | P15 | Watch | 1009 | 525 | 113 | 52.0 | 21.5 | 19.24 | 22.17 | 0.446 | 19.24 |
| training_v1_stride30 | P18 | Earring | 1849 | 1397 | 1324 | 75.6 | 94.8 | 5.49 | 8.64 | 0.929 | -0.77 |
| training_v1_stride30 | P18 | Ring | 1849 | 136 | 114 | 7.4 | 83.8 | 10.71 | 22.31 | 0.587 | 5.28 |
| training_v1_stride30 | P18 | Watch | 1849 | 1006 | 152 | 54.4 | 15.1 | 25.56 | 33.87 | 0.407 | 24.26 |
| training_v1_stride30 | P19 | Earring | 904 | 867 | 831 | 95.9 | 95.8 | 10.32 | 12.24 | 0.735 | -4.06 |
| training_v1_stride30 | P19 | Ring | 904 | 204 | 99 | 22.6 | 48.5 | 15.63 | 19.77 | -0.002 | 10.51 |
| training_v1_stride30 | P19 | Watch | 904 | 658 | 19 | 72.8 | 2.9 | 22.27 | 25.53 | 0.407 | 22.27 |
| training_v1_stride30 | P20 | Earring | 448 | 229 | 229 | 51.1 | 100.0 | 4.75 | 6.54 | 0.885 | -1.96 |
| training_v1_stride30 | P20 | Ring | 448 | 14 | 11 | 3.1 | 78.6 | 7.21 | 9.66 | 0.651 | 1.24 |
| training_v1_stride30 | P20 | Watch | 448 | 111 | 0 | 24.8 | 0.0 |  |  |  |  |

## 与 full / <0.1 / <0.2 的比较

当前报告的 motion threshold 是 `<0.2`。这里把它和 full report、严格 no-motion `<0.1`、以及已计算的 `<0.2` 结果做 sensitivity comparison。

### 有效配对行统计

| Comparison | Metric | Paired MAE rows | Candidate lower MAE | Paired R rows | Candidate higher R | Coverage rows | Candidate higher coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <0.2 vs full | RMSSD | 79 | 37 | 71 | 33 | 96 | 70 |
| <0.2 vs full | SDNN | 79 | 39 | 71 | 26 | 96 | 70 |
| <0.2 vs <0.1 | RMSSD | 59 | 19 | 50 | 29 | 96 | 11 |
| <0.2 vs <0.1 | SDNN | 59 | 20 | 50 | 29 | 96 | 11 |

### Threshold 总览（RMSSD）

| Threshold | Dataset | Subset windows | Valid preds | Coverage % | Mean RMSSD MAE | Median RMSSD R |
| --- | --- | --- | --- | --- | --- | --- |
| <0.1 | strict_reference | 567 | 479 | 84.5 | 14.48 | 0.383 |
| <0.2 | strict_reference | 1708 | 1131 | 66.2 | 15.07 | 0.384 |
| <0.1 | training_v1_stride30 | 5737 | 4771 | 83.2 | 15.51 | 0.336 |
| <0.2 | training_v1_stride30 | 16960 | 11218 | 66.1 | 14.33 | 0.282 |

### Dataset x Device 汇总比较（RMSSD, <0.2 vs full）

| Dataset | Device | MAE full | MAE <0.2 | R full | R <0.2 | Coverage full | Coverage <0.2 | Subset % <0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 14.99 | 13.20 | 0.362 | 0.546 | 80.9 | 93.0 | 55.1 |
| strict_reference | Ring | 11.84 | 12.73 | 0.403 | 0.602 | 32.5 | 64.4 | 12.8 |
| strict_reference | Watch | 17.25 | 19.55 | 0.421 | 0.265 | 10.6 | 21.1 | 32.2 |
| training_v1_stride30 | Earring | 15.13 | 13.62 | 0.394 | 0.425 | 80.9 | 93.1 | 54.5 |
| training_v1_stride30 | Ring | 11.92 | 12.35 | 0.429 | 0.230 | 33.1 | 67.2 | 12.3 |
| training_v1_stride30 | Watch | 16.18 | 16.79 | 0.282 | 0.278 | 10.7 | 20.7 | 32.6 |

### Dataset x Device 汇总比较（SDNN, <0.2 vs full）

| Dataset | Device | MAE full | MAE <0.2 | R full | R <0.2 | Coverage full | Coverage <0.2 | Subset % <0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 6.42 | 6.04 | 0.866 | 0.825 | 80.9 | 93.0 | 55.1 |
| strict_reference | Ring | 9.87 | 9.55 | 0.636 | 0.732 | 32.5 | 64.4 | 12.8 |
| strict_reference | Watch | 15.25 | 16.37 | 0.572 | 0.534 | 10.6 | 21.1 | 32.2 |
| training_v1_stride30 | Earring | 6.26 | 5.76 | 0.892 | 0.867 | 80.9 | 93.1 | 54.5 |
| training_v1_stride30 | Ring | 10.46 | 10.33 | 0.675 | 0.587 | 33.1 | 67.2 | 12.3 |
| training_v1_stride30 | Watch | 15.49 | 15.53 | 0.495 | 0.442 | 10.7 | 20.7 | 32.6 |

解释：threshold 越宽，subset windows 通常越多；但它逐渐从 strict no-motion 变成 low-motion / low-to-moderate-motion sensitivity analysis。Coverage 的分母是 threshold 子集内部窗口数，不能直接等同于完整报告的 overall coverage。

## 重要解析

- 这个报告回答的是“在数据集内低运动窗口中，当前 PPG heuristic baseline 表现如何”。
- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。
- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。
- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。
- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。
- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。

## 输出文件

- Markdown report: `current_best_baseline_motion_lt_0p2_strict_vs_training_stride30_by_participant_device.md`
- Machine-readable table: `current_best_baseline_motion_lt_0p2_strict_vs_training_stride30_by_participant_device.csv`

Generated in 0.0 seconds.

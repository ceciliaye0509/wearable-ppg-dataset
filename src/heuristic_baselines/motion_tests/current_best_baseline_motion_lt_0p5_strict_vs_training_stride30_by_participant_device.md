# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30

本报告只在指定 motion threshold 子集上重新计算当前最佳 raw-aligned baseline。

## Motion Threshold 定义

- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < 0.5`。
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
| `strict_reference` | 5118 | 3542 | non-overlapping 5 min windows, filtered to per-device no-motion windows |
| `training_v1_stride30` | 51192 | 35242 | 30 s stride rolling windows, filtered to per-device no-motion windows |

## Dataset x Device 汇总（RMSSD）

| Dataset | Device | Participants | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | Mean participant RMSSD MAE | Median participant R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 16 | 1706 | 1440 | 1199 | 84.4 | 83.3 | 15.50 | 0.363 |
| strict_reference | Ring | 16 | 1706 | 833 | 394 | 48.8 | 47.3 | 11.87 | 0.419 |
| strict_reference | Watch | 16 | 1706 | 1269 | 147 | 74.4 | 11.6 | 17.33 | 0.309 |
| training_v1_stride30 | Earring | 16 | 17064 | 14411 | 12014 | 84.5 | 83.4 | 15.63 | 0.393 |
| training_v1_stride30 | Ring | 16 | 17064 | 8292 | 4040 | 48.6 | 48.7 | 12.02 | 0.428 |
| training_v1_stride30 | Watch | 16 | 17064 | 12539 | 1474 | 73.5 | 11.8 | 16.31 | 0.276 |

## 每个参与者 x 每个设备结果（RMSSD）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms | Raw accel no-motion | Motion accel no-motion | Green selected | IR selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 83 | 48 | 77.6 | 57.8 | 11.57 | 13.13 | 0.050 | -5.88 | 1.50 | 0.28 | 22 | 26 |
| strict_reference | P1 | Ring | 107 | 14 | 10 | 13.1 | 71.4 | 14.79 | 17.95 | -0.092 | -4.77 | 9.98 | 0.32 | 8 | 2 |
| strict_reference | P1 | Watch | 107 | 58 | 2 | 54.2 | 3.4 | 22.58 | 22.82 |  | 22.58 | 2.25 | 0.28 | 0 | 2 |
| strict_reference | P3 | Earring | 361 | 198 | 166 | 54.8 | 83.8 | 20.19 | 23.08 | 0.061 | -18.59 | 9.76 | 0.19 | 76 | 90 |
| strict_reference | P3 | Ring | 361 | 105 | 37 | 29.1 | 35.2 | 12.48 | 14.19 | 0.535 | -8.50 | 9.90 | 0.31 | 34 | 3 |
| strict_reference | P3 | Watch | 361 | 175 | 23 | 48.5 | 13.1 | 9.49 | 11.55 | -0.247 | 6.86 | 9.86 | 0.27 | 18 | 5 |
| strict_reference | P4 | Earring | 129 | 123 | 86 | 95.3 | 69.9 | 14.18 | 19.21 | 0.326 | -7.58 | 9.78 | 0.21 | 42 | 44 |
| strict_reference | P4 | Ring | 129 | 73 | 27 | 56.6 | 37.0 | 6.57 | 8.14 | 0.816 | 0.12 | 9.82 | 0.30 | 24 | 3 |
| strict_reference | P4 | Watch | 129 | 91 | 14 | 70.5 | 15.4 | 13.99 | 17.21 | 0.571 | 13.99 | 9.81 | 0.20 | 13 | 1 |
| strict_reference | P5 | Earring | 14 | 14 | 9 | 100.0 | 64.3 | 19.10 | 21.74 | 0.552 | -18.84 | 10.02 | 0.22 | 1 | 8 |
| strict_reference | P5 | Ring | 14 | 14 | 1 | 100.0 | 7.1 | 13.83 | 13.83 |  | -13.83 | 9.95 | 0.27 | 1 | 0 |
| strict_reference | P5 | Watch | 14 | 14 | 1 | 100.0 | 7.1 | 8.21 | 8.21 |  | 8.21 | 10.05 | 0.27 | 0 | 1 |
| strict_reference | P6 | Earring | 119 | 117 | 112 | 98.3 | 95.7 | 13.44 | 17.08 | 0.167 | -4.58 | 9.69 | 0.13 | 3 | 109 |
| strict_reference | P6 | Ring | 119 | 113 | 60 | 95.0 | 53.1 | 13.32 | 16.40 | 0.344 | 11.08 | 9.82 | 0.18 | 50 | 10 |
| strict_reference | P6 | Watch | 119 | 105 | 28 | 88.2 | 26.7 | 27.51 | 28.54 | 0.708 | 27.51 | 9.94 | 0.30 | 28 | 0 |
| strict_reference | P7 | Earring | 62 | 62 | 61 | 100.0 | 98.4 | 5.16 | 7.30 | 0.843 | 2.89 | 9.91 | 0.12 | 6 | 55 |
| strict_reference | P7 | Ring | 62 | 47 | 26 | 75.8 | 55.3 | 15.67 | 19.86 | 0.428 | 15.67 | 9.99 | 0.34 | 26 | 0 |
| strict_reference | P7 | Watch | 62 | 61 | 7 | 98.4 | 11.5 | 19.39 | 20.17 | 0.198 | 19.39 | 9.74 | 0.22 | 7 | 0 |
| strict_reference | P8 | Earring | 136 | 122 | 68 | 89.7 | 55.7 | 15.87 | 21.34 | 0.088 | 13.90 | 9.60 | 0.27 | 32 | 36 |
| strict_reference | P8 | Ring | 136 | 20 | 16 | 14.7 | 80.0 | 14.58 | 16.32 | 0.510 | 14.54 | 10.04 | 0.43 | 16 | 0 |
| strict_reference | P8 | Watch | 136 | 102 | 6 | 75.0 | 5.9 | 23.04 | 23.78 | 0.577 | 23.04 | 9.87 | 0.28 | 6 | 0 |
| strict_reference | P9 | Earring | 136 | 133 | 129 | 97.8 | 97.0 | 7.60 | 9.70 | 0.818 | -1.61 | 9.86 | 0.10 | 46 | 83 |
| strict_reference | P9 | Ring | 136 | 111 | 71 | 81.6 | 64.0 | 8.47 | 11.11 | 0.910 | 6.82 | 9.90 | 0.23 | 68 | 3 |
| strict_reference | P9 | Watch | 136 | 125 | 9 | 91.9 | 7.2 | 13.66 | 17.79 | 0.892 | 12.31 | 9.89 | 0.21 | 9 | 0 |
| strict_reference | P10 | Earring | 66 | 52 | 52 | 78.8 | 100.0 | 19.42 | 20.57 | 0.052 | -19.25 | 10.15 | 0.37 | 1 | 51 |
| strict_reference | P10 | Ring | 66 | 15 | 9 | 22.7 | 60.0 | 9.27 | 12.80 | 0.419 | 1.69 | 9.90 | 0.41 | 9 | 0 |
| strict_reference | P10 | Watch | 66 | 47 | 0 | 71.2 | 0.0 |  |  |  |  | 9.98 | 0.37 | 0 | 0 |
| strict_reference | P11 | Earring | 37 | 30 | 23 | 81.1 | 76.7 | 30.00 | 38.27 | -0.201 | -30.00 | 9.95 | 0.23 | 6 | 17 |
| strict_reference | P11 | Ring | 37 | 15 | 10 | 40.5 | 66.7 | 12.71 | 18.30 | 0.141 | -11.79 | 10.10 | 0.41 | 10 | 0 |
| strict_reference | P11 | Watch | 37 | 25 | 4 | 67.6 | 16.0 | 16.56 | 19.47 | -0.144 | 16.56 | 9.89 | 0.27 | 4 | 0 |
| strict_reference | P12 | Earring | 78 | 73 | 64 | 93.6 | 87.7 | 7.91 | 10.14 | 0.531 | 1.38 | 9.89 | 0.14 | 51 | 13 |
| strict_reference | P12 | Ring | 78 | 61 | 16 | 78.2 | 26.2 | 11.61 | 14.72 | 0.531 | 10.00 | 9.95 | 0.34 | 15 | 1 |
| strict_reference | P12 | Watch | 78 | 65 | 25 | 83.3 | 38.5 | 12.91 | 15.18 | 0.248 | 11.96 | 9.86 | 0.18 | 25 | 0 |
| strict_reference | P13 | Earring | 42 | 23 | 22 | 54.8 | 95.7 | 22.20 | 23.21 | 0.781 | -22.20 | 10.02 | 0.27 | 17 | 5 |
| strict_reference | P13 | Ring | 42 | 11 | 4 | 26.2 | 36.4 | 7.86 | 11.58 | 0.410 | -5.12 | 9.91 | 0.37 | 3 | 1 |
| strict_reference | P13 | Watch | 42 | 31 | 0 | 73.8 | 0.0 |  |  |  |  | 9.89 | 0.30 | 0 | 0 |
| strict_reference | P15 | Earring | 100 | 100 | 85 | 100.0 | 85.0 | 12.40 | 15.71 | 0.066 | 2.51 | 9.82 | 0.14 | 41 | 44 |
| strict_reference | P15 | Ring | 100 | 50 | 30 | 50.0 | 60.0 | 13.27 | 15.86 | 0.113 | 9.92 | 9.93 | 0.28 | 26 | 4 |
| strict_reference | P15 | Watch | 100 | 93 | 10 | 93.0 | 10.8 | 32.72 | 34.18 | -0.123 | 32.72 | 9.83 | 0.20 | 10 | 0 |
| strict_reference | P18 | Earring | 187 | 182 | 151 | 97.3 | 83.0 | 12.14 | 15.75 | 0.560 | -8.68 | 9.89 | 0.15 | 11 | 140 |
| strict_reference | P18 | Ring | 187 | 92 | 56 | 49.2 | 60.9 | 13.54 | 15.91 | 0.263 | 7.17 | 9.95 | 0.30 | 54 | 2 |
| strict_reference | P18 | Watch | 187 | 162 | 15 | 86.6 | 9.3 | 22.82 | 29.40 | 0.371 | 20.87 | 9.84 | 0.20 | 13 | 2 |
| strict_reference | P19 | Earring | 87 | 86 | 83 | 98.9 | 96.5 | 22.23 | 24.32 | 0.400 | -17.93 | 9.85 | 0.09 | 20 | 63 |
| strict_reference | P19 | Ring | 87 | 64 | 12 | 73.6 | 18.8 | 11.29 | 14.99 | -0.169 | -0.40 | 9.79 | 0.28 | 12 | 0 |
| strict_reference | P19 | Watch | 87 | 84 | 2 | 96.6 | 2.4 | 15.77 | 15.89 |  | 15.77 | 9.86 | 0.17 | 2 | 0 |
| strict_reference | P20 | Earring | 45 | 42 | 40 | 93.3 | 95.2 | 14.56 | 16.90 | 0.474 | -12.81 | 9.67 | 0.22 | 27 | 13 |
| strict_reference | P20 | Ring | 45 | 28 | 9 | 62.2 | 32.1 | 10.63 | 12.09 | 0.456 | -5.89 | 9.94 | 0.33 | 8 | 1 |
| strict_reference | P20 | Watch | 45 | 31 | 1 | 68.9 | 3.2 | 3.97 | 3.97 |  | -3.97 | 9.86 | 0.25 | 1 | 0 |
| training_v1_stride30 | P1 | Earring | 1087 | 823 | 535 | 75.7 | 65.0 | 12.22 | 14.50 | -0.049 | -6.29 | 1.50 | 0.27 | 260 | 275 |
| training_v1_stride30 | P1 | Ring | 1087 | 139 | 93 | 12.8 | 66.9 | 15.54 | 20.14 | -0.304 | -2.43 | 9.96 | 0.33 | 69 | 24 |
| training_v1_stride30 | P1 | Watch | 1087 | 600 | 13 | 55.2 | 2.2 | 21.26 | 21.80 | -0.017 | 21.26 | 2.26 | 0.27 | 0 | 13 |
| training_v1_stride30 | P3 | Earring | 3589 | 1962 | 1642 | 54.7 | 83.7 | 20.28 | 23.17 | 0.065 | -18.90 | 9.76 | 0.19 | 744 | 898 |
| training_v1_stride30 | P3 | Ring | 3589 | 1048 | 375 | 29.2 | 35.8 | 11.83 | 14.08 | 0.536 | -8.66 | 9.90 | 0.31 | 349 | 26 |
| training_v1_stride30 | P3 | Watch | 3589 | 1720 | 243 | 47.9 | 14.1 | 9.47 | 11.63 | -0.150 | 7.04 | 9.86 | 0.27 | 204 | 39 |
| training_v1_stride30 | P4 | Earring | 1297 | 1255 | 909 | 96.8 | 72.4 | 13.92 | 18.78 | 0.359 | -6.43 | 9.78 | 0.21 | 463 | 446 |
| training_v1_stride30 | P4 | Ring | 1297 | 713 | 253 | 55.0 | 35.5 | 6.85 | 8.75 | 0.776 | 1.61 | 9.81 | 0.30 | 230 | 23 |
| training_v1_stride30 | P4 | Watch | 1297 | 909 | 140 | 70.1 | 15.4 | 16.03 | 18.70 | 0.631 | 15.58 | 9.81 | 0.20 | 128 | 12 |
| training_v1_stride30 | P5 | Earring | 144 | 144 | 94 | 100.0 | 65.3 | 20.20 | 22.17 | 0.575 | -20.14 | 10.02 | 0.22 | 5 | 89 |
| training_v1_stride30 | P5 | Ring | 144 | 144 | 20 | 100.0 | 13.9 | 11.17 | 13.63 | 0.581 | -11.17 | 9.95 | 0.27 | 9 | 11 |
| training_v1_stride30 | P5 | Watch | 144 | 144 | 13 | 100.0 | 9.0 | 12.59 | 15.05 | 0.079 | 9.51 | 10.05 | 0.27 | 7 | 6 |
| training_v1_stride30 | P6 | Earring | 1132 | 1118 | 1057 | 98.8 | 94.5 | 13.79 | 17.66 | 0.132 | -4.05 | 9.69 | 0.13 | 28 | 1029 |
| training_v1_stride30 | P6 | Ring | 1132 | 1070 | 622 | 94.5 | 58.1 | 13.12 | 16.05 | 0.404 | 11.01 | 9.82 | 0.18 | 542 | 80 |
| training_v1_stride30 | P6 | Watch | 1132 | 989 | 245 | 87.4 | 24.8 | 26.95 | 28.12 | 0.662 | 26.95 | 9.93 | 0.30 | 244 | 1 |
| training_v1_stride30 | P7 | Earring | 610 | 610 | 604 | 100.0 | 99.0 | 4.98 | 7.16 | 0.839 | 2.58 | 9.91 | 0.12 | 80 | 524 |
| training_v1_stride30 | P7 | Ring | 610 | 439 | 249 | 72.0 | 56.7 | 16.59 | 20.58 | 0.452 | 16.49 | 9.99 | 0.35 | 249 | 0 |
| training_v1_stride30 | P7 | Watch | 610 | 592 | 73 | 97.0 | 12.3 | 19.43 | 20.13 | 0.225 | 19.43 | 9.74 | 0.22 | 73 | 0 |
| training_v1_stride30 | P8 | Earring | 1381 | 1238 | 685 | 89.6 | 55.3 | 16.59 | 22.58 | 0.119 | 15.01 | 9.60 | 0.28 | 316 | 369 |
| training_v1_stride30 | P8 | Ring | 1381 | 206 | 154 | 14.9 | 74.8 | 15.04 | 16.95 | 0.332 | 15.01 | 10.05 | 0.44 | 146 | 8 |
| training_v1_stride30 | P8 | Watch | 1381 | 1018 | 62 | 73.7 | 6.1 | 21.64 | 22.92 | 0.048 | 21.64 | 9.86 | 0.28 | 62 | 0 |
| training_v1_stride30 | P9 | Earring | 1383 | 1359 | 1299 | 98.3 | 95.6 | 7.38 | 9.48 | 0.846 | -1.91 | 9.86 | 0.11 | 450 | 849 |
| training_v1_stride30 | P9 | Ring | 1383 | 1154 | 732 | 83.4 | 63.4 | 8.91 | 11.95 | 0.889 | 7.15 | 9.90 | 0.23 | 696 | 36 |
| training_v1_stride30 | P9 | Watch | 1383 | 1264 | 90 | 91.4 | 7.1 | 13.39 | 16.46 | 0.868 | 12.30 | 9.89 | 0.20 | 87 | 3 |
| training_v1_stride30 | P10 | Earring | 645 | 507 | 506 | 78.6 | 99.8 | 18.97 | 20.21 | 0.072 | -18.74 | 10.15 | 0.37 | 9 | 497 |
| training_v1_stride30 | P10 | Ring | 645 | 162 | 100 | 25.1 | 61.7 | 9.53 | 12.60 | 0.325 | 3.60 | 9.90 | 0.43 | 100 | 0 |
| training_v1_stride30 | P10 | Watch | 645 | 448 | 0 | 69.5 | 0.0 |  |  |  |  | 9.97 | 0.36 | 0 | 0 |
| training_v1_stride30 | P11 | Earring | 350 | 283 | 219 | 80.9 | 77.4 | 30.21 | 38.00 | -0.276 | -28.72 | 9.95 | 0.24 | 62 | 157 |
| training_v1_stride30 | P11 | Ring | 350 | 114 | 94 | 32.6 | 82.5 | 12.56 | 17.71 | 0.186 | -12.01 | 10.10 | 0.38 | 94 | 0 |
| training_v1_stride30 | P11 | Watch | 350 | 218 | 46 | 62.3 | 21.1 | 17.14 | 18.83 | -0.287 | 17.14 | 9.89 | 0.25 | 46 | 0 |
| training_v1_stride30 | P12 | Earring | 805 | 747 | 666 | 92.8 | 89.2 | 7.73 | 10.11 | 0.480 | 1.16 | 9.89 | 0.14 | 555 | 111 |
| training_v1_stride30 | P12 | Ring | 805 | 600 | 173 | 74.5 | 28.8 | 11.31 | 14.45 | 0.547 | 10.29 | 9.95 | 0.32 | 157 | 16 |
| training_v1_stride30 | P12 | Watch | 805 | 664 | 237 | 82.5 | 35.7 | 13.84 | 16.19 | 0.282 | 12.57 | 9.86 | 0.18 | 237 | 0 |
| training_v1_stride30 | P13 | Earring | 431 | 231 | 212 | 53.6 | 91.8 | 22.66 | 23.80 | 0.753 | -22.58 | 10.02 | 0.26 | 162 | 50 |
| training_v1_stride30 | P13 | Ring | 431 | 106 | 41 | 24.6 | 38.7 | 9.08 | 11.22 | 0.797 | -8.31 | 9.93 | 0.35 | 32 | 9 |
| training_v1_stride30 | P13 | Watch | 431 | 321 | 4 | 74.5 | 1.2 | 1.74 | 2.06 | 0.968 | -0.12 | 9.89 | 0.30 | 4 | 0 |
| training_v1_stride30 | P15 | Earring | 1009 | 1007 | 847 | 99.8 | 84.1 | 11.95 | 15.12 | 0.049 | 1.80 | 9.82 | 0.14 | 416 | 431 |
| training_v1_stride30 | P15 | Ring | 1009 | 530 | 295 | 52.5 | 55.7 | 14.39 | 17.21 | 0.092 | 10.86 | 9.92 | 0.28 | 278 | 17 |
| training_v1_stride30 | P15 | Watch | 1009 | 930 | 126 | 92.2 | 13.5 | 29.52 | 31.67 | -0.001 | 29.52 | 9.83 | 0.20 | 126 | 0 |
| training_v1_stride30 | P18 | Earring | 1849 | 1811 | 1493 | 97.9 | 82.4 | 12.63 | 16.67 | 0.470 | -7.80 | 9.90 | 0.15 | 75 | 1418 |
| training_v1_stride30 | P18 | Ring | 1849 | 952 | 600 | 51.5 | 63.0 | 14.88 | 17.85 | 0.177 | 7.74 | 9.95 | 0.30 | 569 | 31 |
| training_v1_stride30 | P18 | Watch | 1849 | 1566 | 157 | 84.7 | 10.0 | 24.45 | 31.21 | 0.276 | 22.75 | 9.84 | 0.19 | 144 | 13 |
| training_v1_stride30 | P19 | Earring | 904 | 900 | 856 | 99.6 | 95.1 | 22.32 | 24.53 | 0.428 | -18.37 | 9.85 | 0.10 | 217 | 639 |
| training_v1_stride30 | P19 | Ring | 904 | 634 | 159 | 70.1 | 25.1 | 13.12 | 16.36 | -0.195 | 0.75 | 9.79 | 0.27 | 158 | 1 |
| training_v1_stride30 | P19 | Watch | 904 | 866 | 21 | 95.8 | 2.4 | 15.37 | 16.27 | 0.386 | 15.37 | 9.86 | 0.17 | 21 | 0 |
| training_v1_stride30 | P20 | Earring | 448 | 416 | 390 | 92.9 | 93.8 | 14.29 | 16.64 | 0.431 | -12.17 | 9.67 | 0.23 | 265 | 125 |
| training_v1_stride30 | P20 | Ring | 448 | 281 | 80 | 62.7 | 28.5 | 8.46 | 10.15 | 0.463 | -4.27 | 9.95 | 0.34 | 75 | 5 |
| training_v1_stride30 | P20 | Watch | 448 | 290 | 4 | 64.7 | 1.4 | 1.83 | 2.30 | 0.868 | -1.42 | 9.85 | 0.24 | 4 | 0 |

## 每个参与者 x 每个设备结果（SDNN）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 83 | 48 | 77.6 | 57.8 | 4.32 | 5.54 | 0.850 | -0.02 |
| strict_reference | P1 | Ring | 107 | 14 | 10 | 13.1 | 71.4 | 9.07 | 12.28 | 0.412 | 0.94 |
| strict_reference | P1 | Watch | 107 | 58 | 2 | 54.2 | 3.4 | 19.18 | 20.15 |  | 19.18 |
| strict_reference | P3 | Earring | 361 | 198 | 166 | 54.8 | 83.8 | 6.37 | 7.73 | 0.919 | -4.93 |
| strict_reference | P3 | Ring | 361 | 105 | 37 | 29.1 | 35.2 | 9.40 | 13.45 | 0.693 | 5.18 |
| strict_reference | P3 | Watch | 361 | 175 | 23 | 48.5 | 13.1 | 9.85 | 12.68 | 0.469 | 8.51 |
| strict_reference | P4 | Earring | 129 | 123 | 86 | 95.3 | 69.9 | 6.57 | 9.72 | 0.800 | -2.23 |
| strict_reference | P4 | Ring | 129 | 73 | 27 | 56.6 | 37.0 | 7.57 | 11.71 | 0.687 | 1.27 |
| strict_reference | P4 | Watch | 129 | 91 | 14 | 70.5 | 15.4 | 15.52 | 18.52 | 0.135 | 8.60 |
| strict_reference | P5 | Earring | 14 | 14 | 9 | 100.0 | 64.3 | 9.48 | 11.70 | 0.785 | -3.65 |
| strict_reference | P5 | Ring | 14 | 14 | 1 | 100.0 | 7.1 | 1.00 | 1.00 |  | -1.00 |
| strict_reference | P5 | Watch | 14 | 14 | 1 | 100.0 | 7.1 | 18.68 | 18.68 |  | 18.68 |
| strict_reference | P6 | Earring | 119 | 117 | 112 | 98.3 | 95.7 | 5.39 | 7.91 | 0.812 | 0.08 |
| strict_reference | P6 | Ring | 119 | 113 | 60 | 95.0 | 53.1 | 11.22 | 16.27 | 0.461 | 10.18 |
| strict_reference | P6 | Watch | 119 | 105 | 28 | 88.2 | 26.7 | 20.74 | 23.32 | 0.657 | 20.71 |
| strict_reference | P7 | Earring | 62 | 62 | 61 | 100.0 | 98.4 | 2.92 | 3.95 | 0.973 | 2.63 |
| strict_reference | P7 | Ring | 62 | 47 | 26 | 75.8 | 55.3 | 14.50 | 17.30 | 0.601 | 14.50 |
| strict_reference | P7 | Watch | 62 | 61 | 7 | 98.4 | 11.5 | 12.12 | 13.10 | 0.394 | 12.12 |
| strict_reference | P8 | Earring | 136 | 122 | 68 | 89.7 | 55.7 | 7.59 | 10.80 | 0.715 | 7.01 |
| strict_reference | P8 | Ring | 136 | 20 | 16 | 14.7 | 80.0 | 9.14 | 12.13 | 0.666 | 9.14 |
| strict_reference | P8 | Watch | 136 | 102 | 6 | 75.0 | 5.9 | 19.01 | 21.39 | 0.095 | 19.01 |
| strict_reference | P9 | Earring | 136 | 133 | 129 | 97.8 | 97.0 | 3.45 | 6.44 | 0.945 | 1.09 |
| strict_reference | P9 | Ring | 136 | 111 | 71 | 81.6 | 64.0 | 6.91 | 10.10 | 0.922 | 6.10 |
| strict_reference | P9 | Watch | 136 | 125 | 9 | 91.9 | 7.2 | 9.96 | 13.85 | 0.660 | 6.01 |
| strict_reference | P10 | Earring | 66 | 52 | 52 | 78.8 | 100.0 | 5.49 | 6.19 | 0.905 | -4.88 |
| strict_reference | P10 | Ring | 66 | 15 | 9 | 22.7 | 60.0 | 7.54 | 9.05 | 0.833 | 6.75 |
| strict_reference | P10 | Watch | 66 | 47 | 0 | 71.2 | 0.0 |  |  |  |  |
| strict_reference | P11 | Earring | 37 | 30 | 23 | 81.1 | 76.7 | 13.25 | 16.36 | 0.623 | -8.47 |
| strict_reference | P11 | Ring | 37 | 15 | 10 | 40.5 | 66.7 | 8.56 | 11.52 | 0.764 | 0.24 |
| strict_reference | P11 | Watch | 37 | 25 | 4 | 67.6 | 16.0 | 8.63 | 9.85 | 0.843 | 8.63 |
| strict_reference | P12 | Earring | 78 | 73 | 64 | 93.6 | 87.7 | 5.28 | 7.88 | 0.952 | 3.33 |
| strict_reference | P12 | Ring | 78 | 61 | 16 | 78.2 | 26.2 | 7.00 | 10.07 | 0.893 | 3.73 |
| strict_reference | P12 | Watch | 78 | 65 | 25 | 83.3 | 38.5 | 8.91 | 11.68 | 0.815 | 4.99 |
| strict_reference | P13 | Earring | 42 | 23 | 22 | 54.8 | 95.7 | 6.24 | 6.77 | 0.969 | -5.39 |
| strict_reference | P13 | Ring | 42 | 11 | 4 | 26.2 | 36.4 | 12.88 | 15.35 | 0.403 | 9.68 |
| strict_reference | P13 | Watch | 42 | 31 | 0 | 73.8 | 0.0 |  |  |  |  |
| strict_reference | P15 | Earring | 100 | 100 | 85 | 100.0 | 85.0 | 7.23 | 9.76 | 0.778 | 3.67 |
| strict_reference | P15 | Ring | 100 | 50 | 30 | 50.0 | 60.0 | 12.83 | 15.67 | 0.555 | 10.78 |
| strict_reference | P15 | Watch | 100 | 93 | 10 | 93.0 | 10.8 | 18.98 | 20.90 | 0.487 | 18.98 |
| strict_reference | P18 | Earring | 187 | 182 | 151 | 97.3 | 83.0 | 5.66 | 8.40 | 0.934 | -1.45 |
| strict_reference | P18 | Ring | 187 | 92 | 56 | 49.2 | 60.9 | 14.37 | 19.51 | 0.789 | 12.03 |
| strict_reference | P18 | Watch | 187 | 162 | 15 | 86.6 | 9.3 | 31.44 | 44.96 | 0.246 | 29.83 |
| strict_reference | P19 | Earring | 87 | 86 | 83 | 98.9 | 96.5 | 10.21 | 11.62 | 0.745 | -3.87 |
| strict_reference | P19 | Ring | 87 | 64 | 12 | 73.6 | 18.8 | 12.21 | 14.44 | 0.404 | 8.52 |
| strict_reference | P19 | Watch | 87 | 84 | 2 | 96.6 | 2.4 | 19.87 | 20.03 |  | 19.87 |
| strict_reference | P20 | Earring | 45 | 42 | 40 | 93.3 | 95.2 | 4.72 | 5.97 | 0.946 | -1.54 |
| strict_reference | P20 | Ring | 45 | 28 | 9 | 62.2 | 32.1 | 7.48 | 9.89 | 0.894 | 6.07 |
| strict_reference | P20 | Watch | 45 | 31 | 1 | 68.9 | 3.2 | 2.62 | 2.62 |  | 2.62 |
| training_v1_stride30 | P1 | Earring | 1087 | 823 | 535 | 75.7 | 65.0 | 4.47 | 5.77 | 0.852 | -0.11 |
| training_v1_stride30 | P1 | Ring | 1087 | 139 | 93 | 12.8 | 66.9 | 8.81 | 12.53 | 0.486 | 1.97 |
| training_v1_stride30 | P1 | Watch | 1087 | 600 | 13 | 55.2 | 2.2 | 24.26 | 25.40 | -0.380 | 24.26 |
| training_v1_stride30 | P3 | Earring | 3589 | 1962 | 1642 | 54.7 | 83.7 | 6.63 | 8.51 | 0.887 | -4.92 |
| training_v1_stride30 | P3 | Ring | 3589 | 1048 | 375 | 29.2 | 35.8 | 7.52 | 11.13 | 0.700 | 3.48 |
| training_v1_stride30 | P3 | Watch | 3589 | 1720 | 243 | 47.9 | 14.1 | 10.77 | 14.10 | 0.422 | 9.52 |
| training_v1_stride30 | P4 | Earring | 1297 | 1255 | 909 | 96.8 | 72.4 | 6.58 | 10.15 | 0.849 | -1.25 |
| training_v1_stride30 | P4 | Ring | 1297 | 713 | 253 | 55.0 | 35.5 | 8.32 | 13.76 | 0.661 | 2.76 |
| training_v1_stride30 | P4 | Watch | 1297 | 909 | 140 | 70.1 | 15.4 | 16.04 | 20.29 | 0.567 | 10.69 |
| training_v1_stride30 | P5 | Earring | 144 | 144 | 94 | 100.0 | 65.3 | 6.54 | 7.84 | 0.930 | -5.17 |
| training_v1_stride30 | P5 | Ring | 144 | 144 | 20 | 100.0 | 13.9 | 4.82 | 6.98 | 0.772 | 1.54 |
| training_v1_stride30 | P5 | Watch | 144 | 144 | 13 | 100.0 | 9.0 | 13.41 | 14.96 | 0.313 | 12.73 |
| training_v1_stride30 | P6 | Earring | 1132 | 1118 | 1057 | 98.8 | 94.5 | 5.84 | 8.80 | 0.783 | 0.64 |
| training_v1_stride30 | P6 | Ring | 1132 | 1070 | 622 | 94.5 | 58.1 | 11.52 | 16.10 | 0.638 | 10.61 |
| training_v1_stride30 | P6 | Watch | 1132 | 989 | 245 | 87.4 | 24.8 | 19.36 | 22.25 | 0.621 | 19.35 |
| training_v1_stride30 | P7 | Earring | 610 | 610 | 604 | 100.0 | 99.0 | 2.79 | 3.84 | 0.972 | 2.30 |
| training_v1_stride30 | P7 | Ring | 610 | 439 | 249 | 72.0 | 56.7 | 14.54 | 17.44 | 0.645 | 14.54 |
| training_v1_stride30 | P7 | Watch | 610 | 592 | 73 | 97.0 | 12.3 | 15.45 | 17.42 | 0.457 | 15.45 |
| training_v1_stride30 | P8 | Earring | 1381 | 1238 | 685 | 89.6 | 55.3 | 8.05 | 11.61 | 0.721 | 7.65 |
| training_v1_stride30 | P8 | Ring | 1381 | 206 | 154 | 14.9 | 74.8 | 11.76 | 14.82 | 0.540 | 11.76 |
| training_v1_stride30 | P8 | Watch | 1381 | 1018 | 62 | 73.7 | 6.1 | 14.67 | 17.94 | 0.374 | 14.61 |
| training_v1_stride30 | P9 | Earring | 1383 | 1359 | 1299 | 98.3 | 95.6 | 3.29 | 5.95 | 0.955 | 0.68 |
| training_v1_stride30 | P9 | Ring | 1383 | 1154 | 732 | 83.4 | 63.4 | 7.65 | 12.70 | 0.865 | 6.47 |
| training_v1_stride30 | P9 | Watch | 1383 | 1264 | 90 | 91.4 | 7.1 | 11.39 | 14.97 | 0.671 | 6.53 |
| training_v1_stride30 | P10 | Earring | 645 | 507 | 506 | 78.6 | 99.8 | 5.29 | 6.16 | 0.911 | -4.69 |
| training_v1_stride30 | P10 | Ring | 645 | 162 | 100 | 25.1 | 61.7 | 10.26 | 12.31 | 0.653 | 9.99 |
| training_v1_stride30 | P10 | Watch | 645 | 448 | 0 | 69.5 | 0.0 |  |  |  |  |
| training_v1_stride30 | P11 | Earring | 350 | 283 | 219 | 80.9 | 77.4 | 13.22 | 16.33 | 0.754 | -8.12 |
| training_v1_stride30 | P11 | Ring | 350 | 114 | 94 | 32.6 | 82.5 | 6.94 | 9.15 | 0.836 | -1.72 |
| training_v1_stride30 | P11 | Watch | 350 | 218 | 46 | 62.3 | 21.1 | 10.09 | 11.13 | 0.675 | 10.09 |
| training_v1_stride30 | P12 | Earring | 805 | 747 | 666 | 92.8 | 89.2 | 4.84 | 7.62 | 0.941 | 2.74 |
| training_v1_stride30 | P12 | Ring | 805 | 600 | 173 | 74.5 | 28.8 | 9.89 | 14.28 | 0.807 | 6.71 |
| training_v1_stride30 | P12 | Watch | 805 | 664 | 237 | 82.5 | 35.7 | 10.32 | 14.19 | 0.770 | 7.33 |
| training_v1_stride30 | P13 | Earring | 431 | 231 | 212 | 53.6 | 91.8 | 6.43 | 7.13 | 0.933 | -5.01 |
| training_v1_stride30 | P13 | Ring | 431 | 106 | 41 | 24.6 | 38.7 | 9.82 | 13.55 | 0.783 | 6.10 |
| training_v1_stride30 | P13 | Watch | 431 | 321 | 4 | 74.5 | 1.2 | 15.19 | 17.11 | -0.241 | 15.19 |
| training_v1_stride30 | P15 | Earring | 1009 | 1007 | 847 | 99.8 | 84.1 | 6.85 | 9.55 | 0.786 | 3.88 |
| training_v1_stride30 | P15 | Ring | 1009 | 530 | 295 | 52.5 | 55.7 | 11.82 | 15.74 | 0.566 | 10.30 |
| training_v1_stride30 | P15 | Watch | 1009 | 930 | 126 | 92.2 | 13.5 | 19.75 | 22.87 | 0.483 | 19.75 |
| training_v1_stride30 | P18 | Earring | 1849 | 1811 | 1493 | 97.9 | 82.4 | 5.87 | 9.59 | 0.915 | -0.81 |
| training_v1_stride30 | P18 | Ring | 1849 | 952 | 600 | 51.5 | 63.0 | 15.66 | 22.55 | 0.687 | 12.30 |
| training_v1_stride30 | P18 | Watch | 1849 | 1566 | 157 | 84.7 | 10.0 | 25.44 | 33.59 | 0.394 | 23.70 |
| training_v1_stride30 | P19 | Earring | 904 | 900 | 856 | 99.6 | 95.1 | 10.28 | 12.17 | 0.737 | -4.15 |
| training_v1_stride30 | P19 | Ring | 904 | 634 | 159 | 70.1 | 25.1 | 14.81 | 19.15 | 0.219 | 11.41 |
| training_v1_stride30 | P19 | Watch | 904 | 866 | 21 | 95.8 | 2.4 | 22.79 | 25.96 | 0.495 | 22.79 |
| training_v1_stride30 | P20 | Earring | 448 | 416 | 390 | 92.9 | 93.8 | 4.81 | 6.55 | 0.915 | -1.62 |
| training_v1_stride30 | P20 | Ring | 448 | 281 | 80 | 62.7 | 28.5 | 8.29 | 10.80 | 0.799 | 7.17 |
| training_v1_stride30 | P20 | Watch | 448 | 290 | 4 | 64.7 | 1.4 | 3.47 | 3.89 | 0.978 | 3.47 |

## 与 full / <0.1 / <0.2 的比较

当前报告的 motion threshold 是 `<0.5`。这里把它和 full report、严格 no-motion `<0.1`、以及已计算的 `<0.2` 结果做 sensitivity comparison。

### 有效配对行统计

| Comparison | Metric | Paired MAE rows | Candidate lower MAE | Paired R rows | Candidate higher R | Coverage rows | Candidate higher coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <0.5 vs full | RMSSD | 93 | 21 | 88 | 23 | 96 | 75 |
| <0.5 vs full | SDNN | 93 | 28 | 88 | 25 | 96 | 75 |
| <0.5 vs <0.1 | RMSSD | 59 | 20 | 50 | 30 | 96 | 11 |
| <0.5 vs <0.1 | SDNN | 59 | 17 | 50 | 33 | 96 | 11 |
| <0.5 vs <0.2 | RMSSD | 79 | 32 | 71 | 33 | 96 | 13 |
| <0.5 vs <0.2 | SDNN | 79 | 29 | 71 | 36 | 96 | 13 |

### Threshold 总览（RMSSD）

| Threshold | Dataset | Subset windows | Valid preds | Coverage % | Mean RMSSD MAE | Median RMSSD R |
| --- | --- | --- | --- | --- | --- | --- |
| <0.1 | strict_reference | 567 | 479 | 84.5 | 14.48 | 0.383 |
| <0.2 | strict_reference | 1708 | 1131 | 66.2 | 15.07 | 0.384 |
| <0.5 | strict_reference | 3542 | 1740 | 49.1 | 14.79 | 0.400 |
| <0.1 | training_v1_stride30 | 5737 | 4771 | 83.2 | 15.51 | 0.336 |
| <0.2 | training_v1_stride30 | 16960 | 11218 | 66.1 | 14.33 | 0.282 |
| <0.5 | training_v1_stride30 | 35242 | 17528 | 49.7 | 14.62 | 0.359 |

### Dataset x Device 汇总比较（RMSSD, <0.5 vs full）

| Dataset | Device | MAE full | MAE <0.5 | R full | R <0.5 | Coverage full | Coverage <0.5 | Subset % <0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 14.99 | 15.50 | 0.362 | 0.363 | 80.9 | 83.3 | 84.4 |
| strict_reference | Ring | 11.84 | 11.87 | 0.403 | 0.419 | 32.5 | 47.3 | 48.8 |
| strict_reference | Watch | 17.25 | 17.33 | 0.421 | 0.309 | 10.6 | 11.6 | 74.4 |
| training_v1_stride30 | Earring | 15.13 | 15.63 | 0.394 | 0.393 | 80.9 | 83.4 | 84.5 |
| training_v1_stride30 | Ring | 11.92 | 12.02 | 0.429 | 0.428 | 33.1 | 48.7 | 48.6 |
| training_v1_stride30 | Watch | 16.18 | 16.31 | 0.282 | 0.276 | 10.7 | 11.8 | 73.5 |

### Dataset x Device 汇总比较（SDNN, <0.5 vs full）

| Dataset | Device | MAE full | MAE <0.5 | R full | R <0.5 | Coverage full | Coverage <0.5 | Subset % <0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 6.42 | 6.51 | 0.866 | 0.877 | 80.9 | 83.3 | 84.4 |
| strict_reference | Ring | 9.87 | 9.48 | 0.636 | 0.687 | 32.5 | 47.3 | 48.8 |
| strict_reference | Watch | 15.25 | 15.39 | 0.572 | 0.478 | 10.6 | 11.6 | 74.4 |
| training_v1_stride30 | Earring | 6.26 | 6.36 | 0.892 | 0.899 | 80.9 | 83.4 | 84.5 |
| training_v1_stride30 | Ring | 10.46 | 10.15 | 0.675 | 0.674 | 33.1 | 48.7 | 48.6 |
| training_v1_stride30 | Watch | 15.49 | 15.49 | 0.495 | 0.483 | 10.7 | 11.8 | 73.5 |

解释：threshold 越宽，subset windows 通常越多；但它逐渐从 strict no-motion 变成 low-motion / low-to-moderate-motion sensitivity analysis。Coverage 的分母是 threshold 子集内部窗口数，不能直接等同于完整报告的 overall coverage。

## 重要解析

- 这个报告回答的是“在数据集内低运动窗口中，当前 PPG heuristic baseline 表现如何”。
- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。
- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。
- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。
- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。
- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。

## 输出文件

- Markdown report: `current_best_baseline_motion_lt_0p5_strict_vs_training_stride30_by_participant_device.md`
- Machine-readable table: `current_best_baseline_motion_lt_0p5_strict_vs_training_stride30_by_participant_device.csv`

Generated in 0.0 seconds.

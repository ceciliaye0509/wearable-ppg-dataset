# Current Best Heuristic Baseline: No-Motion Strict Reference vs Training Stride30

本报告只在指定 motion threshold 子集上重新计算当前最佳 raw-aligned baseline。

## Motion Threshold 定义

- 按每个 window、每个 device 独立判断：`accel_motion_mean_mag < 1.0`。
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
| `strict_reference` | 5118 | 4333 | non-overlapping 5 min windows, filtered to per-device no-motion windows |
| `training_v1_stride30` | 51192 | 43367 | 30 s stride rolling windows, filtered to per-device no-motion windows |

## Dataset x Device 汇总（RMSSD）

| Dataset | Device | Participants | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | Mean participant RMSSD MAE | Median participant R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 16 | 1706 | 1528 | 1254 | 89.6 | 82.1 | 15.22 | 0.362 |
| strict_reference | Ring | 16 | 1706 | 1325 | 487 | 77.7 | 36.8 | 11.99 | 0.403 |
| strict_reference | Watch | 16 | 1706 | 1480 | 147 | 86.8 | 9.9 | 17.33 | 0.309 |
| training_v1_stride30 | Earring | 16 | 17064 | 15318 | 12561 | 89.8 | 82.0 | 15.35 | 0.394 |
| training_v1_stride30 | Ring | 16 | 17064 | 13229 | 4956 | 77.5 | 37.5 | 12.07 | 0.429 |
| training_v1_stride30 | Watch | 16 | 17064 | 14820 | 1481 | 86.8 | 10.0 | 16.30 | 0.276 |

## 每个参与者 x 每个设备结果（RMSSD）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms | Raw accel no-motion | Motion accel no-motion | Green selected | IR selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 107 | 66 | 100.0 | 61.7 | 11.49 | 13.49 | -0.065 | -5.55 | 1.59 | 0.38 | 24 | 42 |
| strict_reference | P1 | Ring | 107 | 32 | 17 | 29.9 | 53.1 | 14.00 | 16.87 | -0.245 | -1.12 | 9.81 | 0.57 | 15 | 2 |
| strict_reference | P1 | Watch | 107 | 93 | 2 | 86.9 | 2.2 | 22.58 | 22.82 |  | 22.58 | 2.06 | 0.45 | 0 | 2 |
| strict_reference | P3 | Earring | 361 | 204 | 168 | 56.5 | 82.4 | 20.24 | 23.10 | 0.061 | -18.66 | 9.76 | 0.20 | 77 | 91 |
| strict_reference | P3 | Ring | 361 | 183 | 39 | 50.7 | 21.3 | 12.26 | 13.95 | 0.504 | -8.04 | 9.99 | 0.47 | 36 | 3 |
| strict_reference | P3 | Watch | 361 | 201 | 23 | 55.7 | 11.4 | 9.49 | 11.55 | -0.247 | 6.86 | 9.89 | 0.33 | 18 | 5 |
| strict_reference | P4 | Earring | 129 | 128 | 88 | 99.2 | 68.8 | 14.36 | 19.28 | 0.336 | -7.91 | 9.79 | 0.22 | 43 | 45 |
| strict_reference | P4 | Ring | 129 | 108 | 27 | 83.7 | 25.0 | 6.57 | 8.14 | 0.816 | 0.12 | 9.87 | 0.44 | 24 | 3 |
| strict_reference | P4 | Watch | 129 | 122 | 14 | 94.6 | 11.5 | 13.99 | 17.21 | 0.571 | 13.99 | 9.84 | 0.32 | 13 | 1 |
| strict_reference | P5 | Earring | 14 | 14 | 9 | 100.0 | 64.3 | 19.10 | 21.74 | 0.552 | -18.84 | 10.02 | 0.22 | 1 | 8 |
| strict_reference | P5 | Ring | 14 | 14 | 1 | 100.0 | 7.1 | 13.83 | 13.83 |  | -13.83 | 9.95 | 0.27 | 1 | 0 |
| strict_reference | P5 | Watch | 14 | 14 | 1 | 100.0 | 7.1 | 8.21 | 8.21 |  | 8.21 | 10.05 | 0.27 | 0 | 1 |
| strict_reference | P6 | Earring | 119 | 119 | 114 | 100.0 | 95.8 | 13.27 | 16.94 | 0.167 | -4.43 | 9.69 | 0.14 | 3 | 111 |
| strict_reference | P6 | Ring | 119 | 119 | 60 | 100.0 | 50.4 | 13.32 | 16.40 | 0.344 | 11.08 | 9.83 | 0.20 | 50 | 10 |
| strict_reference | P6 | Watch | 119 | 119 | 28 | 100.0 | 23.5 | 27.51 | 28.54 | 0.708 | 27.51 | 9.98 | 0.34 | 28 | 0 |
| strict_reference | P7 | Earring | 62 | 62 | 61 | 100.0 | 98.4 | 5.16 | 7.30 | 0.843 | 2.89 | 9.91 | 0.12 | 6 | 55 |
| strict_reference | P7 | Ring | 62 | 62 | 27 | 100.0 | 43.5 | 15.66 | 19.70 | 0.435 | 15.66 | 10.01 | 0.42 | 27 | 0 |
| strict_reference | P7 | Watch | 62 | 62 | 7 | 100.0 | 11.3 | 19.39 | 20.17 | 0.198 | 19.39 | 9.75 | 0.22 | 7 | 0 |
| strict_reference | P8 | Earring | 136 | 130 | 69 | 95.6 | 53.1 | 15.70 | 21.19 | 0.081 | 13.64 | 9.60 | 0.29 | 33 | 36 |
| strict_reference | P8 | Ring | 136 | 108 | 42 | 79.4 | 38.9 | 14.94 | 17.06 | 0.206 | 14.93 | 10.13 | 0.67 | 42 | 0 |
| strict_reference | P8 | Watch | 136 | 127 | 6 | 93.4 | 4.7 | 23.04 | 23.78 | 0.577 | 23.04 | 9.89 | 0.35 | 6 | 0 |
| strict_reference | P9 | Earring | 136 | 135 | 129 | 99.3 | 95.6 | 7.60 | 9.70 | 0.818 | -1.61 | 9.86 | 0.11 | 46 | 83 |
| strict_reference | P9 | Ring | 136 | 133 | 72 | 97.8 | 54.1 | 8.43 | 11.05 | 0.908 | 6.65 | 9.93 | 0.30 | 69 | 3 |
| strict_reference | P9 | Watch | 136 | 134 | 9 | 98.5 | 6.7 | 13.66 | 17.79 | 0.892 | 12.31 | 9.90 | 0.23 | 9 | 0 |
| strict_reference | P10 | Earring | 66 | 63 | 61 | 95.5 | 96.8 | 18.68 | 19.94 | 0.014 | -18.41 | 10.17 | 0.41 | 1 | 60 |
| strict_reference | P10 | Ring | 66 | 58 | 18 | 87.9 | 31.0 | 10.71 | 13.75 | 0.403 | 4.02 | 9.95 | 0.60 | 18 | 0 |
| strict_reference | P10 | Watch | 66 | 60 | 0 | 90.9 | 0.0 |  |  |  |  | 9.99 | 0.41 | 0 | 0 |
| strict_reference | P11 | Earring | 37 | 32 | 24 | 86.5 | 75.0 | 29.06 | 37.50 | -0.126 | -29.06 | 9.95 | 0.25 | 7 | 17 |
| strict_reference | P11 | Ring | 37 | 29 | 13 | 78.4 | 44.8 | 15.30 | 20.46 | 0.341 | -14.60 | 10.13 | 0.56 | 13 | 0 |
| strict_reference | P11 | Watch | 37 | 31 | 4 | 83.8 | 12.9 | 16.56 | 19.47 | -0.144 | 16.56 | 9.91 | 0.34 | 4 | 0 |
| strict_reference | P12 | Earring | 78 | 73 | 64 | 93.6 | 87.7 | 7.91 | 10.14 | 0.531 | 1.38 | 9.89 | 0.14 | 51 | 13 |
| strict_reference | P12 | Ring | 78 | 71 | 17 | 91.0 | 23.9 | 12.28 | 15.34 | 0.489 | 10.77 | 9.96 | 0.39 | 16 | 1 |
| strict_reference | P12 | Watch | 78 | 73 | 25 | 93.6 | 34.2 | 12.91 | 15.18 | 0.248 | 11.96 | 9.87 | 0.23 | 25 | 0 |
| strict_reference | P13 | Earring | 42 | 42 | 38 | 100.0 | 90.5 | 19.44 | 20.96 | 0.779 | -19.44 | 10.14 | 0.41 | 28 | 10 |
| strict_reference | P13 | Ring | 42 | 29 | 7 | 69.0 | 24.1 | 5.64 | 8.97 | 0.646 | -3.31 | 9.96 | 0.59 | 6 | 1 |
| strict_reference | P13 | Watch | 42 | 40 | 0 | 95.2 | 0.0 |  |  |  |  | 9.92 | 0.37 | 0 | 0 |
| strict_reference | P15 | Earring | 100 | 100 | 85 | 100.0 | 85.0 | 12.40 | 15.71 | 0.066 | 2.51 | 9.82 | 0.14 | 41 | 44 |
| strict_reference | P15 | Ring | 100 | 94 | 35 | 94.0 | 37.2 | 14.09 | 17.59 | 0.009 | 11.20 | 10.01 | 0.46 | 31 | 4 |
| strict_reference | P15 | Watch | 100 | 100 | 10 | 100.0 | 10.0 | 32.72 | 34.18 | -0.123 | 32.72 | 9.84 | 0.24 | 10 | 0 |
| strict_reference | P18 | Earring | 187 | 187 | 153 | 100.0 | 81.8 | 12.31 | 15.93 | 0.562 | -8.90 | 9.90 | 0.16 | 11 | 142 |
| strict_reference | P18 | Ring | 187 | 167 | 91 | 89.3 | 54.5 | 12.94 | 15.42 | 0.332 | 7.52 | 10.01 | 0.47 | 89 | 2 |
| strict_reference | P18 | Watch | 187 | 179 | 15 | 95.7 | 8.4 | 22.82 | 29.40 | 0.371 | 20.87 | 9.86 | 0.24 | 13 | 2 |
| strict_reference | P19 | Earring | 87 | 87 | 84 | 100.0 | 96.6 | 22.40 | 24.50 | 0.388 | -18.15 | 9.85 | 0.10 | 21 | 63 |
| strict_reference | P19 | Ring | 87 | 83 | 12 | 95.4 | 14.5 | 11.29 | 14.99 | -0.169 | -0.40 | 9.82 | 0.37 | 12 | 0 |
| strict_reference | P19 | Watch | 87 | 87 | 2 | 100.0 | 2.3 | 15.77 | 15.89 |  | 15.77 | 9.87 | 0.19 | 2 | 0 |
| strict_reference | P20 | Earring | 45 | 45 | 41 | 100.0 | 91.1 | 14.43 | 16.75 | 0.478 | -12.73 | 9.67 | 0.25 | 27 | 14 |
| strict_reference | P20 | Ring | 45 | 35 | 9 | 77.8 | 25.7 | 10.63 | 12.09 | 0.456 | -5.89 | 9.97 | 0.40 | 8 | 1 |
| strict_reference | P20 | Watch | 45 | 38 | 1 | 84.4 | 2.6 | 3.97 | 3.97 |  | -3.97 | 9.86 | 0.34 | 1 | 0 |
| training_v1_stride30 | P1 | Earring | 1087 | 1087 | 722 | 100.0 | 66.4 | 12.24 | 14.83 | -0.116 | -6.04 | 1.59 | 0.38 | 287 | 435 |
| training_v1_stride30 | P1 | Ring | 1087 | 345 | 173 | 31.7 | 50.1 | 14.92 | 19.04 | -0.294 | 2.99 | 9.82 | 0.59 | 146 | 27 |
| training_v1_stride30 | P1 | Watch | 1087 | 956 | 13 | 87.9 | 1.4 | 21.26 | 21.80 | -0.017 | 21.26 | 2.06 | 0.44 | 0 | 13 |
| training_v1_stride30 | P3 | Earring | 3589 | 2039 | 1668 | 56.8 | 81.8 | 20.41 | 23.27 | 0.061 | -19.04 | 9.76 | 0.21 | 760 | 908 |
| training_v1_stride30 | P3 | Ring | 3589 | 1813 | 389 | 50.5 | 21.5 | 11.61 | 13.89 | 0.513 | -8.40 | 10.00 | 0.48 | 363 | 26 |
| training_v1_stride30 | P3 | Watch | 3589 | 1982 | 243 | 55.2 | 12.3 | 9.47 | 11.63 | -0.150 | 7.04 | 9.89 | 0.33 | 204 | 39 |
| training_v1_stride30 | P4 | Earring | 1297 | 1285 | 921 | 99.1 | 71.7 | 14.05 | 18.86 | 0.364 | -6.66 | 9.78 | 0.22 | 470 | 451 |
| training_v1_stride30 | P4 | Ring | 1297 | 1097 | 265 | 84.6 | 24.2 | 7.00 | 8.97 | 0.767 | 2.00 | 9.88 | 0.45 | 242 | 23 |
| training_v1_stride30 | P4 | Watch | 1297 | 1236 | 143 | 95.3 | 11.6 | 15.99 | 18.62 | 0.634 | 15.55 | 9.84 | 0.33 | 131 | 12 |
| training_v1_stride30 | P5 | Earring | 144 | 144 | 94 | 100.0 | 65.3 | 20.20 | 22.17 | 0.575 | -20.14 | 10.02 | 0.22 | 5 | 89 |
| training_v1_stride30 | P5 | Ring | 144 | 144 | 20 | 100.0 | 13.9 | 11.17 | 13.63 | 0.581 | -11.17 | 9.95 | 0.27 | 9 | 11 |
| training_v1_stride30 | P5 | Watch | 144 | 144 | 13 | 100.0 | 9.0 | 12.59 | 15.05 | 0.079 | 9.51 | 10.05 | 0.27 | 7 | 6 |
| training_v1_stride30 | P6 | Earring | 1132 | 1132 | 1071 | 100.0 | 94.6 | 13.67 | 17.55 | 0.133 | -3.97 | 9.69 | 0.14 | 31 | 1040 |
| training_v1_stride30 | P6 | Ring | 1132 | 1132 | 626 | 100.0 | 55.3 | 13.13 | 16.03 | 0.406 | 11.03 | 9.83 | 0.21 | 546 | 80 |
| training_v1_stride30 | P6 | Watch | 1132 | 1127 | 249 | 99.6 | 22.1 | 26.88 | 28.04 | 0.663 | 26.88 | 9.97 | 0.33 | 248 | 1 |
| training_v1_stride30 | P7 | Earring | 610 | 610 | 604 | 100.0 | 99.0 | 4.98 | 7.16 | 0.839 | 2.58 | 9.91 | 0.12 | 80 | 524 |
| training_v1_stride30 | P7 | Ring | 610 | 601 | 272 | 98.5 | 45.3 | 16.42 | 20.26 | 0.465 | 16.32 | 10.01 | 0.43 | 272 | 0 |
| training_v1_stride30 | P7 | Watch | 610 | 610 | 73 | 100.0 | 12.0 | 19.43 | 20.13 | 0.225 | 19.43 | 9.75 | 0.23 | 73 | 0 |
| training_v1_stride30 | P8 | Earring | 1381 | 1331 | 696 | 96.4 | 52.3 | 16.42 | 22.42 | 0.108 | 14.75 | 9.60 | 0.30 | 327 | 369 |
| training_v1_stride30 | P8 | Ring | 1381 | 1113 | 387 | 80.6 | 34.8 | 15.40 | 17.45 | 0.209 | 15.37 | 10.13 | 0.66 | 379 | 8 |
| training_v1_stride30 | P8 | Watch | 1381 | 1297 | 62 | 93.9 | 4.8 | 21.64 | 22.92 | 0.048 | 21.64 | 9.89 | 0.36 | 62 | 0 |
| training_v1_stride30 | P9 | Earring | 1383 | 1378 | 1301 | 99.6 | 94.4 | 7.37 | 9.47 | 0.846 | -1.91 | 9.86 | 0.11 | 452 | 849 |
| training_v1_stride30 | P9 | Ring | 1383 | 1356 | 740 | 98.0 | 54.6 | 8.92 | 11.94 | 0.886 | 7.04 | 9.92 | 0.30 | 704 | 36 |
| training_v1_stride30 | P9 | Watch | 1383 | 1367 | 90 | 98.8 | 6.6 | 13.39 | 16.46 | 0.868 | 12.30 | 9.90 | 0.23 | 87 | 3 |
| training_v1_stride30 | P10 | Earring | 645 | 617 | 592 | 95.7 | 95.9 | 18.33 | 19.66 | 0.059 | -17.98 | 10.17 | 0.42 | 14 | 578 |
| training_v1_stride30 | P10 | Ring | 645 | 556 | 190 | 86.2 | 34.2 | 9.19 | 12.33 | 0.326 | 3.14 | 9.95 | 0.60 | 190 | 0 |
| training_v1_stride30 | P10 | Watch | 645 | 589 | 0 | 91.3 | 0.0 |  |  |  |  | 9.99 | 0.43 | 0 | 0 |
| training_v1_stride30 | P11 | Earring | 350 | 306 | 223 | 87.4 | 72.9 | 29.85 | 37.70 | -0.243 | -28.39 | 9.95 | 0.26 | 66 | 157 |
| training_v1_stride30 | P11 | Ring | 350 | 270 | 126 | 77.1 | 46.7 | 16.00 | 20.84 | 0.402 | -15.58 | 10.13 | 0.56 | 126 | 0 |
| training_v1_stride30 | P11 | Watch | 350 | 298 | 46 | 85.1 | 15.4 | 17.14 | 18.83 | -0.287 | 17.14 | 9.91 | 0.35 | 46 | 0 |
| training_v1_stride30 | P12 | Earring | 805 | 753 | 666 | 93.5 | 88.4 | 7.73 | 10.11 | 0.480 | 1.16 | 9.89 | 0.14 | 555 | 111 |
| training_v1_stride30 | P12 | Ring | 805 | 725 | 178 | 90.1 | 24.6 | 11.77 | 15.00 | 0.518 | 10.77 | 9.97 | 0.39 | 161 | 17 |
| training_v1_stride30 | P12 | Watch | 805 | 752 | 237 | 93.4 | 31.5 | 13.84 | 16.19 | 0.282 | 12.57 | 9.88 | 0.24 | 237 | 0 |
| training_v1_stride30 | P13 | Earring | 431 | 431 | 385 | 100.0 | 89.3 | 19.12 | 20.71 | 0.763 | -19.08 | 10.15 | 0.41 | 260 | 125 |
| training_v1_stride30 | P13 | Ring | 431 | 296 | 69 | 68.7 | 23.3 | 6.96 | 9.47 | 0.750 | -4.34 | 9.96 | 0.57 | 60 | 9 |
| training_v1_stride30 | P13 | Watch | 431 | 413 | 4 | 95.8 | 1.0 | 1.74 | 2.06 | 0.968 | -0.12 | 9.92 | 0.36 | 4 | 0 |
| training_v1_stride30 | P15 | Earring | 1009 | 1009 | 849 | 100.0 | 84.1 | 11.97 | 15.15 | 0.048 | 1.75 | 9.82 | 0.14 | 418 | 431 |
| training_v1_stride30 | P15 | Ring | 1009 | 939 | 343 | 93.1 | 36.5 | 14.77 | 17.88 | 0.063 | 11.72 | 10.01 | 0.45 | 323 | 20 |
| training_v1_stride30 | P15 | Watch | 1009 | 1001 | 126 | 99.2 | 12.6 | 29.52 | 31.67 | -0.001 | 29.52 | 9.84 | 0.23 | 126 | 0 |
| training_v1_stride30 | P18 | Earring | 1849 | 1845 | 1505 | 99.8 | 81.6 | 12.69 | 16.74 | 0.471 | -7.90 | 9.90 | 0.16 | 75 | 1430 |
| training_v1_stride30 | P18 | Ring | 1849 | 1656 | 934 | 89.6 | 56.4 | 14.18 | 16.89 | 0.263 | 8.77 | 10.01 | 0.47 | 903 | 31 |
| training_v1_stride30 | P18 | Watch | 1849 | 1776 | 157 | 96.1 | 8.8 | 24.45 | 31.21 | 0.276 | 22.75 | 9.86 | 0.25 | 144 | 13 |
| training_v1_stride30 | P19 | Earring | 904 | 904 | 860 | 100.0 | 95.1 | 22.38 | 24.59 | 0.424 | -18.45 | 9.85 | 0.10 | 221 | 639 |
| training_v1_stride30 | P19 | Ring | 904 | 849 | 162 | 93.9 | 19.1 | 13.01 | 16.23 | -0.191 | 0.67 | 9.81 | 0.37 | 161 | 1 |
| training_v1_stride30 | P19 | Watch | 904 | 899 | 21 | 99.4 | 2.3 | 15.37 | 16.27 | 0.386 | 15.37 | 9.87 | 0.19 | 21 | 0 |
| training_v1_stride30 | P20 | Earring | 448 | 447 | 404 | 99.8 | 90.4 | 14.23 | 16.55 | 0.431 | -12.19 | 9.67 | 0.26 | 274 | 130 |
| training_v1_stride30 | P20 | Ring | 448 | 337 | 82 | 75.2 | 24.3 | 8.65 | 10.35 | 0.452 | -3.77 | 9.97 | 0.40 | 77 | 5 |
| training_v1_stride30 | P20 | Watch | 448 | 373 | 4 | 83.3 | 1.1 | 1.83 | 2.30 | 0.868 | -1.42 | 9.86 | 0.35 | 4 | 0 |

## 每个参与者 x 每个设备结果（SDNN）

| Dataset | Participant | Device | Total windows | No-motion windows | Valid preds | No-motion % | Coverage within no-motion % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 107 | 66 | 100.0 | 61.7 | 4.37 | 5.52 | 0.881 | -0.23 |
| strict_reference | P1 | Ring | 107 | 32 | 17 | 29.9 | 53.1 | 10.76 | 13.58 | 0.248 | 5.45 |
| strict_reference | P1 | Watch | 107 | 93 | 2 | 86.9 | 2.2 | 19.18 | 20.15 |  | 19.18 |
| strict_reference | P3 | Earring | 361 | 204 | 168 | 56.5 | 82.4 | 6.37 | 7.71 | 0.919 | -4.86 |
| strict_reference | P3 | Ring | 361 | 183 | 39 | 50.7 | 21.3 | 9.11 | 13.14 | 0.699 | 4.95 |
| strict_reference | P3 | Watch | 361 | 201 | 23 | 55.7 | 11.4 | 9.85 | 12.68 | 0.469 | 8.51 |
| strict_reference | P4 | Earring | 129 | 128 | 88 | 99.2 | 68.8 | 6.49 | 9.62 | 0.817 | -2.24 |
| strict_reference | P4 | Ring | 129 | 108 | 27 | 83.7 | 25.0 | 7.57 | 11.71 | 0.687 | 1.27 |
| strict_reference | P4 | Watch | 129 | 122 | 14 | 94.6 | 11.5 | 15.52 | 18.52 | 0.135 | 8.60 |
| strict_reference | P5 | Earring | 14 | 14 | 9 | 100.0 | 64.3 | 9.48 | 11.70 | 0.785 | -3.65 |
| strict_reference | P5 | Ring | 14 | 14 | 1 | 100.0 | 7.1 | 1.00 | 1.00 |  | -1.00 |
| strict_reference | P5 | Watch | 14 | 14 | 1 | 100.0 | 7.1 | 18.68 | 18.68 |  | 18.68 |
| strict_reference | P6 | Earring | 119 | 119 | 114 | 100.0 | 95.8 | 5.32 | 7.84 | 0.819 | 0.10 |
| strict_reference | P6 | Ring | 119 | 119 | 60 | 100.0 | 50.4 | 11.22 | 16.27 | 0.461 | 10.18 |
| strict_reference | P6 | Watch | 119 | 119 | 28 | 100.0 | 23.5 | 20.74 | 23.32 | 0.657 | 20.71 |
| strict_reference | P7 | Earring | 62 | 62 | 61 | 100.0 | 98.4 | 2.92 | 3.95 | 0.973 | 2.63 |
| strict_reference | P7 | Ring | 62 | 62 | 27 | 100.0 | 43.5 | 14.27 | 17.06 | 0.606 | 14.27 |
| strict_reference | P7 | Watch | 62 | 62 | 7 | 100.0 | 11.3 | 12.12 | 13.10 | 0.394 | 12.12 |
| strict_reference | P8 | Earring | 136 | 130 | 69 | 95.6 | 53.1 | 7.50 | 10.73 | 0.712 | 6.89 |
| strict_reference | P8 | Ring | 136 | 108 | 42 | 79.4 | 38.9 | 11.87 | 16.21 | 0.491 | 11.85 |
| strict_reference | P8 | Watch | 136 | 127 | 6 | 93.4 | 4.7 | 19.01 | 21.39 | 0.095 | 19.01 |
| strict_reference | P9 | Earring | 136 | 135 | 129 | 99.3 | 95.6 | 3.45 | 6.44 | 0.945 | 1.09 |
| strict_reference | P9 | Ring | 136 | 133 | 72 | 97.8 | 54.1 | 6.90 | 10.06 | 0.923 | 6.10 |
| strict_reference | P9 | Watch | 136 | 134 | 9 | 98.5 | 6.7 | 9.96 | 13.85 | 0.660 | 6.01 |
| strict_reference | P10 | Earring | 66 | 63 | 61 | 95.5 | 96.8 | 5.11 | 5.87 | 0.939 | -4.59 |
| strict_reference | P10 | Ring | 66 | 58 | 18 | 87.9 | 31.0 | 9.52 | 12.01 | 0.636 | 8.51 |
| strict_reference | P10 | Watch | 66 | 60 | 0 | 90.9 | 0.0 |  |  |  |  |
| strict_reference | P11 | Earring | 37 | 32 | 24 | 86.5 | 75.0 | 13.20 | 16.20 | 0.628 | -7.61 |
| strict_reference | P11 | Ring | 37 | 29 | 13 | 78.4 | 44.8 | 10.22 | 12.97 | 0.645 | 0.30 |
| strict_reference | P11 | Watch | 37 | 31 | 4 | 83.8 | 12.9 | 8.63 | 9.85 | 0.843 | 8.63 |
| strict_reference | P12 | Earring | 78 | 73 | 64 | 93.6 | 87.7 | 5.28 | 7.88 | 0.952 | 3.33 |
| strict_reference | P12 | Ring | 78 | 71 | 17 | 91.0 | 23.9 | 10.70 | 19.54 | 0.714 | -0.59 |
| strict_reference | P12 | Watch | 78 | 73 | 25 | 93.6 | 34.2 | 8.91 | 11.68 | 0.815 | 4.99 |
| strict_reference | P13 | Earring | 42 | 42 | 38 | 100.0 | 90.5 | 5.16 | 6.01 | 0.968 | -4.56 |
| strict_reference | P13 | Ring | 42 | 29 | 7 | 69.0 | 24.1 | 11.51 | 13.52 | 0.591 | 9.69 |
| strict_reference | P13 | Watch | 42 | 40 | 0 | 95.2 | 0.0 |  |  |  |  |
| strict_reference | P15 | Earring | 100 | 100 | 85 | 100.0 | 85.0 | 7.23 | 9.76 | 0.778 | 3.67 |
| strict_reference | P15 | Ring | 100 | 94 | 35 | 94.0 | 37.2 | 13.10 | 16.18 | 0.529 | 11.33 |
| strict_reference | P15 | Watch | 100 | 100 | 10 | 100.0 | 10.0 | 18.98 | 20.90 | 0.487 | 18.98 |
| strict_reference | P18 | Earring | 187 | 187 | 153 | 100.0 | 81.8 | 5.82 | 8.75 | 0.929 | -1.66 |
| strict_reference | P18 | Ring | 187 | 167 | 91 | 89.3 | 54.5 | 12.61 | 17.06 | 0.827 | 10.75 |
| strict_reference | P18 | Watch | 187 | 179 | 15 | 95.7 | 8.4 | 31.44 | 44.96 | 0.246 | 29.83 |
| strict_reference | P19 | Earring | 87 | 87 | 84 | 100.0 | 96.6 | 10.25 | 11.64 | 0.740 | -3.98 |
| strict_reference | P19 | Ring | 87 | 83 | 12 | 95.4 | 14.5 | 12.21 | 14.44 | 0.404 | 8.52 |
| strict_reference | P19 | Watch | 87 | 87 | 2 | 100.0 | 2.3 | 19.87 | 20.03 |  | 19.87 |
| strict_reference | P20 | Earring | 45 | 45 | 41 | 100.0 | 91.1 | 4.83 | 6.07 | 0.940 | -1.27 |
| strict_reference | P20 | Ring | 45 | 35 | 9 | 77.8 | 25.7 | 7.48 | 9.89 | 0.894 | 6.07 |
| strict_reference | P20 | Watch | 45 | 38 | 1 | 84.4 | 2.6 | 2.62 | 2.62 |  | 2.62 |
| training_v1_stride30 | P1 | Earring | 1087 | 1087 | 722 | 100.0 | 66.4 | 4.55 | 5.87 | 0.872 | -0.24 |
| training_v1_stride30 | P1 | Ring | 1087 | 345 | 173 | 31.7 | 50.1 | 11.40 | 14.95 | 0.362 | 7.40 |
| training_v1_stride30 | P1 | Watch | 1087 | 956 | 13 | 87.9 | 1.4 | 24.26 | 25.40 | -0.380 | 24.26 |
| training_v1_stride30 | P3 | Earring | 3589 | 2039 | 1668 | 56.8 | 81.8 | 6.63 | 8.49 | 0.888 | -4.91 |
| training_v1_stride30 | P3 | Ring | 3589 | 1813 | 389 | 50.5 | 21.5 | 7.57 | 11.21 | 0.690 | 3.58 |
| training_v1_stride30 | P3 | Watch | 3589 | 1982 | 243 | 55.2 | 12.3 | 10.77 | 14.10 | 0.422 | 9.52 |
| training_v1_stride30 | P4 | Earring | 1297 | 1285 | 921 | 99.1 | 71.7 | 6.55 | 10.10 | 0.851 | -1.29 |
| training_v1_stride30 | P4 | Ring | 1297 | 1097 | 265 | 84.6 | 24.2 | 8.47 | 13.68 | 0.667 | 3.15 |
| training_v1_stride30 | P4 | Watch | 1297 | 1236 | 143 | 95.3 | 11.6 | 15.91 | 20.13 | 0.570 | 10.53 |
| training_v1_stride30 | P5 | Earring | 144 | 144 | 94 | 100.0 | 65.3 | 6.54 | 7.84 | 0.930 | -5.17 |
| training_v1_stride30 | P5 | Ring | 144 | 144 | 20 | 100.0 | 13.9 | 4.82 | 6.98 | 0.772 | 1.54 |
| training_v1_stride30 | P5 | Watch | 144 | 144 | 13 | 100.0 | 9.0 | 13.41 | 14.96 | 0.313 | 12.73 |
| training_v1_stride30 | P6 | Earring | 1132 | 1132 | 1071 | 100.0 | 94.6 | 5.79 | 8.75 | 0.790 | 0.64 |
| training_v1_stride30 | P6 | Ring | 1132 | 1132 | 626 | 100.0 | 55.3 | 11.50 | 16.07 | 0.638 | 10.60 |
| training_v1_stride30 | P6 | Watch | 1132 | 1127 | 249 | 99.6 | 22.1 | 19.64 | 22.58 | 0.641 | 19.63 |
| training_v1_stride30 | P7 | Earring | 610 | 610 | 604 | 100.0 | 99.0 | 2.79 | 3.84 | 0.972 | 2.30 |
| training_v1_stride30 | P7 | Ring | 610 | 601 | 272 | 98.5 | 45.3 | 14.26 | 17.10 | 0.652 | 14.24 |
| training_v1_stride30 | P7 | Watch | 610 | 610 | 73 | 100.0 | 12.0 | 15.45 | 17.42 | 0.457 | 15.45 |
| training_v1_stride30 | P8 | Earring | 1381 | 1331 | 696 | 96.4 | 52.3 | 7.95 | 11.52 | 0.728 | 7.52 |
| training_v1_stride30 | P8 | Ring | 1381 | 1113 | 387 | 80.6 | 34.8 | 12.95 | 16.91 | 0.545 | 12.68 |
| training_v1_stride30 | P8 | Watch | 1381 | 1297 | 62 | 93.9 | 4.8 | 14.67 | 17.94 | 0.374 | 14.61 |
| training_v1_stride30 | P9 | Earring | 1383 | 1378 | 1301 | 99.6 | 94.4 | 3.29 | 5.94 | 0.955 | 0.68 |
| training_v1_stride30 | P9 | Ring | 1383 | 1356 | 740 | 98.0 | 54.6 | 7.65 | 12.66 | 0.865 | 6.47 |
| training_v1_stride30 | P9 | Watch | 1383 | 1367 | 90 | 98.8 | 6.6 | 11.39 | 14.97 | 0.671 | 6.53 |
| training_v1_stride30 | P10 | Earring | 645 | 617 | 592 | 95.7 | 95.9 | 5.04 | 5.93 | 0.960 | -4.35 |
| training_v1_stride30 | P10 | Ring | 645 | 556 | 190 | 86.2 | 34.2 | 9.92 | 12.20 | 0.645 | 9.37 |
| training_v1_stride30 | P10 | Watch | 645 | 589 | 0 | 91.3 | 0.0 |  |  |  |  |
| training_v1_stride30 | P11 | Earring | 350 | 306 | 223 | 87.4 | 72.9 | 13.17 | 16.25 | 0.752 | -7.78 |
| training_v1_stride30 | P11 | Ring | 350 | 270 | 126 | 77.1 | 46.7 | 8.25 | 10.57 | 0.746 | -1.36 |
| training_v1_stride30 | P11 | Watch | 350 | 298 | 46 | 85.1 | 15.4 | 10.09 | 11.13 | 0.675 | 10.09 |
| training_v1_stride30 | P12 | Earring | 805 | 753 | 666 | 93.5 | 88.4 | 4.84 | 7.62 | 0.941 | 2.74 |
| training_v1_stride30 | P12 | Ring | 805 | 725 | 178 | 90.1 | 24.6 | 11.26 | 17.31 | 0.758 | 4.88 |
| training_v1_stride30 | P12 | Watch | 805 | 752 | 237 | 93.4 | 31.5 | 10.32 | 14.19 | 0.770 | 7.33 |
| training_v1_stride30 | P13 | Earring | 431 | 431 | 385 | 100.0 | 89.3 | 5.14 | 6.06 | 0.944 | -4.08 |
| training_v1_stride30 | P13 | Ring | 431 | 296 | 69 | 68.7 | 23.3 | 10.66 | 15.17 | 0.665 | 8.45 |
| training_v1_stride30 | P13 | Watch | 431 | 413 | 4 | 95.8 | 1.0 | 15.19 | 17.11 | -0.241 | 15.19 |
| training_v1_stride30 | P15 | Earring | 1009 | 1009 | 849 | 100.0 | 84.1 | 6.86 | 9.55 | 0.787 | 3.87 |
| training_v1_stride30 | P15 | Ring | 1009 | 939 | 343 | 93.1 | 36.5 | 12.40 | 16.38 | 0.565 | 11.07 |
| training_v1_stride30 | P15 | Watch | 1009 | 1001 | 126 | 99.2 | 12.6 | 19.75 | 22.87 | 0.483 | 19.75 |
| training_v1_stride30 | P18 | Earring | 1849 | 1845 | 1505 | 99.8 | 81.6 | 5.91 | 9.66 | 0.914 | -0.88 |
| training_v1_stride30 | P18 | Ring | 1849 | 1656 | 934 | 89.6 | 56.4 | 14.29 | 20.06 | 0.734 | 11.73 |
| training_v1_stride30 | P18 | Watch | 1849 | 1776 | 157 | 96.1 | 8.8 | 25.44 | 33.59 | 0.394 | 23.70 |
| training_v1_stride30 | P19 | Earring | 904 | 904 | 860 | 100.0 | 95.1 | 10.31 | 12.21 | 0.731 | -4.21 |
| training_v1_stride30 | P19 | Ring | 904 | 849 | 162 | 93.9 | 19.1 | 14.76 | 19.08 | 0.214 | 11.42 |
| training_v1_stride30 | P19 | Watch | 904 | 899 | 21 | 99.4 | 2.3 | 22.79 | 25.96 | 0.495 | 22.79 |
| training_v1_stride30 | P20 | Earring | 448 | 447 | 404 | 99.8 | 90.4 | 4.87 | 6.57 | 0.913 | -1.36 |
| training_v1_stride30 | P20 | Ring | 448 | 337 | 82 | 75.2 | 24.3 | 8.56 | 11.17 | 0.799 | 7.47 |
| training_v1_stride30 | P20 | Watch | 448 | 373 | 4 | 83.3 | 1.1 | 3.47 | 3.89 | 0.978 | 3.47 |

## 与 full / <0.1 / <0.2 的比较

当前报告的 motion threshold 是 `<1`。这里把它和 full report、严格 no-motion `<0.1`、以及已计算的 `<0.2` 结果做 sensitivity comparison。

### 有效配对行统计

| Comparison | Metric | Paired MAE rows | Candidate lower MAE | Paired R rows | Candidate higher R | Coverage rows | Candidate higher coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <1 vs full | RMSSD | 93 | 7 | 88 | 27 | 96 | 61 |
| <1 vs full | SDNN | 93 | 11 | 88 | 19 | 96 | 61 |
| <1 vs <0.1 | RMSSD | 59 | 19 | 50 | 30 | 96 | 11 |
| <1 vs <0.1 | SDNN | 59 | 17 | 50 | 33 | 96 | 11 |
| <1 vs <0.2 | RMSSD | 79 | 31 | 71 | 31 | 96 | 13 |
| <1 vs <0.2 | SDNN | 79 | 28 | 71 | 36 | 96 | 13 |

### Threshold 总览（RMSSD）

| Threshold | Dataset | Subset windows | Valid preds | Coverage % | Mean RMSSD MAE | Median RMSSD R |
| --- | --- | --- | --- | --- | --- | --- |
| <0.1 | strict_reference | 567 | 479 | 84.5 | 14.48 | 0.383 |
| <0.2 | strict_reference | 1708 | 1131 | 66.2 | 15.07 | 0.384 |
| <1 | strict_reference | 4333 | 1888 | 43.6 | 14.74 | 0.371 |
| <0.1 | training_v1_stride30 | 5737 | 4771 | 83.2 | 15.51 | 0.336 |
| <0.2 | training_v1_stride30 | 16960 | 11218 | 66.1 | 14.33 | 0.282 |
| <1 | training_v1_stride30 | 43367 | 18998 | 43.8 | 14.54 | 0.386 |

### Dataset x Device 汇总比较（RMSSD, <1 vs full）

| Dataset | Device | MAE full | MAE <1 | R full | R <1 | Coverage full | Coverage <1 | Subset % <1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 14.99 | 15.22 | 0.362 | 0.362 | 80.9 | 82.1 | 89.6 |
| strict_reference | Ring | 11.84 | 11.99 | 0.403 | 0.403 | 32.5 | 36.8 | 77.7 |
| strict_reference | Watch | 17.25 | 17.33 | 0.421 | 0.309 | 10.6 | 9.9 | 86.8 |
| training_v1_stride30 | Earring | 15.13 | 15.35 | 0.394 | 0.394 | 80.9 | 82.0 | 89.8 |
| training_v1_stride30 | Ring | 11.92 | 12.07 | 0.429 | 0.429 | 33.1 | 37.5 | 77.5 |
| training_v1_stride30 | Watch | 16.18 | 16.30 | 0.282 | 0.276 | 10.7 | 10.0 | 86.8 |

### Dataset x Device 汇总比较（SDNN, <1 vs full）

| Dataset | Device | MAE full | MAE <1 | R full | R <1 | Coverage full | Coverage <1 | Subset % <1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 6.42 | 6.42 | 0.866 | 0.900 | 80.9 | 82.1 | 89.6 |
| strict_reference | Ring | 9.87 | 10.00 | 0.636 | 0.636 | 32.5 | 36.8 | 77.7 |
| strict_reference | Watch | 15.25 | 15.39 | 0.572 | 0.478 | 10.6 | 9.9 | 86.8 |
| training_v1_stride30 | Earring | 6.26 | 6.26 | 0.892 | 0.900 | 80.9 | 82.0 | 89.8 |
| training_v1_stride30 | Ring | 10.46 | 10.54 | 0.675 | 0.666 | 33.1 | 37.5 | 77.5 |
| training_v1_stride30 | Watch | 15.49 | 15.50 | 0.495 | 0.483 | 10.7 | 10.0 | 86.8 |

解释：threshold 越宽，subset windows 通常越多；但它逐渐从 strict no-motion 变成 low-motion / low-to-moderate-motion sensitivity analysis。Coverage 的分母是 threshold 子集内部窗口数，不能直接等同于完整报告的 overall coverage。

## 重要解析

- 这个报告回答的是“在数据集内低运动窗口中，当前 PPG heuristic baseline 表现如何”。
- no-motion 是 per-device 判断，不是 per-window 全设备统一判断，因此同一个参与者的不同设备 no-motion 窗口数可能差很多。
- `No-motion %` 很低时，MAE/R 可能不稳定，因为用于评估的窗口太少。
- `Coverage within no-motion %` 表示 no-motion 子集中有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。
- `R` 至少需要 3 个有效预测点才会计算；少于 3 个时 CSV/MD 中会留空。
- stride30 数据高度重叠，适合观察开发/训练视角，不应当按独立测试样本解释。

## 输出文件

- Markdown report: `current_best_baseline_motion_lt_1p0_strict_vs_training_stride30_by_participant_device.md`
- Machine-readable table: `current_best_baseline_motion_lt_1p0_strict_vs_training_stride30_by_participant_device.csv`

Generated in 1779.3 seconds.

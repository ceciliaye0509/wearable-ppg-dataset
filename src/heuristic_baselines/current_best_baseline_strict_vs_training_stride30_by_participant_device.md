# Current Best Heuristic Baseline: Strict Reference vs Training Stride30

本报告使用当前 raw-aligned NPZ 重新计算 baseline，不依赖数据集里保存的 PPG-derived metadata。

## Baseline 定义

- 方法：`device_best_sqi_green_or_ir__hrv_from_ppg`。
- 每个 window、每个 device 独立评估 `ppg_green` 与 `ppg_ir`。
- 每一路 PPG 使用 `heuristic_baselines.algorithms.hrv.hrv_from_ppg` 计算 PRV/HRV，包含 PPG peak detection、IBI validity、IBI CV、SQI、RMSSD 上限和亚采样 peak refinement。
- 若 green 和 IR 都有效，则只用 PPG 自身信息选择 SQI 更高的一路；不跨设备 fusion，也不使用 ECG label 选通道。
- 运动解释默认使用数据集里的 `accel_motion_mean_mag`，`<0.1` 作为 no-motion 参考阈值；baseline 选择本身没有用 ECG label。

## 数据集

| Dataset | Windows | Interpretation |
|---|---:|---|
| `strict_reference` | 1706 | non-overlapping 5 min windows, primary reference-style evaluation |
| `training_v1_stride30` | 17064 | 30 s stride rolling windows, heavily overlapping development/training view |

## Dataset x Device 汇总（RMSSD）

| Dataset | Device | Participants | Windows | Valid preds | Coverage % | Mean participant RMSSD MAE | Median participant R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | Earring | 16 | 1706 | 1380 | 80.9 | 14.99 | 0.362 |
| strict_reference | Ring | 16 | 1706 | 555 | 32.5 | 11.84 | 0.403 |
| strict_reference | Watch | 16 | 1706 | 180 | 10.6 | 17.25 | 0.421 |
| training_v1_stride30 | Earring | 16 | 17064 | 13800 | 80.9 | 15.13 | 0.394 |
| training_v1_stride30 | Ring | 16 | 17064 | 5649 | 33.1 | 11.92 | 0.429 |
| training_v1_stride30 | Watch | 16 | 17064 | 1822 | 10.7 | 16.18 | 0.282 |

## 每个参与者 x 每个设备结果（RMSSD）

| Dataset | Participant | Device | Windows | Valid preds | Coverage % | MAE ms | RMSE ms | R | Bias ms | Raw mean accel | Motion mean accel | No-motion % | Green selected | IR selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 66 | 61.7 | 11.49 | 13.49 | -0.065 | -5.55 | 1.59 | 0.38 | 0.9 | 24 | 42 |
| strict_reference | P1 | Ring | 107 | 29 | 27.1 | 12.96 | 15.56 | -0.046 | -1.40 | 8.47 | 2.03 | 0.0 | 26 | 3 |
| strict_reference | P1 | Watch | 107 | 2 | 1.9 | 22.58 | 22.82 |  | 22.58 | 1.92 | 0.56 | 0.9 | 0 | 2 |
| strict_reference | P3 | Earring | 361 | 292 | 80.9 | 16.84 | 20.17 | 0.219 | -14.01 | 6.17 | 3.78 | 13.3 | 118 | 174 |
| strict_reference | P3 | Ring | 361 | 93 | 25.8 | 11.50 | 13.34 | 0.505 | -9.30 | 6.83 | 3.50 | 0.3 | 79 | 14 |
| strict_reference | P3 | Watch | 361 | 56 | 15.5 | 8.31 | 10.39 | 0.471 | 4.77 | 6.24 | 3.88 | 0.8 | 30 | 26 |
| strict_reference | P4 | Earring | 129 | 88 | 68.2 | 14.36 | 19.28 | 0.336 | -7.91 | 9.79 | 0.23 | 27.1 | 43 | 45 |
| strict_reference | P4 | Ring | 129 | 27 | 20.9 | 6.57 | 8.14 | 0.816 | 0.12 | 9.95 | 0.61 | 5.4 | 24 | 3 |
| strict_reference | P4 | Watch | 129 | 14 | 10.9 | 13.99 | 17.21 | 0.571 | 13.99 | 9.86 | 0.38 | 17.1 | 13 | 1 |
| strict_reference | P5 | Earring | 14 | 9 | 64.3 | 19.10 | 21.74 | 0.552 | -18.84 | 10.02 | 0.22 | 0.0 | 1 | 8 |
| strict_reference | P5 | Ring | 14 | 1 | 7.1 | 13.83 | 13.83 |  | -13.83 | 9.95 | 0.27 | 0.0 | 1 | 0 |
| strict_reference | P5 | Watch | 14 | 1 | 7.1 | 8.21 | 8.21 |  | 8.21 | 10.05 | 0.27 | 7.1 | 0 | 1 |
| strict_reference | P6 | Earring | 119 | 114 | 95.8 | 13.27 | 16.94 | 0.167 | -4.43 | 9.69 | 0.14 | 23.5 | 3 | 111 |
| strict_reference | P6 | Ring | 119 | 60 | 50.4 | 13.32 | 16.40 | 0.344 | 11.08 | 9.83 | 0.20 | 11.8 | 50 | 10 |
| strict_reference | P6 | Watch | 119 | 28 | 23.5 | 27.51 | 28.54 | 0.708 | 27.51 | 9.98 | 0.34 | 1.7 | 28 | 0 |
| strict_reference | P7 | Earring | 62 | 61 | 98.4 | 5.16 | 7.30 | 0.843 | 2.89 | 9.91 | 0.12 | 35.5 | 6 | 55 |
| strict_reference | P7 | Ring | 62 | 27 | 43.5 | 15.66 | 19.70 | 0.435 | 15.66 | 10.01 | 0.42 | 1.6 | 27 | 0 |
| strict_reference | P7 | Watch | 62 | 7 | 11.3 | 19.39 | 20.17 | 0.198 | 19.39 | 9.75 | 0.22 | 0.0 | 7 | 0 |
| strict_reference | P8 | Earring | 136 | 69 | 50.7 | 15.70 | 21.19 | 0.081 | 13.64 | 9.60 | 0.35 | 0.0 | 33 | 36 |
| strict_reference | P8 | Ring | 136 | 42 | 30.9 | 14.94 | 17.06 | 0.206 | 14.93 | 10.18 | 0.80 | 0.0 | 42 | 0 |
| strict_reference | P8 | Watch | 136 | 6 | 4.4 | 23.04 | 23.78 | 0.577 | 23.04 | 9.90 | 0.44 | 1.5 | 6 | 0 |
| strict_reference | P9 | Earring | 136 | 129 | 94.9 | 7.60 | 9.70 | 0.818 | -1.61 | 9.86 | 0.12 | 64.0 | 46 | 83 |
| strict_reference | P9 | Ring | 136 | 72 | 52.9 | 8.43 | 11.05 | 0.908 | 6.65 | 9.93 | 0.32 | 5.9 | 69 | 3 |
| strict_reference | P9 | Watch | 136 | 9 | 6.6 | 13.66 | 17.79 | 0.892 | 12.31 | 9.90 | 0.25 | 9.6 | 9 | 0 |
| strict_reference | P10 | Earring | 66 | 62 | 93.9 | 18.44 | 19.79 | 0.074 | -18.05 | 10.17 | 0.47 | 0.0 | 2 | 60 |
| strict_reference | P10 | Ring | 66 | 18 | 27.3 | 10.71 | 13.75 | 0.403 | 4.02 | 9.99 | 0.74 | 0.0 | 18 | 0 |
| strict_reference | P10 | Watch | 66 | 0 | 0.0 |  |  |  |  | 10.02 | 0.52 | 0.0 | 0 | 0 |
| strict_reference | P11 | Earring | 37 | 24 | 64.9 | 29.06 | 37.50 | -0.126 | -29.06 | 9.94 | 0.47 | 0.0 | 7 | 17 |
| strict_reference | P11 | Ring | 37 | 13 | 35.1 | 15.30 | 20.46 | 0.341 | -14.60 | 10.56 | 1.04 | 0.0 | 13 | 0 |
| strict_reference | P11 | Watch | 37 | 4 | 10.8 | 16.56 | 19.47 | -0.144 | 16.56 | 10.16 | 0.65 | 10.8 | 4 | 0 |
| strict_reference | P12 | Earring | 78 | 65 | 83.3 | 7.89 | 10.10 | 0.536 | 1.46 | 9.89 | 0.24 | 21.8 | 52 | 13 |
| strict_reference | P12 | Ring | 78 | 17 | 21.8 | 12.28 | 15.34 | 0.489 | 10.77 | 10.12 | 0.57 | 0.0 | 16 | 1 |
| strict_reference | P12 | Watch | 78 | 25 | 32.1 | 12.91 | 15.18 | 0.248 | 11.96 | 9.96 | 0.34 | 20.5 | 25 | 0 |
| strict_reference | P13 | Earring | 42 | 38 | 90.5 | 19.44 | 20.96 | 0.779 | -19.44 | 10.14 | 0.41 | 0.0 | 28 | 10 |
| strict_reference | P13 | Ring | 42 | 8 | 19.0 | 5.02 | 8.40 | 0.707 | -2.99 | 10.05 | 0.81 | 0.0 | 7 | 1 |
| strict_reference | P13 | Watch | 42 | 0 | 0.0 |  |  |  |  | 9.93 | 0.41 | 7.1 | 0 | 0 |
| strict_reference | P15 | Earring | 100 | 85 | 85.0 | 12.40 | 15.71 | 0.066 | 2.51 | 9.82 | 0.14 | 41.0 | 41 | 44 |
| strict_reference | P15 | Ring | 100 | 35 | 35.0 | 14.09 | 17.59 | 0.009 | 11.20 | 10.04 | 0.50 | 2.0 | 31 | 4 |
| strict_reference | P15 | Watch | 100 | 10 | 10.0 | 32.72 | 34.18 | -0.123 | 32.72 | 9.84 | 0.24 | 18.0 | 10 | 0 |
| strict_reference | P18 | Earring | 187 | 153 | 81.8 | 12.31 | 15.93 | 0.562 | -8.90 | 9.90 | 0.16 | 35.3 | 11 | 142 |
| strict_reference | P18 | Ring | 187 | 92 | 49.2 | 12.87 | 15.35 | 0.332 | 7.50 | 10.06 | 0.57 | 1.1 | 90 | 2 |
| strict_reference | P18 | Watch | 187 | 15 | 8.0 | 22.82 | 29.40 | 0.371 | 20.87 | 9.88 | 0.29 | 15.0 | 13 | 2 |
| strict_reference | P19 | Earring | 87 | 84 | 96.6 | 22.40 | 24.50 | 0.388 | -18.15 | 9.85 | 0.10 | 64.4 | 21 | 63 |
| strict_reference | P19 | Ring | 87 | 12 | 13.8 | 11.29 | 14.99 | -0.169 | -0.40 | 9.84 | 0.42 | 0.0 | 12 | 0 |
| strict_reference | P19 | Watch | 87 | 2 | 2.3 | 15.77 | 15.89 |  | 15.77 | 9.87 | 0.19 | 13.8 | 2 | 0 |
| strict_reference | P20 | Earring | 45 | 41 | 91.1 | 14.43 | 16.75 | 0.478 | -12.73 | 9.67 | 0.25 | 11.1 | 27 | 14 |
| strict_reference | P20 | Ring | 45 | 9 | 20.0 | 10.63 | 12.09 | 0.456 | -5.89 | 10.12 | 0.70 | 2.2 | 8 | 1 |
| strict_reference | P20 | Watch | 45 | 1 | 2.2 | 3.97 | 3.97 |  | -3.97 | 9.93 | 0.51 | 0.0 | 1 | 0 |
| training_v1_stride30 | P1 | Earring | 1087 | 722 | 66.4 | 12.24 | 14.83 | -0.116 | -6.04 | 1.59 | 0.38 | 2.4 | 287 | 435 |
| training_v1_stride30 | P1 | Ring | 1087 | 323 | 29.7 | 13.52 | 16.91 | -0.194 | 1.96 | 8.49 | 2.00 | 0.0 | 293 | 30 |
| training_v1_stride30 | P1 | Watch | 1087 | 13 | 1.2 | 21.26 | 21.80 | -0.017 | 21.26 | 1.93 | 0.55 | 1.5 | 0 | 13 |
| training_v1_stride30 | P3 | Earring | 3589 | 2894 | 80.6 | 16.92 | 20.18 | 0.239 | -14.19 | 6.16 | 3.79 | 13.6 | 1163 | 1731 |
| training_v1_stride30 | P3 | Ring | 3589 | 912 | 25.4 | 11.21 | 13.17 | 0.564 | -9.38 | 6.83 | 3.51 | 0.5 | 771 | 141 |
| training_v1_stride30 | P3 | Watch | 3589 | 576 | 16.0 | 8.63 | 10.81 | 0.496 | 5.18 | 6.23 | 3.89 | 0.8 | 333 | 243 |
| training_v1_stride30 | P4 | Earring | 1297 | 921 | 71.0 | 14.05 | 18.86 | 0.364 | -6.66 | 9.79 | 0.23 | 25.6 | 470 | 451 |
| training_v1_stride30 | P4 | Ring | 1297 | 265 | 20.4 | 7.00 | 8.97 | 0.767 | 2.00 | 9.95 | 0.60 | 5.6 | 242 | 23 |
| training_v1_stride30 | P4 | Watch | 1297 | 143 | 11.0 | 15.99 | 18.62 | 0.634 | 15.55 | 9.85 | 0.38 | 14.6 | 131 | 12 |
| training_v1_stride30 | P5 | Earring | 144 | 94 | 65.3 | 20.20 | 22.17 | 0.575 | -20.14 | 10.02 | 0.22 | 2.1 | 5 | 89 |
| training_v1_stride30 | P5 | Ring | 144 | 20 | 13.9 | 11.17 | 13.63 | 0.581 | -11.17 | 9.95 | 0.27 | 0.0 | 9 | 11 |
| training_v1_stride30 | P5 | Watch | 144 | 13 | 9.0 | 12.59 | 15.05 | 0.079 | 9.51 | 10.05 | 0.27 | 11.8 | 7 | 6 |
| training_v1_stride30 | P6 | Earring | 1132 | 1071 | 94.6 | 13.67 | 17.55 | 0.133 | -3.97 | 9.69 | 0.14 | 22.6 | 31 | 1040 |
| training_v1_stride30 | P6 | Ring | 1132 | 626 | 55.3 | 13.13 | 16.03 | 0.406 | 11.03 | 9.83 | 0.21 | 11.8 | 546 | 80 |
| training_v1_stride30 | P6 | Watch | 1132 | 249 | 22.0 | 26.88 | 28.04 | 0.663 | 26.88 | 9.97 | 0.34 | 1.9 | 248 | 1 |
| training_v1_stride30 | P7 | Earring | 610 | 604 | 99.0 | 4.98 | 7.16 | 0.839 | 2.58 | 9.91 | 0.12 | 30.7 | 80 | 524 |
| training_v1_stride30 | P7 | Ring | 610 | 272 | 44.6 | 16.42 | 20.26 | 0.465 | 16.32 | 10.02 | 0.44 | 1.1 | 272 | 0 |
| training_v1_stride30 | P7 | Watch | 610 | 73 | 12.0 | 19.43 | 20.13 | 0.225 | 19.43 | 9.75 | 0.23 | 0.0 | 73 | 0 |
| training_v1_stride30 | P8 | Earring | 1381 | 696 | 50.4 | 16.42 | 22.42 | 0.108 | 14.75 | 9.60 | 0.35 | 0.0 | 327 | 369 |
| training_v1_stride30 | P8 | Ring | 1381 | 387 | 28.0 | 15.40 | 17.45 | 0.209 | 15.37 | 10.19 | 0.81 | 0.0 | 379 | 8 |
| training_v1_stride30 | P8 | Watch | 1381 | 62 | 4.5 | 21.64 | 22.92 | 0.048 | 21.64 | 9.91 | 0.44 | 0.7 | 62 | 0 |
| training_v1_stride30 | P9 | Earring | 1383 | 1301 | 94.1 | 7.37 | 9.47 | 0.846 | -1.91 | 9.86 | 0.12 | 65.5 | 452 | 849 |
| training_v1_stride30 | P9 | Ring | 1383 | 740 | 53.5 | 8.92 | 11.94 | 0.886 | 7.04 | 9.93 | 0.32 | 6.1 | 704 | 36 |
| training_v1_stride30 | P9 | Watch | 1383 | 90 | 6.5 | 13.39 | 16.46 | 0.868 | 12.30 | 9.90 | 0.25 | 12.4 | 87 | 3 |
| training_v1_stride30 | P10 | Earring | 645 | 600 | 93.0 | 18.28 | 19.62 | 0.072 | -17.84 | 10.17 | 0.48 | 0.0 | 18 | 582 |
| training_v1_stride30 | P10 | Ring | 645 | 191 | 29.6 | 9.24 | 12.37 | 0.326 | 3.22 | 10.00 | 0.75 | 0.0 | 191 | 0 |
| training_v1_stride30 | P10 | Watch | 645 | 0 | 0.0 |  |  |  |  | 10.02 | 0.53 | 0.0 | 0 | 0 |
| training_v1_stride30 | P11 | Earring | 350 | 223 | 63.7 | 29.85 | 37.70 | -0.243 | -28.39 | 9.95 | 0.46 | 0.9 | 66 | 157 |
| training_v1_stride30 | P11 | Ring | 350 | 126 | 36.0 | 16.00 | 20.84 | 0.402 | -15.58 | 10.55 | 1.02 | 0.0 | 126 | 0 |
| training_v1_stride30 | P11 | Watch | 350 | 54 | 15.4 | 16.19 | 18.04 | -0.107 | 15.96 | 10.15 | 0.64 | 15.4 | 54 | 0 |
| training_v1_stride30 | P12 | Earring | 805 | 671 | 83.4 | 7.72 | 10.09 | 0.483 | 1.19 | 9.89 | 0.24 | 22.4 | 560 | 111 |
| training_v1_stride30 | P12 | Ring | 805 | 178 | 22.1 | 11.77 | 15.00 | 0.518 | 10.77 | 10.11 | 0.56 | 0.2 | 161 | 17 |
| training_v1_stride30 | P12 | Watch | 805 | 237 | 29.4 | 13.84 | 16.19 | 0.282 | 12.57 | 9.95 | 0.34 | 20.1 | 237 | 0 |
| training_v1_stride30 | P13 | Earring | 431 | 385 | 89.3 | 19.12 | 20.71 | 0.763 | -19.08 | 10.15 | 0.41 | 0.0 | 260 | 125 |
| training_v1_stride30 | P13 | Ring | 431 | 86 | 20.0 | 6.40 | 8.78 | 0.759 | -4.01 | 10.04 | 0.78 | 0.0 | 77 | 9 |
| training_v1_stride30 | P13 | Watch | 431 | 4 | 0.9 | 1.74 | 2.06 | 0.968 | -0.12 | 9.93 | 0.40 | 3.2 | 4 | 0 |
| training_v1_stride30 | P15 | Earring | 1009 | 849 | 84.1 | 11.97 | 15.15 | 0.048 | 1.75 | 9.82 | 0.14 | 42.6 | 418 | 431 |
| training_v1_stride30 | P15 | Ring | 1009 | 343 | 34.0 | 14.77 | 17.88 | 0.063 | 11.72 | 10.03 | 0.50 | 2.1 | 323 | 20 |
| training_v1_stride30 | P15 | Watch | 1009 | 126 | 12.5 | 29.52 | 31.67 | -0.001 | 29.52 | 9.84 | 0.24 | 17.9 | 126 | 0 |
| training_v1_stride30 | P18 | Earring | 1849 | 1505 | 81.4 | 12.69 | 16.74 | 0.471 | -7.90 | 9.90 | 0.17 | 35.9 | 75 | 1430 |
| training_v1_stride30 | P18 | Ring | 1849 | 936 | 50.6 | 14.17 | 16.88 | 0.264 | 8.77 | 10.06 | 0.58 | 1.0 | 905 | 31 |
| training_v1_stride30 | P18 | Watch | 1849 | 157 | 8.5 | 24.45 | 31.21 | 0.276 | 22.75 | 9.88 | 0.29 | 16.3 | 144 | 13 |
| training_v1_stride30 | P19 | Earring | 904 | 860 | 95.1 | 22.38 | 24.59 | 0.424 | -18.45 | 9.85 | 0.10 | 60.0 | 221 | 639 |
| training_v1_stride30 | P19 | Ring | 904 | 162 | 17.9 | 13.01 | 16.23 | -0.191 | 0.67 | 9.85 | 0.43 | 0.9 | 161 | 1 |
| training_v1_stride30 | P19 | Watch | 904 | 21 | 2.3 | 15.37 | 16.27 | 0.386 | 15.37 | 9.87 | 0.20 | 13.9 | 21 | 0 |
| training_v1_stride30 | P20 | Earring | 448 | 404 | 90.2 | 14.23 | 16.55 | 0.431 | -12.19 | 9.67 | 0.26 | 11.8 | 274 | 130 |
| training_v1_stride30 | P20 | Ring | 448 | 82 | 18.3 | 8.65 | 10.35 | 0.452 | -3.77 | 10.14 | 0.73 | 1.6 | 77 | 5 |
| training_v1_stride30 | P20 | Watch | 448 | 4 | 0.9 | 1.83 | 2.30 | 0.868 | -1.42 | 9.93 | 0.52 | 0.9 | 4 | 0 |

## 每个参与者 x 每个设备结果（SDNN）

| Dataset | Participant | Device | Windows | Valid preds | Coverage % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_reference | P1 | Earring | 107 | 66 | 61.7 | 4.37 | 5.52 | 0.881 | -0.23 |
| strict_reference | P1 | Ring | 107 | 29 | 27.1 | 10.51 | 13.53 | 0.481 | 6.83 |
| strict_reference | P1 | Watch | 107 | 2 | 1.9 | 19.18 | 20.15 |  | 19.18 |
| strict_reference | P3 | Earring | 361 | 292 | 80.9 | 6.46 | 8.37 | 0.851 | -2.35 |
| strict_reference | P3 | Ring | 361 | 93 | 25.8 | 7.21 | 10.96 | 0.779 | 2.02 |
| strict_reference | P3 | Watch | 361 | 56 | 15.5 | 7.77 | 11.21 | 0.693 | 3.83 |
| strict_reference | P4 | Earring | 129 | 88 | 68.2 | 6.49 | 9.62 | 0.817 | -2.24 |
| strict_reference | P4 | Ring | 129 | 27 | 20.9 | 7.57 | 11.71 | 0.687 | 1.27 |
| strict_reference | P4 | Watch | 129 | 14 | 10.9 | 15.52 | 18.52 | 0.135 | 8.60 |
| strict_reference | P5 | Earring | 14 | 9 | 64.3 | 9.48 | 11.70 | 0.785 | -3.65 |
| strict_reference | P5 | Ring | 14 | 1 | 7.1 | 1.00 | 1.00 |  | -1.00 |
| strict_reference | P5 | Watch | 14 | 1 | 7.1 | 18.68 | 18.68 |  | 18.68 |
| strict_reference | P6 | Earring | 119 | 114 | 95.8 | 5.32 | 7.84 | 0.819 | 0.10 |
| strict_reference | P6 | Ring | 119 | 60 | 50.4 | 11.22 | 16.27 | 0.461 | 10.18 |
| strict_reference | P6 | Watch | 119 | 28 | 23.5 | 20.74 | 23.32 | 0.657 | 20.71 |
| strict_reference | P7 | Earring | 62 | 61 | 98.4 | 2.92 | 3.95 | 0.973 | 2.63 |
| strict_reference | P7 | Ring | 62 | 27 | 43.5 | 14.27 | 17.06 | 0.606 | 14.27 |
| strict_reference | P7 | Watch | 62 | 7 | 11.3 | 12.12 | 13.10 | 0.394 | 12.12 |
| strict_reference | P8 | Earring | 136 | 69 | 50.7 | 7.50 | 10.73 | 0.712 | 6.89 |
| strict_reference | P8 | Ring | 136 | 42 | 30.9 | 11.87 | 16.21 | 0.491 | 11.85 |
| strict_reference | P8 | Watch | 136 | 6 | 4.4 | 19.01 | 21.39 | 0.095 | 19.01 |
| strict_reference | P9 | Earring | 136 | 129 | 94.9 | 3.45 | 6.44 | 0.945 | 1.09 |
| strict_reference | P9 | Ring | 136 | 72 | 52.9 | 6.90 | 10.06 | 0.923 | 6.10 |
| strict_reference | P9 | Watch | 136 | 9 | 6.6 | 9.96 | 13.85 | 0.660 | 6.01 |
| strict_reference | P10 | Earring | 66 | 62 | 93.9 | 5.05 | 5.83 | 0.961 | -4.49 |
| strict_reference | P10 | Ring | 66 | 18 | 27.3 | 9.52 | 12.01 | 0.636 | 8.51 |
| strict_reference | P10 | Watch | 66 | 0 | 0.0 |  |  |  |  |
| strict_reference | P11 | Earring | 37 | 24 | 64.9 | 13.20 | 16.20 | 0.628 | -7.61 |
| strict_reference | P11 | Ring | 37 | 13 | 35.1 | 10.22 | 12.97 | 0.645 | 0.30 |
| strict_reference | P11 | Watch | 37 | 4 | 10.8 | 8.63 | 9.85 | 0.843 | 8.63 |
| strict_reference | P12 | Earring | 78 | 65 | 83.3 | 5.27 | 7.84 | 0.954 | 3.36 |
| strict_reference | P12 | Ring | 78 | 17 | 21.8 | 10.70 | 19.54 | 0.714 | -0.59 |
| strict_reference | P12 | Watch | 78 | 25 | 32.1 | 8.91 | 11.68 | 0.815 | 4.99 |
| strict_reference | P13 | Earring | 42 | 38 | 90.5 | 5.16 | 6.01 | 0.968 | -4.56 |
| strict_reference | P13 | Ring | 42 | 8 | 19.0 | 11.32 | 13.13 | 0.630 | 9.72 |
| strict_reference | P13 | Watch | 42 | 0 | 0.0 |  |  |  |  |
| strict_reference | P15 | Earring | 100 | 85 | 85.0 | 7.23 | 9.76 | 0.778 | 3.67 |
| strict_reference | P15 | Ring | 100 | 35 | 35.0 | 13.10 | 16.18 | 0.529 | 11.33 |
| strict_reference | P15 | Watch | 100 | 10 | 10.0 | 18.98 | 20.90 | 0.487 | 18.98 |
| strict_reference | P18 | Earring | 187 | 153 | 81.8 | 5.82 | 8.75 | 0.929 | -1.66 |
| strict_reference | P18 | Ring | 187 | 92 | 49.2 | 12.76 | 17.19 | 0.825 | 10.92 |
| strict_reference | P18 | Watch | 187 | 15 | 8.0 | 31.44 | 44.96 | 0.246 | 29.83 |
| strict_reference | P19 | Earring | 87 | 84 | 96.6 | 10.25 | 11.64 | 0.740 | -3.98 |
| strict_reference | P19 | Ring | 87 | 12 | 13.8 | 12.21 | 14.44 | 0.404 | 8.52 |
| strict_reference | P19 | Watch | 87 | 2 | 2.3 | 19.87 | 20.03 |  | 19.87 |
| strict_reference | P20 | Earring | 45 | 41 | 91.1 | 4.83 | 6.07 | 0.940 | -1.27 |
| strict_reference | P20 | Ring | 45 | 9 | 20.0 | 7.48 | 9.89 | 0.894 | 6.07 |
| strict_reference | P20 | Watch | 45 | 1 | 2.2 | 2.62 | 2.62 |  | 2.62 |
| training_v1_stride30 | P1 | Earring | 1087 | 722 | 66.4 | 4.55 | 5.87 | 0.872 | -0.24 |
| training_v1_stride30 | P1 | Ring | 1087 | 323 | 29.7 | 11.72 | 15.09 | 0.517 | 9.08 |
| training_v1_stride30 | P1 | Watch | 1087 | 13 | 1.2 | 24.26 | 25.40 | -0.380 | 24.26 |
| training_v1_stride30 | P3 | Earring | 3589 | 2894 | 80.6 | 6.64 | 8.84 | 0.836 | -2.57 |
| training_v1_stride30 | P3 | Ring | 3589 | 912 | 25.4 | 6.55 | 9.82 | 0.815 | 1.07 |
| training_v1_stride30 | P3 | Watch | 3589 | 576 | 16.0 | 8.34 | 12.14 | 0.700 | 5.18 |
| training_v1_stride30 | P4 | Earring | 1297 | 921 | 71.0 | 6.55 | 10.10 | 0.851 | -1.29 |
| training_v1_stride30 | P4 | Ring | 1297 | 265 | 20.4 | 8.47 | 13.68 | 0.667 | 3.15 |
| training_v1_stride30 | P4 | Watch | 1297 | 143 | 11.0 | 15.91 | 20.13 | 0.570 | 10.53 |
| training_v1_stride30 | P5 | Earring | 144 | 94 | 65.3 | 6.54 | 7.84 | 0.930 | -5.17 |
| training_v1_stride30 | P5 | Ring | 144 | 20 | 13.9 | 4.82 | 6.98 | 0.772 | 1.54 |
| training_v1_stride30 | P5 | Watch | 144 | 13 | 9.0 | 13.41 | 14.96 | 0.313 | 12.73 |
| training_v1_stride30 | P6 | Earring | 1132 | 1071 | 94.6 | 5.79 | 8.75 | 0.790 | 0.64 |
| training_v1_stride30 | P6 | Ring | 1132 | 626 | 55.3 | 11.50 | 16.07 | 0.638 | 10.60 |
| training_v1_stride30 | P6 | Watch | 1132 | 249 | 22.0 | 19.64 | 22.58 | 0.641 | 19.63 |
| training_v1_stride30 | P7 | Earring | 610 | 604 | 99.0 | 2.79 | 3.84 | 0.972 | 2.30 |
| training_v1_stride30 | P7 | Ring | 610 | 272 | 44.6 | 14.26 | 17.10 | 0.652 | 14.24 |
| training_v1_stride30 | P7 | Watch | 610 | 73 | 12.0 | 15.45 | 17.42 | 0.457 | 15.45 |
| training_v1_stride30 | P8 | Earring | 1381 | 696 | 50.4 | 7.95 | 11.52 | 0.728 | 7.52 |
| training_v1_stride30 | P8 | Ring | 1381 | 387 | 28.0 | 12.95 | 16.91 | 0.545 | 12.68 |
| training_v1_stride30 | P8 | Watch | 1381 | 62 | 4.5 | 14.67 | 17.94 | 0.374 | 14.61 |
| training_v1_stride30 | P9 | Earring | 1383 | 1301 | 94.1 | 3.29 | 5.94 | 0.955 | 0.68 |
| training_v1_stride30 | P9 | Ring | 1383 | 740 | 53.5 | 7.65 | 12.66 | 0.865 | 6.47 |
| training_v1_stride30 | P9 | Watch | 1383 | 90 | 6.5 | 11.39 | 14.97 | 0.671 | 6.53 |
| training_v1_stride30 | P10 | Earring | 645 | 600 | 93.0 | 5.03 | 5.91 | 0.963 | -4.31 |
| training_v1_stride30 | P10 | Ring | 645 | 191 | 29.6 | 9.95 | 12.23 | 0.626 | 9.23 |
| training_v1_stride30 | P10 | Watch | 645 | 0 | 0.0 |  |  |  |  |
| training_v1_stride30 | P11 | Earring | 350 | 223 | 63.7 | 13.17 | 16.25 | 0.752 | -7.78 |
| training_v1_stride30 | P11 | Ring | 350 | 126 | 36.0 | 8.25 | 10.57 | 0.746 | -1.36 |
| training_v1_stride30 | P11 | Watch | 350 | 54 | 15.4 | 12.36 | 16.00 | 0.552 | 12.36 |
| training_v1_stride30 | P12 | Earring | 805 | 671 | 83.4 | 4.85 | 7.62 | 0.942 | 2.77 |
| training_v1_stride30 | P12 | Ring | 805 | 178 | 22.1 | 11.26 | 17.31 | 0.758 | 4.88 |
| training_v1_stride30 | P12 | Watch | 805 | 237 | 29.4 | 10.32 | 14.19 | 0.770 | 7.33 |
| training_v1_stride30 | P13 | Earring | 431 | 385 | 89.3 | 5.14 | 6.06 | 0.944 | -4.08 |
| training_v1_stride30 | P13 | Ring | 431 | 86 | 20.0 | 9.94 | 14.11 | 0.683 | 8.17 |
| training_v1_stride30 | P13 | Watch | 431 | 4 | 0.9 | 15.19 | 17.11 | -0.241 | 15.19 |
| training_v1_stride30 | P15 | Earring | 1009 | 849 | 84.1 | 6.86 | 9.55 | 0.787 | 3.87 |
| training_v1_stride30 | P15 | Ring | 1009 | 343 | 34.0 | 12.40 | 16.38 | 0.565 | 11.07 |
| training_v1_stride30 | P15 | Watch | 1009 | 126 | 12.5 | 19.75 | 22.87 | 0.483 | 19.75 |
| training_v1_stride30 | P18 | Earring | 1849 | 1505 | 81.4 | 5.91 | 9.66 | 0.914 | -0.88 |
| training_v1_stride30 | P18 | Ring | 1849 | 936 | 50.6 | 14.31 | 20.07 | 0.734 | 11.75 |
| training_v1_stride30 | P18 | Watch | 1849 | 157 | 8.5 | 25.44 | 33.59 | 0.394 | 23.70 |
| training_v1_stride30 | P19 | Earring | 904 | 860 | 95.1 | 10.31 | 12.21 | 0.731 | -4.21 |
| training_v1_stride30 | P19 | Ring | 904 | 162 | 17.9 | 14.76 | 19.08 | 0.214 | 11.42 |
| training_v1_stride30 | P19 | Watch | 904 | 21 | 2.3 | 22.79 | 25.96 | 0.495 | 22.79 |
| training_v1_stride30 | P20 | Earring | 448 | 404 | 90.2 | 4.87 | 6.57 | 0.913 | -1.36 |
| training_v1_stride30 | P20 | Ring | 448 | 82 | 18.3 | 8.56 | 11.17 | 0.799 | 7.47 |
| training_v1_stride30 | P20 | Watch | 448 | 4 | 0.9 | 3.47 | 3.89 | 0.978 | 3.47 |

## 重要解析

- `strict_reference` 是更适合报告 generalization / evaluation 的表，因为 5 分钟窗口不重叠。
- `training_v1_stride30` 的窗口高度重叠，同一个生理片段会出现多次；它适合训练/开发稳定性观察，不应被当作独立测试样本数。
- Coverage 表示每个 device 的窗口中，有多少窗口能从 green/IR 至少一路得到有效 PRV 估计。Coverage 低通常意味着该 device/channel 在这些窗口里的 peak detection 或 QC 不稳定。
- Bias 是 `PPG-derived - ECG-derived`。负 bias 表示 baseline 系统性低估 ECG HRV。
- `Raw mean accel` 保留原始 magnitude 便于追溯；`Motion mean accel` 和 `No-motion %` 用于解释运动状态。如果某参与者/设备 motion 高且 no-motion 比例低，PPG baseline 表现差更可能是运动伪差导致。
- Green/IR selected counts 展示 device 内部 best-SQI 选择更偏向哪一路；这有助于判断某个设备是否主要依赖某个 wavelength。

## Strict Reference: RMSSD 最好与最困难的参与者设备

### 最好 10 组（按 RMSSD MAE）

| Participant | Device | Windows | Valid preds | Coverage % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P20 | Watch | 45 | 1 | 2.2 | 3.97 | 3.97 |  | -3.97 |
| P13 | Ring | 42 | 8 | 19.0 | 5.02 | 8.40 | 0.707 | -2.99 |
| P7 | Earring | 62 | 61 | 98.4 | 5.16 | 7.30 | 0.843 | 2.89 |
| P4 | Ring | 129 | 27 | 20.9 | 6.57 | 8.14 | 0.816 | 0.12 |
| P9 | Earring | 136 | 129 | 94.9 | 7.60 | 9.70 | 0.818 | -1.61 |
| P12 | Earring | 78 | 65 | 83.3 | 7.89 | 10.10 | 0.536 | 1.46 |
| P5 | Watch | 14 | 1 | 7.1 | 8.21 | 8.21 |  | 8.21 |
| P3 | Watch | 361 | 56 | 15.5 | 8.31 | 10.39 | 0.471 | 4.77 |
| P9 | Ring | 136 | 72 | 52.9 | 8.43 | 11.05 | 0.908 | 6.65 |
| P20 | Ring | 45 | 9 | 20.0 | 10.63 | 12.09 | 0.456 | -5.89 |

### 最困难 10 组（按 RMSSD MAE）

| Participant | Device | Windows | Valid preds | Coverage % | MAE ms | RMSE ms | R | Bias ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P15 | Watch | 100 | 10 | 10.0 | 32.72 | 34.18 | -0.123 | 32.72 |
| P11 | Earring | 37 | 24 | 64.9 | 29.06 | 37.50 | -0.126 | -29.06 |
| P6 | Watch | 119 | 28 | 23.5 | 27.51 | 28.54 | 0.708 | 27.51 |
| P8 | Watch | 136 | 6 | 4.4 | 23.04 | 23.78 | 0.577 | 23.04 |
| P18 | Watch | 187 | 15 | 8.0 | 22.82 | 29.40 | 0.371 | 20.87 |
| P1 | Watch | 107 | 2 | 1.9 | 22.58 | 22.82 |  | 22.58 |
| P19 | Earring | 87 | 84 | 96.6 | 22.40 | 24.50 | 0.388 | -18.15 |
| P13 | Earring | 42 | 38 | 90.5 | 19.44 | 20.96 | 0.779 | -19.44 |
| P7 | Watch | 62 | 7 | 11.3 | 19.39 | 20.17 | 0.198 | 19.39 |
| P5 | Earring | 14 | 9 | 64.3 | 19.10 | 21.74 | 0.552 | -18.84 |

## 输出文件

- Markdown report: `current_best_baseline_strict_vs_training_stride30_by_participant_device.md`
- Machine-readable table: `current_best_baseline_strict_vs_training_stride30_by_participant_device.csv`

Generated in 2095.0 seconds.

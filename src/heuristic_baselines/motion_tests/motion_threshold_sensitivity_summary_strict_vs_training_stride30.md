# Motion threshold sensitivity report

本报告比较 `accel_motion_mean_mag` 阈值 `<0.1`、`<0.2`、`<0.5`、`<1.0` 对 current-best heuristic baseline 的影响，并保留 full set 作为参照。

## 读法

- `windows kept`：被 motion 条件保留下来的窗口数；full 表示不加 motion 条件。
- `coverage within kept %`：保留下来的窗口里，baseline 最终能输出有效 HRV 的比例。
- `coverage of all %`：相对于原始全部窗口，最终有有效 HRV 输出的比例；这个指标更能反映实际可用数据量。
- `mean MAE ms` 是 participant-device level MAE 的平均值；`median R` 是 participant-device level R 的中位数。

## 主要结论

- 在 `training_v1_stride30` 的 RMSSD 上，`<0.1` 只保留 11.21% 窗口，coverage of all 为 9.32%；`<0.2` 保留 33.13%，coverage of all 为 21.91%。这两个阈值比较适合说明“很静止时”的表现，但可用窗口偏少。
- `<0.5` 和 `<1.0` 明显提高可用窗口：training RMSSD coverage of all 分别为 34.24% 和 37.11%，更符合放宽 no-motion 条件看 HRV 结果。
- `<1.0` 在这批数据里接近弱过滤：training RMSSD 保留 84.71% 窗口，结果更像 full；`<0.5` 是更折中的 motion sensitivity 选择。
- 和 full 相比，motion filter 通常牺牲 overall coverage；是否“更好”要同时看 MAE/R 和 coverage，不能只看某一个误差指标。

## Overall: RMSSD

| motion filter | dataset | metric | windows kept | kept % | valid preds | coverage within kept % | coverage of all % | mean MAE ms | median R | MAE rows | R rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict_reference | RMSSD | 5118 | 100.00 | 2115 | 41.32 | 41.32 | 14.58 | 0.39 | 46 | 41 |
| full | training_v1_stride30 | RMSSD | 51192 | 100.00 | 21271 | 41.55 | 41.55 | 14.37 | 0.40 | 47 | 47 |
| <0.1 | strict_reference | RMSSD | 567.00 | 11.08 | 479.00 | 84.48 | 9.36 | 14.48 | 0.38 | 28 | 19 |
| <0.1 | training_v1_stride30 | RMSSD | 5737 | 11.21 | 4771 | 83.16 | 9.32 | 15.51 | 0.34 | 31 | 31 |
| <0.2 | strict_reference | RMSSD | 1708 | 33.37 | 1131 | 66.22 | 22.10 | 15.07 | 0.38 | 38 | 32 |
| <0.2 | training_v1_stride30 | RMSSD | 16960 | 33.13 | 11218 | 66.14 | 21.91 | 14.33 | 0.28 | 41 | 39 |
| <0.5 | strict_reference | RMSSD | 3542 | 69.21 | 1740 | 49.12 | 34.00 | 14.79 | 0.40 | 46 | 41 |
| <0.5 | training_v1_stride30 | RMSSD | 35242 | 68.84 | 17528 | 49.74 | 34.24 | 14.62 | 0.36 | 47 | 47 |
| <1.0 | strict_reference | RMSSD | 4333 | 84.66 | 1888 | 43.57 | 36.89 | 14.74 | 0.37 | 46 | 41 |
| <1.0 | training_v1_stride30 | RMSSD | 43367 | 84.71 | 18998 | 43.81 | 37.11 | 14.54 | 0.39 | 47 | 47 |

## Overall: SDNN

| motion filter | dataset | metric | windows kept | kept % | valid preds | coverage within kept % | coverage of all % | mean MAE ms | median R | MAE rows | R rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict_reference | SDNN | 5118 | 100.00 | 2115 | 41.32 | 41.32 | 10.31 | 0.71 | 46 | 41 |
| full | training_v1_stride30 | SDNN | 51192 | 100.00 | 21271 | 41.55 | 41.55 | 10.64 | 0.73 | 47 | 47 |
| <0.1 | strict_reference | SDNN | 567.00 | 11.08 | 479.00 | 84.48 | 9.36 | 9.57 | 0.83 | 28 | 19 |
| <0.1 | training_v1_stride30 | SDNN | 5737 | 11.21 | 4771 | 83.16 | 9.32 | 9.74 | 0.68 | 31 | 31 |
| <0.2 | strict_reference | SDNN | 1708 | 33.37 | 1131 | 66.22 | 22.10 | 10.32 | 0.76 | 38 | 32 |
| <0.2 | training_v1_stride30 | SDNN | 16960 | 33.13 | 11218 | 66.14 | 21.91 | 10.44 | 0.67 | 41 | 39 |
| <0.5 | strict_reference | SDNN | 3542 | 69.21 | 1740 | 49.12 | 34.00 | 10.25 | 0.76 | 46 | 41 |
| <0.5 | training_v1_stride30 | SDNN | 35242 | 68.84 | 17528 | 49.74 | 34.24 | 10.57 | 0.72 | 47 | 47 |
| <1.0 | strict_reference | SDNN | 4333 | 84.66 | 1888 | 43.57 | 36.89 | 10.40 | 0.71 | 46 | 41 |
| <1.0 | training_v1_stride30 | SDNN | 43367 | 84.71 | 18998 | 43.81 | 37.11 | 10.67 | 0.73 | 47 | 47 |

## Paired comparison vs full: RMSSD

| motion filter | dataset | metric | paired MAE rows | MAE better rows | MAE worse rows | mean ΔMAE vs full ms | paired R rows | R better rows | R worse rows | median ΔR vs full | mean Δcoverage(all) pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <0.1 | strict_reference | RMSSD | 28 | 17.00 | 9.00 | -0.06 | 19 | 6.00 | 12.00 | -0.07 | -31.66 |
| <0.1 | training_v1_stride30 | RMSSD | 31 | 19.00 | 12.00 | 0.63 | 31 | 12.00 | 19.00 | -0.08 | -32.29 |
| <0.2 | strict_reference | RMSSD | 38 | 16.00 | 14.00 | -0.03 | 32 | 16.00 | 11.00 | 0.00 | -18.31 |
| <0.2 | training_v1_stride30 | RMSSD | 41 | 21.00 | 19.00 | -0.34 | 39 | 17.00 | 21.00 | -0.00 | -18.99 |
| <0.5 | strict_reference | RMSSD | 46 | 8.00 | 16.00 | 0.21 | 41 | 10.00 | 14.00 | 0.00 | -4.96 |
| <0.5 | training_v1_stride30 | RMSSD | 47 | 13.00 | 20.00 | 0.25 | 47 | 13.00 | 20.00 | 0.00 | -5.10 |
| <1.0 | strict_reference | RMSSD | 46 | 0.00 | 8.00 | 0.16 | 41 | 0.00 | 8.00 | 0.00 | -1.57 |
| <1.0 | training_v1_stride30 | RMSSD | 47 | 1.00 | 9.00 | 0.16 | 47 | 1.00 | 9.00 | 0.00 | -1.67 |

## Paired comparison vs full: SDNN

| motion filter | dataset | metric | paired MAE rows | MAE better rows | MAE worse rows | mean ΔMAE vs full ms | paired R rows | R better rows | R worse rows | median ΔR vs full | mean Δcoverage(all) pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <0.1 | strict_reference | SDNN | 28 | 18.00 | 8.00 | -1.02 | 19 | 7.00 | 11.00 | -0.01 | -31.66 |
| <0.1 | training_v1_stride30 | SDNN | 31 | 22.00 | 9.00 | -1.05 | 31 | 9.00 | 22.00 | -0.05 | -32.29 |
| <0.2 | strict_reference | SDNN | 38 | 15.00 | 15.00 | -0.07 | 32 | 13.00 | 14.00 | 0.00 | -18.31 |
| <0.2 | training_v1_stride30 | SDNN | 41 | 24.00 | 16.00 | -0.50 | 39 | 13.00 | 25.00 | -0.04 | -18.99 |
| <0.5 | strict_reference | SDNN | 46 | 11.00 | 13.00 | -0.06 | 41 | 11.00 | 13.00 | 0.00 | -4.96 |
| <0.5 | training_v1_stride30 | SDNN | 47 | 17.00 | 16.00 | -0.07 | 47 | 14.00 | 19.00 | 0.00 | -5.10 |
| <1.0 | strict_reference | SDNN | 46 | 2.00 | 6.00 | 0.09 | 41 | 2.00 | 6.00 | 0.00 | -1.57 |
| <1.0 | training_v1_stride30 | SDNN | 47 | 6.00 | 4.00 | 0.03 | 47 | 4.00 | 6.00 | 0.00 | -1.67 |

## Device-level summary: RMSSD

| motion filter | dataset | device | metric | windows kept | kept % | valid preds | coverage within kept % | coverage of all % | mean MAE ms | median R | MAE rows | R rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict_reference | Earring | RMSSD | 1706 | 100.00 | 1380 | 80.89 | 80.89 | 14.99 | 0.36 | 16 | 16 |
| full | strict_reference | Ring | RMSSD | 1706 | 100.00 | 555.00 | 32.53 | 32.53 | 11.84 | 0.40 | 16 | 15 |
| full | strict_reference | Watch | RMSSD | 1706 | 100.00 | 180.00 | 10.55 | 10.55 | 17.25 | 0.42 | 14 | 10 |
| full | training_v1_stride30 | Earring | RMSSD | 17064 | 100.00 | 13800 | 80.87 | 80.87 | 15.13 | 0.39 | 16 | 16 |
| full | training_v1_stride30 | Ring | RMSSD | 17064 | 100.00 | 5649 | 33.10 | 33.10 | 11.92 | 0.43 | 16 | 16 |
| full | training_v1_stride30 | Watch | RMSSD | 17064 | 100.00 | 1822 | 10.68 | 10.68 | 16.18 | 0.28 | 15 | 15 |
| <0.1 | strict_reference | Earring | RMSSD | 406.00 | 23.80 | 394.00 | 97.04 | 23.09 | 11.97 | 0.36 | 11 | 10 |
| <0.1 | strict_reference | Ring | RMSSD | 36.00 | 2.11 | 33.00 | 91.67 | 1.93 | 12.09 | 0.74 | 7 | 3 |
| <0.1 | strict_reference | Watch | RMSSD | 125.00 | 7.33 | 52.00 | 41.60 | 3.05 | 18.90 | 0.38 | 10 | 6 |
| <0.1 | training_v1_stride30 | Earring | RMSSD | 4070 | 23.85 | 3948 | 97.00 | 23.14 | 13.99 | 0.29 | 12 | 12 |
| <0.1 | training_v1_stride30 | Ring | RMSSD | 372.00 | 2.18 | 325.00 | 87.37 | 1.90 | 14.24 | 0.43 | 9 | 9 |
| <0.1 | training_v1_stride30 | Watch | RMSSD | 1295 | 7.59 | 498.00 | 38.46 | 2.92 | 18.49 | 0.36 | 10 | 10 |
| <0.2 | strict_reference | Earring | RMSSD | 940.00 | 55.10 | 874.00 | 92.98 | 51.23 | 13.20 | 0.55 | 15 | 15 |
| <0.2 | strict_reference | Ring | RMSSD | 219.00 | 12.84 | 141.00 | 64.38 | 8.26 | 12.73 | 0.60 | 11 | 7 |
| <0.2 | strict_reference | Watch | RMSSD | 549.00 | 32.18 | 116.00 | 21.13 | 6.80 | 19.55 | 0.26 | 12 | 10 |
| <0.2 | training_v1_stride30 | Earring | RMSSD | 9301 | 54.51 | 8658 | 93.09 | 50.74 | 13.62 | 0.43 | 15 | 15 |
| <0.2 | training_v1_stride30 | Ring | RMSSD | 2092 | 12.26 | 1406 | 67.21 | 8.24 | 12.35 | 0.23 | 12 | 11 |
| <0.2 | training_v1_stride30 | Watch | RMSSD | 5567 | 32.62 | 1154 | 20.73 | 6.76 | 16.79 | 0.28 | 14 | 13 |
| <0.5 | strict_reference | Earring | RMSSD | 1440 | 84.41 | 1199 | 83.26 | 70.28 | 15.50 | 0.36 | 16 | 16 |
| <0.5 | strict_reference | Ring | RMSSD | 833.00 | 48.83 | 394.00 | 47.30 | 23.09 | 11.87 | 0.42 | 16 | 15 |
| <0.5 | strict_reference | Watch | RMSSD | 1269 | 74.38 | 147.00 | 11.58 | 8.62 | 17.33 | 0.31 | 14 | 10 |
| <0.5 | training_v1_stride30 | Earring | RMSSD | 14411 | 84.45 | 12014 | 83.37 | 70.41 | 15.63 | 0.39 | 16 | 16 |
| <0.5 | training_v1_stride30 | Ring | RMSSD | 8292 | 48.59 | 4040 | 48.72 | 23.68 | 12.02 | 0.43 | 16 | 16 |
| <0.5 | training_v1_stride30 | Watch | RMSSD | 12539 | 73.48 | 1474 | 11.76 | 8.64 | 16.31 | 0.28 | 15 | 15 |
| <1.0 | strict_reference | Earring | RMSSD | 1528 | 89.57 | 1254 | 82.07 | 73.51 | 15.22 | 0.36 | 16 | 16 |
| <1.0 | strict_reference | Ring | RMSSD | 1325 | 77.67 | 487.00 | 36.75 | 28.55 | 11.99 | 0.40 | 16 | 15 |
| <1.0 | strict_reference | Watch | RMSSD | 1480 | 86.75 | 147.00 | 9.93 | 8.62 | 17.33 | 0.31 | 14 | 10 |
| <1.0 | training_v1_stride30 | Earring | RMSSD | 15318 | 89.77 | 12561 | 82.00 | 73.61 | 15.35 | 0.39 | 16 | 16 |
| <1.0 | training_v1_stride30 | Ring | RMSSD | 13229 | 77.53 | 4956 | 37.46 | 29.04 | 12.07 | 0.43 | 16 | 16 |
| <1.0 | training_v1_stride30 | Watch | RMSSD | 14820 | 86.85 | 1481 | 9.99 | 8.68 | 16.30 | 0.28 | 15 | 15 |

## Device-level summary: SDNN

| motion filter | dataset | device | metric | windows kept | kept % | valid preds | coverage within kept % | coverage of all % | mean MAE ms | median R | MAE rows | R rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict_reference | Earring | SDNN | 1706 | 100.00 | 1380 | 80.89 | 80.89 | 6.42 | 0.87 | 16 | 16 |
| full | strict_reference | Ring | SDNN | 1706 | 100.00 | 555.00 | 32.53 | 32.53 | 9.87 | 0.64 | 16 | 15 |
| full | strict_reference | Watch | SDNN | 1706 | 100.00 | 180.00 | 10.55 | 10.55 | 15.25 | 0.57 | 14 | 10 |
| full | training_v1_stride30 | Earring | SDNN | 17064 | 100.00 | 13800 | 80.87 | 80.87 | 6.26 | 0.89 | 16 | 16 |
| full | training_v1_stride30 | Ring | SDNN | 17064 | 100.00 | 5649 | 33.10 | 33.10 | 10.46 | 0.68 | 16 | 16 |
| full | training_v1_stride30 | Watch | SDNN | 17064 | 100.00 | 1822 | 10.68 | 10.68 | 15.49 | 0.50 | 15 | 15 |
| <0.1 | strict_reference | Earring | SDNN | 406.00 | 23.80 | 394.00 | 97.04 | 23.09 | 5.19 | 0.93 | 11 | 10 |
| <0.1 | strict_reference | Ring | SDNN | 36.00 | 2.11 | 33.00 | 91.67 | 1.93 | 7.66 | 0.71 | 7 | 3 |
| <0.1 | strict_reference | Watch | SDNN | 125.00 | 7.33 | 52.00 | 41.60 | 3.05 | 15.71 | 0.47 | 10 | 6 |
| <0.1 | training_v1_stride30 | Earring | SDNN | 4070 | 23.85 | 3948 | 97.00 | 23.14 | 5.91 | 0.85 | 12 | 12 |
| <0.1 | training_v1_stride30 | Ring | SDNN | 372.00 | 2.18 | 325.00 | 87.37 | 1.90 | 9.15 | 0.68 | 9 | 9 |
| <0.1 | training_v1_stride30 | Watch | SDNN | 1295 | 7.59 | 498.00 | 38.46 | 2.92 | 14.87 | 0.35 | 10 | 10 |
| <0.2 | strict_reference | Earring | SDNN | 940.00 | 55.10 | 874.00 | 92.98 | 51.23 | 6.04 | 0.82 | 15 | 15 |
| <0.2 | strict_reference | Ring | SDNN | 219.00 | 12.84 | 141.00 | 64.38 | 8.26 | 9.55 | 0.73 | 11 | 7 |
| <0.2 | strict_reference | Watch | SDNN | 549.00 | 32.18 | 116.00 | 21.13 | 6.80 | 16.37 | 0.53 | 12 | 10 |
| <0.2 | training_v1_stride30 | Earring | SDNN | 9301 | 54.51 | 8658 | 93.09 | 50.74 | 5.76 | 0.87 | 15 | 15 |
| <0.2 | training_v1_stride30 | Ring | SDNN | 2092 | 12.26 | 1406 | 67.21 | 8.24 | 10.33 | 0.59 | 12 | 11 |
| <0.2 | training_v1_stride30 | Watch | SDNN | 5567 | 32.62 | 1154 | 20.73 | 6.76 | 15.53 | 0.44 | 14 | 13 |
| <0.5 | strict_reference | Earring | SDNN | 1440 | 84.41 | 1199 | 83.26 | 70.28 | 6.51 | 0.88 | 16 | 16 |
| <0.5 | strict_reference | Ring | SDNN | 833.00 | 48.83 | 394.00 | 47.30 | 23.09 | 9.48 | 0.69 | 16 | 15 |
| <0.5 | strict_reference | Watch | SDNN | 1269 | 74.38 | 147.00 | 11.58 | 8.62 | 15.39 | 0.48 | 14 | 10 |
| <0.5 | training_v1_stride30 | Earring | SDNN | 14411 | 84.45 | 12014 | 83.37 | 70.41 | 6.36 | 0.90 | 16 | 16 |
| <0.5 | training_v1_stride30 | Ring | SDNN | 8292 | 48.59 | 4040 | 48.72 | 23.68 | 10.15 | 0.67 | 16 | 16 |
| <0.5 | training_v1_stride30 | Watch | SDNN | 12539 | 73.48 | 1474 | 11.76 | 8.64 | 15.49 | 0.48 | 15 | 15 |
| <1.0 | strict_reference | Earring | SDNN | 1528 | 89.57 | 1254 | 82.07 | 73.51 | 6.42 | 0.90 | 16 | 16 |
| <1.0 | strict_reference | Ring | SDNN | 1325 | 77.67 | 487.00 | 36.75 | 28.55 | 10.00 | 0.64 | 16 | 15 |
| <1.0 | strict_reference | Watch | SDNN | 1480 | 86.75 | 147.00 | 9.93 | 8.62 | 15.39 | 0.48 | 14 | 10 |
| <1.0 | training_v1_stride30 | Earring | SDNN | 15318 | 89.77 | 12561 | 82.00 | 73.61 | 6.26 | 0.90 | 16 | 16 |
| <1.0 | training_v1_stride30 | Ring | SDNN | 13229 | 77.53 | 4956 | 37.46 | 29.04 | 10.54 | 0.67 | 16 | 16 |
| <1.0 | training_v1_stride30 | Watch | SDNN | 14820 | 86.85 | 1481 | 9.99 | 8.68 | 15.50 | 0.48 | 15 | 15 |

## Files

- Aggregate CSV: `motion_threshold_sensitivity_summary_strict_vs_training_stride30.csv`
- Participant-device long CSV: `motion_threshold_sensitivity_participant_device_long_strict_vs_training_stride30.csv`
- Per-threshold detailed CSV/MD files remain in the same folder.

## Notes

- 这里的 motion threshold 使用 dataset 内的 `accel_motion_mean_mag`，也就是去重力后的 window-level motion magnitude。
- `0p5` / `1p0` 文件名只是为了避免文件名里出现小数点；报告正文仍写成 `<0.5` / `<1.0`。

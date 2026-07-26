# v1.2 Baseline Motion Threshold < 1.0

本报告只使用 `training_stride30`，并先取同一批 `participant + window_index` common-window 交集。

- Motion filter: all reported `device x channel` rows satisfy `accel_motion_mean_mag < 1.0`.
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: the same common-motion windows for every device/channel at this threshold.
- MAE/R are computed on each device/channel's QC-valid predictions inside this common-motion subset.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 13201 | 5670 | 42.95% | 11.06 ms | 0.548 | 10.98 ms | 0.796 |
| 2 | `Ring` | `ppg_ir` | 13201 | 3385 | 25.64% | 14.18 ms | 0.474 | 26.99 ms | 0.600 |
| 3 | `Watch` | `ppg_green` | 13201 | 3181 | 24.10% | 14.70 ms | 0.431 | 21.22 ms | 0.604 |
| 4 | `Earring` | `ppg_ir` | 13201 | 9368 | 70.96% | 15.54 ms | 0.442 | 7.53 ms | 0.853 |
| 5 | `Earring` | `ppg_green` | 13201 | 11372 | 86.14% | 17.08 ms | 0.436 | 6.02 ms | 0.928 |
| 6 | `Watch` | `ppg_ir` | 13201 | 2438 | 18.47% | 77.23 ms | -0.039 | 96.34 ms | 0.136 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.306 [0.124, 0.558] | 92.31% [82.25, 97.63] |
| `Earring` | `ppg_ir` | 0.284 [0.125, 0.485] | 72.06% [61.67, 87.76] |
| `Ring` | `ppg_green` | 0.410 [0.260, 0.561] | 42.78% [27.90, 51.99] |
| `Ring` | `ppg_ir` | 0.226 [-0.016, 0.490] | 24.41% [19.06, 34.65] |
| `Watch` | `ppg_green` | 0.351 [0.003, 0.448] | 21.86% [8.41, 29.21] |
| `Watch` | `ppg_ir` | -0.080 [-0.173, 0.301] | 16.89% [12.00, 21.30] |

## Output

- CSV: `v1_2_motion_lt_1p0_training_stride30.csv`

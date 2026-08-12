# v1.2 Baseline Motion Threshold < 0.5

本报告只使用 `training_stride30`，并先取同一批 `participant + window_index` common-window 交集。

- Motion filter: all reported `device x channel` rows satisfy `accel_motion_mean_mag < 0.5`.
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: the same common-motion windows for every device/channel at this threshold.
- MAE/R are computed on each device/channel's QC-valid predictions inside this common-motion subset.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 7996 | 4395 | 54.96% | 10.79 ms | 0.590 | 11.15 ms | 0.780 |
| 2 | `Ring` | `ppg_ir` | 7996 | 2919 | 36.51% | 14.23 ms | 0.492 | 27.17 ms | 0.607 |
| 3 | `Watch` | `ppg_green` | 7996 | 2642 | 33.04% | 14.39 ms | 0.443 | 20.69 ms | 0.596 |
| 4 | `Earring` | `ppg_ir` | 7996 | 6321 | 79.05% | 14.84 ms | 0.489 | 7.71 ms | 0.838 |
| 5 | `Earring` | `ppg_green` | 7996 | 7251 | 90.68% | 16.35 ms | 0.489 | 5.96 ms | 0.922 |
| 6 | `Watch` | `ppg_ir` | 7996 | 1925 | 24.07% | 74.78 ms | -0.039 | 93.11 ms | 0.158 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.272 [0.087, 0.641] | 95.12% [88.61, 97.29] |
| `Earring` | `ppg_ir` | 0.317 [0.186, 0.518] | 75.40% [66.77, 92.36] |
| `Ring` | `ppg_green` | 0.425 [0.119, 0.561] | 58.87% [45.39, 66.49] |
| `Ring` | `ppg_ir` | 0.235 [0.109, 0.486] | 38.98% [29.13, 53.16] |
| `Watch` | `ppg_green` | 0.325 [-0.117, 0.372] | 28.77% [8.33, 54.43] |
| `Watch` | `ppg_ir` | -0.214 [-0.274, 0.213] | 21.04% [16.64, 25.65] |

## Output

- CSV: `v1_2_motion_lt_0p5_training_stride30.csv`

# v1.2 Baseline Motion Threshold < 0.1

本报告只使用 `training_stride30`，并先取同一批 `participant + window_index` common-window 交集。

- Motion filter: all reported `device x channel` rows satisfy `accel_motion_mean_mag < 0.1`.
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: the same common-motion windows for every device/channel at this threshold.
- MAE/R are computed on each device/channel's QC-valid predictions inside this common-motion subset.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Earring` | `ppg_green` | 118 | 115 | 97.46% | 9.17 ms | 0.762 | 7.73 ms | 0.706 |
| 2 | `Earring` | `ppg_ir` | 118 | 114 | 96.61% | 9.34 ms | 0.660 | 9.89 ms | 0.562 |
| 3 | `Ring` | `ppg_green` | 118 | 76 | 64.41% | 12.25 ms | 0.733 | 14.34 ms | 0.562 |
| 4 | `Watch` | `ppg_green` | 118 | 49 | 41.53% | 12.26 ms | 0.403 | 26.89 ms | 0.085 |
| 5 | `Ring` | `ppg_ir` | 118 | 67 | 56.78% | 13.94 ms | 0.270 | 26.74 ms | 0.061 |
| 6 | `Watch` | `ppg_ir` | 118 | 72 | 61.02% | 47.42 ms | -0.515 | 52.74 ms | -0.360 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.431 [-0.215, 0.891] | 100.00% [100.00, 100.00] |
| `Earring` | `ppg_ir` | 0.147 [0.022, 0.464] | 100.00% [100.00, 100.00] |
| `Ring` | `ppg_green` | 0.482 [-0.024, 0.637] | 100.00% [89.47, 100.00] |
| `Ring` | `ppg_ir` | 0.434 [-0.332, 0.671] | 84.21% [66.67, 100.00] |
| `Watch` | `ppg_green` | 0.586 [0.254, 0.602] | 50.00% [36.51, 68.42] |
| `Watch` | `ppg_ir` | -0.501 [-0.529, 0.172] | 28.57% [10.53, 66.67] |

## Output

- CSV: `v1_2_motion_lt_0p1_training_stride30.csv`

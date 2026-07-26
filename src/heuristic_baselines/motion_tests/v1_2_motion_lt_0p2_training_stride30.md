# v1.2 Baseline Motion Threshold < 0.2

本报告只使用 `training_stride30`，并先取同一批 `participant + window_index` common-window 交集。

- Motion filter: all reported `device x channel` rows satisfy `accel_motion_mean_mag < 0.2`.
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: the same common-motion windows for every device/channel at this threshold.
- MAE/R are computed on each device/channel's QC-valid predictions inside this common-motion subset.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 1367 | 995 | 72.79% | 10.78 ms | 0.731 | 10.73 ms | 0.792 |
| 2 | `Watch` | `ppg_green` | 1367 | 576 | 42.14% | 13.67 ms | 0.616 | 21.38 ms | 0.479 |
| 3 | `Ring` | `ppg_ir` | 1367 | 720 | 52.67% | 14.33 ms | 0.611 | 30.20 ms | 0.522 |
| 4 | `Earring` | `ppg_ir` | 1367 | 1235 | 90.34% | 14.42 ms | 0.642 | 7.55 ms | 0.872 |
| 5 | `Earring` | `ppg_green` | 1367 | 1311 | 95.90% | 14.93 ms | 0.632 | 6.16 ms | 0.914 |
| 6 | `Watch` | `ppg_ir` | 1367 | 560 | 40.97% | 69.04 ms | -0.063 | 82.88 ms | -0.065 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.613 [0.229, 0.676] | 100.00% [96.30, 100.00] |
| `Earring` | `ppg_ir` | 0.276 [0.126, 0.564] | 97.67% [76.43, 100.00] |
| `Ring` | `ppg_green` | 0.612 [-0.081, 0.837] | 70.54% [61.31, 89.06] |
| `Ring` | `ppg_ir` | 0.406 [-0.281, 0.728] | 55.77% [43.65, 73.65] |
| `Watch` | `ppg_green` | 0.467 [0.253, 0.581] | 40.48% [16.06, 67.73] |
| `Watch` | `ppg_ir` | -0.210 [-0.303, 0.122] | 29.33% [25.50, 44.28] |

## Output

- CSV: `v1_2_motion_lt_0p2_training_stride30.csv`

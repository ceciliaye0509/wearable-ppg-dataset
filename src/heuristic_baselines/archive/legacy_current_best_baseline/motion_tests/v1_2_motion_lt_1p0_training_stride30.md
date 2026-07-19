# v1.2 Baseline Motion Threshold < 1.0

本报告只使用 `training_stride30`，并按每个 `device x channel` 独立应用 motion filter。

- Motion filter: `accel_motion_mean_mag < 1.0`
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: windows inside this motion subset for that device/channel.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 13229 | 5670 | 42.86% | 11.06 ms | 0.548 | 10.98 ms | 0.796 |
| 2 | `Ring` | `ppg_ir` | 13229 | 3390 | 25.63% | 14.22 ms | 0.472 | 27.01 ms | 0.599 |
| 3 | `Watch` | `ppg_green` | 14820 | 3236 | 21.84% | 14.80 ms | 0.423 | 21.29 ms | 0.604 |
| 4 | `Earring` | `ppg_ir` | 15318 | 10327 | 67.42% | 15.49 ms | 0.427 | 7.50 ms | 0.855 |
| 5 | `Earring` | `ppg_green` | 15318 | 13110 | 85.59% | 17.50 ms | 0.416 | 6.06 ms | 0.930 |
| 6 | `Watch` | `ppg_ir` | 14820 | 2615 | 17.65% | 76.90 ms | -0.051 | 96.49 ms | 0.124 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.204 [0.115, 0.536] | 91.31% [81.80, 96.23] |
| `Earring` | `ppg_ir` | 0.282 [0.142, 0.482] | 70.76% [58.16, 85.77] |
| `Ring` | `ppg_green` | 0.410 [0.260, 0.561] | 42.78% [27.90, 50.80] |
| `Ring` | `ppg_ir` | 0.226 [-0.016, 0.490] | 24.41% [19.05, 34.57] |
| `Watch` | `ppg_green` | 0.325 [-0.010, 0.399] | 20.32% [7.22, 26.85] |
| `Watch` | `ppg_ir` | -0.112 [-0.201, 0.305] | 16.41% [13.15, 20.76] |

## Output

- CSV: `v1_2_motion_lt_1p0_training_stride30.csv`

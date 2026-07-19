# v1.2 Baseline Motion Threshold < 0.2

本报告只使用 `training_stride30`，并按每个 `device x channel` 独立应用 motion filter。

- Motion filter: `accel_motion_mean_mag < 0.2`
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: windows inside this motion subset for that device/channel.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 2092 | 1485 | 70.98% | 10.89 ms | 0.677 | 10.20 ms | 0.775 |
| 2 | `Ring` | `ppg_ir` | 2092 | 1143 | 54.64% | 13.76 ms | 0.572 | 28.27 ms | 0.517 |
| 3 | `Earring` | `ppg_ir` | 9301 | 7421 | 79.79% | 14.24 ms | 0.456 | 7.36 ms | 0.848 |
| 4 | `Watch` | `ppg_green` | 5567 | 2020 | 36.29% | 14.37 ms | 0.513 | 21.04 ms | 0.609 |
| 5 | `Earring` | `ppg_green` | 9301 | 8522 | 91.62% | 16.30 ms | 0.454 | 5.84 ms | 0.929 |
| 6 | `Watch` | `ppg_ir` | 5567 | 1428 | 25.65% | 73.65 ms | -0.035 | 91.11 ms | 0.126 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.354 [0.090, 0.615] | 94.30% [87.15, 97.25] |
| `Earring` | `ppg_ir` | 0.294 [0.008, 0.471] | 80.05% [65.25, 91.21] |
| `Ring` | `ppg_green` | 0.400 [-0.026, 0.731] | 64.30% [57.83, 80.57] |
| `Ring` | `ppg_ir` | 0.286 [-0.149, 0.365] | 59.09% [43.53, 65.42] |
| `Watch` | `ppg_green` | 0.319 [-0.085, 0.382] | 37.64% [15.26, 51.67] |
| `Watch` | `ppg_ir` | -0.239 [-0.304, -0.104] | 23.23% [13.86, 28.39] |

## Output

- CSV: `v1_2_motion_lt_0p2_training_stride30.csv`

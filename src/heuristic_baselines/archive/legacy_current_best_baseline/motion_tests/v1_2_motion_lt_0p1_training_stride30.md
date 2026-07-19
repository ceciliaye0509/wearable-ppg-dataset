# v1.2 Baseline Motion Threshold < 0.1

本报告只使用 `training_stride30`，并按每个 `device x channel` 独立应用 motion filter。

- Motion filter: `accel_motion_mean_mag < 0.1`
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: windows inside this motion subset for that device/channel.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 372 | 296 | 79.57% | 10.48 ms | 0.806 | 10.18 ms | 0.812 |
| 2 | `Watch` | `ppg_green` | 1295 | 646 | 49.88% | 11.74 ms | 0.599 | 21.74 ms | 0.521 |
| 3 | `Ring` | `ppg_ir` | 372 | 258 | 69.35% | 12.52 ms | 0.722 | 23.80 ms | 0.556 |
| 4 | `Earring` | `ppg_ir` | 4070 | 3606 | 88.60% | 14.83 ms | 0.403 | 7.46 ms | 0.783 |
| 5 | `Earring` | `ppg_green` | 4070 | 3821 | 93.88% | 17.66 ms | 0.375 | 6.19 ms | 0.896 |
| 6 | `Watch` | `ppg_ir` | 1295 | 517 | 39.92% | 66.79 ms | -0.115 | 82.71 ms | -0.092 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.338 [0.180, 0.683] | 97.02% [93.37, 100.00] |
| `Earring` | `ppg_ir` | 0.243 [0.075, 0.569] | 94.42% [83.26, 100.00] |
| `Ring` | `ppg_green` | 0.195 [-0.211, 0.871] | 87.26% [65.99, 91.19] |
| `Ring` | `ppg_ir` | 0.111 [-0.196, 0.671] | 73.53% [58.48, 84.30] |
| `Watch` | `ppg_green` | 0.447 [0.126, 0.714] | 45.44% [8.13, 84.72] |
| `Watch` | `ppg_ir` | 0.020 [-0.151, 0.611] | 37.57% [11.80, 47.68] |

## Output

- CSV: `v1_2_motion_lt_0p1_training_stride30.csv`

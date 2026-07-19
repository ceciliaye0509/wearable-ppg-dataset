# v1.2 Baseline Motion Threshold < 0.5

本报告只使用 `training_stride30`，并按每个 `device x channel` 独立应用 motion filter。

- Motion filter: `accel_motion_mean_mag < 0.5`
- Baseline: frozen v1.2 device x channel parameters, both green and IR reported.
- Denominator: windows inside this motion subset for that device/channel.

## Aggregate Results

| Rank | Device | Channel | Motion windows | Valid preds | Coverage within motion | RMSSD MAE | RMSSD R | SDNN MAE | SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 8292 | 4540 | 54.75% | 10.83 ms | 0.580 | 11.06 ms | 0.779 |
| 2 | `Ring` | `ppg_ir` | 8292 | 3053 | 36.82% | 14.14 ms | 0.476 | 27.13 ms | 0.597 |
| 3 | `Watch` | `ppg_green` | 12539 | 3177 | 25.34% | 14.78 ms | 0.424 | 21.31 ms | 0.598 |
| 4 | `Earring` | `ppg_ir` | 14411 | 9824 | 68.17% | 15.50 ms | 0.421 | 7.49 ms | 0.856 |
| 5 | `Earring` | `ppg_green` | 14411 | 12388 | 85.96% | 17.47 ms | 0.415 | 6.08 ms | 0.931 |
| 6 | `Watch` | `ppg_ir` | 12539 | 2341 | 18.67% | 77.19 ms | -0.051 | 96.16 ms | 0.151 |

## Participant R/Coverage Variability

| Device | Channel | Participant R median [IQR] | Participant coverage median [IQR] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.211 [0.112, 0.538] | 92.41% [82.38, 96.65] |
| `Earring` | `ppg_ir` | 0.288 [0.131, 0.483] | 71.84% [60.05, 87.08] |
| `Ring` | `ppg_green` | 0.423 [0.119, 0.561] | 54.99% [45.26, 66.28] |
| `Ring` | `ppg_ir` | 0.235 [-0.023, 0.486] | 38.02% [29.05, 51.63] |
| `Watch` | `ppg_green` | 0.327 [0.012, 0.428] | 22.43% [8.24, 33.99] |
| `Watch` | `ppg_ir` | -0.166 [-0.269, 0.240] | 18.15% [10.95, 21.01] |

## Output

- CSV: `v1_2_motion_lt_0p5_training_stride30.csv`

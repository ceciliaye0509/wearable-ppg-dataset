# v1.2 noQC Training-Stride30 Ablation

本目录汇总当前 v1.2 双通道 frozen baseline 在 `training_stride30` 上关闭所有 QC gate 后的结果。

## Dataset / Inputs

- Role: `training_stride30`
- Dataset: `synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30`
- Channel metrics: `/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/heuristic_baselines/outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/v1_1_plus_neurokit_channel_metrics.csv`
- Frozen choices: `/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/heuristic_baselines/outputs/formal_v1_2_report_both_channels/v1_2_both_channels_frozen_choices.csv`
- Baseline: keep frozen `device x channel` peak methods from v1.2; report both green and IR.
- noQC definition: do not apply SQI, valid-IBI-ratio, IBI-correction-ratio, IBI-CV, correlation, or RMSSD-range gates.
- Remaining validity requirement: PPG HRV and ECG HRV must both be finite, otherwise MAE/R cannot be computed.

## Current QC Baseline

当前 v1.2 QC baseline 对每个窗口应用 frozen `device x channel` gate。QC 包括：

- `ppg_valid_sample_ratio >= 0.90`
- `ppg_sqi >= frozen min_sqi`，当前 frozen 值按 `device x channel` 为 `0.30` 或 `0.50`
- `ppg_valid_ibi_ratio >= frozen min_valid_ibi`，当前 frozen 值按 `device x channel` 为 `0.70 / 0.80 / 0.85 / 0.90`
- `ppg_ibi_correction_ratio` 必须有限，且 `<= frozen max_correction`，当前 frozen 值按 `device x channel` 为 `0.20 / 0.30 / 0.35`
- `ppg_ibi_cv` 必须有限，且 `<= frozen max_ibi_cv`，当前 frozen 值按 `device x channel` 为 `0.30 / 0.35`
- `ppg_rmssd_ms` 必须有限，且 `<= 200 ms`
- `ppg_sdnn_ms` 必须有限

当前 frozen 参数展开：

- Earring green: `min_sqi=0.30`, `min_valid_ibi=0.85`, `max_correction=0.20`, `max_ibi_cv=0.35`, `max_rmssd=200 ms`
- Earring IR: `min_sqi=0.50`, `min_valid_ibi=0.90`, `max_correction=0.20`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`
- Ring green: `min_sqi=0.30`, `min_valid_ibi=0.80`, `max_correction=0.20`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`
- Ring IR: `min_sqi=0.50`, `min_valid_ibi=0.90`, `max_correction=0.30`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`
- Watch green: `min_sqi=0.30`, `min_valid_ibi=0.90`, `max_correction=0.30`, `max_ibi_cv=0.30`, `max_rmssd=200 ms`
- Watch IR: `min_sqi=0.50`, `min_valid_ibi=0.70`, `max_correction=0.35`, `max_ibi_cv=0.35`, `max_rmssd=200 ms`

各 `device x channel` frozen QC gate：

| Device | Channel | Frozen gate | min SQI | min valid IBI | max correction | max IBI CV | max RMSSD |
|---|---|---|---:|---:|---:|---:|---:|
| `Earring` | `ppg_green` | `gate_sqi030_ibi085_corr020_cv035_rmssd200` | 0.30 | 0.85 | 0.20 | 0.35 | 200 ms |
| `Earring` | `ppg_ir` | `gate_sqi050_ibi090_corr020_cv030_rmssd200` | 0.50 | 0.90 | 0.20 | 0.30 | 200 ms |
| `Ring` | `ppg_green` | `gate_sqi030_ibi080_corr020_cv030_rmssd200` | 0.30 | 0.80 | 0.20 | 0.30 | 200 ms |
| `Ring` | `ppg_ir` | `gate_sqi050_ibi090_corr030_cv030_rmssd200` | 0.50 | 0.90 | 0.30 | 0.30 | 200 ms |
| `Watch` | `ppg_green` | `gate_sqi030_ibi090_corr030_cv030_rmssd200` | 0.30 | 0.90 | 0.30 | 0.30 | 200 ms |
| `Watch` | `ppg_ir` | `gate_sqi050_ibi070_corr035_cv035_rmssd200` | 0.50 | 0.70 | 0.35 | 0.35 | 200 ms |

## Coverage Definition

noQC ablation 的 coverage 口径：

```text
noQC coverage = n_finite_HRV / n_total

n_finite_HRV = PPG HRV 和 ECG HRV 都是有限值的窗口数
```

noQC 不应用 SQI、valid IBI ratio、IBI correction ratio、IBI CV、correlation 或 RMSSD range gate；它衡量的是 frozen detector/bandpass 能产出有限 HRV 的上限。因此 noQC coverage 可以接近 100%，但如果 detector 找不到足够 peaks、IBI 序列不足、PPG HRV 是 NaN/inf，coverage 仍然不会等于 100%。

这也是为什么不同 detector 会改变 coverage：不同 detector 会产生不同 peak 序列，进而改变 IBI 数量、IBI 稳定性、SQI、correction ratio 和最终 HRV 是否为有限值。coverage 上升不一定代表 HRV agreement 变好；它必须和 MAE、R、bias 一起解释。

## noQC vs Current QC Baseline

| Rank | Device | Channel | noQC valid / total | noQC coverage | noQC RMSSD MAE | noQC RMSSD R | QC coverage | QC RMSSD MAE | QC RMSSD R | Coverage gain | MAE change | R change | noQC SDNN MAE | noQC SDNN R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Ring` | `ppg_green` | 17064 / 17064 | 100.00% | 20.22 ms | 0.310 | 38.60% | 11.54 ms | 0.495 | 61.40 pp | 8.68 ms | -0.186 | 39.39 ms | 0.366 |
| 2 | `Earring` | `ppg_green` | 17062 / 17064 | 99.99% | 20.89 ms | 0.152 | 84.22% | 17.19 ms | 0.414 | 15.76 pp | 3.70 ms | -0.262 | 14.18 ms | 0.483 |
| 3 | `Earring` | `ppg_ir` | 16170 / 17064 | 94.76% | 24.51 ms | 0.112 | 66.39% | 15.35 ms | 0.428 | 28.38 pp | 9.16 ms | -0.316 | 23.85 ms | 0.339 |
| 4 | `Ring` | `ppg_ir` | 17064 / 17064 | 100.00% | 31.42 ms | 0.111 | 23.73% | 14.42 ms | 0.358 | 76.27 pp | 17.00 ms | -0.247 | 69.21 ms | 0.238 |
| 5 | `Watch` | `ppg_green` | 17064 / 17064 | 100.00% | 34.22 ms | 0.043 | 22.55% | 14.05 ms | 0.364 | 77.45 pp | 20.17 ms | -0.321 | 69.84 ms | 0.174 |
| 6 | `Watch` | `ppg_ir` | 17063 / 17064 | 99.99% | 84.68 ms | -0.065 | 20.07% | 68.85 ms | -0.136 | 79.93 pp | 15.83 ms | 0.071 | 111.30 ms | 0.075 |

## Participant Variability

| Device | Channel | Participant RMSSD R median [IQR] | Participant coverage median [min, max] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.152 [0.111, 0.278] | 100.00% [99.90, 100.00] |
| `Earring` | `ppg_ir` | 0.149 [0.013, 0.279] | 99.57% [73.93, 100.00] |
| `Ring` | `ppg_green` | 0.191 [0.103, 0.397] | 100.00% [100.00, 100.00] |
| `Ring` | `ppg_ir` | 0.047 [0.004, 0.218] | 100.00% [100.00, 100.00] |
| `Watch` | `ppg_green` | -0.027 [-0.138, 0.158] | 100.00% [100.00, 100.00] |
| `Watch` | `ppg_ir` | -0.018 [-0.108, 0.072] | 100.00% [99.97, 100.00] |

## Interpretation

- noQC coverage 是 frozen detector/bandpass 能产出有限 HRV 的上限，不等于真实可用质量。
- 如果 noQC coverage 接近 100% 但 MAE/R 变差，说明原 QC gate 主要是在移除错误 peak/IBI 窗口，而不是 detector 算不出来。
- 如果 noQC coverage 仍明显低于 100%，说明即使不做 QC，部分窗口也无法产生有限 HRV prediction。

## Output Files

- `v1_2_noQC_training_stride30_eval.csv`
- `v1_2_noQC_training_stride30_participant_eval.csv`
- `v1_2_noQC_vs_QC_training_stride30_comparison.csv`
- `summary.json`

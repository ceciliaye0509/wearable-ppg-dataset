# v1.2 noQC no-fill no-IBI-correction Training-Stride30 Ablation

本目录汇总当前 v1.2 双通道 frozen baseline 在 `training_stride30` 上关闭所有 QC gate、关闭 baseline-level `fill_missing`、关闭 IBI artifact correction 后的结果。

## Definition

保留：

- `ppg_resampled` 输入，也就是 dataset 生成阶段已经完成的 raw PPG 线性重采样。
- v1.2 frozen `device x channel` peak method。
- frozen peak method 对应的 bandpass 参数。
- SciPy `find_peaks` detector 和 prominence/distance 设置。
- 生理 IBI 范围过滤：`300-2000 ms`，用于从 detected peaks 得到可计算 HRV 的 NN interval。

关闭：

- 所有 frozen QC gate：SQI、valid IBI ratio、IBI correction ratio、IBI CV、RMSSD 上限等。
- baseline-level `_fill_missing`：如果输入 window 含 NaN/inf，本次不做线性补点。
- IBI artifact correction：不再用局部中位数替换异常 IBI。
- peak interpolation/refinement：当前 frozen choices 本来也不含 `_refine`。

因此本 ablation 的处理链是：

```text
raw PPG -> dataset-level ppg_resampled -> bandpass -> SciPy peak detector -> physiological IBI filter -> RMSSD/SDNN
```

注意：`ppg_resampled` 中由 dataset 生成阶段写入的 invalid zero-filled 点仍保留为 0；本实验只是关闭 baseline-level `_fill_missing`，不是从原始 timestamp PPG 重新截窗。

## Inputs

- Dataset directory: `/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/heuristic_baselines/outputs/synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30`
- Frozen choices: `/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/heuristic_baselines/outputs/formal_v1_2_report_both_channels/v1_2_both_channels_frozen_choices.csv`
- Role: `training_stride30`

## Aggregate Results

| Rank | Device | Channel | Valid / total | Coverage | RMSSD MAE | RMSSD R | RMSSD bias | SDNN MAE | SDNN R | Peak method |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `Earring` | `ppg_green` | 17062 / 17064 | 99.99% | 76.09 ms | 0.017 | 63.64 ms | 45.90 ms | 0.220 | `scipy_bp05_40_prom025_corr02` |
| 2 | `Earring` | `ppg_ir` | 16170 / 17064 | 94.76% | 134.42 ms | 0.032 | 129.10 ms | 87.51 ms | 0.185 | `scipy_bp05_40_prom025_corr02` |
| 3 | `Ring` | `ppg_green` | 17064 / 17064 | 100.00% | 207.63 ms | 0.155 | 207.04 ms | 145.69 ms | 0.197 | `scipy_bp07_35_prom025_corr02` |
| 4 | `Watch` | `ppg_green` | 17064 / 17064 | 100.00% | 299.41 ms | -0.028 | 299.38 ms | 214.12 ms | 0.161 | `scipy_bp07_35_prom025_corr02` |
| 5 | `Ring` | `ppg_ir` | 17064 / 17064 | 100.00% | 303.85 ms | 0.007 | 303.66 ms | 217.23 ms | 0.128 | `scipy_bp07_35_prom025_corr02` |
| 6 | `Watch` | `ppg_ir` | 17063 / 17064 | 99.99% | 361.17 ms | -0.132 | 361.16 ms | 263.19 ms | 0.103 | `scipy_bp07_35_prom025_corr03` |

## Participant Variability

| Device | Channel | Participant RMSSD R median [IQR] | Participant coverage median [min, max] |
|---|---|---:|---:|
| `Earring` | `ppg_green` | 0.057 [-0.024, 0.171] | 100.00% [99.90, 100.00] |
| `Earring` | `ppg_ir` | 0.069 [-0.073, 0.215] | 99.57% [73.93, 100.00] |
| `Ring` | `ppg_green` | 0.047 [-0.030, 0.286] | 100.00% [100.00, 100.00] |
| `Ring` | `ppg_ir` | 0.024 [-0.062, 0.089] | 100.00% [100.00, 100.00] |
| `Watch` | `ppg_green` | -0.040 [-0.235, 0.117] | 100.00% [100.00, 100.00] |
| `Watch` | `ppg_ir` | -0.074 [-0.187, 0.055] | 100.00% [99.97, 100.00] |

## Comparison With v1.2 noQC Training-Stride30 Ablation

对比对象是现有 `v1.2 noQC Training-Stride30 Ablation`，即保留 `_fill_missing` 和 IBI correction、但关闭 QC gate 的结果。

| Device | Channel | This coverage | noQC coverage | Coverage change | This RMSSD MAE | noQC RMSSD MAE | MAE change | This RMSSD R | noQC RMSSD R | R change | This SDNN MAE | noQC SDNN MAE | SDNN MAE change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Earring` | `ppg_green` | 99.99% | 99.99% | 0.00 pp | 76.09 ms | 20.89 ms | 55.19 ms | 0.017 | 0.152 | -0.135 | 45.90 ms | 14.18 ms | 31.73 ms |
| `Earring` | `ppg_ir` | 94.76% | 94.76% | 0.00 pp | 134.42 ms | 24.51 ms | 109.91 ms | 0.032 | 0.112 | -0.080 | 87.51 ms | 23.85 ms | 63.65 ms |
| `Ring` | `ppg_green` | 100.00% | 100.00% | 0.00 pp | 207.63 ms | 20.22 ms | 187.42 ms | 0.155 | 0.310 | -0.155 | 145.69 ms | 39.39 ms | 106.30 ms |
| `Watch` | `ppg_green` | 100.00% | 100.00% | 0.00 pp | 299.41 ms | 34.22 ms | 265.19 ms | -0.028 | 0.043 | -0.070 | 214.12 ms | 69.84 ms | 144.27 ms |
| `Ring` | `ppg_ir` | 100.00% | 100.00% | 0.00 pp | 303.85 ms | 31.42 ms | 272.44 ms | 0.007 | 0.111 | -0.105 | 217.23 ms | 69.21 ms | 148.02 ms |
| `Watch` | `ppg_ir` | 99.99% | 99.99% | -0.00 pp | 361.17 ms | 84.68 ms | 276.49 ms | -0.132 | -0.065 | -0.067 | 263.19 ms | 111.30 ms | 151.89 ms |

### Summary

- 本 ablation 相比现有 `v1.2 noQC Training-Stride30 Ablation` 明显变差。所有 6 个 `device x channel` 的 RMSSD MAE 都上升，R 也整体下降。
- Coverage 基本没有变化：Earring green、Earring IR、Ring green、Ring IR、Watch green 的 coverage change 都是 `0.00 pp`；Watch IR 只有浮点舍入级别的 `-0.00 pp`。因此关闭 baseline-level `_fill_missing` 没有实质性改变 coverage。
- 主要变化来自关闭 IBI artifact correction。RMSSD MAE 增幅从 `55.19 ms` 到 `276.49 ms`，SDNN MAE 增幅从 `31.73 ms` 到 `151.89 ms`。这说明在关闭 QC gate 时，IBI correction 是抑制异常 IBI 对 RMSSD/SDNN 破坏的关键步骤。
- `Earring ppg_green`: RMSSD MAE change `55.19 ms`, R change `-0.135`, coverage change `0.00 pp`.
- `Earring ppg_ir`: RMSSD MAE change `109.91 ms`, R change `-0.080`, coverage change `0.00 pp`.
- `Ring ppg_green`: RMSSD MAE change `187.42 ms`, R change `-0.155`, coverage change `0.00 pp`.
- `Watch ppg_green`: RMSSD MAE change `265.19 ms`, R change `-0.070`, coverage change `0.00 pp`.
- `Ring ppg_ir`: RMSSD MAE change `272.44 ms`, R change `-0.105`, coverage change `0.00 pp`.
- `Watch ppg_ir`: RMSSD MAE change `276.49 ms`, R change `-0.067`, coverage change `-0.00 pp`.

结论：本 ablation 的性能明显差于现有 noQC，但 coverage 几乎完全不变。因此，本次差异不是由可计算窗口数量变化造成的，而是由不做 IBI correction 后异常 IBI 直接进入 RMSSD/SDNN 计算造成的。当前 `training_stride30` 的 `ppg_resampled` 输入大多已经是 finite，baseline-level `_fill_missing` 对这个实验的实际影响很小。

## Output Files

- `v1_2_noQC_noFill_noCorr_training_stride30_eval.csv`
- `v1_2_noQC_noFill_noCorr_training_stride30_participant_eval.csv`
- `v1_2_noQC_noFill_noCorr_training_stride30_predictions.csv`
- `v1_2_noQC_noFill_noCorr_vs_noQC_training_stride30_comparison.csv`
- `summary.json`

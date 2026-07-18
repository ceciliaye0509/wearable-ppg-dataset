# HRV/PPG Baseline 正式实验边界


## 当前数据集

- 开发 / 连续监测数据集：
  `synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30`
- 最终严格窗口级评估数据集：
  `synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100`

两个数据集来自同一批原始参与者。`strict_reference` 使用不重叠窗口，可以降低时间冗余，但它不是受试者独立的留出测试集。

## 使用字段

当前 rawaligned NPZ 不存储 PPG-derived baseline metadata。正式 baseline 必须从以下字段重新计算 PPG-derived 指标：

- `ppg_resampled`
- `ppg_valid_mask_resampled`
- `ppg_valid_sample_ratio`
- `accel_mean_mag`
- `accel_motion_mean_mag`

ECG 字段只作为参考标签：

- `ecg_r_peak_times_rel_ms`
- `ecg_r_peak_indices_grid`
- `ecg_rmssd_ms`
- `ecg_sdnn_ms`

ECG label 不能用于逐窗口选择 channel / device，也不能用于 development set 以外的参数选择。

## 正式选择规则

正式 baseline 选择只能使用开发数据：

- 使用 `training_stride30` 选择参数。
- 冻结一个 device-wise heuristic rule。
- 在 `strict_reference` 上评估冻结后的规则。
- Earring、Ring、Watch 必须分别报告。

允许：

- 每个设备单独评估。
- 每个设备可以使用在开发数据上选出的固定最佳通道。
- 可以使用一套统一的设备内部 green/IR quality-based selection 规则。
- 可以使用 PPG-only quality features：SQI、valid IBI ratio、IBI correction ratio、`accel_mean_mag`、`accel_motion_mean_mag`。

禁止：

- 跨设备 median fusion。
- 跨设备 best selection。
- 使用 ECG label 做逐窗口 channel / device selection。
- 在 `strict_reference` 上调参。
- 把 `strict_reference` 称为受试者独立测试集。

## 候选集合

候选集合应保持小而明确。

当前已完成 full-cohort recomputation 的正式候选：

- Detector：SciPy `find_peaks`
- Bandpass：`0.7-3.5 Hz`、`0.5-4.0 Hz`
- Prominence：`0.25 * std`
- IBI correction threshold：`0.20`
- SQI gate：`0.4` 或 `0.5`
- valid IBI ratio gate：`0.8` 或 `0.9`
- physiological IBI range：`300-2000 ms`

只有在重新计算完整开发集指标后，以下候选才可以进入正式冻结：

- NeuroKit Elgendi raw clean-only。
- 外部 bandpass + NeuroKit Elgendi double-clean。
- SciPy prominence `0.30`。
- IBI correction threshold `0.30`。

抽样检查可以用于 triage，但抽样检查不能冻结正式 primary baseline。

## 当前正式脚本

运行：

```bash
python src/heuristic_baselines/formal_freeze_v1_unified_baseline.py
```

这个脚本会：

1. 读取已经重新计算好的通道级指标。
2. 只使用 `training_stride30` rows 做候选选择。
3. 冻结一个统一处理规则。
4. 只把冻结后的规则应用到 `strict_reference`。
5. 输出正式结果，不写出 `strict_reference` 候选扫描。

## 归档内容

`archive/` 目录保存历史探索脚本和旧报告，不属于当前正式 v1-primary unified heuristic baseline 主流程。

其中 `archive/legacy_current_best_baseline/` 包含旧版 `current_best_baseline` 和 motion threshold sensitivity 分析。它们只用于历史追溯，不用于正式调参或最终报告。

`archive/legacy_step_matrix_diagnostics/` 包含旧版 4-device step-matrix 消融和诊断脚本。它们包含 `Necklace` 或旧设备命名，也不用于当前 3-device 正式 baseline。

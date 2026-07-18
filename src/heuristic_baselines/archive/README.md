# 归档说明

这个目录保存历史探索脚本和旧实验报告。

归档内容不属于当前正式 v1-primary unified heuristic baseline 主流程。保留它们的目的只是为了回看旧结果、追踪实验演进，或者在需要时复跑历史分析。

当前正式 baseline 主流程仍在上一级目录：

- `../evaluate_devicewise_recomputed_peaks_v1.py`
- `../formal_freeze_v1_unified_baseline.py`
- `../README_formal_baseline_experiment.md`

不要用归档脚本选择正式 baseline 参数，也不要把归档报告作为正式 v1 primary baseline 的最终结果。

## 归档分组

- `legacy_current_best_baseline/`
  - 逻辑：直接用旧版 `hrv_from_ppg()` 从 PPG 估计 HRV，再按 SQI 在 green/IR 中选通道；motion 版本额外按加速度阈值筛窗口。
  - 和当前版本区别：当前正式 v1 从 `ppg_resampled` 重新计算 peaks / IBI / SQI / QC，并且只用 `training_stride30` 冻结统一规则，不使用旧的逐窗口 SQI-best current-best 结论。
- `legacy_step_matrix_diagnostics/`
  - 逻辑：旧版 4-device / Necklace 消融实验和诊断图，用来比较插值、IBI 校正、阈值门限、降采样等处理步骤。
  - 和当前版本区别：当前正式 v1 是 3-device（Earring/Ring/Watch）统一 heuristic baseline，不再使用旧 4-device step-matrix 作为正式调参依据。

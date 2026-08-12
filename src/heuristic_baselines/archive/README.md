# 归档说明

这个目录保存历史探索脚本、旧 pipeline、专题消融和不再作为当前主线的实验报告。

归档内容不属于当前主线 baseline 流程。保留它们的目的只是为了回看旧结果、追踪实验演进，或者在需要时复跑历史分析。

当前仍在上一级目录保留的主线/参考流程：

- `../evaluate_rawslots_baseline_ablation.py`
- `../formal_freeze_v1_2_device_channel_params.py`
- `../formal_freeze_v1_2_report_both_channels.py`
- `../evaluate_devicewise_recomputed_peaks_v1.py`
- `../formal_freeze_v1_unified_baseline.py`
- `../README_formal_baseline_experiment.md`
- `../README_baseline_optimization_summary.md`

不要用归档脚本选择新的正式 baseline 参数，也不要把归档报告作为当前 baseline 的最终结果。

## 归档分组

- `legacy_current_best_baseline/`
  - 逻辑：直接用旧版 `hrv_from_ppg()` 从 PPG 估计 HRV，再按 SQI 在 green/IR 中选通道；motion 版本额外按加速度阈值筛窗口。
  - 和当前版本区别：当前正式 v1 从 `ppg_resampled` 重新计算 peaks / IBI / SQI / QC，并且只用 `training_stride30` 冻结统一规则，不使用旧的逐窗口 SQI-best current-best 结论。
- `legacy_step_matrix_diagnostics/`
  - 逻辑：旧版 4-device / Necklace 消融实验和诊断图，用来比较插值、IBI 校正、阈值门限、降采样等处理步骤。
  - 和当前版本区别：当前正式 v1 是 3-device（Earring/Ring/Watch）统一 heuristic baseline，不再使用旧 4-device step-matrix 作为正式调参依据。
- `legacy_5min_pipeline/`
  - 旧的 5 分钟窗口 HR/HRV pipeline 入口，包括 `runner.py`、`hrv_runner.py`、`eval_ppg_vs_ecg.py`、`run_all.py`、`bench_participant.py` 和 `README_HRV.md`。
  - 当前 rawaligned/rawslots baseline 不再通过这些脚本产出正式指标。
- `noqc_ablation/`
  - v1.2 no-QC、no-fill、no-IBI-correction 专题消融。
  - 用于解释 QC/填补/校正的影响，不作为正式 baseline 入口。
- `motion_threshold_sensitivity/`
  - v1.2 motion threshold sensitivity 脚本和报告。
  - 属于专题分析，不参与当前默认 freeze。
- `activity_hrv_case_study/`
  - ECG-only activity HRV case study 图、CSV 和生成脚本。
  - 属于个案分析，不参与 baseline freeze。

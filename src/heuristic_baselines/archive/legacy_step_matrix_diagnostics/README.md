# 旧版 step-matrix 与诊断脚本归档

这个目录保存旧版 4-device / Necklace 时代的消融实验和诊断脚本。

这些脚本不属于当前正式 v1-primary unified heuristic baseline 主流程。保留它们只是为了回看旧实验设计、旧图表或旧阈值诊断。

## 文件

- `legacy_full_step_matrix_v2.py`
  - 逻辑：对旧版 4-device 数据跑 step-matrix 消融，组合测试亚采样插值、IBI artifact correction、阈值门限、运动质控、降采样等处理步骤。
  - 和当前版本区别：当前正式 v1 不做这套大矩阵调参，而是使用小候选集合，在 `training_stride30` 上冻结一条统一 heuristic rule。
- `legacy_generate_scatter_plots.py`
  - 逻辑：基于旧 step-matrix 的每个步骤生成 PPG RMSSD vs ECG RMSSD 散点图，用于可视化不同步骤相对 baseline 的变化。
  - 和当前版本区别：当前正式结果主要看冻结规则在 training/strict 上的 MAE、RMSE、R、bias、coverage，不再用旧 step 图作为正式证据。
- `legacy_plot_valid_sample_ratio_dist.py`
  - 逻辑：统计旧版 4-device rawaligned 数据里的 `ppg_valid_sample_ratio` 分布，辅助判断旧阈值如 0.50 是否合理。
  - 和当前版本区别：当前正式 v1 使用 3-device 数据，并且质量门控以重新计算后的 SQI、valid IBI ratio、correction ratio、IBI CV 等为主。

## 为什么归档

- 它们面向旧 4-device 设置，包含 `Necklace` 或 apple/garmin/polar/samsung 等旧设备命名。
- 当前正式 baseline 使用 3-device 数据：`Earring`、`Ring`、`Watch`。
- 当前正式 baseline 需要从 `training_stride30` 冻结统一规则，再在 `strict_reference` 上最终评估。

因此，这些脚本只用于历史追溯，不用于正式调参或最终报告。

# 旧版 current-best baseline 归档

这个目录保存旧版 `current_best_baseline` 系列脚本和 motion threshold 探索报告。

这些脚本使用旧逻辑：

- 直接调用 `hrv_from_ppg()` 计算 PPG HRV。
- 在 `ppg_green` / `ppg_ir` 之间按 SQI 选择通道。
- `no-motion` 版本额外使用 `accel_motion_mean_mag` 阈值筛选窗口。

这些结果是探索性材料，不用于当前正式 v1-primary unified heuristic baseline。

## 文件

- `legacy_evaluate_current_best_baseline_report.py`
  - 逻辑：遍历 full-window 数据，对每个设备的 green/IR 都调用 `hrv_from_ppg()`，再按 SQI、valid IBI ratio、IBI correction ratio 选择一个窗口级最佳通道。
  - 和当前版本区别：当前正式 v1 不再用这个旧函数直接产出最终 baseline，而是先全量重算 peaks/IBI/SQI，再在开发集冻结统一规则。
- `legacy_evaluate_current_best_baseline_no_motion_report.py`
  - 逻辑：在旧版 current-best 逻辑外加一层 `accel_motion_mean_mag < threshold` 的窗口筛选，用来观察低运动窗口下的表现。
  - 和当前版本区别：当前正式 v1 的 primary baseline 不把 motion threshold 作为正式核心规则；motion 信息最多作为后续分析变量，不能替代冻结后的统一规则。
- `motion_tests/`
  - 逻辑：保存旧版 motion threshold sensitivity 的 Markdown/CSV，对比 full、motion < 0.1、0.2、0.5、1.0 等阈值下的结果。
  - 和当前版本区别：这些报告是历史探索输出，不是当前正式 `formal_v1_unified_baseline` 的结果。

`motion_tests/` 里的 Markdown 是当时生成的历史报告，内容保留原貌；它们不是当前维护文档，也不作为正式 baseline 结果引用。

## 当前正式流程

正式流程请回到两级上级目录使用：

- `../../evaluate_devicewise_recomputed_peaks_v1.py`
- `../../formal_freeze_v1_unified_baseline.py`
- `../../README_formal_baseline_experiment.md`

归档脚本只用于历史追溯，不用于正式调参或最终报告。

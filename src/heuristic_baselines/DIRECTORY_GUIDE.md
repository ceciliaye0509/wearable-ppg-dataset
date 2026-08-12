# heuristic_baselines 目录指南

本文说明当前顶层文件用途，以及哪些内容已经移入 `archive/`。

## 当前主线

| 文件/目录 | 作用 |
|---|---|
| `evaluate_rawslots_baseline_ablation.py` | 从 `synced_3device_rawaligned_training_v1_stride30_rawslots` 全量重跑 rawslot/resampled、peak/foot、IBI correction、QC/no-QC 消融，并冻结 device x channel 结果。 |
| `evaluate_rawslots_detector_fiducials.py` | detector-native peak vs foot/onset 轻量验证脚本；默认每个 participant 抽样 5 个窗口、单进程、只跑 `scipy,qppgfast`，避免 16GB 机器误触发全量重任务。PWD/MSPTD/qppgfast 在 5 分钟窗口上计算较重，目前不作为 frozen baseline 输出。 |
| `qppgfast_devicewise_v1.py` | rawslots 正式、不可调的 qppgfast baseline 入口：strict interpolation 100 Hz/gap 100 ms、`0.7-3.5 Hz`、IBI correction `0.20`、Earring peak + Ring/Watch native foot/onset；同时产出同输入 SciPy-peak 对照。按 participant 流式处理，适合 16GB 机器跑全量。 |
| `build_rawslots_qppgfast_devicewise_report_tables.py` | 从正式 full16 输出生成主研究笔记所用的 6 通道、共同有效窗口与宏平均 CSV。 |
| `summarize_rawslots_waveform_consensus_freeze.py` | 汇总不读取 ECG、fiducial 或 IBI 的 `waveform_consensus_v1` 极性敏感性验证：5 个 60 秒段中至少 4 段形态方向一致才接受，否则标为 ambiguous；输出训练/独立 holdout 的预定义冻结验收。 |
| `formal_freeze_v1_2_device_channel_params.py` | v1.2 device/channel-specific 参数冻结基础逻辑。 |
| `formal_freeze_v1_2_report_both_channels.py` | v1.2 双通道正式报告入口；green 和 IR 都汇报，不做 best-channel 压缩。 |
| `README_baseline_optimization_summary.md` | 当前 baseline 优化历史和主结论汇总。 |
| `README_formal_baseline_experiment.md` | formal baseline 的实验边界、数据角色和禁止事项。 |

## 保留的历史参考

| 文件 | 作用 |
|---|---|
| `evaluate_devicewise_recomputed_peaks_v1.py` | v1 recomputed peaks baseline 候选扫描。 |
| `freeze_v1_primary_unified_baseline.py` | v1 primary unified freeze 的共享逻辑。 |
| `formal_freeze_v1_unified_baseline.py` | v1 conservative reference freeze。 |
| `evaluate_devicewise_recomputed_peaks_v1_1.py` | v1.1 扩展候选扫描。 |
| `evaluate_neurokit_fullcohort_cached_v1_1.py` | NeuroKit Elgendi full-cohort 候选补算。 |
| `formal_freeze_v1_1_unified_baseline.py` | v1.1 formal freeze。 |

## 基础模块

| 文件/目录 | 作用 |
|---|---|
| `algorithms/` | HR/HRV/PRV 相关算法实现，包括 PWD、MSPTD、FFT、autocorr、HeartPy、NeuroKit、SQA 和 HRV 指标工具。 |
| `config.py` | 路径、participant、device、channel、输出目录等配置。 |
| `preprocess.py` | PPG detrend + bandpass 工具。 |
| `io_utils.py` | participant ID、窗口 NPZ 路径、PPG 字段读取等辅助函数。 |
| `requirements.txt` | 依赖列表。 |
| `__init__.py` | Python package 标识。 |

## 输出目录

| 目录 | 作用 |
|---|---|
| `outputs/rawslots_baseline_ablation_v1/` | 最新 rawslots 全量消融和冻结报告。 |
| `outputs/rawslots_qppgfast_devicewise_v1_full16/` | `qppgfast_devicewise_v1` 的全 16 人端到端复现、冻结配置、逐窗口明细和 SciPy-peak 验收对照。 |
| `outputs/rawslots_qppgfast_waveform_consensus_freeze_v1/` | waveform-only 极性一致性规则的已冻结敏感性结果；独立 holdout coverage/R 未达预定义要求，因此不替代正式的 `peak_train` 极性规则。 |
| `outputs/formal_v1_2_report_both_channels/` | 当前推荐 v1.2 双通道报告。 |
| `outputs/formal_v1_2_device_channel_params/` | v1.2 参数冻结结果。 |
| `outputs/formal_v1_unified_baseline/` | v1 conservative reference 结果。 |
| `outputs/synced_3device_*` | 数据集产物，体积较大；暂时保留在原位，避免打断现有脚本默认路径。 |

## 已归档

| 目录 | 内容 |
|---|---|
| `archive/legacy_5min_pipeline/` | 老 5min pipeline：`runner.py`、`hrv_runner.py`、`eval_ppg_vs_ecg.py`、`run_all.py`、`bench_participant.py`、`README_HRV.md`。 |
| `archive/noqc_ablation/noQC/` | v1.2 no-QC、no-fill、no-IBI-correction 专题消融。 |
| `archive/motion_threshold_sensitivity/` | v1.2 motion threshold sensitivity 脚本和报告。 |
| `archive/activity_hrv_case_study/activity_hrv_case_study/` | ECG-only activity HRV 个案分析图、CSV 和生成脚本。 |
| `archive/diagnostics/` | 一次性诊断脚本，例如 SciPy vs NeuroKit participant/device 诊断。 |
| `archive/legacy_current_best_baseline/` | 旧版 current-best baseline。 |
| `archive/legacy_step_matrix_diagnostics/` | 旧版 step-matrix 诊断实验。 |

## 清理规则

- `__pycache__/`、`.DS_Store`、临时 smoke 输出和 `.cache/` 不保留。
- 当前主线结果不移动。
- 大体积数据集产物若要进一步整理，建议先统一改脚本默认路径，再从 `outputs/` 移出。
- detector-native foot vs peak 验证优先使用轻量命令：`python evaluate_rawslots_detector_fiducials.py`。如需全量，必须显式加 `--allow-full`，且建议只跑单个 detector。

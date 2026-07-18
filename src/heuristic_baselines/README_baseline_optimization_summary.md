# HRV/PPG Baseline 优化汇总

本文是当前 `src/heuristic_baselines/` 目录下的 正式baseline 优化记录。它只总结已经落到当前代码或当前输出目录里的内容，不把想法、抽样试验或已经归档的旧逻辑当作当前正式 baseline。

## 当前结论

- 当前 primary comparison 仍建议使用 `formal_v1_unified_baseline`。
- v1.1 扩展候选后没有在 primary unified 边界内超过 v1；它保留为一次候选扩展记录。
- full-cohort NeuroKit Elgendi candidates 已经补算并放入同一个 formal unified freeze 框架比较；结果仍然没有替代当前 SciPy primary baseline。
- v1.1 的 device-wise analysis 显示：如果允许每个设备使用不同处理参数，Earring 和 Ring 可以更好；但这不满足“所有设备同一套处理办法，只允许设备内固定通道不同”的公平对比边界。
- Watch 的 RMSSD 覆盖率仍然偏低；full-cohort NeuroKit 没有解决这个问题，下一步更应该看 gate / IBI 质量门控。

## 实验边界

对应文件：

- `README_formal_baseline_experiment.md`

简单逻辑：

- 明确当前两个数据集的角色：`training_stride30` 用作开发/参数选择，`strict_reference` 用作冻结后最终窗口级评估。
- 明确 `strict_reference` 不是 subject-independent held-out test。
- 明确当前 NPZ 不保存 PPG-derived metadata，因此 PPG peaks、IBI、SQI、HRV 必须从 `ppg_resampled` 重新计算。
- 明确禁止跨设备 fusion、跨设备 best selection、用 ECG label 逐窗口选通道或调参。
- 明确正式脚本只应把 `strict_reference` 用在冻结后的评估上。

这部分来自你发送的实验边界要求，并且现在已经保留为项目内 README。

## v1：重新计算 PPG-derived baseline 指标

对应文件：

- `evaluate_devicewise_recomputed_peaks_v1.py`

简单逻辑：

- 从 `ppg_resampled` 重新计算 PPG peaks、IBI、RMSSD、SDNN、SQI、valid IBI ratio、IBI correction ratio、IBI CV。
- v1 候选包含 SciPy `find_peaks`，bandpass 为 `0.7-3.5 Hz` 和 `0.5-4.0 Hz`，prominence 为 `0.25 * std`，IBI correction threshold 为 `0.20`。
- 质量门控包含 `ppg_valid_sample_ratio >= 0.90`、SQI、valid IBI ratio、IBI correction ratio、IBI CV、RMSSD 上限等条件。
- 同时生成固定通道策略和设备内部 `SQI-best` 策略，但选择只能在 `training_stride30` 上完成。
- 这个脚本是“每设备独立 baseline 评估脚本”：它也可以为 Earring、Ring、Watch 各自选出表现最好的 device-wise 策略，用于 stronger reference / analysis；它不是最终 primary unified baseline。

对应输出：

- `outputs/all_participants_devicewise_baseline_v1_recomputed_peaks/`

## v1 primary unified baseline：统一处理规则冻结

对应文件：

- `freeze_v1_primary_unified_baseline.py`

简单逻辑：

- 所有设备共用同一套 peak detector、bandpass、IBI correction 和 QC gate。
- 允许每个设备在 `ppg_green` / `ppg_ir` 中选择一个固定最佳通道。
- 不允许 Earring、Ring、Watch 各自使用不同 bandpass 或不同 QC gate 作为 primary baseline。
- `formal_freeze_v1_unified_baseline.py` 只用 `training_stride30` 产生候选总结和冻结规则，再把冻结规则应用到 `strict_reference`；它不会输出 strict 上的候选扫描，避免继续用 strict 调参。

v1 formal 结果：

| 设备 | strict RMSSD MAE | strict RMSSD R | strict 覆盖率 | strict SDNN R |
|---|---:|---:|---:|---:|
| Earring | 16.66 ms | 0.310 | 77.32% | 0.774 |
| Ring | 12.52 ms | 0.448 | 57.74% | 0.716 |
| Watch | 14.07 ms | 0.347 | 22.57% | 0.589 |

对应输出：

- `outputs/formal_v1_unified_baseline/`

## v1.1：候选扩展和峰值细化接口

对应文件：

- `evaluate_devicewise_recomputed_peaks_v1_1.py`

简单逻辑：

- v1.1 在 v1 基础上扩展了 SciPy prominence `0.30`（更严格的 peak detection）、IBI correction threshold `0.30`（允许 IBI 序列有更多波动）、coverage-friendly gate `gate_sqi035_ibi08_corr03_cv35_rmssd200`。
- 增加了 `refine_peaks` 字段和 peak refinement 候选；默认不全量运行 refinement，需要显式使用 `--include-refinement`。
- 增加 `--base-channel-metrics-csv`，可以复用 v1 已经算好的基础候选，避免重复计算。
- 增加 `--workers`，使用线程并行处理参与者。

v1.1 device-wise analysis 结果：

| 设备 | strict RMSSD MAE | strict RMSSD R | strict 覆盖率 | 解释 |
|---|---:|---:|---:|---|
| Earring | 15.51 ms | 0.413 | 67.23% | 比 unified v1 的 MAE/R 更好，但覆盖率下降 |
| Ring | 11.66 ms | 0.490 | 38.16% | 比 unified v1 的 MAE/R 更好，但覆盖率下降 |
| Watch | 14.07 ms | 0.347 | 22.57% | 基本沿用 v1 结果 |

注意：这张表允许不同设备使用不同处理参数，因此只作为 stronger reference / analysis，不作为 primary comparison。

对应输出：

- `outputs/all_participants_devicewise_baseline_v1_1_recomputed_peaks/`

## v1.1 formal unified baseline：重新冻结但不替代 v1

对应文件：

- `formal_freeze_v1_1_unified_baseline.py`

简单逻辑：

- 在 v1.1 候选集合上重新做 primary unified freeze。
- 仍然只使用 `training_stride30` 选择参数，`strict_reference` 只用于冻结后的评估。
- v1.1 formal 重新冻结后仍选择 v1 的同一套统一规则：
  `scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200`
- 因此 v1.1 formal 不替代 v1，而是记录“扩展候选后没有改变 primary unified 规则”。

v1.1 formal 结果与 v1 formal 相同：

| 设备 | strict RMSSD MAE | strict RMSSD R | strict 覆盖率 |
|---|---:|---:|---:|
| Earring | 16.66 ms | 0.310 | 77.32% |
| Ring | 12.52 ms | 0.448 | 57.74% |
| Watch | 14.07 ms | 0.347 | 22.57% |

对应输出：

- `outputs/formal_v1_1_unified_baseline/`

## full-cohort NeuroKit candidate：没有替代 SciPy primary baseline

对应文件：

- `evaluate_neurokit_fullcohort_cached_v1_1.py`
- `formal_freeze_v1_1_unified_baseline.py`
- `diagnose_scipy_vs_neurokit_participants.py`
- `README_neurokit_fullcohort_comparison.md`

执行的三步：

1. 补算 full-cohort NeuroKit Elgendi candidates。
2. 将 SciPy + NeuroKit 放入同一个 formal unified freeze 框架比较。
3. 生成 primary SciPy vs best NeuroKit unified 的 per-participant error diagnosis。

简单逻辑：

- NeuroKit 补算包含两个 candidate：
  - `nk_elgendi_raw_cleanonly_corr02`
  - `nk_elgendi_bp07_35_doubleclean_corr02`
- 为了避免长任务中断后重跑，`evaluate_neurokit_fullcohort_cached_v1_1.py` 按 `role + participant` 写缓存。
- 合并后生成 `outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/v1_1_plus_neurokit_channel_metrics.csv`，再用 formal freeze 脚本只在 `training_stride30` 上重新选择统一规则。
- `diagnose_scipy_vs_neurokit_participants.py` 只读取已有结果，不重新计算 peaks，用于比较 participant/device 级误差和 QC 失败原因。

formal unified 结果：

- 加入 full-cohort NeuroKit 后，冻结规则仍然选中当前 SciPy v1 规则：
  `scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200`
- 因此 NeuroKit 没有替代 current primary SciPy baseline。

Training 上 Top unified candidates：

| 排名 | Detector | 方法简写 | mean RMSSD MAE | mean R | mean coverage | min coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | SciPy | `bp07_35_prom025_corr02 + gate_sqi04_ibi08_corr03_cv30` | 14.43 ms | 0.378 | 52.67% | 22.83% |
| 2 | SciPy | `bp07_35_prom025_corr02 + gate_sqi035_ibi08_corr03_cv35` | 14.44 ms | 0.376 | 52.71% | 22.87% |
| 3 | SciPy | `bp07_35_prom030_corr02 + gate_sqi035_ibi08_corr03_cv35` | 14.76 ms | 0.368 | 52.33% | 22.83% |
| 4 | SciPy | `bp07_35_prom030_corr02 + gate_sqi04_ibi08_corr03_cv30` | 14.76 ms | 0.368 | 52.31% | 22.79% |
| 5 | NeuroKit | `bp07_35_doubleclean_corr02 + gate_sqi04_ibi09_corr02_cv25` | 15.09 ms | 0.377 | 48.13% | 20.48% |

结论：

- 最佳 NeuroKit unified candidate 的 training mean RMSSD MAE 比 primary SciPy 高约 `0.66 ms`。
- NeuroKit mean coverage 也更低：`48.13%` vs `52.67%`。
- NeuroKit mean R 接近 SciPy，但没有超过 SciPy。

per-participant diagnosis 的 strict_reference 结果：

| 设备 | 方法 | mean MAE | median MAE | mean R | mean coverage |
|---|---|---:|---:|---:|---:|
| Earring | primary SciPy | 17.76 ms | 16.44 ms | 0.262 | 78.38% |
| Earring | best NeuroKit | 18.18 ms | 16.52 ms | 0.363 | 75.11% |
| Ring | primary SciPy | 11.99 ms | 11.66 ms | 0.386 | 56.88% |
| Ring | best NeuroKit | 12.10 ms | 12.07 ms | 0.369 | 48.15% |
| Watch | primary SciPy | 12.74 ms | 10.67 ms | 0.297 | 23.32% |
| Watch | best NeuroKit | 21.33 ms | 12.81 ms | 0.222 | 21.70% |

QC 诊断：

- NeuroKit 在三个设备上都带来更高的 `valid IBI` 失败比例。
- Ring 和 Watch 的主要瓶颈仍然是 `IBI correction ratio`。
- Watch 的 NeuroKit pass-all 约 `20.34%`，primary SciPy pass-all 约 `21.87%`，所以 NeuroKit 没有解决 Watch 覆盖率低的问题。

对应输出：

- `outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/`
- `outputs/formal_v1_1_plus_neurokit_unified_baseline/`
- `outputs/diagnosis_scipy_vs_neurokit_participants/`

## 归档整理

对应文件：

- `archive/README.md:1-22`
- `archive/legacy_current_best_baseline/README.md:1-35`
- `archive/legacy_step_matrix_diagnostics/README.md:1-25`

简单逻辑：

- 旧版 `current_best_baseline`、motion threshold sensitivity、step-matrix、valid sample ratio 分布图等已经归档。
- 归档脚本只用于历史追溯，不用于正式调参或最终报告。
- 当前正式 baseline 主流程应回到：
  - `evaluate_devicewise_recomputed_peaks_v1.py`
  - `formal_freeze_v1_unified_baseline.py`
  - `README_formal_baseline_experiment.md`

这一步保留了历史结果，但避免它们和当前 primary unified baseline 混在一起。

## 没有作为正式结论的内容

- P1/P3/P5/P18 等小样本只用于 debug / triage，不用于正式冻结。
- 旧版 stored-metadata baseline 不作为当前正式 baseline。
- 跨设备 fusion、跨设备 best selection 没有进入当前正式 baseline。
- NeuroKit Elgendi 已完成 full-cohort candidate 比较，但没有成为当前 frozen primary baseline。
- peak refinement 目前有代码接口和抽样 smoke check，但还没有作为 full-cohort formal primary 结果。

## 计划与疑问

1. 是否处理 Watch 覆盖率低？Earring/Ring 在 unified 规则下被牺牲？
   - 当前 Watch strict RMSSD 覆盖率只有 22.57%。
   -  v1.1 device-wise analysis 说明 Earring/Ring 各自使用不同参数时会更好。
   - 已完成 SciPy vs NeuroKit per-participant diagnosis；Watch 主要瓶颈仍然是 IBI correction ratio，NeuroKit 还增加了 valid IBI failure。

2. 是否做 v1.2 gate-only optimization？
   - 可以只在现有 recomputed channel metrics 上扩展 gate，例如 `valid_ibi >= 0.70/0.75/0.80`、`cv <= 0.30/0.35/0.40`、`corr <= 0.30/0.35`、`sqi >= 0.35/0.40`。
   - 目标是提升 Watch 覆盖率，同时不明显牺牲 Earring/Ring。
   - 仍然必须只在 `training_stride30` 上选 gate，再冻结到 `strict_reference`。

3. 是否使用 NeuroKit 以外的工具？
   - 可以，但要作为明确候选进入同一套 development/freeze 流程。
   - 可考虑 HeartPy、MSPTD、qppgfast、pyPPG 或自定义 SciPy refinement。
   - 不能因为某个工具在 strict_reference 上看起来好就直接替换；必须在 development 上选择并冻结。

5. 是否继续 peak refinement full run？
   - 当前 NeuroKit full-cohort 结果说明“换 detector”没有解决 Watch 覆盖率问题。
   - 如果后续 gate-only 仍失败，再考虑 peak refinement full run。
   - 已有接口是 `evaluate_devicewise_recomputed_peaks_v1_1.py --include-refinement`。

6. 是否需要 subject-level held-out split？
   - 如果论文要声称泛化到新 participant，仍然需要。
   - 当前 `training_stride30` 和 `strict_reference` 是同一批 participant 的不同窗口组织方式，不是 subject-independent test。

## 推荐下一步

先停止继续盲目扩展 detector。下一步建议做 v1.2 gate-only optimization：只读取当前 v1.1 + NeuroKit channel metrics，在 `training_stride30` 上更细地 sweep `valid IBI ratio`、`IBI correction ratio`、`IBI CV` 等 gate，再冻结到 `strict_reference`。

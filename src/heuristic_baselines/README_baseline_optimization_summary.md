# HRV/PPG Baseline 优化汇总

本文是当前 `src/heuristic_baselines/` 目录下的 正式baseline 优化记录。它只总结已经落到当前代码或当前输出目录里的内容，不把想法、抽样试验或已经归档的旧逻辑当作当前正式 baseline。

## 当前结论

- 当前更强的 heuristic comparison 建议新增报告 `formal_v1_2_report_both_channels`；如果需要与早期严格统一规则比较，保留 `formal_v1_unified_baseline` 作为 conservative reference。
- v1.1 扩展候选后没有在 primary unified 边界内超过 v1；它保留为一次候选扩展记录。
- full-cohort NeuroKit Elgendi candidates 已经补算并放入同一个 formal unified freeze 框架比较；结果仍然没有替代当前 SciPy primary baseline。
- v1.2 按新的边界允许每个 `device x channel` 使用不同 peak/bandpass/gate 参数，但 green 和 IR 都必须汇报，不再在通道间只选一个最佳。
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

## Coverage 是怎么算的

本文里的 coverage / 覆盖率不是原始 PPG 信号存在率，也不是 peak detector 的召回率，而是某个 baseline 在指定窗口集合里最终能给出有效 HRV prediction 的比例。

普通 QC baseline 的口径：

```text
coverage = n_valid / n_total

n_total = 当前 dataset role 下该 device 或 device x channel 的窗口数
n_valid = PPG HRV 和 ECG HRV 都是有限值，且该窗口通过 frozen QC gate 的窗口数
```

因此当前 v1/v1.1/v1.2 表里的 training 覆盖率由两步共同决定：

1. detector / bandpass / peak method 是否能从 `ppg_resampled` 产生足够 peaks，并计算出有限的 PPG RMSSD/SDNN。
2. 该窗口是否通过已冻结的 quality gate，例如 `ppg_valid_sample_ratio`、SQI、valid IBI ratio、IBI correction ratio、IBI CV、RMSSD 上限等。

motion threshold sensitivity 的口径略有不同：分母不是完整 `training_stride30`，而是先取同一批 common-motion windows。具体做法是：每个 threshold 先取所有汇报的 `device x channel` 都满足 `accel_motion_mean_mag < threshold` 的 `participant + window_index` 交集，再在这批 common-motion windows 内计算各通道 QC-valid coverage、MAE 和 R。

```text
coverage_within_motion = n_valid_after_QC_inside_common_motion_subset / n_common_motion_windows
```

所以 motion 表中的 coverage 只能解释“在同一批 common-motion 子集内部，baseline 能保留多少窗口”，不能直接和完整 training_stride30 coverage 混为一谈。

noQC ablation 的详细 coverage 口径和结果移到 `noQC/README.md`。简短地说，noQC coverage 衡量 frozen detector/bandpass 能产出有限 HRV 的上限；coverage 上升不一定代表 HRV agreement 变好，仍必须和 MAE、R、bias 一起解释。

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

v1 formal 的 training_stride30 结果，按 RMSSD MAE 从好到差排序：

| 设备 | training RMSSD MAE | training RMSSD R | training 覆盖率 | training SDNN R |
|---|---:|---:|---:|---:|
| Ring | 12.53 ms | 0.446 | 57.93% | 0.712 |
| Watch | 14.18 ms | 0.364 | 22.83% | 0.580 |
| Earring | 16.57 ms | 0.323 | 77.26% | 0.758 |

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

v1.1 device-wise analysis 的 training_stride30 结果，按 RMSSD MAE 从好到差排序：

| 设备 | training RMSSD MAE | training RMSSD R | training 覆盖率 | 解释 |
|---|---:|---:|---:|---|
| Ring | 11.54 ms | 0.495 | 38.57% | 比 unified v1 的 MAE/R 更好，但覆盖率下降 |
| Watch | 14.18 ms | 0.364 | 22.83% | 基本沿用 v1 结果 |
| Earring | 15.40 ms | 0.419 | 66.98% | 比 unified v1 的 MAE/R 更好，但覆盖率下降 |

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

v1.1 formal 的 training_stride30 结果与 v1 formal 相同，按 RMSSD MAE 从好到差排序：

| 设备 | training RMSSD MAE | training RMSSD R | training 覆盖率 |
|---|---:|---:|---:|
| Ring | 12.53 ms | 0.446 | 57.93% |
| Watch | 14.18 ms | 0.364 | 22.83% |
| Earring | 16.57 ms | 0.323 | 77.26% |

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

per-participant diagnosis 说明：

- 现有 `diagnosis_scipy_vs_neurokit_participants` artifact 只保存了 `strict_reference` 诊断表，没有对应的 `training_stride30` participant-level diagnosis。
- 为保持本 summary 的结果口径一致，这里不再列出该 strict 表；如需继续比较 SciPy vs NeuroKit 的 participant-level 波动，应补生成 training_stride30 版本后再汇报。

QC 诊断：

- NeuroKit 在三个设备上都带来更高的 `valid IBI` 失败比例。
- Ring 和 Watch 的主要瓶颈仍然是 `IBI correction ratio`。
- Watch 的 NeuroKit pass-all 约 `20.34%`，primary SciPy pass-all 约 `21.87%`，所以 NeuroKit 没有解决 Watch 覆盖率低的问题。

对应输出：

- `outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/`
- `outputs/formal_v1_1_plus_neurokit_unified_baseline/`
- `outputs/diagnosis_scipy_vs_neurokit_participants/`

## v1.2：允许设备/通道参数不同，并同时报告 green 和 IR

对应文件：

- `formal_freeze_v1_2_report_both_channels.py`
- `formal_freeze_v1_2_device_channel_params.py`

新的科学边界：

- 总体仍使用同一套透明方法族：从 `ppg_resampled` 重新计算 peaks、IBI、SQI/QC 和 HRV。
- 参数选择只使用 `training_stride30`。
- `strict_reference` 只在参数冻结后评估。
- 不做 subject-level held-out split。
- 不训练模型，不做跨设备 fusion，不用 ECG label 做逐窗口选择。
- 允许每个 `device x channel` 使用自己的 peak method / bandpass / IBI correction threshold / QC gate。
- 不允许 green 和 IR 只选一个最佳；两个通道都冻结、都报告。

当前已完成的非-refinement full-cohort 结果使用现有 `v1_1_plus_neurokit_channel_metrics.csv`，候选包含 SciPy 和 NeuroKit，但不含 full refinement。冻结单位为 `device x channel`。

冻结后的 training_stride30 RMSSD 结果，按 RMSSD MAE 从好到差排序。这里的 Coverage 是完整 `training_stride30` common-window 原始分母 `17064` 下的 QC-valid 覆盖率；MAE/R 在各通道 QC-valid 窗口上计算。

| 排名 | 设备 | 通道 | Peak/gate 简写 | RMSSD MAE | RMSSD R | Coverage |
|---:|---|---|---|---:|---:|---:|
| 1 | Ring | `ppg_green` | `bp07_35_prom025_corr02 + sqi030/ibi080/corr020/cv030` | 11.54 ms | 0.495 | 38.60% |
| 2 | Watch | `ppg_green` | `bp07_35_prom025_corr02 + sqi030/ibi090/corr030/cv030` | 14.05 ms | 0.364 | 22.55% |
| 3 | Ring | `ppg_ir` | `bp07_35_prom025_corr02 + sqi050/ibi090/corr030/cv030` | 14.42 ms | 0.358 | 23.73% |
| 4 | Earring | `ppg_ir` | `bp05_40_prom025_corr02 + sqi050/ibi090/corr020/cv030` | 15.35 ms | 0.428 | 66.39% |
| 5 | Earring | `ppg_green` | `bp05_40_prom025_corr02 + sqi030/ibi085/corr020/cv035` | 17.19 ms | 0.414 | 84.22% |
| 6 | Watch | `ppg_ir` | `bp07_35_prom025_corr03 + sqi050/ibi070/corr035/cv035` | 68.85 ms | -0.136 | 20.07% |

同一批冻结结果按 SDNN MAE 从好到差排序：

| 排名 | 设备 | 通道 | Peak/gate 简写 | SDNN MAE | SDNN R | Coverage |
|---:|---|---|---|---:|---:|---:|
| 1 | Earring | `ppg_green` | `bp05_40_prom025_corr02 + sqi030/ibi085/corr020/cv035` | 6.02 ms | 0.928 | 84.22% |
| 2 | Earring | `ppg_ir` | `bp05_40_prom025_corr02 + sqi050/ibi090/corr020/cv030` | 7.51 ms | 0.852 | 66.39% |
| 3 | Ring | `ppg_green` | `bp07_35_prom025_corr02 + sqi030/ibi080/corr020/cv030` | 10.63 ms | 0.796 | 38.60% |
| 4 | Watch | `ppg_green` | `bp07_35_prom025_corr02 + sqi030/ibi090/corr030/cv030` | 20.96 ms | 0.581 | 22.55% |
| 5 | Ring | `ppg_ir` | `bp07_35_prom025_corr02 + sqi050/ibi090/corr030/cv030` | 25.16 ms | 0.592 | 23.73% |
| 6 | Watch | `ppg_ir` | `bp07_35_prom025_corr03 + sqi050/ibi070/corr035/cv035` | 88.38 ms | 0.156 | 20.07% |

解读：

- v1.2 不再把通道选择混成一个 winner-take-all 结果；green 和 IR 的差异直接暴露。
- Earring 的 IR RMSSD MAE 更低，但 green 覆盖率更高且 SDNN R 更高。
- Ring 的 green 明显优于 IR。
- Watch 的 green 仍是可用通道；Watch IR 虽然按 coverage 目标勉强可汇报，但 RMSSD 误差和相关性都很差，不应作为 Watch 的主要结论。
- 到目前为止，v1.2 的最优冻结项仍主要来自 SciPy，NeuroKit 没有在这些 `device x channel` 冻结项里胜出。

各设备、通道的最佳结果总结：

| 设备 | 最佳 RMSSD 通道 | 最佳 training RMSSD 结果 | 最佳 SDNN 通道 | 最佳 training SDNN 结果 | 结论 |
|---|---|---|---|---|---|
| Earring | `ppg_ir` | `15.35 ms` MAE，R `0.428`，覆盖率 `66.39%` | `ppg_green` | `6.02 ms` MAE，R `0.928`，覆盖率 `84.22%` | RMSSD 最佳是 IR；SDNN 和覆盖率最佳是 green，因此 Earring 应同时报告两通道，分别强调不同优势。 |
| Ring | `ppg_green` | `11.54 ms` MAE，R `0.495`，覆盖率 `38.60%` | `ppg_green` | `10.63 ms` MAE，R `0.796`，覆盖率 `38.60%` | green 在 RMSSD、SDNN、R 和 coverage 上都优于 IR，是 Ring 的主要 baseline 通道。 |
| Watch | `ppg_green` | `14.05 ms` MAE，R `0.364`，覆盖率 `22.55%` | `ppg_green` | `20.96 ms` MAE，R `0.581`，覆盖率 `22.55%` | green 是 Watch 唯一可作为 baseline 解读的通道；IR 的 RMSSD/SDNN 都明显失败，应保留汇报但不作为主要结论。 |

参与者之间的 R / coverage 波动：

| 设备 | 通道 | participant RMSSD MAE median [min, max] | participant R median [IQR] | participant coverage median [IQR] | coverage range |
|---|---|---:|---:|---:|---:|
| Watch | `ppg_green` | 10.36 ms [3.89, 27.94] | 0.325 [0.086, 0.399] | 24.57% [8.19, 28.42] | 3.12-55.56% |
| Ring | `ppg_green` | 11.31 ms [7.55, 20.72] | 0.414 [0.260, 0.561] | 36.91% [30.59, 45.96] | 17.40-66.52% |
| Ring | `ppg_ir` | 14.07 ms [7.63, 23.31] | 0.228 [-0.016, 0.490] | 21.34% [15.93, 32.23] | 11.95-52.21% |
| Earring | `ppg_ir` | 14.84 ms [5.50, 32.38] | 0.282 [0.161, 0.482] | 70.84% [55.91, 85.54] | 29.11-97.54% |
| Earring | `ppg_green` | 16.07 ms [6.34, 33.37] | 0.206 [0.151, 0.532] | 91.38% [81.07, 95.39] | 34.03-100.00% |
| Watch | `ppg_ir` | 73.99 ms [46.71, 99.08] | -0.116 [-0.197, 0.302] | 16.56% [14.16, 24.49] | 5.10-41.71% |

说明：

- coverage 的 participant 间波动很大，尤其 Watch green 的范围是 `3.12-55.56%`，Watch IR 是 `5.10-41.71%`；因此 Watch 的总覆盖率低不是均匀下降，而是部分 participant 可用、部分 participant 几乎不可用。
- R 也有明显 participant 差异。Ring green 的 median R 最高，为 `0.414`；Watch IR 的 median R 为 `-0.116`，即使部分 participant 可过 gate，整体仍应视为失败通道。
- Earring green 的 participant median coverage 很高（`91.38%`），但 median R 低于 Earring IR；因此它的优势是覆盖率，不是 RMSSD agreement。

motion threshold sensitivity：

当前 v1.2 frozen baseline 已在 `training_stride30` 上按 `accel_motion_mean_mag < 0.1 / 0.2 / 0.5 / 1.0` 重算。阈值过滤已改为 common-window intersection：每个 threshold 先取所有汇报的 `device x channel` 都满足 motion 条件的同一批 `participant + window_index`，分母是 common-motion windows。

| 设备 | 通道 | Common motion windows `<0.1 -> <1.0` | Coverage `<0.1 -> <1.0` | RMSSD MAE `<0.1 -> <1.0` | RMSSD R `<0.1 -> <1.0` | 影响 |
|---|---|---:|---:|---:|---:|---|
| Earring | `ppg_green` | 118 -> 13201 | 97.46% -> 86.14% | 9.17 -> 17.08 ms | 0.762 -> 0.436 | common low-motion 子集下误差最低；阈值放宽后样本更充分但 MAE 上升。 |
| Earring | `ppg_ir` | 118 -> 13201 | 96.61% -> 70.96% | 9.34 -> 15.54 ms | 0.660 -> 0.442 | common low-motion 子集下接近 green；阈值放宽后仍是 Earring 的主要 RMSSD 通道。 |
| Ring | `ppg_green` | 118 -> 13201 | 64.41% -> 42.95% | 12.25 -> 11.06 ms | 0.733 -> 0.548 | 在 common-motion 横向比较中保持 Ring 最佳；严格阈值样本很少。 |
| Ring | `ppg_ir` | 118 -> 13201 | 56.78% -> 25.64% | 13.94 -> 14.18 ms | 0.270 -> 0.474 | coverage 和 RMSSD agreement 均弱于 Ring green；严格阈值下 R 不稳定。 |
| Watch | `ppg_green` | 118 -> 13201 | 41.53% -> 24.10% | 12.26 -> 14.70 ms | 0.403 -> 0.431 | Watch 的可用通道；common-motion 下 coverage 仍明显低于 Earring/Ring green。 |
| Watch | `ppg_ir` | 118 -> 13201 | 61.02% -> 18.47% | 47.42 -> 77.23 ms | -0.515 -> -0.039 | 即使使用 common low-motion 子集，IR 仍明显失败。 |

结论：

- `<0.1` 最能体现 common low-motion 下的最好情况，但 common windows 只有 `118`，代表性有限。
- `<0.2` 是较平衡的 common low-motion sensitivity：common windows 增至 `1367`，Ring green R `0.731`，Watch green R `0.616`。
- `<0.5` 和 `<1.0` 更接近 common low-to-moderate motion/full training 行为，窗口数多，但 Ring/Watch 的 motion 子集内 coverage 仍明显低于 Earring。
- motion threshold 主要解释低运动 common subset 中各通道表现如何变化；Watch IR 无论如何都失败。

对应输出：

- `outputs/formal_v1_2_report_both_channels/`
- `outputs/formal_v1_2_device_channel_params/`
- `archive/legacy_current_best_baseline/motion_tests/v1_2_motion_threshold_sensitivity_training_stride30.md`
- `archive/legacy_current_best_baseline/motion_tests/v1_2_motion_threshold_sensitivity_training_stride30.csv`
- `archive/legacy_current_best_baseline/motion_tests/v1_2_motion_threshold_sensitivity_participant_training_stride30.csv`

补充：

- `formal_freeze_v1_2_device_channel_params.py` 早先跑过一个“每设备选一个通道”的版本；它只作为中间诊断，不作为当前推荐报告边界。
- peak refinement full run 本轮尝试过使用 `evaluate_devicewise_recomputed_peaks_v1_1.py --include-refinement` 补算，但该脚本是一次性全量写盘，约 30 分钟仍未产生可用输出，已中断。下一步应改成按 `role + participant` 缓存的 refinement 补算，再并入 `formal_freeze_v1_2_report_both_channels.py` 重新冻结。

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
  v1.2 这轮尝试了 full refinement 一次性补算，但因运行时间过长且无中间缓存已中断；不把 refinement 当作正式结论。

## 推荐下一步

1. 写一个按 `role + participant` 缓存的 refinement 补算脚本，避免全量任务中断后重来。
2. 将 refinement channel metrics 与当前 SciPy + NeuroKit full-cohort metrics 合并。
3. 用 `formal_freeze_v1_2_report_both_channels.py` 重新冻结 `device x channel` 参数。
4. 若 Watch IR 仍然显著失败，将其作为负面通道结果如实报告，而不是用 best-channel selection 隐藏。

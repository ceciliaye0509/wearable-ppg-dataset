# Full-cohort NeuroKit Candidate 比较结果

本文记录本轮按顺序执行的 3 个步骤：

1. 补算 full-cohort NeuroKit candidates。
2. 将 SciPy + NeuroKit 放入同一个 formal unified freeze 框架比较。
3. 生成 per-participant error diagnosis。

本记录是实验 README，不是论文正文。

## 运行边界

- 参数选择仍然只使用 `training_stride30`。
- `strict_reference` 只用于冻结后的评估和诊断。
- 不做跨设备 fusion。
- 不做跨设备 best selection。
- 不使用 ECG label 逐窗口选择通道或调参。
- NeuroKit candidate 只作为 detector candidate 加入候选池，不自动替代当前 primary baseline。

## 新增/使用的代码

| 文件 | 作用 |
|---|---|
| `evaluate_neurokit_fullcohort_cached_v1_1.py` | 按 `role + participant` 缓存补算 NeuroKit Elgendi full-cohort candidates，并与已有 v1.1 SciPy channel metrics 合并 |
| `formal_freeze_v1_1_unified_baseline.py` | 复用 formal freeze 框架，对 SciPy + NeuroKit 统一候选进行 training-only freeze |
| `diagnose_scipy_vs_neurokit_participants.py` | 比较 primary SciPy 和最佳 NeuroKit unified candidate 的 participant/device 级误差与 QC 失败原因 |

`evaluate_devicewise_recomputed_peaks_v1_1.py` 也做了一个小改动：当传入 `--base-channel-metrics-csv` 时，会读取已有 `peak_method` 名称并跳过已存在的方法，只补算缺失 candidates。这个改动用于避免重算已有 v1.1 SciPy 指标。

## 输出目录

| 目录 | 含义 |
|---|---|
| `outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/` | v1.1 SciPy 指标 + full-cohort NeuroKit 指标；包含可恢复缓存 |
| `outputs/formal_v1_1_plus_neurokit_unified_baseline/` | SciPy + NeuroKit 进入同一 formal unified freeze 后的结果 |
| `outputs/diagnosis_scipy_vs_neurokit_participants/` | primary SciPy vs best NeuroKit unified 的 participant/device 级诊断 |

## Step 1：full-cohort NeuroKit candidate

补算内容：

- `nk_elgendi_raw_cleanonly_corr02`
- `nk_elgendi_bp07_35_doubleclean_corr02`

缓存策略：

- 每个 `role + participant` 单独写入 `.cache/neurokit_rows/`。
- 共有 32 个缓存单元：`strict_reference` 16 个 participant，`training_stride30` 16 个 participant。
- 这样中断后可以继续，不需要从头重跑。

合并后的主文件：

- `outputs/all_participants_devicewise_baseline_v1_1_plus_neurokit/v1_1_plus_neurokit_channel_metrics.csv`

## Step 2：formal unified freeze comparison

加入 full-cohort NeuroKit 后，formal unified freeze 仍然选中当前 SciPy v1 规则：

`scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200`

也就是说：**NeuroKit 没有替代当前 primary SciPy baseline**。

### Training 上满足 coverage 约束的 Top unified candidates

| 排名 | 方法 | Detector | Strategy | mean RMSSD MAE | mean R | mean coverage | min coverage |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200` | SciPy | fixed channel | 14.43 ms | 0.378 | 52.67% | 22.83% |
| 2 | `scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi035_ibi08_corr03_cv35_rmssd200` | SciPy | fixed channel | 14.44 ms | 0.376 | 52.71% | 22.87% |
| 3 | `scipy_bp07_35_prom030_corr02__fixed_best_channel_per_device__gate_sqi035_ibi08_corr03_cv35_rmssd200` | SciPy | fixed channel | 14.76 ms | 0.368 | 52.33% | 22.83% |
| 4 | `scipy_bp07_35_prom030_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200` | SciPy | fixed channel | 14.76 ms | 0.368 | 52.31% | 22.79% |
| 5 | `nk_elgendi_bp07_35_doubleclean_corr02__fixed_best_channel_per_device__gate_sqi04_ibi09_corr02_cv25_rmssd200` | NeuroKit | fixed channel | 15.09 ms | 0.377 | 48.13% | 20.48% |

解读：

- 最佳 NeuroKit unified candidate 比 primary SciPy 的 training mean RMSSD MAE 高约 `0.66 ms`。
- NeuroKit 的 mean coverage 也更低：`48.13%` vs `52.67%`。
- NeuroKit mean R 与 SciPy 接近，但没有超过 SciPy。

## Step 3：per-participant diagnosis

诊断比较对象：

| 标签 | 方法 |
|---|---|
| `primary_scipy` | `scipy_bp07_35_prom025_corr02__fixed_best_channel_per_device__gate_sqi04_ibi08_corr03_cv30_rmssd200` |
| `best_neurokit_unified` | `nk_elgendi_bp07_35_doubleclean_corr02__fixed_best_channel_per_device__gate_sqi04_ibi09_corr02_cv25_rmssd200` |

### strict_reference 参与者级均值

| 设备 | 方法 | mean MAE | median MAE | mean R | mean coverage |
|---|---|---:|---:|---:|---:|
| Earring | primary SciPy | 17.76 ms | 16.44 ms | 0.262 | 78.38% |
| Earring | best NeuroKit | 18.18 ms | 16.52 ms | 0.363 | 75.11% |
| Ring | primary SciPy | 11.99 ms | 11.66 ms | 0.386 | 56.88% |
| Ring | best NeuroKit | 12.10 ms | 12.07 ms | 0.369 | 48.15% |
| Watch | primary SciPy | 12.74 ms | 10.67 ms | 0.297 | 23.32% |
| Watch | best NeuroKit | 21.33 ms | 12.81 ms | 0.222 | 21.70% |

解读：

- Earring：NeuroKit 的 participant-level mean R 更高，但 MAE 和 coverage 略差。
- Ring：SciPy 的 MAE、R、coverage 都略好。
- Watch：NeuroKit 明显更差，主要受少数 participant 的大误差影响。

### strict_reference 平均 QC 失败原因

| 设备 | 方法 | pass all | fail SQI | fail valid IBI | fail correction | fail IBI CV |
|---|---|---:|---:|---:|---:|---:|
| Earring | primary SciPy | 78.38% | 2.90% | 6.08% | 17.80% | 3.99% |
| Earring | best NeuroKit | 75.11% | 2.90% | 10.20% | 22.55% | 5.23% |
| Ring | primary SciPy | 56.88% | 0.00% | 3.77% | 42.77% | 0.35% |
| Ring | best NeuroKit | 48.15% | 0.00% | 11.88% | 51.32% | 4.46% |
| Watch | primary SciPy | 21.87% | 0.00% | 8.28% | 77.40% | 1.44% |
| Watch | best NeuroKit | 20.34% | 0.00% | 32.41% | 79.16% | 6.73% |

解读：

- NeuroKit 在三个设备上都带来了更高的 `valid IBI` 失败比例。
- Ring 和 Watch 的主要瓶颈仍然是 `IBI correction ratio`。
- Watch 的 NeuroKit 结果额外出现很高的 valid IBI failure，说明 NeuroKit 对 Watch 的峰序列稳定性没有改善。

## 当前结论

1. **primary baseline 不应换成 NeuroKit。**
   - full-cohort formal freeze 仍选择 SciPy。
   - 最佳 NeuroKit unified candidate 在 training 上 MAE 和 coverage 都不如 primary SciPy。

2. **NeuroKit 可以作为 stronger reference / analysis 保留。**
   - 它在 Earring 的 participant-level R 上有一点优势。
   - 但这个优势不足以抵消整体 MAE / coverage 的下降。

3. **Watch 覆盖率低不是 NeuroKit 能直接解决的问题。**
   - primary SciPy Watch pass all 约 `21.87%`。
   - best NeuroKit Watch pass all 约 `20.34%`。
   - Watch 主要问题仍然是 IBI correction ratio 过高，NeuroKit 还增加了 valid IBI failure。

## 下一步建议

优先做 v1.2 gate-only optimization，而不是继续扩展 detector：

- 针对 `IBI correction ratio` 和 `valid IBI ratio` 做更细的 training-only gate sweep。
- 重点观察 Watch coverage 能否提升，同时 Earring/Ring 的 MAE/R 不明显变差。
- 仍然保持 primary unified 边界：所有设备同一套处理参数，只允许每个设备固定最佳通道。

暂时不建议立刻做 peak refinement full run。理由是这次 NeuroKit 诊断显示，Watch 的主要问题更像是 IBI 质量门控和异常 IBI 比例，而不是简单换 detector 就能解决。

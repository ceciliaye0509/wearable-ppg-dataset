# Fold-0：qPPG × SegNet/TCN 开发期对照报告

## 研究范围与当前状态

本文是 **fold-0 inner-validation 开发期报告**，不是正式四折交叉验证或 outer-test 结果。它在同一冻结 raw-slot 合同下对照四种模型：

1. raw causal TCN；
2. qPPGFast + causal-TCN 残差模型；
3. raw partner-style SegNet；
4. qPPGFast + partner-style-SegNet 残差模型。

这些开发运行均未加载 outer-test 参与者，也没有启动正式四折。原始 NPZ 下载与 staged raw-slot cache 在本工作之前已完成；本工作没有重新下载或重建 cache，也没有读取、展示或重建 Hugging Face token。

## 运行环境

| 项目 | 内容 | Coverage |
|---|---|---|
| 计算节点 | Purdue A30 节点 `gilbreth-d005` | 不适用（运行环境） |
| 主训练环境 | `/scratch/gilbreth/qiuyue/daily_hrv/envs/water-legacy/bin/python` | 不适用（运行环境） |
| Python | 3.8.20 | 不适用（运行环境） |
| NumPy | 1.24.3 | 不适用（运行环境） |
| PyTorch / CUDA | 2.0.0 / 11.8；CUDA 可用 | 不适用（运行环境） |
| 环境原则 | 原有 `water-legacy` 保持不变；未安装、升级或修改包 | 不适用（运行环境） |
| 仅 qPPG 特征提取 | 使用已有 `anaconda/2024.10-py312` 中的 SciPy 1.13.1，因为 `water-legacy` 不含 SciPy；未修改任何环境 | 不适用（运行环境） |

## 冻结实验合同

| 维度 | 固定设置 | Coverage |
|---|---|---|
| 原始数据源 | `synced_3device_rawaligned_training_v1_stride30_rawslots` | 不适用（数据合同） |
| 已完成缓存 | `artifacts/ml_hrv/rawslot_cache` | 不适用（数据合同） |
| 设备 / PPG 输入 | Earring，green 单通道，100 Hz raw-slot | 不适用（数据合同） |
| 窗口 / 更新 | 5 分钟窗口，30 秒 stride；10 秒 encoder segment | 不适用（数据合同） |
| 标签 | ECG-corrected RMSSD 和 SDNN；仅使用 ECG label QC 通过窗口 | 不适用（数据合同） |
| Fold-0 训练参与者 | P20、P9、P7、P5、P18、P15、P19、P1、P6、P4 | 不适用（非验证指标） |
| Fold-0 inner validation | P10、P12（Earring 共 `n=1,178` 个窗口） | 1,178/1,178 = 100.0% |
| 未加载的 outer test | P8、P11、P3、P13 | 0/1,178 = 0.0%（开发期未读取） |
| 训练单位 | 每 batch 4 个窗口；每 epoch 128 次参数更新 | 不适用（训练设置） |

qPPG 特征 CSV 由同一 raw-slot 合同的 PPG 信号生成，不含 ECG 标签。它含 qPPG RMSSD/SDNN 基线，以及 9 个质量特征：有效性、peak 数、有效 IBI 比例、IBI CV、IBI correction ratio、SQI、有效采样比例、最大缺口和 motion。Fold-0 训练行的 qPPG 有效比例为 75.1%，验证行为 95.1%。所有 qPPG 特征的插补、标准化与无效 qPPG 基线 fallback 中位数均仅在训练行拟合。

## 对原 SegNet 的处理与对照

### 原 partner-style SegNet 路径

`PartnerSegNetEncoder` 是仓库中对伙伴 SegNet 的忠实实现：对每个 10 秒片段的原始 PPG 及其 mask，依次使用三层 stride-2 的 `Conv1d → BatchNorm → ReLU`。`SegNetMeanHead` 对 30 个 segment token 取平均，再预测两个 HRV 输出。原有的宽泛 baseline 配置为 `src/ml_hrv/configs/segnet_rawslot_baseline.json`，使用三种设备和默认 green+IR 输入。

### 为本次公平对照所做的 SegNet 适配

SegNet **主体架构未改变**：仍采用 `direct_architecture: segnet_mean`、token dimension 64、无 timestamp jitter。为使它可与 TCN 在同一合同下对照，配置层面做了以下限制：

- 将输入限定为 Earring + green、100 Hz，而非原先的三设备默认设置；
- 保留 5 分钟窗口和冻结的 fold-0 split；
- 在标准化 log-HRV 空间采用 direct Huber loss，替代 learned heteroscedastic NLL。这是因为此前 direct-TCN gate 出现了不确定性逃逸/近似常数预测风险；
- 使用面向时限的 6 epoch × 128 updates 开发 gate，并允许 early stopping；这不是正式生产训练计划。

### qPPG 残差处理

对 TCN 或 SegNet，qPPG residual 都表示：

```text
最终标准化 log-HRV = 标准化 qPPG 基线 + 神经网络预测的修正量（residual）
```

神经网络仍然直接读取 raw PPG，并额外读取仅来自 PPG 的、按训练集标准化的 qPPG 质量特征及 `qppg_valid` 标记。qPPG 无效窗口并不删除：其 qPPG 基线改用训练集 qPPG 中位数，同时保留无效标记让模型学习 fallback 情况。该设计检验神经网络能否修正生理 qPPG 估计，而非从头学习整个 HRV 计算过程。

## 创建和修改的代码

以下均为仓库相对路径。

| 路径 | 代码处理 / 用途 | Coverage |
|---|---|---|
| `src/ml_hrv/models/encoder.py` | 原有 `PartnerSegNetEncoder`；本次作为伙伴风格 SegNet encoder 使用，主体未修改。 | 不适用（代码清单） |
| `src/ml_hrv/models/direct.py` | 为 `DirectHRVHead` 和 `SegNetMeanHead` 增加可选 qPPG-quality conditioning。 | 不适用（代码清单） |
| `src/ml_hrv/models/pipeline.py` | 增加残差组装：标准化 qPPG 基线 + 神经网络 direct correction。 | 不适用（代码清单） |
| `src/ml_hrv/data/qppg.py` | **新建。** 加载按 participant/window/device/channel 键控的 qPPG CSV，并实现训练集专用的中位数插补与标准化。 | 不适用（代码清单） |
| `src/ml_hrv/data/dataset.py` | 将 qPPG 字段 join 到 raw-slot sample，并在 batch 中保留 qPPG validity/base。 | 不适用（代码清单） |
| `src/ml_hrv/config.py` | 新增 qPPG 特征表和 residual 开关配置。 | 不适用（代码清单） |
| `src/ml_hrv/training/losses.py` | 新增开发期 Huber direct-loss 选项。 | 不适用（代码清单） |
| `src/ml_hrv/training/trainer.py` | 用 train-only target scaler 编码 qPPG base，并将 qPPG scaler state 保存进 checkpoint。 | 不适用（代码清单） |
| `src/ml_hrv/training/checkpointing.py` | 保存可选的 qPPG feature-scaler state。 | 不适用（代码清单） |
| `src/ml_hrv/evaluation/evaluator.py` | 在评估时喂入 qPPG residual 输入，并写出 qPPG base/validity/residual 字段。 | 不适用（代码清单） |
| `src/ml_hrv/scripts/train.py` | 仅用训练行拟合 qPPG scaler，再提供给 validation 数据。 | 不适用（代码清单） |
| `src/ml_hrv/scripts/evaluate_inner_validation.py` | 在 inner-validation 前重新计算并核验 train-only target/qPPG scaler。 | 不适用（代码清单） |
| `src/ml_hrv/scripts/build_qppgfast_features.py` | **此前新建。** 从 frozen raw-slot 构建不含标签的 qPPGFast 特征表。 | 不适用（代码清单） |
| `src/ml_hrv/scripts/analyze_qppg_residual.py` | **新建。** 只读地按 all/valid/invalid qPPG 窗口审计 qPPG base 与 residual。 | 不适用（代码清单） |
| `src/ml_hrv/configs/teacher_green_earring_causal_tcn_huber_dev.json` | Raw TCN 开发 gate 配置。 | 不适用（代码清单） |
| `src/ml_hrv/configs/teacher_green_earring_causal_tcn_qppg_residual_dev.json` | qPPG + TCN residual 配置。 | 不适用（代码清单） |
| `src/ml_hrv/configs/teacher_green_earring_segnet_huber_dev.json` | **新建。** Raw SegNet fold-0 开发 gate 配置。 | 不适用（代码清单） |
| `src/ml_hrv/configs/teacher_green_earring_segnet_qppg_residual_dev.json` | **新建。** qPPG + SegNet residual fold-0 开发 gate 配置。 | 不适用（代码清单） |

相关开发提交：`e3ce855`（Huber gate）、`3198888`（inner-validation evaluator）、`1d29ffa`（qPPG residual 路径）、`76b31f9`（SegNet gates）和 `dfc07bb`（qPPG 分层审计）。

## 结果：fold-0 inner validation

train-median baseline 指的是：对每个验证窗口，一律输出 fold-0 训练标签的中位数。它不是伙伴模型，也不读取 PPG。

本报告的 **Coverage** 一律指“该行指标实际纳入评估的窗口数 / fold-0 全部 ECG-QC inner-validation 窗口数（1,178）× 100%”。例如 qPPG-valid 的 coverage 为 `1,120 / 1,178 × 100% = 95.1%`。它**不是** raw PPG 的 slot 有效采样率；后者的单窗口算式为 `mean(mask)`，未用于这些汇总指标表。

| 模型 | Coverage | RMSSD MAE (ms) | RMSSD r | RMSSD R² | RMSSD 预测 SD (ms) | SDNN MAE (ms) | SDNN r | SDNN R² | SDNN 预测 SD (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train-median baseline | 1,178/1,178 = 100.0% | 7.620 | n/a | ~0 | 0.000 | 15.301 | n/a | ~0 | 0.000 |
| Raw TCN（Huber gate） | 1,178/1,178 = 100.0% | 7.371 | -0.104 | -0.142 | 0.026 | 14.942 | 0.106 | -0.015 | 0.038 |
| qPPG + TCN residual | 1,178/1,178 = 100.0% | 14.197 | 0.233 | -3.756 | 14.684 | 7.067 | 0.889 | 0.767 | 16.812 |
| Raw SegNet（Huber gate） | 1,178/1,178 = 100.0% | 7.373 | -0.066 | -0.171 | 2.853 | 14.215 | 0.217 | 0.040 | 4.702 |
| qPPG + SegNet residual | 1,178/1,178 = 100.0% | 13.310 | 0.231 | -3.443 | 14.599 | **6.830** | **0.889** | **0.778** | 15.872 |

作为尺度参考，验证标签 SD：RMSSD 为 8.135 ms，SDNN 为 20.385 ms。

## 结果分析

1. **Raw direct 模型不足。** Raw TCN 对两种输出几乎都是常数；raw SegNet 虽有更多变化，但 SDNN 信号仍弱，RMSSD 没有超过 median baseline。
2. **qPPG augmentation 对 SDNN 很有效。** qPPG + SegNet 的 SDNN 最好（MAE 6.830 ms，`r=0.889`，`R²=0.778`），相比 train-median baseline 的 MAE 约降低 55%。qPPG + TCN 很接近（MAE 7.067 ms）。两者差异仅 0.237 ms；一个 inner-validation split 不足以宣称架构优胜。
3. **当前 qPPG residual 设计不适用于 RMSSD。** 两个 qPPG residual 模型的 RMSSD R² 均明显为负，预测 SD 约 14.6 ms，明显高于 8.1 ms 的真实 target SD。这是 RMSSD correction 过度波动，而不是此前“预测几乎不变”的坍缩。
4. **不能将 RMSSD 和 SDNN 平均成一个总分。** 这样会掩盖 SDNN 的明显收益与 RMSSD 的失败；两者必须分别报告。

## RMSSD 专用诊断：train-only 校准 + qPPG-valid gate

由于 RMSSD 是研究重点，进行了一个无神经网络、无验证标签参与拟合的诊断。步骤如下：

1. 仅在 fold-0 训练参与者的 `qppg_valid=true` 窗口上拟合 log-affine 校准：`log(ECG RMSSD) = a × log(qPPG RMSSD) + b`；
2. 将该校准应用于 P10/P12 的 qPPG-valid 窗口；
3. 对 qPPG-invalid 窗口，输出训练集 **ECG RMSSD 标签中位数**，即 train-only supervised fallback；
4. 验证标签只用于最终评估，未参与 slope、intercept 或 fallback 的拟合。

训练集中用于校准的有效窗口数为 5,112，拟合结果为 `a=0.2760`、`b=2.9902`，训练集 ECG RMSSD fallback 中位数为 61.964 ms。斜率明显小于 1，表示原始 qPPG RMSSD 的变化幅度过大，需要向训练集典型范围收缩。

| RMSSD 方法 | 评估范围 | Coverage | MAE (ms) | r | R² | 预测 SD (ms) |
|---|---|---:|---:|---:|---:|---:|
| 原始 qPPG base | qPPG-valid（n=1,120） | 1,120/1,178 = 95.1% | 19.072 | 0.544 | -6.446 | 22.786 |
| Train-only 校准 qPPG | qPPG-valid（n=1,120） | 1,120/1,178 = 95.1% | 7.120 | 0.507 | -0.191 | 6.589 |
| 校准 qPPG + valid gate | 全部验证窗口（n=1,178） | 1,178/1,178 = 100.0% | **7.114** | 0.497 | -0.203 | 6.430 |
| Train-median baseline | 全部验证窗口（n=1,178） | 1,178/1,178 = 100.0% | 7.620 | n/a | ~0 | 0.000 |

qPPG-invalid 的 58 个窗口采用训练 ECG RMSSD 中位数 fallback 后，MAE 为 6.992 ms；该组的相关系数不定义，因为预测为常数。

该诊断将总体 RMSSD MAE 降至 7.114 ms，优于 train-median、raw TCN 和 raw SegNet，但 R² 仍为负。因此它是当前最合理的 **RMSSD 专用 baseline**，不能被表述为已经解决 RMSSD 的窗口间变异预测。

诊断输出位置：

```text
artifacts/ml_hrv/dev_tcn_qppg_residual_fold0/inner_validation/rmssd_trainonly_calibration_gate.json
```

## RMSSD 专用开发实验：受限 SegNet residual（负结果）

为检验“是否只是此前 residual 太自由”这一问题，又只在 fold-0 进行了一个 RMSSD 专用的开发实验。它仍未加载 outer test，也未启动其他 fold。与前述四臂对照相比，本实验的限制更严格：

1. 仅拟合 RMSSD；SDNN 不参与损失或模型选择；
2. qPPG-valid 窗口的 base 使用同一套仅由训练参与者拟合的 log-affine 校准；
3. qPPG-invalid 窗口强制输出训练 ECG RMSSD 中位数，不允许神经网络修正；
4. qPPG-valid 窗口的 residual 采用 `tanh` 限幅；上限是训练有效行中校准误差绝对值的 log-space 第 90 分位数（0.3091），不使用验证标签；
5. 使用伙伴风格的 SegNet encoder，6 epoch × 128 updates；最佳 validation Huber loss 出现在 epoch 4（0.2933）。

结果如下。校准 qPPG + gate 的数值来自上一节完全相同的 fold-0、训练集专用校准诊断，作为该实验必须超过的参考线。

| RMSSD 方法 | Coverage | MAE (ms) | RMSE (ms) | bias (ms) | r | R² | 预测 SD (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train-median baseline | 1,178/1,178 = 100.0% | 7.620 | 8.978 | 3.805 | n/a | -0.219 | 0.000 |
| Train-only 校准 qPPG + valid gate | 1,178/1,178 = 100.0% | **7.114** | — | — | **0.497** | -0.203 | 6.430 |
| 校准 qPPG + valid-gated bounded SegNet residual | 1,178/1,178 = 100.0% | 8.191 | 10.185 | 3.161 | 0.018 | -0.569 | 5.408 |

这不是“模型还需要更多 epoch”的信号：validation Huber 在 epoch 4 已达到最低，随后 epoch 5–6 恶化；而且即使 residual 已限幅，RMSSD 排序仍从校准 qPPG 的 `r=0.497` 几乎降为零。该模型也劣于不读 PPG 的 train-median baseline（8.191 > 7.620 ms）。

**开发期结论：** 对 RMSSD，当前证据不支持在校准 qPPG 基线上增加 SegNet residual；也没有理由继续优先尝试 TCN residual。应把“train-only qPPG 校准 + qPPG-valid gate + invalid 时训练集 median fallback”保留为当前 RMSSD 方法。后续若扩大到其他 fold，应验证这个冻结的无神经 residual baseline 是否稳定，而不是为 RMSSD 启动 SegNet/TCN residual 的正式四折训练。

本实验输出位置：

```text
artifacts/ml_hrv/dev_segnet_qppg_rmssd_bounded_fold0/inner_validation_metrics.json
artifacts/ml_hrv/dev_segnet_qppg_rmssd_bounded_fold0/inner_validation_window_predictions.csv
artifacts/ml_hrv/dev_segnet_qppg_rmssd_bounded_fold0/checkpoints/fold_0_rmssd_bounded.pt
```

## 当前决策与下一步

- **不要**启动 RMSSD neural-residual 的正式四折训练，也不要为 RMSSD 继续优先尝试 TCN residual。
- 保留本次四个 fold-0 开发输出，并运行只读 `analyze_qppg_residual.py`，比较全部 / qPPG-valid / qPPG-invalid 窗口。
- 诊断已表明：RMSSD 的主要问题是 valid qPPG 的尺度/偏差，以及 neural correction 的跨参与者泛化失败；invalid fallback 不是主要瓶颈。
- 后续 RMSSD 工作应先冻结并验证“校准 qPPG + qPPG-valid gate”这一无神经 residual baseline。任何后续四折决定都必须继续使用同一冻结合同，并分别报告 RMSSD 与 SDNN。

## Artifact 位置

| 变体 | Coverage | 运行目录 | Inner-validation 输出 |
|---|---:|---|---|
| Raw TCN | 1,178/1,178 = 100.0% | `artifacts/ml_hrv/dev_tcn_huber_fold0` | `inner_validation/inner_validation_metrics.json`、`inner_validation/window_predictions.csv` |
| qPPG + TCN | 1,178/1,178 = 100.0% | `artifacts/ml_hrv/dev_tcn_qppg_residual_fold0` | `inner_validation/inner_validation_metrics.json`、`inner_validation/window_predictions.csv` |
| Raw SegNet | 1,178/1,178 = 100.0% | `artifacts/ml_hrv/dev_segnet_raw_fold0` | `inner_validation/inner_validation_metrics.json`、`inner_validation/window_predictions.csv` |
| qPPG + SegNet | 1,178/1,178 = 100.0% | `artifacts/ml_hrv/dev_segnet_qppg_residual_fold0` | `inner_validation/inner_validation_metrics.json`、`inner_validation/window_predictions.csv` |
| RMSSD 校准 qPPG + bounded SegNet（开发负结果） | 1,178/1,178 = 100.0% | `artifacts/ml_hrv/dev_segnet_qppg_rmssd_bounded_fold0` | `inner_validation_metrics.json`、`inner_validation_window_predictions.csv` |

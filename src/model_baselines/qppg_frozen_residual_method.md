# qPPGfast + Frozen PPG Embedding 的 HRV 校正方法

## 1. 方法定位

这项实验不是用机器学习替换 qPPGfast，而是把 qPPGfast 作为主要的生理学基线，再测试 frozen PPG embedding 是否能够解释并修正 qPPGfast 的剩余误差。

核心问题是：

> 当 qPPGfast 已经能够较好地估计 5-minute SDNN/RMSSD 时，预训练 PPG encoder 提取的波形形态信息，能否进一步判断 heuristic 在哪些窗口会高估或低估 HRV？

因此，qPPGfast 的输出始终保留在最终预测中，机器学习模型只负责进行保守校正。

## 2. 整体流程

对于同一个 5-minute PPG window，同时运行两条支路：

```text
5-minute PPG window
├── qPPGfast
│   ├── heuristic SDNN / RMSSD
│   ├── mean / median IBI
│   ├── detected peak count
│   ├── valid IBI count
│   ├── rejected IBI ratio
│   └── signal-quality features
│
└── Frozen PulsePPG or PaPaGei encoder
    └── fixed-dimensional waveform embedding

qPPG features + frozen embedding
→ leakage-safe Ridge
→ predicted qPPG residual
→ corrected SDNN / RMSSD
```

Frozen 表示预训练 encoder 的参数不更新。它只负责将 PPG waveform 转换为 embedding，只有下游 Ridge 会使用训练参与者的数据进行拟合。

## 3. Ridge 学习什么？

对于每个训练窗口，首先计算 qPPGfast 相对于 ECG reference 的误差：

```text
SDNN residual  = ECG SDNN  − qPPG SDNN
RMSSD residual = ECG RMSSD − qPPG RMSSD
```

例如：

```text
qPPG SDNN = 55 ms
ECG SDNN  = 62 ms
Residual  = +7 ms
```

这表示 qPPGfast 在该窗口低估了 7 ms。

Ridge 使用 qPPG-derived features 和 frozen embedding 预测 residual：

```text
[qPPG features, frozen embedding] → predicted residual
```

测试时的最终输出为：

```text
Final SDNN  = qPPG SDNN  + predicted SDNN residual
Final RMSSD = qPPG RMSSD + predicted RMSSD residual
```

即：

$$
\widehat{y}_{\mathrm{final}}
=
y_{\mathrm{qPPG}}
+
f_{\mathrm{Ridge}}(x_{\mathrm{qPPG}}, z_{\mathrm{frozen}}).
$$

其中 $x_{\mathrm{qPPG}}$ 是 heuristic 和质量特征，$z_{\mathrm{frozen}}$ 是预训练 encoder 的 embedding。

## 4. 为什么使用 Ridge？

Frozen embedding 通常有数百个维度，而且不同维度可能高度相关。普通线性回归容易学到很大的系数并过拟合训练参与者。

Ridge 的训练目标为：

$$
\sum_i (r_i-\widehat r_i)^2
+
\alpha\sum_j w_j^2,
$$

其中第一项要求预测 residual 接近真实 residual，第二项限制模型系数过大。

- 较小的 $\alpha$：允许较灵活的 correction，但更容易过拟合。
- 较大的 $\alpha$：产生更保守的 correction。
- 极大的 $\alpha$：模型接近不使用 embedding，只输出很小或接近平均值的 correction。

这适合当前任务，因为 qPPGfast 已经是较强 baseline；我们的目标是测试 embedding 是否提供增量信息，而不是让复杂模型推翻 heuristic 预测。

## 5. 数据如何对齐？

qPPGfast 结果、frozen embedding 和 ECG label 必须按照以下标识严格合并：

```text
participant + window_index + device + channel
```

每一行必须代表完全相同的 PPG window。不能只依赖 CSV 行顺序，也不能把不同设备或通道的输出混合。

推荐的建模字段包括：

```text
participant
window_index
device
channel
qppg_sdnn_ms
qppg_rmssd_ms
mean_ibi_ms
median_ibi_ms
ibi_std_ms
peak_count
valid_ibi_count
rejected_ibi_ratio
valid_sample_ratio
embedding_0 ... embedding_511
ecg_sdnn_corrected_ms
ecg_rmssd_corrected_ms
```

如果 qPPGfast 目前没有输出全部中间特征，第一版最低限度可以使用：

```text
qPPG SDNN + qPPG RMSSD + available quality features + frozen embedding
```

## 6. Leakage-safe nested LOSO

正式评估采用 participant-level outer LOSO。

以 P1 为测试参与者为例：

1. P1 的所有窗口只用于最终测试。
2. 其余 15 名参与者构成 outer-training data。
3. 在 outer-training participants 内使用 participant-grouped inner CV 选择 Ridge alpha。
4. StandardScaler 只能在当前 training partition 上拟合。
5. 选定 alpha 后，使用全部 outer-training participants 重新拟合 scaler 和 Ridge。
6. 最后在完全未见过的 P1 上预测 residual。

禁止以下操作：

- 使用全部 16 名参与者拟合 scaler；
- 使用测试参与者选择 alpha；
- 根据测试结果选择 correction strength；
- 在不同方法使用不同窗口时直接比较 MAE。

## 7. 必须完成的四组对照

| 实验 | 输入 | 目的 |
|---|---|---|
| qPPG-only | qPPGfast SDNN/RMSSD | 强 heuristic baseline |
| Embedding-only | Frozen embedding | 检查预训练表征单独包含多少 HRV 信息 |
| Feature fusion | qPPG features + embedding | 直接预测 ECG SDNN/RMSSD |
| Residual correction | qPPG features + embedding | 只预测 qPPGfast 的剩余误差 |

最重要的比较是：

```text
Residual correction vs qPPG-only
```

如果 residual correction 没有在相同 held-out windows 上稳定优于 qPPG-only，就不能认为 frozen embedding 提供了有用的增量信息。

## 8. 评估指标

对 SDNN 和 RMSSD 分别报告：

- participant-fold MAE mean ± std；
- participant-fold R² mean ± std；
- participant-fold Pearson r mean ± std；
- pooled held-out-window MAE、R² 和 Pearson r；
- true/predicted standard deviation；
- 每个 fold 选择的 Ridge alpha；
- native coverage；
- common-window coverage 和 common-window metrics。

同时建议报告相对于 qPPG-only 的 paired improvement：

```text
ΔMAE = corrected MAE − qPPG-only MAE
```

因此：

- $\Delta\mathrm{MAE}<0$ 表示 correction 改善；
- $\Delta\mathrm{MAE}>0$ 表示 correction 使结果变差。

## 9. 如何解释最终结果？

### 如果 residual correction 稳定改善

说明 frozen embedding 捕捉到了 qPPGfast 未显式建模的波形形态、信号质量或设备差异，可作为 heuristic 的补充信息。

### 如果只改善 SDNN、不改善 RMSSD

说明 embedding 可能捕捉到较慢或全局的 variability，但没有充分保留 RMSSD 所依赖的相邻 beat-to-beat variation。

### 如果 pooled metrics 改善、participant-fold metrics 不改善

模型可能主要学习了参与者之间的平均差异，而没有稳定改善新参与者内部的窗口变化。

### 如果 correction 使结果变差

说明 embedding 没有提供可靠的增量信息，或 Ridge 学到了 participant/domain-specific bias。此时应保留 qPPGfast 作为最终 baseline，而不是因为 ML 更复杂就采用 corrected model。

## 10. 方法名称建议

英文可使用：

**Physiology-Informed Frozen-Embedding Residual Correction**

或更具体地写为：

**qPPGfast + Frozen PulsePPG Residual Ridge**

论文中的简短描述：

> We retained qPPGfast as the physiological baseline and trained a regularized linear model to predict its residual error from heuristic quality features and frozen PPG embeddings. All preprocessing, scaling, and hyperparameter selection were performed within participant-grouped training folds.

## 11. 当前实验优先级

1. 确认 qPPGfast prediction CSV 的字段及窗口标识。
2. 将 qPPGfast 结果与现有 frozen embedding 按 window key 对齐。
3. 先实现 Ridge residual correction，不使用 MLP，也不 fine-tune encoder。
4. 在单个 outer fold 上做 smoke test，确认没有 leakage 和行错位。
5. 运行完整 16-fold LOSO。
6. 与 qPPG-only、embedding-only 和 direct feature fusion 进行 paired comparison。


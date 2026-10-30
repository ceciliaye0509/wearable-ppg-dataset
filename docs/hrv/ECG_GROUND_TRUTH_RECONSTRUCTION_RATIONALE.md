# 为什么需要重建 ECG RMSSD ground truth 标签

## 结论

当前 ECG 标签是 **经 IBI correction 的、由 Pan–Tompkins 产生的参考标签**。

原始 ECG 仍然完整保留；因此建议是重建由 ECG 派生出的 R-peak、RR 和 HRV
标签，而不是丢弃数据或重新采集数据。

这一判断主要来自**心搏序列和相邻心搏定位**的错误，而不只是一个恒定的全局
时间延迟。

## 如何发现问题

我们进行了独立、盲法的 raw ECG 审阅：

- 审阅范围为发布训练队列中 `P1`、`P5`、`P13`、`P15` 四位参与者的 16 个
  五分钟窗口。
- 每个窗口抽取两个 20 秒片段（65–85 s、215–235 s），总计 **32 个片段**。
- 审阅者查看 raw ECG 并逐个点击 R apex；界面不显示 detector 标记、
  预测 peak 或 RMSSD。
- 得到人工 R apex。Pan–Tompkins 在完整五分钟窗口上运行，再只在盲标片段内与人工 R apex比较。
- detector 与人工共识以预先设定的 150 ms 容差进行一对一事件匹配。该容差仅
  用于事件计数，**不表示 150 ms 的峰位偏差对 RMSSD 是可接受的**。

审阅材料和结果分析脚本位于：

- `src/heuristic_baselines/outputs/blinded_ecg_rpeak_review_v2_training_cohort/`
- `src/heuristic_baselines/analyze_blinded_rpeak_annotations.py`

## 量化结果

### 两位人工审阅者的内部一致性很高

| 指标 | 结果 |
|---|---:|
| 审阅者 1 点击数 | 685 |
| 审阅者 2 点击数 | 682 |
| 匹配后的人共识 R apex 数 | 680 |
| 相对于平均点击数的一致率 | 99.49% |
| 两位审阅者的绝对位置差中位数 | 0 ms |

因此，下面的 detector 比较结果并非由两位审阅者对可见 raw 波形 apex 的大范围
分歧造成。

### Pan–Tompkins 报告的 peak 并不位于 raw R apex

| 指标 | 结果 |
|---|---:|
| 审阅片段内的 detector peak 数 | 770 |
| 人工共识 R apex 数 | 680 |
| 一对一匹配 peak 数 | 651 |
| 未匹配 detector peak 的人工 apex 数（漏检） | 29 |
| 未匹配人工 apex 的 detector peak 数（额外检测） | 119 |
| Precision | 84.55% |
| Recall | 95.74% |
| F1 | 89.79% |
| 已匹配 peak 的 detector 时间 − 人工 apex 时间中位数 | **+84.55 ms** |
| 已匹配 peak 绝对时间偏差的 95% 分位数 | 86.0 ms |

`+84.55 ms` 表示 Pan–Tompkins 报告的 sample 晚于人工在 raw ECG 上标出的 R apex。
按数据约 128–130 Hz 的采样率计算，约为 11 个 samples。这是**峰位定位误差**，
不是“RMSSD 延后了 84.55 ms”。

如果每个 beat 都有严格相同的时间平移，RR interval 和 RMSSD 不会改变。因此，
这个系统性偏移本身并不是重建 RMSSD 标量标签的主要理由；但它已说明发布的
`ecg_r_peak_times_*` 并不是 raw R-apex annotation。

### IBI correction 未消除已观察到的 RMSSD 差异

发布数据的主目标为 `ecg_rmssd_corrected_ms`。其标签生成流程是：

1. 根据 Pan–Tompkins peak 计算 RR interval；
2. 仅保留 300–2000 ms 的生理范围内 IBI；
3. 若 IBI 相对 11 点局部中位数的偏差大于 20%，用该局部中位数替换；
4. 在 corrected IBI 序列上计算 RMSSD。

我们将完全相同的规则，分别应用到每个盲标片段的人工共识 apex 和 detector
peak 上。这是一个**片段级诊断**，不能替代完整五分钟人工标签研究。

| corrected RMSSD 比较 | 结果 |
|---|---:|
| 审阅的 20 秒片段数 | 32 |
| 两种序列都可计算 corrected RMSSD 的片段数 | 23 |
| detector − 人工 corrected RMSSD 的中位数 | **+31.9 ms** |
| corrected RMSSD 绝对误差中位数 | **31.9 ms** |
| corrected RMSSD 平均绝对误差 | 33.1 ms |
| corrected RMSSD 绝对误差 90% 分位数 | 56.1 ms |
| correction 前的绝对误差中位数 | 30.7 ms |
| 被 correction 的 IBI 数：人工 / detector | 2 / 4 |

两种序列的片段级 correction ratio 中位数均为零。因此 correction 在这次审计中
很少实际发生，也没有消除 RMSSD 差异。它不能将 detector 点重定位到真实 R apex，
不能恢复漏检 beat，也不能可靠地删除额外检测 beat。

## 当前 RMSSD target 为什么不可靠

核心风险不是“每个 beat 都晚 85 ms”，而是 detector 的**心搏时间序列**与人工
apex 序列不同：

- 存在漏检和额外检测；
- 相对人工 apex 的时间偏移并非每个 beat 都严格恒定；
- RMSSD 会对连续 RR interval 的差值平方，因此相邻 beat 上不相同的小定位误差
  可能被放大。

IBI correction 可以压低明显异常的 interval，但不能替代正确的 beat identity 和
fiducial 定位。它甚至可能让存在 peak 检测错误的序列在数值上显得更合理，但不改变
底层 peak annotation 的错误。

本次审计尚未估计整个发布数据集、完整五分钟窗口上的误差分布。因此，它并不证明
每一个窗口的 RMSSD 都错误了 31.9 ms；但它已经表明，现有流程未达到将该 RMSSD
target 称为“经过独立验证的 ECG ground truth”所需的证据标准。

## 建议的重建流程

1. **冻结当前版本作为 legacy reference。** 保留 raw data 和当前值，但在分析和
   模型结论中将其命名为 `pantompkins_corrected_reference`，而不是 ECG ground
   truth。
2. **建立真正的验证集。** 在跨参与者、跨活动、跨信号形态和跨质量等级的完整
   五分钟窗口上，由两位审阅者标注 raw-ECG R apex，并在算法评估前解决分歧。
3. **只将 detector 用作候选生成器。** 在该验证集上评估候选 ECG detector 加上
   raw-waveform apex refinement 的组合。不要直接选择“最近局部极值”，因为波形
   极性和形态不同，最近极值可能是 S 波而不是 R apex。
4. **同时采用 beat-level 与 HRV-level 的验收标准。** 根据漏检/额外检测率、apex
   timing error，以及相对人工共识的完整五分钟 corrected-RMSSD error 选择流程；
   不能只看 event F1。
5. **谨慎重建 RR 和 RMSSD。** 保留 RR interval 的原始相邻关系；无效或不确定
   interval 应成为 run boundary，不能在过滤后被悄悄当作相邻 interval。仅对已
   验证的心搏序列进行 IBI correction，记录 correction ratio，并剔除 correction
   比例过高的窗口。
6. **批量重处理全队列并进行版本化。** 为每个窗口发布 refined peak time、
   uncorrected/corrected RR list、HRV 值、QC 状态、算法版本和置信度/拒绝原因。

只有在第 4 步证实完整窗口 RMSSD error 足够小之后，重建后的数值才应作为主要的
ECG ground-truth target。

## 相关实现位置

- 当前 ECG 标签生成：
  `src/prepare_windowed_dataset/generate_rawaligned_4device_dataset.py`
  （`_ecg_label_for_window`）
- 当前主标签导出：
  `src/prepare_windowed_dataset/generate_rawslots_training_stride30_dataset.py`
- IBI correction helper：
  `src/heuristic_baselines/algorithms/hrv.py`
- 盲法审阅分析：
  `src/heuristic_baselines/analyze_blinded_rpeak_annotations.py`

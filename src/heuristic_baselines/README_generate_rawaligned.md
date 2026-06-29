# generate_rawaligned_4device_dataset.py 技术文档

## 概述

本脚本用于从原始 PPG/ECG 时间序列生成**基于原始时间线对齐**的多设备 PPG 数据集，以 ECG R-peak 为主标签。与早期基于预切 5 分钟窗口的 v2 生成器不同，本版本直接从原始数据出发，在共享绝对时间网格上切窗、重采样、质控。

### 核心设计理念

1. **Raw-aligned**：不依赖预对齐的窗口数据，直接从各设备原始时间戳出发对齐
2. **ECG R-peak 主标签**：使用 NeuroKit2 Pan-Tompkins 算法从 ECG 信号检测 R-peak，作为 HRV 的 ground truth
3. **PPG 质量仅作元数据**：PPG SQI、运动指标、PPG 波峰及 PPG-derived HRV 不参与窗口筛选决策，仅保存为元数据

---

## 生成流水线（8 步）

```
原始数据
  │
  ▼
┌─────────────────────────────────────────────────┐
│ 第 1 步：加载原始数据                             │
│   加载指定参与者的 PPG（多设备）和 ECG（Polar）    │
│   数据文件（.npz 格式）                           │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 2 步：生成候选窗口                             │
│   找到所有设备 + ECG 的公共时间重叠区间            │
│   按 stride 间隔在公共区间上等距切分窗口           │
│   窗口起点对齐到 stride 的整数倍                   │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 3 步：★ 边界对齐检查 + ECG 标签质控            │
│   对每个候选窗口：                                │
│   (a) 检查各设备在窗口起止处的偏移 ≤ tolerance    │
│   (b) 提取 ECG 片段，Pan-Tompkins 检测 R-peak    │
│   (c) 计算 RR 间期、RMSSD、SDNN、HR 等 HRV 指标  │
│   (d) 综合 7 项条件判断 ECG 标签是否可信          │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 4 步：预分配输出数组                           │
│   仅对通过边界+ECG 筛选的窗口分配内存              │
│   ppg 形状: (N, n_devices, n_channels, target_len)│
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 5 步：★★ PPG 重采样 + 波峰检测 + 质控         │
│   对每个窗口的每个设备/通道：                      │
│   (a) 将原始 PPG 重采样到统一频率的时间网格       │
│   (b) 计算有效采样覆盖率                          │
│   (c) PPG 波峰检测和 HRV 计算（仅作元数据）       │
│   (d) PPG 单通道质控（仅作元数据）                │
│   (e) 运动占比计算（仅作元数据）                  │
│   若任一设备/通道的采样覆盖率 < 阈值 → 标记丢弃   │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 6 步：最终筛选                                 │
│   仅保留所有设备/通道的 PPG 采样覆盖率均达标的窗口 │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 7 步：★ 保存为压缩 .npz 文件                   │
│   每个参与者一个文件，包含配置、PPG 数据、ECG 标签 │
│   PPG 元数据和质控信息                             │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ 第 8 步：输出汇总统计 CSV                         │
│   记录候选窗口数、各阶段通过数、保留数、            │
│   平均质量指标等                                   │
└─────────────────────────────────────────────────┘
```

---

## 窗口纳入规则（三级筛选）

一个窗口最终被保留，必须同时满足以下三个条件：

| 筛选阶段 | 条件 | 说明 |
|---|---|---|
| **1. 边界对齐** | 所有 PPG 设备在窗口起止处的偏移 ≤ `alignment_tolerance_sec` | 确保各设备在该窗口有数据覆盖 |
| **2. ECG 标签质控** | 通过 7 项综合质控 | 确保 ECG R-peak 标签可信 |
| **3. PPG 采样覆盖** | 所有设备/通道的 `ppg_valid_sample_ratio` ≥ `min_valid_sample_ratio` | 重采样后有效样本比例达标 |

> **注意**：PPG SQI、运动占比、PPG 波峰质量、PPG-derived HRV **不参与**窗口筛选，仅作为元数据保存。

### ECG 标签质控 7 项条件

| 条件 | 失败原因标记 | 说明 |
|---|---|---|
| ECG 有效样本覆盖率 ≥ `min_ecg_valid_sample_ratio` | `low_ecg_sample_ratio` | 窗口内 ECG 信号完整性 |
| 有效 IBI 比例 ≥ `ecg_min_valid_ibi_ratio` | `low_ecg_valid_ibi_ratio` | RR 间期在生理范围内的比例 |
| R-peak 数量 ≥ max(3, 基于最低心率的下限) | `too_few_ecg_peaks` | 检测到足够的心跳 |
| RMSSD 有限（非 NaN） | `invalid_ecg_hrv` | HRV 计算成功 |
| IBI 校正比例 ≤ `DEFAULT_MAX_IBI_CORRECTION_RATIO` | `high_ecg_ibi_correction_ratio` | 异常间期修正量不过大 |
| 心率在 [`min_hr_bpm`, `max_hr_bpm`] 范围内 | `ecg_hr_out_of_range` | 心率在生理合理范围 |
| RMSSD ≤ `DEFAULT_MAX_RMSSD_MS` | `ecg_rmssd_too_high` | HRV 不异常偏高 |

---

## 关键参数

### 窗口与采样参数

| 参数 | 命令行选项 | 默认值 | 说明 |
|---|---|---|---|
| 窗口时长 | `--window-sec` | 300 | 每个窗口的时长（秒），即 5 分钟 |
| 窗口步长 | `--stride-sec` | 300 | 窗口之间的间隔（秒）。等于 window-sec 时无重叠；小于时有重叠（适合训练） |
| 目标采样率 | `--target-fs` | 50.0 | PPG 重采样的目标频率（Hz）。100.0 用于更高分辨率 |
| 边界对齐容差 | `--alignment-tolerance-sec` | 2.0 | 允许设备在窗口起止处的最大偏移（秒） |
| 最低有效采样比 | `--min-valid-sample-ratio` | 0.50 | PPG 重采样后有效样本占比下限（0.90 = 允许最多 10% sample loss） |
| 最大源间隔 | `--max-source-gap-ms` | 500.0 | 重采样时原始样本的最大允许间隔（毫秒） |

### ECG 质控参数

| 参数 | 命令行选项 | 默认值 | 说明 |
|---|---|---|---|
| ECG 最低有效 IBI 比 | `--ecg-min-valid-ibi-ratio` | 0.80 | RR 间期在生理范围内的最低比例（1.0 = 全部 RR 必须有效） |
| ECG 最低有效采样比 | `--min-ecg-valid-sample-ratio` | 0.95 | ECG 信号在窗口内的最低覆盖率 |
| 最低心率 | `--min-hr-bpm` | 30.0 | 允许的最低心率（BPM） |
| 最高心率 | `--max-hr-bpm` | 200.0 | 允许的最高心率（BPM） |

### 运动检测参数（仅用于元数据）

| 参数 | 命令行选项 | 默认值 | 说明 |
|---|---|---|---|
| 运动段长 | `--motion-seg-sec` | 10.0 | 加速度计数据按此长度分段计算标准差 |
| 运动阈值百分位 | `--motion-percentile` | 75.0 | 取所有段标准差的此百分位数作为运动阈值 |

### 其他参数

| 参数 | 命令行选项 | 默认值 | 说明 |
|---|---|---|---|
| PPG 设备列表 | `--devices` | 全部 4 设备 | 逗号分隔的设备名（如 `Earring,Ring,Watch`） |
| 排除的参与者 | `--exclude` | P2,P14,P16,P17 | 排除的参与者 ID |
| PPG 预处理模式 | `--preprocess-mode` | bandpass | PPG 波峰检测前的预处理方式 |
| 最大窗口数 | `--max-windows` | 无限制 | 调试用：限制每个参与者的最大候选窗口数 |

---

## 输出文件

### 每参与者文件 `{dataset_name}_{pid}.npz`

| 字段 | 形状 | 说明 |
|---|---|---|
| **PPG 数据** | | |
| `ppg_50hz` | (N, D, 2, L) | 重采样后的 PPG 信号（字段名沿用历史，实际采样率由 `target_fs` 决定） |
| `ppg_valid_mask_50hz` | (N, D, 2, L) | 每个重采样点是否由附近原始样本支持 |
| `ppg_valid_sample_ratio` | (N, D, 2) | 每设备/通道的有效样本覆盖率 |
| **ECG 主标签** | | |
| `ecg_r_peak_times_rel_ms` | (N,) object | R-peak 相对于窗口起点的时间（毫秒），**主标签** |
| `ecg_r_peak_indices_50hz` | (N,) object | R-peak 映射到采样网格的索引 |
| `ecg_rr_intervals_ms` | (N,) object | RR 间期（毫秒） |
| `ecg_rr_intervals_corrected_ms` | (N,) object | 校正后 RR 间期 |
| `ecg_rmssd_ms` | (N,) | RMSSD（毫秒） |
| `ecg_sdnn_ms` | (N,) | SDNN（毫秒） |
| **ECG 质控** | | |
| `ecg_label_qc_pass` | (N,) bool | ECG 标签是否通过质控 |
| `ecg_label_qc_reason` | (N,) object | 质控失败原因（`ok` 表示通过） |
| `ecg_valid_ibi_ratio` | (N,) | 有效 IBI 比例 |
| `ecg_qrs_sqi` | (N,) | QRS 信号质量指数 |
| `ecg_hr_bpm` | (N,) | 心率（BPM） |
| **PPG 元数据** | | |
| `ppg_peak_times_rel_ms` | (N, D, 2) object | PPG 波峰时间（元数据） |
| `ppg_ibi_ms` | (N, D, 2) object | PPG IBI（元数据） |
| `ppg_rmssd_ms` | (N, D, 2) | PPG RMSSD（元数据） |
| `ppg_sqi` | (N, D, 2) | PPG 信号质量指数（元数据） |
| `ppg_quality_flag` | (N, D, 2) bool | PPG 质量标记（元数据） |
| `motion_fraction` | (N, D) | 运动占比（元数据） |

> N = 窗口数，D = 设备数，L = target_len

### 汇总文件 `{dataset_name}_summary.csv`

记录每个参与者的窗口统计：候选窗口数、边界对齐通过数、ECG 质控通过数、PPG 采样通过数、最终保留数、各质量指标的均值。

### 配置文件 `config.json`

保存完整的生成参数，可用于复现。

---

## 运行示例

### 标准 4 设备 50Hz strict reference（stride=300，无重叠）

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  src/heuristic_baselines/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_4device_rawaligned_strict_reference \
  --stride-sec 300
```

### 3 设备 100Hz、ECG IBI 100%、sample 90%（训练用，stride=60）

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  src/heuristic_baselines/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90 \
  --devices Earring,Ring,Watch \
  --target-fs 100.0 \
  --stride-sec 60 \
  --min-valid-sample-ratio 0.90 \
  --ecg-min-valid-ibi-ratio 1.0
```

### 3 设备 100Hz strict reference、sample 90%（评估用，stride=300）

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  src/heuristic_baselines/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90 \
  --devices Earring,Ring,Watch \
  --target-fs 100.0 \
  --stride-sec 300 \
  --min-valid-sample-ratio 0.90 \
  --ecg-min-valid-ibi-ratio 1.0
```

---

## 已生成数据集一览

| 数据集名 | 设备 | 频率 | stride | sample阈值 | ECG IBI | 窗口数 |
|---|---|---|---|---|---|---|
| `synced_4device_rawaligned_strict_reference` | 4 设备 | 50 Hz | 300s | 0.50 | 0.80 | — |
| `synced_4device_rawaligned_training_v1` | 4 设备 | 50 Hz | 60s | 0.50 | 0.80 | — |
| `synced_3device_rawaligned_strict_reference_100hz_ecgibi100` | 3 设备 | 100 Hz | 300s | 0.50 | 1.00 | 1803 |
| `synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90` | 3 设备 | 100 Hz | 300s | 0.90 | 1.00 | 1718 |
| `synced_3device_rawaligned_training_v1_100hz_ecgibi100` | 3 设备 | 100 Hz | 60s | 0.50 | 1.00 | — |
| `synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90` | 3 设备 | 100 Hz | 60s | 0.90 | 1.00 | 8608 |

---

## 依赖

- Python 3.10+
- NumPy, Pandas, SciPy
- NeuroKit2（ECG R-peak 检测）
- 项目内部模块：`config`, `algorithms.hrv`, `generate_synced_4device_dataset`, `io_utils`

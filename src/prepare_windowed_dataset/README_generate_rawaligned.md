# generate_rawaligned_4device_dataset.py 生成说明

## 当前对应关系确认

这份 README 对应当前脚本：

`/Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/prepare_windowed_dataset/generate_rawaligned_4device_dataset.py`

当前脚本的真实行为是：

1. 从 `snowballlab/Multisite-PPG/raw_data` 的原始 PPG/ECG 时间线生成 raw-aligned 数据集。
2. 默认设备是 `Earring`, `Ring`, `Watch` 三个设备。
3. 默认通道是 `ppg_green`, `ppg_ir`。
4. PPG 会重采样到统一时间网格，作为模型输入保存到 `ppg_resampled`。
5. ECG R-peak 是主标签，保存到 `ecg_r_peak_times_rel_ms`。
6. PPG peak、PPG IBI、PPG-derived HRV、PPG SQI、PPG quality flag 不写入数据集。
7. 加速度保存两个窗口级 per-device 特征：
   - `accel_mean_mag`: 原始三轴加速度幅值均值。
   - `accel_motion_mean_mag`: 去重力后的 motion 幅值均值。
8. 使用 `numpy.savez_compressed` 保存每个参与者的 `.npz`。

因此，当前最终版 Hugging Face 数据集与脚本的对应关系是成立的；README 以当前脚本逻辑为准。

---

## 数据集目标

本脚本用于生成适合 HRV baseline 和训练模型共同使用的 raw-aligned 数据集。数据集只保留两类信息：

1. baseline 和模型都需要的原始/对齐输入，例如 PPG、mask、加速度 motion 特征。
2. ECG reference 标签和 ECG 标签质控信息。

不再把用于某个 baseline 的 PPG 派生结果提前写进数据集。PPG peak、PPG IBI、PPG HRV、PPG SQI 等应在 baseline 或模型评估代码中按算法版本实时计算。

---

## 生成流程

```text
原始 Multisite-PPG raw_data
  |
  v
加载每个参与者的 Earring/Ring/Watch PPG 和 Polar ECG
  |
  v
取所有设备和 ECG 的公共时间重叠区间
  |
  v
按 window_sec 和 stride_sec 在绝对时间线上切 5 分钟窗口
  |
  v
检查 PPG 设备边界是否在 alignment_tolerance_sec 内
  |
  v
对 ECG 窗口运行 NeuroKit2 Pan-Tompkins R-peak 检测
  |
  v
执行 ECG 标签质控
  |
  v
把 PPG green/IR 重采样到 target_fs 网格
  |
  v
检查所有设备/通道的 PPG sample coverage
  |
  v
计算 accel_mean_mag 和 accel_motion_mean_mag
  |
  v
保存每参与者一个 compressed NPZ，并输出 summary CSV/README/config
```

---

## 窗口纳入规则

一个窗口会被保留，当且仅当同时满足以下条件：

| 阶段 | 条件 | 说明 |
|---|---|---|
| 边界对齐 | 每个 PPG 设备在窗口起点和终点附近都有样本，偏移不超过 `alignment_tolerance_sec` | 避免某设备窗口头尾缺数据 |
| ECG 标签质控 | `ecg_label_qc_pass == True` | 保证 ECG reference 可用 |
| PPG 采样覆盖 | 每个设备、每个通道的 `ppg_valid_sample_ratio >= min_valid_sample_ratio` | 当前最终版用 `0.90`，允许最多约 10% sample loss |

PPG 派生质量指标不参与筛选，因为当前数据集不保存这些 baseline 派生信息。

---

## ECG 标签质控

ECG 标签由窗口内 Polar ECG 原始信号产生。脚本使用 NeuroKit2 的 Pan-Tompkins 1985 方法检测 R-peak，然后计算 RR/HRV 指标并做质控。

窗口必须满足：

| 条件 | 默认值/规则 | 失败原因 |
|---|---:|---|
| ECG 有效样本覆盖率 >= `min_ecg_valid_sample_ratio` | `1.0` | `low_ecg_sample_ratio` |
| 有效 IBI 比例 >= `ecg_min_valid_ibi_ratio` | `1.0` | `low_ecg_valid_ibi_ratio` |
| R-peak 数量 >= `max(3, floor(window_sec * min_hr_bpm / 60))` | 默认最低心率 `30 bpm` | `too_few_ecg_peaks` |
| RMSSD 可计算且有限 | 必须有限 | `invalid_ecg_hrv` |
| ECG IBI 校正比例 <= `DEFAULT_MAX_IBI_CORRECTION_RATIO` | `0.2` | `high_ecg_ibi_correction_ratio` |
| 心率在 [`min_hr_bpm`, `max_hr_bpm`] | 默认 `[30, 200] bpm` | `ecg_hr_out_of_range` |
| RMSSD <= `DEFAULT_MAX_RMSSD_MS` | `200 ms` | `ecg_rmssd_too_high` |

注意：数据集中同时保存未校正的 `ecg_rr_intervals_ms` 和校正后的 `ecg_rr_intervals_corrected_ms`。

---

## 加速度字段

脚本保存 per-window、per-device 的两个加速度特征。

### `accel_mean_mag`

原始三轴加速度幅值的窗口均值：

```text
m_t = sqrt(accel_x_t^2 + accel_y_t^2 + accel_z_t^2)
accel_mean_mag = mean(m_t)
```

这个字段保留原始 magnitude，包含重力分量。

### `accel_motion_mean_mag`

去重力后的窗口级 motion feature：

```text
accel_motion_mean_mag = mean(abs(m_t - gravity_reference))
```

`gravity_reference` 的选择规则：

| 原始 magnitude 中位数 | gravity_reference |
|---:|---:|
| 7.0 到 12.5 | `9.80665` |
| 0.7 到 1.3 | `1.0` |
| 其他情况 | 该设备该文件的 magnitude 中位数 |

后续 baseline/model 默认应优先使用 `accel_motion_mean_mag` 来判断运动状态。例如老师要求的 no-motion 阈值可以在 baseline/model 中用 `accel_motion_mean_mag < 0.1` 来判断，而不是在数据集生成阶段提前写标签。

---

## 关键参数

### 命令行参数

| 参数 | 默认值 | 当前最终版使用 | 说明 |
|---|---:|---:|---|
| `--participants` | `None` | 默认扫描 raw root 后排除指定参与者 | 可用逗号指定参与者 |
| `--exclude` | `P2,P14,P16,P17` | `P2,P14,P16,P17` | 排除无效或不纳入最终版的参与者 |
| `--devices` | `Earring,Ring,Watch` | `Earring,Ring,Watch` | 当前脚本只支持这 3 个设备 |
| `--window-sec` | `300` | `300` | 5 分钟窗口 |
| `--stride-sec` | `300` | strict: `300`; training: `60` 或 `30` | 窗口步长 |
| `--target-fs` | `50.0` | `100.0` | 最终版使用 100 Hz |
| `--alignment-tolerance-sec` | `2.0` | `2.0` | 设备窗口边界容差 |
| `--min-valid-sample-ratio` | `0.50` | `0.90` | PPG 采样覆盖阈值 |
| `--max-source-gap-ms` | `500.0` | `500.0` | 重采样时最近原始样本的最大允许距离 |
| `--ecg-min-valid-ibi-ratio` | `1.0` | `1.0` | ECG valid IBI ratio |
| `--min-ecg-valid-sample-ratio` | `1.0` | `1.0` | ECG 有效采样覆盖率 |
| `--min-hr-bpm` | `30.0` | `30.0` | ECG 质控最低心率 |
| `--max-hr-bpm` | `200.0` | `200.0` | ECG 质控最高心率 |
| `--max-windows` | `None` | `None` | 调试用窗口数量限制 |

### 当前最终版公共配置

| 项目 | 值 |
|---|---|
| 设备 | `Earring`, `Ring`, `Watch` |
| PPG 通道 | `ppg_green`, `ppg_ir` |
| 采样率 | `100 Hz` |
| 窗口长度 | `300 s` |
| 窗口样本数 | `30000` |
| PPG sample threshold | `0.90` |
| ECG valid IBI ratio | `1.0` |
| ECG valid sample ratio | `1.0` |
| ECG detector | `window_neurokit2_pantompkins1985` |
| primary label | `ecg_r_peak_times_rel_ms` |

---

## 输出结构

每个数据集目录包含：

```text
config.json
README.md
{dataset_name}_summary.csv
{dataset_name}_{participant}.npz
{dataset_name}_{participant}_summary.csv
```

每个参与者一个 `.npz` 文件。

---

## NPZ 字段

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `config_json` | scalar string | 当前参与者文件的生成配置 |
| `participant` | scalar string | 参与者 ID |
| `devices` | `(D,)` | 设备列表 |
| `channels` | `(2,)` | `ppg_green`, `ppg_ir` |
| `t0_ms`, `t1_ms` | `(N,)` | 窗口绝对起止时间，毫秒 |
| `max_start_diff_ms`, `max_end_diff_ms` | `(N,) float32` | 设备边界最大偏移 |
| `ppg_resampled` | `(N, D, 2, L) float32` | 重采样后的 raw PPG，模型输入 |
| `ppg_valid_mask_resampled` | `(N, D, 2, L) bool` | 每个重采样点是否由附近原始样本支持 |
| `ppg_valid_sample_ratio` | `(N, D, 2) float32` | 每设备/通道采样覆盖率 |
| `accel_mean_mag` | `(N, D) float32` | 原始加速度 magnitude 窗口均值 |
| `accel_motion_mean_mag` | `(N, D) float32` | 去重力后的 motion magnitude 窗口均值 |
| `ecg_r_peak_times_rel_ms` | `(N,) object of float32 arrays` | ECG R-peak 相对窗口起点时间，主标签 |
| `ecg_r_peak_indices_grid` | `(N,) object of int32 arrays` | R-peak 映射到 target_fs 网格的索引 |
| `ecg_r_peak_amplitudes_raw` | `(N,) object of float32 arrays` | R-peak 原始 ECG 振幅 |
| `ecg_rr_intervals_ms` | `(N,) object of float32 arrays` | 未校正 RR interval |
| `ecg_rr_intervals_corrected_ms` | `(N,) object of float32 arrays` | 校正后 RR interval |
| `ecg_rmssd_ms` | `(N,) float32` | ECG reference RMSSD |
| `ecg_sdnn_ms` | `(N,) float32` | ECG reference SDNN |
| `ecg_valid_sample_ratio` | `(N,) float32` | ECG 有效采样覆盖率 |
| `ecg_valid_ibi_ratio` | `(N,) float32` | ECG 有效 IBI 比例 |
| `ecg_ibi_correction_ratio` | `(N,) float32` | ECG IBI 校正比例 |
| `ecg_qc_pass`, `ecg_label_qc_pass` | `(N,) bool` | ECG 质控是否通过 |
| `ecg_qc_reason`, `ecg_label_qc_reason` | `(N,) object` | ECG 质控原因 |
| `ecg_qrs_sqi` | `(N,) float32` | 简单 ECG QRS 质量分数 |
| `ecg_peak_count` | `(N,) int32` | R-peak 数量 |
| `ecg_hr_bpm` | `(N,) float32` | ECG 心率 |
| `ecg_rr_cv` | `(N,) float32` | RR coefficient of variation |
| `ecg_rpeak_detector_agreement` | `(N,) float32` | 当前填充为 NaN，预留字段 |

其中 `N` 是保留窗口数，`D` 是设备数，`L = window_sec * target_fs`。当前最终版 `D = 3`，`L = 30000`。

---

## dtype 策略

为降低 NPZ 体积，当前脚本保存时使用：

| 字段类别 | dtype |
|---|---|
| `ppg_resampled` | `float32` |
| `ppg_valid_mask_resampled` | `bool` |
| `ppg_valid_sample_ratio` | `float32` |
| `accel_mean_mag`, `accel_motion_mean_mag` | `float32` |
| ECG HRV/QC 数值字段 | `float32` |
| ECG peak count | `int32` |
| ECG grid indices | object arrays of `int32` |
| ECG ragged float arrays | object arrays of `float32` |
| 字符串/object metadata | 保持 object/string |

---

## 当前最终版三个数据集

Hugging Face `snowballlab/HRV` 当前最终版保留 3 个数据集目录。

| 数据集 | 用途 | stride | 本地状态 |
|---|---|---:|---|
| `synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100` | strict reference / evaluation | `300 s` | 本地保留 |
| `synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100` | training | `60 s` | HF 保留，本地已清理 |
| `synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30` | training | `30 s` | 本地保留 |

本地已确认：

| 数据集 | 参与者数 | 本地窗口数 |
|---|---:|---:|
| strict reference stride 300 | 16 | 1706 |
| training stride 30 | 16 | 17064 |

stride 60 数据集在 Hugging Face 上保留，本地副本之前已删除以节省空间。

---

## 运行示例

### strict reference: stride 300

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  /Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/prepare_windowed_dataset/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_3device_rawaligned_strict_reference_100hz_ecgibi100_sample90_ecgsample100 \
  --raw-root /path/to/Multisite-PPG/raw_data \
  --devices Earring,Ring,Watch \
  --target-fs 100.0 \
  --window-sec 300 \
  --stride-sec 300 \
  --min-valid-sample-ratio 0.90 \
  --ecg-min-valid-ibi-ratio 1.0 \
  --min-ecg-valid-sample-ratio 1.0
```

### training: stride 60

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  /Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/prepare_windowed_dataset/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100 \
  --raw-root /path/to/Multisite-PPG/raw_data \
  --devices Earring,Ring,Watch \
  --target-fs 100.0 \
  --window-sec 300 \
  --stride-sec 60 \
  --min-valid-sample-ratio 0.90 \
  --ecg-min-valid-ibi-ratio 1.0 \
  --min-ecg-valid-sample-ratio 1.0
```

### training: stride 30

```bash
/Users/jiqiyu/Desktop/Daily_HRV/venv/bin/python \
  /Users/jiqiyu/Desktop/Daily_HRV/wearable-ppg-dataset/src/prepare_windowed_dataset/generate_rawaligned_4device_dataset.py \
  --dataset-name synced_3device_rawaligned_training_v1_100hz_ecgibi100_sample90_ecgsample100_stride30 \
  --raw-root /path/to/Multisite-PPG/raw_data \
  --devices Earring,Ring,Watch \
  --target-fs 100.0 \
  --window-sec 300 \
  --stride-sec 30 \
  --min-valid-sample-ratio 0.90 \
  --ecg-min-valid-ibi-ratio 1.0 \
  --min-ecg-valid-sample-ratio 1.0
```

---

## 注意事项

1. 当前脚本名仍叫 `generate_rawaligned_4device_dataset.py`，但当前 `DEVICES` 常量是 3 个设备：`Earring`, `Ring`, `Watch`。
2. 默认 `RAW_ROOT` 来自 `src/prepare_windowed_dataset/config.py` 里的 `WINDOW_INPUT_ROOT`。如果本地没有 `Multisite-PPG/raw_data`，需要显式传入 `--raw-root`。
3. 脚本有断点续跑逻辑：如果 `{dataset_name}_{pid}.npz` 已存在，会跳过该参与者。
4. 如果修改了脚本逻辑但想重新生成某个参与者，需要先删除对应参与者的 `.npz` 和 summary，或输出到新的 dataset name。
5. 数据集不保存 PPG-derived 标签。baseline/model 应自行从 `ppg_resampled` 计算 PPG peak、IBI、PRV/HRV、SQI 等。
6. 模型训练时可直接读取 `ppg_resampled` 和 `ppg_valid_mask_resampled`；运动相关分析优先使用 `accel_motion_mean_mag`。

---

## 依赖

- Python
- NumPy
- Pandas
- NeuroKit2
- 项目内部模块：
  - `prepare_windowed_dataset.config`
  - `heuristic_baselines.algorithms.hrv`

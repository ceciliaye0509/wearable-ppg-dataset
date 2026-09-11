# Raw-slot SegNet baseline 与加速审计

更新日期：2026-09-11

## 冻结状态

四折 participant-disjoint SegNet 已按同一协议完成，随后冻结；没有在 fold 之间修改训练、
验证或 early-stopping 规则。

- 输入：单设备 `ppg_rawslot_values` green + IR 与 `ppg_rawslot_mask`；没有读取
  `ppg_resampled_values`。
- 标签顺序：`ecg_rmssd_corrected_ms`、`ecg_sdnn_corrected_ms`。
- 样本：11,788 个窗口、35,364 个 window-device rows、16 participants。
- 完整性：`artifacts/ml_hrv/segnet_rawslot_formal/FROZEN.json` 固定 config、folds、4 个
  checkpoint、各 fold 指标与逐窗 CSV 的 SHA-256。
- 最佳 epoch / 停止 epoch：fold 0 为 4/12，fold 1 为 1/9，fold 2 为 2/10，fold 3 为 3/11。

## 主结果

以下为 participant-macro；95% CI 的 bootstrap 单位也是 participant，而不是相互重叠的
30 s stride windows。

| Target | MAE ms (95% CI) | RMSE ms | bias ms | r | R² | CCC |
|---|---:|---:|---:|---:|---:|---:|
| RMSSD | 12.036 (10.189–13.961) | 14.103 | -1.550 | 0.013 | -2.410 | 0.001 |
| SDNN | 14.799 (12.501–17.525) | 18.975 | 0.245 | 0.063 | -0.923 | 0.026 |

| Fold | RMSSD MAE / r | SDNN MAE / r |
|---|---:|---:|
| 0 | 14.958 / 0.049 | 11.549 / 0.171 |
| 1 | 11.395 / 0.009 | 14.558 / -0.039 |
| 2 | 11.585 / 0.018 | 15.698 / 0.088 |
| 3 | 10.207 / -0.026 | 17.392 / 0.032 |

结论很明确：MAE 比冻结的 qPPG heuristic RMSSD participant-macro MAE 20.028 ms 低，
但相关、CCC 近零且 R² 为负，说明 SegNet 主要学到了回归到总体均值，**没有学会可靠的
continuous tracking，因此不是 SOTA**。qPPG 的 mean participant coverage 为 79.52%，而
本数据 release 本身已做 coverage 预筛；两者 coverage 口径不同，不能把 ML 的 100% 输出率
解释成更高的原始流 coverage。

设备 participant-macro RMSSD MAE 为 Earring 12.000、Ring 11.710、Watch 12.398 ms；
SDNN 为 14.900、14.929、14.569 ms。三设备 precision-fusion upper bound 的 window-micro
RMSSD MAE 是 10.724 ms、SDNN MAE 是 15.439 ms，但 r 分别为 -0.099、-0.069；这不是
足够的互补性证据，不启用部署 fusion。

### Coverage 审计

这里需要区分三个概念：PPG observed-slot coverage、模型实际输出 coverage，以及按 confidence
拒识后的 retained coverage。当前 Hugging Face training release 已经做过约 90% 的 PPG
coverage 预筛，所以以下结果只描述“已通过预筛的数据”，不能外推为全天原始流 coverage。

各设备 rawslot coverage 分布：

| Device | rows | mean | median | min | 95th percentile | max |
|---|---:|---:|---:|---:|---:|---:|
| Earring | 11,788 | 94.66% | 94.83% | 90.00% | 96.57% | 99.37% |
| Ring | 11,788 | 94.70% | 94.85% | 90.00% | 96.49% | 99.24% |
| Watch | 11,788 | 94.96% | 95.07% | 90.03% | 97.08% | 99.40% |

按 PPG coverage 分层的 window-micro 误差：

| Coverage | rows (%) | RMSSD MAE ms | SDNN MAE ms |
|---|---:|---:|---:|
| 90–95% | 18,798 (53.16%) | 10.881 | 15.984 |
| 95–98% | 16,439 (46.49%) | 10.920 | 15.271 |
| 98–100% | 127 (0.36%) | 8.359 | 13.079 |

同一 coverage 桶内的设备结果如下；98–100% 桶样本很少，不能据此排设备名次：

| Coverage | Device (n) | RMSSD MAE ms | SDNN MAE ms |
|---|---|---:|---:|
| 90–95% | Earring (6,601) | 11.040 | 15.891 |
| 90–95% | Ring (6,559) | 10.704 | 15.915 |
| 90–95% | Watch (5,638) | 10.902 | 16.174 |
| 95–98% | Earring (5,121) | 10.360 | 15.340 |
| 95–98% | Ring (5,218) | 10.706 | 15.736 |
| 95–98% | Watch (6,100) | 11.573 | 14.814 |
| 98–100% | Earring (66) | 7.362 | 13.657 |
| 98–100% | Ring (11) | 4.776 | 9.572 |
| 98–100% | Watch (50) | 10.463 | 13.087 |

冻结 CSV 中模型对 35,360/35,364 rows 给出 accepted，输出 coverage 为 99.9887%（报告四舍
五入为 100.0%）。其余 4 rows 的 coverage 本应正好是 90%，但 float32 表示成
`0.899999976`，被严格 `<0.90` 比较误拒；未来 evaluator 已加 `1e-6` 容差，冻结 CSV 不回写。

按 baseline confidence 排序后的 risk–coverage 是 window-micro：

| Retained coverage | RMSSD MAE ms | SDNN MAE ms |
|---:|---:|---:|
| 100% | 10.890 | 15.642 |
| 90% | 10.685 | 15.540 |
| 80% | 10.714 | 15.378 |
| 50% | 10.784 | 15.653 |

拒掉一半窗口并没有稳定降低误差，进一步说明 SegNet 的 confidence 没有排序风险的能力。
未来 uncertainty 模型必须同时报告 participant-macro accuracy、原始流 acceptance coverage、
coverage 分层及 risk–coverage，不能只报已筛选窗口上的 MAE。

基线 direct log-variance 固定为 0，confidence 约 0.73，高误差窗与其他窗的平均 confidence
几乎相同。因此当前 confidence 只是接口占位，不能用于临床或产品拒识。motion strata 中
high-motion MAE 反而较低，可能来自 participant/标签分布混杂，不能作“运动改善模型”的
因果解释。

### Train-only 常数基线审计

为确认网络是否真的使用了 PPG，每一折只用该折 train participants 的唯一窗口拟合常数，
再对原封不动的 outer test rows 评分：

| Predictor | RMSSD participant-macro MAE ms | SDNN participant-macro MAE ms |
|---|---:|---:|
| train arithmetic mean | 11.114 | 14.910 |
| train median | **11.112** | **14.257** |
| train geometric mean | 11.207 | 14.363 |
| raw-slot SegNet | 12.036 | 14.799 |

SegNet 比 train-median 常数在 RMSSD 上差 0.924 ms，在 SDNN 上差 0.542 ms。因此它不仅
没有达到 SOTA，也没有通过“优于不看波形的 train-only constant”这一最低门槛。后续
causal-direct、beat 和 fusion 均必须同时超过 constant、qPPG 与 SegNet；否则不进入正式
候选。完整逐 fold 常数及评分保存在
`artifacts/ml_hrv/segnet_rawslot_formal/constant_baseline_audit.json`。

## 基于原代码以及相关论文的补充点

本节复核了伙伴代码的两个状态：开始本工作时的 `fe9e454`，以及 push 前从远端同步到的
`abc465c`。后者已经吸收了一批重要修复，所以以下结论明确区分“早期问题已修复”和
“raw continuous HRV 仍需补充”，避免用旧审计结果评价新代码。

### 原代码中值得保留的部分

伙伴 HRV 扩展并非完全错误，它提供了三个有价值的起点：

1. [`hrv_ext.py`](../model_baselines/supervised/hrv_ext.py) 已把 5 min 波形切为 30 个 10 s
   segment，并用同一个 CNN 编码，再做 mean/LSTM aggregation；这就是当前 SegNet 复现所
   保留的核心结构。
2. 它对 HRV target 先取 log，再只用训练参与者拟合 mean/std；当前实现继续使用 train-only
   log-ms scaler。
3. HRV loader 把整个 test participant 留出，并另取一个 participant 做 validation，优于把
   同一人的重叠窗口随机分进 train/test。当前实现进一步固定为可复现的四折
   participant-disjoint 协议。
4. 远端最新版已经使用 `ecg_sdnn_corrected_ms`、`ecg_rmssd_corrected_ms`，并新增
   green/IR/both channel 选项、PPG band-pass + polyphase resampling、Kazemi-style/dilated
   peak detector、weighted BCE + Dice、只用 validation 选择 peak threshold，以及 event F1、
   peak count、RR correction ratio 和解码 HRV coverage。这些都应作为更强的伙伴 baseline
   保留，而不是在新目录重复造一套弱版本。
5. 远端最新版也修正了 RR removal 后的 adjacency：被删除的 interval 两侧不再被错误拼成
   一对来计算 RMSSD。新 pipeline 的 NaN separator 测试继续保留，作为同一生理约束的显式
   数据契约。

因此，新 pipeline 是在保留原框架有效部分的基础上修正数据和生理语义，而不是因为 SegNet
结果不好就完全推倒重写。

### 从早期版本到远端最新版，再到本 pipeline

| 审计项 | 早期伙伴代码 | 远端最新版 `abc465c` | raw-slot/SOTA 路径仍需做的补充 | 状态 |
|---|---|---|---|---|
| 主 PPG | resampled PPG、普通 mean/std z-normalization | 字段名明确为 `ppg_resampled_values`；peak 路径增加缺失值线性填充、零相位 band-pass 与 polyphase resampling | 主输入固定为未做波形插值的 `ppg_rawslot_values`；mask 与 relative timestamp jitter 独立入模，只对 observed slots 做 window-local robust normalization。伙伴 peak 预处理可作为 resampled 对照，但零相位滤波不能直接代表 causal 部署 | raw-slot 数据契约已完成 |
| Ground truth | 曾使用未明确 corrected 的 target | 已改为 `[ecg_sdnn_corrected_ms, ecg_rmssd_corrected_ms]` | 本项目公开契约固定 RMSSD→SDNN；checkpoint 必须保存 `target_order` 与 train-only scaler，防止虽然各自代码内部一致、跨 pipeline 却交换列语义 | 已完成 |
| 5 min direct aggregation | 30×10 s shared encoder + mean/LSTM | 保持该结构，是有价值的强 baseline | mean 对 30 个 token 等权且无显式 causal state；正式 direct 模型用 causal TCN，并输入 segment coverage、mask 和 timing | SegNet 已冻结；TCN 待正式四折 |
| Beat target 与架构 | ECG R peak 直接画到 PPG sample；基础 PeakNet | 新增 Kazemi-style target、dilated/Kazemi backbone、weighted BCE + Dice，并在 validation 选 threshold | target 仍没有建模 ECG electrical event 到 PPG pulse 的生理延迟；需在 100–700 ms 内做 latent delay search/offset head，outer test 不参与 delay 或 threshold 调参 | delay-aware 实现已完成，待正式训练 |
| RR adjacency | 早期范围过滤后直接对压缩数组 `diff`，可能跨被删 interval 计算 RMSSD | raw、remove 路径已用原位置 mask 保留 adjacency；median correction 保持序列长度 | 继续以 NaN separator 和单元测试覆盖断点，并增加 extra/missed beat 的 merge/split 审计 | 核心修复两边均有；高级纠错待验证 |
| Coverage 与失败 | 早期只对 finite HRV 窗口评分 | 现在打印 event precision/recall/F1、count、correction ratio 和 decoded-HRV coverage | peak loader 默认先要求 `ppg_resampled_valid_sample_ratio >= 0.95`，所以该 coverage 是“预筛后的条件 coverage”；仍需每个输入窗口的预测或 reject reason、raw PPG/acceptance/risk–coverage，以及未来未预筛 raw-session challenge set | 逐窗 schema 与 evaluator 已完成；challenge set 按要求只保留建议 |
| 设备与采样 | 各设备独立训练/评估 | HRV loader 仍通过 `device_name` 一次选一台设备 | 一个共享模型联合训练 Earring/Ring/Watch，但每个样本只含一台设备；participant/device balanced sampler；部署固定 single-device green+IR，多设备同步输入只做融合上界 | 已实现，待正式比较 |
| 产物与复现 | checkpoint 主要保存 model/optimizer | 最新训练仍只把这两项写入 checkpoint；HRV direct 未保存完整逐窗记录，peak 只有可选 spot-check audit | 保存 scaler、target order、fold、config、history、RNG、每窗 CSV、run metadata 与 content-hash freeze manifest | 已完成 |
| 可信度 | 没有预测方差、校准或明确拒识语义 | 新增解码失败 coverage，但没有校准的预测 uncertainty | direct heteroscedastic head、beat SQI、confidence/reject reason、calibration 与 high-error overconfidence audit | 接口已完成；需正式训练验证 |

原代码还有一个可复现性细节：`train_sup()` 在调用 `set_seed()` 前先执行
`np.random.randint()` 来生成最终 seed，因此进程级初始 NumPy 状态可能改变结果。新 pipeline
直接使用固定整数 seed，并把 RNG state 写入 checkpoint。这个修正不回写伙伴目录。

### 论文带来的模型与评估补充

| 论文证据 | 对本项目的补充 | 可比性边界 |
|---|---|---|
| Kazemi et al., *Sensors* 2022，[DOI 10.3390/s22166054](https://doi.org/10.3390/s22166054) | 噪声下的 dilated CNN beat detection 支持保留独立 beat branch、dense heatmap 与 event-level 评价 | 论文包含重采样处理；不能据此把插值 PPG 重新设为本项目主输入 |
| Schäfer & Vagedes, *International Journal of Cardiology* 2013，[DOI 10.1016/j.ijcard.2012.03.119](https://doi.org/10.1016/j.ijcard.2012.03.119) | PRV 不能在所有状态下自动等同 ECG-HRV；因此主标签保持 ECG corrected HRV，并分别审计 beat timing、RR correction 和最终 HRV | 这是综述性证据，不是本数据集上的性能 benchmark |
| Bent et al., *npj Digital Medicine* 2020，[DOI 10.1038/s41746-020-0226-6](https://doi.org/10.1038/s41746-020-0226-6) | wearable optical measurement error受设备与活动条件影响，支持 per-device、motion 和 failure-tail 分层，而不是只报总体平均 | 主要研究 optical HR accuracy，不能直接给 RMSSD/SDNN 的目标误差 |
| Sarhaddi et al., *PLOS ONE* 2022，[DOI 10.1371/journal.pone.0268361](https://doi.org/10.1371/journal.pone.0268361) | awake/activity 条件下 HRV 通常比 sleep 更难，支持无 accel/标量 accel/原始 XYZ 消融以及 high-motion challenge | Samsung 手表和 24 h 场景不同，数值不能与本模型直接横比 |
| O’Grady et al., *Sensors* 2024，[DOI 10.3390/s24196220](https://doi.org/10.3390/s24196220) | 支持 serial measurement、agreement 与 Bland–Altman 报告，提醒不能只看相关 | 研究的是 Apple Watch 导出 SDNN；不是 continuous RMSSD benchmark |
| Flatt et al., *Sensors* 2025，[DOI 10.3390/s25103165](https://doi.org/10.3390/s25103165) | 随 HRV 增大出现 proportional bias/heteroscedasticity，支持 log target、预测不确定度、误差随 target 分层和 Bland–Altman | 受控 resting wristband 场景比本项目连续活动场景容易 |
| Sinichi et al., PsyArXiv preprint 2026，[DOI 10.31234/osf.io/qc2b3_v1](https://doi.org/10.31234/osf.io/qc2b3_v1) | 把 posture、activity、sleep/wake 下“何时有效”作为问题，支持 confidence 与 reject reason 成为正式输出 | 尚为 preprint，结论需标明证据等级 |
| Tao et al., *BMC Sports Science, Medicine and Rehabilitation* 2026，[DOI 10.1186/s13102-026-01960-x](https://doi.org/10.1186/s13102-026-01960-x) | 支持补充短窗结果，但短窗应使用自己的 ECG beat/RR label，不继承 5 min scalar target | 90 s、seated/low-motion、participant-mean 结果只能作辅助参照 |

### 由代码与论文共同形成的晋级门槛

后续模型不能只凭更低的总体 MAE 晋级，至少必须满足：

1. participant-macro RMSSD 与 SDNN 同时优于每折 train-only median constant；并报告
   r、R²、CCC、bias 和 participant bootstrap CI，防止再次出现“MAE 尚可但只会猜均值”。
2. 在相同 participant split 下报告 Earring/Ring/Watch、motion、PPG coverage、target
   magnitude 与 device×coverage；98–100% coverage 当前只有 127 rows，不能据小样本挑设备。
3. beat branch 必须先通过 event F1、timing MAE、count MAE 和 RR correction ratio，再看
   RMSSD/SDNN；不能只在能成功 decode 的窗口上算误差。
4. uncertainty 必须让保留比例下降时 risk 稳定下降，并检查 high-error overconfidence；当前
   SegNet 拒掉 50% 窗口仍没有稳定收益，未通过该门槛。
5. direct 与 beat 只有在 held-out inner validation 上表现出误差互补并超过最佳单路径至少
   0.5 ms MAE，才允许启用 fusion；outer test 不参与该决定。
6. 最终结果需另加未做 PPG coverage 预筛的真实 raw-session challenge set。在此之前只能称为
   “已预筛窗口上的 ECG-referenced HRV estimation”，不能声称全天候 continuous coverage。

## 缓存与 DataLoader 实测

新的 cache 派生文件按每个 participant/window/device/channel 单独保存 median、MAD、std；
只使用该 trailing crop 自身的 observed slots。300 s 与 60 s 使用不同文件，不共享统计，
也不拟合 train/global/test 分布。16 participants 的两个时长一次性回填耗时 39.47 s。
15 个真实 P5 window-device 的旧/新归一化最大绝对差为 `9.5367e-7`，自动测试容差为
`1e-6`。

Apple M4、macOS 26.6.2、Python 3.13.7、Torch 2.12.0；fold 0，5 warm-up batches，
100 train batches + 30 个顺序 validation batches。数值只涵盖加载、归一化与 collate。

| 设置 | train device-samples/s | sequential val device-samples/s |
|---|---:|---:|
| workers 0, threads 4, batch 4, cached stats | 4,003.6 | 6,115.3 |
| workers 0, threads 4, batch 4, legacy median/MAD | 1,160.4 | 1,257.4 |
| workers 0, threads 1, batch 4, cached stats | 658.7 | 1,377.9 |
| workers 0, threads 4, batch 2, cached stats | 911.1 | 6,223.5 |
| workers 0, threads 4, batch 8, cached stats | 831.8 | 1,814.7 |

同规格下 cached stats 对 train 是 3.45×、对 validation 是 4.86×。validation/test dataset
本来就按 participant → window → device 建立 refs 且 `shuffle=False`，现在正式保留该顺序，
减少 mmap/page-cache 抖动。

`num_workers=2/4` 在当前 Codex 沙箱无法启动 `torch_shm_manager`，报 `Operation not permitted`；
沙箱外 benchmark 的授权又被执行环境额度拒绝。因此没有把失败误写成“多 worker 更慢”，
也不会默认打开 persistent workers。换到正常终端后应原样重跑 `benchmark_loader.py` 再决定。

eager FP32 causal-direct forward + backward + AdamW（loader 不计入）实测：

| CPU threads | matched windows/batch | device samples/batch | s/step | samples/s |
|---:|---:|---:|---:|---:|
| 1 | 4 | 12 | 1.789 | 6.71 |
| 2 | 4 | 12 | 1.652 | 7.26 |
| 4 | 4 | 12 | 1.592 | 7.54 |
| 8 | 4 | 12 | 1.619 | 7.41 |
| 4 | 2 | 6 | 0.753 | 7.97 |
| 4 | 8 | 24 | 3.549 | 6.76 |

batch 2 只有约 5.7% 吞吐优势，却会在固定 windows/epoch 下翻倍 optimizer steps 并改变优化
协议；所以第一版 causal/beat 正式对比统一选择 threads=4、workers=0、batch=4、eager FP32。
`pin_memory`、FP16 不用于 CPU。`torch.compile`、CPU BF16、更多 workers 与并行 folds 均不
默认启用，必须另做完整 benchmark；并行 folds 尤其可能争抢同一 mmap/page cache。

采用上述设置的 causal-direct fold-0 开发短跑（2 epochs × 16 train batches，每次 inner-val
最多 8 batches）已通过，best checkpoint 在 epoch 1，inner-val loss 0.07427。该命令使用
`--skip-test-evaluation`，没有加载或评估 outer test；它只证明新路径可训练，不能作为模型
效果或调参结论。

## Beat 路径加速与无泄漏预训练

100–700 ms、20 ms step 的 31 个 delay candidates 已从 Python candidate loop 改成一次
batched gather/correlation。`B=12, T=30,000`、4 CPU threads、20 次调用的实测为：

| 实现 | ms/call | 数值差 |
|---|---:|---:|
| candidate loop | 10.515 | reference |
| batched gather | 5.224 | max abs = 0 |

即约 2.01×；训练仍只做 delay-marginalized loss，完整 beat decode 只放在 validation/test。
新增 `beat_pretrain_60s.json` 与强制 `--short-window-seconds 30/60`：这一阶段只优化 ECG beat
heatmap、offset、delay 和 SQI，不读取 outer-test waveform、不 decode outer test，也不把
5 min RMSSD/SDNN 标量继承给短 crop。1 train batch + 1 inner-val batch 的 smoke run 已通过。

训练入口支持按 fold 校验 split 后从 causal-direct checkpoint 初始化，并可先冻结 pretrained
shared encoder 若干 epoch 热身 beat/context head，再解冻；是否采用必须由固定 inner-val/单折短跑决定。开发期固定 seed，
额外 seeds 只跑最终候选，outer test 禁止调参。

beat 正式评估现在还会保存逐窗 one-to-one event TP/FP/FN、precision、recall、F1、timing
MAE 和 count error，并在 `metrics.json` 汇总。reference 定义为 ECG R peak 加模型预测的
ECG→PPG pulse delay，matching tolerance 固定 100 ms。这样能把“beat detection 是否正确”
与“RR 修正后 HRV 是否碰巧接近”分开检查；当前尚无正式训练完成的 beat checkpoint，所以
这里没有伪造或报告 event 数值。

## MPS 与后续数据管线

环境是 arm64 Apple M4，系统报告 `Metal: Supported`，Torch 报 `mps.is_built() == True`；但
`mps.is_available() == False`、device count 为 0。独立 Swift Metal 调用
`MTLCreateSystemDefaultDevice()` 也返回 `nil`，所以根因是当前执行沙箱没有暴露 Metal
device，而不是 Intel、旧 macOS 或 PyTorch 未编译 MPS。当前正式配置继续 CPU；未来迁移
CUDA 后才 benchmark autocast BF16/FP16、pin-memory/non-blocking 与更大 batch。

由于 5 min/30 s 的窗口重叠 90%，已对 v2 的可行性做完整审计：11,788 个窗口中有
11,118 对相邻连续窗口、670 个连续 runs；所有连续 pair 的 raw value、mask、jitter overlap
逐 slot 比较均为 0 mismatch。另从原始 NPZ 检查 P5 的 87 个连续 pair，raw timestamp 也是
0 mismatch。按 run 边界计算，canonical store 理论上可减少 84.88% 重复 slots。

独立 P5 prototype 已实际生成：v1 目录 112 MB，canonical v2 为 18 MB（84.19% slot reduction），
93/93 个窗口重建后的 value/mask/jitter 均完全一致。由此可以继续开发 session-level mmap
或 chunked Zarr，但仍不覆盖当前 window-cache。全量切换前还要验证 label mapping、统计
归一化、冷/热 page-cache 吞吐和随机 sampler 行为；保持为独立 `data_pipeline_v2`，不能
混入本次冻结 SegNet 或进行中的正式对比。审计和 prototype manifest 分别在
`artifacts/ml_hrv/benchmarks/window_overlap_audit.json` 与
`artifacts/ml_hrv/canonical_v2_prototype/P5/manifest.json`。

如以后降低 validation 频率，patience 必须按 validation events 而非 epochs 计数，并作为
所有比较模型共享的新协议。未做 coverage 预筛的 challenge set 仍建议从上游 raw session
构建并冻结，但按当前要求不在本阶段生成。

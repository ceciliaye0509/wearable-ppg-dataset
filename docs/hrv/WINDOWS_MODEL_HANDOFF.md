# Windows模型线迁移说明（2026-10-30）

这次迁移的工作是：在Windows工作站完成M1适配，再按同事配置复跑模型。Mac继续做heuristic研究。两条线使用同一份已冻结的v4.3 ECG参考。

## 项目从哪里看

```text
Daily_HRV/
├── wearable-ppg-dataset/       Git代码仓库
│   ├── docs/hrv/              当前状态、迁移说明、历史模型报告
│   ├── experiments/hrv_v4_3/   本次数据绑定、四折、配置、源窗口索引
│   ├── tools/                 迁移完整性检查、模型环境检查
│   ├── requirements/          Windows模型环境依赖
│   ├── research/ecg_v4_3_release/  已完成ECG发布流程的代码留档
│   └── src/
│       ├── model_baselines/   同事Kazemi峰检测对照；本次模型主线
│       ├── ml_hrv/            之前的直接HRV/SegNet研究；历史与次级比较
│       ├── heuristic_baselines/  heuristic、极性研究、人工标注网页
│       └── prepare_windowed_dataset/  早期切窗/准备代码
├── data/ecg_v4_3/             在新电脑上复制的完整冻结数据，不进Git
└── transfer/                 离线Git bundle、迁移说明、验证记录
```

源Mac的`analysis/`保存实验记录和历史数据集，目前没有Git仓库。正式数据仍留在原路径，不移动、不删除。`hrv-research-notes/`是另一个笔记仓库；这次模型迁移不依赖它。旧分析脚本、默认HF下载和早期切窗入口均不能替代已验收的v4.3包。

## 当前固定的东西

- v4.3：15个participant文件；14,036候选窗、10,271合格ECG参考、3,765不合格。14人有合格标签。
- ECG协议：`d39ea6d77a4f753824948ab25b77e08ab62c457a5cc8e1b67dfcb9b341ee95b8`。
- 数据文件清单SHA256：`bcd593f655b52d6d0217a01d456b3cdb33382d91e09a2e7fbd8b3909ab23d0d0`。
- 保留97个局部人工修复窗、7个既有拒绝窗和4个完整人工G1例外；2,717负向分钟、34个翻转区间。
- 主参考是校正NN代理；原始RR目标另作同窗次级分析。模型或PPG得分不能用来挑选ECG真值。
- heuristic仍为`qppgfast_devicewise_corr`。六通道宏平均RMSSD MAE/r为41.432342ms/0.263864，SDNN为45.070483ms/0.459693；coverage为合格参考的80.761367%、全部候选的59.098034%。这些不是模型成绩。
- 模型未按v4.3重训，HF尚无这个版本。旧标签缓存和旧scaler不能复用。

模型对照规格在`experiments/hrv_v4_3/model_comparison.json`：Earring绿光、100→50Hz、导数通道、Kazemi、batch8、lr1e-4、最多20epoch、patience5、实际训练seed1。沿用同事原四折，只排除P5/P6，不重新分配剩余participant。v4.3窗口数已重新计算：

| 折 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 0 | 6,230 | 1,142 | 2,899 |
| 1 | 6,389 | 1,158 | 2,724 |
| 2 | 6,808 | 1,322 | 2,141 |
| 3 | 6,870 | 894 | 2,507 |

这些也是当前Earring绿光输入筛选后的窗口数。模型目标字段按名字读取：同事代码的顺序是SDNN、RMSSD，旧`ml_hrv`的顺序相反。

## 只需要复制两项

1. `analysis/windows_model_transfer_20261030_v1/`整个迁移目录，包含离线Git bundle和验证记录。
2. `analysis/ecg_training_release_v4_3_export12_20261029_v1/dataset/synced_3device_rawaligned_training_v4_3_stride30_rawslots_fullmanual_export12_20261029/`整个数据目录，约14.14GiB。

把第二项复制为`D:\Daily_HRV\data\ecg_v4_3`，保留里面所有文件名、子目录和`checksums.json`。把第一项复制为`D:\Daily_HRV\transfer`。无需复制Mac虚拟环境、旧缓存、全部原始录制或其它历史数据版本。缓存和输出建议放SSD，先检查SSD单独剩余空间。

在Windows PowerShell中恢复代码：

```powershell
cd D:\Daily_HRV
git clone -b hrv-followup .\transfer\wearable-ppg-dataset.bundle wearable-ppg-dataset
cd wearable-ppg-dataset
git switch -c model-v4.3-workstation
git remote set-url origin https://github.com/ceciliaye0509/wearable-ppg-dataset.git
```

bundle包含这次本地commit，不依赖先推送GitHub。`transfer_receipt.json`记录应恢复的commit；核对`git rev-parse HEAD`。Mac和工作站后续使用不同工作分支和输出目录，避免互相覆盖。

## Windows环境和迁移检查

使用Python3.11及以上；下面以3.11为例。源仓库的`src/model_baselines/environment.yaml`是旧Linux/Python3.8环境导出，不能直接用于这台Windows电脑。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements/model-workstation.txt
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
nvidia-smi
.\.venv\Scripts\python.exe tools/verify_hrv_transfer.py --data-dir D:\Daily_HRV\data\ecg_v4_3 --report runs/transfer_verification.json
.\.venv\Scripts\python.exe tools/check_model_workstation.py --require-cuda --report runs/workstation_environment.json
```

PyTorch2.12.0/CUDA12.6安装方式来自[官方版本说明](https://pytorch.org/get-started/previous-versions/#v2120)。安装前确认驱动兼容该运行时；具体GPU算子以本机前后向检查为准，不以显卡型号或`is_available()`单项代替。源码Mac使用torch2.12.0；同事实际环境尚未完整提供，不能宣称跨机器逐位复现。环境检查只使用合成batch8、2输入通道、15,000采样点，执行前后向、不更新参数，不证明真实数据训练已通过。

数据检查默认核对全部冻结校验和及四折计数；`--quick`只作排错，会跳过大文件哈希，不算完整迁移验收。工具不会重建标签或改写数据。输出保存在被Git忽略的`runs/`。

## 到新电脑后先做什么

本次提交完成了迁移准备，**没有完成M1，也没有提供可直接开始正式训练的四折启动命令**。`model_comparison.json`是待实现的对照规格，旧训练脚本不会自动读取它。

按下面顺序继续：

1. 让四折入口读取已保存的14人划分。旧`_group4_split`仍硬性要求16人，直接运行会报错；不得通过重分组绕过。
2. loader、缓存和预测导出同时保留完整NPZ行号、`source_window_index`及窗口起止时间；筛选后行号不能冒充原始窗口索引。导出候选窗拒绝原因，保留各coverage分母。
3. 用v4.3重建50Hz输入/峰目标缓存。按participant分批处理，缓存绑定数据、协议和预处理版本；训练集scaler只在本折训练participant上拟合。与直接读取v4.3的小样本逐项核对。
4. 真数据batch8完成前后向及一次小样本流程检查，核对峰目标、单位、输出长度、索引和验证选阈值逻辑，之后再完整跑四折。
5. 先报告固定同事配置结果，再预先固定少量优化方案。checkpoint和decoder阈值只用验证participant选择；测试集只做最终评价。旧预测换参考的影响与真正重训的变化分开报告。

Windows做模型时，Mac可继续研究heuristic。新heuristic实验写新目录，不覆盖当前冻结baseline或ECG标签。测试集调参、恢复已关闭的时间戳/来源问题、修改ECG门槛均不属于这次模型迁移。

## 给工作站上的助手直接使用

> 先阅读docs/hrv/WINDOWS_MODEL_HANDOFF.md、experiments/hrv_v4_3/model_comparison.json、folds.json和dataset_manifest.json。我的数据放在D:\Daily_HRV\data\ecg_v4_3。先执行完整迁移校验与CUDA合成前后向检查，再实现本文列出的M1适配和新缓存；小样本核验通过后执行同事配置的模型基准。保留固定四折、seed1、Earring绿光50Hz和原训练预算；保留原始窗口索引与所有分母。不要修改已冻结ECG参考，不要使用旧v4.1标签缓存或历史scaler。每一步报告实际结果后继续。

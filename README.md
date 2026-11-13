# Wearable PPG HRV

正式代码保留两条独立流程：**HF raw-slot v1** 和 **v4.6 segment250**。两者只共享无版本含义的数值/IO工具；生成阶段均从原始 PPG、ECG 计算窗口，不读取另一版本的数据集。

## 目录

```text
src/hrv/
  common/       切窗、raw-slot、IO、RR、qppg与评估工具
  hf_rawslot/   HF早期版本：generate / baseline / model
  v46/          v4.6：generate / PPG分段 / ECG / baseline / model
src/model_baselines/  其他作者的模型，原路径、原内容和文件名保留
configs/v46_reference/ 人工修复、源极性及排除记录；不是旧版窗口或自动标签缓存
```

自己的模型和未定稿研究位于仓库外 `../research_workspace/`；一天 ECG/活动变化等记录在其 `supplementary_experiments/`。旧 `rawaligned` 生成入口已退出正式源码，其必要数值函数已提取到 `common/`。数据集名称中的历史 `rawaligned` 字样保留，避免改坏已有文件身份。

## 运行

在仓库根目录安装 `pip install -e '.[model]'`，或设置 `PYTHONPATH=src`。原始数据需使用对应版本记录的源文件；v4.6参考极性操作绑定 HF `snowballlab/Multisite-PPG` 的提交 `126068a2b550a935d3503d15169f79c3dd02c368`，不能任意替换成极性状态不同的原始数据。

### HF raw-slot v1

```sh
python -m hrv.hf_rawslot.generate \
  --raw-root /path/to/raw_data \
  --dataset-name synced_3device_rawaligned_training_v1_stride30_rawslots \
  --out-dir /path/to/hf_rawslot
python -m hrv.hf_rawslot.baseline --dataset-dir /path/to/hf_rawslot --output-dir /path/to/results/hf
```

该生成器取自 `bcf9cf4` 发布时期：300s/stride30/100Hz，原始槽位容差5ms、覆盖率90%，Pan-Tompkins ECG-QC参与窗口筛选，默认排除 P2/P14/P16/P17，**不排除P5**。它不同于后来的 Pan-ungated 工作区版本。原始数据若不同于当时发布源，不能仅凭代码相同声称逐字节复现。

模型仍使用同事原代码，参数原样转发：

```sh
python -m hrv.hf_rawslot.model --task peak --device_name earring \
  --peak_backbone kazemi --data_dir /path/to/hf_rawslot
```

### v4.6

```sh
python -m hrv.v46.generate --raw-root /path/to/raw_data --output-dir /path/to/v46
python -m hrv.v46.baseline --dataset-dir /path/to/v46 --output-dir /path/to/results/v46
python -m hrv.v46.model eval --dataset-dir /path/to/v46 --output-dir /path/to/results/model \
  --folds /path/to/folds.json --checkpoint-dir /path/to/checkpoints --decoder-dir /path/to/decoders
```

固定300s/stride30/100Hz。用原始未增强PPG计算公共时间窗、边界与90%槽位覆盖，之后才做源QC、最多250ms插值和长缺口分段；新边界保护3s，禁止跨片段构造RR。模型按片段处理，在一个原窗口内归一化，训练与解码共享有效区。观察不足标记默认仅供复查。

ECG从原始信号重新检测：Rodrigues2021、局部50ms、受保护启动上下文、固定G0/G1/G2和RR规则。参考输入保留明确批准的人审锚点修复、四个整窗例外和拒绝决定，按绝对时间匹配，不扩大人审范围。

v4.6以每32窗一个NPZ分片保存，减少一次性内存；`dataset_manifest.json`列出分片。不要用旧整窗loader直接读取，应使用本版本的入口。`--limit`可作小规模检查，`--window-start-ms`可重建指定绝对时间窗口。HF版本仍保持原参与者NPZ格式。

分段训练适配器也可用 `python -m hrv.v46.model train ... --folds ...`，只在train更新、val选模，不在test调参。此为新训练，不代表旧权重已重训完成。自己的研究模型入口见 `../research_workspace/models/mymodel.py`。

## 验证与边界

`python -m unittest discover -s tests` 检查切窗与版本依赖、缺口和RR邻接、masked loss及HF发布口径。迁移时固定源样例的数值对照记录见 `docs/RESTRUCTURE_VERIFICATION.json`。

重构保持原科学规则；新的独立生成流程已进行固定样例验证，尚未全量重建20人数据或重新训练完整模型。现有冻结数据、标注、权重与正式数据指针保留原位。此前的版本封存脚本属于历史证据，不作为新入口。

## 复查说明（2026-11-13）

HF模型包装入口要求显式指定`--data_dir`和`--task`，相对数据路径按调用位置解析。其他作者原模型中的默认目录及旧`--segment-dataset-manifest`分支原样保留，但新HF入口不会走这些旧默认/分支；分段数据使用`hrv.v46.model`。

v4.6训练按历史的参与者、参与者内窗口打乱顺序分批；目标固定为Kazemi五点标记，解码固定300ms间距，不依赖外部同名环境变量。初始化顺序也与历史适配器对齐。HF baseline除新的按通道汇总外，另写`historical_*_summary.csv`保留原参与者平均口径，两种汇总不可混称。

仓库外qppg残差实验需显式提供`--qppg-feature-csv`；不再默认读取旧特征缓存。

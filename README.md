# MTG-GAN 复现代码

验证记录入口：[总览、主线与待办](summary.md) · [最新有效结果](results.md) · [修改与验证历史](history.md)。每轮验证后同步维护三份记录；详细检查日志保留在 `docs/`。

当前实验范围为 **BGL / Spirit / Thunderbird 六方向迁移**：[任务清单](benchmarks/bgl_spirit_thunderbird/manifest.json) · [改造方案](docs/six_direction_upgrade_plan.md) · [新版改动与硬件评估](docs/hardware_and_v2_changes.md)。`inductive_v2` 代码、六份配置与串行入口已完成，静态检查通过；**按用户要求，运行测试与实验均待批准**。

依据 Jing Zhang、Chao Luo、Rui Shao 的 *MTG-GAN: A Masked Temporal Graph Generative Adversarial Network for Cross-Domain System Log Anomaly Detection*（DOI `10.1109/TNSM.2026.3654642`）独立实现，并参考张警学位论文第三章。

**当前已开始 BGL → Thunderbird 小样本真实实验，最新进度与指标见 [summary.md](summary.md) 和 [results.md](results.md)。** 尚未进行全量论文性能复现。这不是作者原始实现；实现假设见 [方法对应说明](docs/methodology.md)。

## 已实现

- 原始日志读取、Drain、时间窗/会话切分、增量多阶时序图。
- 冻结 Sentence-BERT 特征、GCN 对比预训练、因果膨胀 TCN。
- Gumbel-Softmax 与随机双掩码、动态混合权重、下一图生成。
- Wasserstein GAN、梯度惩罚、特征扰动损失、广义谱损失。
- 无标签目标域适配、关键事件局部评分、全局谱评分、历史动态阈值。
- 分阶段 CLI、检查点、逐样本预测 CSV、Precision/Recall/F1/Accuracy/AUC。

## 目录

```text
mtg_gan/       config、data、dataset、model、losses、scoring、training、cli
configs/       七个迁移方向的配置
  inductive_v2/ BGL / Spirit / Thunderbird 六方向新版配置
scripts/       生成配置，不启动实验
tests/         合成日志、模型数值和完整链路检查
docs/          方法对应与验证记录
summary.md     当前主线、缺点和待补实验
results.md     最新或最佳的有效结果
history.md     按时间追加的修改尝试与结果
runs/          后续运行时生成的预处理、模型和结果
prepared/      新版按域与指纹共享的数据、语义缓存（待运行生成）
source_runs/   新版每域每种子的源检查点（待运行生成）
```

## 安装

建议 Python 3.10+，先按硬件安装 PyTorch，再执行：

```powershell
Set-Location 'D:\PaperWork\毕业论文\mtg_gan_reproduction'
python -m pip install -e '.[test]'
python -m mtg_gan --help
python -m mtg_gan inspect --config configs/bgl_to_thunderbird.json
```

`inspect` 只读取首行和文件大小，不加载语义模型、不预处理全量日志、不训练。配置中的相对路径以配置文件所在目录为基准。

默认语义模型为 `sentence-transformers/all-MiniLM-L6-v2`，384 维，两域共享。这是论文未指定具体模型时的复现选择。`local_files_only=true`，当前不下载权重。后续需提前缓存模型，或把 `semantics.model` 指向本地 SentenceTransformer 目录；允许下载时可自行改为 `false`。更换编码器需同步修改 `model.input_dim` 并重新编码、训练。

本机原先全局缓存缺少权重；现已将官方固定版本下载到 `models/all-MiniLM-L6-v2/`，校验并验证离线加载成功。pilot 配置已指向该本地目录，其他配置若仍使用模型名称，需要将 `semantics.model` 改为这个目录的绝对路径。生产编码器没有随机特征回退。

权重准备脚本为 `python scripts/download_minilm.py`，只下载指定 revision 的权重、分词器和配置，不下载其他推理格式或执行远程代码，校验清单写入模型目录。脚本需要网络访问。

## 数据

输入目录：`D:/PaperWork/count_granger/dataset`。原始文件只读，不搬移。

| 系统 | 文件 | 划分 |
|---|---|---|
| BGL | `BGL/BGL.log` | 6h 时间窗，1h 步长 |
| Thunderbird | `Thunderbird/Thunderbird.log` | 1h 固定时间窗 |
| HDFS | `HDFS/HDFS.log`、`HDFS/anomaly_label.csv` | 完整 block 会话 |
| OpenStack | `OpenStack/OpenStack_labeled.log` | 本地统一标签格式，1min 时间窗 |
| Spirit | `Spirit/Spirit1G.log` | 扩展设置：1h 时间窗 |

论文的日志跨域任务为 BGL↔Thunderbird。另有 BGL↔HDFS、BGL↔OpenStack、BGL→Spirit 的扩展配置；Spirit 未出现在论文实验中。指定目录没有 SWaT/WADI，未实现其工业时序处理器。

默认 60/20/20 切分是补全设置。时间窗数据先按原始记录排序切分，再各自构窗；HDFS 按 block ID 稳定哈希分组。Drain 仅拟合各域训练部分。旧版未见模板进入 UNKNOWN；新版保留冻结匹配后未匹配的规范化文本并使用冻结编码器表示，训练频次仅统计训练部分，候选仅来自历史和训练词表。候选外的新事件记录 novelty 分量，不进入生成器历史输入。`max_nodes` 超出会报错，不静默截断。

`max_lines=null` 为全量读取。Thunderbird 约 31.8GB，SQLite 预处理需要足够磁盘。调试可显式限行，HDFS 还必须设置 `allow_partial_sessions=true` 才允许截断会话。前缀子集结果不能视为全量结果。

## 六方向新版入口（待批准运行）

新架构在 `mtg_gan/inductive_model.py`，旧架构保留为 `legacy`。新版共享节点对掩码/解码器和图读出消除编号专属参数，并加入磁盘缓存、时间边界、连边预热、谱归一化与分项校准。这是归纳式扩展，检测效果尚未验证。

```powershell
# 默认仅显示计划；本轮未运行，以下作为后续操作入口
python scripts/run_six_directions.py
# 用户批准并通过合成、数据和资源检查后，串行运行单种子六方向
python scripts/run_six_directions.py --execute --seeds 42
# 方法固定后再开展三个种子，完整汇总要求各方向覆盖一致
python scripts/run_six_directions.py --execute --seeds 42 43 44
python scripts/summarize_suite.py
```

六份配置位于 `configs/inductive_v2/`，均使用指定目录的三份完整日志与项目内本地 MiniLM。每种子训练三个源模型，六次目标适配独立从各自源模型开始；共享按域缓存。默认 batch=1、512 候选，源正常验证选模与校准，目标训练不读标签，动态阈值关闭。候选容量不足会报错，实际全量容量与资源仍须检查。

## Legacy 分阶段命令（诊断基线）

以下为全量配置的分阶段命令；当前小样本实验使用独立的 `configs/bgl_to_thunderbird_pilot_20261026.json`，详情见三份记录入口。

```powershell
# 预处理两域并编码训练词表
python -m mtg_gan prepare --config configs/bgl_to_thunderbird.json
python -m mtg_gan encode --config configs/bgl_to_thunderbird.json

# 源域正常样本：GCN 预训练和 GAN 训练
python -m mtg_gan pretrain --config configs/bgl_to_thunderbird.json
python -m mtg_gan train-source --config configs/bgl_to_thunderbird.json

# 适配前评估：源域测试、目标域测试
python -m mtg_gan calibrate --config configs/bgl_to_thunderbird.json --stage source
python -m mtg_gan evaluate --config configs/bgl_to_thunderbird.json --stage source --role source
python -m mtg_gan evaluate --config configs/bgl_to_thunderbird.json --stage source --role target

# 目标域训练部分无标签适配，然后重新校准和评估
python -m mtg_gan adapt-target --config configs/bgl_to_thunderbird.json
python -m mtg_gan calibrate --config configs/bgl_to_thunderbird.json --stage target
python -m mtg_gan evaluate --config configs/bgl_to_thunderbird.json --stage target --role target
```

`prepare` 不覆盖已有结果，修改切分、窗口、容量时请换 `output_dir`。训练会覆盖同名阶段检查点，变更实验设置也应使用新目录。所有轮数、超参数和损失权重都是未调优的复现默认值。

默认阈值用源域正常验证样本校准。可设 `score.calibration=target_unlabeled_train` 使用目标训练分数分位数；此方式不读取目标标签，可能受到异常污染，两种协议须分开报告。动态阈值只利用过去已接受分数。输出分阶段检查点、损失历史、校准参数、逐样本 CSV 和评估 JSON；单一类别的 AUC 为 `null`。评分设置或检查点变化后需重新校准。

## 验证和资源

```powershell
# 本机如出现 NumPy/PyTorch OpenMP 冲突，仅对当前终端设置：
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONUTF8 = '1'
python -m pytest -q --basetemp tmp/pytest
```

测试使用临时合成日志及特征，不下载模型，不训练真实数据。**当前新版包含 24 个测试函数，本轮只做静态编译，尚未执行测试。** 旧版 15 项通过不能代表新版已通过。检查记录见 [validation.md](docs/validation.md)。

使用原生 PyTorch，不依赖 PyG/DGL。稠密掩码约 `O(B W N²)`，谱运算约 `O(B N³)`。本机 i7-14650HX、约 16GB RAM、RTX 5060 Laptop 8GB 具备新版串行验证的开跑条件，仍需测量全量节点容量和峰值；详见 [资源评估](docs/hardware_and_v2_changes.md)。旧版槽位对应仍是诊断缺口，新版拟通过参数共享修正，尚未验证迁移效果。

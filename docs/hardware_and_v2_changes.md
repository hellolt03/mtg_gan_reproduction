# 六方向新版代码与本机资源评估

更新：2026-10-27。按用户要求，本轮完成代码、配置和静态检查，**不运行单元测试、模型前向、语义编码、数据预处理、训练或评估**。运行效果需要用户同意后验证。

## 已完成的代码改动

旧实现继续使用 `model.architecture=legacy`，旧 pilot 检查点和指标保留。六份新配置使用 `inductive_v2`，属于针对已发现缺陷的归纳式扩展，不能称为作者原实现或已经复现论文性能。

| 改动 | 目的 | 代码 |
|---|---|---|
| 共享节点对掩码及有向 Q/K 解码器 | 消除新模板对应未训练输出行的问题，参数不依赖模板编号或词表容量 | `mtg_gan/inductive_model.py` |
| 图编码与 mean/max 读出判别器 | 替换展平 N² 的巨型 FFN，保留语义和连边信息 | `mtg_gan/inductive_model.py`、`factory.py` |
| 三域独立、六方向共享磁盘缓存 | 原始日志流式进入 SQLite；同域切分、Drain、特征复用，减少重复工作 | `mtg_gan/cache.py` |
| 片段边界、时间空档与节点出现掩码 | 不跨 split、采样 segment 或超过配置的空档建立历史，区分出现节点和预测候选 | `mtg_gan/inductive_dataset.py` |
| 正负边平衡 BCE、源域预测预热 | 为稀疏连边预测提供直接学习信号，再加入 GAN | `mtg_gan/losses.py`、`training.py` |
| 归一化 Laplacian 与按节点数缩放 | 缓解旧热核饱和及规模效应，增加谱梯度、预测密度日志 | `mtg_gan/losses.py`、`scoring.py`、`training.py` |
| 源正常验证集分项校准 | 局部、全局与新节点分量分别按中位数/IQR 标准化后融合，固定源阈值作为首轮协议 | `mtg_gan/scoring.py`、`training.py` |
| 源模型复用、独立适配、汇总 | 每种子训练三个源模型，分别用于两个目标；缺方向或种子不齐时不输出六方向宏平均 | `scripts/run_six_directions.py`、`summarize_suite.py` |

数据使用现有三份文件的完整范围：BGL.log、Spirit1G.log、Thunderbird.log，原文件只读。Spirit 是现有 1G 文件范围。时间排序后按原始记录 60/20/20 切分，再分别构窗；BGL 6h 窗口/1h 步长，其他两域 1h/1h。

Drain 仅在各域训练部分拟合，之后冻结；未匹配文本保留规范化文本，由冻结 MiniLM 编码。虽然语义缓存可预先存储各 split 的冻结推理结果，生成器候选只来自历史出现节点和训练频次词表，未来图的边、出现节点及标签不参与候选选择。未来落在候选外的事件不静默消失，而记录为 `novelty` 比例。目标训练标签不用于筛选或选模；源训练和源验证的正常筛选会使用源标签。

源模型按正常验证集的平衡连边损失选模，目标适配固定 5 轮；不使用目标测试 F1 选轮数。默认预训练 10 轮、源 GAN 最多 20 轮、源预测预热 3 轮，均是待验证的初始设置。分项缩放不能保证改善排序或 AUC，新增损失和新节点权重也需要消融。

## 六方向配置与执行入口

配置位于 `configs/inductive_v2/`：BGL→Spirit、BGL→Thunderbird、Spirit→BGL、Spirit→Thunderbird、Thunderbird→BGL、Thunderbird→Spirit。

目录分工：`prepared/inductive_v2/{domain}/{fingerprint}/` 共享数据；`source_runs/inductive_v2/{source}/seed{seed}/` 源模型；`runs/inductive_v2/{source}_to_{target}/seed{seed}/` 独立适配与评估。上述运行目录本轮尚未创建。原始输入、语义模型路径已做存在性检查。

以下命令留待批准后使用，本轮没有执行：

```powershell
Set-Location 'D:\PaperWork\毕业论文\mtg_gan_reproduction'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONUTF8 = '1'
# 先做合成检查与资源验证，不能将其当作真实检测成绩
python -m pytest -q --basetemp tmp/pytest_inductive_v2
# 默认只显示六方向计划，不预处理、不训练
python scripts/run_six_directions.py
# 正式执行一个种子的六个方向；获得用户批准后再启动
python scripts/run_six_directions.py --execute --seeds 42
# 协议固定后再开展三种子实验
python scripts/run_six_directions.py --execute --seeds 42 43 44
python scripts/summarize_suite.py
```

## 本机硬件与运行判断

2026-10-27 通过只读系统查询核实；可用内存、磁盘和显存会随进程变化。[CPU/内存/磁盘快照](../benchmarks/bgl_spirit_thunderbird/hardware_20261027.json)。GPU 使用 `nvidia-smi` 读取，没有加载模型。

| 资源 | 本次读取 | 判断 |
|---|---|---|
| CPU | i7-14650HX，16 核/24 线程 | 具备解析、CPU 语义编码和缓存准备条件；完整日志解析仍可能耗时较长 |
| 内存 | 15.73 GiB，总量约 16GB；当前可用 6.44 GiB | 是主要约束，准备与训练串行，关闭其他占内存应用；不把全量日志读入内存 |
| GPU | RTX 5060 Laptop，8151 MiB；当前空闲 7842 MiB | 对新版 batch=1、候选上限 512 的初始配置具备开跑条件，峰值待测 |
| 磁盘 | D 盘空闲 222.23 GiB；C 盘空闲 280.38 GiB | 可开始准备；SQLite 排序、索引与 VACUUM 会增加临时占用，最终容量仍须测量 |
| Python | 3.13.9 | 已完成静态编译，当前新版依赖运行兼容性还没验证 |

**结论：可以开始分阶段、串行验证，无需先升级硬件；不能据此保证全量六方向所有设置都能直接完成。** 初始配置 batch=1、上下文 4 图、每样本最多 512 候选、MiniLM 在 CPU 按 32 模板编码、DataLoader 单进程。六方向不并发，语义编码与训练也不并发。32GB 内存会更宽裕，但并非本轮开跑前提。

按当前层定义静态计算，新版生成器约 115,969 参数、判别器约 41,601 参数，总计约 15.76 万；不是实例化实测值。旧判别器审计值为 94,638,273。参数减少不等于显存和耗时按比例减少：稠密图、中间激活、梯度惩罚和 O(N³) 谱运算仍有成本。

`max_nodes=50000` 是缓存词表容量，**不是每张图分配 50000×50000 矩阵**。单样本候选上限由 `candidate_nodes=512` 单独控制。如果历史实际需要超过 512 节点，程序明确报错，不截断历史；届时依据训练分区节点数和资源测量决定增大容量或调整窗口，不能按测试成绩选择。Thunderbird 原文件约 31.8GB，尚未测量新版全量处理时间或空间，暂不给耗时承诺。

## 本轮检查与批准后待办

[静态检查产物](../benchmarks/bgl_spirit_thunderbird/static_validation_20261027.json)：模块、脚本和测试语法编译通过；六方向、三份源模型路径、共享域协议、原文件及本地权重路径检查通过；没有创建新版缓存、源训练或迁移运行目录。现有 24 个测试函数，其中新增 9 个，**本轮均未执行**。旧版 15 项通过仅是历史事实，不验证新版。

批准后先执行合成测试（包括置换、padding、无未来输入、片段边界、梯度、校准、源模型复用和旧检查点兼容），再在固定连续数据上检查实际节点容量、数据可用性、RAM/显存及谱梯度。确认三域源模型能学习后，再依次完成六方向一个种子；持久性预测基线、结构/损失/评分消融、目标污染和动态阈值仍待补。未有新版 F1、AP 或 AUC，不能声称本次修改已改善检测效果。

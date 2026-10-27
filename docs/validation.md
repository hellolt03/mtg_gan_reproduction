# 代码验证记录

持续记录入口：[总览](../summary.md) · [最新有效结果](../results.md) · [修改历史](../history.md)。本页保留初始代码检查的详细证据；后续新轮次同步更新三个入口。

日期：2026-10-26。本页保留初始代码实现阶段的检查，彼时未开展真实训练。后续真实 MiniLM 编码及首轮小样本实验已完成，见 [实验报告](experiment_bgl_thunderbird_pilot_20261026.md) 和 [最新结果](../results.md)。

## 已完成

- `python -m compileall -q mtg_gan scripts`：通过。
- `python -m mtg_gan --help`：八个分阶段命令均可列出。
- `python -m pytest -q --basetemp tmp/pytest_delivery`：最终复查 **14 passed in 12.83s**。
- 七份配置均能解析，源/目标原始日志路径存在，HDFS 标签文件存在。
- BGL、HDFS、OpenStack、Spirit、Thunderbird 各只读抽取前 **1,000 行**，读取器全部成功解析；不是全文件格式保证。
- 真实数据运行目录 `runs/` 未创建。
- 在 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下尝试真实 MiniLM 离线加载：缓存配置存在，但加载因缺少 `pytorch_model.bin` / `model.safetensors` 失败。没有下载权重；真实语义编码未验证成功，已在 README 说明准备方法。测试中的语义接口使用替身编码器，仅验证形状和调用契约。

## 测试覆盖

| 范围 | 实际检查 |
|---|---|
| 原始格式 | BGL epoch 与微秒、缺省位置字段、syslog 组件、Spirit 自由文本冒号、OpenStack 统一格式、HDFS 多 block |
| 时序构图 | 有向边、多阶边、自环、训练/验证/测试时间边界 |
| 词表 | 训练部分拟合、测试新模板进入 UNKNOWN、容量超限拒绝 |
| 标签隔离 | 目标训练批次没有标签，翻转目标标签不改变训练张量，禁止目标正常标签筛选 |
| HDFS | 完整 block 分组、稳定 split、前缀不包含最后事件的边 |
| 语义接口 | 共享模型参数、离线选项、填充零特征、维度不符报错 |
| 模型 | TCN 因果性、掩码梯度、填充节点零预测、确定性评分 |
| 谱运算 | 同图零迹权重损失、重复特征值下有限梯度、对称拉普拉斯 |
| 评分 | 扰动公式两方向区分、InfoNCE、历史阈值先判断后更新 |
| 全链路 | 临时合成日志→Drain→合成特征→GCN 预训练→源训练→目标适配→校准→评估与 CSV |

合成全链路每阶段只有一个 epoch，特征由测试夹具生成；这只验证程序执行，不作为任何数据集的检测成绩。生产编码器路径不会悄悄回退为随机或哈希特征。

## 环境

Python 当前环境中验证的版本：PyTorch `2.12.0+cu132`，NumPy `2.3.5`，scikit-learn `1.7.2`，Drain3 `0.9.11`，sentence-transformers `5.5.1`。CUDA 可用，训练设备默认自动选择。命令仅对当前进程设置 `MKL_THREADING_LAYER=SEQUENTIAL`、`PYTHONUTF8=1`。

## 尚未验证

全量数据解析、实际 Drain 模板数及内存、完整节点容量下的 GPU 训练资源、训练收敛、任何真实数据集 Precision/Recall/F1/AUC、与论文表格的性能一致性。方法缺口见 [methodology.md](methodology.md)，不能把合成检查通过等同于论文结果复现。

## 启动真实实验后的补充检查

增加 padding 缩减保持稀疏图不变的测试，并验证排除谱分解中的纯 padding 后全局范数保持不变。复查 `python -m pytest -q --basetemp tmp/pytest_pilot_scoring`：**15 passed in 15.16s**。官方真实 MiniLM 的 SHA-256、离线加载、两域模板编码均通过；真实 pilot 的训练与评估情况单列在实验报告，不覆盖以上历史检查。

## 2026-10-27：新版仅静态检查，运行待批准

`inductive_v2` 代码与六方向配置已写入。`python -m compileall -q mtg_gan scripts tests` 通过；纯配置检查确认六个有向组合、同域配置一致、共享三源路径、日志及本地权重存在，且没有新版缓存或运行目录。[静态产物](../benchmarks/bgl_spirit_thunderbird/static_validation_20261027.json)。

AST 统计为 24 个测试函数，其中新版文件新增 9 个。**本轮没有执行 pytest、模型实例化/前向、编码、预处理、训练或评估**；上面的 15 项通过属于旧版修订。待运行范围包括置换/padding、参数与词表容量解耦、无未来输入、片段边界、梯度、分项校准、源检查点复用、旧配置兼容和合成全链路。详见 [本轮说明](hardware_and_v2_changes.md)。

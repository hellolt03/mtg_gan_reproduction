# MTG-GAN 最新有效结果

更新：2026-10-27。总览见 [summary.md](summary.md)，过往修改与检查见 [history.md](history.md)。本页只保留当前最新或最佳的有效结果。

## 真实数据实验

最新结果为已完成的 BGL → Thunderbird 小样本 pilot。**链路已跑通，当前尚未形成有效跨域检测。** [配置](configs/bgl_to_thunderbird_pilot_20261026.json) · [完整报告](docs/experiment_bgl_thunderbird_pilot_20261026.md)。

| 评估对象 | 测试数 | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 源模型 → BGL | 312 | 80.00% | 10.13% | **17.98%** | 0.6851 |
| 源模型 → Thunderbird | 14 | 21.43% | 100.00% | **35.29%** | 0.2121 |
| 适配后 → Thunderbird | 14 | 21.43% | 100.00% | **35.29%** | 0.1818 |

目标适配前后均为 TP=3、FP=11、TN=FN=0，全部判异常。源域 TP=8、FP=2、TN=231、FN=71。目标标签不参与训练和校准；两阶段分别以源域 113 个正常验证样本校准，初始阈值 9.2836 / 9.6555。目标仅 3 个异常，且输入由不连续片段抽样得到，不能与论文全量结果直接比较。

原始指标：[源域](runs/bgl_to_thunderbird_pilot_20261026/source_source_test_metrics.json) · [直接迁移](runs/bgl_to_thunderbird_pilot_20261026/source_target_test_metrics.json) · [适配后](runs/bgl_to_thunderbird_pilot_20261026/target_target_test_metrics.json)。本页保留本轮可比结果，过往实现检查过程见历史。

新版 `inductive_v2` 代码与六份配置已写入，但按用户要求尚未运行测试、预处理、训练或评估，以上旧版指标保持不变。三域六方向没有新版性能结果；[任务清单](benchmarks/bgl_spirit_thunderbird/manifest.json) 明确为待批准运行。不能据静态检查认定新版已改善效果。

## 最新代码验证

| 检查 | 最新已记录结果 | 范围 |
|---|---|---|
| 新版语法编译 | **通过** | `compileall` 检查模块、脚本和测试文件，不运行模型 |
| 新版六方向配置 | **通过** | 方向齐全、同域协议一致、三源路径复用、数据及本地权重路径存在 |
| 新版运行验证 | **未执行，待用户批准** | 24 个测试函数（新增 9 个）均未在本轮执行；无新版前向、编码、预处理、训练或评估 |
| 新版运行目录 | **尚未创建** | 缓存、源模型、迁移运行目录不存在，与未启动状态一致 |

[本轮静态检查产物](benchmarks/bgl_spirit_thunderbird/static_validation_20261027.json)。旧版最近自动测试为 **15 passed in 15.16s**，仅是旧修订历史，不能作为新版通过证据。官方 MiniLM 权重校验、旧 pilot 两域真实语义编码已完成；新版没有重新编码。此前细节见 [验证记录](docs/validation.md)。

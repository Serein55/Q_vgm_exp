# Q-VGM v2

依据用户提供的 [实施计划](../Q_VGM_v2_复现实现与实验计划.md)，独立实现 v2。现有 `qvgm/` 的 v5 核心未修改。当前只完成第一批算法模块，尚无 v2 训练结果。

## 已实现文件

| 文件 | 用途 |
|---|---|
| `configs/libero_spatial_v2.yaml` | 路径、结构参数、工程假设；尚不能交给 v5 runner 执行 |
| `qvgm_v2/data/buffer.py` | schema-2 单条 transition 转 stepwise 奖励、终止和有效位；不负责加载数据集 |
| `qvgm_v2/models/state_encoder.py` | RL token 与 proprio 投影拼接、LayerNorm；输入 proprio 需在训练集统计量下归一化 |
| `qvgm_v2/models/critic.py` | 两个逐位置 Q、逐位置最小值求和、逐位置 V |
| `qvgm_v2/algorithms/stepwise_iql.py` | TD 目标、expectile/Q 损失和 EMA；不含训练循环 |
| `tests/test_core.py` | 手工目标、边界语义、逐位置聚合、梯度隔离和 EMA 测试 |

从父项目 `Q_vgm/` 执行：

```bash
PYTHONPATH=Q_vgm_v2 ../RLinf/.venv-openpi-robotwin/bin/python -m unittest discover -s Q_vgm_v2/tests -v
```

## 数学与边界

状态为 `LN([z_rl; Linear(p)])`，维数 2304。两个 Q 各输出 5 维，每个隐藏层重新注入 35 维动作。标量分数是 `sum_i min(Q1_i,Q2_i)`。

普通位置的目标为 `r_i + γ(1-d_i)V_(i+1)(s)`；末位置使用 `V_0(s')`。V 对 detached target-Q 的逐位置最小值做 expectile 回归（τ=0.8）。Q 目标不会反传到 V。

终止位保留，其后位置无效。后继 chunk 不完整时关闭跨 chunk bootstrap。配置默认将 timeout 视为有限时域终止，**这是待核实的工程选择**；若显式选择 continuing timeout，partial tail 缺少后继位置时不回归该末位置，避免伪造零目标。此替代分支尚无环境验证。

## 后续顺序

数据审计及 demo 动作归一化 → 500 SFT rollouts + demos → AE 预训练 → encoder/proprio/Q/V 联合训练 → Q-selection / guidance 环境验证 → actor。

当前尚未实现完整 buffer、数据导入、联合训练循环、guidance 和 actor。配置中的空预算不得自动补成正式训练参数。最终效果与 v5 不同也不能单独归因于某一个架构变化，需要消融。

## 全新 SFT rollout 采集

从 `Q_vgm/` 运行 `bash Q_vgm_v2/scripts/collect_sft.sh`，使用 GPU 2、3，按任务分工，共 500 episodes。采集配置在 `configs/collect_sft.yaml`，复用父项目配置加载器，因此该文件中的路径以 `Q_vgm/` 为基准。模型训练配置仍以本目录为基准。

每条 episode 记录 `success_once`（策略执行阶段至少成功一次）、seed、initial_state_index、observations、prefixes 和逐步 transition；首次成功终止。结果保存在 `artifacts/v2_sft_500/`。随时汇总：

```bash
../RLinf/.venv-openpi-robotwin/bin/python Q_vgm_v2/scripts/summarize_collection.py Q_vgm_v2/artifacts/v2_sft_500/buffer
```

脚本只统计已原子落盘的 episode，不会因断点重启重复计算。

## Expert demos 与最终审计

- `scripts/download_demos.py`：下载固定版本的官方 Spatial HDF5 数据。
- `configs/import_demos.yaml`：demo 来源、输出目录和转换约定。
- `scripts/import_libero_demos.py --task-ids 0 --limit 1`：单条预检；使用动作前 simulator state 渲染、checkpoint 原生动作归一化、末状态成功复核。
- `bash Q_vgm_v2/scripts/convert_demos.sh`：GPU 0、1 批量转换；按 episode 断点续跑。
- `scripts/audit_dataset.py`：检查已落盘数据，生成 `artifacts/dataset_report.json` 和 `combined_manifest.json`。缺数据时 ready=false。
- `bash Q_vgm_v2/scripts/finish_dataset.sh`：等待 SFT 与 demos 结束再进行最终审计，写 `数据准备状态.md`。

Python 脚本均使用 `../RLinf/.venv-openpi-robotwin/bin/python`，工作目录为父项目 `Q_vgm/`。官方原始 Spatial 为 500 demos；论文 432 条子集尚未确认。转换不等于完整动力学重放；奖励来自 HDF5，末状态复核另存字段。

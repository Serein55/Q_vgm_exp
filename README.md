# Q-VGM v5：最小 offline 复现

本仓库只保留 v5 offline 主线及必要测试。v2、检索/控制器改进分支、消融产物、旧文档和运行日志已删除；外部 RLinf 环境与 few-shot SFT 权重未改动。许可证文件保留。本文是唯一说明文档。

## 已完成的工作与结论

- 实现 few-shot π0.5 rollout、冻结前缀表征的 RLT 自编码器、IQL critic、Q 梯度引导与 velocity-matching actor，以及 LIBERO 评估。
- v5 发布权重兼容配置：采集150条轨迹，RLT训练10000步，critic训练6000步，actor训练500步。主线数据、权重、特征缓存保留在 `artifacts/spatial_v5_h10/`。
- 同一 Q-VGM 评估 harness 下，发布输入条件 SFT 为376/500；v5字面输入条件为26/500。用户提供的另一 RLinf harness 成绩为400/500，不能跨 harness 直接比较。
- 默认 min、mean、加 proprio 及二者组合的配对引导诊断均未确认提升。默认 min 的464对结果：控制82.11%、引导81.47%，14胜17负，p≈0.72。
- 此前常数动作偏置对照为141/500，真critic引导actor为148/500，二者配对检验p≈0.49；均明显低于同harness的SFT 376/500。引导位移约91–94%的方差由全局方向解释，支持“引导主要表现为固定偏置”的诊断，但不证明论文idea一般无效。
- 曾尝试检索、特权状态、监督排序等改进，没有确认稳定收益。另行尝试的v2已按要求全部删除；其代码、数据、权重、结果文件均不属于当前pipeline。
- 尚未实现持续采样、更新buffer、交替更新critic/actor的online RL。当前保留的是完成过、但未达到论文效果的offline实现，不是官方实现。

## 数学与数据 pipeline

```text
外部 few-shot SFT + LIBERO
  → 固定探索 rollout：150 episodes
  → buffer：前缀tokens、观测、动作块、奖励、终止信息
  → RLT自编码器：重建冻结前缀表征，缓存 z
  → IQL：Q回归 r + γ V(s′)，V做expectile回归
  → 动作梯度上升 A⁺ = A + η ∇A Q(z,A)
  → 用局部velocity matching把动作修正蒸馏到actor
  → 同一harness评估SFT与actor的success_once
```

RLT采用带mask的表征重建损失；critic为ensemble，V默认以min聚合为expectile目标（τ=0.8）；actor在后段flow步骤使用Q引导目标。细节以 `qvgm/algorithms/` 和训练脚本为准。探索温度2.0、flow noise 0.08及部分offline预算是工程选取，不能视为作者公开的完整超参。

主配置保留发布SFT的H10、非离散state输入，执行前5步×7维动作；10次去噪，环境上限220步。`libero_spatial_paper.yaml`另提供H5、离散state的字面配置，仅供明确对照；不可与主线混用数据或直接比较基线。

## 文件与路径

| 路径 | 用途 |
|---|---|
| `configs/libero_spatial.yaml` | Python、SFT、normalization、tokenizer、LIBERO等外部路径及基础设置 |
| `configs/libero_spatial_checkpoint.yaml` | v5算法采用发布权重输入约定的训练配置 |
| `configs/libero_spatial_paper.yaml` | 论文输入约定的字面对照配置 |
| `qvgm/models/` | π0.5适配、RLT、critic |
| `qvgm/algorithms/` | IQL、动作引导、velocity-matching loss |
| `qvgm/data/`、`qvgm/envs/` | replay和LIBERO接口 |
| `qvgm/training.py`、`qvgm/config.py` | 训练、公用配置与产物一致性检查 |
| `scripts/` | 预检、采集、RLT/critic/actor训练和SFT/actor评估，共7个入口 |
| `tests/` | 保留算法和输入约定测试 |
| `artifacts/spatial_v5_h10/buffer/` | 150条轨迹及采集配置/flow检查；不保留采集日志 |
| `artifacts/spatial_v5_h10/rl_token_full.pt` | RLT权重及训练状态 |
| `artifacts/spatial_v5_h10/features_full.pt` | 冻结z缓存及buffer签名 |
| `artifacts/spatial_v5_h10/critic_full.pt` | critic权重 |
| `artifacts/spatial_v5_h10/actor_full.pt` | actor权重 |

产物约25GiB，主要是数据与模型，不是日志。特征与模型记录buffer签名；不要重采后直接拼接旧缓存。路径集中在YAML，默认外部Python为 `../RLinf/.venv-openpi-robotwin/bin/python`。运行入口默认GPU2，可用 `CUDA_VISIBLE_DEVICES` 覆盖。

## 使用

在仓库根目录执行。已有主线产物可直接评估；重新训练请使用新run名，避免覆盖保留结果：

```bash
bash run.sh preflight --config configs/libero_spatial_checkpoint.yaml

# 以下为重跑命令，不是当前正在运行的任务。
bash run.sh collect_rollouts --config configs/libero_spatial.yaml --run v5_new --check-flow
bash run.sh train_rl_token --config configs/libero_spatial_checkpoint.yaml --run v5_new --tag full
bash run.sh train_critic --config configs/libero_spatial_checkpoint.yaml --run v5_new --tag full
bash run.sh train_offline_qvgm --config configs/libero_spatial_checkpoint.yaml --run v5_new --tag full --steps 500

bash run.sh eval_sft --config configs/libero_spatial_checkpoint.yaml --name v5_sft
bash run.sh eval_qvgm --config configs/libero_spatial_checkpoint.yaml --actor-checkpoint artifacts/spatial_v5_h10/actor_full.pt --name v5_actor
```

`collect_rollouts --task-ids`支持任务子集；评估支持`--episodes-per-task`。训练续跑须显式`--resume`并保持配置、数据一致。将来主动运行会生成新日志，此次清理删除的是既有日志，不移除训练所需的运行记录能力。

# Q-VGM idea 实验

当前结论：原论文 offline 复现失败，且失败已定位到机制层——引导量几乎与状态无关，等价于常数动作偏置（2026-09-28，见下）。检索策略独立起点确认未通过门槛（389 vs 375）。特权状态参数化 critic 诊断已完成：geometry/full_state 不优于视觉 mean-pool（OOF 1143–1160 vs 1193/2048），瓶颈判定为标签/候选设计而非视觉表征；离散九候选线已到天花板。

**few-shot 基线补测已完成**（2026-09-28 00:02，`scripts/eval_sft_baseline.sh`，逐回合 `episodes.jsonl` 在 `artifacts/eval/sftbase_*/`）：同一批 SFT 权重在 Q-VGM harness 内 **发布条件 376/500、v5 字面条件 26/500**，RLinf harness 为 400/500。两个后果：(1) 52/500 的同条件对照是 26/500；历史 α=0 组（41/50=82%）经核查 500 步 loss 全为 0、actor 从未更新，**那 82% 就是 SFT 本身，不是训练过的对照**，所以旧链条"条件不匹配→训练修复到 82%→引导砸到 10%"不成立，正确表述是「同条件 SFT → 加引导蒸馏 → 10–30%」；(2) **基线是 harness 依赖的**（376 vs 400，差 24 回合集中在 t5/t8），RLinf harness 的 400/500 只在该 harness 内成立，跨 harness 比较一律无效。

**offline 复现的因果定位已完成**（2026-09-28 07:15，`idea改进实验.md` 末节、`复现过程.md` 阶段 3/4）：数据、AE、critic、actor 全部重建后，四个 critic 臂（论文默认 min 聚合、mean 聚合、+proprio、两者）的**配对反事实 A/B 全部未通过门槛**（Δ≤0、p≥0.087、`corr(q_gain,Δ)`≈0），且实际引导位移的 **91–94% 方差由一个与状态无关的方向解释**。据此构造的常数偏置对照 `q(z,a)=a·m` 在 500 回合上与真 critic 不可区分（141/500 对 148/500，McNemar p=0.49），两者都相对 SFT 基线 376/500 崩溃（p≈1e-56）。**结论：状态条件 Q 对崩溃没有可测贡献，本次复现中的"价值引导"等价于注入固定旋转偏置并被 500 步 velocity matching 积分成腕部姿态漂移。** 因此继续改 critic 聚合、加 proprio/特权状态、调 α/LR 都无着力点；要让引导存在，必须先让 ∇_A Q 随状态变化，而当前标签（chunk 级稀疏成功、150 条轨迹、normalized 旋转维 std 仅 0.04–0.07）与九候选线的"标签噪声与信号同阶"是同一个瓶颈。当前无后台实验在跑，8 卡显存归零。

**最新机制发现**（`idea改进实验.md` 末节）：t9 的失败不是感知/语言/够不到，而是**最后落点精度**——真值追踪显示取碗与搬运成功（0.5 m），成功回合落点离盘心 4–8 mm 且盘子不动，t9 失败回合偏 32–62 mm 并把盘子撞开 50–120 mm。场景含两个黑碗（柜顶为目标、炉面为干扰物），录像中"柜顶还有碗"不是抓错碗。该机制对口方向 4（部署状态上的 DAgger 式数据聚合），不对口 1/2/3/9/10。

- [主要实验结果与当前进度](idea改进实验.md)
- [数学原理、论文差异](论文算法核对.md)
- [数据流与文件用法](数据PIPELINE.md)
- [完整复现流水与时间线](复现过程.md)

2026-09-26 按用户要求删除失败实验的权重、数据、日志及一次性脚本，历史只保留文档结论。外部 RLinf 权重、环境及其他项目未改动。本机另有不入库的大体积产物与归档目录，见 `.gitignore`（`artifacts/`、`archive/`）。

运行入口：`bash Q_vgm/run.sh <脚本名> [参数]`。Q-VGM 主配置 `configs/libero_spatial.yaml`（v5 字面条件见 `configs/libero_spatial_paper.yaml`，critic 消融臂见 `configs/arms/`）；失败诊断配置 `configs/diag/`；真值追踪 `scripts/diag_grasp_trace.py`。offline 线可整体重跑：`scripts/collect_buffer_h10.sh`（5 卡并行采 150 episodes，约 10 分钟）→ `train_rl_token`（约 100 分钟）→ `scripts/critic_arms.sh`（四臂 critic + 梯度诊断）→ `scripts/counterfactual_arms.sh`（配对环境 A/B + `counterfactual_report.py` 汇总）→ `scripts/actor_eval.sh <min|mean|prop|meanprop|bias>`（500 步 actor + 500 回合评估，切分与基线一致）；`scripts/guidance_direction.py` 测引导位移的全局方向占比，`--bias-critic` 用 `ConstantDirectionQ` 做常数偏置对照。保留的标签可直接用于后续改进；已完成的无效模型已删除，不必重复采集。

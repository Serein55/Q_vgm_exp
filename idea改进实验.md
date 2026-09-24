# 基于 Q 引导思路的改进实验

2026-09-24，用户授权按五步方案尝试。目标改为验证价值引导能否改善 SFT；此分支不是论文复现。旧实验与权重保留。两轮均已完成候选采集、critic 拟合和验收，均未通过预设教师门槛；尚未启动 actor 蒸馏。

## 五步如何落地

| 步骤 | 本轮实现与验收 |
| --- | --- |
| 1 同状态动作回报 | task 1/9 各 6 条来源轨迹，每条取初态和轨迹中点，共 24 个状态；5 种候选×2 组后续噪声，共 240 次续跑。只替换首个动作 chunk，之后统一冻结 SFT。 |
| 2 critic 动作排序 | 冻结 RL-token，加 8 维 proprio，仅输入 critic。用候选平均折扣回报做 MSE 与同状态 pairwise 排序训练；按完整 episode 分割训练/验证/测试。 |
| 3 候选选择 | 对照原动作、旧 Q 选优、新 critic 选优、候选内 oracle。验证和测试都需平均回报提升 >0.01 且成功率不下降，才通过初步门槛。oracle 只诊断候选上限，不能作为实际可用教师。 |
| 4 动作约束 | 候选含原始 Q ascent 与椭球投影后的 ascent；控制器输入空间旋转半径更小，不在归一化空间对各维等幅修改。 |
| 5 actor 蒸馏 | 有效教师通过留出验收后，还需闭环 teacher 检查，再开始小预算 actor 更新和参考策略约束。未通过则停止，不训练 actor。配置中的 actor 预算目前只是预留值。 |

## 数据与方法

配置 [configs/idea_recovery.yaml](configs/idea_recovery.yaml) 继承原发布 SFT 输入约定；不修改 stateless SFT。来源 `spatial_paper_checkpoint/buffer`，新目录 `artifacts/idea_recovery_v1`。来源 episode 0/1/2 为训练，3 为验证，4/5 为测试；两个 task 分别遵循同一划分，两种采样时刻和全部候选/重复跟随 episode 一起划分。

这不是严格只读固定离线数据的算法：新采集了模拟器反事实回报标签，单独计入环境交互预算；没有把它们混入原 IQL buffer，也没有启动在线 actor–critic 循环。初态/中点是进度代理，不声称已标注接近、抓取、搬运、放置四个语义阶段。当前只有 12 个训练状态、4 个验证状态、8 个测试状态，属于小样本可行性检查。

候选：`base`、`resample`、`perturb`、`q_raw`、`q_trust`。每个起点的候选只生成一次，两次续跑只更换后续 SFT 噪声；不同候选共享同一组后续噪声。环境通过固定 seed、重放原动作前缀重建，每次验证 simulator/state/image 起点一致性。模拟器状态最大差要求 ≤1e-6；图像允许极小渲染误差。成功立即结束，超时回报为 0；首块之后共 n 步成功的回报为 γ^(n−1)。

受约束动作：设输出反归一化对角尺度为 S，候选改变量为 Δa，则控制器输入改变量 δ=SΔa。取 `D=diag(0.04,0.04,0.04,0.012,0.012,0.012,0.03)`，投影到 `||D⁻¹δ||₂≤1`（对整个 5×7 chunk 求范数）。数值是控制器输入单位，不是米/弧度；控制器内部裁剪仍存在。扰动候选同样使用该椭球。原 Q 引导 α=0.05、J=3，用作对照，不当作可信标签。

新 critic 为三个小型独立 MLP，输入为 RL-token 的训练集 PCA 投影、proprio 与动作。PCA 保留最多 8 维，归一化和 PCA 均只使用训练组。每个 head 按来源 episode bootstrap，600 次 AdamW 更新；MSE 加 0.1 倍同状态排序损失，回报差 ≤0.02 的候选对不作排序标签。每 10 步用验证目标选择 checkpoint。选择分数 `mean−std` 和切换阈值 0.02 预先固定；head std 只是启发式，不是校准置信区间。测试回报仅用于最终验收，不能再据其反复调参。

新 critic 估计的是“执行该动作后跟随冻结 SFT”的回报，不是最优 Q。有限回报样本、bootstrap 集成和 PCA 都是工程选择；若失败不能据此否定价值引导 idea。

## 文件与运行方式

- `scripts/recovery_candidates.py`：生成配对单 chunk 反事实标签。支持按完整状态分片续跑，已有 `.pt` 不重复计算；中断的不完整状态需重跑，`episodes.jsonl` 因此只作运行日志，正式统计以 `.pt` 为准。
- `scripts/recovery_fit.py`：只在全部 worker complete 后拟合与验收，拒绝覆盖已有模型；CPU 执行，不读取原 actor 的训练优化器。
- `scripts/recovery_watch.py`：tmux 托管的衔接器，检测 worker 提前退出则报错；采集完成后运行 fit。无论 gate 是否通过，都不会自动启动尚未验证的 actor 训练。
- `artifacts/idea_recovery_v1/worker*/`：每状态一个 `.pt`，包含当前实际观测、z、候选动作、旧 Q、重复回报和起点误差。
- `candidate_critic.pt`：模型、训练集变换参数、配置和数据文件列表；`candidate_report.json` 为分组策略选择比较；`critic_history.json` 为验证选择记录。
- `pipeline_status.json` 与 `worker*.log`、`fit.log`：状态和日志。

```bash
# 从项目父目录执行；worker 0/1/2 分别在 GPU 0/1/2。
CUDA_VISIBLE_DEVICES=0 bash Q_vgm/run.sh recovery_candidates --config configs/idea_recovery.yaml --worker 0
bash Q_vgm/run.sh recovery_fit --config configs/idea_recovery.yaml
```

当前 tmux：`qvgm-recovery-data0/1/2`、`qvgm-recovery-watch`。它们脱离对话运行。脚本 Ruff 已通过；拟合器另用 /tmp 下明确标记的合成数据检查序列化和分组路径，合成结果不进入实验报告。


## 本轮结果（2026-09-24）

三个 worker 均完成 80 次，共 24 个状态/240 次候选续跑。新 critic 完成 600 次更新，由验证目标选中第 250 步。下表成功率仅指该组状态上的配对续跑，不是任务从初态开始的正式评估。

| 测试组方法 | 平均折扣回报 | 成功续跑数（共 16） |
| --- | --- | --- |
| 原 SFT 候选 | 0.297638 | 11 |
| 旧 Q 选优 | 0.260581 | 9 |
| 新 critic 选优 | 0.297631 | 12 |
| 固定受约束 Q 候选 | 0.304077 | 12 |
| 根据已知回报事后选最佳候选 | 0.312703 | 12 |

新 critic 在训练/验证/测试组的平均回报增量分别为 +0.017287 / +0.014845 / −0.000007。验证通过，但测试未达到预先设定的 >0.01 增量，故 `actor_gate.passed=false`，自动流程停于 `stopped_at_teacher_gate`。成功次数增加 1 次是正面信号，但任务 1 完成速度下降抵消了任务 9 的收益；不以事后更换验收指标将本轮改判为通过。

测试组分任务：task 1 新选择的回报增量 −0.025593、成功率不变；task 9 为 +0.025579、成功率增加 0.125。新 critic 排序准确率在训练组为 32/37=86.5%，验证组 7/9=77.8%，测试组 9/21=42.9%；旧 Q 的测试排序为 4/21=19.0%。只计平均回报差 >0.02 的候选对，这些候选对互有关联，且回报仅估计两次；不作显著性或总体优劣结论。

固定 `q_trust` 在测试组略好，但验证组回报增量为 −0.000462，没有满足两组均改善的要求，因此不据测试表现事后选它作为教师。事后最佳候选回报也仅比原动作高 0.015065；这是有噪声的小样本事后上限，不能当作真实可实现收益。

完整结果：[candidate_report.json](artifacts/idea_recovery_v1/candidate_report.json)；数据检查：[collection_audit.json](artifacts/idea_recovery_v1/collection_audit.json)。模型与训练集变换参数在 `candidate_critic.pt`。所有文件位于独立分支，未改原训练数据或 actor。

## 能判断什么，以及下一步

1. 旧 Q 在全部 24 个状态都把原始 Q ascent 候选排第一，却不能在测试续跑中兑现收益。它确实存在当前候选分布下的评分可靠性问题。
2. 新 critic 降低了旧评分造成的退化，训练组排序也能学到；测试泛化仍不足。不能把多项同时变动后的效果单独归因于 proprio。
3. 修改幅度约束有值得扩大验证的信号，但本轮没有通过跨划分验收。24 个状态只覆盖 12 条来源轨迹，训练部分更少，不能据此否定 idea。
4. 第 5 步没有执行，原因是教师验收失败，并非 actor 训练报错。没有生成 recovery actor checkpoint，也没有把事后 oracle 动作当作可泛化教师。

若继续下一轮，优先扩大不同轨迹/阶段的同状态候选数据及续跑次数，再用新的、未看过的测试起点验证。另需事前决定主指标：若主要追求 `success_once`，可考虑以成功概率训练/验收，完成步数作次指标；不能在当前测试结果上改规则后声称成功。架构/数据量/指标变化均应另开版本。本轮未自动安排这些后续实验。


## 第二轮：扩大数据、成功率主目标（已完成）

配置 [idea_recovery_v2.yaml](configs/idea_recovery_v2.yaml)，产物 `artifacts/idea_recovery_v2`，不覆盖第一轮。task 1/9 各取来源 episode 0–14，每条初态/中点，共 60 个状态；每状态 5 候选、每候选 4 组后续噪声，共 **1200 次续跑**。GPU 0/1/2/3 对应四个 worker，每个计划 300 次。没有直接复用第一轮回报文件；当前配置从头采集，以固定的一致候选和配对重复生成完整标签。

训练 episode 0–9（40 状态），验证 10–11（8 状态），测试 12–14（12 状态）。第一轮测试 episode 4/5 转入新版训练，不能再用于新版的独立测试。新版验证/测试尚无已用过的候选回报标签，但旧 AE/IQL 训练见过来源 buffer 的这些轨迹，因此仅是新候选 critic 的监督划分，不是整个系统从未见过的环境泛化验证。后续正式 teacher/actor 检查仍需新的起点。

主要改动是监督目标由平均折扣回报改为候选成功比例；归一化、PCA、模型 checkpoint 仍只由训练/验证确定。测试仅验收一次。验收条件在结果出来前固定：**验证与测试整体成功率都提高，而且每个任务在这两组中成功率均不下降**；选中的模型不能是初始化 step 0。平均折扣回报保留为辅助指标。每候选 4 次仍有较大噪声，不能仅凭通过该门槛宣称已经稳定改善策略。

三个阶段 `采集→拟合→教师验收` 由 `qvgm-recovery-v2-watch` 托管；worker tmux 为 `qvgm-recovery-v2-data0/1/2/3`。未过门槛则停；通过则标为等待教师闭环检查，不直接训练 actor。`pipeline_status.json`、`worker*.log`、`watch.log` 记录状态，`launch_manifest.json` 保存本次脚本/配置 hash。

```bash
CUDA_VISIBLE_DEVICES=0 bash Q_vgm/run.sh recovery_candidates --config configs/idea_recovery_v2.yaml --worker 0
RLinf/.venv-openpi-robotwin/bin/python Q_vgm/scripts/recovery_watch.py --config Q_vgm/configs/idea_recovery_v2.yaml
```

新旧目标的拟合/保存/划分流程分别用合成数据检查通过；合成数据保留在 /tmp，不混入实验。Ruff 通过。本轮结果见下节。

### 并发提速

用户授权充分使用 GPU 后，执行并发从每卡 1 个串行进程调整为每卡 3 个，总计 12 个进程（仍只用 GPU 0–3）。原 worker 编号/实验配置/随机种子保持不变，新增 `--lanes 3 --lane 0|1|2` 将每个 worker 的任务进一步互斥分片。切换在完整状态组写盘后进行，保留 17 组=340 次续跑，无需重采已完成组。保存边界见 `parallel_switch_boundary.json`，文件 hash 见 `parallel_switch_manifest.json`。

新日志为 `workerN/episodes_laneL.jsonl`，完成标记 `status_laneL.txt`；历史 `episodes.jsonl` 保留。总进度不能再只看原日志，应合并这些日志或以已保存的 `.pt` 为准。`recovery_watch.py --lanes 3` 检查全部 lane，完成后才写 worker complete 并启动拟合。tmux 为 `qvgm-recovery-v2-dataN-laneL`，监控名称保持不变。分片覆盖/不重复检查和 Ruff 通过。


## 第二轮结果

1200 次续跑全部完成，共 60 个状态。新 critic 训练 600 步，验证选中第 100 步。新监督目标是成功比例；以下平均折扣回报为辅助指标。

| 分组与方法 | 成功续跑数 | 平均折扣回报 |
| --- | --- | --- |
| 验证：原动作 | 17/32 | 0.248073 |
| 验证：旧 Q 选优 | 16/32 | 0.237184 |
| 验证：新 critic 选优 | 17/32 | 0.248925 |
| 测试：原动作 | 26/48 | 0.247539 |
| 测试：旧 Q 选优 | 27/48 | 0.263979 |
| 测试：新 critic 选优 | 28/48 | 0.285388 |
| 测试：固定受约束 Q 候选 | 28/48 | 0.277295 |

测试成功率增加 4.17 个百分点，折扣回报增加 0.03785，但验证成功率持平，因此**仍未通过“验证和测试都提高”的预设门槛**。流程状态 `stopped_at_teacher_gate`，没有继续闭环 teacher 或 actor 训练，也未根据测试结果修改阈值。

测试增加的 2 次成功全部来自 **task 9 / episode 13 / 中间状态 22**，选择的是 `q_raw`；其余测试状态的成功次数均未改变。不能将其描述成多个独立任务/起点上的稳定改善。测试 48 次续跑来自 12 个状态、6 条来源轨迹，重复噪声和同轨迹两时刻并不独立。

新 critic 的非平局候选对排序：训练 66/86，验证 8/8，测试 11/16；旧 Q 分别为 30/86、0/8、8/16。样本少且候选对关联，不能只凭排序数字认定已经解决泛化。`oracle_candidate` 在第二轮按成功比例选取，同成功比例时选最先出现的候选，因此它的辅助折扣回报不是该指标上界；新选择的折扣回报高于此项不代表超过 oracle。

固定 `q_trust` 的测试成功数也是 28/48，但验证比原动作少 1 次成功，同样不能据测试结果直接采纳。小幅限幅不保证每个状态都受益。

结果：[candidate_report.json](artifacts/idea_recovery_v2/candidate_report.json)。全量检查：[collection_audit.json](artifacts/idea_recovery_v2/collection_audit.json)：60 个唯一状态，各含 20 条完整候选/重复记录，标签及形状通过；并发切换前的 17 个结果文件 hash 全部不变。模型及训练集变换保存于 `candidate_critic.pt`。

当前结论：比第一轮出现了更明确的局部正面信号，但仍不足以放行蒸馏或宣称 idea 已有效。下一步若继续，应冻结这版模型、阈值和候选规则，在新的独立起点确认收益；不要继续在已看过的测试组上挑模型。本轮没有排队第三轮实验。

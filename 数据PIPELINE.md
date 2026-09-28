# 数据 pipeline 与文件用法

## 当前流程

冻结外部 few-shot SFT → 模拟器训练起点 → 初始/请求五十步状态 → 九候选各十六次配对续跑 → 每状态 .pt → 与原 96 组数据合并 → 明确训练/开发清单 → 三种子监督 critic → 复用开发标签汇总。

本轮采集、训练均已完成，主要结果见 idea改进实验.md；无效权重已删除。这是使用模拟器收集监督数据后离线训练的改进分支，不是原论文纯 offline 复现，也不是 actor online RL。online actor 更新尚未实现。

每个 .pt 包含 raw observation、mean_pool、base/候选动作、实际前缀和成功记录。新增数据仅用于训练；原 episode 23–26 的 32 组开发标签不变。controller_training 原来指向 pilot 的文件已实体化。fresh controller 采集不再加载无关的旧 replay/AE。

| 文件 | 用途 |
| --- | --- |
| configs/libero_spatial.yaml | 外部 Python、SFT、LIBERO、tokenizer 路径及基础设置 |
| configs/libero_spatial_checkpoint.yaml | 发布模型精度与输入约定 |
| configs/controller_training.yaml | 原 96 组配置；不必重新采集 |
| configs/controller_coverage.yaml | 当前扩充采集；GPU 由 CUDA_VISIBLE_DEVICES 指定 |
| configs/libero_spatial_paper.yaml | 保留论文输入约定测试所需配置，不代表当前采集 |
| scripts/recovery_candidates.py | --config … --worker N；当前采集工作进程，支持跳过已完成组 |
| scripts/recovery_watch.py | --config …；监测已有 tmux worker，不启动采集 |
| scripts/controller_coverage_pipeline.py | 等待采集，检查 128/32 划分及哈希，训练并汇总 |
| scripts/train_controller_critic.py | --manifest 清单 --run 输出名 --seed 7108 --dropout 0.15 --steps 600 --eval-every 5 |
| scripts/controller_training_report.py | --manifest 清单 --prefix controller_coverage；汇总三种子和固定候选控制 |
| scripts/preflight.py | 依赖/路径预检 |
| qvgm/ | 模型适配、环境、算法及数据模块 |
| tests/ | 数学及输入约定单元测试 |

`bash Q_vgm/run.sh <脚本名> [参数]` 自动选择 YAML 中的 Python。默认 GPU 2；本轮曾在 GPU 0–3 运行，现已完成。

artifacts/controller_coverage/training_status.json 已记录完成及权重删除。training_manifest.json 保留全部数据文件哈希和划分。重复日志与已汇总的模型报告已清理，结果见 idea改进实验.md。若主动重训，pipeline 会重新生成模型、报告和日志；它不是常驻新实验调度器。

## 历史 offline / online 边界

历史纯 offline：SFT 探索采集固定 replay → token AE → IQL → Q-guided actor → LIBERO 评估。结果失败，replay、AE、critic、actor 和一次性诊断已删除，仅保留文档主要结果及算法模块。若重做需重新采集与训练。没有实现持续交互、buffer 更新、critic/actor 交替更新的 online 训练流程。


补充入口：`bash Q_vgm/run.sh train_candidate_policy --method linear`（默认）或 `--method forest`，只读取现有 training_manifest.json。以训练 episode 四折比较直接选动作的策略，开发标签不参与线性配置选择；文档中的开发集此前已被使用，因此仍仅为探索。最新两分支均失败，模型已删除，复跑会生成新输出。


直接策略入口：`train_candidate_policy --method direct` 使用 token 均值 PCA，`--method direct_image` 使用 .pt 内保存的两路 RGB 空间特征。共用 direct_candidate_policy.py，以已有成功标签直接优化候选选择概率；不调用环境、不需要 GPU。配置由训练 episode 四折选取，再汇总固定开发标签；任务回退也只用训练 OOF。产物目录分别 direct_policy_guarded/direct_image_policy，已有目录拒绝覆盖。本轮失败产物已清理，复跑会重新训练；主要结果见实验文档。


当前保留的检索候选：`train_candidate_policy --method retrieval` 调用 retrieval_policy.py，按 task 与 actual_prefix_chunks>0 筛选训练邻居。select_actions(train_z, train_side, train_y, query_z, query_side, metric, neighbors) 返回候选索引、固定参考动作索引、训练库中的近邻索引；模型输入不包含查询成功标签。policy.pt 的 side 顺序为 proprio8、base动作35、actual_prefix_chunks1、task one-hot[1,5,6,9]4；mean_pool 为 z。加载其训练数组及 metric/neighbors 即可调用该接口。当前默认固定 prior strength=2、切换阈值 0.02，改这些值须作为新实验。

artifacts/retrieval_policy 保留模型、训练选择报告、开发选择及重载审计，不删除此尚有希望的候选。已有目录拒绝覆盖。当前无后台任务。


### 冻结检索确认与推理接口

`FrozenRetrievalPolicy(checkpoint, expected_sha256).choose(mean_pool, proprio, base_action, task, prefix_chunks)` 返回动作候选编号、固定参考编号及训练库邻居编号。mean_pool 为 2048 维；base_action 为 5×7 归一化动作；prefix_chunks 为已执行完整块数。仅支持 task 1/5/6/9，权重 hash 与算法常量不符会拒绝执行。动作编号按 checkpoint.candidate_names 定义，controller 修正幅度固定 0.2。不能把归一化动作的 0.2 当物理控制输入的 0.2。

configs/retrieval_confirmation.yaml 锁定测试起点和模型；recovery_candidates.py 在 frozen_retrieval 模式仅执行 base/reference/selected 三种唯一动作，部分候选标签文件不得作为原九候选训练数据。每组先写 .selection.json 再收集回报。confirm_retrieval.py --watch 等待 worker 完成，校验哈希、配对完整性及选择重现，写 summary.json。

finish_retrieval.py 托管条件后续阶段：第一阶段门槛不通过则结束，通过才启动 retrieval_closed_loop.py 的 20 个 worker，再生成 artifacts/retrieval_closed_loop/summary.json。第一阶段总状态 completion_status.json；闭环各 worker 写完整配对 JSON，完成后再汇总。部署检查使用新噪声但复用确认起点，应与独立状态测试区分。当前任务已托管，勿重复启动。


补充训练内部筛选：`train_candidate_policy --method learned_retrieval` 学习低秩收益距离；`--method guarded_retrieval` 检查状态支持和近邻收益离散程度回退。均由 learn_retrieval_metric.py 实现，不加载 retrieval_confirmation 数据。只有训练 OOF 严格优于原检索才进入开发比较；本轮均未达到，因此没有新环境交互，失败产物已清理。


仿真状态诊断：privileged_state.py 从原 training_manifest.json 读取 160 个状态及保存前缀，重放后核对图像和机器人状态，提取几何/动力学特征。缓存文件含 source_sha256 与 feature_names；--fit 校验来源哈希及每任务特征顺序，再用原 128/32 划分诊断检索。没有新增候选成功标签，45–49 确认数据不在输入中。状态见 artifacts/privileged_state/status.json；完整报告 report.json。

仿真状态缓存已完成 160/160，并通过来源哈希与重放核对。geometry/full_state 近邻诊断未超过原视觉检索的训练 OOF，因此未读开发收益；缓存保留供后续不同模型使用，不视为已经训练出完整状态 critic。当前 status.json=complete。

## offline v5 复现线（2026-09-28 重建）

数据流：`scripts/collect_buffer_h10.sh`（GPU 0-4 按任务并行，150 episodes 约 10 分钟，写 `artifacts/spatial_v5_h10/buffer/`，内含采集签名 `config.json` 与 `flow_check.json`）→ `train_rl_token`（10000 步约 100 分钟，写 `rl_token_full.pt` 与 `features_full.pt`；后者是逐 episode 的冻结 z 缓存，之后 critic/actor/诊断都不再跑 VLA）→ `scripts/critic_arms.sh`（四臂各 6000 步，写 `critic_full.pt`、`critic_diagnostics_full.json`、`critic_gradient_audit_full.json`）→ `scripts/counterfactual_arms.sh`（每臂按任务 8 卡分片，写 `cf_<arm>_t<task>/episodes.jsonl`，再由 `scripts/counterfactual_report.py` 汇总到 `artifacts/counterfactual_report.json`）→ `scripts/actor_eval.sh <arm>`（500 步 actor + 500 回合评估，写 `artifacts/eval/qvgm_<arm>_t{a,b}-*/`，任务切分与 `eval_sft_baseline.sh` 一致以便配对）。

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `artifacts/spatial_v5_h10` | buffer、features、`min` 臂 critic/actor、`cf_min_*`、`guidance_direction_full.json` | 25G |
| `artifacts/spatial_v5_{mean,prop,meanprop}` | 三个消融臂的 critic、诊断与 `cf_*`（buffer/features 为符号链接） | 各 246M |
| `artifacts/spatial_v5_bias` | 常数偏置对照的 actor（无 critic） | 5.8G |

复用与失效规则：三个消融臂的 `buffer`/`features_full.pt` 指向 `spatial_v5_h10`，删掉主 run 会让它们同时失效；重采会改变 `buffer_signature`，而 features/critic/actor 都带该签名校验（不符即报错），所以换数据必须整条重跑，不能只补一段。`guidance_direction_full.json` 的 `global_direction` 是 `--bias-critic` 对照的唯一输入。critic 状态是否含 proprio 由配置与 checkpoint 双重记录，两者不一致时 `train_offline_qvgm`/`diagnose_critic` 直接报错而不是静默降维。

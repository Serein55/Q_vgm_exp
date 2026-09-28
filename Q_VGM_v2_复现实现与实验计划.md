# Q-VGM v2 复现实现与实验计划

## 0. 目标

在当前 `Q_vgm_exp` 仓库中新增独立目录：

```text
Q_vgm_v2/
```

用于复现 arXiv:2606.08015v2，而不修改现有 v5 主线的核心实现。

目标分成三层：

1. **算法结构复现**：严格实现 v2 的 stepwise IQL critic、RL-token + proprio state、clipped Q-gradient ascent、keep-best、late-step residual velocity matching。
2. **机制验证**：在训练 actor 之前，首先验证 critic 的动作排序和动作梯度在环境中确实有意义。
3. **性能复现**：最终在 LIBERO-Spatial 上比较 few-shot SFT 与 Q-VGM。

论文 v2 在 LIBERO-Spatial 报告：

| Method | Spatial SR |
|---|---:|
| few-shot SFT | 85.6% |
| Q selection | 90.2% |
| Q guidance | 93.8% |
| Action distillation | 93.4% |
| Q-VGM | 96.2% |

因此复现不能只看最终 actor；**Q selection 和 test-time Q guidance 本身就是非常有用的 critic sanity check**。

---

# 1. 为什么必须单独维护 `Q_vgm_v2`

当前 `qvgm/` 已经围绕 v5 建立了：

```text
chunk-level IQL
10-head scalar Q
frozen/cached RL token
normalized Q gradient
first-non-improvement stop
150 self-rollouts
```

而 v2 的核心是：

```text
stepwise IQL
2-head clipped double Q
RL-token encoder + critic joint training
explicit proprio projection
gradient clipping
global keep-best
500 eval rollouts + expert demos
```

两者在数据结构和 critic API 上已经不兼容。

因此不要直接修改：

```text
qvgm/algorithms/iql.py
qvgm/models/critic.py
qvgm/algorithms/q_guidance.py
```

否则以后无法区分 v2/v5 实验。

推荐原则：

```text
qvgm/          → 保留现有 v5 / diagnostics
Q_vgm_v2/      → 独立维护 v2
```

但是一些稳定的基础设施可以复用，不必复制。

---

# 2. 推荐目录结构

建议建立：

```text
Q_vgm_v2/
├── README.md
├── run_v2.sh
│
├── configs/
│   ├── libero_spatial_v2.yaml
│   └── smoke_v2.yaml
│
├── qvgm_v2/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── buffer.py
│   │   └── libero_demos.py
│   │
│   ├── models/
│   │   ├── state_encoder.py
│   │   └── critic.py
│   │
│   ├── algorithms/
│   │   ├── stepwise_iql.py
│   │   ├── q_guidance.py
│   │   └── qvgm_loss.py
│   │
│   └── training.py
│
├── scripts/
│   ├── collect_eval_rollouts.py
│   ├── import_libero_demos.py
│   ├── build_buffer.py
│   ├── train_rl_token.py
│   ├── train_critic.py
│   ├── cache_critic_states.py
│   ├── diagnose_critic.py
│   ├── eval_q_selection.py
│   ├── eval_q_guidance.py
│   ├── train_qvgm.py
│   └── eval_qvgm.py
│
└── tests/
    ├── test_buffer.py
    ├── test_stepwise_iql.py
    ├── test_critic.py
    ├── test_state_encoder.py
    ├── test_q_guidance.py
    └── test_qvgm_loss.py
```

其中 `Q_vgm_v2/qvgm_v2/` 只写 v2 特有逻辑。

---

# 3. 哪些现有代码可以直接复用

## 3.1 `Pi05Flow`：基本直接复用

继续使用：

```python
from qvgm.models.pi05_adapter import (
    Pi05Flow,
    load_sft_policy,
)
```

当前实现已经提供：

```text
encode_context()
predict_velocity()
sample_with_intermediates()
unnormalize()
enable_actor_training()
```

这些接口覆盖 v2 所需的 flow-matching trajectory 和 local velocity prediction。

不建议复制一个 `pi05_adapter_v2.py`。

唯一建议新增的公共接口是：

```python
normalize_actions(...)
```

因为 v2 要加入 LIBERO expert demos。SFT rollout 中的 action 已经保存为 normalized action，但 LIBERO demo 里的 action 是 controller/environment 坐标。

因此必须保证：

\[
A_{\rm demo}^{env}
\rightarrow
A_{\rm demo}^{normalized}
\]

使用和 π0.5 checkpoint 完全相同的 `norm_stats`。

建议增加 round-trip test：

```python
a_norm = normalize_actions(a_env)
a_back = unnormalize(a_norm)

assert max_abs(a_back - a_env) < tolerance
```

---

## 3.2 LIBERO env：直接复用

继续复用：

```python
from qvgm.envs.libero_env import ...
```

无需为 v2 再写 environment wrapper。

---

## 3.3 `RLTTokenTransformer`：architecture 直接复用

当前：

```text
qvgm/models/rl_token.py
```

已经包含：

```python
RLTTokenEncoder
RLTTokenDecoder
RLTTokenTransformer
```

以及：

```python
encode()
encode_flat()
decode()
reconstruct()
loss()
```

足够实现 v2，不需要重写 RL-token architecture。

但训练流程必须改变。

---

# 4. RL-token：v5 训练流程不能复用

当前 v5：

```text
prefix
 ↓
train RL-token AE
 ↓
freeze AE
 ↓
cache features_xxx.pt
 ↓
critic 只训练 Q/V
```

v2 应改成：

```text
prefix
 ↓
pretrain AE
 ↓
initialize critic state encoder
 ↓
critic training:
    RL-token encoder 继续更新
    +
    Q/V 更新
    +
    reconstruction regularizer
```

对应：

\[
L_{\rm critic}
=
L_{\rm IQL}
+
\alpha_{\rm rec}L_{\rm recon}.
\]

因此当前 `features_full.pt` 不能直接作为 v2 critic 的最终 state。

## 推荐实现

AE pretraining 后保存：

```text
rl_token_pretrained.pt
```

critic training 时：

```python
prefix, mask = buffer.prefix_batch(...)
recon_loss, info = rlt(prefix, mask)
z_rl = info["z_rl"]
state = state_encoder(z_rl, proprio)
```

再进入 critic。

### Decoder 处理建议

第一版采用：

```text
encoder: train
decoder: freeze
critic: train
V head: train
VLA backbone: freeze
```

reconstruction loss 仍可通过 frozen decoder 反传到 encoder。

如果后续能确认作者实现中 decoder 也 joint train，再增加：

```yaml
train_decoder_with_critic: true
```

---

# 5. v2 critic state

定义：

\[
s
=
\operatorname{LayerNorm}
\left(
[z_{\rm rl}\Vert W_pp]
\right)
\in\mathbb R^{2304}.
\]

其中：

\[
z_{\rm rl}\in\mathbb R^{2048},
\]

proprio projection：

\[
W_pp\in\mathbb R^{256}.
\]

新增：

```text
Q_vgm_v2/qvgm_v2/models/state_encoder.py
```

建议接口：

```python
class CriticStateEncoder(nn.Module):
    def __init__(
        self,
        proprio_dim=8,
        proprio_embed_dim=256,
        rl_token_dim=2048,
    ):
        ...

    def forward(self, z_rl, proprio):
        p = self.proprio_proj(proprio)
        s = torch.cat([z_rl, p], dim=-1)
        return self.norm(s)
```

最终：

```text
[B, 2048] + [B, 256]
        ↓
     [B, 2304]
```

当前仓库已有 `proprio_stats()` / `proprio_features()`，normalization helper 可以借鉴。

不要简单使用：

```python
torch.cat([z, raw_8d_proprio], -1)
```

---

# 6. v2 critic 必须重新实现

当前 v5 critic：

```python
Q(s,A) -> scalar
```

10 heads。

v2：

```text
Q1(s,A) -> [Q1^0,...,Q1^4]
Q2(s,A) -> [Q2^0,...,Q2^4]
```

即：

\[
Q_n(s,A)\in\mathbb R^H,
\qquad H=5.
\]

建议结构：

- 2 个 Q heads；
- hidden dimensions `[1024, 512]`；
- action chunk \(A\in\mathbb R^{5\times7}\)，flatten 为 35 维；
- action 在每一 hidden layer 重新注入；
- Value head 输出 \(V(s)\in\mathbb R^5\)。

推荐接口：

```python
class StepwiseQHead(nn.Module):
    def forward(self, state, action):
        # state:  [B,2304]
        # action: [B,5,7]
        # return: [B,5]
```

```python
class DoubleStepwiseCritic(nn.Module):
    def forward(self, state, action):
        q1 = ...
        q2 = ...
        return q1, q2

    def minimum(self, state, action):
        q1, q2 = self(...)
        return torch.minimum(q1, q2)

    def score(self, state, action):
        return self.minimum(state, action).sum(-1)
```

统一 guidance score：

\[
Q(s,A)
=
\sum_{i=0}^{H-1}
\min(Q_1^{(i)},Q_2^{(i)}).
\]

以后所有 guidance 都调用：

```python
critic.score(state, action)
```

---

# 7. Stepwise IQL

当前 v5 使用：

\[
R_{\rm chunk}+\gamma^nV(s')
\]

作为整个 chunk 的 TD target。

v2 应改成 stepwise target。

对于：

\[
A_t=[a_{t,0},...,a_{t,H-1}],
\]

buffer 提供：

\[
r_0,...,r_{H-1}
\]

和：

\[
d_0,...,d_{H-1}.
\]

target：

\[
y_i=
\begin{cases}
r_i+\gamma(1-d_i)V^{(i+1)}(s), & i<H-1,\\
r_i+\gamma(1-d_i)V^{(0)}(s'), & i=H-1.
\end{cases}
\]

还需要：

- partial chunk validity mask；
- mask 保留到第一个 terminal（包含 terminal）；
- partial next chunk 时 boundary bootstrap 要 mask；
- Value expectile target 使用 double-Q minimum；
- expectile \(\tau=0.8\)。

因此重写：

```text
Q_vgm_v2/qvgm_v2/algorithms/stepwise_iql.py
```

建议：

```python
class StepwiseIQL:
    def update(self, batch):
        ...
```

batch 至少包含：

```python
{
    "state":          [B,2304],
    "next_state":     [B,2304],
    "action":         [B,5,7],
    "rewards":        [B,5],
    "dones":          [B,5],
    "valid":          [B,5],
    "boundary_valid": [B],
}
```

---

# 8. Buffer schema：原始 rollout 大部分可以复用

当前 rollout shard 已保存：

```python
transition = {
    "action": ...
    "rewards": ...
    "steps": ...
    "terminated": ...
    "truncated": ...
}
```

因此可以构造 stepwise transition。

例如：

```text
action.shape = [5,7]
rewards      = [0,0,1]
steps        = 3
terminated   = True
```

转换为：

```text
rewards = [0,0,1,0,0]
valid   = [1,1,1,0,0]
dones   = [0,0,1,0,0]
```

建议新建：

```text
Q_vgm_v2/qvgm_v2/data/buffer.py
```

提供：

```python
stepwise_transition(e, i)
```

不要修改 v5 `qvgm/data/replay_buffer.py`。

---

# 9. v2 数据必须重新准备

正式 v2 不应继续使用 150 条 v5 rollout 作为唯一正式 buffer。

目标：

```text
500 SFT evaluation rollouts
+
LIBERO expert demos
```

Spatial：

```text
10 tasks × 50 episodes = 500 episodes
```

## 9.1 SFT rollout

建议 evaluation collection：

```yaml
temperature: 1.0
flow_noise: 0.0
episodes_per_task: 50
```

这里 `T=1, sigma=0` 应标注为 implementation assumption / standard inference protocol，而不是未核实的论文显式数值。

单独包装：

```text
Q_vgm_v2/scripts/collect_eval_rollouts.py
```

防止误用 v5 config。

---

# 10. LIBERO expert demos

新增：

```text
Q_vgm_v2/scripts/import_libero_demos.py
```

将 demo 转成统一 episode representation：

```python
episode = {
    "source": "rollout" | "demo",
    "task_id": ...,
    "observations": [...],
    "prefixes": [...],
    "transitions": [...]
}
```

其中 demo action 必须经过：

```text
environment action
        ↓
π0.5 normalization
        ↓
normalized [5,7] chunk
```

不能直接把 LIBERO action 当成 critic action。

---

# 11. 不要直接合并 dataset 文件

建议：

```text
artifacts/v2_spatial/
├── rollout_buffer/
├── demo_buffer/
└── combined_manifest.json
```

`V2ReplayBuffer`：

```python
V2ReplayBuffer(
    rollout_dir=...,
    demo_dir=...,
)
```

后续方便做：

```text
rollout-only
demo-only
rollout+demos
```

消融。

---

# 12. RL-token pretraining

训练 500 rollouts + demos 的 prefix。

按 episode split train / validation。

记录：

```text
reconstruction_mse
reconstruction_cosine
```

保存：

```text
best_rl_token.pt
```

建议 gate：held-out reconstruction cosine > 0.95 后再进入 critic。

---

# 13. Critic training 时不能使用 stale feature cache

critic 阶段：

```text
prefix
   ↓
trainable RL-token encoder
   ↓
z_rl
   ↓
proprio projection
   ↓
LayerNorm
   ↓
s
   ↓
Double Stepwise Q + V
```

loss：

\[
L=L_Q+L_V+\alpha_{\rm rec}L_{\rm recon}.
\]

critic training 每一个 batch 都必须访问 prefix，不能直接加载 frozen `features.pt`。

critic 完成以后才：

```text
freeze RL-token encoder
freeze state encoder
freeze Q
```

然后生成：

```text
critic_states_final.pt
```

final cache 绑定：

```text
critic checkpoint hash
buffer signature
RL-token checkpoint
```

---

# 14. v2 Q-guidance 必须重新写

当前 v5：

```python
direction = grad / ||grad||
candidate = action + alpha * direction
```

并且 first non-improvement stop。

v2 应实现：

\[
A^{j+1}=A^j+\alpha\operatorname{clip}_G(\nabla_AQ).
\]

得到：

\[
A^0,A^1,\dots,A^J
\]

最后：

\[
j^\star=\arg\max_jQ(s,A^j).
\]

即：

```text
不是 normalized gradient
不是 first failure stop
而是 gradient norm clipping + global keep-best
```

新增：

```text
Q_vgm_v2/qvgm_v2/algorithms/q_guidance.py
```

推荐：

```python
def clip_gradient(grad, max_norm):
    norm = grad.flatten(1).norm(dim=-1)
    scale = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
    return grad * scale.view(-1, 1, 1)
```

然后：

```python
candidates = [action]

for _ in range(J):
    grad = dQdA(...)
    grad = clip_gradient(grad, G)
    action = action + alpha * grad
    candidates.append(action)

scores = stack(Q(candidate))
best = candidates[argmax(scores)]
```

必须有单元测试。

---

# 15. v2 velocity matching

这部分与现有实现接近，可以复用主体：

\[
\hat A_{\rm base}^{[k]}
=
x^{[k]}+(1-\tau_k)v_{\rm base}(x^{[k]},\tau_k).
\]

\[
\hat h_Q^{[k]}
=
\frac{\hat A_Q^{[k]}-\hat A_{\rm base}^{[k]}}{1-\tau_k}.
\]

\[
L_{\rm align}
=
\sum_{k=K-M}^{K-1}
\left\|
(v_\theta-v_{\rm base})
-
\operatorname{sg}[\hat h_Q]
\right\|^2.
\]

trajectory、base endpoint、critic candidate、target correction 全 detach，gradient 只通过 local actor velocity prediction。

当前 `qvgm_loss.py` 主体可复用，但需替换 guidance API 和 critic score API。

---

# 16. Action horizon 问题仍然存在

论文 critic 使用：

\[
H=5,\quad d_a=7.
\]

released π0.5 checkpoint 的实际网络 contract：

```text
policy action horizon = 10
policy action dim     = 32
```

建议 v2 第一版保持：

```yaml
policy_action_horizon: 10
policy_action_dim: 32
critic_horizon: 5
critic_action_dim: 7
```

不要为了字面 H=5 重新实例化 checkpoint。

因此准确名称应为：

> **Q-VGM v2 algorithm on the released few-shot checkpoint contract**

而不是 exact author checkpoint reproduction。

---

# 17. Actor interface

critic 训练完成后：

```text
buffer state
   ↓
final frozen RL encoder
   ↓
final state encoder
   ↓
cache s ∈ R^2304
```

保存：

```text
critic_states_final.pt
```

actor training：

```text
observation
    ↓
Pi05Flow.encode_context()
    ↓
actor trajectory

cached state s
    ↓
frozen critic
```

这样 VLA forward 与 critic-state forward 解耦。

---

# 18. 建议的主要接口

## Buffer

```python
class V2ReplayBuffer:
    def transition(self, e, i):
        return {
            "prefix": ...,
            "prefix_mask": ...,
            "proprio": ...,
            "next_prefix": ...,
            "next_prefix_mask": ...,
            "next_proprio": ...,
            "action": ...,       # [5,7]
            "rewards": ...,      # [5]
            "dones": ...,        # [5]
            "valid": ...,        # [5]
            "boundary_valid": ...,
        }
```

## State encoder

```python
state_encoder(z_rl, proprio) -> [B,2304]
```

## Critic

```python
q1, q2 = critic(state, action)      # [B,5], [B,5]
q_min = critic.minimum(state, action) # [B,5]
score = critic.score(state, action)   # [B]
```

## Value

```python
value(state)  # [B,5]
```

## Guidance

```python
improve_actions(
    score_fn,
    state,
    action,
    ascent_steps,
    alpha,
    grad_clip,
)
```

## Actor loss

```python
local_velocity_loss(
    actor,
    base,
    context,
    critic_state,
    critic,
    tau,
    x_tau,
    cfg,
)
```

---

# 19. 单元测试必须先完成

至少写六类测试。

## 19.1 Stepwise TD target

手工设：

```text
H = 5
rewards = [0,0,0,0,1]
dones   = [0,0,0,0,1]
```

验证：

\[
y_0=\gamma V^1(s),\quad
...,
\quad y_4=1.
\]

再测试：

```text
partial chunk
terminal at i=2
timeout
```

## 19.2 Double-Q minimum

确认：

\[
Q_{\min}=\min(Q_1,Q_2)
\]

逐 position 发生，而不是：

```text
min(sum(Q1), sum(Q2))
```

## 19.3 score

验证：

\[
Q(s,A)=\sum_i\min(Q_1^{(i)},Q_2^{(i)}).
\]

## 19.4 Gradient clipping

构造：

\[
\nabla_AQ=(3,4),\quad G=2.
\]

clip 后 norm 应为 2，而不是 1。

## 19.5 Keep-best

例如：

```text
Q(A0)=1
Q(A1)=2
Q(A2)=1.5
Q(A3)=2.5
```

最终应选 `A3`，不能在 `A2` 时停止。

## 19.6 Gradient isolation

确认：

```text
critic.grad          = None
base_policy.grad     = None
trajectory.grad      = None
current actor local velocity 有 grad
```

---

# 20. 实验顺序

不要直接：

```text
critic → actor 500 steps → 看成功率
```

v2 建立严格 gate。

## Phase 0：baseline

同一个 harness 下评估 few-shot SFT：

```text
50 episodes/task
10 tasks
= 500
```

这些 episodes 同时作为 v2 rollout dataset。

记录：

```text
overall success
per-task success
episode length
seed
initial-state index
```

## Phase 1：构建 dataset

目标：

```text
500 SFT evaluation rollouts
+
LIBERO expert demos
```

输出：

```text
dataset_report.json
```

包括：

```text
rollout episodes
demo episodes
success/failure count
transition count
valid action positions
action mean/std
per-dimension std
proprio mean/std
```

尤其检查 rotation action std。

## Phase 2：RL-token

建议 gate：

```text
held-out cosine > 0.95
```

输出：

```text
rl_token_pretrained.pt
rl_token_history.jsonl
```

## Phase 3：critic

训练：

```text
RL-token encoder
+
proprio projection
+
LayerNorm
+
Double Q
+
V
```

记录：

```text
q_loss
v_loss
recon_loss
Q success/failure
Q1-Q2 disagreement
gradient norm
gradient energy per action dimension
gradient cosine across states
global gradient direction explained variance
```

## Phase 4：critic 环境验证

### 4.1 Q selection

同 state 采多个 SFT action：

\[
A_1,\dots,A_N.
\]

选择：

\[
A^\star=\arg\max_iQ(s,A_i).
\]

如果 Q-selection 不优于 SFT，就暂时不要 actor training。

### 4.2 Test-time Q guidance

直接：

```text
SFT action
 ↓
v2 clipped ∇Q ascent
 ↓
execute improved action
```

不训练 actor。

原则：

\[
\boxed{
\text{如果 critic guidance 本身不能帮助环境，
就没有理由期待把它 distill 进 actor 后会突然变好。}
}
\]

## Phase 5：same-state counterfactual

对同一 simulator state：

```text
A_base
vs
A_Q
```

保持：

```text
同 initial state
同 continuation noise
同 frozen SFT continuation
```

测：

\[
\Delta Return=R(A_Q)-R(A_{\rm base}).
\]

同时统计：

\[
\operatorname{corr}
(
Q(A_Q)-Q(A_{\rm base}),
\Delta Return
).
\]

只有环境收益为正且 gradient 不再退化成 global constant direction，才进入 actor。

## Phase 6：actor Q-VGM

critic gate 通过后再训练 actor。

先：

```text
25 steps
```

smoke test。

然后：

```text
100
250
500
```

分阶段保存 actor。

每阶段先：

```text
10 episodes/task
```

快速 evaluation；正式 candidate 再 50 episodes/task。

---

# 21. 必须有 zero-guidance control

保留：

```text
alpha = 0
```

控制。

理论上：

\[
\hat h_Q=0
\]

actor 不应发生实质更新，其 success 应接近同 harness SFT。

如果：

```text
zero-guidance ≠ SFT
```

先查 pipeline，不能分析 critic。

---

# 22. 推荐正式实验矩阵

| ID | Dataset | Critic | Guidance | Actor |
|---|---|---|---|---|
| A | — | — | — | SFT |
| B | rollout+demos | v2 | Q selection | frozen |
| C | rollout+demos | v2 | test-time Q guidance | frozen |
| D | rollout+demos | v2 | α=0 | Q-VGM actor |
| E | rollout+demos | v2 | v2 guidance | Q-VGM actor |

如果 E 有提升，再做：

```text
rollout-only
single Q
no proprio
no reconstruction
no per-layer injection
```

---

# 23. Hyperparameter 管理

结构参数建议固定：

```text
H = 5
action dim = 7
RL token = 2048
RL-token layers = 2
heads = 8
proprio projection = 256
critic state = 2304
critic Q heads = 2
critic hidden = [1024,512]
IQL expectile = 0.8
late steps M = 5
```

对于未充分公开的参数：

```text
alpha_rec
critic LR
target EMA coefficient
critic update count
J
alpha
gradient clipping G
actor LR
actor batch
actor update count
```

必须在 config 中标为 engineering assumption。

---

# 24. 第一版 config 建议

```yaml
version: qvgm_v2

data:
  rollout_episodes_per_task: 50
  use_demos: true

model:
  policy_action_horizon: 10
  policy_action_dim: 32
  denoising_steps: 10

env:
  action_chunk: 5
  action_dim: 7

rl_token:
  dim: 2048
  layers: 2
  heads: 8

critic:
  q_heads: 2
  widths: [1024, 512]
  proprio_dim: 8
  proprio_embed_dim: 256
  expectile: 0.8

  # Engineering assumptions
  lr: 0.0001
  target_ema: 0.005
  alpha_rec: null
  steps: null

guidance:
  late_steps: 5
  # Engineering defaults
  ascent_steps: 3
  ascent_step_size: 0.05
  grad_clip: 1.0

actor:
  lr: 0.000005
  steps: 500
```

---

# 25. 关于 `grad_clip_G`

第一版可先设：

```yaml
gradient_clip_G: 1.0
alpha: 0.05
J: 3
```

则每一步 displacement 上界：

\[
\alpha G=0.05,
\]

三步最大：

\[
0.15.
\]

这样与现有 v5 normalized-gradient 实验的 displacement 同量级，方便比较。

但这是工程设计，不应写成论文已公布超参。

---

# 26. `alpha_rec` 不建议直接猜

因为：

\[
L_{\rm IQL}
\]

和：

\[
L_{\rm recon}
\]

数值尺度可能不同。

建议先运行 100 个 critic warmup batches，只记录：

```text
unweighted IQL loss
unweighted recon loss
```

再选择 \(\alpha_{\rm rec}\)，使 reconstruction term 初始约占总 critic loss 的 5%–20%，然后固定。

---

# 27. 推荐训练依赖图

```text
few-shot SFT checkpoint
        │
        ├─────────────┐
        │             │
        ▼             ▼
500 eval rollouts   LIBERO demos
        │             │
        └──────┬──────┘
               ▼
          unified buffer
               │
               ▼
       frozen VLA prefixes
               │
               ▼
        RL-token pretrain
               │
               ▼
     joint critic training
               │
        ┌──────┴────────┐
        ▼               ▼
 final state cache    frozen critic
        │               │
        └──────┬────────┘
               ▼
      critic diagnostics
               │
          PASS ONLY
               ▼
       Q selection test
               │
          PASS ONLY
               ▼
        Q guidance test
               │
          PASS ONLY
               ▼
      counterfactual A/B
               │
          PASS ONLY
               ▼
        actor Q-VGM
               │
               ▼
          actor eval
```

---

# 28. 文件复用总结

| 当前文件 | v2 处理 |
|---|---|
| `qvgm/models/pi05_adapter.py` | ✅ 直接复用 |
| `qvgm/envs/libero_env.py` | ✅ 直接复用 |
| `qvgm/models/rl_token.py` | ✅ architecture 直接复用 |
| `qvgm/config.py` | ✅ 可复用 config/runtime 思路 |
| `qvgm/data/replay_buffer.py` | ⚠️ 底层格式可参考，写 v2 wrapper |
| `qvgm/models/critic.py` | ❌ 不复用 |
| `qvgm/algorithms/iql.py` | ❌ 不复用 |
| `qvgm/algorithms/q_guidance.py` | ❌ 不复用 |
| `qvgm/algorithms/qvgm_loss.py` | ⚠️ 主体复用，换 guidance API |
| `scripts/train_rl_token.py` | ⚠️ pretrain 主体可复用 |
| `scripts/train_critic.py` | ❌ 重写 |
| `scripts/train_offline_qvgm.py` | ⚠️ actor skeleton 可复用 |
| v5 recovery/constant bias scripts | ❌ 不进入 v2 主实现 |

---

# 29. 第一批真正应该实现的文件

第一阶段：

```text
Q_vgm_v2/
├── configs/libero_spatial_v2.yaml
│
├── qvgm_v2/
│   ├── data/buffer.py
│   ├── models/state_encoder.py
│   ├── models/critic.py
│   └── algorithms/stepwise_iql.py
│
└── tests/
    ├── test_buffer.py
    ├── test_state_encoder.py
    ├── test_critic.py
    └── test_stepwise_iql.py
```

第二阶段：

```text
collect_eval_rollouts
import_libero_demos
train_rl_token
train_critic
```

第三阶段：

```text
q_guidance
Q selection
Q guidance
```

critic 环境验证通过后，再实现：

```text
qvgm_loss
train_qvgm
eval_qvgm
```

---

# 30. 复现成功标准

## Algorithmic reproduction

满足：

```text
v2 critic architecture 正确
stepwise IQL 正确
500 rollouts + demos
RL-token joint training
Q selection / Q guidance 有实际提升
Q-VGM actor 相对 SFT 有稳定提升
```

可认为 v2 algorithmic reproduction successful。

## Numerical reproduction

只有当：

```text
SFT baseline ≈ 85.6%
```

且 checkpoint、harness、数据条件能和论文充分对应时，才应该拿论文的 96.2% 作为数值复现目标。

如果起点仍然是 75–80%，则不能简单把“没到 96.2%”定义成失败。

---

# 31. v2 项目的核心诊断目标

最大价值不是单纯再跑一遍旧论文，而是回答：

\[
\boxed{
v2\text{ 的 critic 设计是否能避免 v5 中出现的状态无关动作梯度？}
}
\]

重点比较：

### v5

\[
\nabla_AQ(s,A)
\approx
\text{constant direction}.
\]

### v2

检查不同状态下的 \(\nabla_AQ\) 是否真正随：

```text
state
task phase
proprioception
action
```

而变化。

如果 v2：

```text
Q-selection improves
Q-guidance improves
gradient is state-dependent
```

而 v5 不行，那么问题基本可定位到 v5 critic/data simplification 导致 action-gradient quality 丢失。

如果 v2 仍退化成 constant direction，则更应该怀疑：

```text
checkpoint contract
action normalization
demo normalization
prefix/state construction
LIBERO harness
```

---

# 32. 最终执行原则

先冻结 v5，不再继续围绕：

```text
alpha
LR
proprio arm
critic aggregation
```

反复调参。

新建 `Q_vgm_v2/`，严格按：

\[
\boxed{
\text{data}
\rightarrow
\text{RL-token}
\rightarrow
\text{stepwise critic}
\rightarrow
\text{critic environment validation}
\rightarrow
\text{actor}
}
\]

推进。

尤其：

\[
\boxed{
\textbf{不要在 Q-selection / Q-guidance 通过之前训练 v2 actor。}
}
\]

这是本次 v2 复现相较 v5 最重要的流程改进。

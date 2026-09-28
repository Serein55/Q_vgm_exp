"""Collect only SFT self-rollouts, resumable at episode boundaries."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--run", default="spatial_offline")
    p.add_argument("--episodes-per-task", type=int)
    p.add_argument("--task-ids", type=int, nargs="+")
    p.add_argument("--check-flow", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy

    n = args.episodes_per_task or cfg["offline"]["episodes_per_task"]
    task_ids = args.task_ids if args.task_ids is not None else cfg["env"]["task_ids"]
    if n <= 0:
        p.error("episodes-per-task must be positive")
    out = Path(cfg["paths"]["artifacts"]) / args.run / "buffer"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "config.json"

    # Keep original semantics immutable when resuming/expanding episode budget.
    def collection_signature(config):
        return dict(
            model=config["model"],
            env=config["env"],
            seed=config["runtime"]["seed"],
            checkpoint=config["paths"]["checkpoint"],
            norm_stats=config["paths"]["norm_stats"],
            temperature=config["offline"]["temperature"],
            sigma=config["offline"]["flow_noise"],
            gamma=config["offline"]["gamma"],
        )

    if manifest.exists():
        if collection_signature(json.loads(manifest.read_text())) != collection_signature(cfg):
            raise ValueError(
                "Existing buffer uses different collection settings; choose a new --run"
            )
    else:
        manifest.write_text(json.dumps(cfg, indent=2))
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    if len(set(task_ids)) != len(task_ids) or any(t < 0 or t >= suite.n_tasks for t in task_ids):
        raise ValueError("Invalid task ids")
    checked = False
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial = suite.get_task_init_states(task_id)
        if n > len(initial):
            raise ValueError("Insufficient distinct initial states")
        env = make_env(cfg, task, cfg["runtime"]["seed"])
        try:
            for episode in range(n):
                dest = out / f"task{task_id:02d}_episode{episode:03d}.pt"
                if dest.exists():
                    continue
                start = time.monotonic()
                seed = cfg["runtime"]["seed"] + 100000 + task_id * 10000 + episode
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                obs = env.set_init_state(initial[episode])
                for _ in range(cfg["env"]["settle_steps"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                observations, prefixes, transitions = [], [], []
                steps, success, success_once = 0, False, False
                inp = observation_for_policy(obs, task.language)
                context = flow.encode_context([inp])

                def store_state(inp, context):
                    observations.append(
                        {
                            k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                            for k, v in inp.items()
                        }
                    )
                    prefixes.append(
                        context["prefix"][0, context["pad"][0]].cpu().to(torch.bfloat16)
                    )

                store_state(inp, context)
                if args.check_flow and not checked:
                    rng_state = torch.cuda.get_rng_state()
                    noise = torch.randn(
                        1,
                        cfg["model"]["action_horizon"],
                        cfg["model"]["action_dim"],
                        device=flow.device,
                    )
                    clean, traj = flow.sample_with_intermediates(context, noise)
                    ours = flow.unnormalize(clean, context)[0]
                    official = policy.infer(inp, noise=noise[0].cpu().numpy())["actions"]
                    # Different float timestep accumulation can shift bf16 activations.
                    error = float(np.max(np.abs(ours - official)))
                    if not np.allclose(ours, official, atol=0.02, rtol=0.02):
                        raise AssertionError(f"Flow/OpenPI sampling mismatch: {error}")
                    print(
                        json.dumps(
                            dict(
                                flow_max_abs_error=error,
                                trajectory_detached=all(not x.requires_grad for _, x in traj),
                            )
                        ),
                        flush=True,
                    )
                    (out / "flow_check.json").write_text(json.dumps(dict(max_abs_error=error)))
                    torch.cuda.set_rng_state(rng_state)
                    checked = True
                while steps < cfg["env"]["max_steps"]:
                    normalized, _ = flow.sample_with_intermediates(
                        context,
                        temperature=cfg["offline"]["temperature"],
                        sigma=cfg["offline"]["flow_noise"],
                    )
                    executed = flow.unnormalize(normalized, context)[0][
                        : cfg["env"]["action_chunk"]
                    ]
                    if not np.isfinite(executed).all():
                        raise ValueError("Non-finite sampled actions")
                    rewards = []
                    terminated = False
                    for action in executed:
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        success = bool(env.check_success())
                        success_once = success_once or success
                        rewards.append(float(success))  # Sparse success reward.
                        terminated = bool(done or success)
                        if terminated or steps >= cfg["env"]["max_steps"]:
                            break
                    inp = observation_for_policy(obs, task.language)
                    context = flow.encode_context([inp])
                    store_state(inp, context)
                    transitions.append(
                        dict(
                            action=normalized[
                                0, : cfg["env"]["action_chunk"], : cfg["env"]["action_dim"]
                            ].cpu(),
                            reward=sum(
                                cfg["offline"]["gamma"] ** i * r for i, r in enumerate(rewards)
                            ),
                            rewards=torch.tensor(rewards),
                            steps=len(rewards),
                            terminated=terminated,
                            truncated=not terminated and steps >= cfg["env"]["max_steps"],
                        )
                    )
                    if terminated:
                        break
                record = dict(
                    schema=2,
                    source="rollout",
                    policy_source="fewshot_sft",
                    task_id=task_id,
                    episode=episode,
                    seed=seed,
                    success=success,
                    success_once=success_once,
                    initial_state_index=episode,
                    observations=observations,
                    # Proprio tokenization can change prefix length within an episode.
                    prefixes=prefixes,
                    transitions=transitions,
                )
                temp = dest.with_suffix(".tmp")
                torch.save(record, temp)
                temp.replace(dest)
                metrics = dict(
                    task_id=task_id,
                    episode=episode,
                    success=success,
                    success_once=success_once,
                    initial_state_index=episode,
                    steps=steps,
                    chunks=len(transitions),
                    seconds=time.monotonic() - start,
                )
                with (out / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
        finally:
            env.close()
    print(f"Buffer ready: {out}", flush=True)


if __name__ == "__main__":
    main()

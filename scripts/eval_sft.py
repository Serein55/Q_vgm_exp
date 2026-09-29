"""Standalone SFT evaluation. Run via run.sh to select the YAML interpreter."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("--name", default="sft")
    parser.add_argument("--actor-checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.episodes_per_task is not None:
        if args.episodes_per_task < 1:
            parser.error("episodes-per-task must be positive")
        cfg["evaluation"]["episodes_per_task"] = args.episodes_per_task
    if args.task_ids is not None:
        cfg["env"]["task_ids"] = args.task_ids
    if args.cpu_smoke:
        cfg["runtime"].update(device="cpu", mujoco_gl="osmesa")
    setup_runtime(cfg)

    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import infer_actions, load_sft_policy

    seed = cfg["runtime"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    task_ids = cfg["env"]["task_ids"]
    if len(set(task_ids)) != len(task_ids) or any(i < 0 or i >= suite.n_tasks for i in task_ids):
        raise ValueError("Invalid or duplicate task ids")
    count = cfg["evaluation"]["episodes_per_task"]
    states = {i: suite.get_task_init_states(i) for i in task_ids}
    if any(count > len(s) for s in states.values()):
        raise ValueError("Not enough distinct LIBERO initial states")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(cfg["paths"]["artifacts"]) / "eval" / f"{args.name}-{stamp}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"OUTPUT={out}", flush=True)
    start = time.monotonic()
    policy = None if args.env_only else load_sft_policy(cfg)
    if args.actor_checkpoint:
        if args.env_only:
            parser.error("Cannot combine env-only with actor-checkpoint")
        from qvgm.models.pi05_adapter import Pi05Flow

        ck = torch.load(args.actor_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if ck["config"]["model"] != cfg["model"]:
            raise ValueError("Evaluation model configuration differs from actor checkpoint")
        flow = Pi05Flow(policy, cfg)
        flow.enable_actor_training(ck["settings"].get("master_dtype", "float32"))
        expected = {n for n, p in flow.model.named_parameters() if p.requires_grad}
        if set(ck["actor"]) != expected:
            raise ValueError("Actor checkpoint has missing or unexpected parameters")
        flow.model.load_state_dict(ck["actor"], strict=False)
        flow.model.requires_grad_(False).eval()
        policy._qvgm_flow = flow
        del ck
        (out / "actor_checkpoint.txt").write_text(str(Path(args.actor_checkpoint).resolve()))
    print(f"Model ready in {time.monotonic() - start:.1f}s", flush=True)
    records = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env = make_env(cfg, task, seed)
        try:
            for episode in range(count):
                # Same seed and initial state for paired SFT/Q-VGM evaluation.
                episode_seed = seed + task_id * 10000 + episode
                rng = np.random.default_rng(episode_seed)
                env.seed(episode_seed)
                env.reset()
                obs = env.set_init_state(states[task_id][episode])
                for _ in range(cfg["env"]["settle_steps"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                success = False
                steps = 0
                inference_ms = []
                ep_start = time.monotonic()
                while steps < cfg["env"]["max_steps"]:
                    inp = observation_for_policy(obs, task.language)
                    if args.env_only:
                        np.savez_compressed(out / f"observation-{task_id}.npz", **inp)
                        print(
                            f"Environment rendered: task={task_id}, image={inp['observation/image'].shape}",
                            flush=True,
                        )
                        break
                    result = infer_actions(policy, inp, cfg, rng)
                    inference_ms.append(result["policy_timing"]["infer_ms"])
                    for action in result["actions"][: cfg["env"]["action_chunk"]]:
                        obs, reward, done, info = env.step(action.tolist())
                        steps += 1
                        # LIBERO's task success is distinct from our time limit.
                        success = bool(env.check_success())
                        if success or done or steps >= cfg["env"]["max_steps"]:
                            break
                    if success or done:
                        break
                record = dict(
                    task_id=task_id,
                    episode=episode,
                    seed=episode_seed,
                    success=success,
                    steps=steps,
                    seconds=time.monotonic() - ep_start,
                    mean_inference_ms=float(np.mean(inference_ms)) if inference_ms else None,
                )
                records.append(record)
                with (out / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        finally:
            env.close()
    summary = dict(
        status="env_smoke" if args.env_only else "complete",
        episodes=len(records),
        successes=sum(r["success"] for r in records),
        success_rate=None if args.env_only else float(np.mean([r["success"] for r in records])),
        seconds=time.monotonic() - start,
        config=cfg,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()

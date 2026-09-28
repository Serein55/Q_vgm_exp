"""Frozen retrieval policy deployed every chunk, with paired SFT/fixed controls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_policy import FrozenRetrievalPolicy

from qvgm.config import load_config, setup_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--worker", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy

    c = cfg["recovery"]
    out = Path(cfg["paths"]["artifacts"]) / "retrieval_closed_loop" / f"worker{args.worker}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.txt").write_text("running\n")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    retrieval = FrozenRetrievalPolicy(cfg["paths"]["retrieval_policy"], c["retrieval_sha256"])
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    flow.enable_actor_training(cfg["offline"]["actor"]["master_dtype"])
    flow.model.requires_grad_(False)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    jobs = [(t, e, k) for t in c["task_ids"] for e in range(45, 50) for k in range(4)]
    H, D = cfg["model"]["action_horizon"], cfg["model"]["action_dim"]
    for job, (task_id, episode, repeat) in enumerate(jobs):
        if job % c["workers"] != args.worker:
            continue
        dest = out / f"task{task_id:02d}_episode{episode:03d}_repeat{repeat:02d}.json"
        if dest.exists():
            continue
        task = suite.get_task(task_id)
        initial = suite.get_task_init_states(task_id)[episode]
        env_seed = cfg["runtime"]["seed"] + 800000 + task_id * 10000 + episode
        # New noise stream; initialization IDs reused from stage one, so not a new
        # independent held-out state set. Compare closed-loop behavior only.
        noise_seed = cfg["runtime"]["seed"] + 4000000 + task_id * 10000 + episode * 100 + repeat
        results = {}
        common_state = None
        common_obs = None
        for method in ["base", "fixed", "retrieval"]:
            np.random.seed(env_seed)
            torch.manual_seed(env_seed)
            env = make_env(cfg, task, env_seed)
            try:
                env.seed(env_seed)
                env.reset()
                obs = env.set_init_state(initial)
                for _ in range(cfg["env"]["settle_steps"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                inp = observation_for_policy(obs, task.language)
                if common_state is None:
                    common_state = env.get_sim_state().copy()
                    common_obs = inp
                state_error = float(np.max(np.abs(env.get_sim_state() - common_state)))
                image_error = max(
                    float(np.abs(inp[k].astype(float) - common_obs[k].astype(float)).mean())
                    for k in ["observation/image", "observation/wrist_image"]
                )
                assert state_error <= 1e-6 and image_error <= 1
                rng = np.random.default_rng(noise_seed)
                elapsed = 0
                chunk_index = 0
                success = False
                choices = []
                while elapsed < cfg["env"]["max_steps"]:
                    inp = observation_for_policy(obs, task.language)
                    context = flow.encode_context([inp])
                    eps = torch.from_numpy(rng.standard_normal((1, H, D)).astype(np.float32))
                    base, _ = flow.sample_with_intermediates(context, eps)
                    action = flow.unnormalize(base, context)[0, :5].copy()
                    if method == "retrieval":
                        pooled = (
                            context["prefix"][0, context["pad"][0]].float().mean(0).cpu().numpy()
                        )
                        pick, _, _ = retrieval.choose(
                            pooled,
                            inp["observation/state"],
                            base[0, :5, :7].cpu().numpy(),
                            task_id,
                            chunk_index,
                        )
                    else:
                        pick = 0 if method == "base" else 2
                    choices.append(retrieval.ck["candidate_names"][pick])
                    if pick:
                        axis = [0, 1, 2, 6][(pick - 1) // 2]
                        sign = 1 if pick % 2 else -1
                        action[:, axis] = (
                            sign
                            if axis == 6
                            else np.clip(action[:, axis] + sign * c["controller_delta"], -1, 1)
                        )
                    for a in action:
                        obs, _, done, _ = env.step(a.tolist())
                        elapsed += 1
                        success = bool(env.check_success())
                        if done or success or elapsed >= cfg["env"]["max_steps"]:
                            break
                    chunk_index += 1
                    if done or success:
                        break
                results[method] = dict(
                    success=success,
                    steps=elapsed,
                    choices=choices,
                    sim_error=state_error,
                    image_error=image_error,
                )
            finally:
                env.close()
        payload = dict(
            task=task_id,
            episode=episode,
            repeat=repeat,
            checkpoint_sha256=retrieval.sha256,
            results=results,
        )
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload) + "\n")
        tmp.replace(dest)
        print(json.dumps(payload), flush=True)
    (out / "status.txt").write_text("complete\n")


if __name__ == "__main__":
    main()

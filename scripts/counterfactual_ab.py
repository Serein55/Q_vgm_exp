"""Paired counterfactual A/B: clean SFT chunk vs Q-ascent chunk at matched states.

Answers whether grad_A Q points at real environment improvement: the same replayed
state is continued twice with identical noise, once with the policy chunk and once
with the chunk moved by normalized Q ascent. Only that first chunk is modified; the
rest of the episode is frozen SFT. Nothing here trains or evaluates the Q-VGM actor.
"""

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/libero_spatial.yaml")
    p.add_argument("--run", default="spatial_v5_h10")
    p.add_argument("--tag", default="full")
    p.add_argument("--name", required=True)
    p.add_argument("--task-ids", nargs="+", type=int)
    p.add_argument("--episodes-per-task", type=int, default=2)
    p.add_argument("--states-per-episode", type=int, default=3)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05])
    p.add_argument("--state-tolerance", type=float, default=1e-3)
    args = p.parse_args()
    if Path(args.name).name != args.name:
        p.error("name must be a single directory name")
    for key, value in (
        ("episodes-per-task", args.episodes_per_task),
        ("states-per-episode", args.states_per_episode),
        ("repeats", args.repeats),
    ):
        if value < 1:
            p.error(f"{key} must be positive")
    if not all(0 <= a < float("inf") for a in args.alphas) or 0.0 not in args.alphas:
        p.error("alphas must be finite, nonnegative, and include the zero control")
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import buffer_signature, make_critic, proprio_features

    device = cfg["runtime"]["device"]
    root = Path(cfg["paths"]["artifacts"]) / args.run
    out = root / args.name
    out.mkdir(exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(config=cfg, args=vars(args)), indent=2))
    (out / "status.txt").write_text("running\n")

    def mark_unfinished():
        if (out / "status.txt").read_text().strip() == "running":
            (out / "status.txt").write_text("failed\n")

    atexit.register(mark_unfinished)

    buffer = ReplayBuffer(root / "buffer")
    signature = buffer_signature(buffer)
    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    if features["buffer_signature"] != signature:
        raise ValueError("Stale feature cache")
    ck = torch.load(root / f"critic_{args.tag}.pt", weights_only=True, map_location="cpu")
    if ck["buffer_signature"] != signature:
        raise ValueError("Critic provenance mismatch")
    stats = ck.get("proprio_stats")
    extra = 0 if stats is None else int(stats["mean"].numel())
    critic = make_critic(cfg, extra).to(device).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    del ck

    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    h, d = cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
    horizon, dim = cfg["model"]["action_horizon"], cfg["model"]["action_dim"]
    ascent = cfg["offline"]["actor"]
    alphas = sorted(set(args.alphas))

    def critic_state(e, index):
        z = features["z"][e][index][None].to(device)
        if stats is not None:
            z = torch.cat([z, proprio_features(buffer, [(e, index)], stats).to(device)], -1)
        return z

    def guide(z, normalized, alpha):
        improved, metrics = improve_actions(
            critic.mean,
            z,
            normalized[:, :h, :d],
            steps=ascent["ascent_steps"],
            alpha=alpha,
        )
        normalized[:, :h, :d] = improved
        return metrics

    records = []
    selected = [
        (e, ep)
        for e, ep in enumerate(buffer.episodes)
        if ep["episode"] < args.episodes_per_task
        and (args.task_ids is None or ep["task_id"] in args.task_ids)
    ]
    for e, ep in selected:
        task = suite.get_task(ep["task_id"])
        initial = suite.get_task_init_states(ep["task_id"])[ep["episode"]]
        chunks = len(ep["transitions"])
        for k in range(args.states_per_episode):
            state_index = int(chunks * (k + 1) / (args.states_per_episode + 1))
            if state_index >= chunks:
                continue
            recorded = buffer.observation(e, state_index)
            z = critic_state(e, state_index)
            for repeat in range(args.repeats):
                # One seed per (state, repeat) shared by every alpha keeps the A/B paired.
                pair_seed = (
                    cfg["runtime"]["seed"]
                    + 5_000_000
                    + ep["task_id"] * 100_000
                    + ep["episode"] * 1_000
                    + state_index * 10
                    + repeat
                )
                for alpha in alphas:
                    start = time.monotonic()
                    np.random.seed(ep["seed"])
                    torch.manual_seed(ep["seed"])
                    env = make_env(cfg, task, ep["seed"])
                    record = dict(
                        task=ep["task_id"],
                        episode=ep["episode"],
                        state_index=state_index,
                        repeat=repeat,
                        alpha=alpha,
                    )
                    try:
                        env.seed(ep["seed"])
                        env.reset()
                        obs = env.set_init_state(initial)
                        for _ in range(cfg["env"]["settle_steps"]):
                            obs, _, _, _ = env.step(DUMMY_ACTION)
                        elapsed = 0
                        for i in range(state_index):
                            inp = buffer.observation(e, i)
                            transformed = policy._input_transform(dict(inp))
                            context = {
                                "state": torch.as_tensor(transformed["state"], device=flow.device)[
                                    None
                                ]
                            }
                            normalized = torch.zeros(1, horizon, dim, device=flow.device)
                            transition = ep["transitions"][i]
                            normalized[0, :h, :d] = transition["action"].to(flow.device)
                            for action in flow.unnormalize(normalized, context)[0][
                                : transition["steps"]
                            ]:
                                obs, _, done, _ = env.step(action.tolist())
                                elapsed += 1
                                if done or env.check_success():
                                    raise RuntimeError("Replay terminated early")
                        reconstructed = observation_for_policy(obs, task.language)
                        record["state_error"] = float(
                            np.max(
                                np.abs(
                                    reconstructed["observation/state"]
                                    - recorded["observation/state"]
                                )
                            )
                        )
                        record["image_error"] = max(
                            float(
                                np.mean(
                                    np.abs(
                                        reconstructed[key].astype(float)
                                        - recorded[key].astype(float)
                                    )
                                )
                            )
                            for key in ("observation/image", "observation/wrist_image")
                        )
                        if record["state_error"] > args.state_tolerance:
                            raise RuntimeError(f"Replay drifted: state={record['state_error']:.3e}")
                        rng = np.random.default_rng(pair_seed)
                        context = flow.encode_context([reconstructed])
                        noise = torch.from_numpy(
                            rng.standard_normal((1, horizon, dim)).astype(np.float32)
                        )
                        normalized, _ = flow.sample_with_intermediates(context, noise)
                        record.update(guide(z, normalized, alpha))
                        success, post_steps = False, 0
                        while elapsed < cfg["env"]["max_steps"]:
                            for action in flow.unnormalize(normalized, context)[0][:h]:
                                obs, _, done, _ = env.step(action.tolist())
                                elapsed += 1
                                post_steps += 1
                                success = bool(env.check_success())
                                if success or done or elapsed >= cfg["env"]["max_steps"]:
                                    break
                            if success or done or elapsed >= cfg["env"]["max_steps"]:
                                break
                            context = flow.encode_context(
                                [observation_for_policy(obs, task.language)]
                            )
                            noise = torch.from_numpy(
                                rng.standard_normal((1, horizon, dim)).astype(np.float32)
                            )
                            normalized, _ = flow.sample_with_intermediates(context, noise)
                        record.update(success=success, steps=post_steps)
                    except RuntimeError as error:
                        record.update(success=None, error=str(error))
                    finally:
                        env.close()
                    record["seconds"] = time.monotonic() - start
                    records.append(record)
                    with (out / "episodes.jsonl").open("a") as f:
                        f.write(json.dumps(dict(record, tag=args.tag, run=args.run)) + "\n")
                    print(json.dumps(record), flush=True)

    ok = [r for r in records if r.get("success") is not None]
    summary = {
        "records": len(records),
        "usable": len(ok),
        "skipped": len(records) - len(ok),
        "max_state_error": max((r["state_error"] for r in ok), default=None),
        "per_alpha": {
            str(a): dict(n=len(rs), successes=sum(r["success"] for r in rs))
            for a in alphas
            if (rs := [r for r in ok if r["alpha"] == a])
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "status.txt").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

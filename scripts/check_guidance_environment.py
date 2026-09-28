"""Counterfactual clean-action Q guidance from reconstructed replay states.

Diagnostic only: frozen SFT with one-chunk or repeated clean-action guidance.
Results never enter the training buffer or evaluate the trained Q-VGM actor.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run", default="spatial_offline")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--name", default="guidance_environment_check_v2")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.005, 0.05])
    parser.add_argument("--intervention", choices=["once", "repeated"], default="once")
    parser.add_argument("--encoder-bf16", action="store_true")
    parser.add_argument("--state-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        parser.error("episodes-per-task must be positive")
    if not all(0 <= x < float("inf") for x in args.alphas) or 0.0 not in args.alphas:
        parser.error("alphas must be finite, nonnegative, and include zero")
    if not 0 <= args.state_fraction < 1:
        parser.error("state-fraction must be in [0, 1)")
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import buffer_signature, make_autoencoder, make_critic

    root = Path(cfg["paths"]["artifacts"]) / args.run
    if Path(args.name).name != args.name:
        parser.error("name must be a single directory name")
    out = root / args.name
    out.mkdir(exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(config=cfg, args=vars(args)), indent=2))
    (out / "status.txt").write_text("running\n")

    def mark_unfinished():
        if (out / "status.txt").read_text().strip() == "running":
            (out / "status.txt").write_text("failed\n")

    atexit.register(mark_unfinished)
    torch.manual_seed(cfg["runtime"]["seed"])
    buffer = ReplayBuffer(root / "buffer")
    ck = torch.load(root / "critic_full.pt", weights_only=True, map_location="cpu")
    if ck["buffer_signature"] != buffer_signature(buffer):
        raise ValueError("Provenance mismatch")
    critic = make_critic(cfg).to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    del ck
    ck = torch.load(root / "rl_token_full.pt", weights_only=True, mmap=True, map_location="cpu")
    if ck["buffer_signature"] != buffer_signature(buffer):
        raise ValueError("AE provenance mismatch")
    encoder = make_autoencoder(cfg["offline"]["rl_token"]).encoder
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ck["model"].items() if k.startswith("encoder.")}
    )
    encoder = encoder.to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    del ck
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    flow.enable_actor_training(cfg["offline"]["actor"].get("master_dtype", "float32"))
    flow.model.requires_grad_(False)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    h, d = cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
    horizon, dim = cfg["model"]["action_horizon"], cfg["model"]["action_dim"]
    records = []

    def guide(context, normalized, alpha):
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.encoder_bf16),
        ):
            prefix = context["prefix"][:, context["pad"][0]].float()
            z = encoder(prefix).flatten(1).float()
        improved, metrics = improve_actions(
            critic.mean,
            z,
            normalized[:, :h, :d],
            steps=cfg["offline"]["actor"]["ascent_steps"],
            alpha=alpha,
        )
        normalized[:, :h, :d] = improved
        return metrics

    for e, ep in enumerate(buffer.episodes):
        if ep["episode"] >= args.episodes_per_task:
            continue
        if args.task_ids is not None and ep["task_id"] not in args.task_ids:
            continue
        task = suite.get_task(ep["task_id"])
        state_index = int(len(ep["transitions"]) * args.state_fraction)
        original = buffer.observation(e, state_index)
        initial = suite.get_task_init_states(ep["task_id"])[ep["episode"]]
        env = None
        common_start = None
        common_sim_state = None
        try:
            for alpha in sorted(set(args.alphas)):
                start = time.monotonic()
                np.random.seed(ep["seed"])
                torch.manual_seed(ep["seed"])
                if env is not None:
                    env.close()
                env = make_env(cfg, task, ep["seed"])
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
                        "state": torch.as_tensor(transformed["state"], device=flow.device)[None]
                    }
                    normalized = torch.zeros(1, horizon, dim, device=flow.device)
                    transition = ep["transitions"][i]
                    normalized[0, :h, :d] = transition["action"].to(flow.device)
                    executed = flow.unnormalize(normalized, context)[0]
                    for action in executed[: transition["steps"]]:
                        obs, _, done, _ = env.step(action.tolist())
                        elapsed += 1
                        if done or env.check_success():
                            raise RuntimeError(
                                "Replay reached a terminal before the selected state"
                            )
                reconstructed = observation_for_policy(obs, task.language)
                original_state_error = float(
                    np.max(
                        np.abs(reconstructed["observation/state"] - original["observation/state"])
                    )
                )
                if common_start is None:
                    common_start = reconstructed
                    common_sim_state = env.get_sim_state().copy()
                sim_state_error = float(np.max(np.abs(env.get_sim_state() - common_sim_state)))
                state_error = float(
                    np.max(
                        np.abs(
                            reconstructed["observation/state"] - common_start["observation/state"]
                        )
                    )
                )
                image_error = max(
                    float(
                        np.mean(
                            np.abs(reconstructed[k].astype(float) - common_start[k].astype(float))
                        )
                    )
                    for k in ("observation/image", "observation/wrist_image")
                )
                if state_error > 1e-3 or image_error > 1.0 or sim_state_error > 1e-6:
                    raise RuntimeError(
                        f"Counterfactual starts differ: state={state_error}, image={image_error}, sim={sim_state_error}"
                    )
                context = flow.encode_context([reconstructed])
                rng = np.random.default_rng(
                    cfg["runtime"]["seed"] + ep["task_id"] * 10000 + ep["episode"]
                )
                noise = torch.from_numpy(rng.standard_normal((1, horizon, dim)).astype(np.float32))
                normalized, _ = flow.sample_with_intermediates(context, noise)
                metrics = guide(context, normalized, alpha)
                chunk_metrics = [metrics]
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
                    context = flow.encode_context([observation_for_policy(obs, task.language)])
                    noise = torch.from_numpy(
                        rng.standard_normal((1, horizon, dim)).astype(np.float32)
                    )
                    normalized, _ = flow.sample_with_intermediates(context, noise)
                    if args.intervention == "repeated":
                        chunk_metrics.append(guide(context, normalized, alpha))
                record = dict(
                    task=ep["task_id"],
                    episode=ep["episode"],
                    state_index=state_index,
                    alpha=alpha,
                    success=success,
                    steps=post_steps,
                    discounted_return=cfg["offline"]["gamma"] ** (post_steps - 1)
                    if success
                    else 0.0,
                    state_error=state_error,
                    image_error=image_error,
                    original_state_error=original_state_error,
                    sim_state_error=sim_state_error,
                    seconds=time.monotonic() - start,
                    intervention=args.intervention,
                    guided_chunks=len(chunk_metrics),
                    chunk_metrics=chunk_metrics,
                    **metrics,
                )
                records.append(record)
                with (out / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        finally:
            if env is not None:
                env.close()
    summary = {
        str(alpha): dict(episodes=len(rs), successes=sum(r["success"] for r in rs))
        for alpha in sorted(set(args.alphas))
        if (rs := [r for r in records if r["alpha"] == alpha])
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "status.txt").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

"""Paired task-1 videos, action-coordinate checks, and fixed-input alignment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/libero_spatial_checkpoint.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from libero.libero import benchmark
    from openpi import transforms
    from openpi.shared.normalize import load

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import buffer_signature, make_critic

    root = Path(cfg["paths"]["artifacts"])
    run = root / "spatial_paper_checkpoint"
    out = run / "task1_diagnosis"
    out.mkdir(exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    flows = {}
    paths = {
        "zero": root / "spatial_offline_zero_guidance/actor_full.pt",
        "rl": run / "actor_full.pt",
    }
    for label, path in paths.items():
        policy = load_sft_policy(cfg)
        ck = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if ck["config"]["model"] != cfg["model"]:
            raise ValueError("Model config mismatch")
        flow = Pi05Flow(policy, cfg)
        flow.enable_actor_training(ck["settings"].get("master_dtype", "float32"))
        expected = {n for n, p in flow.model.named_parameters() if p.requires_grad}
        if set(ck["actor"]) != expected:
            raise ValueError("Incomplete actor checkpoint")
        flow.model.load_state_dict(ck["actor"], strict=False)
        flow.model.requires_grad_(False).eval()
        flows[label] = flow
        del ck
    print("Models ready", flush=True)
    stats = load(Path(cfg["paths"]["norm_stats"]).parent)
    normalizer = transforms.Normalize(stats, use_quantiles=True)
    scale = (np.asarray(stats["actions"].q99[:7]) - np.asarray(stats["actions"].q01[:7]) + 1e-6) / 2
    buffer = ReplayBuffer(run / "buffer")
    features = torch.load(run / "features_full.pt", weights_only=True)
    ck = torch.load(run / "critic_full.pt", weights_only=True, map_location="cpu")
    if any(x["buffer_signature"] != buffer_signature(buffer) for x in (features, ck)):
        raise ValueError("Feature/critic provenance mismatch")
    critic = make_critic(cfg).to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    del ck
    e = next(i for i, f in enumerate(buffer.files) if f.name == "task01_episode000.pt")
    n = len(buffer.episodes[e]["transitions"])
    fixed = []
    rng = np.random.default_rng(7)
    for i in sorted({0, n // 2, n - 1}):
        context = flows["rl"].encode_context([buffer.observation(e, i)])
        noise = torch.from_numpy(rng.standard_normal((1, 10, 32)).astype(np.float32))
        _, trajectory = flows["rl"].sample_with_intermediates(context, noise)
        z = features["z"][e][i : i + 1].to(cfg["runtime"]["device"])
        for tau, x in trajectory[-cfg["offline"]["actor"]["late_steps"] :]:
            with torch.no_grad():
                ref = flows["zero"].predict_velocity(x, tau, context).float()
                prediction = flows["rl"].predict_velocity(x, tau, context).float()
            endpoint = (x + (1 - tau) * ref)[:, :5, :7]
            improved, metrics = improve_actions(critic.mean, z, endpoint, 3, 0.05)
            desired = (improved - endpoint) / (1 - tau)
            actual = (prediction - ref)[:, :5, :7]
            residual = actual - desired
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(1), desired.flatten(1)
            ).item()
            fixed.append(
                dict(
                    state_index=i,
                    tau=tau,
                    target_norm=desired.norm().item(),
                    actual_norm=actual.norm().item(),
                    cosine=cosine,
                    loss=residual.square().sum().item(),
                    actual_per_dim=actual.mean((0, 1)).tolist(),
                    target_per_dim=desired.mean((0, 1)).tolist(),
                    **metrics,
                )
            )
    (out / "alignment.json").write_text(json.dumps(fixed, indent=2))
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    task = suite.get_task(1)
    initial_states = suite.get_task_init_states(1)
    if args.episodes > len(initial_states):
        raise ValueError("Too many episodes")
    summary, coordinates = [], None
    for episode in range(args.episodes):
        episode_seed = cfg["runtime"]["seed"] + 10000 + episode
        common_start = None
        for label, flow in flows.items():
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env = make_env(cfg, task, episode_seed)
            writer = imageio.get_writer(out / f"{label}_episode{episode}.mp4", fps=20)
            try:
                env.seed(episode_seed)
                env.reset()
                obs = env.set_init_state(initial_states[episode])
                for _ in range(cfg["env"]["settle_steps"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                if common_start is None:
                    common_start = env.get_sim_state().copy()
                start_error = float(np.max(np.abs(env.get_sim_state() - common_start)))
                if start_error > 1e-6:
                    raise RuntimeError(f"Unequal paired initial simulator states: {start_error}")
                rng = np.random.default_rng(episode_seed)
                steps, chunk, success = 0, 0, False
                while steps < cfg["env"]["max_steps"]:
                    inp = observation_for_policy(obs, task.language)
                    context = flow.encode_context([inp])
                    noise = torch.from_numpy(rng.standard_normal((1, 10, 32)).astype(np.float32))
                    own, _ = flow.sample_with_intermediates(context, noise)
                    other = flows["rl" if label == "zero" else "zero"]
                    counterfactual, _ = other.sample_with_intermediates(context, noise)
                    actions = flow.unnormalize(own, context)[0][:5]
                    alternative = other.unnormalize(counterfactual, context)[0][:5]
                    if coordinates is None:
                        back = normalizer({"actions": actions})["actions"]
                        analytical = (own[0, :5, :7].cpu().numpy() + 1) * scale + np.asarray(
                            stats["actions"].q01[:7]
                        )
                        jacobian = []
                        for d in range(7):
                            changed = own.clone()
                            changed[:, :, d] += 0.001
                            jacobian.append(
                                ((flow.unnormalize(changed, context)[0][:5] - actions) / 0.001)
                                .mean(0)
                                .tolist()
                            )
                        coordinates = dict(
                            scale=scale.tolist(),
                            measured_jacobian=jacobian,
                            roundtrip_max_abs=float(
                                np.max(np.abs(back - own[0, :5, :7].cpu().numpy()))
                            ),
                            formula_max_abs=float(np.max(np.abs(analytical - actions))),
                        )
                        (out / "coordinates.json").write_text(json.dumps(coordinates, indent=2))
                    record = dict(
                        episode=episode,
                        policy=label,
                        chunk=chunk,
                        start_step=steps,
                        normalized_actions=own[0, :5, :7].cpu().tolist(),
                        environment_actions=actions.tolist(),
                        other_policy_actions_same_state=alternative.tolist(),
                        same_state_action_difference_l2=float(
                            np.linalg.norm(actions - alternative)
                        ),
                        state_before=inp["observation/state"].tolist(),
                    )
                    for action in actions:
                        writer.append_data(
                            np.concatenate(
                                [inp["observation/image"], inp["observation/wrist_image"]], axis=1
                            )
                        )
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        success = bool(env.check_success())
                        inp = observation_for_policy(obs, task.language)
                        if success or done or steps >= cfg["env"]["max_steps"]:
                            break
                    record.update(
                        end_step=steps,
                        success=success,
                        state_after=inp["observation/state"].tolist(),
                    )
                    with (out / "chunks.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    if success or done:
                        break
                    chunk += 1
                result = dict(
                    episode=episode,
                    policy=label,
                    success=success,
                    steps=steps,
                    initial_sim_state_max_difference=start_error,
                )
                summary.append(result)
                print(json.dumps(result), flush=True)
            finally:
                writer.close()
                env.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

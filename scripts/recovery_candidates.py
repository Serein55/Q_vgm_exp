"""Matched single-chunk interventions with identical frozen-SFT continuations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/idea_recovery.yaml")
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--lane", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import buffer_signature, make_autoencoder, make_critic, save_checkpoint

    settings = cfg["recovery"]
    if not 0 <= args.lane < args.lanes:
        ap.error("invalid lane")
    if not 0 <= args.worker < settings["workers"]:
        ap.error("invalid worker")
    root = Path(cfg["paths"]["artifacts"]) / settings["run"]
    root.mkdir(exist_ok=True)
    out = root / f"worker{args.worker}"
    out.mkdir(exist_ok=True)
    snapshot = json.dumps(cfg, sort_keys=True, indent=2)
    if (out / "config.json").exists() and (out / "config.json").read_text() != snapshot:
        raise ValueError("Resume configuration differs")
    if not (out / "config.json").exists():
        (out / "config.json").write_text(snapshot)
    status_path = out / ("status.txt" if args.lanes == 1 else f"status_lane{args.lane}.txt")
    episode_log = out / ("episodes.jsonl" if args.lanes == 1 else f"episodes_lane{args.lane}.jsonl")
    status_path.write_text("running\n")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    source = Path(cfg["paths"]["artifacts"]) / settings["source_run"]
    replay = ReplayBuffer(source / "buffer")
    signature = buffer_signature(replay)
    ck = torch.load(source / "critic_full.pt", weights_only=True, map_location="cpu")
    assert ck["buffer_signature"] == signature
    critic = make_critic(cfg).to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    del ck
    ck = torch.load(source / "rl_token_full.pt", weights_only=True, mmap=True, map_location="cpu")
    assert ck["buffer_signature"] == signature
    encoder = make_autoencoder(cfg["offline"]["rl_token"]).encoder
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ck["model"].items() if k.startswith("encoder.")}
    )
    encoder.to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    del ck
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    flow.enable_actor_training(cfg["offline"]["actor"]["master_dtype"])
    flow.model.requires_grad_(False)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    h, d = cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
    H, D = cfg["model"]["action_horizon"], cfg["model"]["action_dim"]
    device = flow.device
    radii = torch.tensor(settings["action_radii"], device=device)

    # Infer affine output scale through the actual output transform.
    def output_scale(context):
        a = torch.zeros(1, H, D, device=device)
        b = a.clone()
        b[:, :, :d] = 1
        return torch.as_tensor(
            flow.unnormalize(b, context)[0, 0] - flow.unnormalize(a, context)[0, 0],
            device=device,
            dtype=torch.float32,
        )

    jobs = [
        (e, fraction)
        for e, ep in enumerate(replay.episodes)
        if ep["task_id"] in settings["task_ids"] and ep["episode"] < settings["episodes"]
        for fraction in settings["fractions"]
    ]
    for job, (e, fraction) in enumerate(jobs):
        if job % settings["workers"] != args.worker:
            continue
        if (job // settings["workers"]) % args.lanes != args.lane:
            continue
        ep = replay.episodes[e]
        index = int(len(ep["transitions"]) * fraction)
        key = f"task{ep['task_id']:02d}_episode{ep['episode']:03d}_state{index:03d}"
        dest = out / f"{key}.pt"
        if dest.exists():
            continue
        task = suite.get_task(ep["task_id"])
        initial = suite.get_task_init_states(ep["task_id"])[ep["episode"]]
        env = None
        common = None
        candidate_seed = (
            cfg["runtime"]["seed"] + ep["task_id"] * 10000 + ep["episode"] * 100 + index
        )
        records = []
        names = ["base", "resample", "perturb", "q_raw", "q_trust"]
        try:
            for repeat in range(settings["repeats"]):
                for cidx, name in enumerate(names):
                    if env is not None:
                        env.close()
                    np.random.seed(ep["seed"])
                    torch.manual_seed(ep["seed"])
                    env = make_env(cfg, task, ep["seed"])
                    env.seed(ep["seed"])
                    env.reset()
                    obs = env.set_init_state(initial)
                    for _ in range(cfg["env"]["settle_steps"]):
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    elapsed = 0
                    for i in range(index):
                        inp = replay.observation(e, i)
                        state = torch.as_tensor(
                            policy._input_transform(dict(inp))["state"], device=device
                        )[None]
                        normalized = torch.zeros(1, H, D, device=device)
                        t = ep["transitions"][i]
                        normalized[0, :h, :d] = t["action"].to(device)
                        for action in flow.unnormalize(normalized, {"state": state})[
                            0, : t["steps"]
                        ]:
                            obs, _, done, _ = env.step(action.tolist())
                            elapsed += 1
                            if done or env.check_success():
                                raise RuntimeError("Prefix terminated before intervention")
                    inp = observation_for_policy(obs, task.language)
                    if common is None:
                        common = env.get_sim_state().copy()
                        common_obs = inp
                        context = flow.encode_context([inp])
                        rng = np.random.default_rng(candidate_seed)

                        def noise():
                            return torch.from_numpy(
                                rng.standard_normal((1, H, D)).astype(np.float32)
                            )

                        base, _ = flow.sample_with_intermediates(context, noise())
                        alternative, _ = flow.sample_with_intermediates(context, noise())
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            z = (
                                encoder(context["prefix"][:, context["pad"][0]].float())
                                .flatten(1)
                                .float()
                            )
                        scale = output_scale(context)
                        delta = torch.from_numpy(rng.standard_normal((h, d)).astype(np.float32)).to(
                            device
                        )
                        delta = delta / delta.norm().clamp_min(1e-8) * settings["trust_radius"]
                        perturbed = base.clone()
                        perturbed[0, :h, :d] += delta * radii / scale
                        raw, _ = improve_actions(
                            critic.mean, z, base[:, :h, :d], steps=3, alpha=0.05
                        )
                        raw_full = base.clone()
                        raw_full[:, :h, :d] = raw
                        # Project into a rotation-sensitive ellipsoid in controller coordinates.
                        scaled_delta = (raw - base[:, :h, :d]) * scale / radii
                        scaled_delta *= min(
                            1.0,
                            settings["trust_radius"] / float(scaled_delta.norm().clamp_min(1e-8)),
                        )
                        trusted = base.clone()
                        trusted[:, :h, :d] += scaled_delta * radii / scale
                        candidates = torch.cat([base, alternative, perturbed, raw_full, trusted])
                        with torch.no_grad():
                            scores = critic.mean(
                                z.expand(len(names), -1), candidates[:, :h, :d]
                            ).cpu()
                        actions_env = np.stack(
                            [flow.unnormalize(x[None], context)[0, :h] for x in candidates]
                        )
                        saved_observation = {
                            k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                            for k, v in inp.items()
                        }
                    sim_error = float(np.max(np.abs(env.get_sim_state() - common)))
                    image_error = max(
                        float(np.abs(inp[k].astype(float) - common_obs[k].astype(float)).mean())
                        for k in ["observation/image", "observation/wrist_image"]
                    )
                    state_error = float(
                        np.max(np.abs(inp["observation/state"] - common_obs["observation/state"]))
                    )
                    if sim_error > 1e-6 or state_error > 1e-3 or image_error > 1:
                        raise RuntimeError(
                            f"Unequal starts: {sim_error}, {state_error}, {image_error}"
                        )
                    # Candidate and prefix fixed; only continuation noise varies across repeats.
                    rng_follow = np.random.default_rng(candidate_seed + 1000000 + repeat)
                    normalized = candidates[cidx : cidx + 1].clone()
                    context = {
                        "state": torch.as_tensor(
                            policy._input_transform(dict(inp))["state"], device=device
                        )[None]
                    }
                    n = 0
                    success = False
                    while elapsed < cfg["env"]["max_steps"]:
                        for action in flow.unnormalize(normalized, context)[0, :h]:
                            obs, _, done, _ = env.step(action.tolist())
                            elapsed += 1
                            n += 1
                            success = bool(env.check_success())
                            if success or done or elapsed >= cfg["env"]["max_steps"]:
                                break
                        if success or done or elapsed >= cfg["env"]["max_steps"]:
                            break
                        context = flow.encode_context([observation_for_policy(obs, task.language)])
                        eps = torch.from_numpy(
                            rng_follow.standard_normal((1, H, D)).astype(np.float32)
                        )
                        normalized, _ = flow.sample_with_intermediates(context, eps)
                    row = dict(
                        candidate=name,
                        repeat=repeat,
                        success=success,
                        steps=n,
                        return_value=cfg["offline"]["gamma"] ** (n - 1) if success else 0.0,
                        sim_error=sim_error,
                        state_error=state_error,
                        image_error=image_error,
                    )
                    records.append(row)
                    with episode_log.open("a") as f:
                        f.write(json.dumps(dict(key=key, **row)) + "\n")
                    print(json.dumps(dict(key=key, **row)), flush=True)
            save_checkpoint(
                dest,
                dict(
                    key=key,
                    task=ep["task_id"],
                    episode=ep["episode"],
                    state_index=index,
                    fraction=fraction,
                    buffer_signature=signature,
                    z=z.cpu()[0],
                    observation=saved_observation,
                    candidates=candidates.cpu(),
                    candidate_names=names,
                    old_q=scores,
                    environment_actions=torch.from_numpy(actions_env),
                    records=records,
                ),
            )
        finally:
            if env is not None:
                env.close()
    status_path.write_text("complete\n")


if __name__ == "__main__":
    main()

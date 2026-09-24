"""Compare samplers and trained actions on saved observations, without environment rollouts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run", default="spatial_offline")
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import numpy as np
    import torch

    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import buffer_signature, make_critic

    torch.manual_seed(cfg["runtime"]["seed"])
    root = Path(cfg["paths"]["artifacts"]) / args.run
    destination = root / f"actor_diagnostics_{args.tag}.json"
    if destination.exists():
        raise FileExistsError(destination)
    buffer = ReplayBuffer(root / "buffer")
    samples = []
    for e, path in enumerate(buffer.files):
        if path.stem.endswith("episode000"):
            length = len(buffer.episodes[e]["transitions"])
            samples.extend((e, i) for i in sorted({0, length // 2, length - 1}))
    if not samples:
        raise ValueError("No episode000 shards for diagnosis")
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    rng = np.random.default_rng(cfg["runtime"]["seed"])
    noises = [
        rng.standard_normal((cfg["model"]["action_horizon"], cfg["model"]["action_dim"])).astype(
            np.float32
        )
        for _ in samples
    ]
    observations = [buffer.observation(e, i) for e, i in samples]
    records = [dict(shard=buffer.files[e].name, state_index=i) for e, i in samples]
    actions = {}
    environment_actions = {}
    for stage in ("native_sft", "flow_sft", "amp_sft", "trained_actor"):
        if stage == "amp_sft":
            flow.enable_actor_training()
            expected = {n for n, p in flow.model.named_parameters() if p.requires_grad}
            flow.model.requires_grad_(False)
        if stage == "trained_actor":
            ck = torch.load(
                root / f"actor_{args.tag}.pt", weights_only=True, mmap=True, map_location="cpu"
            )
            if (
                ck["buffer_signature"] != buffer_signature(buffer)
                or ck["config"]["model"] != cfg["model"]
            ):
                raise ValueError("Actor provenance mismatch")
            if set(ck["actor"]) != expected:
                raise ValueError("Incomplete actor checkpoint")
            flow.model.load_state_dict(ck["actor"], strict=False)
            del ck
        values = []
        env_values = []
        for observation, noise in zip(observations, noises, strict=True):
            if stage == "native_sft":
                action = policy.infer(observation, noise=noise)["actions"]
            else:
                context = flow.encode_context([observation])
                normalized, _ = flow.sample_with_intermediates(
                    context, torch.from_numpy(noise)[None]
                )
                action = flow.unnormalize(normalized, context)[0]
                if stage in ("amp_sft", "trained_actor"):
                    values.append(normalized[0].cpu())
            if stage not in ("amp_sft", "trained_actor"):
                values.append(torch.from_numpy(action))
            env_values.append(torch.from_numpy(action))
        actions[stage] = torch.stack(values)
        environment_actions[stage] = torch.stack(env_values)
        print(json.dumps(dict(stage=stage, observations=len(values))), flush=True)

    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    ck = torch.load(root / f"critic_{args.tag}.pt", weights_only=True, map_location="cpu")
    signature = buffer_signature(buffer)
    if any(x["buffer_signature"] != signature for x in (features, ck)):
        raise ValueError("Critic/features provenance mismatch")
    critic = make_critic(cfg).to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    h, d = cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
    baseline = actions["amp_sft"][:, :h, :d]
    trained = actions["trained_actor"][:, :h, :d]
    with torch.no_grad():
        z = torch.stack([features["z"][e][i] for e, i in samples]).to(flow.device)
        qb = critic.mean(z, baseline.to(flow.device)).cpu()
        qt = critic.mean(z, trained.to(flow.device)).cpu()
    parity = (actions["native_sft"] - actions["flow_sft"])[:, :h, :d]
    amp_delta = (environment_actions["amp_sft"] - environment_actions["flow_sft"])[:, :h, :d]
    delta = trained - baseline
    for j, record in enumerate(records):
        record.update(
            native_flow_max_abs=float(parity[j].abs().max()),
            amp_flow_max_abs=float(amp_delta[j].abs().max()),
            normalized_action_delta_l2=float(delta[j].norm()),
            q_sft=float(qb[j]),
            q_actor=float(qt[j]),
            gripper_sign_change=float(
                ((trained[j, :, -1] > 0) != (baseline[j, :, -1] > 0)).float().mean()
            ),
        )
    report = dict(
        config=cfg,
        buffer_signature=signature,
        samples=records,
        native_flow_max_abs=float(parity.abs().max()),
        amp_flow_max_abs=float(amp_delta.abs().max()),
        normalized_action_delta_l2_mean=float(delta.flatten(1).norm(dim=1).mean()),
        normalized_action_delta_abs_per_dim=delta.abs().mean((0, 1)).tolist(),
        q_sft_mean=float(qb.mean()),
        q_actor_mean=float(qt.mean()),
        q_gain_mean=float((qt - qb).mean()),
        note="Saved training states, one fixed noise per state; Q is not measured return.",
    )
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("config", "samples")}), flush=True
    )


if __name__ == "__main__":
    main()

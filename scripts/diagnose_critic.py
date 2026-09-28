"""Audit ensemble action gradients on saved transitions; no new robot rollouts."""

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

    import torch

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.training import buffer_signature, make_critic, proprio_features

    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    root = Path(cfg["paths"]["artifacts"]) / args.run
    dest = root / f"critic_gradient_audit_{args.tag}.json"
    if dest.exists():
        raise FileExistsError(dest)
    buffer = ReplayBuffer(root / "buffer")
    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    checkpoint = torch.load(root / f"critic_{args.tag}.pt", weights_only=True, map_location="cpu")
    signature = buffer_signature(buffer)
    if any(c["buffer_signature"] != signature for c in (features, checkpoint)):
        raise ValueError("Buffer provenance mismatch")
    stats = checkpoint.get("proprio_stats")
    extra = 0 if stats is None else int(stats["mean"].numel())
    critic = make_critic(cfg, extra).to(cfg["runtime"]["device"]).eval().requires_grad_(False)
    critic.load_state_dict(checkpoint["critic"])
    selected = torch.randperm(len(buffer.transitions))[:512].tolist()
    samples = [buffer.transitions[i] for i in selected]
    gradients, values, gains = [], [], []
    for offset in range(0, len(samples), 32):
        batch = samples[offset : offset + 32]
        z = torch.stack([features["z"][e][i] for e, i in batch]).to(cfg["runtime"]["device"])
        if stats is not None:
            z = torch.cat([z, proprio_features(buffer, batch, stats).to(z.device)], -1)
        actions = torch.stack(
            [buffer.episodes[e]["transitions"][i]["action"] for e, i in batch]
        ).to(z.device)
        per_head = []
        for head in critic.heads:
            a = actions.detach().requires_grad_(True)
            per_head.append(torch.autograd.grad(head(z, a).sum(), a)[0].flatten(1))
        gradients.append(torch.stack(per_head, 1).cpu())
        with torch.no_grad():
            values.append(critic(z, actions).cpu())
        improved, _ = improve_actions(critic.mean, z, actions, steps=3, alpha=0.05)
        with torch.no_grad():
            gains.append((critic(z, improved) - critic(z, actions)).cpu())
    gradients, values, gains = map(torch.cat, (gradients, values, gains))
    unit = torch.nn.functional.normalize(gradients, dim=-1)
    h = gradients.shape[1]
    pairs = torch.triu_indices(h, h, offset=1)
    cosine = torch.einsum("bhd,bjd->bhj", unit, unit)[:, pairs[0], pairs[1]]
    mean_gradient = gradients.mean(1)
    normalized = torch.nn.functional.normalize(mean_gradient, dim=-1)
    # Length of the average unit vector: 1 means identical directions across states.
    concentration = normalized.mean(0).norm()
    dims = mean_gradient.reshape(len(samples), cfg["env"]["action_chunk"], cfg["env"]["action_dim"])
    energy = dims.square().sum((0, 1))
    report = dict(
        buffer_signature=signature,
        sampled_transitions=len(samples),
        head_gradient_cosine_mean=float(cosine.mean()),
        head_gradient_cosine_negative_fraction=float((cosine < 0).float().mean()),
        mean_gradient_direction_concentration=float(concentration),
        action_dimension_gradient_energy_fraction=(energy / energy.sum()).tolist(),
        q_head_std_mean=float(values.std(-1).mean()),
        improved_action_head_gain_mean=gains.mean(0).tolist(),
        improved_action_negative_head_gain_fraction=float((gains < 0).float().mean()),
        task_gradient_direction_concentration={},
        note="In-sample diagnostic only. Ensemble agreement/Q increase do not establish correct gradients or higher environment returns.",
    )
    for task in range(10):
        mask = torch.tensor(
            [buffer.files[e].name.startswith(f"task{task:02d}_") for e, _ in samples]
        )
        if mask.any():
            report["task_gradient_direction_concentration"][str(task)] = float(
                normalized[mask].mean(0).norm()
            )
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(dest)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

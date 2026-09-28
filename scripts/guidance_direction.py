"""How much of the Q-ascent displacement is a state-independent constant direction?

The gradient audit already reports direction concentration of the raw gradient. This
measures the displacement actually applied to the action chunk after keep-best ascent,
which is what the velocity-matching target distils into the actor.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/libero_spatial.yaml")
    p.add_argument("--run", default="spatial_v5_h10")
    p.add_argument("--tag", default="full")
    p.add_argument("--samples", type=int, default=1024)
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import torch

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.training import buffer_signature, make_critic, proprio_features

    device = cfg["runtime"]["device"]
    root = Path(cfg["paths"]["artifacts"]) / args.run
    dest = root / f"guidance_direction_{args.tag}.json"
    if dest.exists():
        raise FileExistsError(dest)
    buffer = ReplayBuffer(root / "buffer")
    signature = buffer_signature(buffer)
    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    ck = torch.load(root / f"critic_{args.tag}.pt", weights_only=True, map_location="cpu")
    if features["buffer_signature"] != signature or ck["buffer_signature"] != signature:
        raise ValueError("Provenance mismatch")
    stats = ck.get("proprio_stats")
    extra = 0 if stats is None else int(stats["mean"].numel())
    critic = make_critic(cfg, extra).to(device).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])

    torch.manual_seed(cfg["runtime"]["seed"])
    ids = torch.randperm(len(buffer.transitions))[: args.samples].tolist()
    pairs = [buffer.transitions[i] for i in ids]
    ascent = cfg["offline"]["actor"]
    deltas = []
    for offset in range(0, len(pairs), 64):
        batch = pairs[offset : offset + 64]
        z = torch.stack([features["z"][e][i] for e, i in batch]).to(device)
        if stats is not None:
            z = torch.cat([z, proprio_features(buffer, batch, stats).to(z.device)], -1)
        a = torch.stack([buffer.episodes[e]["transitions"][i]["action"] for e, i in batch]).to(z)
        improved, _ = improve_actions(
            critic.mean,
            z,
            a,
            steps=ascent["ascent_steps"],
            alpha=ascent["ascent_step_size"],
        )
        deltas.append((improved - a).flatten(1).cpu())
    delta = torch.cat(deltas)
    mean_direction = torch.nn.functional.normalize(delta.mean(0), dim=0)
    norm = delta.norm(dim=-1)
    projection = delta @ mean_direction
    # R^2 of the displacement explained by one global direction.
    explained = (projection.square() / norm.square().clamp_min(1e-12)).mean()
    cosine = (projection / norm.clamp_min(1e-12)).mean()
    dims = delta.reshape(len(delta), cfg["env"]["action_chunk"], cfg["env"]["action_dim"])
    energy = dims.square().sum((0, 1))
    report = dict(
        run=args.run,
        tag=args.tag,
        samples=len(delta),
        displacement_norm_mean=float(norm.mean()),
        global_direction_explained_variance=float(explained),
        cosine_to_global_direction_mean=float(cosine),
        cosine_negative_fraction=float((projection < 0).float().mean()),
        dimension_energy_fraction=(energy / energy.sum()).tolist(),
        global_direction=mean_direction.tolist(),
        note=(
            "explained_variance near 1 means Q ascent applies an almost state-independent "
            "offset, so velocity matching distils a constant action bias rather than a "
            "state-conditional value improvement."
        ),
    )
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(dest)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("global_direction", "note")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

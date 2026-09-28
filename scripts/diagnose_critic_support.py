"""Read-only finite-difference and marginal action-support checks on a fixed critic."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/libero_spatial_checkpoint.yaml")
    parser.add_argument("--run", default="spatial_paper_checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import torch

    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.training import buffer_signature, make_critic

    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    root = Path(cfg["paths"]["artifacts"]) / args.run
    dest = root / "critic_support_audit.json"
    if dest.exists():
        raise FileExistsError(dest)
    buffer = ReplayBuffer(root / "buffer")
    features = torch.load(root / "features_full.pt", weights_only=True)
    ck = torch.load(root / "critic_full.pt", weights_only=True, map_location="cpu")
    signature = buffer_signature(buffer)
    if any(x["buffer_signature"] != signature for x in (features, ck)):
        raise ValueError("Buffer provenance mismatch")
    device = cfg["runtime"]["device"]
    critic = make_critic(cfg).to(device).eval().requires_grad_(False)
    critic.load_state_dict(ck["critic"])
    report = {"buffer_signature": signature, "tasks": {}}
    actor = cfg["offline"]["actor"]
    # Fixed diagnostic constants, not training hyperparameters.
    report["settings"] = dict(
        seed=cfg["runtime"]["seed"],
        max_samples_per_task=128,
        quantiles=[0.01, 0.99],
        ascent_steps=actor["ascent_steps"],
        ascent_step_size=actor["ascent_step_size"],
    )
    for task in range(10):
        pairs = [
            (e, i)
            for e, i in buffer.transitions
            if buffer.files[e].name.startswith(f"task{task:02d}_")
        ]
        actions = torch.stack(
            [buffer.episodes[e]["transitions"][i]["action"] for e, i in pairs]
        ).to(device)
        population = actions.flatten(0, 1)
        low, high = torch.quantile(population, torch.tensor([0.01, 0.99], device=device), dim=0)
        std = population.std(0)
        ids = torch.randperm(len(pairs))[:128].tolist()
        z = torch.stack([features["z"][pairs[k][0]][pairs[k][1]] for k in ids]).to(device)
        a = actions[ids]
        improved, metrics = improve_actions(
            critic.mean, z, a, steps=actor["ascent_steps"], alpha=actor["ascent_step_size"]
        )
        delta = improved - a

        def outside(x):
            return ((x < low) | (x > high)).float().mean((0, 1)).tolist()

        report["tasks"][str(task)] = dict(
            transitions=len(pairs),
            sampled_transitions=len(ids),
            normalized_action_std=std.tolist(),
            mean_guidance_per_dimension=delta.mean((0, 1)).tolist(),
            rms_guidance_over_marginal_std=(
                delta.square().mean((0, 1)).sqrt() / std.clamp_min(1e-8)
            ).tolist(),
            outside_1_99_percentile_before=outside(a),
            outside_1_99_percentile_after=outside(improved),
            **metrics,
        )
    # Double precision limits cancellation when checking the derivative itself.
    critic.double()
    a = a[:16].double().requires_grad_(True)
    z = z[:16].double()
    grad = torch.autograd.grad(critic.mean(z, a).sum(), a)[0]
    direction = torch.nn.functional.normalize(grad.flatten(1), dim=1).reshape_as(a)
    eps = 1e-4
    with torch.no_grad():
        finite_difference = (
            critic.mean(z, a + eps * direction) - critic.mean(z, a - eps * direction)
        ) / (2 * eps)
    analytic = (grad * direction).sum((1, 2))
    report["finite_difference"] = dict(
        task=9,
        samples=len(a),
        dtype="float64",
        epsilon=eps,
        max_absolute_error=float((finite_difference - analytic).abs().max()),
        max_relative_error=float(
            ((finite_difference - analytic).abs() / analytic.abs().clamp_min(1e-12)).max()
        ),
    )
    report["limitations"] = (
        "Marginal per-task ranges pooled over chunk positions are not conditional support. "
        "These checks use recorded actions, not the actor's reference endpoints. "
        "Finite differences validate the learned model derivative, not its true-return direction."
    )
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(dest, flush=True)


if __name__ == "__main__":
    main()

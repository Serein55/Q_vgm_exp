"""Fit an offline IQL ensemble and measure clean-action gradient sensitivity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.training import buffer_signature, log, make_critic, save_checkpoint, stage_args


def main():
    args, cfg, settings, root = stage_args("critic")
    import time

    import torch

    from qvgm.algorithms.iql import IQL
    from qvgm.algorithms.q_guidance import improve_actions
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.models.critic import Value

    buffer = ReplayBuffer(root / "buffer")
    signature = buffer_signature(buffer)
    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    if features["buffer_signature"] != signature:
        raise ValueError("Stale feature cache")
    device = cfg["runtime"]["device"]
    z, nz, actions, rewards, steps, done, success = [], [], [], [], [], [], []
    for e, i in buffer.transitions:
        t = buffer.episodes[e]["transitions"][i]
        z.append(features["z"][e][i])
        nz.append(features["z"][e][i + 1])
        actions.append(t["action"])
        rewards.append(t["reward"])
        steps.append(t["steps"])
        done.append(t["terminated"])
        success.append(buffer.episodes[e]["success"])
    data = dict(
        z=torch.stack(z),
        next_z=torch.stack(nz),
        action=torch.stack(actions),
        reward=torch.tensor(rewards),
        steps=torch.tensor(steps),
        terminated=torch.tensor(done),
        success=torch.tensor(success),
    )
    data = {k: v.to(device) for k, v in data.items()}
    critic = make_critic(cfg).to(device)
    value = Value(cfg["offline"]["rl_token"]["dim"], settings["widths"]).to(device)
    trainer = IQL(critic, value, settings, cfg["offline"]["gamma"])
    dest = root / f"critic_{args.tag}.pt"
    start_step = 0
    if dest.exists():
        if not args.resume:
            raise FileExistsError(dest)
        ck = torch.load(dest, weights_only=True, map_location=device)
        if ck["buffer_signature"] != signature or ck["settings"] != settings:
            raise ValueError("Resume mismatch")
        critic.load_state_dict(ck["critic"])
        value.load_state_dict(ck["value"])
        trainer.target.load_state_dict(ck["target"])
        trainer.qopt.load_state_dict(ck["qopt"])
        trainer.vopt.load_state_dict(ck["vopt"])
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        start_step = ck["step"]
    log(
        root / f"critic_{args.tag}.jsonl",
        dict(event="start", settings=settings, **buffer.summary()),
    )
    start = time.monotonic()
    for step in range(start_step, settings["steps"]):
        ids = torch.randint(len(z), (settings["batch_size"],), device=device)
        metrics = trainer.update({k: v[ids] for k, v in data.items()})
        if (step + 1) % 100 == 0 or step == start_step:
            metrics.update(step=step + 1, seconds=time.monotonic() - start)
            log(root / f"critic_{args.tag}.jsonl", metrics)
        if (step + 1) % 1000 == 0 or step + 1 == settings["steps"]:
            save_checkpoint(
                dest,
                dict(
                    critic=critic.state_dict(),
                    value=value.state_dict(),
                    target=trainer.target.state_dict(),
                    qopt=trainer.qopt.state_dict(),
                    vopt=trainer.vopt.state_dict(),
                    step=step + 1,
                    settings=settings,
                    config=cfg,
                    buffer_signature=signature,
                    cuda_rng=torch.cuda.get_rng_state(),
                ),
            )
    critic.eval().requires_grad_(False)
    ids = torch.randperm(len(z), device=device)[:256]
    _, diagnostics = improve_actions(critic.mean, data["z"][ids], data["action"][ids])
    with torch.no_grad():
        scores = torch.cat(
            [
                critic.mean(data["z"][i : i + 256], data["action"][i : i + 256])
                for i in range(0, len(z), 256)
            ]
        )
        for label in [True, False]:
            mask = data["success"] == label
            diagnostics["q_success" if label else "q_failure"] = (
                scores[mask].mean().item() if mask.any() else None
            )
    diagnostics.update(event="diagnostics", step=settings["steps"])
    log(root / f"critic_{args.tag}.jsonl", diagnostics)
    (root / f"critic_diagnostics_{args.tag}.json").write_text(
        __import__("json").dumps(diagnostics, indent=2)
    )
    if not diagnostics["grad_norm"] > 1e-10 or not diagnostics["q_gain"] > 0:
        raise RuntimeError("Critic guidance gate failed; do not train actor")


if __name__ == "__main__":
    main()

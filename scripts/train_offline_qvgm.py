"""Frozen offline critic/reference; one actor step per sum of late-step losses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.training import (
    buffer_signature,
    log,
    make_critic,
    proprio_enabled,
    proprio_features,
    save_checkpoint,
    stage_args,
)


def main():
    args, cfg, settings, root = stage_args("actor")
    # Reject resuming older checkpoints with rounded bf16 supervision targets.
    settings["loss_precision"] = "float32"
    settings["bias_critic"] = bool(args.bias_critic)
    import copy
    import json
    import random
    import time

    import torch

    from qvgm.algorithms.qvgm_loss import local_velocity_loss
    from qvgm.data.replay_buffer import ReplayBuffer
    from qvgm.models.critic import ConstantDirectionQ
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy

    buffer = ReplayBuffer(root / "buffer")
    signature = buffer_signature(buffer)
    features = torch.load(root / f"features_{args.tag}.pt", weights_only=True)
    if features["buffer_signature"] != signature:
        raise ValueError("Buffer/feature provenance mismatch")
    device = cfg["runtime"]["device"]
    stats = None
    if args.bias_critic:
        direction = json.loads((root / f"guidance_direction_{args.tag}.json").read_text())
        critic = ConstantDirectionQ(
            direction["global_direction"], cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
        )
    else:
        diagnostics = json.loads((root / f"critic_diagnostics_{args.tag}.json").read_text())
        if not diagnostics["q_gain"] > 0 or not diagnostics["grad_norm"] > 1e-10:
            raise RuntimeError("Critic action-gradient gate failed")
        ck = torch.load(root / f"critic_{args.tag}.pt", weights_only=True, map_location="cpu")
        if ck["buffer_signature"] != signature:
            raise ValueError("Critic provenance mismatch")
        stats = ck.get("proprio_stats")
        if proprio_enabled(cfg) != (stats is not None):
            raise ValueError("critic.proprio does not match the critic checkpoint")
        extra = 0 if stats is None else int(stats["mean"].numel())
        critic = make_critic(cfg, extra)
        critic.load_state_dict(ck["critic"])
        del ck
    critic = critic.to(device).eval().requires_grad_(False)
    policy = load_sft_policy(cfg)
    actor = Pi05Flow(policy, cfg)
    # Share frozen VLM; only duplicate the small action expert and projections.
    ref_model = copy.copy(actor.model)
    ref_model._modules = actor.model._modules.copy()
    ref_model.paligemma_with_expert = copy.copy(actor.model.paligemma_with_expert)
    ref_model.paligemma_with_expert._modules = actor.model.paligemma_with_expert._modules.copy()
    ref_model.paligemma_with_expert.gemma_expert = copy.deepcopy(
        actor.model.paligemma_with_expert.gemma_expert
    )
    for name in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
        setattr(ref_model, name, copy.deepcopy(getattr(actor.model, name)))
    reference = copy.copy(actor)
    reference.model = ref_model
    reference.actor_training = True  # Same autocast arithmetic; weights stay frozen.
    ref_model.requires_grad_(False)
    params = actor.enable_actor_training(settings.get("master_dtype", "float32"))
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=0.0)
    trainable_names = {n for n, p in actor.model.named_parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in reference.model.parameters())
    reference_parameters = dict(reference.model.named_parameters())
    assert all(
        p.data_ptr() != reference_parameters[n].data_ptr()
        for n, p in actor.model.named_parameters()
        if n in trainable_names
    )
    dest = root / f"actor_{args.tag}.pt"
    start_step = 0
    if dest.exists():
        if not args.resume:
            raise FileExistsError(dest)
        ck = torch.load(dest, weights_only=True, map_location="cpu")
        if ck["buffer_signature"] != signature or ck["settings"] != settings:
            raise ValueError("Resume mismatch")
        missing, unexpected = actor.model.load_state_dict(ck["actor"], strict=False)
        if unexpected or trainable_names.intersection(missing):
            raise ValueError("Incomplete actor checkpoint")
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        random.setstate(ck["random_state"])
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    log(
        root / f"actor_{args.tag}.jsonl",
        dict(event="start", settings=settings, trainable_parameters=sum(p.numel() for p in params)),
    )
    start = time.monotonic()
    batch = settings["batch_size"]
    micro = settings.get("micro_batch", 1)
    K, M = cfg["model"]["denoising_steps"], settings["late_steps"]
    if not 0 < M <= K:
        raise ValueError("Invalid late-step count")
    for step in range(start_step, settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        samples = random.choices(buffer.transitions, k=batch)
        sums = {}
        for offset in range(0, batch, micro):
            ids = samples[offset : offset + micro]
            context = actor.encode_context([buffer.observation(e, i) for e, i in ids])
            z = torch.stack([features["z"][e][i] for e, i in ids]).to(device)
            if stats is not None:
                z = torch.cat([z, proprio_features(buffer, ids, stats).to(device)], -1)
            _, trajectory = actor.sample_with_intermediates(context)
            for tau, x in trajectory[K - M :]:
                loss, metrics = local_velocity_loss(
                    actor,
                    reference,
                    context,
                    z,
                    critic,
                    tau,
                    x,
                    settings,
                    cfg["env"]["action_chunk"],
                    cfg["env"]["action_dim"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite alignment loss")
                (loss * len(ids) / batch).backward()
                for key, value in dict(metrics, loss=loss.item()).items():
                    sums[key] = sums.get(key, 0.0) + value * len(ids) / batch
        grad = torch.nn.utils.clip_grad_norm_(
            params, settings["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        assert all(p.grad is None for p in critic.parameters())
        assert all(p.grad is None for p in reference.model.parameters())
        sums.update(step=step + 1, actor_grad_norm=float(grad), seconds=time.monotonic() - start)
        # Loss is a SUM across late steps; guidance diagnostics are averages.
        for key in [
            "q_before",
            "q_after",
            "q_gain",
            "displacement",
            "accept_rate",
            "grad_norm",
            "correction_norm",
        ]:
            sums[key] /= M
        log(root / f"actor_{args.tag}.jsonl", sums)
        if (step + 1) % 25 == 0 or step + 1 == settings["steps"]:
            save_checkpoint(
                dest,
                dict(
                    actor={
                        n: t for n, t in actor.model.state_dict().items() if n in trainable_names
                    },
                    optimizer=optimizer.state_dict(),
                    step=step + 1,
                    settings=settings,
                    config=cfg,
                    buffer_signature=signature,
                    random_state=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state(),
                ),
            )
    print(f"Offline actor saved: {dest}", flush=True)


if __name__ == "__main__":
    main()

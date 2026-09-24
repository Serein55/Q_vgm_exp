"""Pretrain the RLT autoencoder on frozen policy prefix states."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.training import buffer_signature, log, make_autoencoder, save_checkpoint, stage_args


def main():
    args, cfg, settings, root = stage_args("rl_token")
    import random
    import time

    import torch

    from qvgm.data.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(root / "buffer")
    signature = buffer_signature(buffer)
    device = cfg["runtime"]["device"]
    model = make_autoencoder(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.0)
    dest = root / f"rl_token_{args.tag}.pt"
    start_step = 0
    if dest.exists():
        if not args.resume:
            raise FileExistsError(f"{dest}; use --resume or a new --tag")
        ck = torch.load(dest, weights_only=True, map_location="cpu")
        if ck["buffer_signature"] != signature or ck["settings"] != settings:
            raise ValueError("Resume configuration/buffer mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        random.setstate(ck["random_state"])
        torch.set_rng_state(ck["torch_rng"])
    batch = settings["batch_size"]
    micro = settings.get("micro_batch", 1)
    # Legacy runs held out 20%; the paper-reading profile uses the whole buffer.
    fraction = settings.get("validation_episode_fraction", 0.2)
    if not 0 <= fraction < 1:
        raise ValueError("validation_episode_fraction must be in [0, 1)")
    validation_episodes = set()
    for task in {ep["task_id"] for ep in buffer.episodes}:
        episodes = [e for e, ep in enumerate(buffer.episodes) if ep["task_id"] == task]
        if fraction > 0 and len(episodes) >= 2:
            validation_episodes.update(episodes[-max(1, int(len(episodes) * fraction)) :])
    train = [s for s in buffer.states if s[0] not in validation_episodes]
    valid = [s for s in buffer.states if s[0] in validation_episodes]
    if not train:
        raise ValueError("No autoencoder training states")
    log(
        root / f"rl_token_{args.tag}.jsonl",
        dict(
            event="start",
            settings=settings,
            parameters=sum(p.numel() for p in model.parameters()),
            **buffer.summary(),
        ),
    )
    start = time.monotonic()
    for step in range(start_step, settings["steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = random.choices(train, k=batch)
        total = 0.0
        for offset in range(0, batch, micro):
            ids = indices[offset : offset + micro]
            x, mask = buffer.prefix_batch(ids, device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
            ):
                loss, _ = model.loss(x, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss")
            (loss * len(ids) / batch).backward()
            total += loss.item() * len(ids) / batch
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 10 == 0 or step == start_step:
            metrics = dict(
                step=step + 1,
                reconstruction_mse=total,
                grad_norm=float(grad),
                seconds=time.monotonic() - start,
            )
            if str(device).startswith("cuda"):
                metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            if valid:
                model.eval()
                with (
                    torch.no_grad(),
                    torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
                    ),
                ):
                    x, mask = buffer.prefix_batch(valid[:micro], device)
                    metrics["validation_mse"] = model.loss(x, mask)[0].item()
            log(root / f"rl_token_{args.tag}.jsonl", metrics)
        if (step + 1) % settings.get("checkpoint_interval", 500) == 0 or step + 1 == settings[
            "steps"
        ]:
            save_checkpoint(
                dest,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step + 1,
                    settings=settings,
                    config=cfg,
                    buffer_signature=signature,
                    random_state=random.getstate(),
                    torch_rng=torch.get_rng_state(),
                ),
            )
    # Cache frozen embeddings once; critic training never runs the VLA/autoencoder.
    model.eval().requires_grad_(False)
    encoded = []
    with torch.no_grad():
        for e, ep in enumerate(buffer.episodes):
            parts = []
            for i in range(0, len(ep["prefixes"]), micro):
                indices = [(e, j) for j in range(i, min(i + micro, len(ep["prefixes"])))]
                x, mask = buffer.prefix_batch(indices, device)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
                ):
                    parts.append(model.encode_flat(x, mask).float().cpu())
            encoded.append(torch.cat(parts))
    save_checkpoint(
        root / f"features_{args.tag}.pt",
        dict(z=encoded, buffer_signature=signature, settings=settings),
    )
    print(f"Frozen features ready: {len(encoded)} episodes", flush=True)


if __name__ == "__main__":
    main()

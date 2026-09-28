"""Shared offline stage helpers and checkpoint provenance."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from qvgm.config import load_config, setup_runtime

PROPRIO_KEY = "observation/state"


def stage_args(stage):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--run", default="spatial_offline")
    p.add_argument("--tag", default="full")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--resume", action="store_true")
    if stage == "actor":
        p.add_argument("--ascent-step-size", type=float)
        p.add_argument("--lr", type=float)
        p.add_argument(
            "--bias-critic",
            action="store_true",
            help="Replace the learned critic with q(z,a)=a.m using the measured global direction",
        )
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    settings = dict(cfg["offline"][stage])
    if stage == "actor" and args.ascent_step_size is not None:
        if not 0 <= args.ascent_step_size < float("inf"):
            p.error("ascent-step-size must be finite and nonnegative")
        settings["ascent_step_size"] = args.ascent_step_size
    if stage == "actor" and args.lr is not None:
        if not 0 < args.lr < float("inf"):
            p.error("lr must be finite and positive")
        settings["lr"] = args.lr
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.micro_batch is not None:
        settings["micro_batch"] = args.micro_batch
    if settings.get("steps") is None or settings["steps"] <= 0:
        p.error("Set an explicit positive --steps; this value is unspecified in the paper")
    if settings.get("batch_size", 1) < 1 or settings.get("micro_batch", 1) < 1:
        p.error("Batch sizes must be positive")
    root = Path(cfg["paths"]["artifacts"]) / args.run
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    random.seed(cfg["runtime"]["seed"])
    return args, cfg, settings, root


def buffer_signature(buffer):
    return hashlib.sha256(
        "\n".join(f"{p.name}:{p.stat().st_size}" for p in buffer.files).encode()
    ).hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def log(path, metrics):
    print(json.dumps(metrics), flush=True)
    with Path(path).open("a") as f:
        f.write(json.dumps(metrics) + "\n")


def make_autoencoder(config, seq_len=1024):
    from qvgm.models.rl_token import RLTTokenTransformer

    return RLTTokenTransformer(
        input_dim=2048,
        embed_dim=config["dim"],
        prefix_seq_len=seq_len,
        num_layers=config["layers"],
        num_heads=config["heads"],
        mlp_ratio=config.get("mlp_ratio", 4.0),
    )


def make_critic(cfg, extra_state_dim=0):
    from qvgm.models.critic import ChunkCritic

    c = cfg["offline"]["critic"]
    return ChunkCritic(
        critic_state_dim(cfg, extra_state_dim),
        cfg["env"]["action_chunk"],
        cfg["env"]["action_dim"],
        c["heads"],
        c["widths"],
    )


def critic_state_dim(cfg, extra_state_dim=0):
    return cfg["offline"]["rl_token"]["dim"] + extra_state_dim


def proprio_enabled(cfg):
    """Critic-side proprioception only; the policy conditioning stays stateless."""
    return bool(cfg["offline"]["critic"].get("proprio", False))


def proprio_stats(buffer):
    rows = torch.stack(
        [
            torch.as_tensor(buffer.observation(e, i)[PROPRIO_KEY], dtype=torch.float32)
            for e, i in buffer.states
        ]
    )
    if not torch.isfinite(rows).all():
        raise ValueError("Non-finite proprioception in buffer")
    return dict(mean=rows.mean(0), std=rows.std(0).clamp_min(1e-3))


def proprio_features(buffer, pairs, stats):
    rows = torch.stack(
        [
            torch.as_tensor(buffer.observation(e, i)[PROPRIO_KEY], dtype=torch.float32)
            for e, i in pairs
        ]
    )
    return (rows - stats["mean"]) / stats["std"]

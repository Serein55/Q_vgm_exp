"""CPU-only replay label and saved IQL target audit; no parameter updates."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from qvgm.algorithms.iql import chunk_target, expectile_loss
from qvgm.config import load_config
from qvgm.data.replay_buffer import ReplayBuffer
from qvgm.models.critic import Value
from qvgm.training import buffer_signature, make_critic


def main():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/libero_spatial_checkpoint.yaml"
    )
    torch.set_num_threads(4)
    root = Path(cfg["paths"]["artifacts"]) / "spatial_paper_checkpoint"
    dest = root / "label_target_audit.json"
    if dest.exists():
        raise FileExistsError(dest)
    replay = ReplayBuffer(root / "buffer")
    features = torch.load(root / "features_full.pt", weights_only=True, map_location="cpu")
    ck = torch.load(root / "critic_full.pt", weights_only=True, map_location="cpu")
    sig = buffer_signature(replay)
    assert ck["buffer_signature"] == features["buffer_signature"] == sig
    gamma = cfg["offline"]["gamma"]
    errors, reward_error, rows = [], 0.0, []
    for e, ep in enumerate(replay.episodes):
        ts = ep["transitions"]
        if len(features["z"][e]) != len(ts) + 1:
            errors.append([e, "feature count"])
        total = sum(t["steps"] for t in ts)
        if not (
            ts[-1]["terminated"] == ep["success"]
            and ts[-1]["truncated"] == (not ep["success"])
            and (ep["success"] or total == cfg["env"]["max_steps"])
        ):
            errors.append([e, "episode ending"])
        for i, t in enumerate(ts):
            rs = t["rewards"].tolist()
            reward_error = max(
                reward_error, abs(t["reward"] - sum(gamma**j * r for j, r in enumerate(rs)))
            )
            expected = [0.0] * t["steps"]
            if i == len(ts) - 1 and ep["success"]:
                expected[-1] = 1.0
            if rs != expected or (i < len(ts) - 1 and (t["terminated"] or t["truncated"])):
                errors.append([e, i, "reward/early ending"])
            if t["action"].shape != (5, 7):
                errors.append([e, i, "action shape"])
            rows.append((e, i, t))
    critic, target = make_critic(cfg), make_critic(cfg)
    value = Value(cfg["offline"]["rl_token"]["dim"], ck["settings"]["widths"])
    for model, key in [(critic, "critic"), (target, "target"), (value, "value")]:
        model.load_state_dict(ck[key])
        model.eval().requires_grad_(False)
    collected = {k: [] for k in ["q", "y", "v", "target_q", "next_v"]}
    with torch.no_grad():
        for off in range(0, len(rows), 64):
            batch = rows[off : off + 64]
            z = torch.stack([features["z"][e][i] for e, i, t in batch])
            nz = torch.stack([features["z"][e][i + 1] for e, i, t in batch])
            a = torch.stack([t["action"] for e, i, t in batch])
            reward = torch.tensor([t["reward"] for e, i, t in batch])
            steps = torch.tensor([t["steps"] for e, i, t in batch])
            done = torch.tensor([t["terminated"] for e, i, t in batch])
            nv = value(nz)
            y = chunk_target(reward, steps, done, nv, gamma)
            independent = torch.tensor(
                [
                    t["reward"] + (0 if t["terminated"] else gamma ** t["steps"] * float(nv[j]))
                    for j, (e, i, t) in enumerate(batch)
                ]
            )
            torch.testing.assert_close(y, independent, atol=1e-6, rtol=1e-6)
            for key, tensor in dict(
                q=critic(z, a), y=y, v=value(z), target_q=target(z, a).min(-1).values, next_v=nv
            ).items():
                collected[key].append(tensor)
    data = {k: torch.cat(v) for k, v in collected.items()}
    assert all(torch.isfinite(v).all() for v in data.values())
    groups = {}
    for name, predicate in [
        ("terminal", lambda t: t["terminated"]),
        ("truncated", lambda t: t["truncated"]),
        ("ordinary", lambda t: not (t["terminated"] or t["truncated"])),
    ]:
        mask = torch.tensor([predicate(t) for e, i, t in rows])
        groups[name] = dict(
            count=int(mask.sum()),
            target_mean=float(data["y"][mask].mean()),
            q_mean=float(data["q"][mask].mean()),
            next_value_mean=float(data["next_v"][mask].mean()),
            q_target_rmse=float((data["q"][mask] - data["y"][mask, None]).square().mean().sqrt()),
        )
    report = dict(
        buffer_signature=sig,
        episodes=len(replay.episodes),
        transitions=len(rows),
        errors=errors,
        reward_max_abs_error=reward_error,
        independent_target_check="passed (atol=rtol=1e-6)",
        short_chunks=sum(t["steps"] < 5 for e, i, t in rows),
        groups=groups,
        q_target_rmse=float((data["q"] - data["y"][:, None]).square().mean().sqrt()),
        value_expectile_loss=float(
            expectile_loss(data["target_q"] - data["v"], ck["settings"]["expectile"])
        ),
        target_min=float(data["y"].min()),
        target_max=float(data["y"].max()),
        note="In-sample fixed checkpoint residuals; not held-out accuracy. Time-limit bootstrap is an explicit modeling choice. Provenance signature checks filenames/sizes, not file contents.",
    )
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if errors or reward_error > 1e-6:
        raise RuntimeError("Replay label audit failed")


if __name__ == "__main__":
    main()

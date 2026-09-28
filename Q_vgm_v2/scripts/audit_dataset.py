"""Audit committed schema-2 shards; never declare an incomplete dataset ready."""

import argparse
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("Q_vgm_v2/artifacts"))
    args = p.parse_args()
    report = {"sources": {}, "errors": []}
    manifest = []
    for source, directory in [("rollout", "v2_sft_500"), ("demo", "v2_expert")]:
        counts = {str(t): dict(episodes=0, successes=0) for t in range(10)}
        transitions = positions = 0
        action_sum = torch.zeros(7, dtype=torch.float64)
        action_sq = action_sum.clone()
        all_valid = True
        for path in sorted((args.root / directory / "buffer").glob("task*_episode*.pt")):
            try:
                ep = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
                ts, obs, prefixes = ep["transitions"], ep["observations"], ep["prefixes"]
                assert ep["schema"] == 2 and ep["source"] == source
                assert len(obs) == len(prefixes) == len(ts) + 1 and len(ts) > 0
                task, episode = ep["task_id"], ep["episode"]
                assert task in range(10) and episode in range(50)
                assert path.name == f"task{task:02d}_episode{episode:03d}.pt"
                for o, prefix in zip(obs, prefixes):
                    assert o["observation/state"].shape == (8,)
                    assert torch.isfinite(o["observation/state"]).all()
                    assert o["observation/image"].shape == (256, 256, 3)
                    assert o["observation/wrist_image"].shape == (256, 256, 3)
                    assert prefix.ndim == 2 and prefix.shape[1] == 2048 and len(prefix) > 0
                    assert torch.isfinite(prefix).all()
                any_reward = False
                for i, t in enumerate(ts):
                    n = t["steps"]
                    assert 0 < n <= 5 and t["action"].shape == (5, 7)
                    assert t["rewards"].shape == (n,)
                    assert torch.isfinite(t["action"]).all() and torch.isfinite(t["rewards"]).all()
                    assert bool(((t["rewards"] == 0) | (t["rewards"] == 1)).all())
                    assert (
                        abs(
                            t["reward"]
                            - sum(0.99**j * float(r) for j, r in enumerate(t["rewards"]))
                        )
                        < 1e-6
                    )
                    assert not (t["terminated"] and t["truncated"])
                    if i < len(ts) - 1:
                        assert n == 5 and not t["terminated"] and not t["truncated"]
                    else:
                        assert t["terminated"] or t["truncated"]
                    any_reward |= bool(t["rewards"].any())
                    actions = t["action"][:n].double()
                    action_sum += actions.sum(0)
                    action_sq += actions.square().sum(0)
                    positions += n
                assert bool(ep["success"]) == any_reward
                eligible = True
                if source == "rollout":
                    assert ep["success_once"] == any_reward
                    assert ep["initial_state_index"] == episode
                else:
                    eligible = bool(ep["terminal_success_rechecked"])
                    if not eligible:
                        report["errors"].append(
                            f"{path}: source success not confirmed at restored terminal"
                        )
                        all_valid = False
                counts[str(task)]["episodes"] += 1
                counts[str(task)]["successes"] += int(ep["success"])
                transitions += len(ts)
                manifest.append(
                    dict(
                        source=source,
                        path=str(path.resolve()),
                        task_id=task,
                        episode=episode,
                        bytes=path.stat().st_size,
                        eligible=eligible,
                    )
                )
            except (AssertionError, KeyError, ValueError) as error:
                all_valid = False
                report["errors"].append(f"{path}: {type(error).__name__}: {error}")
        n = sum(t["episodes"] for t in counts.values())
        mean = action_sum / max(positions, 1)
        std = (action_sq / max(positions, 1) - mean.square()).clamp_min(0).sqrt()
        report["sources"][source] = dict(
            episodes=n,
            per_task=counts,
            transitions=transitions,
            valid_action_positions=positions,
            action_mean=mean.tolist(),
            action_std=std.tolist(),
            complete=all(t["episodes"] == 50 for t in counts.values()),
            valid=all_valid,
        )
    report["ready"] = all(s["complete"] and s["valid"] for s in report["sources"].values())
    report["paper_demo_subset_match"] = False
    report["note"] = "Official raw Spatial has 500 demos; paper 432-demo subset is not identified."
    for name, data in [
        ("dataset_report.json", report),
        ("combined_manifest.json", dict(ready=report["ready"], episodes=manifest)),
    ]:
        dest = args.root / name
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(dest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

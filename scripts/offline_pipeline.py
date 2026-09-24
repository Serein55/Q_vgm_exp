"""Resumable sequential full offline run; never launches a new SFT baseline."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import ROOT, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/libero_spatial.yaml"))
    p.add_argument("--run", default="spatial_offline")
    p.add_argument("--actor-steps", type=int, default=500)
    p.add_argument("--tag", default="full")
    args = p.parse_args()
    if args.actor_steps <= 0:
        p.error("actor-steps must be positive")
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["artifacts"]) / args.run
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    status_path = root / f"pipeline_{args.tag}.json"
    identity = dict(config=cfg, actor_steps=args.actor_steps, tag=args.tag)
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous and previous["identity"] != identity:
        raise ValueError("Existing pipeline has different settings; use a new tag/run")
    completed = previous.get("completed", [])
    common = ["--config", str(Path(args.config).resolve()), "--run", args.run]
    stages = [
        ("collect_rollouts", common + ["--check-flow"]),
        ("train_rl_token", common + ["--tag", args.tag]),
        ("train_critic", common + ["--tag", args.tag]),
        ("train_offline_qvgm", common + ["--tag", args.tag, "--steps", str(args.actor_steps)]),
        (
            "eval_qvgm",
            [
                "--config",
                str(Path(args.config).resolve()),
                "--name",
                f"qvgm_{args.tag}",
                "--actor-checkpoint",
                str(root / f"actor_{args.tag}.pt"),
            ],
        ),
    ]
    files = dict(
        train_rl_token=root / f"rl_token_{args.tag}.pt",
        train_critic=root / f"critic_{args.tag}.pt",
        train_offline_qvgm=root / f"actor_{args.tag}.pt",
    )

    def report(stage, state, **extra):
        status = dict(
            identity=identity,
            completed=completed,
            stage=stage,
            status=state,
            pid=os.getpid(),
            updated=datetime.now(timezone.utc).isoformat(),
            **extra,
        )
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)
        print(json.dumps({k: v for k, v in status.items() if k != "identity"}), flush=True)
        with (ROOT / "复现过程.md").open("a") as f:
            f.write(
                f"\n- 自动流程 `{args.tag}`：`{stage}` {state}（{status['updated']}），详见 `artifacts/{args.run}/pipeline_{args.tag}.json`。\n"
            )

    for stage, arguments in stages:
        if stage in completed:
            continue
        if stage in files and files[stage].exists():
            arguments += ["--resume"]
        report(stage, "running")
        logfile = root / f"{stage}_{args.tag}.log"
        with logfile.open("a") as output:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / f"{stage}.py"), *arguments],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            report(stage, "failed", returncode=result.returncode, log=str(logfile))
            raise SystemExit(result.returncode)
        completed.append(stage)
        report(stage, "complete", log=str(logfile))
    report("offline_pipeline", "complete")


if __name__ == "__main__":
    main()

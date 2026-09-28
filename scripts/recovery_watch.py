"""Persist candidate collection -> critic fitting; never launch actor before gates."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=str(Path(__file__).resolve().parents[1] / "configs/controller_coverage.yaml"),
)
parser.add_argument("--lanes", type=int, default=1)
args = parser.parse_args()
if args.lanes < 1:
    parser.error("lanes must be positive")
config_path = str(Path(args.config).resolve())
cfg = load_config(config_path)
c = cfg["recovery"]
root = Path(cfg["paths"]["artifacts"]) / c["run"]
status = root / "pipeline_status.json"


def write(stage, **kwargs):
    status.write_text(json.dumps(dict(stage=stage, **kwargs), indent=2) + "\n")


write("collecting")
while True:
    completed = []
    for i in range(c["workers"]):
        worker_done = True
        for lane in range(args.lanes):
            filename = "status.txt" if args.lanes == 1 else f"status_lane{lane}.txt"
            p = root / f"worker{i}" / filename
            done = p.exists() and p.read_text().strip() == "complete"
            worker_done = worker_done and done
            if not done:
                suffix = "" if args.lanes == 1 else f"-lane{lane}"
                session = f"{c.get('session_prefix', 'qvgm-recovery-data')}{i}{suffix}"
                result = subprocess.run(
                    ["tmux", "display-message", "-p", "-t", session, "#{pane_dead}"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode or result.stdout.strip() == "1":
                    write(
                        "failed",
                        worker=i,
                        lane=lane,
                        reason="collection process exited before completion",
                    )
                    sys.exit(1)
        completed.append(worker_done)
        if worker_done and args.lanes > 1:
            (root / f"worker{i}/status.txt").write_text("complete\n")
    if all(completed):
        break
    time.sleep(30)
write("collection_complete", actor_started=False)

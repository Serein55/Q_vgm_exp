"""Read committed shards so resume/log duplication cannot inflate success counts."""

import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("buffer", type=Path)
a = p.parse_args()
counts = {i: {"episodes": 0, "success_once": 0} for i in range(10)}
for path in sorted(a.buffer.glob("task*_episode*.pt")):
    ep = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    row = counts[ep["task_id"]]
    row["episodes"] += 1
    row["success_once"] += int(ep["success_once"])
for row in counts.values():
    row["rate"] = row["success_once"] / row["episodes"] if row["episodes"] else None
n = sum(r["episodes"] for r in counts.values())
k = sum(r["success_once"] for r in counts.values())
report = dict(
    episodes=n,
    expected=500,
    success_once=k,
    rate=k / n if n else None,
    complete=all(r["episodes"] == 50 for r in counts.values()),
    per_task=counts,
)
out = a.buffer.parent / "success_once_summary.json"
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(report, indent=2) + "\n")
tmp.replace(out)
print(json.dumps(report, indent=2))

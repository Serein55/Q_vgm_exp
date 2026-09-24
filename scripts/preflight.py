"""Inspect assets, checkpoint schema and GPU without loading model weights."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import torch
    from safetensors import safe_open

    paths = cfg["paths"]
    checks = {k: Path(v).exists() for k, v in paths.items()}
    weights = Path(paths["checkpoint"]) / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as f:
        tensor_count = len(f.keys())
        action_shape = f.get_slice("action_out_proj.weight").get_shape()
    stats = json.loads(Path(paths["norm_stats"]).read_text())["norm_stats"]
    versions = {}
    for name in ["torch", "transformers", "mujoco", "openpi", "safetensors"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unpackaged"
    report = dict(
        paths=checks,
        versions=versions,
        tensors=tensor_count,
        action_out_shape=action_shape,
        norm_dimensions={k: len(v["mean"]) for k, v in stats.items()},
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    destination = Path(paths["artifacts"]) / "preflight.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(checks.values()) or not report["cuda_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

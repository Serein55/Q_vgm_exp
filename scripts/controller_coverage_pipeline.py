"""Wait for training-only coverage collection, then fit against fixed development labels."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "artifacts/controller_coverage"
    status = data / "training_status.json"

    def write(stage, **extra):
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(stage=stage, **extra), indent=2) + "\n")
        tmp.replace(status)

    def run(script, *args):
        subprocess.run([sys.executable, str(root / "scripts" / script), *args], check=True)

    try:
        write("collecting_training_states")
        run("recovery_watch.py", "--config", str(root / "configs/controller_coverage.yaml"))
        import torch

        old = sorted((root / "artifacts/controller_training").glob("worker*/*.pt"))
        new = sorted(data.glob("worker*/*.pt"))
        assert len(old) == 96 and len(new) == 64
        files = old + new
        rows = [torch.load(p, weights_only=True) for p in files]
        assert len({r["key"] for r in rows}) == 160
        train, val = [], []
        hashes = {}
        for i, (p, r) in enumerate(zip(files, rows)):
            assert r["task"] in [1, 5, 6, 9] and r["state_index"] in [0, 10]
            assert r["candidate_method"] == "controller_grid"
            assert len(r["records"]) == 144
            assert len({(t["candidate"], t["repeat"]) for t in r["records"]}) == 144
            if i >= 96:
                assert 27 <= r["episode"] <= 34
            else:
                assert 15 <= r["episode"] <= 26
            (val if 23 <= r["episode"] <= 26 else train).append(r["key"])
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert len(train) == 128 and len(val) == 32
        manifest = data / "training_manifest.json"
        manifest.write_text(
            json.dumps(
                dict(
                    files=[str(p) for p in files],
                    train_keys=train,
                    validation_keys=val,
                    sha256=hashes,
                    note="Original 32 development groups unchanged. Additional groups are training only. No independent confirmation.",
                ),
                indent=2,
            )
            + "\n"
        )
        write("fitting", train_groups=128, validation_groups=32)
        for seed in [7108, 7119, 7130]:
            name = f"controller_coverage_{seed}"
            if not (root / "artifacts" / name / "report.json").exists():
                run(
                    "train_controller_critic.py",
                    "--manifest",
                    str(manifest),
                    "--run",
                    name,
                    "--seed",
                    str(seed),
                    "--dropout",
                    "0.15",
                    "--steps",
                    "600",
                    "--eval-every",
                    "5",
                )
        run(
            "controller_training_report.py",
            "--manifest",
            str(manifest),
            "--prefix",
            "controller_coverage",
        )
        write("complete", report=str(root / "artifacts/controller_coverage_summary/report.json"))
    except Exception as exc:
        write("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

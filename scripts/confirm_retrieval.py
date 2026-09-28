"""Audit complete frozen-policy paired confirmation; never fit on confirmation labels."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_policy import FrozenRetrievalPolicy

from qvgm.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/retrieval_confirmation.yaml"),
    )
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["recovery"]
    root = Path(cfg["paths"]["artifacts"]) / c["run"]
    status = root / "confirmation_status.json"

    def write(stage, **kwargs):
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(stage=stage, **kwargs), indent=2) + "\n")
        tmp.replace(status)

    try:
        protocol = json.loads((root / "protocol.json").read_text())
        assert protocol["config"] == cfg
        if args.watch:
            write("collecting")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("recovery_watch.py")),
                    "--config",
                    args.config,
                ],
                check=True,
            )
        for path, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        policy = FrozenRetrievalPolicy(cfg["paths"]["retrieval_policy"], c["retrieval_sha256"])
        rows = [torch.load(p, weights_only=True) for p in sorted(root.glob("worker*/*.pt"))]
        expected = {
            (t, e, s)
            for t in c["task_ids"]
            for e in range(c["fresh_first_episode"], c["fresh_first_episode"] + c["episodes"])
            for s in [0, c["fresh_prefix_chunks"]]
        }
        assert len(rows) == len(expected) == 40
        assert {(r["task"], r["episode"], r["state_index"]) for r in rows} == expected
        base, fixed, selected = [], [], []
        for r in rows:
            assert r["retrieval_sha256"] == policy.sha256
            pick, reference, donors = policy.choose(
                r["mean_pool"].numpy(),
                r["observation"]["observation/state"].numpy(),
                r["candidates"][0, :5, :7].numpy(),
                r["task"],
                r["actual_prefix_chunks"],
            )
            assert r["candidate_names"][pick] == r["retrieval_selected"]
            assert r["candidate_names"][reference] == r["retrieval_reference"] == "x_minus"
            assert donors == r["retrieval_donors"]
            names = {"base", "x_minus", r["retrieval_selected"]}
            pairs = {(t["candidate"], t["repeat"]) for t in r["records"]}
            assert pairs == {(n, k) for n in names for k in range(c["repeats"])}
            assert len(r["records"]) == len(pairs)
            assert all(
                t["sim_error"] <= 1e-6 and t["state_error"] <= 1e-3 and t["image_error"] <= 1
                for t in r["records"]
            )
            for dest, name in [
                (base, "base"),
                (fixed, "x_minus"),
                (selected, r["retrieval_selected"]),
            ]:
                dest.append(
                    [
                        next(
                            t["success"]
                            for t in r["records"]
                            if t["candidate"] == name and t["repeat"] == k
                        )
                        for k in range(c["repeats"])
                    ]
                )
        arrays = {
            k: np.array(v, float)
            for k, v in [("base", base), ("fixed", fixed), ("retrieval", selected)]
        }
        report = dict(
            protocol="Frozen model; new initial-state IDs; single-chunk intervention then frozen SFT. Not continuous closed-loop evaluation.",
            checkpoint_sha256=policy.sha256,
            groups=len(rows),
            trials_per_method=arrays["base"].size,
            actual_rollouts=sum(len(r["records"]) for r in rows),
            successes={k: int(v.sum()) for k, v in arrays.items()},
            per_task={},
            comparisons={},
        )
        for t in c["task_ids"]:
            mask = np.array([r["task"] == t for r in rows])
            report["per_task"][str(t)] = {k: int(v[mask].sum()) for k, v in arrays.items()}
        for name in ["base", "fixed"]:
            delta = (arrays["retrieval"] - arrays[name]).mean(1)
            rng = np.random.default_rng(7108)
            draws = []
            for t in c["task_ids"]:
                es = sorted({r["episode"] for r in rows if r["task"] == t})
                clusters = np.array(
                    [delta[[r["task"] == t and r["episode"] == e for r in rows]].mean() for e in es]
                )
                draws.append(rng.choice(clusters, (20000, len(clusters)), replace=True).mean(1))
            report["comparisons"][name] = dict(
                gain=float(delta.mean()),
                cluster_interval90=np.quantile(np.mean(draws, axis=0), [0.05, 0.95]).tolist(),
            )
        report["single_intervention_gate_passed"] = all(
            v["cluster_interval90"][0] > 0 for v in report["comparisons"].values()
        )
        report["note"] = (
            "Gate preregistered before rollouts: positive lower fifth bootstrap percentile versus both controls. Only 20 task/episode clusters; not proof of general reliability. No test-label fitting."
        )
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        write(
            "complete",
            gate_passed=report["single_intervention_gate_passed"],
            next_step="closed_loop_check_required"
            if report["single_intervention_gate_passed"]
            else "do_not_deploy_as_improved_policy",
        )
        doc = Path(__file__).resolve().parents[1] / "idea改进实验.md"
        with doc.open("a") as f:
            f.write(
                "\n冻结检索独立起点确认已完成："
                + json.dumps(report["successes"], ensure_ascii=False)
                + f"，每策略 {report['trials_per_method']} 次；单次干预门槛 {'通过，尚需闭环确认' if report['single_intervention_gate_passed'] else '未通过，不作有效策略发布'}。完整报告 artifacts/retrieval_confirmation/summary.json。\n"
            )
        print(json.dumps(report), flush=True)
    except Exception as exc:
        write("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

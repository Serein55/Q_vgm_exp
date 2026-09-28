"""Complete frozen confirmation; conditionally run a small closed-loop check."""

import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config


def main():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/retrieval_confirmation.yaml"
    cfg = load_config(config)
    art = Path(cfg["paths"]["artifacts"])
    status = art / "retrieval_confirmation/completion_status.json"

    def write(stage, **kwargs):
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(stage=stage, **kwargs), indent=2) + "\n")
        tmp.replace(status)

    try:
        write("waiting_for_single_intervention")
        while True:
            p = art / "retrieval_confirmation/confirmation_status.json"
            state = json.loads(p.read_text()) if p.exists() else {}
            if state.get("stage") == "failed":
                raise RuntimeError(state)
            if state.get("stage") == "complete":
                break
            live = subprocess.run(
                ["tmux", "has-session", "-t", "qvgm-retrieval-confirm-watch"], capture_output=True
            )
            if live.returncode:
                if p.exists() and json.loads(p.read_text()).get("stage") == "complete":
                    break
                raise RuntimeError("Confirmation watcher exited before completion")
            time.sleep(30)
        summary = json.loads((art / "retrieval_confirmation/summary.json").read_text())
        if not summary["single_intervention_gate_passed"]:
            write(
                "finished_without_confirmed_improvement",
                single_intervention_passed=False,
                closed_loop_run=False,
            )
            return
        dest = art / "retrieval_closed_loop"
        dest.mkdir(exist_ok=False)
        protocol = json.loads(
            (art / "retrieval_confirmation/closed_loop_protocol.json").read_text()
        )
        for path, digest in protocol["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        (dest / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        sessions = []
        for i in range(cfg["recovery"]["workers"]):
            session = f"qvgm-retrieval-closed-{i}"
            sessions.append(session)
            command = (
                shlex.join(
                    [
                        "env",
                        f"CUDA_VISIBLE_DEVICES={i % 4}",
                        "OPENBLAS_NUM_THREADS=4",
                        cfg["paths"]["python"],
                        str(root / "scripts/retrieval_closed_loop.py"),
                        "--config",
                        str(config),
                        "--worker",
                        str(i),
                    ]
                )
                + " > "
                + shlex.quote(str(dest / f"worker{i}.log"))
                + " 2>&1"
            )
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, "-c", str(root), command], check=True
            )
        write("checking_closed_loop")
        while True:
            done = []
            for i, session in enumerate(sessions):
                p = dest / f"worker{i}/status.txt"
                complete = p.exists() and p.read_text().strip() == "complete"
                done.append(complete)
                if (
                    not complete
                    and subprocess.run(
                        ["tmux", "has-session", "-t", session], capture_output=True
                    ).returncode
                ):
                    raise RuntimeError(f"Closed-loop worker {i} exited early")
            if all(done):
                break
            time.sleep(30)
        rows = [json.loads(p.read_text()) for p in sorted(dest.glob("worker*/task*.json"))]
        assert len(rows) == 80 and len({(r["task"], r["episode"], r["repeat"]) for r in rows}) == 80
        assert {(r["task"], r["episode"], r["repeat"]) for r in rows} == {
            (t, e, k) for t in [1, 5, 6, 9] for e in range(45, 50) for k in range(4)
        }
        assert all(r["checkpoint_sha256"] == cfg["recovery"]["retrieval_sha256"] for r in rows)
        outcomes = {
            m: np.array([r["results"][m]["success"] for r in rows], float)
            for m in ["base", "fixed", "retrieval"]
        }
        report = dict(
            trials_per_method=80,
            successes={m: int(v.sum()) for m, v in outcomes.items()},
            comparisons={},
            per_task={
                str(t): {
                    m: int(v[[r["task"] == t for r in rows]].sum()) for m, v in outcomes.items()
                }
                for t in [1, 5, 6, 9]
            },
            note="Continuous intervention; same held-out initialization IDs as first stage, fresh noise. Conditional deployment check, not a second independent state test.",
        )
        for m in ["base", "fixed"]:
            delta = outcomes["retrieval"] - outcomes[m]
            rng = np.random.default_rng(7108)
            draws = []
            for t in [1, 5, 6, 9]:
                cl = np.array(
                    [
                        delta[[r["task"] == t and r["episode"] == e for r in rows]].mean()
                        for e in range(45, 50)
                    ]
                )
                draws.append(rng.choice(cl, (20000, 5), replace=True).mean(1))
            report["comparisons"][m] = dict(
                gain=float(delta.mean()),
                cluster_interval90=np.quantile(np.mean(draws, axis=0), [0.05, 0.95]).tolist(),
            )
        report["closed_loop_gate_passed"] = all(
            v["cluster_interval90"][0] > 0 for v in report["comparisons"].values()
        )
        (dest / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        write(
            "complete",
            single_intervention_passed=True,
            closed_loop_passed=report["closed_loop_gate_passed"],
            report=str(dest / "summary.json"),
        )
        with (root / "idea改进实验.md").open("a") as f:
            f.write(
                "\n冻结检索连续闭环检查完成："
                + json.dumps(report["successes"], ensure_ascii=False)
                + "，每策略 80 回合；门槛"
                + ("通过" if report["closed_loop_gate_passed"] else "未通过")
                + "。同一确认起点、新噪声的条件部署检查，不是第二批独立状态测试。见 artifacts/retrieval_closed_loop/summary.json。\n"
            )
    except Exception as exc:
        write("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

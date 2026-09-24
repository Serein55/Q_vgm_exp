"""Fit a small proprio-conditioned candidate critic; held-out episode gate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from qvgm.config import load_config


class CandidateCritic(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
                )
                for _ in range(heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/idea_recovery.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["recovery"]
    torch.set_num_threads(4)
    torch.manual_seed(cfg["runtime"]["seed"])
    root = Path(cfg["paths"]["artifacts"]) / c["run"]
    if (root / "candidate_critic.pt").exists():
        raise FileExistsError("Refuse to overwrite fitted model")
    for worker in range(c["workers"]):
        if (root / f"worker{worker}/status.txt").read_text().strip() != "complete":
            raise RuntimeError("Candidate collection incomplete")
    files = sorted(root.glob("worker*/*.pt"))
    expected = len(c["task_ids"]) * c["episodes"] * len(c["fractions"])
    if len(files) != expected:
        raise ValueError(f"Expected {expected} groups, got {len(files)}")
    rows = [torch.load(p, weights_only=True, map_location="cpu") for p in files]
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate groups")
    names = rows[0]["candidate_names"]
    nc = len(names)
    xstate = torch.stack(
        [torch.cat([r["z"].float(), r["observation"]["observation/state"].float()]) for r in rows]
    )
    action = torch.stack([r["candidates"][:, :5, :7].flatten(1) for r in rows])
    returns = torch.tensor(
        [
            [
                [
                    next(
                        t["return_value"]
                        for t in r["records"]
                        if t["candidate"] == name and t["repeat"] == repeat
                    )
                    for repeat in range(c["repeats"])
                ]
                for name in names
            ]
            for r in rows
        ]
    )
    success = torch.tensor(
        [
            [
                [
                    next(
                        float(t["success"])
                        for t in r["records"]
                        if t["candidate"] == name and t["repeat"] == repeat
                    )
                    for repeat in range(c["repeats"])
                ]
                for name in names
            ]
            for r in rows
        ]
    )
    mean_returns = returns.mean(-1)
    target_kind = c.get("target", "return")
    if target_kind not in ("return", "success"):
        raise ValueError("target must be return or success")
    y = success.mean(-1) if target_kind == "success" else mean_returns
    splits = {
        k: torch.tensor([i for i, r in enumerate(rows) if r["episode"] in c[f"{k}_episodes"]])
        for k in ["train", "validation", "test"]
    }
    assert sum(len(v) for v in splits.values()) == len(rows)
    assert all(len(v) for v in splits.values())
    tr = splits["train"]
    va = splits["validation"]
    # Fit all transforms only on training episodes; no held-out state leakage.
    mu = xstate[tr].mean(0)
    std = xstate[tr].std(0).clamp_min(0.05)
    normalized = (xstate - mu) / std
    # PCA reduces the frozen representation to training-supported directions.
    _, _, vh = torch.linalg.svd(normalized[tr, :2048], full_matrices=False)
    basis = vh[: min(8, len(tr) - 1)].T.contiguous()
    state = torch.cat([normalized[:, :2048] @ basis / (2048**0.5), normalized[:, 2048:]], -1)
    amu = action[tr].flatten(0, 1).mean(0)
    astd = action[tr].flatten(0, 1).std(0).clamp_min(0.03)
    x = torch.cat([state[:, None, :].expand(-1, nc, -1), (action - amu) / astd], -1)
    model = CandidateCritic(x.shape[-1], c["critic_heads"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["critic_lr"], weight_decay=0.001)
    groups = sorted({(rows[i]["task"], rows[i]["episode"]) for i in tr.tolist()})
    boot = []
    for _ in model.heads:
        sampled = torch.randint(len(groups), (len(groups),)).tolist()
        boot.append(
            torch.tensor(
                [
                    i
                    for g in sampled
                    for i in tr.tolist()
                    if (rows[i]["task"], rows[i]["episode"]) == groups[g]
                ]
            )
        )

    def objective(pred, target):
        mse = (pred - target).square().mean()
        diff = target[:, :, None] - target[:, None, :]
        mask = diff.abs() > 0.02
        rank = nn.functional.softplus(-(pred[:, :, None] - pred[:, None, :]) * diff.sign() / 0.1)
        ranking = rank[mask].mean() if mask.any() else pred.sum() * 0
        return mse + 0.1 * ranking

    best = float("inf")
    best_state = None
    best_step = 0
    history = []
    for step in range(c["critic_steps"] + 1):
        if step:
            opt.zero_grad()
            loss = sum(
                objective(model.heads[h](x[ids]).squeeze(-1), y[ids]) for h, ids in enumerate(boot)
            ) / len(boot)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                val = float(objective(model(x[va]).mean(-1), y[va]))
            history.append(dict(step=step, validation_objective=val))
            if val < best:
                best = val
                best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        estimates = model(x)
        pred = estimates.mean(-1)
        # Bootstrap spread is heuristic, not a calibrated confidence interval.
        lcb = pred - estimates.std(-1)
        selected = lcb.argmax(-1)
        gain = lcb.gather(1, selected[:, None]).squeeze(1) - lcb[:, 0]
        selected = torch.where(gain > 0.02, selected, torch.zeros_like(selected))
    old_scores = torch.stack([r["old_q"] for r in rows])
    old = old_scores.argmax(-1)
    report = dict(
        candidate_names=names,
        selected_step=best_step,
        target=target_kind,
        validation_objective=best,
        groups=len(rows),
        splits={},
        limitations=[
            "Small pilot; two continuations per candidate.",
            "Candidate critic predicts a fixed-SFT continuation return, not an optimal Q.",
            "Test returns are used once for acceptance, not checkpoint/margin selection.",
            "Bootstrap head spread is not calibrated uncertainty.",
            "No online loop or actor training is triggered by this script.",
        ],
    )
    for split, ids in splits.items():
        idx = ids.tolist()
        chosen = selected[ids]

        def ranking(scores):
            difference = y[ids, :, None] - y[ids, None, :]
            mask = torch.triu(torch.ones(nc, nc, dtype=torch.bool), diagonal=1)[None]
            mask = mask & (difference.abs() > 0.02)
            predicted = scores[ids, :, None] - scores[ids, None, :]
            correct = (predicted * difference > 0).float()
            return dict(
                non_tied_pairs=int(mask.sum()),
                accuracy=float(correct[mask].mean()) if mask.any() else None,
            )

        def metrics(which):
            got = mean_returns[ids, which]
            base = mean_returns[ids, 0]
            succ = success[ids, which].mean(-1)
            bs = success[ids, 0].mean(-1)
            pair = [
                dict(
                    key=rows[i]["key"],
                    candidate=names[int(which[j])],
                    return_difference=float(got[j] - base[j]),
                    success_difference=float(succ[j] - bs[j]),
                )
                for j, i in enumerate(idx)
            ]
            return dict(
                mean_return=float(got.mean()),
                return_gain=float((got - base).mean()),
                mean_success=float(succ.mean()),
                success_gain=float((succ - bs).mean()),
                pairs=pair,
            )

        report["splits"][split] = dict(
            groups=len(ids),
            base=metrics(torch.zeros(len(ids), dtype=torch.long)),
            old_q_selection=metrics(old[ids]),
            new_q_selection=metrics(chosen),
            oracle_candidate=metrics(y[ids].argmax(-1)),
            fixed_candidates={
                name: metrics(torch.full((len(ids),), j, dtype=torch.long))
                for j, name in enumerate(names)
            },
            old_q_ranking=ranking(old_scores),
            new_q_ranking=ranking(pred),
            per_task={
                str(task): dict(
                    groups=sum(rows[i]["task"] == task for i in idx),
                    return_gain=float(
                        torch.stack(
                            [
                                mean_returns[i, selected[i]] - mean_returns[i, 0]
                                for i in idx
                                if rows[i]["task"] == task
                            ]
                        ).mean()
                    ),
                    success_gain=float(
                        torch.stack(
                            [
                                success[i, selected[i]].mean() - success[i, 0].mean()
                                for i in idx
                                if rows[i]["task"] == task
                            ]
                        ).mean()
                    ),
                )
                for task in c["task_ids"]
            },
        )
    gates = [report["splits"][s]["new_q_selection"] for s in ["validation", "test"]]
    if target_kind == "success":
        passed = best_step > 0 and all(m["success_gain"] > 0 for m in gates)
        passed = passed and all(
            metrics["success_gain"] >= 0
            for split in ["validation", "test"]
            for metrics in report["splits"][split]["per_task"].values()
        )
        gate_rule = "Validation AND test success gain > 0; no task success regression on either split; selected step > 0. Return is secondary."
    else:
        passed = best_step > 0 and all(
            m["return_gain"] > 0.01 and m["success_gain"] >= 0 for m in gates
        )
        gate_rule = "Validation AND test mean return gain > 0.01, success does not decrease, selected step > 0."
    report["actor_gate"] = dict(
        passed=passed,
        rule=gate_rule,
        decision="teacher rollout verification required before actor training"
        if passed
        else "stop: no validated teacher improvement; do not train actor",
    )
    torch.save(
        dict(
            model=best_state,
            input_dim=x.shape[-1],
            heads=c["critic_heads"],
            state_mean=mu,
            state_std=std,
            basis=basis,
            action_mean=amu,
            action_std=astd,
            config=cfg,
            files=[str(p) for p in files],
            selected_step=best_step,
        ),
        root / "candidate_critic.pt",
    )
    (root / "candidate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "critic_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (root / "fit_status.txt").write_text("complete\n")
    print(
        json.dumps(
            dict(
                selected_step=best_step,
                gate=report["actor_gate"],
                results={
                    s: {
                        k: {n: v for n, v in m.items() if n != "pairs"}
                        for k, m in vals.items()
                        if k in ["base", "old_q_selection", "new_q_selection", "oracle_candidate"]
                    }
                    for s, vals in report["splits"].items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

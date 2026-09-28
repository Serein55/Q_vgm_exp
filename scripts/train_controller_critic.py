"""Train a nonlinear finite-action critic from controller candidate supervision."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/controller_training",
        help="Completed training data directory",
    )
    ap.add_argument("--manifest", type=Path, help="Explicit file paths and train/validation keys")
    ap.add_argument("--seed", type=int, default=7108)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--run")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--prior-residual", action="store_true")
    ap.add_argument(
        "--objective", choices=["regression", "listwise", "best_reference"], default="regression"
    )
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts" / (args.root.name + "_critic" if args.root else "controller_neural_v1")
    if args.run:
        out = root / "artifacts" / args.run
    out.mkdir(exist_ok=True)
    assert not (out / "report.json").exists()
    rows = []
    sources = [args.root]
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if manifest:
        rows = [torch.load(p, weights_only=True) for p in manifest["files"]]
        sources = []
    for source in sources:
        rows += [torch.load(p, weights_only=True) for p in sorted(source.glob("worker*/*.pt"))]
    rows.sort(key=lambda r: r["key"])
    if manifest:
        boundary = None
    elif args.root:
        configs = [json.loads(p.read_text()) for p in args.root.glob("worker*/config.json")]
        assert configs and all(c == configs[0] for c in configs)
        c = configs[0]["recovery"]
        assert len(rows) == len(c["task_ids"]) * c["episodes"] * len(c["fractions"])
        boundary = c["train_before_episode"]
    else:
        assert len(rows) == 32
        boundary = 22
    assert len({r["key"] for r in rows}) == len(rows)
    for r in rows:
        assert len(r["records"]) == 144
        assert len({(t["candidate"], t["repeat"]) for t in r["records"]}) == 144
    names = rows[0]["candidate_names"]
    assert len(names) == 9 and all(r["candidate_names"] == names for r in rows)
    x = torch.stack(
        [
            torch.cat(
                [
                    r["mean_pool"],
                    r["observation"]["observation/state"],
                    r["candidates"][0, :5, :7].flatten(),
                ]
            )
            for r in rows
        ]
    ).float()
    y = torch.tensor(
        [
            [
                [
                    next(
                        x["success"]
                        for x in r["records"]
                        if x["candidate"] == n and x["repeat"] == k
                    )
                    for k in range(16)
                ]
                for n in names
            ]
            for r in rows
        ],
        dtype=torch.float32,
    )
    target = y.mean(-1)
    adv = target - target[:, :1]
    train_keys = (
        set(manifest["train_keys"])
        if manifest
        else {r["key"] for r in rows if r["episode"] < boundary}
    )
    val_keys = (
        set(manifest["validation_keys"])
        if manifest
        else {r["key"] for r in rows if r["episode"] >= boundary}
    )
    assert train_keys and val_keys and not train_keys & val_keys
    assert train_keys | val_keys == {r["key"] for r in rows}
    assert not (
        {(r["task"], r["episode"]) for r in rows if r["key"] in train_keys}
        & {(r["task"], r["episode"]) for r in rows if r["key"] in val_keys}
    )
    tr = torch.tensor([i for i, r in enumerate(rows) if r["key"] in train_keys])
    va = torch.tensor([i for i, r in enumerate(rows) if r["key"] in val_keys])
    mu = x[tr].mean(0)
    std = x[tr].std(0).clamp_min(0.05)
    x = (x - mu) / std
    model = nn.Sequential(
        nn.Linear(x.shape[-1], 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
        nn.Linear(64, 9),
    )
    if args.dropout:
        model = nn.Sequential(
            nn.Linear(x.shape[-1], 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 9),
        )
    prior = (
        torch.logit(target[tr].mean(0).clamp(0.01, 0.99)) if args.prior_residual else torch.zeros(9)
    )
    if args.prior_residual:
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    reference = int(target[tr].mean(0).argmax())
    ref_target = target - target[:, reference : reference + 1]
    history = []
    best = float("inf")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                logits_all = model(x) + prior
                p = logits_all.sigmoid()
                pa = p - p[:, :1]
                val = float((pa[va, 1:] - adv[va, 1:]).square().mean())
                train = float((pa[tr, 1:] - adv[tr, 1:]).square().mean())
            history.append({"step": step, "train_adv_mse": train, "validation_adv_mse": val})
            criterion = val
            if args.objective == "listwise":
                criterion = float(
                    -(torch.softmax(target[va] / 0.1, -1) * torch.log_softmax(logits_all[va], -1))
                    .sum(-1)
                    .mean()
                )
            if args.objective == "best_reference":
                ref_pred = p - p[:, reference : reference + 1]
                criterion = float((ref_pred[va] - ref_target[va]).square().mean())
            history[-1]["selection_loss"] = criterion
            if criterion < best:
                best = criterion
                best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == args.steps:
            break
        model.train()
        opt.zero_grad()
        residual = model(x[tr])
        logit = residual + prior
        prob = logit.sigmoid()
        pred_adv = prob - prob[:, :1]
        # Shared state predicts baseline probability and nine conditional action values.
        # Relative loss makes action differences matter, BCE keeps values calibrated to labels.
        loss = (
            pred_adv[:, 1:] - adv[tr, 1:]
        ).square().mean() + 0.1 * nn.functional.binary_cross_entropy_with_logits(logit, target[tr])
        if args.objective == "listwise":
            loss = -(torch.softmax(target[tr] / 0.1, -1) * torch.log_softmax(logit, -1)).sum(
                -1
            ).mean() + 0.1 * nn.functional.binary_cross_entropy_with_logits(logit, target[tr])
        if args.objective == "best_reference":
            ref_pred = prob - prob[:, reference : reference + 1]
            loss = (
                ref_pred - ref_target[tr]
            ).square().mean() + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                logit, target[tr]
            )
        if args.prior_residual:
            loss = loss + 0.01 * residual.square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report = {
        "selected_step": best_step,
        "objective": args.objective,
        "training_reference": names[reference] if args.objective == "best_reference" else "base",
        "seed": args.seed,
        "dropout": args.dropout,
        "prior_residual": args.prior_residual,
        "train_episodes": sorted({r["episode"] for r in rows if r["key"] in train_keys}),
        "validation_episodes": sorted({r["episode"] for r in rows if r["key"] in val_keys}),
        "manifest": str(args.manifest) if args.manifest else None,
        "train_keys": sorted(train_keys),
        "validation_keys": sorted(val_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
        "models": {},
        "note": "Development training only. Episode split and data provenance are recorded in this report. Nine named candidate outputs; no arbitrary-action capability claimed. Final-step train fit diagnoses capacity, not generalization. Threshold fixed .02.",
    }
    for label, state in [("validation_selected", best_state), ("final_fit", final_state)]:
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            p = (model(x) + prior).sigmoid()
            pa = p - p[:, :1]
        pick = pa.argmax(-1)
        pick[pa.max(-1).values <= 0.02] = 0
        chosen = y[torch.arange(len(rows)), pick]
        metrics = {}
        for split, ids in [("train", tr), ("validation", va)]:
            diff = adv[ids, :, None] - adv[ids, None, :]
            pd = pa[ids, :, None] - pa[ids, None, :]
            mask = diff.abs() >= 0.125
            metrics[split] = {
                "base": int(y[ids, 0].sum()),
                "selected": int(chosen[ids].sum()),
                "trials": len(ids) * 16,
                "adv_mse": float((pa[ids, 1:] - adv[ids, 1:]).square().mean()),
                "ranking_acc_gap_at_least_2_of_16": float(
                    ((pd[mask] * diff[mask]) > 0).float().mean()
                )
                if mask.any()
                else None,
                "empirical_oracle": int(target[ids].max(-1).values.sum() * 16),
                "choices": dict(zip(names, torch.bincount(pick[ids], minlength=9).tolist())),
            }
        report["models"][label] = metrics
    torch.save(
        {
            "best_model": best_state,
            "prior_logits": prior,
            "objective": args.objective,
            "prior_residual": args.prior_residual,
            "final_model": final_state,
            "input_dim": x.shape[-1],
            "dropout": args.dropout,
            "seed": args.seed,
            "state_mean": mu,
            "state_std": std,
            "candidate_names": names,
            "keys": [r["key"] for r in rows],
        },
        out / "critic.pt",
    )
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

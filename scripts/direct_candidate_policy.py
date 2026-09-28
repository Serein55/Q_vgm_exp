"""Direct full-information policy optimization, without a learned Q or actor gradient."""

import json

import numpy as np
import torch
from torch import nn


def experiment(root, z, side, y, episodes, tr, va, names, image_features=False):
    out = (
        root / "artifacts" / ("direct_image_policy" if image_features else "direct_policy_guarded")
    )
    out.mkdir(exist_ok=False)
    steps = 300
    seeds = [7108, 7119, 7130]

    def fit(ids, hidden, beta):
        assert not set(ids) & set(va)
        task_means = np.zeros((4, z.shape[1]), np.float32)
        if image_features:
            for ti in range(4):
                task_means[ti] = z[ids][side[ids, -4 + ti] == 1].mean(0)
        centered = z - side[:, -4:] @ task_means
        mu = centered[ids].mean(0).astype(np.float32)
        _, _, vh = torch.linalg.svd(
            torch.from_numpy((centered[ids] - mu).astype(np.float32)), full_matrices=False
        )
        projection = vh[:8].numpy().T
        features = np.concatenate([side, (centered - mu) @ projection], 1).astype(np.float32)
        fm, fs = features[ids].mean(0), features[ids].std(0).clip(0.05)
        x = torch.from_numpy((features - fm) / fs)
        target = torch.tensor(y[ids], dtype=torch.float32)
        prior_log = torch.log_softmax(target.mean(0) / 0.1, 0)
        advantage = target - target[:, target.mean(0).argmax() : target.mean(0).argmax() + 1]
        outputs, heads = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            model = (
                nn.Sequential(nn.Linear(x.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 9))
                if hidden
                else nn.Sequential(nn.Linear(x.shape[1], 9))
            )
            nn.init.zeros_(model[-1].weight)
            nn.init.zeros_(model[-1].bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            for _ in range(steps):
                logp = torch.log_softmax(model(x[ids]) + prior_log, -1)
                prob = logp.exp()
                # Every candidate has observed paired outcomes: no propensity correction.
                loss = (
                    -(prob * advantage).sum(-1).mean()
                    + beta * (prob * (logp - prior_log)).sum(-1).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                outputs.append(torch.softmax(model(x) + prior_log, -1).numpy())
            heads.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return np.mean(outputs, 0), dict(
            heads=heads,
            hidden=hidden,
            beta=beta,
            steps=steps,
            seeds=seeds,
            z_mean=torch.from_numpy(mu),
            image_features=image_features,
            task_means=torch.from_numpy(task_means),
            projection=torch.from_numpy(projection),
            feature_mean=torch.from_numpy(fm),
            feature_std=torch.from_numpy(fs),
            prior_log=prior_log,
            reference=int(target.mean(0).argmax()),
            candidate_names=names,
            side_features="proprio8, base_action35, actual_prefix_chunks1, task_onehot[1,5,6,9]4",
            inference="ensemble mean probabilities then argmax; no base fallback threshold",
        )

    groups = sorted(set(episodes[tr]))
    records = []
    oof_cache = {}
    for hidden in [0, 32]:
        for beta in [0.03, 0.1]:
            pred = np.empty((len(tr), 9))
            fixed = np.zeros(len(tr), int)
            for fold in range(4):
                mask = np.isin(episodes[tr], groups[fold::4])
                a, b = tr[~mask], tr[mask]
                assert not set(episodes[a]) & set(episodes[b])
                probabilities, model = fit(a, hidden, beta)
                pred[mask], fixed[mask] = probabilities[b], model["reference"]
            record = dict(
                hidden=hidden,
                beta=beta,
                oof_selected=int(round(y[tr, pred.argmax(1)].sum() * 16)),
                oof_fixed=int(round(y[tr, fixed].sum() * 16)),
                stochastic_expected_success=float((y[tr] * pred).sum() * 16),
            )
            oof_cache[(hidden, beta)] = (pred.argmax(1).copy(), fixed.copy())
            records.append(record)
            print(record, flush=True)
    best = max(records, key=lambda r: r["oof_selected"])
    report = dict(
        protocol="Direct expected success + KL to training-only action prior. Four episode-group folds, training-only PCA/scaling/prior. Config chosen by training OOF; development is reused exploratory data, not independent confirmation.",
        training_cv=records,
        selected=best,
        training_trials=len(tr) * 16,
    )
    # Predeclared rule: enable a task only if the training episode-bootstrap
    # fifth percentile of its paired OOF gain is positive. No development labels.
    oof_pick, oof_fixed = oof_cache[(best["hidden"], best["beta"])]
    gains = y[tr, oof_pick] - y[tr, oof_fixed]
    gates = {}
    for ti, task in enumerate([1, 5, 6, 9]):
        mask = side[tr, -4 + ti] == 1
        cluster = np.array(
            [gains[mask & (episodes[tr] == e)].mean() for e in np.unique(episodes[tr][mask])]
        )
        rng = np.random.default_rng(7108)
        lower = float(
            np.quantile(rng.choice(cluster, (10000, len(cluster)), replace=True).mean(1), 0.05)
        )
        gates[str(task)] = dict(
            training_oof_gain=float(cluster.mean()), lower5=lower, enabled=lower > 0
        )
    report["task_gates"] = gates
    report["gate_note"] = (
        "Training OOF also selects hyperparameters; bootstrap is a heuristic gate, not a calibrated guarantee."
    )
    if best["oof_selected"] > best["oof_fixed"]:
        pred, model = fit(tr, best["hidden"], best["beta"])
        pick = pred[va].argmax(1)
        report["development"] = dict(
            selected=int(round(y[va, pick].sum() * 16)),
            fixed=int(round(y[va, model["reference"]].sum() * 16)),
            base=int(round(y[va, 0].sum() * 16)),
            trials=len(va) * 16,
            reference=names[model["reference"]],
            choices={n: int((pick == i).sum()) for i, n in enumerate(names)},
            per_task={
                str(t): dict(
                    selected=int(round(y[va, pick][side[va, -4 + ti] == 1].sum() * 16)),
                    fixed=int(round(y[va, model["reference"]][side[va, -4 + ti] == 1].sum() * 16)),
                )
                for ti, t in enumerate([1, 5, 6, 9])
            },
        )
        gated_pick = pick.copy()
        for ti, task in enumerate([1, 5, 6, 9]):
            if not gates[str(task)]["enabled"]:
                gated_pick[side[va, -4 + ti] == 1] = model["reference"]
        report["guarded_development"] = dict(
            selected=int(round(y[va, gated_pick].sum() * 16)),
            fixed=report["development"]["fixed"],
            trials=len(va) * 16,
            choices={n: int((gated_pick == i).sum()) for i, n in enumerate(names)},
        )
        model["task_gates"] = gates
        model["train_keys"] = json.loads(
            (root / "artifacts/controller_coverage/training_manifest.json").read_text()
        )["train_keys"]
        torch.save(model, out / "policy.pt")
    else:
        report["development"] = "Skipped: no training OOF improvement"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

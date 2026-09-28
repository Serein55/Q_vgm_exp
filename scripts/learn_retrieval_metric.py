"""Train-only supervised retrieval distance; preserve the frozen original policy."""

import json

import numpy as np
import torch
from retrieval_policy import select_actions


def experiment(root, z, side, y, episodes, tr, va, names):
    out = root / "artifacts/retrieval_metric"
    out.mkdir(exist_ok=False)
    group = side[:, -4:].argmax(1) * 2 + (side[:, 43] > 0)

    def transform(ids, penalty, blend):
        assert not set(ids) & set(va)
        means = np.stack([z[ids[group[ids] == g]].mean(0) for g in range(8)])
        centered = (z - means[group]).astype(np.float32)
        _, _, vh = torch.linalg.svd(torch.from_numpy(centered[ids]), full_matrices=False)
        projection = vh[:16].numpy().T
        features = centered @ projection
        scale = features[ids].std(0).clip(0.05)
        features = features / scale
        ref = int(y[ids].mean(0).argmax())
        target = y[ids] - y[ids, ref : ref + 1]
        for g in range(8):
            mask = group[ids] == g
            target[mask] -= target[mask].mean(0)
        weights = np.linalg.solve(
            features[ids].T @ features[ids] + penalty * np.eye(16), features[ids].T @ target
        )
        learned = features @ weights
        # Normalize each component only on the relevant training task/stage.
        raw_scale, learned_scale = [], []
        for g in range(8):
            a = z[ids[group[ids] == g]].astype(float)
            b = learned[ids[group[ids] == g]]
            raw_scale.append(
                max(float(np.mean(np.sum((a[:, None] - a[None, :]) ** 2, axis=-1))), 1e-8)
            )
            learned_scale.append(
                max(float(np.mean(np.sum((b[:, None] - b[None, :]) ** 2, axis=-1))), 1e-8)
            )
        mapped = np.concatenate(
            [
                z / np.sqrt(np.array(raw_scale)[group, None]),
                np.sqrt(blend) * learned / np.sqrt(np.array(learned_scale)[group, None]),
            ],
            axis=1,
        )
        checkpoint = dict(
            means=torch.from_numpy(means),
            projection=torch.from_numpy(projection),
            scale=torch.from_numpy(scale),
            weights=torch.from_numpy(weights),
            raw_scale=raw_scale,
            learned_scale=learned_scale,
            blend=blend,
            penalty=penalty,
        )
        return mapped, checkpoint

    def predict(a, b, penalty, blend):
        mapped, model = transform(a, penalty, blend)
        pick, ref, _ = select_actions(mapped[a], side[a], y[a], mapped[b], side[b], "visual", 4)
        return pick, ref, model, mapped

    groups = sorted(set(episodes[tr]))
    records = []
    # Original frozen distance is a training-CV control, not merely fixed action.
    basepick = np.empty(len(tr), int)
    fixed = np.empty(len(tr), int)
    folds = []
    for fold in range(4):
        mask = np.isin(episodes[tr], groups[fold::4])
        a, b = tr[~mask], tr[mask]
        assert not set(episodes[a]) & set(episodes[b])
        folds.append((mask, a, b))
        basepick[mask], fixed[mask], _ = select_actions(
            z[a], side[a], y[a], z[b], side[b], "visual", 4
        )
    baseline = int(round(y[tr, basepick].sum() * 16))
    for penalty in [1.0, 10.0, 100.0]:
        for blend in [0.25, 1.0]:
            picks = np.empty(len(tr), int)
            for mask, a, b in folds:
                picks[mask], _, _, _ = predict(a, b, penalty, blend)
            record = dict(
                penalty=penalty, blend=blend, oof_selected=int(round(y[tr, picks].sum() * 16))
            )
            records.append(record)
            print(record, flush=True)
    best = max(records, key=lambda r: r["oof_selected"])
    report = dict(
        protocol="Four training-episode folds; task/stage centering and PCA16 trained within fold; ridge maps visual features to relative-outcome variation. Blend learned distance with original. Four neighbors, prior2, threshold.02 unchanged. Confirmation IDs45–49 never loaded.",
        training_cv=records,
        selected=best,
        original_retrieval_oof=baseline,
        fixed_oof=int(round(y[tr, fixed].sum() * 16)),
        training_trials=len(tr) * 16,
    )
    if best["oof_selected"] > baseline:
        pick, ref, model, mapped = predict(tr, va, best["penalty"], best["blend"])
        report["development"] = dict(
            selected=int(round(y[va, pick].sum() * 16)),
            fixed=int(round(y[va, ref].sum() * 16)),
            trials=len(va) * 16,
            original_retrieval=309,
        )
        model.update(
            mapped_train=torch.from_numpy(mapped[tr]),
            side=torch.from_numpy(side[tr]),
            success_rate=torch.from_numpy(y[tr]),
            candidate_names=names,
        )
        torch.save(model, out / "metric.pt")
    else:
        report["development"] = "Skipped: failed to beat original retrieval on training OOF"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def support_experiment(root, z, side, y, episodes, tr, va, names):
    out = root / "artifacts/retrieval_support_gate"
    out.mkdir(exist_ok=False)
    group = side[:, -4:].argmax(1) * 2 + (side[:, 43] > 0)

    def predict(a, b, beta, reject_far):
        assert not set(a) & set(va)
        picks, ref, donors = select_actions(z[a], side[a], y[a], z[b], side[b], "visual", 4)
        for qi, q in enumerate(b):
            if picks[qi] == ref:
                continue
            pool = a[group[a] == group[q]]
            bank = z[pool].astype(float)
            pair = ((bank[:, None] - bank[None, :]) ** 2).mean(-1)
            distance = ((bank - z[q]) ** 2).mean(1)
            if reject_far:
                # Calibrate nearest-neighbor support without borrowing same-episode
                # samples. No query statistics or validation labels set the cutoff.
                pair[episodes[pool, None] == episodes[pool][None, :]] = np.inf
                limit = np.quantile(pair.min(1), 0.9)
                if distance.min() > limit:
                    picks[qi] = ref
                    continue
            ids = a[donors[qi]]
            d = ((z[ids].astype(float) - z[q]) ** 2).mean(1)
            w = np.exp(-d / max(float(d.mean()), 1e-8))
            w /= w.sum()
            delta = y[ids, picks[qi]] - y[ids, ref]
            mean = float(w @ delta)
            pool_gain = float((y[pool, picks[qi]] - y[pool, ref]).mean())
            estimate = (4 * mean + 2 * pool_gain) / 6
            # Heuristic dispersion penalty across donor states, not a certified CI.
            variance = float(w @ ((delta - mean) ** 2)) / max(1 - float(w @ w), 1e-8)
            se = np.sqrt(variance * float(w @ w))
            if estimate - beta * se <= 0.02:
                picks[qi] = ref
        return picks, ref

    groups = sorted(set(episodes[tr]))
    records = []
    for beta in [0.0, 0.25, 0.5, 1.0]:
        for reject_far in [False, True]:
            picks = np.empty(len(tr), int)
            for fold in range(4):
                mask = np.isin(episodes[tr], groups[fold::4])
                a, b = tr[~mask], tr[mask]
                assert not set(episodes[a]) & set(episodes[b])
                picks[mask], _ = predict(a, b, beta, reject_far)
            record = dict(
                beta=beta, reject_far=reject_far, oof_selected=int(round(y[tr, picks].sum() * 16))
            )
            records.append(record)
            print(record, flush=True)
    best = max(records, key=lambda r: r["oof_selected"])
    baseline = records[0]["oof_selected"]
    assert baseline == 1175
    report = dict(
        protocol="Training-only episode CV; support quantile .9 calibrated with leave-episode-out donors, weighted inter-donor dispersion penalty. Baseline raw retrieval included. Confirmation data never loaded. Dispersion is not calibrated uncertainty.",
        training_cv=records,
        selected=best,
        original_retrieval_oof=baseline,
    )
    if best["oof_selected"] > baseline:
        picks, ref = predict(tr, va, best["beta"], best["reject_far"])
        report["development"] = dict(
            selected=int(round(y[va, picks].sum() * 16)),
            fixed=int(round(y[va, ref].sum() * 16)),
            original_retrieval=309,
            trials=len(va) * 16,
        )
    else:
        report["development"] = "Skipped: no training OOF gain"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

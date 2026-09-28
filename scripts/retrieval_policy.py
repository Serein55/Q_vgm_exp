"""Same-task, same-stage local action values from stored paired candidate labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def select_actions(train_z, train_side, train_y, query_z, query_side, metric, neighbors):
    """Inference receives training labels only; query labels cannot be accessed."""
    z = np.concatenate([train_z, query_z])
    side = np.concatenate([train_side, query_side])
    y = train_y
    train = np.arange(len(train_z))
    query = np.arange(len(train_z), len(z))
    tasks = side[:, -4:].argmax(1)
    stages = (side[:, 43] > 0).astype(int)
    ref = int(y[train].mean(0).argmax())
    picks = []
    support = []
    for q in query:
        pool = train[(tasks[train] == tasks[q]) & (stages[train] == stages[q])]
        assert len(pool) >= neighbors
        blocks = (
            [z, side[:, :43]]
            if metric == "combined"
            else [z if metric == "visual" else side[:, :43]]
        )
        distance = np.zeros(len(pool))
        for features in blocks:
            # Single train-only scale per block; no test-derived normalization.
            a = features[pool].astype(float)
            scale = np.mean((a[:, None] - a[None, :]) ** 2)
            distance += ((a - features[q]) ** 2).mean(1) / max(scale, 1e-8)
        closest = np.argsort(distance)[:neighbors]
        ids = pool[closest]
        weight = np.exp(-distance[closest] / max(float(distance[closest].mean()), 1e-8))
        weight /= weight.sum()
        # Small prior effective sample size damps noisy local label averages.
        estimate = (neighbors * (weight[:, None] * y[ids]).sum(0) + 2 * y[pool].mean(0)) / (
            neighbors + 2
        )
        pick = int(estimate.argmax())
        if estimate[pick] - estimate[ref] <= 0.02:
            pick = ref
        picks.append(pick)
        support.append(ids.tolist())
    return np.array(picks), ref, support


class FrozenRetrievalPolicy:
    """Inference-only checkpoint interface, with optional immutable hash contract."""

    def __init__(self, checkpoint, expected_sha256=None):
        path = Path(checkpoint)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("Retrieval checkpoint hash mismatch")
        self.ck = torch.load(path, weights_only=True, map_location="cpu")
        if self.ck["prior_strength"] != 2 or self.ck["threshold"] != 0.02:
            raise ValueError("Unsupported policy constants")
        self.z = self.ck["z"].numpy()
        self.side = self.ck["side"].numpy()
        self.y = self.ck["success_rate"].numpy()

    def choose(self, mean_pool, proprio, base_action, task, prefix_chunks):
        if task not in [1, 5, 6, 9]:
            raise ValueError("Retrieval index only covers tasks 1/5/6/9")
        side = np.concatenate(
            [
                np.asarray(proprio).reshape(8),
                np.asarray(base_action).reshape(35),
                [prefix_chunks],
                [float(task == t) for t in [1, 5, 6, 9]],
            ]
        )[None]
        pick, reference, donors = select_actions(
            self.z,
            self.side,
            self.y,
            np.asarray(mean_pool).reshape(1, -1),
            side,
            self.ck["metric"],
            self.ck["neighbors"],
        )
        return int(pick[0]), reference, donors[0]


def experiment(root, z, side, y, episodes, tr, va, names):
    out = root / "artifacts/retrieval_policy"
    out.mkdir(exist_ok=False)
    tasks = side[:, -4:].argmax(1)

    def predict(train, query, metric, neighbors):
        assert not set(train) & set(va)
        pick, ref, support = select_actions(
            z[train], side[train], y[train], z[query], side[query], metric, neighbors
        )
        return pick, ref, [train[ids].tolist() for ids in support]

    groups = sorted(set(episodes[tr]))
    records = []
    for metric in ["visual", "proprio_action", "combined"]:
        for k in [2, 4, 8]:
            selected = np.zeros(len(tr), int)
            fixed = np.zeros(len(tr), int)
            for fold in range(4):
                mask = np.isin(episodes[tr], groups[fold::4])
                a, b = tr[~mask], tr[mask]
                assert not set(episodes[a]) & set(episodes[b])
                selected[mask], fixed[mask], _ = predict(a, b, metric, k)
            record = dict(
                metric=metric,
                neighbors=k,
                oof_selected=int(round(y[tr, selected].sum() * 16)),
                oof_fixed=int(round(y[tr, fixed].sum() * 16)),
            )
            records.append(record)
            print(record, flush=True)
    best = max(records, key=lambda r: r["oof_selected"])
    report = dict(
        protocol="Same task/stage retrieval; four episode-group folds select metric and neighbor count; prior strength 2, threshold .02 fixed. No new rollouts. Development reused, not independent.",
        training_cv=records,
        selected=best,
        training_trials=len(tr) * 16,
    )
    if best["oof_selected"] > best["oof_fixed"]:
        pick, ref, support = predict(tr, va, best["metric"], best["neighbors"])
        torch.save(
            dict(
                z=torch.from_numpy(z[tr]),
                side=torch.from_numpy(side[tr]),
                success_rate=torch.from_numpy(y[tr]),
                metric=best["metric"],
                neighbors=best["neighbors"],
                candidate_names=names,
                train_ids=tr.tolist(),
                prior_strength=2,
                threshold=0.02,
            ),
            out / "policy.pt",
        )
        report["development"] = dict(
            selected=int(round(y[va, pick].sum() * 16)),
            fixed=int(round(y[va, ref].sum() * 16)),
            trials=len(va) * 16,
            per_task={
                str(t): dict(
                    selected=int(round(y[va, pick][tasks[va] == i].sum() * 16)),
                    fixed=int(round(y[va, ref][tasks[va] == i].sum() * 16)),
                )
                for i, t in enumerate([1, 5, 6, 9])
            },
        )
        (out / "selection.json").write_text(
            json.dumps(
                dict(
                    query_ids=va.tolist(),
                    donor_ids=support,
                    candidates=pick.tolist(),
                    reference=names[ref],
                ),
                indent=2,
            )
            + "\n"
        )
    else:
        report["development"] = "Skipped: no training OOF gain"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

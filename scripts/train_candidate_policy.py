"""Honest shallow policy trees; choose actions by return, not value MSE."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fit_tree(x, y, ids, depth, features):
    # Every state supplies labels for every action, so policy value is observed.
    total = y[ids].sum(0)
    best = float(total.max())
    split = None
    if depth:
        for j in features:
            for cut in np.unique(np.quantile(x[ids, j], [0.25, 0.5, 0.75])):
                mask = x[ids, j] <= cut
                if min(mask.sum(), (~mask).sum()) < 8:
                    continue
                left = y[ids[mask]].sum(0)
                gain = float(left.max() + (total - left).max())
                if gain > best + 0.01:
                    best, split = gain, (int(j), float(cut), mask)
    if split is None:
        return {"value": None}
    j, cut, mask = split
    return {
        "feature": j,
        "cut": cut,
        "left": fit_tree(x, y, ids[mask], depth - 1, features),
        "right": fit_tree(x, y, ids[~mask], depth - 1, features),
    }


def calibrate(tree, x, y, ids, prior):
    if "feature" not in tree:
        # Independent episodes estimate leaf values; shrink sparse leaves.
        tree["value"] = ((y[ids].sum(0) + 8 * prior) / (len(ids) + 8)).tolist()
        return
    mask = x[ids, tree["feature"]] <= tree["cut"]
    calibrate(tree["left"], x, y, ids[mask], prior)
    calibrate(tree["right"], x, y, ids[~mask], prior)


def predict(tree, x):
    if "feature" not in tree:
        return np.tile(tree["value"], (len(x), 1))
    mask = x[:, tree["feature"]] <= tree["cut"]
    out = np.empty((len(x), 9))
    out[mask] = predict(tree["left"], x[mask])
    out[~mask] = predict(tree["right"], x[~mask])
    return out


def fit(z, side, y, episodes, use_vision):
    # Fit representation only on this fold's training states.
    mu = z.mean(0)
    _, _, vh = torch.linalg.svd(torch.from_numpy(z - mu), full_matrices=False)
    projection = vh[:8].numpy().T
    x = np.concatenate([side, (z - mu) @ projection], 1) if use_vision else side
    rng = np.random.default_rng(7108)
    trees = []
    groups = np.unique(episodes)
    for _ in range(64):
        shuffled = rng.permutation(groups)
        structure = np.flatnonzero(np.isin(episodes, shuffled[: len(groups) // 2]))
        estimate = np.flatnonzero(~np.isin(episodes, shuffled[: len(groups) // 2]))
        assert not set(episodes[structure]) & set(episodes[estimate])
        features = rng.choice(x.shape[1], max(1, x.shape[1] * 2 // 3), replace=False)
        tree = fit_tree(x, y, structure, 2, features)
        calibrate(tree, x, y, estimate, y[estimate].mean(0))
        trees.append(tree)
    return dict(
        trees=trees,
        mu=mu.tolist(),
        projection=projection.tolist(),
        vision=use_vision,
        reference=int(y.mean(0).argmax()),
    )


def score(model, z, side):
    x = (
        np.concatenate([side, (z - np.array(model["mu"])) @ np.array(model["projection"])], 1)
        if model["vision"]
        else side
    )
    return np.mean([predict(t, x) for t in model["trees"]], 0)


def choose(p, reference):
    pick = p.argmax(1)
    improvement = p[np.arange(len(p)), pick] - p[:, reference]
    pick[improvement <= 0.02] = reference
    return pick


def linear_experiment(root, z, side, y, episodes, tr, va, names):
    """Train-only group CV selects regularization and feature family by policy value."""

    def train(a, vision, penalty):
        mu = z[a].mean(0)
        _, _, vh = torch.linalg.svd(torch.from_numpy(z[a] - mu), full_matrices=False)
        proj = vh[:8].numpy().T
        f = np.concatenate([side, (z - mu) @ proj], 1) if vision else side
        fm, fs = f[a].mean(0), f[a].std(0).clip(0.05)
        f = np.concatenate([np.ones((len(f), 1)), (f - fm) / fs], 1)
        reference = int(y[a].mean(0).argmax())
        target = y[a] - y[a, reference : reference + 1]
        reg = np.eye(f.shape[1]) * penalty
        reg[0, 0] = 1e-8
        weights = np.linalg.solve(f[a].T @ f[a] + reg, f[a].T @ target)
        return f @ weights, dict(
            vision=vision,
            penalty=penalty,
            reference=reference,
            mu=mu.tolist(),
            projection=proj.tolist(),
            feature_mean=fm.tolist(),
            feature_std=fs.tolist(),
            weights=weights.tolist(),
        )

    groups = sorted(set(episodes[tr]))
    records = []
    for vision in [False, True]:
        for penalty in [1.0, 10.0, 100.0, 1000.0]:
            picks, fixed = np.zeros(len(tr), int), np.zeros(len(tr), int)
            for fold in range(4):
                mask = np.isin(episodes[tr], groups[fold::4])
                a, b = tr[~mask], tr[mask]
                pred, model = train(a, vision, penalty)
                picks[mask] = choose(pred[b], model["reference"])
                fixed[mask] = model["reference"]
            records.append(
                dict(
                    vision=vision,
                    penalty=penalty,
                    oof_selected=int(round(y[tr, picks].sum() * 16)),
                    oof_fixed=int(round(y[tr, fixed].sum() * 16)),
                )
            )
    # If all candidates lose, preserve fixed action; do not inspect development outcomes.
    best = max(records, key=lambda r: r["oof_selected"])
    report = dict(
        training_cv=records,
        selected=best,
        threshold=0.02,
        note="Exploratory four-fold episode CV; choosing among fits biases OOF estimate. No independent confirmation.",
    )
    out = root / "artifacts/policy_linear"
    out.mkdir(exist_ok=False)
    if best["oof_selected"] > best["oof_fixed"]:
        pred, model = train(tr, best["vision"], best["penalty"])
        pick = choose(pred[va], model["reference"])
        report["development"] = dict(
            selected=int(round(y[va, pick].sum() * 16)),
            fixed=int(round(y[va, model["reference"]].sum() * 16)),
            trials=len(va) * 16,
            reference=names[model["reference"]],
            choices={n: int((pick == i).sum()) for i, n in enumerate(names)},
        )
        (out / "model.json").write_text(json.dumps(model))
    else:
        report["development"] = "Skipped: no positive training OOF improvement"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "forest",
            "linear",
            "direct",
            "direct_image",
            "retrieval",
            "learned_retrieval",
            "guarded_retrieval",
        ],
        default="linear",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/controller_coverage/training_manifest.json").read_text()
    )
    rows = [torch.load(p, weights_only=True) for p in manifest["files"]]
    names = rows[0]["candidate_names"]
    z = np.stack([r["mean_pool"].numpy() for r in rows]).astype(np.float32)
    side = np.stack(
        [
            np.concatenate(
                [
                    r["observation"]["observation/state"].numpy(),
                    r["candidates"][0, :5, :7].numpy().flatten(),
                    [r["actual_prefix_chunks"]],
                    [float(r["task"] == t) for t in [1, 5, 6, 9]],
                ]
            )
            for r in rows
        ]
    )
    y = np.array(
        [
            [sum(t["success"] for t in r["records"] if t["candidate"] == n) / 16 for n in names]
            for r in rows
        ]
    )
    episodes = np.array([r["episode"] for r in rows])
    tr = np.array([i for i, r in enumerate(rows) if r["key"] in manifest["train_keys"]])
    va = np.array([i for i, r in enumerate(rows) if r["key"] in manifest["validation_keys"]])
    if args.method == "guarded_retrieval":
        from learn_retrieval_metric import support_experiment

        return support_experiment(root, z, side, y, episodes, tr, va, names)
    if args.method == "learned_retrieval":
        from learn_retrieval_metric import experiment

        return experiment(root, z, side, y, episodes, tr, va, names)
    if args.method == "retrieval":
        from retrieval_policy import experiment

        return experiment(root, z, side, y, episodes, tr, va, names)
    if args.method in ["direct", "direct_image"]:
        if args.method == "direct_image":
            z = np.stack(
                [
                    torch.cat(
                        [
                            torch.nn.functional.interpolate(
                                r["observation"][key].permute(2, 0, 1)[None].float() / 255,
                                size=(32, 32),
                                mode="area",
                            ).flatten()
                            for key in ["observation/image", "observation/wrist_image"]
                        ]
                    ).numpy()
                    for r in rows
                ]
            )
        from direct_candidate_policy import experiment

        return experiment(
            root, z, side, y, episodes, tr, va, names, image_features=args.method == "direct_image"
        )
    if args.method == "linear":
        return linear_experiment(root, z, side, y, episodes, tr, va, names)
    groups = sorted(set(episodes[tr]))
    out = root / "artifacts/policy_forest"
    out.mkdir(exist_ok=False)
    report = {
        "protocol": "64 honest trees, depth 2, structure leaf >=8, calibration prior strength 8, fixed switch margin .02; 4 folds grouped by episode across all tasks. Development read after full protocol fixed. No environment interactions.",
        "methods": {},
    }
    for vision in [False, True]:
        name = "vision" if vision else "proprio_action_task_control"
        oof, fixed = np.zeros(len(tr), dtype=int), np.zeros(len(tr), dtype=int)
        for fold in range(4):
            mask = np.isin(episodes[tr], groups[fold::4])
            a, b = tr[~mask], tr[mask]
            model = fit(z[a], side[a], y[a], episodes[a], vision)
            oof[mask] = choose(score(model, z[b], side[b]), model["reference"])
            fixed[mask] = model["reference"]
        model = fit(z[tr], side[tr], y[tr], episodes[tr], vision)
        pick = choose(score(model, z[va], side[va]), model["reference"])
        report["methods"][name] = dict(
            training_oof_selected=int(round(y[tr, oof].sum() * 16)),
            training_oof_fixed=int(round(y[tr, fixed].sum() * 16)),
            training_trials=len(tr) * 16,
            development_selected=int(round(y[va, pick].sum() * 16)),
            development_fixed=int(round(y[va, model["reference"]].sum() * 16)),
            development_base=int(round(y[va, 0].sum() * 16)),
            development_trials=len(va) * 16,
            reference=names[model["reference"]],
            choices={n: int((pick == i).sum()) for i, n in enumerate(names)},
            per_task={
                str(t): dict(
                    selected=int(round(y[va, pick][[rows[i]["task"] == t for i in va]].sum() * 16)),
                    fixed=int(
                        round(
                            y[va, model["reference"]][[rows[i]["task"] == t for i in va]].sum() * 16
                        )
                    ),
                )
                for t in [1, 5, 6, 9]
            },
        )
        (out / f"{name}.json").write_text(json.dumps(model))
        print(name, report["methods"][name], flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

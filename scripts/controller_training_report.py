"""Per-task development diagnostics for trained controller critics and ensemble."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--prefix", default="controller_coverage")
    args = ap.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1] / "artifacts"
    prefix = args.prefix
    out = root / f"{prefix}_summary"
    out.mkdir(exist_ok=True)
    manifest_data = json.loads(args.manifest.read_text())
    rows = [torch.load(p, weights_only=True) for p in manifest_data["files"]]
    count = len(rows)
    names = rows[0]["candidate_names"]
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
    y = np.array(
        [
            [
                [
                    next(
                        t["success"]
                        for t in r["records"]
                        if t["candidate"] == n and t["repeat"] == k
                    )
                    for k in range(16)
                ]
                for n in names
            ]
            for r in rows
        ],
        float,
    )
    tr = [i for i, r in enumerate(rows) if r["episode"] < 23]
    va = [i for i, r in enumerate(rows) if r["episode"] >= 23]
    if args.manifest:
        tr = [i for i, r in enumerate(rows) if r["key"] in manifest_data["train_keys"]]
        va = [i for i, r in enumerate(rows) if r["key"] in manifest_data["validation_keys"]]
        assert len(tr) + len(va) == count and not set(tr) & set(va)
    pred = {}
    manifest = []
    ckpts = []
    for run in [f"{prefix}_{s}" for s in [7108, 7119, 7130]]:
        path = root / run / "critic.pt"
        ck = torch.load(path, weights_only=True)
        drop = ck.get("dropout", 0.0)
        layers = [nn.Linear(ck["input_dim"], 128), nn.LayerNorm(128), nn.SiLU()]
        if drop:
            layers.append(nn.Dropout(drop))
        layers += [nn.Linear(128, 64), nn.SiLU()]
        if drop:
            layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(64, 9))
        m = nn.Sequential(*layers)
        m.load_state_dict(ck["best_model"])
        m.eval()
        with torch.no_grad():
            p = (
                (m((x - ck["state_mean"]) / ck["state_std"]) + ck.get("prior_logits", 0))
                .sigmoid()
                .numpy()
            )
        pred[run] = p - p[:, :1]
        if drop:
            ckpts.append(ck)
            manifest.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    pred[f"{prefix}_ensemble"] = np.mean(
        [pred[f"{prefix}_{s}"] for s in [7108, 7119, 7130]], axis=0
    )
    mean = y[tr].mean((0, 2))
    pred["constant_candidate"] = np.tile(mean - mean[0], (count, 1))
    task_control = np.zeros((count, 9))
    for task in [1, 5, 6, 9]:
        ti = [i for i in tr if rows[i]["task"] == task]
        mean = y[ti].mean((0, 2))
        task_control[[i for i, r in enumerate(rows) if r["task"] == task]] = mean - mean[0]
    pred["task_constant_candidate"] = task_control
    report = {
        "note": "Development diagnostics only; validation already used for checkpoint selection. Cluster intervals are descriptive, not independent confirmation. Task-constant control uses task identity and training labels only.",
        "methods": {},
    }
    for name, p in pred.items():
        pick = p.argmax(-1)
        pick[p.max(-1) <= 0.02] = 0
        selected = y[np.arange(count), pick]
        gain = (selected - y[:, 0]).mean(-1)
        metrics = {}
        for split, ids in [("train", tr), ("validation", va)]:
            per = {}
            draws = []
            rng = np.random.default_rng(7108)
            for task in [1, 5, 6, 9]:
                ti = [i for i in ids if rows[i]["task"] == task]
                eps = sorted({rows[i]["episode"] for i in ti})
                cl = [np.mean([gain[i] for i in ti if rows[i]["episode"] == e]) for e in eps]
                draws.append(rng.choice(cl, (10000, len(cl)), replace=True).mean(1))
                per[str(task)] = {
                    "base": int(y[ti, 0].sum()),
                    "selected": int(selected[ti].sum()),
                    "gain": float(gain[ti].mean()),
                }
            metrics[split] = {
                "base": int(y[ids, 0].sum()),
                "selected": int(selected[ids].sum()),
                "trials": len(ids) * 16,
                "gain": float(gain[ids].mean()),
                "cluster_interval90": np.quantile(np.mean(draws, axis=0), [0.05, 0.95]).tolist(),
                "per_task": per,
                "choices": dict(zip(names, np.bincount(pick[ids], minlength=9).tolist())),
            }
        report["methods"][name] = metrics
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save(
        {
            "heads": ckpts,
            "threshold": 0.02,
            "aggregation": "mean predicted advantage",
            "candidate_names": names,
        },
        out / "ensemble.pt",
    )
    print(json.dumps({k: v["validation"] for k, v in report["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()

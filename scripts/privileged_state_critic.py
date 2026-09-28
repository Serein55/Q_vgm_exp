"""Parametric privileged-state critic diagnostic.

Trains the nine-output candidate critic on privileged simulator features
(geometry / full_state) and on the visual mean-pool arm under one identical
protocol, to decide whether the bottleneck is the visual representation or the
label/candidate design. Config selection uses only the 128 training groups
(4 episode folds, 3-seed ensemble); the 32 development groups are read once
for the selected config.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.02
TASKS = [1, 5, 6, 9]


def load_data():
    manifest = json.loads(
        (ROOT / "artifacts/controller_coverage/training_manifest.json").read_text()
    )
    files = manifest["files"]
    rows = [torch.load(p, weights_only=True) for p in files]
    rows.sort(key=lambda r: r["key"])
    order = {r["key"]: i for i, r in enumerate(rows)}
    for p in files:
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == manifest["sha256"][p]
    names = rows[0]["candidate_names"]
    assert len(names) == 9 and all(r["candidate_names"] == names for r in rows)

    labels = torch.zeros(len(rows), 9, 16)
    for i, r in enumerate(rows):
        assert len(r["records"]) == 144
        for rec in r["records"]:
            labels[i, names.index(rec["candidate"]), rec["repeat"]] = float(rec["success"])

    path_of = {Path(p).stem: p for p in files}
    feats = {}
    for r in rows:
        e = torch.load(ROOT / f"artifacts/privileged_state/{r['key']}.pt", weights_only=True)
        assert e["source_sha256"] == manifest["sha256"][path_of[r["key"]]]
        feats[r["key"]] = e

    base_action = torch.stack([r["candidates"][0, :5, :7].flatten() for r in rows])
    proprio = torch.stack([r["observation"]["observation/state"].float() for r in rows])
    visual = torch.stack([r["mean_pool"].float() for r in rows])
    geometry = torch.stack([feats[r["key"]]["geometry"] for r in rows])
    full_state = torch.stack([feats[r["key"]]["full_state"] for r in rows])
    x_of = {
        "visual": torch.cat([visual, proprio, base_action], 1),
        "geometry": torch.cat([geometry, proprio, base_action], 1),
        "full_state": torch.cat([full_state, proprio, base_action], 1),
    }
    task_ids = torch.tensor([r["task"] for r in rows])
    onehot = torch.zeros(len(rows), len(TASKS))
    onehot[torch.arange(len(rows)), [TASKS.index(t) for t in task_ids]] = 1.0
    episodes = torch.tensor([r["episode"] for r in rows])
    train_keys = set(manifest["train_keys"])
    val_keys = set(manifest["validation_keys"])
    assert not train_keys & val_keys and len(train_keys) == 128 and len(val_keys) == 32
    tr = torch.tensor([order[k] for k in sorted(train_keys)])
    va = torch.tensor([order[k] for k in sorted(val_keys)])
    assert not {episodes[i].item() for i in tr} & {episodes[i].item() for i in va}
    return dict(
        rows=rows,
        labels=labels,
        x_of=x_of,
        onehot=onehot,
        episodes=episodes,
        task_ids=task_ids,
        tr=tr,
        va=va,
        names=names,
    )


def make_model(dim, arch, dropout):
    if arch == "linear":
        return nn.Linear(dim, 9)
    if arch == "32":
        return nn.Sequential(nn.Linear(dim, 32), nn.LayerNorm(32), nn.SiLU(), nn.Linear(32, 9))
    layers = [nn.Linear(dim, 128), nn.LayerNorm(128), nn.SiLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(128, 64), nn.SiLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 9))
    return nn.Sequential(*layers)


def train_one(x, y, idx, ref, seed, arch, dropout, steps, device):
    torch.manual_seed(seed)
    mu = x[idx].mean(0)
    std = x[idx].std(0).clamp_min(0.05)
    xs = ((x - mu) / std)[idx].to(device)
    ys = y[idx].to(device)
    adv = ys - ys[:, ref : ref + 1]
    model = make_model(x.shape[1], arch, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logit = model(xs)
        p = logit.sigmoid()
        pred_adv = p - p[:, ref : ref + 1]
        others = [k for k in range(9) if k != ref]
        loss = (pred_adv[:, others] - adv[:, others]).square().mean() + 0.1 * (
            nn.functional.binary_cross_entropy_with_logits(logit, ys)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        return model, mu, std


def predict_probs(model, mu, std, x, idx, device):
    with torch.no_grad():
        return model(((x - mu) / std)[idx].to(device)).sigmoid().cpu()


def select_from_probs(probs, ref):
    adv = probs - probs[:, ref : ref + 1]
    pick = adv.argmax(-1)
    pick[adv.max(-1).values <= THRESHOLD] = ref
    return pick


def ranking_acc(probs, y, idx):
    pd = probs[idx][:, :, None] - probs[idx][:, None, :]
    diff = y[idx, :, None] - y[idx, None, :]
    mask = diff.abs() >= 0.125
    if not mask.any():
        return None
    return float(((pd[mask] * diff[mask]) > 0).float().mean())


def expected_success(y, idx, pick):
    return int(round(float(y[idx, pick].sum()) * 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7108, 7119, 7130])
    ap.add_argument("--modes", nargs="+", default=["visual", "geometry", "full_state"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)

    out = ROOT / f"artifacts/privileged_state_critic{args.tag}"
    out.mkdir(exist_ok=True)
    assert not (out / "report.json").exists(), "report already exists; refusing to overwrite"

    d = load_data()
    rows, labels, x_of, onehot = d["rows"], d["labels"], d["x_of"], d["onehot"]
    episodes, tr, va, names = d["episodes"], d["tr"], d["va"], d["names"]
    y = labels.mean(-1)
    first8 = labels[..., :8].mean(-1)
    last8 = labels[..., 8:].mean(-1)

    groups = sorted(set(episodes[tr].tolist()))
    folds = [tr[torch.isin(episodes[tr], torch.tensor(groups[f::4]))] for f in range(4)]
    for f in folds:
        rest = tr[~torch.isin(tr, f)]
        assert not set(episodes[f].tolist()) & set(episodes[rest].tolist())

    grid = [
        dict(mode=m, arch=a, dropout=dr, onehot=oh)
        for m, a, dr, oh in itertools.product(
            args.modes, ["linear", "32", "128"], [0.0, 0.15], [0, 1]
        )
        if not (dr and a != "128")
    ]

    def features(cfg, idx_pool):
        x = x_of[cfg["mode"]]
        if cfg["onehot"]:
            x = torch.cat([x, onehot], 1)
        return x

    records = []
    oof_cache = {}
    for cfg in grid:
        x = features(cfg, tr)
        per_seed_oof, per_seed_acc = [], []
        for seed in args.seeds:
            probs = torch.zeros(len(rows), 9)
            for f in folds:
                rest = tr[~torch.isin(tr, f)]
                ref = int(y[rest].mean(0).argmax())
                model, mu, std = train_one(
                    x, y, rest, ref, seed, cfg["arch"], cfg["dropout"], args.steps, device
                )
                probs[f] = predict_probs(model, mu, std, x, f, device)
            per_seed_oof.append(probs)
            per_seed_acc.append(ranking_acc(probs, y, tr))
        ens = torch.stack(per_seed_oof).mean(0)
        oof_cache[json.dumps(cfg, sort_keys=True)] = (ens, per_seed_acc)
        pick = torch.zeros(len(rows), dtype=torch.long)
        for f in folds:
            rest = tr[~torch.isin(tr, f)]
            ref = int(y[rest].mean(0).argmax())
            pick[f] = select_from_probs(ens[f], ref)
        rec = dict(
            cfg,
            oof_selected=expected_success(y, tr, pick[tr]),
            oof_ranking_acc=round(
                float(sum(a for a in per_seed_acc if a is not None) / len(per_seed_acc)), 4
            ),
        )
        records.append(rec)
        print(json.dumps(rec), flush=True)

    # Label-only ceilings (no model): fixed reference and split-half oracle.
    ref_train = int(y[tr].mean(0).argmax())
    fixed_oof = expected_success(y, tr, torch.full((len(tr),), ref_train))
    half_pick = first8[tr].argmax(-1)
    ceilings = {
        "reference_candidate": names[ref_train],
        "fixed_oof_2048": fixed_oof,
        "split_half_select_first8_count_last8_1024": int(
            round(float(last8[tr, half_pick].sum()) * 8)
        ),
        "split_half_fixed_last8_1024": int(round(float(last8[tr, ref_train].sum()) * 8)),
        "split_half_full_oracle_last8_1024": int(
            round(float(last8[tr, last8[tr].argmax(-1)].sum()) * 8)
        ),
        "full_label_oracle_2048": expected_success(y, tr, y[tr].argmax(-1)),
    }
    print(json.dumps(ceilings), flush=True)

    best = max(records, key=lambda r: r["oof_selected"])
    selection = dict(
        selected=best,
        prior_reference=dict(
            visual_retrieval_oof=1175,
            fixed_oof=1100,
            visual_dropout_critic_dev=292,
            retrieval_dev=309,
            fixed_dev=296,
        ),
        note="Selected purely by training OOF ensemble success; development read once below.",
    )

    cfg = {k: best[k] for k in ["mode", "arch", "dropout", "onehot"]}
    x = features(cfg, tr)
    ens_va = torch.zeros(len(rows), 9)
    va_accs = []
    for seed in args.seeds:
        model, mu, std = train_one(
            x, y, tr, ref_train, seed, cfg["arch"], cfg["dropout"], args.steps, device
        )
        probs_va = predict_probs(model, mu, std, x, va, device)
        ens_va[va] += probs_va / len(args.seeds)
        full = torch.zeros(len(rows), 9)
        full[va] = probs_va
        va_accs.append(ranking_acc(full, y, va))
    pick_va = select_from_probs(ens_va[va], ref_train)
    development = dict(
        selected=expected_success(y, va, pick_va),
        fixed=expected_success(y, va, torch.full((len(va),), ref_train)),
        full_label_oracle=expected_success(y, va, y[va].argmax(-1)),
        ranking_acc=round(float(sum(a for a in va_accs if a is not None) / len(va_accs)), 4),
        per_task={
            str(t): expected_success(
                y,
                va[torch.isin(d["task_ids"][va], torch.tensor([t]))],
                pick_va[torch.isin(d["task_ids"][va], torch.tensor([t]))],
            )
            for t in TASKS
        },
        choices=dict(zip(names, torch.bincount(pick_va, minlength=9).tolist())),
    )

    by_mode = {}
    for m in args.modes:
        recs = [r for r in records if r["mode"] == m]
        top = max(recs, key=lambda r: r["oof_selected"])
        by_mode[m] = dict(
            best_oof=top["oof_selected"],
            best_cfg={k: top[k] for k in ["arch", "dropout", "onehot"]},
            mean_ranking_acc=round(sum(r["oof_ranking_acc"] for r in recs) / len(recs), 4),
            max_ranking_acc=max(r["oof_ranking_acc"] for r in recs),
        )

    report = dict(
        protocol=(
            "Privileged-state parametric critic diagnosis. Same 23040 labels, 128/32 split, "
            "4 episode folds, 3-seed probability-mean ensemble, threshold .02 vs training-best "
            "fixed reference, standardization per fold-train. Config chosen by training OOF only."
        ),
        steps=args.steps,
        seeds=args.seeds,
        grid=records,
        ceilings=ceilings,
        by_mode=by_mode,
        selection=selection,
        development=development,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(ceilings=ceilings, by_mode=by_mode, selection=selection, development=development),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

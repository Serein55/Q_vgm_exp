"""Replay saved prefixes to recover simulator state, then compare retrieval features."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def extract(root, manifest, worker, workers):
    cfg = load_config(root / "configs/controller_coverage.yaml")
    setup_runtime(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy

    torch.set_num_threads(2)
    out = root / "artifacts/privileged_state"
    out.mkdir(exist_ok=True)
    status = out / f"worker{worker}.status"
    status.write_text("running\n")
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    for i, path in enumerate(manifest["files"]):
        if i % workers != worker:
            continue
        r = torch.load(path, weights_only=True)
        dest = out / f"{r['key']}.pt"
        if dest.exists():
            continue
        assert r["episode"] < 35
        seed = cfg["runtime"]["seed"] + 800000 + r["task"] * 10000 + r["episode"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        task = suite.get_task(r["task"])
        env = make_env(cfg, task, seed)
        try:
            env.seed(seed)
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(r["task"])[r["episode"]])
            for _ in range(cfg["env"]["settle_steps"]):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            for chunk in r["prefix_actions"].numpy():
                for action in chunk:
                    obs, _, done, _ = env.step(action.tolist())
                    assert not done and not env.check_success()
            inp = observation_for_policy(obs, task.language)
            se = float(
                np.abs(
                    inp["observation/state"] - r["observation"]["observation/state"].numpy()
                ).max()
            )
            ie = max(
                float(
                    np.abs(inp[k].astype(float) - r["observation"][k].numpy().astype(float)).mean()
                )
                for k in ["observation/image", "observation/wrist_image"]
            )
            assert se <= 1e-3 and ie <= 1, (r["key"], se, ie)
            eef = np.array(obs["robot0_eef_pos"])
            features = {}
            for name, body in sorted(env.env.obj_body_id.items()):
                features["object/" + name] = np.array(env.sim.data.body_xpos[body]) - eef
            for i, name in enumerate(env.sim.model.site_names):
                if name and name in env.env.object_sites_dict:
                    features["site/" + name] = np.array(env.sim.data.site_xpos[i]) - eef
            geometry = np.concatenate(list(features.values())).astype(np.float32)
            state = np.concatenate(
                [
                    geometry,
                    env.sim.data.qpos.copy(),
                    env.sim.data.qvel.copy(),
                    np.concatenate(
                        [env.sim.data.body_xquat[b] for _, b in sorted(env.env.obj_body_id.items())]
                    ),
                ]
            ).astype(np.float32)
            payload = dict(
                key=r["key"],
                task=r["task"],
                episode=r["episode"],
                geometry=torch.from_numpy(geometry),
                full_state=torch.from_numpy(state),
                feature_names=list(features),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                state_error=se,
                image_error=ie,
            )
            tmp = dest.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(dest)
            print(
                json.dumps(
                    dict(
                        key=r["key"],
                        geometry_dim=len(geometry),
                        state_dim=len(state),
                        state_error=se,
                        image_error=ie,
                    )
                ),
                flush=True,
            )
        finally:
            env.close()
    status.write_text("complete\n")


def fit(root, manifest):
    import numpy as np
    import torch

    out = root / "artifacts/privileged_state"
    assert not (out / "report.json").exists()
    rows = [torch.load(p, weights_only=True) for p in manifest["files"]]
    extra = [torch.load(out / f"{r['key']}.pt", weights_only=True) for r in rows]
    assert all(
        e["source_sha256"] == hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for e, p in zip(extra, manifest["files"])
    )
    assert len(rows) == len(extra) == 160
    for task in [1, 5, 6, 9]:
        schema = [e["feature_names"] for e in extra if e["task"] == task]
        assert schema and all(names == schema[0] for names in schema)
    names = rows[0]["candidate_names"]
    y = np.array(
        [
            [sum(t["success"] for t in r["records"] if t["candidate"] == n) / 16 for n in names]
            for r in rows
        ]
    )
    tr = np.array([i for i, r in enumerate(rows) if r["key"] in manifest["train_keys"]])
    va = np.array([i for i, r in enumerate(rows) if r["key"] in manifest["validation_keys"]])
    episodes = np.array([r["episode"] for r in rows])
    tasks = np.array([r["task"] for r in rows])
    stage = np.array([int(r["actual_prefix_chunks"] > 0) for r in rows])
    features = {
        mode: {
            t: np.stack([e[mode].numpy() for e in extra if e["task"] == t]) for t in [1, 5, 6, 9]
        }
        for mode in ["geometry", "full_state"]
    }

    def predict(a, b, mode, k):
        picks = np.zeros(len(b), int)
        ref = int(y[a].mean(0).argmax())
        for t in [1, 5, 6, 9]:
            task_ids = np.flatnonzero(tasks == t)
            mapping = {int(v): i for i, v in enumerate(task_ids)}
            f = features[mode][t].astype(float)
            for s in [0, 1]:
                pool = a[(tasks[a] == t) & (stage[a] == s)]
                query_positions = np.flatnonzero((tasks[b] == t) & (stage[b] == s))
                query = b[query_positions]
                if not len(query):
                    continue
                ids = [mapping[int(i)] for i in pool]
                qs = [mapping[int(i)] for i in query]
                std = f[ids].std(0).clip(0.02)
                for pos, q in zip(query_positions, qs):
                    distance = (((f[ids] - f[q]) / std) ** 2).mean(1)
                    nearest = np.argsort(distance)[:k]
                    w = np.exp(-distance[nearest] / max(float(distance[nearest].mean()), 1e-8))
                    w /= w.sum()
                    estimate = (
                        k * (w[:, None] * y[pool[nearest]]).sum(0) + 2 * y[pool].mean(0)
                    ) / (k + 2)
                    pick = int(estimate.argmax())
                    picks[pos] = pick if estimate[pick] - estimate[ref] > 0.02 else ref
        return picks, ref

    groups = sorted(set(episodes[tr]))
    records = []
    for mode in ["geometry", "full_state"]:
        for k in [2, 4, 8]:
            picks = np.zeros(len(tr), int)
            for fold in range(4):
                mask = np.isin(episodes[tr], groups[fold::4])
                a, b = tr[~mask], tr[mask]
                assert not set(episodes[a]) & set(episodes[b])
                picks[mask], _ = predict(a, b, mode, k)
            record = dict(
                features=mode, neighbors=k, oof_selected=int(round(y[tr, picks].sum() * 16))
            )
            records.append(record)
            print(record, flush=True)
    best = max(records, key=lambda r: r["oof_selected"])
    report = dict(
        protocol="Privileged simulator-state diagnosis, not deployable visual policy. Same labels and 128/32 split; 4 episode folds choose features/k. Train-pool scaling floor .02, prior2, switch .02.",
        training_cv=records,
        selected=best,
        training_trials=2048,
        original_visual_oof=1175,
        replay_max_state_error=max(e["state_error"] for e in extra),
        replay_max_image_error=max(e["image_error"] for e in extra),
    )
    if best["oof_selected"] > 1175:
        pick, ref = predict(tr, va, best["features"], best["neighbors"])
        report["development"] = dict(
            selected=int(round(y[va, pick].sum() * 16)),
            fixed=int(round(y[va, ref].sum() * 16)),
            visual_retrieval=309,
            trials=512,
            per_task={
                str(t): int(round(y[va, pick][tasks[va] == t].sum() * 16)) for t in [1, 5, 6, 9]
            },
        )
    else:
        report["development"] = "Skipped: training OOF did not beat visual retrieval"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/controller_coverage/training_manifest.json").read_text()
    )
    if args.fit:
        fit(root, manifest)
    else:
        assert args.worker is not None and 0 <= args.worker < args.workers
        extract(root, manifest, args.worker, args.workers)


if __name__ == "__main__":
    main()

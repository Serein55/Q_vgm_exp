"""Ground-truth grasp trace for a LIBERO task, to replace eyeballing montages.

Runs the frozen SFT policy on a few initial states of one task and records, every
TRACE_EVERY steps, the simulator positions of every scene object plus the gripper site.
Answers three questions the videos cannot answer reliably:
  1. did the target object ever leave its initial support (e.g. the cabinet top)?
  2. which object was closest to / inside the gripper while the gripper was closed?
  3. where did the target end up relative to the placement target?

Usage: bash Q_vgm/run.sh diag_grasp_trace --config configs/libero_spatial.yaml \
         --task-id 9 --trials 0 1 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime

TRACE_EVERY = 10


def body_ids(model, names):
    table = {n: i for i, n in enumerate(model.body_names)}
    return {n: table[n] for n in names if n in table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/libero_spatial.yaml")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--trials", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--name", default="grasp_trace")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_runtime(cfg)

    import numpy as np
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import infer_actions, load_sft_policy

    seed = cfg["runtime"]["seed"]
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    out = Path(cfg["paths"]["artifacts"]) / "diag" / args.name
    out.mkdir(parents=True, exist_ok=True)

    policy = load_sft_policy(cfg)
    env = make_env(cfg, task, seed)
    try:
        model = env.env.sim.model
        all_bodies = list(model.body_names)
        scene = [
            b
            for b in all_bodies
            if any(k in b for k in ("bowl", "plate", "cabinet", "cookie", "ramekin", "stove"))
        ]
        print(f"SCENE_BODIES={scene}", flush=True)
        ids = body_ids(model, scene)
        grip_site = None
        for cand in ("grip_site", "grip_site0", "right_grip_site", "gripper0_grip_site"):
            try:
                grip_site = model.site_name2id(cand)
                break
            except Exception:
                continue
        if grip_site is None:
            raise SystemExit(f"no grip site; sites={list(model.site_names)[:40]}")
        print(f"GRIP_SITE={grip_site}", flush=True)

        for trial in args.trials:
            episode_seed = seed + args.task_id * 10000 + trial
            rng = np.random.default_rng(episode_seed)
            env.seed(episode_seed)
            env.reset()
            obs = env.set_init_state(states[trial])
            for _ in range(cfg["env"]["settle_steps"]):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            init = {n: np.asarray(env.env.sim.data.xpos[i]).tolist() for n, i in ids.items()}
            trace = []
            success = False
            steps = 0
            while steps < cfg["env"]["max_steps"]:
                inp = observation_for_policy(obs, task.language)
                res = infer_actions(policy, inp, cfg, rng)
                for action in res["actions"][: cfg["env"]["action_chunk"]]:
                    obs, reward, done, info = env.step(action.tolist())
                    steps += 1
                    success = bool(env.check_success())
                    if steps % TRACE_EVERY == 0 or success or done:
                        pos = {n: np.asarray(env.env.sim.data.xpos[i]) for n, i in ids.items()}
                        grip = np.asarray(env.env.sim.data.site_xpos[grip_site])
                        near = sorted(
                            ((float(np.linalg.norm(p - grip)), n) for n, p in pos.items())
                        )[:2]
                        trace.append(
                            dict(
                                step=steps,
                                grip=grip.round(3).tolist(),
                                nearest=[(n, round(d, 3)) for d, n in near],
                                **{n: p.round(3).tolist() for n, p in pos.items()},
                            )
                        )
                    if success or done or steps >= cfg["env"]["max_steps"]:
                        break
                if success or done:
                    break

            final = {n: np.asarray(env.env.sim.data.xpos[i]).tolist() for n, i in ids.items()}
            moved = {
                n: round(float(np.linalg.norm(np.asarray(f) - np.asarray(init[n]))), 3)
                for n, f in final.items()
            }
            rec = dict(
                task_id=args.task_id,
                trial=trial,
                seed=episode_seed,
                success=success,
                steps=steps,
                init=init,
                final=final,
                moved=moved,
                trace=trace,
            )
            with (out / "traces.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k != "trace"}), flush=True)
    finally:
        env.close()
    print(f"OUTPUT={out}", flush=True)


if __name__ == "__main__":
    main()

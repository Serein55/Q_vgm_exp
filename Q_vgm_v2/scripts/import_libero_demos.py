"""Convert official HDF5 demos, rendering PRE-action simulator states at 256px.

The official create_dataset script stores post-action RGB/proprio but pre-action
`states`; mixing obs[i] with actions[i] would shift supervision by one step.
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qvgm.config import load_config, setup_runtime


def relocate_xml(xml, assets):
    import robosuite

    root = ET.fromstring(xml)
    for node in root.iter():
        old = node.get("file")
        if not old:
            continue
        if "/robosuite/" in old:
            new = Path(robosuite.__file__).parent / old.rsplit("/robosuite/", 1)[1]
        elif "/assets/" in old:
            new = Path(assets) / old.split("/assets/", 1)[1]
        else:
            new = Path(old)
        if not new.is_file():
            raise FileNotFoundError(new)
        node.set("file", str(new))
    return ET.tostring(root, encoding="unicode")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="Q_vgm_v2/configs/import_demos.yaml")
    p.add_argument("--task-ids", type=int, nargs="+", required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import h5py
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy

    source = Path(cfg["paths"]["demo_source"])
    out = Path(cfg["paths"]["demo_buffer"])
    out.mkdir(parents=True, exist_ok=True)
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    h = cfg["env"]["action_chunk"]
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        path = source / (Path(task.bddl_file).stem + "_demo.hdf5")
        env = make_env(cfg, task, cfg["runtime"]["seed"])
        try:
            with h5py.File(path, "r") as f:
                keys = sorted(f["data"], key=lambda x: int(x.split("_")[-1]))
                for key in keys[: args.limit]:
                    episode = int(key.split("_")[-1])
                    dest = out / f"task{task_id:02d}_episode{episode:03d}.pt"
                    if dest.exists():
                        continue
                    start = time.monotonic()
                    d = f["data"][key]
                    actions, states = d["actions"][:], d["states"][:]
                    rewards, dones = d["rewards"][:], d["dones"][:].astype(bool)
                    if not (len(actions) == len(states) == len(rewards) == len(dones)):
                        raise ValueError("Inconsistent source lengths")
                    if not dones[-1] or dones[:-1].any() or not np.isfinite(actions).all():
                        raise ValueError("Unexpected source terminal/action schema")
                    env.reset()
                    env.reset_from_xml_string(
                        relocate_xml(d.attrs["model_file"], cfg["paths"]["libero_assets"])
                    )
                    observations, prefixes, transitions = [], [], []
                    roundtrip = 0.0
                    # Use recorded pre-action states, retaining the source's action sequence.
                    for j in range(0, len(actions), h):
                        obs = env.set_init_state(states[j])
                        inp = observation_for_policy(obs, task.language)
                        ctx = flow.encode_context([inp])
                        observations.append(
                            {
                                k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                                for k, v in inp.items()
                            }
                        )
                        prefixes.append(ctx["prefix"][0, ctx["pad"][0]].cpu().to(torch.bfloat16))
                        n = min(h, len(actions) - j)
                        chunk = np.repeat(actions[j + n - 1 : j + n], h, axis=0).astype(np.float32)
                        chunk[:n] = actions[j : j + n]
                        # Use EXACT checkpoint input transforms, including action padding.
                        normalized = policy._input_transform(dict(inp, actions=chunk))["actions"]
                        normalized = torch.as_tensor(normalized, device=flow.device).float()[None]
                        restored = flow.unnormalize(normalized, ctx)[0]
                        err = float(np.max(np.abs(restored - chunk)))
                        roundtrip = max(roundtrip, err)
                        if err > 1e-5:
                            raise ValueError(f"Action roundtrip error {err}")
                        terminal = j + n == len(actions)
                        transitions.append(
                            dict(
                                action=normalized[0, :h, :7].cpu(),
                                rewards=torch.tensor(rewards[j : j + n], dtype=torch.float32),
                                reward=float(
                                    sum(
                                        cfg["offline"]["gamma"] ** k * float(rewards[j + k])
                                        for k in range(n)
                                    )
                                ),
                                steps=n,
                                terminated=terminal,
                                truncated=False,
                            )
                        )
                    # Terminal successor is reconstructed by stepping the last recorded action.
                    env.set_init_state(states[-1])
                    final_obs, _, _, _ = env.step(actions[-1].tolist())
                    terminal_success = bool(env.check_success())
                    inp = observation_for_policy(final_obs, task.language)
                    ctx = flow.encode_context([inp])
                    observations.append(
                        {
                            k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                            for k, v in inp.items()
                        }
                    )
                    prefixes.append(ctx["prefix"][0, ctx["pad"][0]].cpu().to(torch.bfloat16))
                    record = dict(
                        schema=2,
                        source="demo",
                        task_id=task_id,
                        episode=episode,
                        source_file=path.name,
                        source_episode=key,
                        source_revision=cfg["demo"]["source_revision"],
                        success=bool(rewards.max() > 0),
                        success_label_source="official_hdf5",
                        terminal_success_rechecked=terminal_success,
                        observation_source="rendered_pre_action_states",
                        observations=observations,
                        prefixes=prefixes,
                        transitions=transitions,
                    )
                    tmp = dest.with_suffix(".tmp")
                    torch.save(record, tmp)
                    tmp.replace(dest)
                    print(
                        json.dumps(
                            dict(
                                task_id=task_id,
                                episode=episode,
                                source="demo",
                                steps=len(actions),
                                chunks=len(transitions),
                                action_roundtrip_max_abs=roundtrip,
                                terminal_success_rechecked=terminal_success,
                                seconds=time.monotonic() - start,
                            )
                        ),
                        flush=True,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    main()

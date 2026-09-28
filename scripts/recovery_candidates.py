"""Matched single-chunk interventions with identical frozen-SFT continuations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import load_config, setup_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/controller_coverage.yaml")
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--lane", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark

    from qvgm.envs.libero_env import DUMMY_ACTION, make_env, observation_for_policy
    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy
    from qvgm.training import save_checkpoint

    settings = cfg["recovery"]
    if not 0 <= args.lane < args.lanes:
        ap.error("invalid lane")
    if not 0 <= args.worker < settings["workers"]:
        ap.error("invalid worker")
    root = Path(cfg["paths"]["artifacts"]) / settings["run"]
    root.mkdir(exist_ok=True)
    out = root / f"worker{args.worker}"
    out.mkdir(exist_ok=True)
    snapshot = json.dumps(cfg, sort_keys=True, indent=2)
    if (out / "config.json").exists() and (out / "config.json").read_text() != snapshot:
        raise ValueError("Resume configuration differs")
    if not (out / "config.json").exists():
        (out / "config.json").write_text(snapshot)
    status_path = out / ("status.txt" if args.lanes == 1 else f"status_lane{args.lane}.txt")
    episode_log = out / ("episodes.jsonl" if args.lanes == 1 else f"episodes_lane{args.lane}.jsonl")
    status_path.write_text("running\n")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.manual_seed(cfg["runtime"]["seed"])
    if not settings.get("controller_candidates") or settings.get("fresh_first_episode") is None:
        raise ValueError("Only fresh controller-grid collection is supported")
    retrieval = None
    if settings.get("frozen_retrieval"):
        from retrieval_policy import FrozenRetrievalPolicy

        if settings.get("reuse_run"):
            raise ValueError("Confirmation cannot reuse prior candidate outcomes")
        retrieval = FrozenRetrievalPolicy(
            cfg["paths"]["retrieval_policy"], settings["retrieval_sha256"]
        )
    policy = load_sft_policy(cfg)
    flow = Pi05Flow(policy, cfg)
    flow.enable_actor_training(cfg["offline"]["actor"]["master_dtype"])
    flow.model.requires_grad_(False)
    suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
    h, d = cfg["env"]["action_chunk"], cfg["env"]["action_dim"]
    H, D = cfg["model"]["action_horizon"], cfg["model"]["action_dim"]
    device = flow.device

    # Infer affine output scale through the actual output transform.
    def output_scale(context):
        a = torch.zeros(1, H, D, device=device)
        b = a.clone()
        b[:, :, :d] = 1
        return torch.as_tensor(
            flow.unnormalize(b, context)[0, 0] - flow.unnormalize(a, context)[0, 0],
            device=device,
            dtype=torch.float32,
        )

    fresh = settings.get("fresh_first_episode")
    if settings["fractions"] != [0.0, 0.5]:
        raise ValueError("Fresh collection expects initial and fixed-prefix states")
    source_episodes = [
        dict(
            task_id=t,
            episode=e,
            seed=cfg["runtime"]["seed"] + 800000 + t * 10000 + e,
            transitions=[],
        )
        for t in settings["task_ids"]
        for e in range(fresh, fresh + settings["episodes"])
    ]
    jobs = [
        (e, fraction)
        for e, ep in enumerate(source_episodes)
        if ep["task_id"] in settings["task_ids"]
        for fraction in settings["fractions"]
    ]
    for job, (e, fraction) in enumerate(jobs):
        if job % settings["workers"] != args.worker:
            continue
        if (job // settings["workers"]) % args.lanes != args.lane:
            continue
        ep = source_episodes[e]
        index = settings["fresh_prefix_chunks"] if fraction else 0
        key = f"task{ep['task_id']:02d}_episode{ep['episode']:03d}_state{index:03d}"
        dest = out / f"{key}.pt"
        if dest.exists():
            continue
        task = suite.get_task(ep["task_id"])
        initial = suite.get_task_init_states(ep["task_id"])[ep["episode"]]
        env = None
        common = None
        candidate_seed = (
            cfg["runtime"]["seed"] + ep["task_id"] * 10000 + ep["episode"] * 100 + index
        )
        records = []
        selected_index = None
        retrieval_reference = None
        retrieval_donors = None
        names = [
            "base",
            "x_plus",
            "x_minus",
            "y_plus",
            "y_minus",
            "z_plus",
            "z_minus",
            "grip_plus",
            "grip_minus",
        ]
        prior = None
        first_repeat = 0
        if settings.get("reuse_run"):
            matches = list(
                (Path(cfg["paths"]["artifacts"]) / settings["reuse_run"]).glob(f"worker*/{key}.pt")
            )
            if len(matches) != 1:
                raise ValueError(f"Expected one source group for {key}")
            prior = torch.load(matches[0], weights_only=True, map_location="cpu")
            assert prior["candidate_names"] == names
            records = list(prior["records"])
            first_repeat = len(records) // len(names)
            assert {(r["candidate"], r["repeat"]) for r in records} == {
                (n, k) for n in names for k in range(first_repeat)
            }
            assert first_repeat < settings["repeats"]
        try:
            prefix_actions = (
                [x.numpy().copy() for x in prior["prefix_actions"]] if prior is not None else []
            )
            if prior is None and index:
                np.random.seed(ep["seed"])
                torch.manual_seed(ep["seed"])
                env = make_env(cfg, task, ep["seed"])
                env.seed(ep["seed"])
                env.reset()
                obs = env.set_init_state(initial)
                for _ in range(cfg["env"]["settle_steps"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                prefix_rng = np.random.default_rng(ep["seed"] + 2000000)
                for _ in range(index):
                    ctx = flow.encode_context([observation_for_policy(obs, task.language)])
                    eps = torch.from_numpy(prefix_rng.standard_normal((1, H, D)).astype(np.float32))
                    act, _ = flow.sample_with_intermediates(ctx, eps)
                    chunk = flow.unnormalize(act, ctx)[0, :h].copy()
                    terminal = False
                    for a in chunk:
                        obs, _, done, _ = env.step(a.tolist())
                        if done or env.check_success():
                            terminal = True
                            break
                    if terminal:
                        break  # Replay only complete nonterminal chunks before this one.
                    prefix_actions.append(chunk)
            for repeat in range(first_repeat, settings["repeats"]):
                for cidx, name in enumerate(names):
                    if selected_index is not None and cidx not in {
                        0,
                        retrieval_reference,
                        selected_index,
                    }:
                        continue
                    if env is not None:
                        env.close()
                    np.random.seed(ep["seed"])
                    torch.manual_seed(ep["seed"])
                    env = make_env(cfg, task, ep["seed"])
                    env.seed(ep["seed"])
                    env.reset()
                    obs = env.set_init_state(initial)
                    for _ in range(cfg["env"]["settle_steps"]):
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    elapsed = 0
                    for chunk in prefix_actions:
                        for action in chunk:
                            obs, _, done, _ = env.step(action.tolist())
                            elapsed += 1
                            if done or env.check_success():
                                raise RuntimeError("Fresh SFT prefix terminated on replay")
                    inp = observation_for_policy(obs, task.language)
                    if common is None:
                        common = env.get_sim_state().copy()
                        common_obs = inp
                        if prior is not None:
                            candidates = prior["candidates"].to(device)
                            pooled = prior["mean_pool"]
                            actions_env = prior["environment_actions"].numpy()
                            saved_observation = prior["observation"]
                            for k in ["observation/image", "observation/wrist_image"]:
                                assert (
                                    np.abs(
                                        inp[k].astype(float)
                                        - saved_observation[k].numpy().astype(float)
                                    ).mean()
                                    <= 1
                                )
                            np.testing.assert_allclose(
                                inp["observation/state"],
                                saved_observation["observation/state"].numpy(),
                                rtol=0,
                                atol=1e-3,
                            )
                        else:
                            context = flow.encode_context([inp])
                            rng = np.random.default_rng(candidate_seed)

                            def noise():
                                return torch.from_numpy(
                                    rng.standard_normal((1, H, D)).astype(np.float32)
                                )

                            pooled = context["prefix"][0, context["pad"][0]].float().mean(0).cpu()
                            base, _ = flow.sample_with_intermediates(context, noise())
                            original_actions = flow.unnormalize(base, context)[0, :h].copy()
                            scale = output_scale(context)
                            proposals = [base]
                            for axis in [0, 1, 2, 6]:
                                for sign in [1, -1]:
                                    desired = original_actions.copy()
                                    desired[:, axis] = (
                                        sign
                                        if axis == 6
                                        else np.clip(
                                            desired[:, axis] + sign * settings["controller_delta"],
                                            -1,
                                            1,
                                        )
                                    )
                                    proposal = base.clone()
                                    proposal[0, :h, :d] += (
                                        torch.as_tensor(desired - original_actions, device=device)
                                        / scale
                                    )
                                    reconstructed = flow.unnormalize(proposal, context)[0, :h]
                                    np.testing.assert_allclose(
                                        reconstructed, desired, rtol=0, atol=1e-6
                                    )
                                    np.testing.assert_allclose(
                                        reconstructed[:, 3:6],
                                        original_actions[:, 3:6],
                                        rtol=0,
                                        atol=1e-6,
                                    )
                                    proposals.append(proposal)
                            candidates = torch.cat(proposals)
                            actions_env = np.stack(
                                [flow.unnormalize(x[None], context)[0, :h] for x in candidates]
                            )
                            saved_observation = {
                                k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                                for k, v in inp.items()
                            }
                        if retrieval is not None:
                            assert names == retrieval.ck["candidate_names"]
                            selected_index, retrieval_reference, retrieval_donors = (
                                retrieval.choose(
                                    pooled.numpy(),
                                    inp["observation/state"],
                                    candidates[0, :h, :d].cpu().numpy(),
                                    ep["task_id"],
                                    len(prefix_actions),
                                )
                            )
                            assert retrieval_reference == names.index("x_minus")
                            # Freeze before any outcome is observed; only selected/base/reference run.
                            (out / f"{key}.selection.json").write_text(
                                json.dumps(
                                    dict(
                                        selected=names[selected_index],
                                        reference=names[retrieval_reference],
                                        donors=retrieval_donors,
                                        checkpoint_sha256=retrieval.sha256,
                                    )
                                )
                                + "\n"
                            )
                    sim_error = float(np.max(np.abs(env.get_sim_state() - common)))
                    image_error = max(
                        float(np.abs(inp[k].astype(float) - common_obs[k].astype(float)).mean())
                        for k in ["observation/image", "observation/wrist_image"]
                    )
                    state_error = float(
                        np.max(np.abs(inp["observation/state"] - common_obs["observation/state"]))
                    )
                    if sim_error > 1e-6 or state_error > 1e-3 or image_error > 1:
                        raise RuntimeError(
                            f"Unequal starts: {sim_error}, {state_error}, {image_error}"
                        )
                    # Candidate and prefix fixed; only continuation noise varies across repeats.
                    rng_follow = np.random.default_rng(candidate_seed + 1000000 + repeat)
                    normalized = candidates[cidx : cidx + 1].clone()
                    context = {
                        "state": torch.as_tensor(
                            policy._input_transform(dict(inp))["state"], device=device
                        )[None]
                    }
                    n = 0
                    success = False
                    while elapsed < cfg["env"]["max_steps"]:
                        chunk_actions = (
                            actions_env[cidx]
                            if n == 0
                            else flow.unnormalize(normalized, context)[0, :h]
                        )
                        for action in chunk_actions:
                            obs, _, done, _ = env.step(action.tolist())
                            elapsed += 1
                            n += 1
                            success = bool(env.check_success())
                            if success or done or elapsed >= cfg["env"]["max_steps"]:
                                break
                        if success or done or elapsed >= cfg["env"]["max_steps"]:
                            break
                        context = flow.encode_context([observation_for_policy(obs, task.language)])
                        eps = torch.from_numpy(
                            rng_follow.standard_normal((1, H, D)).astype(np.float32)
                        )
                        normalized, _ = flow.sample_with_intermediates(context, eps)
                    row = dict(
                        candidate=name,
                        repeat=repeat,
                        success=success,
                        steps=n,
                        return_value=cfg["offline"]["gamma"] ** (n - 1) if success else 0.0,
                        sim_error=sim_error,
                        state_error=state_error,
                        image_error=image_error,
                    )
                    records.append(row)
                    with episode_log.open("a") as f:
                        f.write(json.dumps(dict(key=key, **row)) + "\n")
                    print(json.dumps(dict(key=key, **row)), flush=True)
            save_checkpoint(
                dest,
                dict(
                    key=key,
                    task=ep["task_id"],
                    episode=ep["episode"],
                    state_index=index,
                    fraction=fraction,
                    buffer_signature=None,
                    z=pooled,
                    ae_z=None,
                    candidate_method="controller_grid",
                    mean_pool=pooled,
                    source_kind="fresh_sft",
                    actual_prefix_chunks=len(prefix_actions),
                    prefix_actions=torch.from_numpy(np.stack(prefix_actions))
                    if prefix_actions
                    else torch.empty(0, h, d),
                    observation=saved_observation,
                    candidates=candidates.cpu(),
                    candidate_names=names,
                    old_q=torch.zeros(len(names)),
                    environment_actions=torch.from_numpy(actions_env),
                    records=records,
                    retrieval_selected=names[selected_index]
                    if selected_index is not None
                    else None,
                    retrieval_reference=names[retrieval_reference]
                    if retrieval_reference is not None
                    else None,
                    retrieval_donors=retrieval_donors,
                    retrieval_sha256=retrieval.sha256 if retrieval is not None else None,
                    reused_repeats=first_repeat,
                    reuse_run=settings.get("reuse_run"),
                ),
            )
        finally:
            if env is not None:
                env.close()
    status_path.write_text("complete\n")


if __name__ == "__main__":
    main()

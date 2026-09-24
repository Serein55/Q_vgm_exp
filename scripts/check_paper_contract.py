"""GPU interface check of the literal paper profile; no optimizer or rollout."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvgm.config import ROOT, load_config, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/libero_spatial_paper.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg)

    import numpy as np
    import torch

    from qvgm.models.pi05_adapter import Pi05Flow, load_sft_policy

    out = Path(cfg["paths"]["artifacts"]) / "paper_contract_smoke.json"
    if out.exists():
        raise FileExistsError(out)
    ep = torch.load(
        Path(cfg["paths"]["artifacts"]) / "spatial_offline/buffer/task00_episode000.pt",
        weights_only=True,
        mmap=True,
    )
    obs = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in ep["observations"][0].items()
    }
    policy = load_sft_policy(cfg)
    changed = dict(obs)
    changed["observation/state"] = obs["observation/state"].copy()
    changed["observation/state"][0] += 0.1
    a, b = policy._input_transform(dict(obs)), policy._input_transform(changed)
    token_changes = int(np.count_nonzero(a["tokenized_prompt"] != b["tokenized_prompt"]))
    if token_changes == 0:
        raise AssertionError("Proprio change did not affect prefix tokens")
    flow = Pi05Flow(policy, cfg)
    params = flow.enable_actor_training(cfg["offline"]["actor"]["master_dtype"])
    dtypes = {}
    for p in params:
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    context = flow.encode_context([obs])
    actions, trajectory = flow.sample_with_intermediates(context)
    if actions.shape != (1, 5, 32) or any(x.requires_grad for _, x in trajectory):
        raise AssertionError("Unexpected action horizon or attached trajectory")
    velocity = flow.predict_velocity(trajectory[-1][1], trajectory[-1][0], context)
    target = velocity.detach().float() + 0.01
    loss = (velocity.float()[:, :, :7] - target[:, :, :7]).square().sum()
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError("Missing/nonfinite local actor gradients")
    grad_norm = sum(g.float().square().sum() for g in grads).sqrt().item()
    if not grad_norm > 0:
        raise AssertionError("Zero local actor gradient")
    report = dict(
        config=cfg,
        proprio_changed_tokens=token_changes,
        action_shape=list(actions.shape),
        trajectory_steps=len(trajectory),
        trainable_parameter_dtypes=dtypes,
        local_grad_norm=grad_norm,
        note="Interface/gradient check only; no optimizer update, environment evaluation, or reproduction result.",
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

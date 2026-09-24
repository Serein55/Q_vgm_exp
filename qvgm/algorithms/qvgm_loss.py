"""Local velocity matching: no differentiation through trajectories or targets."""

import torch

from .q_guidance import improve_actions


def local_velocity_loss(actor, reference, context, z, critic, tau, x, cfg, horizon=5, action_dim=7):
    x = x.detach()
    remaining = 1.0 - tau
    if remaining <= 0:
        raise ValueError("Guidance requires tau < 1")
    with torch.no_grad():
        # Keep small Q corrections when the expert returns bf16 velocities.
        vref = reference.predict_velocity(x, tau, context).float()
        endpoint = x + remaining * vref
    clean = endpoint[:, :horizon, :action_dim]
    improved, metrics = improve_actions(
        critic.mean, z, clean, cfg["ascent_steps"], cfg["ascent_step_size"]
    )
    correction = torch.zeros_like(vref)
    correction[:, :horizon, :action_dim] = (improved - clean) / remaining
    target = (vref + correction).detach()
    prediction = actor.predict_velocity(x, tau, context).float()
    # Match the executed chunk only; padded/unexecuted coordinates are not RL actions.
    residual = (prediction - target)[:, :horizon, :action_dim]
    loss = residual.square().flatten(1).sum(-1).mean()
    metrics["correction_norm"] = correction.flatten(1).norm(dim=-1).mean().item()
    return loss, metrics

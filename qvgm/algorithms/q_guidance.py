"""Detached, per-sample normalized Q ascent with permanent first-failure stop."""

import torch


def improve_actions(q_function, z, action, steps=3, alpha=0.05, eps=1e-12):
    best = action.detach().clone()
    with torch.no_grad():
        initial_q = q_function(z.detach(), best)
    score = initial_q.clone()
    active = torch.isfinite(score)
    accepted = torch.zeros(len(best), device=best.device)
    grad_norms = []
    for _ in range(steps):
        with torch.enable_grad():
            x = best.detach().requires_grad_(True)
            values = q_function(z.detach(), x)
            grad = torch.autograd.grad(values.sum(), x)[0]
        norm = grad.flatten(1).norm(dim=-1)
        grad_norms.append(norm.detach())
        good = active & torch.isfinite(norm) & (norm > eps)
        direction = grad / norm.clamp_min(eps).view(-1, *([1] * (grad.ndim - 1)))
        candidate = best + alpha * direction
        with torch.no_grad():
            candidate_q = q_function(z.detach(), candidate)
        take = good & torch.isfinite(candidate_q) & (candidate_q > score)
        shape = take.view(-1, *([1] * (best.ndim - 1)))
        best = torch.where(shape, candidate, best).detach()
        score = torch.where(take, candidate_q, score)
        accepted += take
        active = take  # Never reactivate rejected examples in later rounds.
    return best, dict(
        q_before=initial_q.mean().item(),
        q_after=score.mean().item(),
        q_gain=(score - initial_q).mean().item(),
        displacement=(best - action.detach()).flatten(1).norm(dim=-1).mean().item(),
        accept_rate=(accepted / max(steps, 1)).mean().item(),
        grad_norm=torch.stack(grad_norms).mean().item() if grad_norms else 0.0,
    )

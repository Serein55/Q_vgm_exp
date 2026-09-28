"""Convert schema-2 episodes without changing or relabeling the original files."""

import torch


def stepwise_transition(episode, index, horizon=5, action_dim=7, *, timeout_terminal):
    """Timeout semantics must be chosen explicitly, not inferred from success.

    Boundary bootstrap requires an actual following complete chunk. A continuing
    partial tail has no target for its last position, rather than a false zero.
    """
    transitions = episode["transitions"]
    t = transitions[index]
    n = int(t["steps"])
    action = torch.as_tensor(t["action"]).float()
    reward = torch.as_tensor(t["rewards"]).float()
    if not 0 < n <= horizon or action.shape != (horizon, action_dim):
        raise ValueError("Invalid steps or action shape")
    if reward.shape != (n,) or not torch.isfinite(reward).all() or not torch.isfinite(action).all():
        raise ValueError("Invalid per-step rewards/action")
    terminated, truncated = bool(t["terminated"]), bool(t["truncated"])
    if (terminated or truncated) and index != len(transitions) - 1:
        raise ValueError("Episode has transitions after its end")
    if n < horizon and index != len(transitions) - 1:
        raise ValueError("Unexpected non-final partial chunk")
    rewards = torch.zeros(horizon)
    rewards[:n] = reward
    valid = torch.arange(horizon) < n
    dones = torch.zeros(horizon, dtype=torch.bool)
    dones[n - 1] = terminated or (truncated and timeout_terminal)
    following = transitions[index + 1] if index + 1 < len(transitions) else None
    boundary_valid = (
        not terminated
        and not truncated
        and following is not None
        and int(following["steps"]) == horizon
    )
    return dict(
        action=action,
        rewards=rewards,
        valid=valid,
        dones=dones,
        boundary_valid=torch.tensor(boundary_valid),
        truncated=truncated,
    )

"""Pure loss functions: the trainer owns joint encoder updates and target EMA."""

import torch


def terminal_mask(valid, dones):
    # Include the first terminal itself, exclude every later position.
    earlier = dones.long().cumsum(-1) - dones.long()
    return valid.bool() & (earlier == 0)


@torch.no_grad()
def td_targets(rewards, dones, valid, boundary_valid, value, next_value, gamma):
    valid = terminal_mask(valid, dones)
    bootstrap = torch.cat([value[:, 1:], next_value[:, :1]], dim=-1)
    available = torch.cat([valid[:, 1:], boundary_valid.bool()[:, None]], dim=-1)
    # Missing intra-chunk successor: omit that nonterminal regression target.
    # At the chunk boundary, the paper instead masks bootstrap when unavailable.
    loss_valid = valid.clone()
    loss_valid[:, :-1] &= dones[:, :-1].bool() | available[:, :-1]
    targets = rewards + gamma * (~dones.bool()) * available * bootstrap
    return targets, loss_valid


def masked_mean(x, mask):
    if not bool(mask.any()):
        raise ValueError("No valid positions in batch")
    return x.masked_select(mask).mean()


def iql_losses(q1, q2, value, target_q1, target_q2, next_value, batch, gamma=0.99, expectile=0.8):
    if not 0 < expectile < 1:
        raise ValueError("Expectile must be in (0,1)")
    targets, q_valid = td_targets(
        batch["rewards"],
        batch["dones"],
        batch["valid"],
        batch["boundary_valid"],
        value,
        next_value,
        gamma,
    )
    q_loss = masked_mean((q1 - targets).square() + (q2 - targets).square(), q_valid)
    residual = torch.minimum(target_q1, target_q2).detach() - value
    weight = torch.where(residual > 0, expectile, 1 - expectile)
    v_valid = terminal_mask(batch["valid"], batch["dones"])
    v_loss = masked_mean(weight * residual.square(), v_valid)
    return q_loss, v_loss


@torch.no_grad()
def update_target(target, source, coefficient=0.005):
    if not 0 <= coefficient <= 1:
        raise ValueError("EMA coefficient outside [0,1]")
    target_state, source_state = target.state_dict(), source.state_dict()
    if target_state.keys() != source_state.keys():
        raise ValueError("Target architecture mismatch")
    for name, tensor in target_state.items():
        if tensor.is_floating_point():
            tensor.lerp_(source_state[name], coefficient)
        else:
            tensor.copy_(source_state[name])

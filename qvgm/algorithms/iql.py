"""Offline IQL with chunk-discounted rewards and target-head min expectiles."""

import copy

import torch


def expectile_loss(diff, expectile):
    return (torch.where(diff > 0, expectile, 1 - expectile) * diff.square()).mean()


def chunk_target(reward, steps, terminated, next_value, gamma):
    # Time-limit truncation retains bootstrap; a real terminal does not.
    return reward + gamma**steps * (~terminated).float() * next_value


class IQL:
    def __init__(self, critic, value, cfg, gamma):
        self.critic, self.value, self.cfg, self.gamma = critic, value, cfg, gamma
        # v5 leaves the multi-head aggregation for the expectile target unspecified.
        self.value_target = cfg.get("value_target", "min")
        if self.value_target not in ("min", "mean"):
            raise ValueError("value_target must be 'min' or 'mean'")
        self.target = copy.deepcopy(critic).requires_grad_(False).eval()
        self.qopt = torch.optim.Adam(critic.parameters(), lr=cfg["q_lr"])
        self.vopt = torch.optim.Adam(value.parameters(), lr=cfg["value_lr"])

    def head_target(self, q):
        return q.min(-1).values if self.value_target == "min" else q.mean(-1)

    def update(self, batch):
        z, a = batch["z"], batch["action"]
        with torch.no_grad():
            q = self.head_target(self.target(z, a))
        vloss = expectile_loss(q - self.value(z), self.cfg["expectile"])
        if not torch.isfinite(vloss):
            raise FloatingPointError("Non-finite IQL value loss")
        self.vopt.zero_grad(set_to_none=True)
        vloss.backward()
        self.vopt.step()
        with torch.no_grad():
            y = chunk_target(
                batch["reward"],
                batch["steps"],
                batch["terminated"],
                self.value(batch["next_z"]),
                self.gamma,
            )
        qloss = (self.critic(z, a) - y[:, None]).square().mean(0).sum()
        if not torch.isfinite(qloss):
            raise FloatingPointError("Non-finite IQL Q loss")
        self.qopt.zero_grad(set_to_none=True)
        qloss.backward()
        self.qopt.step()
        with torch.no_grad():
            for target, current in zip(self.target.parameters(), self.critic.parameters()):
                target.lerp_(current, self.cfg["target_ema"])
        return dict(q_loss=qloss.item(), v_loss=vloss.item(), q_mean=q.mean().item())

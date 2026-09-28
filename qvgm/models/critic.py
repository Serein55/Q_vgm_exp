"""Action re-injection at every hidden layer of independent Q heads."""

import torch
from torch import nn


class QHead(nn.Module):
    def __init__(self, state_dim, action_dim, widths):
        super().__init__()
        self.layers = nn.ModuleList()
        for width in widths:
            self.layers.append(nn.Linear(state_dim + action_dim, width))
            state_dim = width
        self.output = nn.Linear(state_dim, 1)

    def forward(self, z, action):
        action = action.flatten(1)
        x = z
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(torch.cat([x, action], -1)))
        return self.output(x).squeeze(-1)


class ChunkCritic(nn.Module):
    def __init__(self, state_dim=2048, horizon=5, action_dim=7, heads=10, widths=(512, 512, 256)):
        super().__init__()
        self.heads = nn.ModuleList(
            [QHead(state_dim, horizon * action_dim, widths) for _ in range(heads)]
        )

    def forward(self, z, action):
        return torch.stack([head(z, action) for head in self.heads], -1)

    def mean(self, z, action):
        return self(z, action).mean(-1)


class Value(nn.Module):
    def __init__(self, state_dim=2048, widths=(512, 512, 256)):
        super().__init__()
        layers = []
        for width in widths:
            layers.extend([nn.Linear(state_dim, width), nn.SiLU()])
            state_dim = width
        self.net = nn.Sequential(*layers, nn.Linear(state_dim, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class ConstantDirectionQ(nn.Module):
    """q(z, a) = a . m: the action gradient is identical at every state.

    Control for Q-VGM guidance. If distilling this reproduces the collapse caused by the
    learned critic, the state-conditional part of Q carries none of the effect.
    """

    def __init__(self, direction, horizon, action_dim):
        super().__init__()
        flat = torch.as_tensor(direction, dtype=torch.float32).flatten()
        if flat.numel() != horizon * action_dim:
            raise ValueError("direction must have horizon * action_dim entries")
        self.register_buffer("direction", flat.reshape(horizon, action_dim))

    def mean(self, z, action):
        return (action * self.direction).flatten(1).sum(-1)

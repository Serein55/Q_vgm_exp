import torch
from torch import nn


class StepwiseQHead(nn.Module):
    def __init__(self, state_dim=2304, horizon=5, action_dim=7, widths=(1024, 512)):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.layers = nn.ModuleList()
        dim = state_dim
        for width in widths:
            self.layers.append(nn.Linear(dim + horizon * action_dim, width))
            dim = width
        self.output = nn.Linear(dim, horizon)

    def forward(self, state, action):
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Expected normalized action [B,H,D]")
        flat = action.flatten(1)
        for layer in self.layers:
            state = torch.relu(layer(torch.cat([state, flat], dim=-1)))
        return self.output(state)


class DoubleStepwiseCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = StepwiseQHead(**kwargs)
        self.q2 = StepwiseQHead(**kwargs)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def minimum(self, state, action):
        return torch.minimum(*self(state, action))

    def score(self, state, action):
        return self.minimum(state, action).sum(-1)


class StepwiseValue(nn.Module):
    def __init__(self, state_dim=2304, horizon=5, widths=(1024, 512)):
        super().__init__()
        layers = []
        for width in widths:
            layers.extend([nn.Linear(state_dim, width), nn.ReLU()])
            state_dim = width
        self.net = nn.Sequential(*layers, nn.Linear(state_dim, horizon))

    def forward(self, state):
        return self.net(state)

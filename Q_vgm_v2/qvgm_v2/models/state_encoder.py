import torch
from torch import nn


class CriticStateEncoder(nn.Module):
    """Input proprio must already use the recorded training-set normalization."""

    def __init__(self, proprio_dim=8, proprio_embed_dim=256, rl_token_dim=2048):
        super().__init__()
        self.proprio_proj = nn.Linear(proprio_dim, proprio_embed_dim)
        self.norm = nn.LayerNorm(rl_token_dim + proprio_embed_dim)

    def forward(self, z_rl, proprio):
        return self.norm(torch.cat([z_rl, self.proprio_proj(proprio)], dim=-1))

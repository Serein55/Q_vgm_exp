"""Memory-mapped episode shards; only local torch tensor/string payloads."""

from pathlib import Path

import torch


class ReplayBuffer:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("task*_episode*.pt"))
        if not self.files:
            raise ValueError(f"No rollout shards in {directory}")
        self.episodes = [torch.load(p, mmap=True, weights_only=True) for p in self.files]
        self.transitions = [
            (e, i) for e, ep in enumerate(self.episodes) for i in range(len(ep["transitions"]))
        ]
        self.states = [
            (e, i) for e, ep in enumerate(self.episodes) for i in range(len(ep["observations"]))
        ]
        for ep in self.episodes:
            if (
                ep["schema"] not in (1, 2)
                or len(ep["observations"]) != len(ep["transitions"]) + 1
                or len(ep["prefixes"]) != len(ep["observations"])
            ):
                raise ValueError("Invalid replay schema")
            for t in ep["transitions"]:
                if (
                    not 0 < t["steps"] <= t["action"].shape[0]
                    or not torch.isfinite(t["action"]).all()
                ):
                    raise ValueError("Invalid action/steps")

    def observation(self, episode, index):
        obs = self.episodes[episode]["observations"][index]
        return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

    def prefix_batch(self, indices, device):
        samples = [self.episodes[e]["prefixes"][i].float() for e, i in indices]
        n = max(s.shape[0] for s in samples)
        x = torch.zeros(len(samples), n, samples[0].shape[1])
        mask = torch.zeros(len(samples), n, dtype=torch.bool)
        for j, s in enumerate(samples):
            x[j, : len(s)], mask[j, : len(s)] = s, True
        return x.to(device), mask.to(device)

    def summary(self):
        return dict(
            episodes=len(self.episodes),
            transitions=len(self.transitions),
            success_ratio=sum(ep["success"] for ep in self.episodes) / len(self.episodes),
        )

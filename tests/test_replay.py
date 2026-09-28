import tempfile
import unittest
from pathlib import Path

import torch

from qvgm.data.replay_buffer import ReplayBuffer
from qvgm.training import buffer_signature, proprio_features, proprio_stats


class ReplayTests(unittest.TestCase):
    def test_variable_proprio_prefixes_within_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            prefixes = [torch.randn(n, 4).bfloat16() for n in (3, 5, 4)]
            episode = dict(
                schema=2,
                task_id=0,
                episode=0,
                success=False,
                observations=[{"prompt": "test"}] * 3,
                prefixes=prefixes,
                transitions=[dict(action=torch.zeros(5, 7), steps=5)] * 2,
            )
            torch.save(episode, Path(directory) / "task00_episode000.pt")
            buffer = ReplayBuffer(directory)
            x, mask = buffer.prefix_batch([(0, 0), (0, 1), (0, 2)], "cpu")
            self.assertEqual(x.shape, (3, 5, 4))
            self.assertEqual(mask.sum(-1).tolist(), [3, 5, 4])
            for i, prefix in enumerate(prefixes):
                torch.testing.assert_close(x[i, : len(prefix)], prefix.float())

    def test_shard_roundtrip_terminal_and_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            for e, length in enumerate([3, 4]):
                record = dict(
                    schema=1,
                    task_id=e,
                    episode=0,
                    success=(e == 0),
                    observations=[{"prompt": "test", "state": torch.ones(8)}] * 2,
                    prefixes=torch.randn(2, length, 4).bfloat16(),
                    transitions=[
                        dict(
                            action=torch.zeros(5, 7),
                            steps=2,
                            reward=0.0,
                            terminated=(e == 0),
                            truncated=(e == 1),
                        )
                    ],
                )
                torch.save(record, Path(d) / f"task{e:02d}_episode000.pt")
            b = ReplayBuffer(d)
            x, m = b.prefix_batch([(0, 0), (1, 1)], "cpu")
            self.assertEqual(x.shape, (2, 4, 4))
            self.assertEqual(m.sum().item(), 7)
            self.assertEqual(b.summary()["success_ratio"], 0.5)
            self.assertEqual(len(buffer_signature(b)), 64)
            self.assertEqual(b.observation(0, 0)["state"].shape, (8,))

    def test_proprio_features_are_z_scored_over_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            states = [torch.tensor([0.0, 2.0]), torch.tensor([2.0, 4.0]), torch.tensor([4.0, 6.0])]
            record = dict(
                schema=2,
                task_id=0,
                episode=0,
                success=False,
                observations=[{"observation/state": s} for s in states],
                prefixes=[torch.randn(3, 4).bfloat16() for _ in states],
                transitions=[dict(action=torch.zeros(5, 7), steps=5) for _ in states[:2]],
            )
            torch.save(record, Path(d) / "task00_episode000.pt")
            buffer = ReplayBuffer(d)
            stats = proprio_stats(buffer)
            torch.testing.assert_close(stats["mean"], torch.tensor([2.0, 4.0]))
            feats = proprio_features(buffer, [(0, 0), (0, 2)], stats)
            self.assertEqual(feats.shape, (2, 2))
            torch.testing.assert_close(feats[0], -feats[1])


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from qvgm.algorithms.iql import IQL, chunk_target, expectile_loss
from qvgm.algorithms.q_guidance import improve_actions
from qvgm.algorithms.qvgm_loss import local_velocity_loss
from qvgm.models.critic import ChunkCritic, ConstantDirectionQ, Value
from qvgm.training import critic_state_dim, make_critic, proprio_enabled


class AlgorithmTests(unittest.TestCase):
    def test_bfloat16_velocity_preserves_small_guidance_target(self):
        class Actor:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(32.0))

            def predict_velocity(self, x, tau, context):
                return self.weight.to(torch.bfloat16).expand_as(x)

        class Q:
            def mean(self, z, x):
                return x.flatten(1).sum(-1)

        actor, reference = Actor(), Actor()
        loss, _ = local_velocity_loss(
            actor,
            reference,
            {},
            torch.zeros(1, 1),
            Q(),
            0.5,
            torch.zeros(1, 1, 1),
            dict(ascent_steps=1, ascent_step_size=0.05),
            horizon=1,
            action_dim=1,
        )
        # 32 + 0.1 rounds back to 32 in bf16, incorrectly erasing supervision.
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), 0.01, places=5)
        loss.backward()
        self.assertLess(actor.weight.grad.item(), 0)
        self.assertIsNone(reference.weight.grad)

    def test_per_sample_keep_best_and_displacement_bound(self):
        # First sample rejects the first proposal; it must remain frozen.
        a = torch.zeros(2, 1, 1, requires_grad=True)
        z = torch.tensor([[0.01], [1.0]])

        def q(z, x):
            return -(x.flatten(1) - z).square().sum(-1)

        improved, metrics = improve_actions(q, z, a, steps=3, alpha=0.1)
        torch.testing.assert_close(improved[:, 0, 0], torch.tensor([0.0, 0.3]))
        self.assertFalse(improved.requires_grad)
        self.assertIsNone(a.grad)
        self.assertGreater(metrics["q_gain"], 0)

    def test_zero_gradient_is_noop(self):
        a = torch.ones(2, 2, 3)
        result, _ = improve_actions(lambda z, x: x.flatten(1).sum(-1) * 0, torch.zeros(2, 1), a)
        torch.testing.assert_close(result, a)

    def test_terminal_and_truncated_chunk_discount(self):
        y = chunk_target(
            torch.tensor([1.0, 2.0]),
            torch.tensor([2, 3]),
            torch.tensor([True, False]),
            torch.tensor([10.0, 10.0]),
            0.5,
        )
        torch.testing.assert_close(y, torch.tensor([1.0, 3.25]))
        self.assertAlmostEqual(expectile_loss(torch.tensor([-1.0, 1.0]), 0.8).item(), 0.5)

    def test_critic_gradient_does_not_accumulate_parameter_grads(self):
        q = ChunkCritic(4, 2, 3, heads=2, widths=(16, 8))
        a = torch.randn(3, 2, 3)
        improved, _ = improve_actions(q.mean, torch.randn(3, 4), a)
        self.assertTrue(torch.isfinite(improved).all())
        self.assertTrue(all(p.grad is None for p in q.parameters()))

    def test_local_loss_detaches_trajectory_reference_and_target(self):
        class Actor:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def predict_velocity(self, x, tau, c):
                return x * self.weight

        class Q:
            def mean(self, z, x):
                return x.flatten(1).sum(-1)

        actor, ref = Actor(), Actor()
        x = torch.ones(2, 3, 4, requires_grad=True)
        loss, _ = local_velocity_loss(
            actor,
            ref,
            {},
            torch.zeros(2, 1),
            Q(),
            0.8,
            x,
            dict(ascent_steps=1, ascent_step_size=0.1),
            horizon=2,
            action_dim=3,
        )
        loss.backward()
        self.assertIsNotNone(actor.weight.grad)
        self.assertIsNone(ref.weight.grad)
        self.assertIsNone(x.grad)
        self.assertLess(actor.weight.grad.item(), 0)

    def test_value_target_aggregation_is_switchable(self):
        critic = ChunkCritic(2, 1, 1, heads=2, widths=(4,))
        value = Value(2, (4,))
        base = dict(q_lr=1e-3, value_lr=1e-3, target_ema=0.005, expectile=0.8)
        q = torch.tensor([[1.0, 3.0]])
        self.assertEqual(IQL(critic, value, base, 0.99).head_target(q).item(), 1.0)
        trainer = IQL(critic, value, dict(base, value_target="mean"), 0.99)
        self.assertEqual(trainer.head_target(q).item(), 2.0)
        with self.assertRaises(ValueError):
            IQL(critic, value, dict(base, value_target="median"), 0.99)

    def test_critic_proprio_widens_state_only(self):
        cfg = {
            "offline": {
                "critic": {"heads": 2, "widths": [8], "proprio": True},
                "rl_token": {"dim": 16},
            },
            "env": {"action_chunk": 5, "action_dim": 7},
        }
        self.assertTrue(proprio_enabled(cfg))
        self.assertEqual(critic_state_dim(cfg, 8), 24)
        critic = make_critic(cfg, 8)
        # 24-dim state plus the flattened 5x7 executed chunk.
        self.assertEqual(critic.heads[0].layers[0].in_features, 24 + 35)
        cfg["offline"]["critic"]["proprio"] = False
        self.assertFalse(proprio_enabled(cfg))
        self.assertEqual(critic_state_dim(cfg), 16)

    def test_constant_direction_q_displacement_is_state_independent(self):
        direction = torch.tensor([2.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        q = ConstantDirectionQ(direction, 2, 3)
        improved, metrics = improve_actions(
            q.mean, torch.randn(4, 5), torch.zeros(4, 2, 3), steps=3, alpha=0.05
        )
        expected = 0.15 * direction.reshape(2, 3) / direction.norm()
        for row in improved:
            torch.testing.assert_close(row, expected, atol=1e-6, rtol=0)
        self.assertAlmostEqual(metrics["displacement"], 0.15, places=5)
        self.assertEqual(metrics["accept_rate"], 1.0)
        with self.assertRaises(ValueError):
            ConstantDirectionQ(direction[:4], 2, 3)


if __name__ == "__main__":
    unittest.main()

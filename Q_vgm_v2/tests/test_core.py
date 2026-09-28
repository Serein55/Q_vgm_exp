import unittest

import torch
from qvgm_v2.algorithms.stepwise_iql import iql_losses, td_targets, update_target
from qvgm_v2.data.buffer import stepwise_transition
from qvgm_v2.models.critic import DoubleStepwiseCritic, StepwiseValue
from qvgm_v2.models.state_encoder import CriticStateEncoder
from torch import nn


class CoreTests(unittest.TestCase):
    def episode(self, n=5, terminated=False, truncated=False):
        return {
            "transitions": [
                dict(
                    action=torch.zeros(5, 7),
                    rewards=torch.zeros(n),
                    steps=n,
                    terminated=terminated,
                    truncated=truncated,
                )
            ]
        }

    def batch(self, transition):
        return {k: v.unsqueeze(0) for k, v in transition.items() if isinstance(v, torch.Tensor)}

    def test_terminal_target_and_partial(self):
        e = self.episode(3, terminated=True)
        e["transitions"][0]["rewards"][-1] = 1
        b = self.batch(stepwise_transition(e, 0, timeout_terminal=True))
        target, mask = td_targets(
            b["rewards"],
            b["dones"],
            b["valid"],
            b["boundary_valid"],
            torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]),
            torch.ones(1, 5),
            0.9,
        )
        torch.testing.assert_close(target[0, :3], torch.tensor([18.0, 27.0, 1.0]))
        self.assertEqual(mask.tolist(), [[True, True, True, False, False]])

    def test_boundary(self):
        e = self.episode()
        e["transitions"] += self.episode()["transitions"]
        b = self.batch(stepwise_transition(e, 0, timeout_terminal=True))
        y, _ = td_targets(
            b["rewards"],
            b["dones"],
            b["valid"],
            b["boundary_valid"],
            torch.ones(1, 5),
            torch.full((1, 5), 7.0),
            0.9,
        )
        self.assertAlmostEqual(y[0, -1].item(), 6.3, places=5)
        e["transitions"][1] = self.episode(3, terminated=True)["transitions"][0]
        b = self.batch(stepwise_transition(e, 0, timeout_terminal=True))
        y, _ = td_targets(
            b["rewards"],
            b["dones"],
            b["valid"],
            b["boundary_valid"],
            torch.ones(1, 5),
            torch.full((1, 5), 7.0),
            0.9,
        )
        self.assertEqual(y[0, -1].item(), 0)

    def test_timeout_choice(self):
        e = self.episode(3, truncated=True)
        for terminal in (True, False):
            b = self.batch(stepwise_transition(e, 0, timeout_terminal=terminal))
            _, mask = td_targets(
                b["rewards"],
                b["dones"],
                b["valid"],
                b["boundary_valid"],
                torch.ones(1, 5),
                torch.ones(1, 5),
                0.99,
            )
            self.assertEqual(bool(mask[0, 2]), terminal)

    def test_full_terminal(self):
        b = self.batch(stepwise_transition(self.episode(terminated=True), 0, timeout_terminal=True))
        b["rewards"][0, -1] = 1
        y, mask = td_targets(
            b["rewards"],
            b["dones"],
            b["valid"],
            b["boundary_valid"],
            torch.arange(5.0).unsqueeze(0),
            torch.ones(1, 5),
            0.9,
        )
        torch.testing.assert_close(y, torch.tensor([[0.9, 1.8, 2.7, 3.6, 1.0]]))
        self.assertTrue(mask.all())

    def test_losses_gradient_isolation(self):
        q1, q2, v, tq1, tq2, nv = [torch.ones(1, 5, requires_grad=True) for _ in range(6)]
        b = self.batch(stepwise_transition(self.episode(terminated=True), 0, timeout_terminal=True))
        ql, vl = iql_losses(q1, q2, v, tq1, tq2, nv, b)
        ql.backward()
        self.assertIsNone(v.grad)
        self.assertIsNone(nv.grad)
        vl.backward()
        self.assertIsNotNone(v.grad)
        self.assertIsNone(tq1.grad)
        self.assertIsNone(tq2.grad)

    def test_state_and_action_grad(self):
        state = CriticStateEncoder()(torch.randn(2, 2048), torch.randn(2, 8))
        self.assertEqual(state.shape, (2, 2304))
        q = DoubleStepwiseCritic(widths=(16, 8))
        a = torch.randn(2, 5, 7, requires_grad=True)
        q.score(state, a).sum().backward()
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertEqual(StepwiseValue(widths=(16, 8))(state).shape, (2, 5))

    def test_positionwise_min(self):
        class Fixed(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = torch.tensor([values])

            def forward(self, *args):
                return self.values

        q = DoubleStepwiseCritic(widths=(4,))
        q.q1, q.q2 = Fixed([0.0, 10.0, 0.0, 10.0, 0.0]), Fixed([10.0, 0.0, 10.0, 0.0, 10.0])
        self.assertEqual(q.score(None, None).item(), 0)

    def test_ema(self):
        a, b = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            a.weight.fill_(0)
            b.weight.fill_(4)
        update_target(a, b, 0.25)
        self.assertEqual(a.weight.item(), 1)

    def test_reject_invalid_rewards(self):
        e = self.episode(3)
        e["transitions"][0]["rewards"] = torch.zeros(5)
        with self.assertRaises(ValueError):
            stepwise_transition(e, 0, timeout_terminal=True)


if __name__ == "__main__":
    unittest.main()

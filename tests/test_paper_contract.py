"""Analytic checks for reference targets and explicit configuration boundaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from qvgm.algorithms.qvgm_loss import local_velocity_loss
from qvgm.config import ROOT, load_config
from qvgm.models.pi05_adapter import Pi05Flow


class PaperContractTests(unittest.TestCase):
    def test_reference_look_forward_and_inverse_remaining_time(self):
        class Flow:
            def __init__(self, value):
                self.weight = torch.nn.Parameter(torch.tensor(value))

            def predict_velocity(self, x, tau, context):
                return self.weight.expand_as(x)

        class Q:
            def mean(self, z, x):
                return -(x.flatten(1) - 1.45).square().sum(-1)

        actor, reference = Flow(3.0), Flow(2.0)
        x = torch.ones(1, 1, 1, requires_grad=True)
        loss, _ = local_velocity_loss(
            actor,
            reference,
            {},
            torch.zeros(1, 1),
            Q(),
            0.8,
            x,
            dict(ascent_steps=1, ascent_step_size=0.05),
            horizon=1,
            action_dim=1,
        )
        # Reference endpoint=1.4; accepted endpoint=1.45; velocity target=2.25.
        self.assertAlmostEqual(loss.item(), (3.0 - 2.25) ** 2, places=5)
        loss.backward()
        self.assertAlmostEqual(actor.weight.grad.item(), 1.5, places=5)
        self.assertIsNone(reference.weight.grad)
        self.assertIsNone(x.grad)

    def test_paper_overlay_preserves_paths_and_changes_explicit_contract(self):
        legacy = load_config()
        paper = load_config(ROOT / "configs/libero_spatial_paper.yaml")
        self.assertEqual(paper["paths"], legacy["paths"])
        self.assertEqual(paper["model"]["action_horizon"], paper["env"]["action_chunk"])
        self.assertTrue(paper["model"]["discrete_state_input"])
        self.assertEqual(paper["offline"]["actor"]["master_dtype"], "checkpoint")
        self.assertEqual(paper["offline"]["actor"]["ascent_step_size"], 0.05)

    def test_config_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "cycle.yaml"
            p.write_text("extends: cycle.yaml\n")
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                load_config(p)

    def test_checkpoint_mode_preserves_mixed_parameter_dtypes(self):
        model = torch.nn.Module()
        model.action_in_proj = torch.nn.Linear(2, 2).to(torch.bfloat16)
        model.action_out_proj = torch.nn.Linear(2, 2)
        model.frozen_backbone = torch.nn.Linear(2, 2)
        flow = Pi05Flow(SimpleNamespace(_model=model), {"runtime": {"device": "cpu"}})
        flow.enable_actor_training("checkpoint")
        self.assertEqual(model.action_in_proj.weight.dtype, torch.bfloat16)
        self.assertEqual(model.action_out_proj.weight.dtype, torch.float32)
        self.assertTrue(model.action_in_proj.weight.requires_grad)
        self.assertFalse(model.frozen_backbone.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()

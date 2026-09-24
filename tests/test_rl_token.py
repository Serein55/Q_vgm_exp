import unittest

import torch

from qvgm.models.rl_token import RLTTokenTransformer


class RLTokenTests(unittest.TestCase):
    def test_masked_reconstruction_and_frozen_input(self):
        torch.manual_seed(1)
        model = RLTTokenTransformer(
            input_dim=4, embed_dim=8, prefix_seq_len=5, num_layers=2, num_heads=2, mlp_ratio=1.0
        )
        x = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        loss, values = model.loss(x, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(values["z_rl"].shape, (2, 8))
        loss.backward()
        self.assertIsNone(x.grad)
        self.assertGreater(sum(p.grad.abs().sum() for p in model.encoder.parameters()).item(), 0.0)
        self.assertGreater(sum(p.grad.abs().sum() for p in model.decoder.parameters()).item(), 0.0)
        with torch.no_grad():
            original = model.encode_flat(x, mask)
            modified = x.detach().clone()
            modified[0, 3:] = 1000
            torch.testing.assert_close(model.encode_flat(modified, mask), original)


if __name__ == "__main__":
    unittest.main()

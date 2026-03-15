from __future__ import annotations

import unittest

import torch

from ngas import DifferentiableNeuralGas as TopLevelDifferentiableNeuralGas
from ngas.models import DifferentiableNeuralGas


class DifferentiableNeuralGasTests(unittest.TestCase):
    def test_constructor_and_exports(self) -> None:
        model = DifferentiableNeuralGas(n_neurons=12, input_dim=3, distance="l2")
        self.assertEqual(model.n_neurons, 12)
        self.assertEqual(model.input_dim, 3)

        top = TopLevelDifferentiableNeuralGas(n_neurons=6, input_dim=2)
        self.assertEqual(top.n_neurons, 6)

    def test_init_points_sets_weights_and_infers_dimension(self) -> None:
        init_points = torch.randn(12, 4)
        model = DifferentiableNeuralGas(n_neurons=12, init_points=init_points)
        self.assertEqual(model.input_dim, 4)
        self.assertTrue(torch.allclose(model.weights.detach(), init_points))

    def test_init_points_rejects_wrong_row_count(self) -> None:
        with self.assertRaises(ValueError):
            DifferentiableNeuralGas(n_neurons=12, init_points=torch.randn(10, 2))

    def test_init_points_rejects_dimension_conflict(self) -> None:
        with self.assertRaises(ValueError):
            DifferentiableNeuralGas(n_neurons=12, input_dim=3, init_points=torch.randn(12, 2))

    def test_forward_returns_scalar_loss(self) -> None:
        torch.manual_seed(0)
        model = DifferentiableNeuralGas(n_neurons=10, input_dim=2, distance="sq_l2")
        data = torch.randn(32, 2)
        loss = model(data)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_forward_details_shapes(self) -> None:
        torch.manual_seed(1)
        model = DifferentiableNeuralGas(
            n_neurons=7,
            input_dim=4,
            neighborhood="inverse",
            rank_temperature=0.3,
        )
        data = torch.randn(20, 4)
        out = model(data, return_details=True)

        self.assertIn("loss", out)
        self.assertIn("distances", out)
        self.assertIn("soft_rank", out)
        self.assertIn("neighborhood", out)
        self.assertEqual(out["distances"].shape, (20, 7))
        self.assertEqual(out["soft_rank"].shape, (20, 7))
        self.assertEqual(out["neighborhood"].shape, (20, 7))

    def test_predict_and_quantization_error(self) -> None:
        torch.manual_seed(2)
        model = DifferentiableNeuralGas(n_neurons=9, input_dim=2, distance="cosine")
        data = torch.randn(40, 2)

        preds = model.predict(data)
        qerr = model.quantization_error(data)
        self.assertEqual(preds.shape, (40,))
        self.assertGreaterEqual(qerr, 0.0)


if __name__ == "__main__":
    unittest.main()

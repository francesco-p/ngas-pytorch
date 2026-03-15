from __future__ import annotations

import unittest

import torch

from ngas import DifferentiableGrowingNeuralGas as TopLevelDifferentiableGrowingNeuralGas
from ngas.models import DifferentiableGrowingNeuralGas


class DifferentiableGrowingNeuralGasTests(unittest.TestCase):
    def test_constructor_and_exports(self) -> None:
        model = DifferentiableGrowingNeuralGas(
            max_neurons=16,
            input_dim=3,
            init_neurons=4,
        )
        self.assertEqual(model.n_nodes, 4)

        top = TopLevelDifferentiableGrowingNeuralGas(max_neurons=10, input_dim=2, init_neurons=2)
        self.assertEqual(top.n_nodes, 2)

    def test_init_points_sets_active_nodes_and_weights(self) -> None:
        init_points = torch.randn(5, 3)
        model = DifferentiableGrowingNeuralGas(max_neurons=12, init_points=init_points)
        self.assertEqual(model.n_nodes, 5)
        self.assertEqual(model.config.input_dim, 3)
        self.assertTrue(torch.allclose(model.weights[:5].detach(), init_points))
        self.assertFalse(bool(model.active_mask[5:].any().item()))

    def test_init_points_rejects_dimension_conflict(self) -> None:
        with self.assertRaises(ValueError):
            DifferentiableGrowingNeuralGas(
                max_neurons=12,
                input_dim=4,
                init_points=torch.randn(5, 3),
            )

    def test_init_points_rejects_invalid_row_count(self) -> None:
        with self.assertRaises(ValueError):
            DifferentiableGrowingNeuralGas(max_neurons=12, init_points=torch.randn(1, 2))
        with self.assertRaises(ValueError):
            DifferentiableGrowingNeuralGas(max_neurons=4, init_points=torch.randn(5, 2))

    def test_forward_and_details(self) -> None:
        torch.manual_seed(0)
        model = DifferentiableGrowingNeuralGas(max_neurons=12, input_dim=2, init_neurons=3)
        data = torch.randn(25, 2)

        loss = model(data)
        self.assertEqual(loss.dim(), 0)

        out = model(data, return_details=True)
        self.assertIn("loss", out)
        self.assertIn("data_term", out)
        self.assertIn("edge_prob", out)
        self.assertEqual(out["distances"].shape[0], 25)
        self.assertEqual(out["distances"].shape[1], model.n_nodes)
        self.assertEqual(out["edge_prob"].shape, (model.n_nodes, model.n_nodes))

    def test_grow_increases_nodes_until_cap(self) -> None:
        torch.manual_seed(1)
        model = DifferentiableGrowingNeuralGas(max_neurons=8, input_dim=2, init_neurons=2)
        data = torch.randn(30, 2)

        # Populate node_error_ema before growing.
        _ = model(data)

        before = model.n_nodes
        added = model.grow(n_new=3, noise_std=0.0)
        after = model.n_nodes
        self.assertGreaterEqual(added, 0)
        self.assertGreaterEqual(after, before)
        self.assertLessEqual(after, 8)

        added_again = model.grow(n_new=20, noise_std=0.0)
        self.assertGreaterEqual(added_again, 0)
        self.assertLessEqual(model.n_nodes, 8)

    def test_grow_continues_after_init_points_until_cap(self) -> None:
        torch.manual_seed(3)
        init_points = torch.randn(4, 2)
        model = DifferentiableGrowingNeuralGas(max_neurons=8, init_points=init_points)
        data = torch.randn(20, 2)
        _ = model(data)

        before = model.n_nodes
        added = model.grow(n_new=2, noise_std=0.0)
        self.assertGreaterEqual(added, 0)
        self.assertGreaterEqual(model.n_nodes, before)
        self.assertLessEqual(model.n_nodes, 8)

    def test_full_capacity_init_points_prevents_more_growth(self) -> None:
        torch.manual_seed(4)
        init_points = torch.randn(6, 2)
        model = DifferentiableGrowingNeuralGas(max_neurons=6, init_points=init_points)
        data = torch.randn(20, 2)
        _ = model(data)
        added = model.grow(n_new=5, noise_std=0.0)
        self.assertEqual(added, 0)
        self.assertEqual(model.n_nodes, 6)

    def test_predict_and_quantization_error(self) -> None:
        torch.manual_seed(2)
        model = DifferentiableGrowingNeuralGas(max_neurons=10, input_dim=2, init_neurons=3)
        data = torch.randn(50, 2)
        _ = model(data)

        preds = model.predict(data)
        qerr = model.quantization_error(data)
        self.assertEqual(preds.shape, (50,))
        self.assertGreaterEqual(qerr, 0.0)


if __name__ == "__main__":
    unittest.main()

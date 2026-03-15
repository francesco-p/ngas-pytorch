from __future__ import annotations

import unittest

import torch

from ngas.models import GrowingNeuralGas


class GrowingNeuralGasTests(unittest.TestCase):
    def test_alias_n_neurons_sets_max_neurons(self) -> None:
        model = GrowingNeuralGas(n_neurons=12)
        self.assertEqual(model.config.max_neurons, 12)

    def test_init_points_starts_with_all_provided_nodes(self) -> None:
        init_points = torch.randn(5, 3)
        model = GrowingNeuralGas(max_neurons=12, init_points=init_points)
        self.assertEqual(model.n_nodes, 5)
        self.assertEqual(model.input_dim, 3)
        self.assertTrue(torch.allclose(model.weights, init_points))

    def test_init_points_rejects_wrong_dimension_conflict(self) -> None:
        with self.assertRaises(ValueError):
            GrowingNeuralGas(max_neurons=12, input_dim=4, init_points=torch.randn(5, 3))

    def test_init_points_rejects_invalid_row_count(self) -> None:
        with self.assertRaises(ValueError):
            GrowingNeuralGas(max_neurons=12, init_points=torch.randn(1, 2))
        with self.assertRaises(ValueError):
            GrowingNeuralGas(max_neurons=4, init_points=torch.randn(5, 2))

    def test_update_initializes(self) -> None:
        torch.manual_seed(3)
        model = GrowingNeuralGas(max_neurons=10, lambda_steps=4)
        sample = torch.randn(2)
        adj, w = model.update(sample)
        self.assertEqual(adj.shape[0], w.shape[0])
        self.assertEqual(w.shape[1], 2)

    def test_growth_happens_when_lambda_reached(self) -> None:
        torch.manual_seed(4)
        model = GrowingNeuralGas(
            max_neurons=16,
            lambda_steps=2,
            lr_winner=0.05,
            lr_neighbor=0.01,
            max_edge_age=20,
        )
        data = torch.randn(64, 2)

        model.fit(data, epochs=1, shuffle=False)
        self.assertGreaterEqual(model.n_nodes, 3)
        self.assertLessEqual(model.n_nodes, 16)

    def test_growth_can_continue_after_init_points(self) -> None:
        torch.manual_seed(4)
        init_points = torch.randn(4, 2)
        data = torch.randn(64, 2)
        model = GrowingNeuralGas(max_neurons=8, lambda_steps=2, init_points=init_points)
        before = model.n_nodes
        model.fit(data, epochs=1, shuffle=False)
        self.assertGreaterEqual(model.n_nodes, before)
        self.assertLessEqual(model.n_nodes, 8)

    def test_full_capacity_init_points_prevents_more_growth(self) -> None:
        torch.manual_seed(6)
        init_points = torch.randn(6, 2)
        model = GrowingNeuralGas(max_neurons=6, lambda_steps=1, init_points=init_points)
        model._insert_new_node()
        self.assertEqual(model.n_nodes, 6)

    def test_predict_and_quantization_error(self) -> None:
        torch.manual_seed(5)
        model = GrowingNeuralGas(max_neurons=10, lambda_steps=3)
        data = torch.randn(50, 3)
        model.fit(data, epochs=1, shuffle=False)

        preds = model.predict(data)
        qerr = model.quantization_error(data)
        self.assertEqual(preds.shape, (50,))
        self.assertGreaterEqual(qerr, 0.0)


if __name__ == "__main__":
    unittest.main()

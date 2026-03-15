from __future__ import annotations

import unittest

import torch

from ngas import InverseNeuralGas as TopLevelInverseNeuralGas
from ngas.models import InverseNeuralGas


class InverseNeuralGasTests(unittest.TestCase):
    def test_constructor_matches_requested_api(self) -> None:
        model = InverseNeuralGas(n_neurons=10, lr=0.02, max_edge_age=100, distance="l2")
        self.assertEqual(model.n_neurons, 10)

    def test_init_points_sets_weights_and_infers_dimension(self) -> None:
        init_points = torch.randn(10, 3)
        model = InverseNeuralGas(n_neurons=10, max_edge_age=16, init_points=init_points)
        self.assertEqual(model.input_dim, 3)
        self.assertTrue(torch.allclose(model.weights, init_points))

    def test_init_points_rejects_wrong_row_count(self) -> None:
        with self.assertRaises(ValueError):
            InverseNeuralGas(n_neurons=10, max_edge_age=16, init_points=torch.randn(11, 2))

    def test_init_points_rejects_dimension_conflict(self) -> None:
        with self.assertRaises(ValueError):
            InverseNeuralGas(
                n_neurons=10,
                max_edge_age=16,
                input_dim=4,
                init_points=torch.randn(10, 2),
            )

    def test_top_level_export(self) -> None:
        model = TopLevelInverseNeuralGas(n_neurons=6, lr=0.02, max_edge_age=16, distance="l2")
        self.assertEqual(model.n_neurons, 6)

    def test_update_initializes_and_updates_shapes(self) -> None:
        torch.manual_seed(0)
        model = InverseNeuralGas(n_neurons=8, lr=0.05, max_edge_age=16, distance="l2")
        sample = torch.randn(2)
        adj, w = model.update(sample)
        self.assertEqual(adj.shape, (8, 8))
        self.assertEqual(w.shape, (8, 2))

    def test_fit_changes_weights(self) -> None:
        torch.manual_seed(1)
        model = InverseNeuralGas(n_neurons=6, lr=0.05, max_edge_age=8, distance="sq_l2")
        data = torch.randn(32, 3)

        model.update(data[0])
        before = model.weights.clone()
        model.fit(data[1:], epochs=2, shuffle=False)

        self.assertFalse(torch.allclose(before, model.weights))

    def test_fit_changes_weights_from_init_points(self) -> None:
        torch.manual_seed(8)
        data = torch.randn(32, 3)
        model = InverseNeuralGas(
            n_neurons=6,
            lr=0.05,
            max_edge_age=8,
            distance="sq_l2",
            init_points=data[:6],
        )
        before = model.weights.clone()
        model.fit(data[6:], epochs=2, shuffle=False)
        self.assertFalse(torch.allclose(before, model.weights))

    def test_predict_and_quantization_error(self) -> None:
        torch.manual_seed(2)
        data = torch.randn(40, 4)
        model = InverseNeuralGas(n_neurons=5, lr=0.03, max_edge_age=10, distance="cosine")
        model.fit(data, epochs=1, shuffle=False)

        preds = model.predict(data)
        qerr = model.quantization_error(data)
        self.assertEqual(preds.shape, (40,))
        self.assertGreaterEqual(qerr, 0.0)

    def test_update_without_topology_keeps_adjacency_unchanged(self) -> None:
        torch.manual_seed(3)
        model = InverseNeuralGas(
            n_neurons=7,
            lr=0.03,
            max_edge_age=10,
            distance="l2",
            update_topology=False,
        )
        sample = torch.randn(3)
        initial_adj = model.adj.clone()
        adj, _ = model.update(sample)

        self.assertTrue(torch.equal(adj, initial_adj))


if __name__ == "__main__":
    unittest.main()

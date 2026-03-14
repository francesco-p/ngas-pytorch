from __future__ import annotations

import unittest

import torch

from ngas.models.ngas import NeuralGas


class NeuralGasTests(unittest.TestCase):
    def test_constructor_matches_requested_api(self) -> None:
        model = NeuralGas(n_neurons=10, lr=0.02, max_edge_age=100, distance="l2")
        self.assertEqual(model.n_neurons, 10)

    def test_update_initializes_and_updates_shapes(self) -> None:
        torch.manual_seed(0)
        model = NeuralGas(n_neurons=8, lr=0.05, max_edge_age=16, distance="l2")
        sample = torch.randn(2)
        adj, w = model.update(sample)
        self.assertEqual(adj.shape, (8, 8))
        self.assertEqual(w.shape, (8, 2))

    def test_fit_changes_weights(self) -> None:
        torch.manual_seed(1)
        model = NeuralGas(n_neurons=6, lr=0.05, max_edge_age=8, distance="sq_l2")
        data = torch.randn(32, 3)

        model.update(data[0])
        before = model.weights.clone()
        model.fit(data[1:], epochs=2, shuffle=False)

        self.assertFalse(torch.allclose(before, model.weights))

    def test_predict_and_quantization_error(self) -> None:
        torch.manual_seed(2)
        data = torch.randn(40, 4)
        model = NeuralGas(n_neurons=5, lr=0.03, max_edge_age=10, distance="cosine")
        model.fit(data, epochs=1, shuffle=False)

        preds = model.predict(data)
        qerr = model.quantization_error(data)
        self.assertEqual(preds.shape, (40,))
        self.assertGreaterEqual(qerr, 0.0)

    def test_update_without_topology_keeps_adjacency_unchanged(self) -> None:
        torch.manual_seed(3)
        model = NeuralGas(
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

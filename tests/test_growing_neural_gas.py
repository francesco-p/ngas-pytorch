from __future__ import annotations

import unittest

import torch

from ngas.models import GrowingNeuralGas


class GrowingNeuralGasTests(unittest.TestCase):
    def test_alias_n_neurons_sets_max_neurons(self) -> None:
        model = GrowingNeuralGas(n_neurons=12)
        self.assertEqual(model.config.max_neurons, 12)

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

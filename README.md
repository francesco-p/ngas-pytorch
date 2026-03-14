# ngas-pytorch

Minimal PyTorch library for classical Neural Gas and Growing Neural Gas.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import torch
from ngas.models import GrowingNeuralGas, InverseNeuralGas, NeuralGas

x = torch.randn(128, 2)

# Original NG formulation (rank neighborhood: exp(-k / lambda))
model = NeuralGas(n_neurons=10, lr=0.02, max_edge_age=100, distance="l2")
model.fit(x, epochs=5)
print(model.weights.shape)  # [10, 2]

# Inverse variant (rank neighborhood: 1 / rank^2)
inv = InverseNeuralGas(n_neurons=10, lr=0.02, max_edge_age=100, distance="l2")
inv.fit(x, epochs=5)


gng = GrowingNeuralGas(max_neurons=32, lambda_steps=50, max_edge_age=100)
gng.fit(x, epochs=3)
print(gng.weights.shape)    # [<=32, 2]
```

## Distances

Supported values:

- `"l2"` (aliases: `"euclidean"`)
- `"sq_l2"` (aliases: `"sqeuclidean"`)
- `"cosine"` (alias: `"cos"`)

## Examples

Run from `ngas-pytorch/`:

```bash
python3 examples/neural_gas_blobs.py --plot
python3 examples/growing_neural_gas_blobs.py --plot
```

Both scripts print summary metrics:

- `neurons`/`nodes`
- `edges`
- `quantization_error`

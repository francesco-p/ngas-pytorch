# ngas-pytorch

<p align="center">
  <img src="examples/banner.png" alt="ngas-pytorch banner" width="900">
</p>

Minimal PyTorch implementations of Neural Gas, Inverse Neural Gas, and Growing Neural Gas for topology-aware vector quantization and graph learning.

## Overview

This repository provides lightweight PyTorch implementations of:

- `NeuralGas`: the classical rank-based Neural Gas algorithm
- `InverseNeuralGas`: an inverse-rank variant with `1 / rank^2` neighborhood weighting
- `GrowingNeuralGas`: a graph-growing version that adapts both topology and model size

The code is designed to stay small and easy to read while still being practical for experiments on CPU or GPU tensors.

## Install

Install the library itself:

```bash
pip install -e .
```

The example notebooks install `matplotlib` inside the notebook so plotting stays out of the base package dependencies.

## Quickstart

```python
import torch
from ngas.models import GrowingNeuralGas, InverseNeuralGas, NeuralGas

x = torch.randn(128, 2)

# Classical Neural Gas
ng = NeuralGas(
    n_neurons=10,
    lr=0.02,
    max_edge_age=100,
    distance="l2",
)
ng.fit(x, epochs=5)
labels = ng.predict(x)
print(ng.weights.shape)           # [10, 2]
print(ng.quantization_error(x))   # average winner distance

# Inverse-rank variant
inv = InverseNeuralGas(
    n_neurons=10,
    lr=0.02,
    max_edge_age=100,
    distance="l2",
)
inv.fit(x, epochs=5)

# Growing Neural Gas
gng = GrowingNeuralGas(
    max_neurons=32,
    lambda_steps=50,
    max_edge_age=100,
    distance="l2",
)
gng.fit(x, epochs=3)
print(gng.weights.shape)          # [<=32, 2]
```

## API Notes

All models expose the same high-level workflow:

1. Create a model with a distance metric and algorithm-specific hyperparameters.
2. Call `fit(data, epochs=..., shuffle=True)`.
3. Use `predict(data)` for winner assignments.
4. Use `quantization_error(data)` as a compact quality metric.

Inputs are standard PyTorch tensors of shape `[N, D]`.

`NeuralGas` and `InverseNeuralGas` also support online updates through `update(sample)`, which is useful for streaming data.

## Streaming Example

For `InverseNeuralGas`, you do not need to call `fit(...)` if your data arrives one sample at a time:

```python
import torch
from ngas.models import InverseNeuralGas

stream = [
    torch.tensor([0.2, -0.1]),
    torch.tensor([0.0, 0.3]),
    torch.tensor([1.1, 0.9]),
]

# input_dim lets you initialize the model before the first sample arrives.
model = InverseNeuralGas(
    n_neurons=16,
    lr=0.03,
    max_edge_age=32,
    distance="l2",
    input_dim=2,
)

for sample in stream:
    adj, weights = model.update(sample)

print(weights.shape)                 # [16, 2]
print(model.predict(torch.stack(stream)))
```

If you prefer, you can also omit `input_dim`; the first call to `update(sample)` will initialize the model automatically from that sample's dimension.

## Distances

Supported distance names:

- `"l2"` (alias: `"euclidean"`)
- `"sq_l2"` (alias: `"sqeuclidean"`)
- `"cosine"` (alias: `"cos"`)

These helpers are also available from the top-level package:

```python
from ngas import SUPPORTED_DISTANCES, normalize_distance_name, pairwise_distance
```

## Examples

Open either notebook from the repository root in Jupyter or VS Code:

```bash
jupyter lab examples/neural_gas_blobs.ipynb
jupyter lab examples/growing_neural_gas_blobs.ipynb
```

Each notebook:

- installs `matplotlib` in a notebook cell
- trains the model on synthetic 2D blobs
- prints summary metrics
- renders the learned graph inline

Reported metrics include:

- `neurons` or `nodes`
- `edges`
- `quantization_error`

## References

The implementations in this repository are based on the original papers:

- Thomas Martinetz and Klaus Schulten, "A 'Neural-Gas' Network Learns Topologies," in *Artificial Neural Networks*, 1991. [PDF](https://www.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf)
- Bernd Fritzke, "A Growing Neural Gas Network Learns Topologies," in *Advances in Neural Information Processing Systems 7*, 1994. [PDF](https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf)

## Citation

If this repository helps your work, you can cite it as:

```bibtex
@software{francesco_p_2026_ngas_pytorch,
  author = {francesco-p},
  title = {ngas-pytorch: PyTorch implementations of Neural Gas and Growing Neural Gas},
  year = {2026},
  url = {https://github.com/francesco-p/ngas-pytorch}
}
```

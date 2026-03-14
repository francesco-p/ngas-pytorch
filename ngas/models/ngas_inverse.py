from __future__ import annotations

from dataclasses import dataclass

import torch

from ngas.metrics import normalize_distance_name, pairwise_distance, point_to_set_distance


@dataclass(frozen=True)
class NeuralGasConfig:
    n_neurons: int = 32
    lr: float = 0.05
    max_edge_age: int = 64
    distance: str = "l2"
    input_dim: int | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    update_topology: bool = True


class InverseNeuralGas:
    """Online Neural Gas variant with inverse squared-rank neighborhood decay."""

    def __init__(
        self,
        n_neurons: int = 32,
        lr: float = 0.05,
        max_edge_age: int = 64,
        distance: str = "l2",
        input_dim: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        update_topology: bool = True,
    ) -> None:
        if n_neurons < 2:
            raise ValueError("n_neurons must be >= 2.")
        if lr <= 0:
            raise ValueError("lr must be > 0.")
        if max_edge_age < 1:
            raise ValueError("max_edge_age must be >= 1.")
        if input_dim is not None and input_dim < 1:
            raise ValueError("input_dim must be >= 1 when provided.")
        if not isinstance(update_topology, bool):
            raise ValueError("update_topology must be a bool.")

        self.config = NeuralGasConfig(
            n_neurons=n_neurons,
            lr=lr,
            max_edge_age=max_edge_age,
            distance=normalize_distance_name(distance),
            input_dim=input_dim,
            device=torch.device(device),
            dtype=dtype,
            update_topology=update_topology,
        )

        self.weights: torch.Tensor | None = None
        self.adj: torch.Tensor = torch.full(
            (n_neurons, n_neurons),
            fill_value=-1,
            dtype=torch.int64,
            device=self.config.device,
        )
        self._initialized = False

        if input_dim is not None:
            self._initialize(input_dim)

    @property
    def n_neurons(self) -> int:
        return self.config.n_neurons

    @property
    def input_dim(self) -> int | None:
        if not self._initialized or self.weights is None:
            return None
        return int(self.weights.size(1))

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    def _initialize(self, input_dim: int) -> None:
        self.weights = torch.rand(
            self.n_neurons,
            input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.adj.fill_(-1)
        self.adj.fill_diagonal_(-1)
        self._initialized = True

    def _ensure_initialized(self, sample: torch.Tensor) -> None:
        if self._initialized:
            return
        self._initialize(sample.numel())

    def _coerce_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample, dtype=self.dtype, device=self.device)
        sample = sample.to(device=self.device, dtype=self.dtype)
        if sample.dim() == 2 and sample.size(0) == 1:
            sample = sample.squeeze(0)
        if sample.dim() != 1:
            raise ValueError("sample must have shape [D] or [1, D].")
        return sample

    @torch.no_grad()
    def update(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._coerce_sample(sample)
        self._ensure_initialized(sample)
        assert self.weights is not None

        if sample.numel() != self.weights.size(1):
            raise ValueError(
                f"sample has dimension {sample.numel()}, but model expects {self.weights.size(1)}."
            )

        distances = point_to_set_distance(self.weights, sample, distance=self.config.distance)
        rank = torch.argsort(distances)

        # Inverse squared-rank neighborhood by rank (winner rank is 1 here).
        influence = (
            1.0
            / torch.arange(
                1,
                rank.numel() + 1,
                device=self.device,
                dtype=self.dtype,
            ).pow(2)
        ).unsqueeze(-1)

        sorted_weights = self.weights[rank]
        sorted_weights.add_(self.config.lr * influence * (sample - sorted_weights))
        self.weights[rank] = sorted_weights

        if self.config.update_topology:
            active = self.adj >= 0
            self.adj[active] += 1

            winner = int(rank[0].item())
            runner_up = int(rank[1].item())
            self.adj[winner, runner_up] = 0
            self.adj[runner_up, winner] = 0

            old_edges = self.adj > self.config.max_edge_age
            self.adj[old_edges] = -1
            self.adj.fill_diagonal_(-1)
        return self.adj, self.weights

    @torch.no_grad()
    def fit(self, data: torch.Tensor, epochs: int = 1, shuffle: bool = True) -> "InverseNeuralGas":
        data = self._coerce_batch(data)
        for _ in range(epochs):
            if shuffle:
                indices = torch.randperm(data.size(0), device=data.device)
                data_epoch = data[indices]
            else:
                data_epoch = data
            for sample in data_epoch:
                self.update(sample)
        return self

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        self._require_initialized()
        assert self.weights is not None
        data = self._coerce_batch(data)
        distances = pairwise_distance(data, self.weights, distance=self.config.distance)
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def quantization_error(self, data: torch.Tensor) -> float:
        self._require_initialized()
        assert self.weights is not None
        data = self._coerce_batch(data)
        distances = pairwise_distance(data, self.weights, distance=self.config.distance)
        min_dist = torch.min(distances, dim=1).values
        return float(min_dist.mean().item())

    def _coerce_batch(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=self.dtype, device=self.device)
        data = data.to(device=self.device, dtype=self.dtype)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.dim() != 2:
            raise ValueError("data must have shape [N, D] or [D].")
        if self._initialized and self.weights is not None and data.size(1) != self.weights.size(1):
            raise ValueError(
                f"data has dimension {data.size(1)}, but model expects {self.weights.size(1)}."
            )
        return data

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Model is not initialized. Call update() or fit() first.")

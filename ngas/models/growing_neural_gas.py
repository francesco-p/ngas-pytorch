from __future__ import annotations

from dataclasses import dataclass

import torch

from ngas.metrics import normalize_distance_name, pairwise_distance, point_to_set_distance


@dataclass(frozen=True)
class GrowingNeuralGasConfig:
    max_neurons: int = 128
    lr_winner: float = 0.05
    lr_neighbor: float = 0.006
    max_edge_age: int = 64
    lambda_steps: int = 100
    alpha: float = 0.5
    beta: float = 0.0005
    distance: str = "l2"
    input_dim: int | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class GrowingNeuralGas:
    """Classical Growing Neural Gas (GNG) with online topology adaptation."""

    def __init__(
        self,
        max_neurons: int | None = None,
        *,
        n_neurons: int | None = None,
        lr_winner: float = 0.05,
        lr_neighbor: float = 0.006,
        max_edge_age: int = 64,
        lambda_steps: int = 100,
        alpha: float = 0.5,
        beta: float = 0.0005,
        distance: str = "l2",
        input_dim: int | None = None,
        init_points: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_neurons is None and n_neurons is None:
            max_neurons = 128
        if max_neurons is None:
            max_neurons = n_neurons
        if n_neurons is not None and max_neurons != n_neurons:
            raise ValueError("Provide either max_neurons or n_neurons, or use the same value.")
        if max_neurons is None or max_neurons < 2:
            raise ValueError("max_neurons must be >= 2.")
        if lr_winner <= 0 or lr_neighbor < 0:
            raise ValueError("lr_winner must be > 0 and lr_neighbor must be >= 0.")
        if max_edge_age < 1:
            raise ValueError("max_edge_age must be >= 1.")
        if lambda_steps < 1:
            raise ValueError("lambda_steps must be >= 1.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        if input_dim is not None and input_dim < 1:
            raise ValueError("input_dim must be >= 1 when provided.")
        if init_points is not None:
            input_dim = self._validate_init_points(init_points, max_neurons, input_dim, dtype, device)

        self.config = GrowingNeuralGasConfig(
            max_neurons=max_neurons,
            lr_winner=lr_winner,
            lr_neighbor=lr_neighbor,
            max_edge_age=max_edge_age,
            lambda_steps=lambda_steps,
            alpha=alpha,
            beta=beta,
            distance=normalize_distance_name(distance),
            input_dim=input_dim,
            device=torch.device(device),
            dtype=dtype,
        )

        self.weights: torch.Tensor | None = None
        self.errors: torch.Tensor | None = None
        self.adj: torch.Tensor | None = None
        self._steps = 0

        if init_points is not None:
            self._initialize_from_points(init_points)
        elif input_dim is not None:
            self._initialize(input_dim)

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    @property
    def n_nodes(self) -> int:
        if self.weights is None:
            return 0
        return int(self.weights.size(0))

    @property
    def input_dim(self) -> int | None:
        if self.weights is None:
            return None
        return int(self.weights.size(1))

    def _initialize(self, input_dim: int, seed_sample: torch.Tensor | None = None) -> None:
        if seed_sample is None:
            center = torch.rand(input_dim, dtype=self.dtype, device=self.device)
        else:
            center = seed_sample.to(device=self.device, dtype=self.dtype)

        jitter = 0.01 * torch.randn(2, input_dim, dtype=self.dtype, device=self.device)
        self.weights = center.unsqueeze(0).repeat(2, 1) + jitter
        self.errors = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.adj = torch.full((2, 2), -1, dtype=torch.int64, device=self.device)
        self.adj[0, 1] = 0
        self.adj[1, 0] = 0

    def _initialize_from_points(self, init_points: torch.Tensor) -> None:
        points = torch.as_tensor(init_points, dtype=self.dtype, device=self.device)
        n_nodes = int(points.size(0))
        self.weights = points.clone()
        self.errors = torch.zeros(n_nodes, dtype=self.dtype, device=self.device)
        self.adj = torch.full((n_nodes, n_nodes), -1, dtype=torch.int64, device=self.device)
        self.adj.fill_diagonal_(-1)

        if n_nodes == 2:
            self.adj[0, 1] = 0
            self.adj[1, 0] = 0
            return

        # Connect each initial node to its nearest neighbor so all provided
        # points begin active without immediately collapsing as isolated nodes.
        distances = pairwise_distance(points, points, distance="sq_l2")
        distances.fill_diagonal_(float("inf"))
        nearest = torch.argmin(distances, dim=1)
        for i, j in enumerate(nearest.tolist()):
            self.adj[i, j] = 0
            self.adj[j, i] = 0

    @staticmethod
    def _validate_init_points(
        init_points: torch.Tensor,
        max_neurons: int,
        input_dim: int | None,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> int:
        points = torch.as_tensor(init_points, dtype=dtype, device=torch.device(device))
        if points.dim() != 2:
            raise ValueError("init_points must have shape [K, D].")
        if points.size(0) < 2 or points.size(0) > max_neurons:
            raise ValueError(
                f"init_points must contain between 2 and {max_neurons} rows."
            )
        if points.size(1) < 1:
            raise ValueError("init_points must have at least one feature dimension.")
        if input_dim is not None and points.size(1) != input_dim:
            raise ValueError(
                f"init_points has dimension {points.size(1)}, but input_dim={input_dim}."
            )
        return int(points.size(1))

    def _coerce_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample, dtype=self.dtype, device=self.device)
        sample = sample.to(device=self.device, dtype=self.dtype)
        if sample.dim() == 2 and sample.size(0) == 1:
            sample = sample.squeeze(0)
        if sample.dim() != 1:
            raise ValueError("sample must have shape [D] or [1, D].")
        return sample

    def _coerce_batch(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=self.dtype, device=self.device)
        data = data.to(device=self.device, dtype=self.dtype)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.dim() != 2:
            raise ValueError("data must have shape [N, D] or [D].")
        return data

    def _assert_initialized(self) -> None:
        if self.weights is None or self.errors is None or self.adj is None:
            raise RuntimeError("Model is not initialized. Call update() or fit() first.")

    def _neighbors_of(self, index: int) -> torch.Tensor:
        self._assert_initialized()
        assert self.adj is not None
        return torch.where(self.adj[index] >= 0)[0]

    def _remove_old_edges(self) -> None:
        self._assert_initialized()
        assert self.adj is not None
        old = self.adj > self.config.max_edge_age
        self.adj[old] = -1
        self.adj.fill_diagonal_(-1)

    def _remove_isolated_nodes(self) -> None:
        self._assert_initialized()
        assert self.adj is not None
        if self.n_nodes <= 2:
            return

        degree = (self.adj >= 0).sum(dim=1)
        keep = degree > 0
        if bool(keep.all()):
            return

        min_alive = 2
        if int(keep.sum().item()) < min_alive:
            _, top = torch.topk(degree, k=min_alive, largest=True)
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[top] = True

        self._filter_nodes(keep)

    def _filter_nodes(self, keep_mask: torch.Tensor) -> None:
        self._assert_initialized()
        assert self.weights is not None and self.errors is not None and self.adj is not None
        self.weights = self.weights[keep_mask]
        self.errors = self.errors[keep_mask]
        self.adj = self.adj[keep_mask][:, keep_mask]
        self.adj.fill_diagonal_(-1)

    def _insert_new_node(self) -> None:
        self._assert_initialized()
        assert self.weights is not None and self.errors is not None and self.adj is not None

        if self.n_nodes >= self.config.max_neurons:
            return

        q = int(torch.argmax(self.errors).item())
        neighbors = self._neighbors_of(q)
        if neighbors.numel() == 0:
            return

        f = int(neighbors[torch.argmax(self.errors[neighbors])].item())
        new_weight = 0.5 * (self.weights[q] + self.weights[f])

        self.weights = torch.cat([self.weights, new_weight.unsqueeze(0)], dim=0)
        new_error = 0.5 * (self.errors[q] + self.errors[f])
        self.errors = torch.cat([self.errors, new_error.unsqueeze(0)], dim=0)

        n_prev = self.adj.size(0)
        expanded = torch.full(
            (n_prev + 1, n_prev + 1),
            fill_value=-1,
            dtype=self.adj.dtype,
            device=self.adj.device,
        )
        expanded[:n_prev, :n_prev] = self.adj
        self.adj = expanded

        r = n_prev
        self.adj[q, f] = -1
        self.adj[f, q] = -1
        self.adj[q, r] = 0
        self.adj[r, q] = 0
        self.adj[f, r] = 0
        self.adj[r, f] = 0

        self.errors[q] *= self.config.alpha
        self.errors[f] *= self.config.alpha
        self.errors[r] = new_error

    @torch.no_grad()
    def update(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._coerce_sample(sample)
        if self.weights is None:
            self._initialize(sample.numel(), seed_sample=sample)

        self._assert_initialized()
        assert self.weights is not None and self.errors is not None and self.adj is not None

        if sample.numel() != self.weights.size(1):
            raise ValueError(
                f"sample has dimension {sample.numel()}, but model expects {self.weights.size(1)}."
            )

        distances = point_to_set_distance(self.weights, sample, distance=self.config.distance)
        s1 = int(torch.argmin(distances).item())

        masked = distances.clone()
        masked[s1] = float("inf")
        s2 = int(torch.argmin(masked).item())

        neighbors = self._neighbors_of(s1)
        if neighbors.numel() > 0:
            self.adj[s1, neighbors] += 1
            self.adj[neighbors, s1] += 1

        self.adj[s1, s2] = 0
        self.adj[s2, s1] = 0

        error_term = distances[s1]
        if self.config.distance == "l2":
            error_term = error_term.pow(2)
        self.errors[s1] += error_term

        self.weights[s1] += self.config.lr_winner * (sample - self.weights[s1])
        neighbors = self._neighbors_of(s1)
        if neighbors.numel() > 0:
            self.weights[neighbors] += self.config.lr_neighbor * (sample - self.weights[neighbors])

        self._remove_old_edges()
        self._remove_isolated_nodes()

        self._steps += 1
        if self._steps % self.config.lambda_steps == 0:
            self._insert_new_node()

        self.errors *= 1.0 - self.config.beta
        return self.adj, self.weights

    @torch.no_grad()
    def fit(self, data: torch.Tensor, epochs: int = 1, shuffle: bool = True) -> "GrowingNeuralGas":
        data = self._coerce_batch(data)
        for _ in range(epochs):
            if shuffle:
                order = torch.randperm(data.size(0), device=data.device)
                data_epoch = data[order]
            else:
                data_epoch = data
            for sample in data_epoch:
                self.update(sample)
        return self

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        self._assert_initialized()
        assert self.weights is not None
        data = self._coerce_batch(data)
        distances = pairwise_distance(data, self.weights, distance=self.config.distance)
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def quantization_error(self, data: torch.Tensor) -> float:
        self._assert_initialized()
        assert self.weights is not None
        data = self._coerce_batch(data)
        distances = pairwise_distance(data, self.weights, distance=self.config.distance)
        min_dist = torch.min(distances, dim=1).values
        return float(min_dist.mean().item())

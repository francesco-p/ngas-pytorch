from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ngas.metrics import normalize_distance_name, pairwise_distance


@dataclass(frozen=True)
class DifferentiableNeuralGasConfig:
    n_neurons: int = 32
    input_dim: int = 2
    distance: str = "l2"
    neighborhood: str = "exponential"
    lambda_value: float = 8.0
    rank_temperature: float = 0.2
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class DifferentiableNeuralGas(nn.Module):
    """Differentiable Neural Gas objective optimized via backpropagation.

    The expected rank of each neuron is approximated with pairwise sigmoid
    comparisons, then mapped to a neighborhood weighting.
    """

    def __init__(
        self,
        n_neurons: int = 32,
        input_dim: int | None = None,
        init_points: torch.Tensor | None = None,
        distance: str = "l2",
        neighborhood: str = "exponential",
        lambda_value: float = 8.0,
        rank_temperature: float = 0.2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_neurons < 2:
            raise ValueError("n_neurons must be >= 2.")
        if lambda_value <= 0:
            raise ValueError("lambda_value must be > 0.")
        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be > 0.")
        if init_points is not None:
            input_dim = self._validate_init_points(init_points, n_neurons, input_dim, dtype, device)
        else:
            if input_dim is None:
                input_dim = 2
            if input_dim < 1:
                raise ValueError("input_dim must be >= 1.")

        neighborhood_key = neighborhood.strip().lower()
        if neighborhood_key not in {"exponential", "inverse"}:
            raise ValueError("neighborhood must be one of: exponential, inverse.")

        self.config = DifferentiableNeuralGasConfig(
            n_neurons=n_neurons,
            input_dim=input_dim,
            distance=normalize_distance_name(distance),
            neighborhood=neighborhood_key,
            lambda_value=lambda_value,
            rank_temperature=rank_temperature,
            device=torch.device(device),
            dtype=dtype,
        )

        if init_points is None:
            w = torch.rand(
                n_neurons,
                input_dim,
                device=self.config.device,
                dtype=self.config.dtype,
            )
        else:
            w = torch.as_tensor(
                init_points,
                device=self.config.device,
                dtype=self.config.dtype,
            ).clone()
        self.weights = nn.Parameter(w)

    @staticmethod
    def _validate_init_points(
        init_points: torch.Tensor,
        n_neurons: int,
        input_dim: int | None,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> int:
        points = torch.as_tensor(init_points, dtype=dtype, device=torch.device(device))
        if points.dim() != 2:
            raise ValueError("init_points must have shape [n_neurons, D].")
        if points.size(0) != n_neurons:
            raise ValueError(
                f"init_points has {points.size(0)} rows, but model expects {n_neurons}."
            )
        if points.size(1) < 1:
            raise ValueError("init_points must have at least one feature dimension.")
        if input_dim is not None and points.size(1) != input_dim:
            raise ValueError(
                f"init_points has dimension {points.size(1)}, but input_dim={input_dim}."
            )
        return int(points.size(1))

    @property
    def n_neurons(self) -> int:
        return self.config.n_neurons

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    def _coerce_batch(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=self.dtype, device=self.device)
        data = data.to(device=self.device, dtype=self.dtype)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.dim() != 2:
            raise ValueError("data must have shape [N, D] or [D].")
        if data.size(1) != self.input_dim:
            raise ValueError(f"data has dimension {data.size(1)}, but model expects {self.input_dim}.")
        return data

    def _soft_rank(self, distances: torch.Tensor) -> torch.Tensor:
        tau = self.config.rank_temperature
        di = distances.unsqueeze(-1)
        dj = distances.unsqueeze(-2)
        prob_greater = torch.sigmoid((di - dj) / tau)
        eye = torch.eye(distances.size(1), device=distances.device, dtype=distances.dtype)
        prob_greater = prob_greater * (1.0 - eye.unsqueeze(0))
        return 1.0 + prob_greater.sum(dim=-1)

    def _neighborhood_weight(self, soft_rank: torch.Tensor) -> torch.Tensor:
        if self.config.neighborhood == "exponential":
            return torch.exp(-(soft_rank - 1.0) / self.config.lambda_value)
        return 1.0 / soft_rank.pow(2)

    def forward(self, data: torch.Tensor, return_details: bool = False):
        x = self._coerce_batch(data)
        distances = pairwise_distance(x, self.weights, distance=self.config.distance)
        soft_rank = self._soft_rank(distances)
        neighborhood = self._neighborhood_weight(soft_rank)

        loss = (neighborhood * distances).mean()

        if return_details:
            return {
                "loss": loss,
                "distances": distances,
                "soft_rank": soft_rank,
                "neighborhood": neighborhood,
            }
        return loss

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        x = self._coerce_batch(data)
        distances = pairwise_distance(x, self.weights, distance=self.config.distance)
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def quantization_error(self, data: torch.Tensor) -> float:
        x = self._coerce_batch(data)
        distances = pairwise_distance(x, self.weights, distance=self.config.distance)
        return float(torch.min(distances, dim=1).values.mean().item())

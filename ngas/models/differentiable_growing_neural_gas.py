from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ngas.metrics import normalize_distance_name, pairwise_distance


@dataclass(frozen=True)
class DifferentiableGrowingNeuralGasConfig:
    max_neurons: int = 64
    input_dim: int = 2
    init_neurons: int = 2
    distance: str = "l2"
    neighborhood: str = "exponential"
    lambda_value: float = 8.0
    rank_temperature: float = 0.2
    topology_influence: float = 0.5
    edge_init_logit: float = -2.0
    edge_sparsity_coeff: float = 1e-3
    edge_length_coeff: float = 1e-2
    error_ema_beta: float = 0.1
    grow_error_decay: float = 0.5
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class DifferentiableGrowingNeuralGas(nn.Module):
    """Differentiable GNG-style model with learnable topology and explicit growth.

    Forward is fully differentiable (prototypes + edge logits). Node growth is an
    explicit discrete operation via `grow()`.
    """

    def __init__(
        self,
        max_neurons: int = 64,
        input_dim: int | None = None,
        init_neurons: int = 2,
        init_points: torch.Tensor | None = None,
        distance: str = "l2",
        neighborhood: str = "exponential",
        lambda_value: float = 8.0,
        rank_temperature: float = 0.2,
        topology_influence: float = 0.5,
        edge_init_logit: float = -2.0,
        edge_sparsity_coeff: float = 1e-3,
        edge_length_coeff: float = 1e-2,
        error_ema_beta: float = 0.1,
        grow_error_decay: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if max_neurons < 2:
            raise ValueError("max_neurons must be >= 2.")
        if lambda_value <= 0:
            raise ValueError("lambda_value must be > 0.")
        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be > 0.")
        if error_ema_beta <= 0 or error_ema_beta > 1:
            raise ValueError("error_ema_beta must be in (0, 1].")
        if grow_error_decay <= 0 or grow_error_decay > 1:
            raise ValueError("grow_error_decay must be in (0, 1].")
        if init_points is not None:
            input_dim, init_neurons = self._validate_init_points(
                init_points,
                max_neurons,
                input_dim,
                dtype,
                device,
            )
        else:
            if input_dim is None:
                input_dim = 2
            if input_dim < 1:
                raise ValueError("input_dim must be >= 1.")
            if init_neurons < 2 or init_neurons > max_neurons:
                raise ValueError("init_neurons must be in [2, max_neurons].")

        neighborhood_key = neighborhood.strip().lower()
        if neighborhood_key not in {"exponential", "inverse"}:
            raise ValueError("neighborhood must be one of: exponential, inverse.")

        self.config = DifferentiableGrowingNeuralGasConfig(
            max_neurons=max_neurons,
            input_dim=input_dim,
            init_neurons=init_neurons,
            distance=normalize_distance_name(distance),
            neighborhood=neighborhood_key,
            lambda_value=lambda_value,
            rank_temperature=rank_temperature,
            topology_influence=topology_influence,
            edge_init_logit=edge_init_logit,
            edge_sparsity_coeff=edge_sparsity_coeff,
            edge_length_coeff=edge_length_coeff,
            error_ema_beta=error_ema_beta,
            grow_error_decay=grow_error_decay,
            device=torch.device(device),
            dtype=dtype,
        )

        w = torch.rand(
            max_neurons,
            input_dim,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        if init_points is not None:
            points = torch.as_tensor(
                init_points,
                device=self.config.device,
                dtype=self.config.dtype,
            )
            w[: points.size(0)] = points
        self.weights = nn.Parameter(w)

        e = torch.full(
            (max_neurons, max_neurons),
            fill_value=edge_init_logit,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.edge_logits = nn.Parameter(e)

        active_mask = torch.zeros(max_neurons, dtype=torch.bool, device=self.config.device)
        active_mask[:init_neurons] = True
        self.register_buffer("active_mask", active_mask)
        self.register_buffer(
            "node_error_ema",
            torch.zeros(max_neurons, dtype=self.config.dtype, device=self.config.device),
        )

    @staticmethod
    def _validate_init_points(
        init_points: torch.Tensor,
        max_neurons: int,
        input_dim: int | None,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> tuple[int, int]:
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
        return int(points.size(1)), int(points.size(0))

    @property
    def n_nodes(self) -> int:
        return int(self.active_mask.sum().item())

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
        if data.size(1) != self.config.input_dim:
            raise ValueError(
                f"data has dimension {data.size(1)}, but model expects {self.config.input_dim}."
            )
        return data

    def _active_indices(self) -> torch.Tensor:
        return torch.where(self.active_mask)[0]

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

    def _edge_prob_matrix(self) -> torch.Tensor:
        logits = 0.5 * (self.edge_logits + self.edge_logits.transpose(0, 1))
        probs = torch.sigmoid(logits)
        probs = probs - torch.diag_embed(torch.diagonal(probs))
        return probs

    def forward(self, data: torch.Tensor, return_details: bool = False):
        x = self._coerce_batch(data)
        idx = self._active_indices()
        w = self.weights[idx]

        distances = pairwise_distance(x, w, distance=self.config.distance)
        soft_rank = self._soft_rank(distances)
        neighborhood = self._neighborhood_weight(soft_rank)

        edge_prob_full = self._edge_prob_matrix()
        edge_prob = edge_prob_full[idx][:, idx]

        if edge_prob.size(0) > 1:
            degree = edge_prob.sum(dim=1) / float(edge_prob.size(0) - 1)
        else:
            degree = torch.zeros(edge_prob.size(0), device=edge_prob.device, dtype=edge_prob.dtype)

        neighborhood = neighborhood * (1.0 + self.config.topology_influence * degree.unsqueeze(0))
        data_term = (neighborhood * distances).mean()

        if edge_prob.numel() > 0:
            proto_dist = pairwise_distance(w, w, distance="sq_l2")
            weighted_len = (edge_prob * proto_dist).sum() / (edge_prob.sum() + 1e-8)
            sparsity = edge_prob.mean()
        else:
            weighted_len = torch.zeros((), device=self.device, dtype=self.dtype)
            sparsity = torch.zeros((), device=self.device, dtype=self.dtype)

        loss = (
            data_term
            + self.config.edge_length_coeff * weighted_len
            + self.config.edge_sparsity_coeff * sparsity
        )

        with torch.no_grad():
            node_err = (neighborhood.detach() * distances.detach()).mean(dim=0)
            prev = self.node_error_ema[idx]
            beta = self.config.error_ema_beta
            self.node_error_ema[idx] = (1.0 - beta) * prev + beta * node_err

        if return_details:
            return {
                "loss": loss,
                "data_term": data_term,
                "edge_length_term": weighted_len,
                "edge_sparsity_term": sparsity,
                "distances": distances,
                "soft_rank": soft_rank,
                "neighborhood": neighborhood,
                "edge_prob": edge_prob,
            }
        return loss

    @torch.no_grad()
    def grow(self, n_new: int = 1, noise_std: float = 0.01) -> int:
        if n_new < 1:
            raise ValueError("n_new must be >= 1.")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0.")

        added = 0
        for _ in range(n_new):
            if self.n_nodes >= self.config.max_neurons:
                break

            idx = self._active_indices()
            local_errors = self.node_error_ema[idx]
            q_local = int(torch.argmax(local_errors).item())
            q = int(idx[q_local].item())

            edge_prob = self._edge_prob_matrix()[idx][:, idx]
            if edge_prob.size(0) > 1:
                row = edge_prob[q_local].clone()
                row[q_local] = -1.0
                f_local = int(torch.argmax(row).item())
                if float(row[f_local].item()) <= 0.0:
                    d = pairwise_distance(
                        self.weights[q].unsqueeze(0),
                        self.weights[idx],
                        distance="sq_l2",
                    ).squeeze(0)
                    d[q_local] = float("inf")
                    f_local = int(torch.argmin(d).item())
                f = int(idx[f_local].item())
            else:
                break

            free_idx = torch.where(~self.active_mask)[0]
            if free_idx.numel() == 0:
                break
            r = int(free_idx[0].item())

            midpoint = 0.5 * (self.weights[q] + self.weights[f])
            if noise_std > 0:
                midpoint = midpoint + noise_std * torch.randn_like(midpoint)
            self.weights[r].copy_(midpoint)
            self.active_mask[r] = True

            self.edge_logits[q, f] = -6.0
            self.edge_logits[f, q] = -6.0

            init = self.config.edge_init_logit
            self.edge_logits[q, r] = init
            self.edge_logits[r, q] = init
            self.edge_logits[f, r] = init
            self.edge_logits[r, f] = init

            new_err = 0.5 * (self.node_error_ema[q] + self.node_error_ema[f])
            self.node_error_ema[q] *= self.config.grow_error_decay
            self.node_error_ema[f] *= self.config.grow_error_decay
            self.node_error_ema[r] = new_err
            added += 1

        return added

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        x = self._coerce_batch(data)
        idx = self._active_indices()
        w = self.weights[idx]
        distances = pairwise_distance(x, w, distance=self.config.distance)
        winners = torch.argmin(distances, dim=1)
        return idx[winners]

    @torch.no_grad()
    def quantization_error(self, data: torch.Tensor) -> float:
        x = self._coerce_batch(data)
        idx = self._active_indices()
        w = self.weights[idx]
        distances = pairwise_distance(x, w, distance=self.config.distance)
        return float(torch.min(distances, dim=1).values.mean().item())

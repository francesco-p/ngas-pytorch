from __future__ import annotations

import torch
import torch.nn.functional as F

SUPPORTED_DISTANCES = ("l2", "sq_l2", "cosine")
_DISTANCE_ALIASES = {
    "l2": "l2",
    "euclidean": "l2",
    "sq_l2": "sq_l2",
    "sqeuclidean": "sq_l2",
    "cosine": "cosine",
    "cos": "cosine",
}


def normalize_distance_name(distance: str) -> str:
    key = distance.strip().lower()
    if key not in _DISTANCE_ALIASES:
        opts = ", ".join(SUPPORTED_DISTANCES)
        raise ValueError(f"Unsupported distance '{distance}'. Expected one of: {opts}.")
    return _DISTANCE_ALIASES[key]


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, distance: str = "l2") -> torch.Tensor:
    metric = normalize_distance_name(distance)

    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("pairwise_distance expects 2D tensors: [N, D] and [M, D].")
    if x.size(-1) != y.size(-1):
        raise ValueError("pairwise_distance expects matching feature dimensions.")

    if metric == "l2":
        return torch.cdist(x, y, p=2)

    if metric == "sq_l2":
        return torch.cdist(x, y, p=2).pow(2)

    x_norm = F.normalize(x, p=2, dim=-1, eps=1e-12)
    y_norm = F.normalize(y, p=2, dim=-1, eps=1e-12)
    similarity = x_norm @ y_norm.transpose(-1, -2)
    return 1.0 - similarity.clamp(-1.0, 1.0)


def point_to_set_distance(points: torch.Tensor, sample: torch.Tensor, distance: str = "l2") -> torch.Tensor:
    if points.dim() != 2:
        raise ValueError("points must have shape [N, D].")

    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    elif sample.dim() != 2 or sample.size(0) != 1:
        raise ValueError("sample must have shape [D] or [1, D].")

    return pairwise_distance(sample, points, distance=distance).squeeze(0)

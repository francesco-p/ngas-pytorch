from __future__ import annotations

import argparse

import torch

from ngas.models import NeuralGas


def make_blobs(n_samples: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    centers = torch.tensor(
        [[-2.0, -1.5], [0.5, 2.0], [2.5, -0.5]],
        dtype=torch.float32,
    )
    chunks = []
    per_center = n_samples // centers.size(0)
    for c in centers:
        points = c + 0.45 * torch.randn(per_center, 2, generator=g)
        chunks.append(points)

    rem = n_samples - per_center * centers.size(0)
    if rem > 0:
        points = centers[0] + 0.45 * torch.randn(rem, 2, generator=g)
        chunks.append(points)

    data = torch.cat(chunks, dim=0)
    perm = torch.randperm(data.size(0), generator=g)
    data = data[perm]
    return data.to(device=device)


def edge_count(adj: torch.Tensor) -> int:
    return int((adj >= 0).sum().item() // 2)


def maybe_plot(data: torch.Tensor, model: NeuralGas) -> None:
    if data.size(1) != 2:
        raise ValueError("Plotting only supports 2D data.")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot") from exc

    pred = model.predict(data).cpu()
    x = data.cpu()
    w = model.weights.cpu()
    adj = model.adj.cpu()

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=pred, s=18, alpha=0.55, cmap="tab10")
    plt.scatter(w[:, 0], w[:, 1], c="black", s=80, marker="x", linewidths=2)

    idx = torch.triu_indices(adj.size(0), adj.size(1), offset=1)
    for i, j in zip(idx[0], idx[1]):
        if int(adj[i, j].item()) >= 0:
            p = w[i]
            q = w[j]
            plt.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=0.8, alpha=0.4)

    plt.title("Neural Gas on Synthetic Blobs")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuralGas on synthetic 2D blobs.")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--n-neurons", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--max-edge-age", type=int, default=80)
    parser.add_argument("--distance", type=str, default="l2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data = make_blobs(n_samples=args.n_samples, seed=args.seed, device=device)

    model = NeuralGas(
        n_neurons=args.n_neurons,
        lr=args.lr,
        max_edge_age=args.max_edge_age,
        distance=args.distance,
        device=device,
    )

    model.fit(data, epochs=args.epochs, shuffle=True)

    qerr = model.quantization_error(data)
    print(f"neurons={model.n_neurons}")
    print(f"edges={edge_count(model.adj)}")
    print(f"quantization_error={qerr:.6f}")

    if args.plot:
        maybe_plot(data, model)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch

from ngas.models import GrowingNeuralGas


def make_blobs(n_samples: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    centers = torch.tensor(
        [[-2.0, -1.5], [0.5, 2.0], [2.5, -0.5], [2.5, 2.8]],
        dtype=torch.float32,
    )
    chunks = []
    per_center = n_samples // centers.size(0)
    for c in centers:
        points = c + 0.40 * torch.randn(per_center, 2, generator=g)
        chunks.append(points)

    rem = n_samples - per_center * centers.size(0)
    if rem > 0:
        points = centers[0] + 0.40 * torch.randn(rem, 2, generator=g)
        chunks.append(points)

    data = torch.cat(chunks, dim=0)
    perm = torch.randperm(data.size(0), generator=g)
    data = data[perm]
    return data.to(device=device)


def edge_count(adj: torch.Tensor) -> int:
    return int((adj >= 0).sum().item() // 2)


def maybe_plot(data: torch.Tensor, model: GrowingNeuralGas) -> None:
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
    plt.scatter(x[:, 0], x[:, 1], c=pred, s=18, alpha=0.55, cmap="tab20")
    plt.scatter(w[:, 0], w[:, 1], c="black", s=85, marker="x", linewidths=2)

    idx = torch.triu_indices(adj.size(0), adj.size(1), offset=1)
    for i, j in zip(idx[0], idx[1]):
        if int(adj[i, j].item()) >= 0:
            p = w[i]
            q = w[j]
            plt.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=0.8, alpha=0.4)

    plt.title("Growing Neural Gas on Synthetic Blobs")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GrowingNeuralGas on synthetic 2D blobs.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--max-neurons", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr-winner", type=float, default=0.05)
    parser.add_argument("--lr-neighbor", type=float, default=0.006)
    parser.add_argument("--max-edge-age", type=int, default=80)
    parser.add_argument("--lambda-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.0005)
    parser.add_argument("--distance", type=str, default="l2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data = make_blobs(n_samples=args.n_samples, seed=args.seed, device=device)

    model = GrowingNeuralGas(
        max_neurons=args.max_neurons,
        lr_winner=args.lr_winner,
        lr_neighbor=args.lr_neighbor,
        max_edge_age=args.max_edge_age,
        lambda_steps=args.lambda_steps,
        alpha=args.alpha,
        beta=args.beta,
        distance=args.distance,
        device=device,
    )

    model.fit(data, epochs=args.epochs, shuffle=True)

    qerr = model.quantization_error(data)
    print(f"nodes={model.n_nodes}")
    print(f"edges={edge_count(model.adj)}")
    print(f"quantization_error={qerr:.6f}")

    if args.plot:
        maybe_plot(data, model)


if __name__ == "__main__":
    main()

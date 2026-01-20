import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from condgen_benchmark.methods.gcds.core import GCDSGenerator


def load_y_test(mode, seed):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", "test.npz"
    )
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y_true = data["Y"]
    return X, y_true


def load_x_normalization_params(mode, seed):
    path = os.path.join(
        "datasets",
        "synthetic",
        mode,
        f"seed_{seed}",
        "x_normalization_params.npz",
    )
    data = np.load(path)
    mean = torch.tensor(data["mean"], dtype=torch.float32)
    std = torch.tensor(data["std"], dtype=torch.float32)
    return mean, std


@torch.no_grad()
def sample_multiple_batches(
    generator, y, num_samples, batch_size, noise_dim, device
):
    loader = DataLoader(
        TensorDataset(y),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    x_hats = []

    for (yb,) in tqdm(loader, desc="Sampling", leave=False):
        yb = yb.to(device)  # [B, y_dim]
        B = yb.shape[0]

        yb_repeat = yb.repeat_interleave(num_samples, dim=0)  # [B * S, y_dim]
        noise = torch.randn(
            B * num_samples, noise_dim, device=device
        )  # [B * S, noise_dim]

        xb = generator(yb_repeat, noise).view(
            B, num_samples, -1
        )  # [B, S, x_dim]
        x_hats.append(xb)

    return torch.cat(x_hats, dim=0)  # [N, S, x_dim]


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cpu"
    )  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y_true = load_y_test(
        args.data_name, args.seed
    )  # [N, y_dim], [N, x_dim]
    x_dim = X.shape[1]
    y_dim = y_true.shape[1]

    generator = GCDSGenerator(
        x_dim=x_dim,
        noise_dim=args.noise_dim,
        y_dim=y_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    ckpt_path = os.path.join(
        "workspaces",
        "gcds",
        args.data_name,
        f"seed_{args.seed}",
        "generator.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    generator.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    print(f"Loaded generator from {ckpt_path} successfully.")
    generator.eval()

    x_hat = sample_multiple_batches(
        generator=generator,
        y=X,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        device=device,
    )  # [N, S, x_dim]

    # Denormalize X_hat
    x_mean, x_std = load_x_normalization_params(args.data_name, args.seed)
    print(f"Loaded X normalization mean: {x_mean.view(-1).numpy()}")
    print(f"Loaded X normalization std:  {x_std.view(-1).numpy()}")

    x_mean = x_mean.view(1, 1, -1).to(x_hat.device)  # [1, 1, x_dim]
    x_std = x_std.view(1, 1, -1).to(x_hat.device)
    x_hat = x_hat * x_std + x_mean

    out_dir = os.path.join(
        "workspaces", "gcds", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sampled.npz")

    np.savez(
        out_path,
        X_hat=x_hat.cpu().numpy(),  # [N, S, x_dim]
        X=X.cpu().numpy(),  # [N, x_dim]
        y_true=y_true,  # [N, y_dim]
    )

    print(f"Saved samples to {out_path}")
    print(f"Shape: {x_hat.shape} (N, S, x_dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="M1")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument(
        "--model-choice",
        type=str,
        choices=["last", "best"],
        default="last",
    )
    args = parser.parse_args()
    main(args)

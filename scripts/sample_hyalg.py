import argparse
import os

import numpy as np
import torch

from condgen_benchmark.methods.hyalg.sampler import sample_conditional
from condgen_benchmark.methods.hyalg.utils import (
    enforce_positive_first_nonzero,
)


def load_dataset(mode, seed):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", "test.npz"
    )
    data = np.load(path)
    Y = data["Y"]
    X = torch.tensor(data["X"], dtype=torch.float32)
    return X, Y


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cpu"
    )  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y_true = load_dataset(args.data_name, args.seed)

    workspace_dir = os.path.join(
        "workspaces", "hyalg", args.data_name, f"seed_{args.seed}"
    )
    ckpt_path = os.path.join(workspace_dir, "theta.npz")
    theta_np = np.load(ckpt_path)["theta"]
    theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
    theta = enforce_positive_first_nonzero(theta)
    theta = theta / torch.norm(theta)

    train_data = np.load(
        os.path.join(
            "datasets",
            "synthetic",
            args.data_name,
            f"seed_{args.seed}",
            "train.npz",
        )
    )
    X_train = torch.tensor(train_data["X"], dtype=torch.float32, device=device)
    Y_train = torch.tensor(train_data["Y"], dtype=torch.float32, device=device)
    if Y_train.shape[-1] == 1:
        Y_train = Y_train.squeeze(-1)
    # x_test = torch.tensor(x_true, dtype=torch.float32, device=device)
    # x_test = torch.tensor(y, dtype=torch.float32, device=device)
    x_test = x.to(device)
    yGrid = torch.linspace(
        X_train.min(), X_train.max(), args.ygrid, device=device
    )

    y_samples = sample_conditional(
        X_train, Y_train, x_test, theta, yGrid, args.H, args.num_samples
    )  # [N, S]

    y_hat = y_samples.unsqueeze(-1)  # [N, S, 1]

    out_path = os.path.join(workspace_dir, "sampled.npz")
    np.savez(
        out_path,
        Y_hat=y_hat.cpu().numpy(),
        X=x.cpu().numpy(),
        Y_true=y_true,
    )

    print(f"Saved samples to {out_path}")
    print(f"Shape: {y_hat.shape} (N, S, y_dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="M1")
    parser.add_argument("--H", type=float, default=0.8)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--ygrid", type=int, default=200)
    args = parser.parse_args()
    main(args)

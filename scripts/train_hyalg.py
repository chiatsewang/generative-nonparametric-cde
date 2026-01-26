import argparse
import os
import time

import numpy as np
import torch
import yaml

from condgen_benchmark.methods.hyalg.loss import S_theta
from condgen_benchmark.methods.hyalg.utils import (
    enforce_positive_first_nonzero,
)


def load_dataset(mode, split, seed):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", f"{split}.npz"
    )
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    if Y.shape[-1] == 1:
        Y = Y.squeeze(-1)
    return X, Y


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cpu"
    )  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y = load_dataset(args.data_name, "train", args.seed)
    X, Y = X.to(device), Y.to(device)

    d = X.shape[1]
    grid_pts = torch.linspace(-1.5, 1.5, 3, device=device)
    mesh = torch.meshgrid(*[grid_pts] * d, indexing="ij")
    A_centers = torch.stack([g.reshape(-1) for g in mesh], dim=1)

    from scipy.optimize import minimize

    theta0 = np.random.randn(d)
    theta0 = enforce_positive_first_nonzero(theta0)
    theta0 /= np.linalg.norm(theta0)

    start_all = time.time()
    result = minimize(
        S_theta(X, Y, A_centers, args.radius, args.h),
        theta0,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda t: np.linalg.norm(t) - 1}],
        options={"disp": True, "maxiter": 500, "ftol": 1e-10},
    )
    t_opt = time.time() - start_all

    theta_hat = torch.tensor(result.x, dtype=torch.float32, device=device)
    theta_hat = enforce_positive_first_nonzero(theta_hat)
    theta_hat = theta_hat / torch.norm(theta_hat)

    print("Optimization completed.")
    print("Estimated theta:", theta_hat.cpu().numpy())
    print("Optimization time:", t_opt)

    workspace_dir = os.path.join(
        "workspaces", "hyalg", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(workspace_dir, exist_ok=True)

    np.savez(
        os.path.join(workspace_dir, "theta.npz"),
        theta=theta_hat.cpu().numpy(),
        t_opt=t_opt,
    )

    config_path = os.path.join(workspace_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230, help="Dataset seed")
    parser.add_argument("--data-name", type=str, default="M1")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.7)
    args = parser.parse_args()
    main(args)

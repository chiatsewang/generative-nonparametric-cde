# scripts/sample_srm.py
import argparse
import os

import numpy as np
import torch
import yaml

from condgen_benchmark.methods.srm.model import SRMGenerator, parse_hidden_dims
from condgen_benchmark.methods.srm.sampler import denormalize_y, sample_srm


def load_test_dataset(data_name, seed):
    path = os.path.join(
        "datasets", "synthetic", data_name, f"seed_{seed}", "test.npz"
    )
    data = np.load(path)

    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = data["Y"] if "Y" in data.files else None

    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y is not None and Y.ndim == 1:
        Y = Y[:, None]

    return X, Y


def load_normalization_params(workspace_dir):
    path = os.path.join(workspace_dir, "Y_normalization_params.npz")
    params = np.load(path)
    mean = params["mean"]
    std = params["std"]
    return mean, std


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device      : {device}")

    workspace_dir = os.path.join(
        "workspaces", "srm", args.data_name, f"seed_{args.seed}"
    )

    config_path = os.path.join(workspace_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    X_test, Y_true = load_test_dataset(args.data_name, args.seed)
    y_mean, y_std = load_normalization_params(workspace_dir)
    y_dim = int(np.asarray(y_mean).reshape(1, -1).shape[1])
    x_dim = X_test.shape[1]

    # Architecture parameters are read from config.yaml saved at training time.
    # This avoids having to remember hidden_dims/noise_dim manually when loading
    # generator.pth, which only contains the model weights.
    if "hidden_dims" in config:
        hidden_dims = parse_hidden_dims(config["hidden_dims"])
    else:
        # Backward compatibility for older SRM runs that used hidden_dim and
        # num_layers instead of a hidden_dims list.
        hidden_dims = [int(config.get("hidden_dim", 50))] * int(
            config.get("num_layers", 2)
        )

    generator = SRMGenerator(
        x_dim=x_dim,
        noise_dim=config["noise_dim"],
        y_dim=y_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    model_path = os.path.join(workspace_dir, "generator.pth")
    generator.load_state_dict(torch.load(model_path, map_location=device))

    y_samples = sample_srm(
        generator=generator,
        x=X_test,
        n_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )
    y_samples = denormalize_y(y_samples, y_mean, y_std).numpy()

    # Match all existing sample_xxx.py scripts in this benchmark.
    # The evaluation script expects workspaces/{algorithm}/{data_name}/seed_{seed}/sampled.npz
    # with keys: Y_hat, X, and Y_true. Do not change these names unless eval.py is changed too.
    out_path = os.path.join(workspace_dir, "sampled.npz")
    np.savez(
        out_path,
        Y_hat=y_samples,  # [N, S, Y_dim]
        X=X_test.numpy(),  # [N, X_dim]
        Y_true=Y_true,  # [N, Y_dim]
    )

    print(f"Saved samples to {out_path}")
    print(f"Shape: {y_samples.shape} (N, S, Y_dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="U1")

    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    print("[SRM] Sampling Settings:")
    print(f" Data name   : {args.data_name}")
    print(f" Seed        : {args.seed}")
    print(f" Num samples : {args.num_samples}")
    print(f" Batch size  : {args.batch_size}")

    main(args)

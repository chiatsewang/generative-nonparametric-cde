import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from condgen_benchmark.methods.ddpm.diffusion import Diffusion
from condgen_benchmark.methods.ddpm.model import (
    ConditionalMLPScalarT,
    parse_hidden_dims,
)
from condgen_benchmark.methods.ddpm.sampler import sample_ddpm


def load_test_data(mode, seed):
    """
    Load test dataset.

    Returns:
        X: torch.Tensor of shape [N, X_dim]
        Y_true: np.ndarray of shape [N, Y_dim]
    """
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", "test.npz"
    )
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y_true = data["Y"]
    return X, Y_true


def load_Y_normalization_params(mode, seed):
    path = os.path.join(
        "workspaces",
        "ddpm",
        mode,
        f"seed_{seed}",
        "Y_normalization_params.npz",
    )
    data = np.load(path)
    mean = torch.tensor(data["mean"], dtype=torch.float32)
    std = torch.tensor(data["std"], dtype=torch.float32)
    return mean, std


@torch.no_grad()
def sample_multiple_batches(
    model, diffusion, X, num_samples, batch_size, Y_dim, device
):
    X = X.to(device)
    Y_hats = []

    for start in tqdm(
        range(0, X.shape[0], batch_size), desc="Sampling", leave=False
    ):
        xb = X[start : start + batch_size]
        B = xb.shape[0]

        xb_repeat = xb.repeat_interleave(num_samples, dim=0)  # [B * S, X_dim]

        Y_hat = sample_ddpm(
            model,
            diffusion,
            xb_repeat,
            shape=(B * num_samples, Y_dim),
            device=device,
        )  # [B * S, Y_dim]

        Y_hat = Y_hat.view(B, num_samples, Y_dim)  # [B, S, Y_dim]
        Y_hats.append(Y_hat)

    return torch.cat(Y_hats, dim=0)  # [N, S, Y_dim]


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"[Device] Using {device}")

    X, Y_true = load_test_data(
        args.data_name, args.seed
    )  # [N, X_dim], [N, Y_dim]
    X_dim = X.shape[1]
    Y_dim = Y_true.shape[1]

    workspace_dir = os.path.join(
        "workspaces", "ddpm", args.data_name, f"seed_{args.seed}"
    )
    config_path = os.path.join(workspace_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "hidden_dims" in config:
        hidden_dims = parse_hidden_dims(config["hidden_dims"])
    elif all(
        key in config for key in ("hidden_dim1", "hidden_dim2", "hidden_dim3")
    ):
        hidden_dims = [
            int(config["hidden_dim1"]),
            int(config["hidden_dim2"]),
            int(config["hidden_dim3"]),
        ]
    elif "hidden_dim" in config:
        hidden_dim = int(config["hidden_dim"])
        hidden_dims = [hidden_dim, hidden_dim // 2]
    else:
        hidden_dims = parse_hidden_dims(None)

    timesteps = int(config.get("timesteps", 300))
    print(f"[DDPM] hidden_dims={hidden_dims}")
    print(f"[DDPM] timesteps={timesteps}")

    model = ConditionalMLPScalarT(
        X_dim,
        Y_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    ckpt_path = os.path.join(
        workspace_dir,
        f"model_{args.model_choice}.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    print(f"Loaded model from {ckpt_path} successfully.")
    model.eval()

    diffusion = Diffusion(timesteps=timesteps, device=device)

    Y_hat = sample_multiple_batches(
        model=model,
        diffusion=diffusion,
        X=X,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        Y_dim=Y_dim,
        device=device,
    )  # [N, S, Y_dim]

    # Denormalize Y_hat
    Y_mean, Y_std = load_Y_normalization_params(args.data_name, args.seed)
    print(f"Loaded Y normalization mean: {Y_mean.view(-1).numpy()}")
    print(f"Loaded Y normalization std:  {Y_std.view(-1).numpy()}")

    Y_mean = Y_mean.view(1, 1, -1).to(Y_hat.device)  # [1, 1, Y_dim]
    Y_std = Y_std.view(1, 1, -1).to(Y_hat.device)  # [1, 1, Y_dim]
    Y_hat = Y_hat * Y_std + Y_mean

    out_dir = os.path.join(
        "workspaces", "ddpm", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sampled.npz")

    np.savez(
        out_path,
        Y_hat=Y_hat.cpu().numpy(),  # [N, S, Y_dim]
        X=X.cpu().numpy(),  # [N, X_dim]
        Y_true=Y_true,
    )  # [N, Y_dim]

    print(f"Saved samples to {out_path}")
    print(f"Shape: {Y_hat.shape} (N, S, Y_dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230, help="Dataset seed")
    parser.add_argument("--data-name", type=str, default="U1")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--model-choice",
        type=str,
        choices=["last", "best"],
        default="best",
        help="Which model checkpoint to use: 'last' or 'best'",
    )
    args = parser.parse_args()
    main(args)

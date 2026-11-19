# scripts/train_srm.py
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from condgen_benchmark.methods.srm.model import SRMGenerator, parse_hidden_dims
from condgen_benchmark.methods.srm.trainer import train_srm


def load_dataset(data_name, split, seed):
    path = os.path.join(
        "datasets", "synthetic", data_name, f"seed_{seed}", f"{split}.npz"
    )
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)

    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    return X, Y


def normalize_tensor(tensor):
    mean = tensor.mean(0, keepdim=True)
    std = tensor.std(0, keepdim=True) + 1e-8
    return (tensor - mean) / std, mean, std


def apply_normalization(tensor, mean, std):
    return (tensor - mean) / (std + 1e-8)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, Y_train = load_dataset(args.data_name, "train", args.seed)
    X_val, Y_val = load_dataset(args.data_name, "valid", args.seed)

    Y_train, y_mean, y_std = normalize_tensor(Y_train)
    Y_val = apply_normalization(Y_val, y_mean, y_std)

    y_mean = y_mean.view(1, -1)
    y_std = y_std.view(1, -1)

    workspace_dir = os.path.join(
        "workspaces", "srm", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(workspace_dir, exist_ok=True)

    norm_path = os.path.join(workspace_dir, "Y_normalization_params.npz")
    np.savez(norm_path, mean=y_mean.numpy(), std=y_std.numpy())
    print(f"Saved Y normalization parameters to {norm_path}")

    x_dim = X_train.shape[1]
    y_dim = Y_train.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    args.hidden_dims = hidden_dims

    print(f"[SRM] x_dim={x_dim} y_dim={y_dim}")
    print(f"[SRM] hidden_dims={hidden_dims}")
    print(f"[SRM] device={device}")

    # The synthetic datasets are tiny relative to GPU memory. Move the full
    # train/validation tensors once and keep all mini-batching on the GPU.
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    generator = SRMGenerator(
        x_dim=x_dim,
        noise_dim=args.noise_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    start = time.time()
    history_train, history_val, epoch_times = train_srm(
        generator=generator,
        x_train=X_train,
        y_train=Y_train,
        x_val=X_val,
        y_val=Y_val,
        batch_size=args.batch_size,
        n_samples=args.sampling_size_for_training,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.energy_beta,
        early_stop=args.early_stop,
        patience=args.patience,
        print_every=args.print_every,
    )
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    torch.save(
        generator.state_dict(), os.path.join(workspace_dir, "generator.pth")
    )
    print(f"Model saved to {workspace_dir}")

    with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    print(f"Config saved to {workspace_dir}/config.yaml")

    np.savez(
        os.path.join(workspace_dir, "losses.npz"),
        train_loss=np.asarray(history_train),
        val_loss=np.asarray(history_val),
    )

    plt.figure(figsize=(6, 4))
    plt.plot(history_train, label="Train SRM loss", linewidth=1.8)
    if len(history_val) > 0:
        plt.plot(history_val, label="Validation SRM loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves (SRM)")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(workspace_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Loss curves saved to {fig_path}")

    with open(os.path.join(workspace_dir, "epoch_times.json"), "w") as f:
        json.dump({"epoch_times": epoch_times}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Match the existing benchmark scripts: default seed=230 and data-name=U1.
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="U1")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--noise-dim", type=int, default=3)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="1024,512,256",
        help="Comma-separated hidden layer sizes, e.g. 1024,512,256.",
    )
    parser.add_argument(
        "--sampling-size-for-training",
        type=int,
        default=30,
        help=(
            "Number of generated Y samples per observation used to estimate "
            "the scoring-rule loss during training. This controls the "
            "Monte Carlo approximation inside training, not the size of "
            "the training dataset and not the final predictive sampling size."
        ),
    )
    parser.add_argument("--energy-beta", type=float, default=1.0)

    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--print-every", type=int, default=1)

    args = parser.parse_args()

    print("[SRM] Experiment Settings:")
    print(f" Data name     : {args.data_name}")
    print(f" Seed          : {args.seed}")
    print(f" Batch size    : {args.batch_size}")
    print(f" Epochs        : {args.epochs}")
    print(f" Hidden dims   : {args.hidden_dims}")
    print(f" Noise dim     : {args.noise_dim}")
    print(f" Training MC samples : {args.sampling_size_for_training}")
    print(f" Energy beta   : {args.energy_beta}")
    print(f" Learning rate : {args.lr}")

    main(args)

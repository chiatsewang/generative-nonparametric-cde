# scripts/train_gcds.py
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from condgen_benchmark.methods.gcds.core import (
    GCDSDiscriminator,
    GCDSGenerator,
)
from condgen_benchmark.methods.gcds.trainer import train_gcds


def load_dataset(mode, split, seed):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", f"{split}.npz"
    )
    data = np.load(path)
    x = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["Y"], dtype=torch.float32)
    return x, y


def normalize_tensor(tensor):
    mean = tensor.mean(0, keepdim=True)
    std = tensor.std(0, keepdim=True) + 1e-8
    normed = (tensor - mean) / std
    return normed, mean, std


def apply_normalization(tensor, mean, std):
    return (tensor - mean) / (std + 1e-8)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    y_train, x_train = load_dataset(args.data_name, "train", args.seed)
    y_val, x_val = load_dataset(args.data_name, "valid", args.seed)

    x_train, x_mean, x_std = normalize_tensor(x_train)
    x_val = apply_normalization(x_val, x_mean, x_std)

    x_mean = x_mean.view(1, -1)
    x_std = x_std.view(1, -1)

    print(f"X mean: {x_mean.numpy().flatten()}")
    print(f"X std:  {x_std.numpy().flatten()}")

    norm_dir = os.path.join(
        "datasets", "synthetic", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(norm_dir, exist_ok=True)
    norm_path = os.path.join(norm_dir, "x_normalization_params.npz")
    np.savez(norm_path, mean=x_mean.numpy(), std=x_std.numpy())
    print(f"Saved X normalization parameters to {norm_path}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    # val_loader = DataLoader(
    #     TensorDataset(x_val, y_val),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    # )

    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]

    print(f"[GCDS] x_dim ={x_dim} y_dim={y_dim}")

    noise_dim = 3

    generator = GCDSGenerator(
        y_dim, noise_dim, x_dim, hidden_dim=args.hidden_dim
    ).to(device)
    discriminator = GCDSDiscriminator(
        x_dim, y_dim, hidden_dims=(args.hidden_dim, args.hidden_dim // 2)
    ).to(device)

    start = time.time()

    history_g, history_d, epoch_times = train_gcds(
        generator,
        discriminator,
        train_loader,
        noise_dim=noise_dim,
        num_epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        device=device,
    )
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    workspace_dir = os.path.join(
        "workspaces", "gcds", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(workspace_dir, exist_ok=True)

    torch.save(
        generator.state_dict(), os.path.join(workspace_dir, "generator.pth")
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(workspace_dir, "discriminator.pth"),
    )
    print(f"Models saved to {workspace_dir}")

    with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    print(f"Config saved to {workspace_dir}/config.yaml")

    # draw traning loss trajectories
    plt.figure(figsize=(6, 4))
    plt.plot(history_g, label="Generator loss", linewidth=1.8)
    plt.plot(history_d, label="Discriminator loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves (GCDS)")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(workspace_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Loss curves saved to {fig_path}")

    # Save epoch eplased time
    with open(os.path.join(workspace_dir, "epoch_times.json"), "w") as f:
        json.dump({"epoch_times": epoch_times}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="M1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for both generator and discriminator (optional).",
    )
    parser.add_argument(
        "--lr-g", type=float, default=None, help="Learning rate for generator."
    )
    parser.add_argument(
        "--lr-d",
        type=float,
        default=None,
        help="Learning rate for discriminator.",
    )
    parser.add_argument("--hidden-dim", type=int, default=50)
    args = parser.parse_args()

    if args.lr is not None:
        args.lr_g = args.lr if args.lr_g is None else args.lr_g
        args.lr_d = args.lr if args.lr_d is None else args.lr_d

    # set the default values for lr
    if args.lr_g is None:
        args.lr_g = 1e-4
    if args.lr_d is None:
        args.lr_d = 1e-4

    # display the given parameters for the simulation
    print("[GCDS] Experiment Settings:")
    print(f" Data name      : {args.data_name}")
    print(f" Seed           : {args.seed}")
    print(f" Batch size     : {args.batch_size}")
    print(f" Epochs         : {args.epochs}")
    print(f" Hidden dim     : {args.hidden_dim}")
    print(f" Learning rate G: {args.lr_g}")
    print(f" Learning rate D: {args.lr_d}")

    main(args)

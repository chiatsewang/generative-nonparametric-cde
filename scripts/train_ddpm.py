import argparse
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from condgen_benchmark.methods.ddpm.diffusion import Diffusion
from condgen_benchmark.methods.ddpm.loss import ddpm_loss
from condgen_benchmark.methods.ddpm.model import (
    ConditionalMLPScalarT,
    parse_hidden_dims,
)
from condgen_benchmark.utils.gpu_batching import batch_indices


def load_dataset(mode, split, seed):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", f"{split}.npz"
    )
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    return X, Y


def normalize_tensor(tensor):
    mean = tensor.mean(0, keepdim=True)
    std = tensor.std(0, keepdim=True) + 1e-8
    normed = (tensor - mean) / std
    return normed, mean, std


def apply_normalization(tensor, mean, std):
    return (tensor - mean) / (std + 1e-8)


@torch.no_grad()
def evaluate(model, diffusion, Y, X, batch_size):
    model.eval()
    device = Y.device
    n = Y.shape[0]
    total_loss = torch.zeros((), device=device)

    for idx in batch_indices(n, batch_size, device, shuffle=False):
        Yb = Y[idx]
        Xb = X[idx]
        tb = torch.randint(
            0, diffusion.timesteps, (Yb.shape[0],), device=device
        )
        tb_norm = (tb.float() / diffusion.timesteps).unsqueeze(1)
        loss = ddpm_loss(model, diffusion, Yb, Xb, tb, tb_norm)
        total_loss += loss.detach() * Yb.shape[0]

    return (total_loss / n).item()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"[Device] Using {device}")

    X_train, Y_train = load_dataset(args.data_name, "train", args.seed)
    X_val, Y_val = load_dataset(args.data_name, "valid", args.seed)

    if args.normalization:
        Y_train, Y_mean, Y_std = normalize_tensor(Y_train)
        Y_val = apply_normalization(Y_val, Y_mean, Y_std)
        print("[DDPM] We will normalize the dependent variable Y")
        print(f"Y mean: {Y_mean.numpy().flatten()}")
        print(f"Y std:  {Y_std.numpy().flatten()}")
    else:
        print("[DDPM] We do not normalize the dependent variable Y")
        d = Y_train.shape[1]
        Y_mean = torch.zeros(1, d, device=Y_train.device, dtype=Y_train.dtype)
        Y_std = torch.ones(1, d, device=Y_train.device, dtype=Y_train.dtype)

    # Synthetic datasets are tiny: transfer once, then batch entirely on GPU.
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    Y_dim = Y_train.shape[1]
    X_dim = X_train.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    args.hidden_dims = hidden_dims
    print(f"[DDPM] hidden_dims={hidden_dims}")

    model = ConditionalMLPScalarT(
        X_dim,
        Y_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_drop_period, gamma=args.lr_drop_factor
    )

    start_all = time.time()
    best_val_loss = float("inf")
    best_model_state = None

    # For early stopping
    best_epoch = 0
    epochs_no_improve = 0

    # For drawing curves and remember each epoch elapsed time
    train_losses, val_losses = [], []
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_epoch = time.time()
        total_loss = torch.zeros((), device=device)
        n_train = Y_train.shape[0]

        for idx in batch_indices(
            n_train, args.batch_size, device, shuffle=True
        ):
            Yb = Y_train[idx]
            Xb = X_train[idx]
            tb = torch.randint(
                0, diffusion.timesteps, (Yb.shape[0],), device=device
            )
            tb_norm = (tb.float() / diffusion.timesteps).unsqueeze(1)

            loss = ddpm_loss(model, diffusion, Yb, Xb, tb, tb_norm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * Yb.shape[0]

        scheduler.step()  # Adjust learning rate
        avg_train_loss = (total_loss / n_train).item()
        val_loss = evaluate(model, diffusion, Y_val, X_val, args.batch_size)

        # Record losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Check if the model is improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stop if not improved after patience times
        if args.early_stop and epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch} (best at {best_epoch}: {best_val_loss:.6f})"
            )
            break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_epoch = time.time()
        elapsed_time = end_epoch - start_epoch
        epoch_times.append(elapsed_time)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}: "
            f"train_loss = {avg_train_loss:.4f} | val_loss = {val_loss:.4f} | "
            f"lr = {current_lr:.5f} | time = {elapsed_time:.2f}s"
        )

    end_all = time.time()
    print(f"Training completed in {end_all - start_all:.2f} seconds.")

    # === Create workspace directory ===
    workspace_dir = os.path.join(
        "workspaces", "ddpm", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(workspace_dir, exist_ok=True)

    # === Save Y normlization params ===
    norm_path = os.path.join(workspace_dir, "Y_normalization_params.npz")
    np.savez(norm_path, mean=Y_mean.numpy(), std=Y_std.numpy())
    print(f"Saved Y normalization parameters to {workspace_dir}")

    # === Save models ===
    model_path = os.path.join(workspace_dir, "model_last.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved last model to {model_path}")

    if best_model_state is not None:
        best_model_path = os.path.join(workspace_dir, "model_best.pth")
        torch.save(best_model_state, best_model_path)
        print(f"Saved best model to {best_model_path}")

    # === Save config ===
    config_path = os.path.join(workspace_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"Saved config to {config_path}")

    # === Save loss curves ===
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(workspace_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")

    # === Save epoch eplased time ===
    with open(os.path.join(workspace_dir, "epoch_times.json"), "w") as f:
        json.dump({"epoch_times": epoch_times}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230, help="Dataset seed")
    parser.add_argument(
        "--data-name", type=str, default="U1", help="Synthetic data model"
    )
    parser.add_argument(
        "--normalization",
        action="store_true",  # if exits: True, otherwise: False for normalization of Y
        help="Standardize Y using train mean/std (default: off)",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation loss (default: off)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs with no improvement to tolerate before early stopping",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="1024,512,256",
        help="Comma-separated hidden layer sizes, e.g. 1024,512,256.",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--lr-drop-period", type=int, default=10)
    args = parser.parse_args()
    main(args)

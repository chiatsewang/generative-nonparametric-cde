import argparse
import copy
import json
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from condgen_benchmark.methods.deepcde.bases.cosine import CosineBasis
from condgen_benchmark.methods.deepcde.bases.wavelet import WaveletBasis
from condgen_benchmark.methods.deepcde.deepcde_pytorch import cde_loss
from condgen_benchmark.methods.deepcde.models import DeepCDEModel
from condgen_benchmark.methods.deepcde.utils import box_transform

matplotlib.use("Agg")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        beta = model(xb)
        loss = criterion(beta, yb)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cpu"
    )  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    def load_dataset(mode, split, seed):
        path = os.path.join(
            "datasets", "synthetic", mode, f"seed_{seed}", f"{split}.npz"
        )
        data = np.load(path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        Y = torch.tensor(data["Y"], dtype=torch.float32)
        return X, Y

    X_train, Y_train = load_dataset(args.data_name, "train", args.seed)
    X_val, Y_val = load_dataset(args.data_name, "valid", args.seed)

    # Normalize X (predictor)
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Box-transform Y (response)
    y_min, y_max = float(Y_train.min()), float(Y_train.max())
    eps = 1e-6
    if args.basis == "cosine":
        basis = CosineBasis(args.n_basis)
    else:
        basis = WaveletBasis(
            n_basis=args.n_basis, family=args.family, n_aux=args.aux
        )

    y_train_unit = box_transform(Y_train.numpy().reshape(-1, 1), y_min, y_max)
    y_val_unit = box_transform(
        np.clip(Y_val.numpy().reshape(-1, 1), y_min + eps, y_max - eps),
        y_min,
        y_max,
    )
    y_train_basis = basis.evaluate(y_train_unit)[:, 1:].astype(np.float32)
    y_val_basis = basis.evaluate(y_val_unit)[:, 1:].astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, torch.from_numpy(y_train_basis)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, torch.from_numpy(y_val_basis)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # model and optimizer
    model = DeepCDEModel(input_dim=X_train.shape[1], n_basis=args.n_basis).to(
        device
    )
    criterion = cde_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop starts from here
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_hist, val_hist = [], []
    epoch_times = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            beta = model(xb)
            loss = criterion(beta, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_hist.append(avg_train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:4d}: train={avg_train_loss:.4f} | val={val_loss:.4f} | time={elapsed:.2f}s"
        )
        epoch_times.append(elapsed)

        # Early stop
        if args.early_stop and no_improve >= args.patience:
            print(
                f"No improvement for {args.patience} epoch(s); early stopping."
            )
            break

    print(
        f"Training completed in {time.time() - t0:.2f}s. Best val={best_val_loss:.4f}"
    )

    # create the workspace folder
    ws = os.path.join(
        "workspaces", "deepcde", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(ws, exist_ok=True)

    # save model files
    torch.save(model.state_dict(), os.path.join(ws, "model_last.pth"))
    if best_state is not None:
        torch.save(best_state, os.path.join(ws, "model_best.pth"))

    # save meta info
    with open(os.path.join(ws, "y_box_range.json"), "w") as f:
        json.dump({"y_min": y_min, "y_max": y_max}, f)
    with open(os.path.join(ws, "basis.json"), "w") as f:
        json.dump(
            (
                {"type": args.basis, "n_basis": int(args.n_basis)}
                if args.basis == "cosine"
                else {
                    "type": "wavelet",
                    "n_basis": int(args.n_basis),
                    "family": args.family,
                    "aux": int(args.aux),
                }
            ),
            f,
        )
    with open(os.path.join(ws, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # save X normalization params
    np.savez(
        os.path.join(ws, "X_normalization_params.npz"),
        mean=mean.numpy(),
        std=std.numpy(),
    )

    # Plot the loss curve (save in the workspace)
    with open(os.path.join(ws, "loss_history.json"), "w") as f:
        json.dump({"train": train_hist, "val": val_hist}, f)
    plt.figure()
    plt.plot(range(1, len(train_hist) + 1), train_hist, label="train")
    plt.plot(range(1, len(val_hist) + 1), val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ws, "loss_curve.png"), dpi=150)
    plt.close()

    # save epoch eplased time
    with open(os.path.join(ws, "epoch_times.json"), "w") as f:
        json.dump({"epoch_times": epoch_times}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="M1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--basis", type=str, default="cosine", choices=["cosine", "wavelet"]
    )
    parser.add_argument("--n-basis", dest="n_basis", type=int, default=31)
    parser.add_argument("--family", type=str, default="db1")
    parser.add_argument("--aux", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)  # learning rate
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="enable patience-based early stopping",
    )  # enable early stop
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="epochs w/o improvement before stop",
    )  # patience: number of steps to wait for the next improvement

    args = parser.parse_args()
    main(args)

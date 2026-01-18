import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from condgen_benchmark.methods.deepcde.bases.cosine import CosineBasis
from condgen_benchmark.methods.deepcde.bases.wavelet import WaveletBasis
from condgen_benchmark.methods.deepcde.deepcde_pytorch import cde_predict
from condgen_benchmark.methods.deepcde.models import DeepCDEModel


def load_test_dataset(mode: str, seed: int):
    """Load X (cond) and Y (true) from test split as numpy arrays."""
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", "test.npz"
    )
    data = np.load(path)
    X = data["X"]  # (N, d)
    Y = data["Y"]  # (N,) or (N,1)
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y.squeeze(1)
    return X, Y


def apply_normalization(
    x_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Standardize by saved mean/std (broadcast)."""
    return (x_tensor - mean) / std


def sample_Y_given_X(
    cde_mat: np.ndarray, y_grid: np.ndarray, num_samples: int
) -> np.ndarray:
    """
    Draw samples per row from discretized pdf on y_grid.
    cde_mat: (N, G) nonnegative pdf values on y_grid
    y_grid:  (G,) increasing grid in y-space
    return:  (N, num_samples, 1)
    """
    N, G = cde_mat.shape
    probs = np.clip(cde_mat, 0.0, None).astype(np.float64)
    sums = probs.sum(axis=1, keepdims=True)
    bad = (sums[:, 0] <= 0.0) | ~np.isfinite(sums[:, 0])
    if np.any(bad):
        probs[bad] = 1.0
        sums[bad] = G
    probs /= sums

    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0

    u = np.random.rand(N, num_samples)
    idx = np.empty((N, num_samples), dtype=np.int64)
    # vectorized searchsorted per row
    for i in range(N):
        idx[i] = np.searchsorted(cdf[i], u[i], side="right")
    np.clip(idx, 0, G - 1, out=idx)

    samples = y_grid[idx]  # (N, S)
    return samples[..., None].astype(np.float32)  # (N, S, 1)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cpu"
    )  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # load test data
    X_np, Y_true = load_test_dataset(args.data_name, args.seed)
    N, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32)

    # define workspace path
    ws = Path("workspaces") / "deepcde" / args.data_name / f"seed_{args.seed}"

    # normalization params
    norm_path = ws / "X_normalization_params.npz"
    norm = np.load(norm_path)
    X_mean = torch.tensor(norm["mean"], dtype=torch.float32).view(
        1, -1
    )  # (1,d)
    X_std = torch.tensor(norm["std"], dtype=torch.float32).view(1, -1)  # (1,d)
    Xz = apply_normalization(X_t, X_mean, X_std)

    # model files
    ckpt = ws / (
        "model_best.pth" if args.model_choice == "best" else "model_last.pth"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # y box range
    with open(ws / "y_box_range.json", "r") as f:
        y_meta = json.load(f)
    y_min, y_max = float(y_meta["y_min"]), float(y_meta["y_max"])

    # basis
    with open(ws / "basis.json", "r") as f:
        bmeta = json.load(f)

    if bmeta["type"] == "cosine":
        basis = CosineBasis(int(bmeta["n_basis"]))
    elif bmeta["type"] == "wavelet":
        basis = WaveletBasis(
            n_basis=int(bmeta["n_basis"]),
            family=bmeta["family"],
            n_aux=int(bmeta["aux"]),
        )
    else:
        raise ValueError(f"Unknown basis type: {bmeta['basis']}")

    # Build model and basis
    model = (
        DeepCDEModel(input_dim=d, n_basis=int(bmeta["n_basis"]))
        .to(device)
        .eval()
    )
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Create grids: u in [0,1], y mapped linearly for reporting/sampling
    G = int(args.ny)
    u_grid_np = np.linspace(
        0.0, 1.0, G, dtype=np.float32
    )  # grid for cde_predict
    y_grid_np = y_min + (y_max - y_min) * u_grid_np  # corresponding y-grid

    # Bump removal parameters
    delta = args.delta
    bin_size = u_grid_np[1] - u_grid_np[0]

    # Evaluate pdf on grid (returns p_Y on y_grid; matches cde_predict signature)
    rows = []
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            xb = Xz[i : i + args.batch_size].to(device)  # (B,d)
            beta = model(xb).detach().cpu().numpy()  # (B, n_basis)
            p_y = cde_predict(
                beta, y_min, y_max, u_grid_np, basis, delta, bin_size
            )  # (B, G), already in y-space
            rows.append(p_y)
    cde_mat = np.vstack(rows).astype(np.float32)  # (N, G)

    # Draw samples for each given X
    Y_hat = sample_Y_given_X(
        cde_mat, y_grid_np, num_samples=args.num_samples
    )  # (N, S, 1)
    Y_hat = np.clip(
        Y_hat, y_min, y_max
    )  # clip ourside the range [y_min, y_max]

    # Save the samples
    out_path = ws / "sampled.npz"
    np.savez(
        out_path,
        Y_hat=Y_hat,  # (N, S, 1) samples of Y|X
        X=X_np,  # (N, d)  inputs (named as in flex sampler)
        Y_true=Y_true,
    )
    print(f"Saved: {out_path}")
    print(
        f"Shapes -> X_hat: {Y_hat.shape}, Y: {X_np.shape}, X_true: {Y_true.shape}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="DeepCDE sampler (flex-like interface)"
    )
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--data-name", type=str, default="M1")
    p.add_argument(
        "--num-samples", type=int, default=2000, help="samples per X"
    )
    p.add_argument(
        "--ny", type=int, default=200, help="number of grid points for u/y"
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--model-choice",
        type=str,
        choices=["last", "best"],
        default="best",
        help="which checkpoint to use",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Absolute area threshold for bump removal (None disables).",
    )
    args = p.parse_args()
    main(args)

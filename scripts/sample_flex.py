import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals, safe_globals

# Add FlexCodeModel to the PyTorch 2.6+ serialization allowlist
# to enable loading pickled model checkpoints
try:
    from flexcode.core import FlexCodeModel

    add_safe_globals([FlexCodeModel])
except Exception:
    pass


# Data loading
def load_test_dataset(data_dir, mode: str, seed: int):
    path = os.path.join(data_dir, mode, f"seed_{seed}", "test.npz")
    data = np.load(path)
    X = data["X"]  # (n, p)
    Y_true = data["Y"]  # (n,) or (n,1)
    if Y_true.ndim == 2 and Y_true.shape[1] == 1:
        Y_true = Y_true.squeeze(1)
    return X, Y_true


# Sampling helper
def sample_Y_given_X(
    cde: np.ndarray, grid: np.ndarray, num_samples: int
) -> np.ndarray:
    """
    cde:  (N, G)  Non-negative values representing the discretized conditional density p(Y | X = X_cond[i]) on the y_grid for each X_cond[i]
    grid: (G,)    Grid values of the response variable Y
    return: (N, S, 1)
    """
    N, G = cde.shape
    probs = np.clip(cde, 0.0, None).astype(np.float64)

    # Normalize row-wise
    probs_sum = probs.sum(axis=1, keepdims=True)
    bad = (probs_sum[:, 0] <= 0.0) | ~np.isfinite(probs_sum[:, 0])
    if np.any(bad):
        probs[bad] = 1.0
        probs_sum[bad] = G
    probs /= probs_sum

    # Build the CDF (set the last entry to 1.0 to avoid numerical issues)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0

    # Draw uniform random numbers
    u = np.random.rand(N, num_samples)

    # Apply searchsorted row-wise
    idx = np.empty((N, num_samples), dtype=np.int64)
    for i in range(N):
        idx[i] = np.searchsorted(cdf[i], u[i], side="right")

    # Safety clipping
    np.clip(idx, 0, G - 1, out=idx)

    samples = grid[idx]  # (N, S)
    return samples[..., None]  # (N, S, 1)


def predict_on_training_grid(model, X_cond: np.ndarray):
    """Predict conditional densities on the response grid stored at training."""
    if not hasattr(model, "y_grid") or model.y_grid is None:
        raise ValueError(
            "The FlexCode checkpoint has no saved response grid (y_grid). "
            "Retrain the model with train-flex to save it."
        )

    y_grid = np.asarray(model.y_grid, dtype=np.float64).reshape(-1)
    if (
        y_grid.size < 2
        or not np.all(np.isfinite(y_grid))
        or not np.all(np.diff(y_grid) > 0)
    ):
        raise ValueError(
            "Saved FlexCode response grid must be finite and increasing."
        )

    prediction = model.predict(X_cond, n_grid=y_grid.size)
    if isinstance(prediction, tuple):
        cde_mat, predicted_grid = prediction
    else:
        cde_mat, predicted_grid = prediction, None

    cde_mat = np.asarray(cde_mat)
    if cde_mat.shape != (len(X_cond), len(y_grid)):
        raise ValueError(
            "FlexCode density shape does not match the training response grid: "
            f"density={cde_mat.shape}, expected={(len(X_cond), len(y_grid))}."
        )

    if predicted_grid is not None:
        predicted_grid = np.asarray(predicted_grid, dtype=np.float64).reshape(
            -1
        )
        if predicted_grid.shape != y_grid.shape or not np.allclose(
            predicted_grid, y_grid, rtol=1e-7, atol=1e-9
        ):
            raise ValueError(
                "FlexCode predict() returned a response grid different from "
                "the one saved during training."
            )

    return cde_mat, y_grid


# Main
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load test set
    X_cond, Y_true = load_test_dataset(
        args.data_dir, args.data_name, args.seed
    )

    # Load trained model (with embedded y_grid)
    ws = Path("workspaces") / "flex" / args.data_name / f"seed_{args.seed}"
    if args.tuned_model:
        print("[FlexCode] Use the tuned model")
        model_path = ws / "model_tuned.pth"
    else:
        print("[FlexCode] Use the original model")
        model_path = ws / "model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        model = torch.load(model_path, weights_only=False)
    except Exception:
        try:
            from flexcode.core import FlexCodeModel

            with safe_globals([FlexCodeModel]):
                model = torch.load(model_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with safe allowlist. {type(e).__name__}: {e}"
            )

    # Use the response grid saved during training. The sampling command
    # must not independently choose the number or locations of grid points.
    cde_mat, y_grid = predict_on_training_grid(model, X_cond)

    # Draw samples of Y|X
    Y_hat = sample_Y_given_X(cde_mat, y_grid, num_samples=args.num_samples)

    # === Save outputs ===
    out_path = ws / "sampled.npz"
    np.savez(
        out_path,
        Y_hat=Y_hat,  # (N, S, 1)
        X=X_cond,  # (N, p)
        Y_true=Y_true,  # (N,)
    )
    print(f"[FlexCode] Saved samples to {out_path}")
    print(
        f"Shapes: Y_hat={Y_hat.shape} (N,S,1), X={np.shape(X_cond)}, Y_true={np.shape(Y_true)}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Sample from FlexCode (grid embedded; no extra npy)"
    )
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--data-name", type=str, default="U1")
    p.add_argument(
        "--data-dir",
        type=str,
        default="datasets/synthetic",
        help="Folder holding the generated datasets",
    )
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--tuned-model",
        action="store_true",
        help="Use tuned FlexCode model (if specified, load *_tuned.pkl instead of default)",
    )
    args = p.parse_args()
    main(args)

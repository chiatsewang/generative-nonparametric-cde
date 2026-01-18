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
def load_test_dataset(mode: str, seed: int):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", "test.npz"
    )
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


# Main
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load test set
    X_cond, Y_true = load_test_dataset(args.data_name, args.seed)

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

    # Predict conditional density on a grid
    pred = model.predict(X_cond, n_grid=args.ny)
    if isinstance(pred, tuple):
        cde_mat, y_grid_pred = pred
    else:
        cde_mat, y_grid_pred = pred, None

    # Decide grid priority: model.y_grid > predict's grid > fallback
    if hasattr(model, "y_grid"):
        y_grid = np.asarray(model.y_grid).ravel()
    elif y_grid_pred is not None:
        y_grid = np.asarray(y_grid_pred).ravel()
    else:
        y_min = float(np.min(Y_true))
        y_max = float(np.max(Y_true))
        y_grid = np.linspace(y_min, y_max, args.ny)

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
    p.add_argument("--data-name", type=str, default="M1")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--ny", type=int, default=200, help="n_grid for predict if needed"
    )
    p.add_argument(
        "--tuned-model",
        action="store_true",
        help="Use tuned FlexCode model (if specified, load *_tuned.pkl instead of default)",
    )
    args = p.parse_args()
    main(args)

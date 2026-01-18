import argparse
import glob
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize repeated simulation results"
    )
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        help="Synthetic data mode (e.g., M1, T3)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ddpm",
        help="Algorithm name (e.g., ddpm, hyalg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(
        "workspaces", args.algorithm, args.data_name, "seed_*/eval.npz"
    )
    eval_files = sorted(glob.glob(pattern))

    if not eval_files:
        print(f"[Error] No eval.npz files found under {pattern}")
        return

    mse_means = []
    mse_stds = []
    w_means = []

    for path in eval_files:
        try:
            data = np.load(path)
            mse_means.append(data["mse_mean"])
            mse_stds.append(data["mse_std"])
            w_means.append(data["w_mean"])
        except Exception as e:
            print(f"[Warning] Failed to read {path}: {e}")

    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)
    w_means = np.array(w_means)

    prefix = f"{args.algorithm.upper()}-{args.data_name}"
    print(f"[{prefix}] Aggregated from {len(mse_means)} seeds:")
    print(f"MSE Mean: {mse_means.mean():.4f} ± {mse_means.std():.4f}")
    print(f"MSE Std : {mse_stds.mean():.4f} ± {mse_stds.std():.4f}")
    print(f"W1 Avg  : {w_means.mean():.4f} ± {w_means.std():.4f}")


if __name__ == "__main__":
    main()

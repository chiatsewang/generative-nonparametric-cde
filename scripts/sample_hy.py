"""Sample Hall--Yao conditional distributions using trained (theta, H)."""

import argparse
import os

import numpy as np

from condgen_benchmark.methods.hy.io import load_split
from condgen_benchmark.methods.hy.sampler import (
    response_grid,
    sample_conditional,
)


def main(args):
    X_train, Y_train = load_split(
        args.data_dir, args.data_name, args.seed, "train"
    )
    X_test, Y_true = load_split(
        args.data_dir, args.data_name, args.seed, "test"
    )
    workspace = os.path.join(
        "workspaces", "hy", args.data_name, f"seed_{args.seed}"
    )
    with np.load(os.path.join(workspace, "theta.npz")) as model:
        theta = model["theta"]
        H = float(model["H"])
    draws = sample_conditional(
        X_train,
        Y_train,
        X_test,
        theta,
        H,
        args.num_samples,
        response_grid(Y_train, args.ny),
        rng=np.random.default_rng(args.seed),
    )
    outfile = os.path.join(workspace, "sampled.npz")
    np.savez(
        outfile,
        Y_hat=draws[:, :, None].astype(np.float32),
        X=X_test.astype(np.float32),
        Y_true=Y_true[:, None].astype(np.float32),
    )
    print(f"[HY] saved {outfile}: {draws.shape[0]} x {draws.shape[1]}, H={H}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="U1")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/synthetic",
        help="Folder holding the generated datasets",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--ny", type=int, default=200)
    main(parser.parse_args())

import argparse
import os

import numpy as np

from condgen_benchmark.data.synthetic import get_synthetic_data_class
from condgen_benchmark.utils.eval_energy_score import oracle_energy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic datasets (U1–U7, O7 and M1–M4) with fixed "
            "train/valid/test sizes and save under seed-specific directories."
        )
    )
    parser.add_argument(
        "--data-name",
        type=str,
        choices=[
            "U1",
            "U2",
            "U3",
            "U4",
            "U5",
            "U6",
            "U7",
            "O7",
            "M1",
            "M2",
            "M3",
            "M4",
        ],
        required=True,
        help="Which synthetic data model to generate.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000).",
    )
    parser.add_argument(
        "--valid-samples",
        type=int,
        default=2000,
        help="Number of validation samples (default: 2000).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2000,
        help="Number of test samples (default: 2000).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/synthetic",
        help="Directory to save .npz files (default: datasets/synthetic).",
    )
    parser.add_argument(
        "--seed", type=int, default=230, help="Random seed (default: 230)."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional data-generation parameter.",
    )

    parser.add_argument(
        "--oracle-samples",
        type=int,
        default=1000,
        help="Conditional samples per test predictor for oracle EScore (default: 1000).",
    )
    parser.add_argument(
        "--escore-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for pairwise Energy Score computation (default: auto).",
    )

    return parser.parse_args()


def save_split(X, Y, out_dir, name, **extra):
    out_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(out_path, X=X, Y=Y, **extra)
    print(f"Saved {name}: {out_path} (X: {X.shape}, Y: {Y.shape})")


def main():
    """
    Generate synthetic dataset and save train/valid/test splits in .npz format
    using the class-based SyntheticData interface.
    """
    args = parse_args()
    if args.oracle_samples < 2:
        raise ValueError("--oracle-samples must be at least 2")

    data_class = get_synthetic_data_class(data_name=args.data_name)

    # Optional parameter
    allowed_args = [
        "seed",
        "gamma",
    ]  # if necessary, you can add other parameters
    init_kwargs = {
        k: v
        for k in allowed_args
        if hasattr(args, k) and (v := getattr(args, k)) is not None
    }
    generator = data_class(**init_kwargs)

    # Display message
    print("=" * 60)
    print(f"Generating synthetic data: {args.data_name}")
    for k, v in init_kwargs.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    total = args.train_samples + args.valid_samples + args.test_samples
    Y, X = generator.generate(total)  # P(Y|X): returns (Y, X)

    indices = np.random.permutation(total)
    train_idx = indices[: args.train_samples]
    valid_idx = indices[
        args.train_samples : args.train_samples + args.valid_samples
    ]
    test_idx = indices[args.train_samples + args.valid_samples :]

    out_dir = os.path.join(args.out_dir, args.data_name, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    save_split(X[train_idx], Y[train_idx], out_dir, "train")
    save_split(X[valid_idx], Y[valid_idx], out_dir, "valid")
    # Compute the oracle with exactly the X_i stored in test.npz, but draw
    # NEW independent responses from each true P(Y | X_i). Do not reuse the
    # single observed Y_i in test.npz. The generator returns rows grouped by
    # test predictor, so reshape to (n_test, oracle_samples, response_dim).
    X_test = X[test_idx]
    Y_oracle, _ = generator.generate(n_samples=args.oracle_samples, X=X_test)
    Y_oracle = Y_oracle.reshape(len(X_test), args.oracle_samples, -1)
    ES_oracle = oracle_energy_score(Y_oracle, device=args.escore_device)
    # Only save the scalar oracle score; the Monte Carlo samples are discarded.
    save_split(
        X_test,
        Y[test_idx],
        out_dir,
        "test",
        ES_oracle=np.float64(ES_oracle),
    )
    print(f"Oracle EScore: {ES_oracle:.6f} (M={args.oracle_samples})")


if __name__ == "__main__":
    main()

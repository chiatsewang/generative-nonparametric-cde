import argparse
import os

import numpy as np

from condgen_benchmark.data.synthetic import get_synthetic_data_class


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic dataset (M1–M10) with fixed "
            "train/valid/test sizes and save under seed-specific directories."
        )
    )
    parser.add_argument(
        "--data-name",
        type=str,
        choices=[
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "M10",
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
        "--rho",
        type=float,
        default=None,
        help="Optional data-generation parameter (currently used only in M3).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional data-generation parameter.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Optional data-generation parameter (currently used only in M9).",
    )

    return parser.parse_args()


def save_split(X, Y, out_dir, name):
    out_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(out_path, X=X, Y=Y)
    print(f"Saved {name}: {out_path} (X: {X.shape}, Y: {Y.shape})")


def main():
    """
    Generate synthetic dataset and save train/valid/test splits in .npz format
    using the class-based SyntheticData interface.
    """
    args = parse_args()

    data_class = get_synthetic_data_class(data_name=args.data_name)

    # Optional parameter
    allowed_args = [
        "seed",
        "rho",
        "c",
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
    save_split(X[test_idx], Y[test_idx], out_dir, "test")


if __name__ == "__main__":
    main()

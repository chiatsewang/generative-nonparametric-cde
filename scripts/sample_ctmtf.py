# scripts/sample_ctmtf.py
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def get_y_dim(npz_path: Path) -> int:
    with np.load(npz_path) as data:
        y = data["Y"]
        if y.ndim == 1:
            return 1
        return int(y.shape[1])


def main(args: argparse.Namespace) -> None:
    data_dir = (
        Path(args.data_dir).resolve() / args.data_name / f"seed_{args.seed}"
    )
    test_path = data_dir / "test.npz"

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    y_dim = get_y_dim(test_path)
    if y_dim != 1:
        raise ValueError(
            f"CTMTF supports only univariate Y, but test.npz has y_dim={y_dim}"
        )

    workspace_dir = (
        ROOT / "workspaces" / "ctmtf" / args.data_name / f"seed_{args.seed}"
    )
    model_path = workspace_dir / "model.rds"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained CTMTF model not found: {model_path}")

    r_sampler = ROOT / "condgen_benchmark" / "methods" / "ctmtf" / "sampler.R"
    if not r_sampler.exists():
        raise FileNotFoundError(f"CTMTF R sampler not found: {r_sampler}")

    cmd = [
        "Rscript",
        str(r_sampler),
        "--data_name",
        args.data_name,
        "--data_dir",
        str(Path(args.data_dir).resolve()),
        "--seed",
        str(args.seed),
        "--num_samples",
        str(args.num_samples),
        "--python_bin",
        sys.executable,
    ]

    print("[CTMTF] Sampling Settings:")
    print(f" Data name   : {args.data_name}")
    print(f" Seed        : {args.seed}")
    print(f" Num samples : {args.num_samples}")
    print(f" Y dim       : {y_dim}")
    print(f" Test file   : {test_path}")
    print(f" Model file  : {model_path}")
    print(f" R sampler   : {r_sampler}")

    subprocess.run(cmd, check=True, cwd=str(ROOT))


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
    main(parser.parse_args())

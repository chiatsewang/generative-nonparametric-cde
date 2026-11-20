# scripts/train_ctmtf.py
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

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
    train_path = data_dir / "train.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    y_dim = get_y_dim(train_path)
    if y_dim != 1:
        raise ValueError(
            f"CTMTF supports only univariate Y, but train.npz has y_dim={y_dim}"
        )

    workspace_dir = (
        ROOT / "workspaces" / "ctmtf" / args.data_name / f"seed_{args.seed}"
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)

    r_trainer = ROOT / "condgen_benchmark" / "methods" / "ctmtf" / "trainer.R"
    if not r_trainer.exists():
        raise FileNotFoundError(f"CTMTF R trainer not found: {r_trainer}")

    config = vars(args).copy()
    config["y_dim"] = y_dim
    config["backend"] = "TransformationForest"

    with open(workspace_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    cmd = [
        "Rscript",
        str(r_trainer),
        "--data_name",
        args.data_name,
        "--data_dir",
        str(Path(args.data_dir).resolve()),
        "--seed",
        str(args.seed),
        "--python_bin",
        sys.executable,
    ]

    print("[CTMTF] Experiment Settings:")
    print(f" Data name : {args.data_name}")
    print(f" Seed      : {args.seed}")
    print(f" Y dim     : {y_dim}")
    print(f" Data dir  : {data_dir}")
    print(f" Workspace : {workspace_dir}")
    print(f" R trainer : {r_trainer}")

    start = time.time()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    elapsed = time.time() - start

    with open(
        workspace_dir / "python_train_wrapper_log.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "elapsed_seconds": elapsed,
                "command": cmd,
                "y_dim": y_dim,
                "backend": "TransformationForest",
            },
            f,
            indent=2,
        )

    print(f"[CTMTF] Training completed in {elapsed:.2f} seconds")
    print(f"[CTMTF] Workspace: {workspace_dir}")


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
    main(parser.parse_args())

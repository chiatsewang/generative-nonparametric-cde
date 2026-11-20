import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def dataset_path(data_dir: str, data_name: str, seed: int, split: str) -> Path:
    return Path(data_dir) / data_name / f"seed_{seed}" / f"{split}.npz"


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    r_script = (
        project_root / "condgen_benchmark" / "methods" / "drf" / "trainer.R"
    )

    train_npz = dataset_path(args.data_dir, args.data_name, args.seed, "train")
    valid_npz = dataset_path(args.data_dir, args.data_name, args.seed, "valid")

    if not train_npz.exists():
        raise FileNotFoundError(f"Dataset not found: {train_npz}")
    if not valid_npz.exists():
        raise FileNotFoundError(f"Dataset not found: {valid_npz}")
    if not r_script.exists():
        raise FileNotFoundError(f"R script not found: {r_script}")

    ws = Path("workspaces") / "drf" / args.data_name / f"seed_{args.seed}"
    ws.mkdir(parents=True, exist_ok=True)

    model_path = ws / "model.rds"
    train_data_path = ws / "train_data.npz"
    config_path = ws / "config.yaml"

    cmd = [
        "Rscript",
        str(r_script),
        "--train-npz",
        str(train_npz),
        "--valid-npz",
        str(valid_npz),
        "--model-out",
        str(model_path),
        "--train-data-out",
        str(train_data_path),
        "--seed",
        str(args.seed),
        "--num-trees",
        str(args.num_trees),
        "--min-node-size",
        str(args.min_node_size),
        "--splitting-rule",
        args.splitting_rule,
        "--python-bin",
        sys.executable,
    ]

    print("[DRF] Running R training script:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)

    print("\n[DRF] Saved:")
    print(" -", model_path)
    print(" -", train_data_path)
    print(" -", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRF using trainer.R")

    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="U1")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/synthetic",
        help="Folder holding the generated datasets",
    )
    parser.add_argument("--num-trees", type=int, default=2000)
    parser.add_argument("--min-node-size", type=int, default=15)
    parser.add_argument("--splitting-rule", type=str, default="FourierMMD")

    main(parser.parse_args())

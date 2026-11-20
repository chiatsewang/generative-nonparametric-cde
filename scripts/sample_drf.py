import argparse
import subprocess
import sys
from pathlib import Path


def dataset_path(data_dir: str, data_name: str, seed: int, split: str) -> Path:
    return Path(data_dir) / data_name / f"seed_{seed}" / f"{split}.npz"


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    r_script = (
        project_root / "condgen_benchmark" / "methods" / "drf" / "sampler.R"
    )

    test_npz = dataset_path(args.data_dir, args.data_name, args.seed, "test")
    train_npz = dataset_path(args.data_dir, args.data_name, args.seed, "train")

    if not test_npz.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_npz}")
    if not train_npz.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_npz}")
    if not r_script.exists():
        raise FileNotFoundError(f"R script not found: {r_script}")

    ws = Path("workspaces") / "drf" / args.data_name / f"seed_{args.seed}"
    model_path = ws / "model.rds"
    out_path = ws / "sampled.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cmd = [
        "Rscript",
        str(r_script),
        "--model",
        str(model_path),
        "--train-npz",
        str(train_npz),
        "--test-npz",
        str(test_npz),
        "--out",
        str(out_path),
        "--seed",
        str(args.seed),
        "--num-samples",
        str(args.num_samples),
        "--python-bin",
        sys.executable,
    ]

    print("[DRF] Running R sampling script:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from DRF using sampler.R"
    )

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

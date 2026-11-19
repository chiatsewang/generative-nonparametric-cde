"""Estimate Hall--Yao theta and select bandwidths for one model/seed."""

import argparse
import os

import numpy as np
import yaml

from condgen_benchmark.methods.hy.io import load_split
from condgen_benchmark.methods.hy.tuning import fit_and_tune


def main(args):
    if (args.h is None) != (args.H is None):
        raise ValueError("Specify --h and --H together, or omit both")
    X_train, Y_train = load_split(args.data_name, args.seed, "train")
    if args.h is None:
        X_valid, Y_valid = load_split(args.data_name, args.seed, "valid")
    else:
        X_valid = Y_valid = None

    hy_training_summary = fit_and_tune(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        radius=args.radius,
        seed=args.seed,
        h=args.h,
        H=args.H,
        n_sub_val=args.n_sub_val,
        n_grid_tuning=args.ny_tuning,
        adam_lr=args.adam_lr,
        adam_max_epochs=args.adam_max_epochs,
        adam_patience=args.adam_patience,
    )

    workspace = os.path.join(
        "workspaces", "hy", args.data_name, f"seed_{args.seed}"
    )
    os.makedirs(workspace, exist_ok=True)
    np.savez(
        os.path.join(workspace, "theta.npz"),
        theta=hy_training_summary["theta"],
        h=hy_training_summary["h"],
        H=hy_training_summary["H"],
        t_opt=hy_training_summary["theta_fit_seconds"],
        t_total=hy_training_summary["total_fit_seconds"],
        success=hy_training_summary["optimization_success"],
    )
    tuning_path = os.path.join(workspace, "tuning.csv")
    if hy_training_summary["bandwidth_energy_scores"]:
        np.savetxt(
            tuning_path,
            np.asarray(hy_training_summary["bandwidth_energy_scores"]),
            delimiter=",",
            header="h,H,EnergyScore",
            comments="",
        )
    elif os.path.exists(tuning_path):
        os.remove(tuning_path)
    with open(
        os.path.join(workspace, "config.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            {
                **vars(args),
                "selected_h": hy_training_summary["h"],
                "selected_H": hy_training_summary["H"],
                "mode": hy_training_summary["mode"],
                "optimization_success": hy_training_summary[
                    "optimization_success"
                ],
                "optimization_message": hy_training_summary[
                    "optimization_message"
                ],
                "theta_fit_diagnostics": hy_training_summary.get(
                    "theta_fit_diagnostics", []
                ),
                "n_val_used": hy_training_summary.get("n_val_used"),
                "theta_fit_seconds": float(
                    hy_training_summary["theta_fit_seconds"]
                ),
                "total_fit_seconds": float(
                    hy_training_summary["total_fit_seconds"]
                ),
            },
            f,
            sort_keys=False,
        )
    print(
        f"[HY] saved theta and bandwidths: {workspace}/theta.npz "
        f"h={hy_training_summary['h']:.3g} H={hy_training_summary['H']:.3g} "
        f"theta={hy_training_summary['theta']}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--data-name", type=str, default="U1")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument(
        "--h",
        type=float,
        default=None,
        help="Bandwidth for theta estimation (specify with --H)",
    )
    parser.add_argument(
        "--H",
        type=float,
        default=None,
        help="CDF bandwidth (specify with --h)",
    )
    parser.add_argument(
        "--n-sub-val",
        type=int,
        default=200,
        help="Number of validation predictors used in tuning",
    )
    parser.add_argument(
        "--ny-tuning",
        type=int,
        default=50,
        help="CDF grid points during tuning",
    )
    parser.add_argument("--adam-lr", type=float, default=1e-2)
    parser.add_argument("--adam-max-epochs", type=int, default=300)
    parser.add_argument("--adam-patience", type=int, default=30)
    main(parser.parse_args())

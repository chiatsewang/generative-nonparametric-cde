import argparse
import copy
import os
from pathlib import Path

import flexcode
import numpy as np
import torch
import yaml
from cdetools.cde_loss import cde_loss
from flexcode.regression_models import NN, RandomForest  # NN = k-NN


def load_dataset(mode: str, split: str, seed: int):
    path = os.path.join(
        "datasets", "synthetic", mode, f"seed_{seed}", f"{split}.npz"
    )
    data = np.load(path)
    x = data["X"]  # (n, p)
    y = data["Y"]  # (n,) or (n, 1) or (n, d)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    return x, y


def dump_model_info(
    model,
    path,
    keys=[
        "basis_system",
        "max_basis",
        "best_basis",
        "bump_threshold",
        "sharpen_alpha",
        "y_min",
        "y_max",
    ],
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            if hasattr(model, k):
                v = getattr(model, k)
                f.write(f"{k}: {v}\n")
    print(f"[FlexCode] saved -> {path}")


def main(args):
    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load training and validation data
    X_train, Y_train = load_dataset(args.data_name, "train", args.seed)
    X_valid, Y_valid = load_dataset(args.data_name, "valid", args.seed)

    # show dataset info (FlexCode)
    print(
        "[FlexCode] Training data shape (X, Y):", X_train.shape, Y_train.shape
    )
    print(
        "[FlexCode] Validation data shape (X, Y):",
        X_valid.shape,
        Y_valid.shape,
    )

    # Since FlexCode assumes a one-dimensional response, we use only the first dimension of Y.
    if Y_train.ndim > 1:
        Y_train = Y_train[:, 0]
    if Y_valid.ndim > 1:
        Y_valid = Y_valid[:, 0]

    # select regressor
    if args.reg == "rf":
        RegrClass = RandomForest
        reg_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "criterion": "squared_error",
        }
    elif args.reg == "knn":  # k-nearest neighbor
        RegrClass = NN
        reg_params = {"k": args.k}
    else:
        raise ValueError("--reg must be one of {rf, knn}")

    # build a flex model
    model = flexcode.FlexCodeModel(
        RegrClass,
        max_basis=args.max_basis,
        basis_system=args.basis,
        regression_params=reg_params,
    )

    # fit
    model.fit(X_train, Y_train)

    # model tune
    model_tuned = copy.deepcopy(model)
    model_tuned.tune(
        X_valid,
        Y_valid,
        bump_threshold_grid=args.bgrid,
        sharpen_grid=args.sgrid,
        n_grid=args.ny,
    )

    # validation: predict returns (cde_matrix, grid)
    cde_valid, y_grid = model.predict(X_valid, n_grid=args.ny)
    y_grid = np.asarray(y_grid).reshape(-1)  # (ny,)
    if cde_valid.shape[0] != Y_valid.shape[0]:
        raise RuntimeError(
            f"Mismatch: cde_valid={cde_valid.shape}, Y_valid={Y_valid.shape}"
        )

    cde_valid_tuned, y_grid_tuned = model_tuned.predict(
        X_valid, n_grid=args.ny
    )
    y_grid_tuned = np.asarray(y_grid_tuned).reshape(-1)  # (ny,)
    if cde_valid_tuned.shape[0] != Y_valid.shape[0]:
        raise RuntimeError(
            f"Mismatch: cde_valid_tuned={cde_valid_tuned.shape}, Y_valid={Y_valid.shape}"
        )

    # cde loss: (cde_estimates, y_grid, y_valid) -> (mean, std)
    cde_loss_mean, cde_loss_std = cde_loss(cde_valid, y_grid, Y_valid)
    print("[FlexCode] CDE Loss: %4.2f ± %.2f" % (cde_loss_mean, cde_loss_std))

    cde_loss_mean_tuned, cde_loss_std_tuned = cde_loss(
        cde_valid_tuned, y_grid_tuned, Y_valid
    )
    print(
        "[FlexCode] CDE Loss (tuned): %4.2f ± %.2f"
        % (cde_loss_mean_tuned, cde_loss_std_tuned)
    )

    # save the files in the workspaces folder
    ws = Path("workspaces") / "flex" / args.data_name / f"seed_{args.seed}"
    ws.mkdir(parents=True, exist_ok=True)

    # y_grid is embedded in the model object

    # for model
    model.y_grid = y_grid
    torch.save(model, ws / "model.pth")

    # for tuned model
    model_tuned.y_grid = y_grid_tuned
    torch.save(model_tuned, ws / "model_tuned.pth")

    # save model dump information
    dump_model_info(model, ws / "model_dump.txt")
    dump_model_info(model_tuned, ws / "model_tuned_dump.txt")

    # save metrics
    np.savez(
        ws / "metrics.npz",
        cde_loss_mean=cde_loss_mean,
        cde_loss_std=cde_loss_std,
    )
    np.savez(
        ws / "metrics_tuned.npz",
        cde_loss_mean=cde_loss_mean_tuned,
        cde_loss_std=cde_loss_std_tuned,
    )

    # save config
    with open(ws / "config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    print(
        "[FlexCode] Saved:",
        ws / "model.pth",
        ws / "metrics.npz",
        ws / "model_tuned.pth",
        ws / "metrics_tuned.npz",
        ws / "config.yaml",
        sep="\n - ",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train FlexCode (no extra grid files)"
    )
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--data-name", type=str, default="M1")

    # select basis
    p.add_argument("--basis", type=str, default="cosine")
    p.add_argument("--max-basis", type=int, default=31)

    # select regressor
    p.add_argument("--reg", type=str, default="rf", choices=["rf", "knn"])

    # RF: parameters
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)

    # k-NN: parameters
    p.add_argument("--k", type=int, default=20)

    # num of grids to predict
    p.add_argument("--ny", type=int, default=200)

    # tune arguments: bump_threshold_grid and sharpen_grid
    p.add_argument(
        "--bgrid",
        type=float,
        nargs="+",
        help="array for bump_threshold_grid for tune function",
    )
    p.add_argument(
        "--sgrid",
        type=float,
        nargs="+",
        help="array for sharpend_grid for tune function",
    )

    args = p.parse_args()
    main(args)

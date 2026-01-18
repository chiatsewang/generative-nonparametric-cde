import argparse
import os

import numpy as np
import ot  # sliced Wasserstein (!pip install POT)
from scipy.stats import wasserstein_distance

from condgen_benchmark.data.synthetic import get_synthetic_data_class


def evaluate(Ys_p_hat, mean_p_true, std_p_true, Ys_p_true):
    """
    Compare empirical conditional mean, std, and Wasserstein distance
    with analytical ground truth.

    Parameters
    ----------
    Ys_p_hat : np.ndarray
        Shape [N, S, d]
        Yi1,...,YiS ~ p_hat(...|Xi) where Xi = (Xi1,...,Xiq)^T (i=1~N)
    mean_p_true : np.ndarray
        Shape [N]
        True E[Y | X]
    std_p_true : np.ndarray
        Shape [N]
        True std[Y | X]
    Ys_p_true : np.ndarray
        Shape [N, S, d]
        Yi1,...,YiS ~ p_true(...|Xi) where Xi = (Xi1,...,Xiq)^T (i=1~N)

    Returns
    -------
    metrics : dict
        Dictionary with MSE of mean/std and Wasserstein-1
    """

    N, S, d = Ys_p_hat.shape

    mean_p_hat = Ys_p_hat.mean(axis=1).squeeze()
    std_p_hat = Ys_p_hat.std(axis=1).squeeze()

    mse_mean = np.mean((mean_p_hat - mean_p_true) ** 2)
    mse_std = np.mean((std_p_hat - std_p_true) ** 2)

    # 　median sequared error
    medse_mean = np.median((mean_p_hat - mean_p_true) ** 2)
    medse_std = np.median((std_p_hat - std_p_true) ** 2)

    if d == 1:
        # Wasserstein distance
        w_distances = np.fromiter(
            (
                wasserstein_distance(Ys_p_hat[i, :, 0], Ys_p_true[i, :, 0])
                for i in range(N)
            ),
            dtype=float,
            count=N,
        )
    else:
        # Sliced Wasserstein distance based on 100 projections
        w_distances = np.fromiter(
            (
                ot.sliced.sliced_wasserstein_distance(
                    Ys_p_hat[i],  # [S, d]
                    Ys_p_true[i],  # [S, d]
                    n_projections=100,
                    p=1,
                    seed=42,
                )
                for i in range(N)
            ),
            dtype=float,
            count=N,
        )

    w_mean = np.mean(w_distances)

    return {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "medse_mean": medse_mean,
        "medse_std": medse_std,
        "mean_p_true": mean_p_true,
        "mean_p_hat": mean_p_hat,
        "std_p_true": std_p_true,
        "std_p_hat": std_p_hat,
        "w_mean": w_mean,
        "w_distances": w_distances,
        "multivariate_response": bool(
            d > 1
        ),  # if True, then w_distances are sliced w-distances
        "response_dim": int(d),
    }


def main(args):
    # Load predicted samples and conditioning variable
    result_path = os.path.join(
        "workspaces",
        args.algorithm,
        args.data_name,
        f"seed_{args.seed}",
        "sampled.npz",
    )
    data = np.load(result_path)
    Ys_p_hat = data["Y_hat"]  # [N, S, d]
    X = data["X"]  # [N, q]

    # Compute analytical ground truth
    data_class = get_synthetic_data_class(data_name=args.data_name)
    model = data_class(seed=args.seed)
    Ys_p_true, _ = model.generate(n_samples=Ys_p_hat.shape[1], X=X)
    Ys_p_true = Ys_p_true.reshape(Ys_p_hat.shape)
    mean_p_true, std_p_true = model.mean_std(X)

    # Evaluate metrics
    metrics = evaluate(Ys_p_hat, mean_p_true, std_p_true, Ys_p_true)
    prefix = f"{args.algorithm.upper()}-{args.data_name}"
    print(
        f"[{prefix}] response_dim={metrics['response_dim']}  multivariate={metrics['multivariate_response']}"
    )
    print(f"[{prefix}] MSE (mean): {metrics['mse_mean']:.6f}")
    print(f"[{prefix}] MSE (std):  {metrics['mse_std']:.6f}")
    print(f"[{prefix}] MedSE (mean): {metrics['medse_mean']:.6f}")
    print(f"[{prefix}] MedSE (std): {metrics['medse_std']:.6f}")
    print(f"[{prefix}] W dist (avg.):  {metrics['w_mean']:.6f}")

    # Save evaluation results
    save_path = os.path.join(
        "workspaces",
        args.algorithm,
        args.data_name,
        f"seed_{args.seed}",
        "eval.npz",
    )
    np.savez_compressed(
        save_path,
        mse_mean=metrics["mse_mean"],
        mse_std=metrics["mse_std"],
        medse_mean=metrics["medse_mean"],
        medse_std=metrics["medse_std"],
        mean_p_true=metrics["mean_p_true"],
        std_p_true=metrics["std_p_true"],
        mean_p_hat=metrics["mean_p_hat"],
        std_p_hat=metrics["std_p_hat"],
        w_mean=metrics["w_mean"],
        w_distances=metrics["w_distances"],
        multivariate_response=metrics["multivariate_response"],
        response_dim=metrics["response_dim"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate conditional generative model outputs"
    )
    parser.add_argument("--seed", type=int, default=230, help="Dataset seed")
    parser.add_argument(
        "--data-name", type=str, default="M1", help="Synthetic data model"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ddpm",
        help="Algorithm name (e.g., ddpm, hyalg)",
    )
    args = parser.parse_args()
    main(args)

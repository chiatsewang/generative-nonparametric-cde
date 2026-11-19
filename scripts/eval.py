import argparse
import os

import numpy as np
import ot  # sliced Wasserstein (!pip install POT)
from scipy.stats import wasserstein_distance

from condgen_benchmark.data.synthetic import get_synthetic_data_class
from condgen_benchmark.utils.eval_energy_score import expected_energy_score


def evaluate(Ys_p_hat, mu_true, sigma_true, Ys_p_true):
    """
    Compare empirical conditional mean, standard deviation, Wasserstein
    distance, and energy score with analytical ground truth.

    Parameters
    ----------
    Ys_p_hat : np.ndarray
        Shape [N, S_hat, d]
        Y_p_hat[i, s, :] is the s-th sample drawn from the fitted
        conditional distribution p_hat(. | X_i).
    mu_true : np.ndarray
        Shape [N]
        True E[Y | X]
    sigma_true : np.ndarray
        Shape [N]
        True std[Y | X]
    Ys_p_true : np.ndarray
        Shape [N, S_true, d]
        Y_p_true[i, t, :] is the t-th sample drawn from the true
        conditional distribution p(. | X_i).

    Returns
    -------
    metrics : dict
        Dictionary with mu/sigma MSE, Wasserstein-1, and energy score
    """

    N, S_hat, d = Ys_p_hat.shape

    # Ys_p_hat has shape (N, S_hat, d):
    #   N     : number of test covariates X_i
    #   S_hat : number of samples drawn from the fitted conditional distribution
    #           p_hat(. | X_i)
    #   d     : response dimension
    #
    # For each test point X_i, estimate the conditional mean and conditional
    # standard deviation of the fitted distribution by empirical summaries over
    # the generated samples. Thus, mu_hat and sigma_hat contain N pointwise
    # estimates: mu_hat[i] ≈ E_{p_hat}[Y | X_i] and
    # sigma_hat[i] ≈ SD_{p_hat}[Y | X_i].
    mu_hat = Ys_p_hat.mean(axis=1).squeeze()
    sigma_hat = Ys_p_hat.std(axis=1).squeeze()

    # Compare pointwise fitted summaries with the true conditional summaries.
    # The final MSEs are averaged over test covariates X_1, ..., X_N.
    mu_mse = np.mean((mu_hat - mu_true) ** 2)
    sigma_mse = np.mean((sigma_hat - sigma_true) ** 2)

    # Median squared errors over test covariates X_1, ..., X_N.
    mu_medse = np.median((mu_hat - mu_true) ** 2)
    sigma_medse = np.median((sigma_hat - sigma_true) ** 2)

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

    # Evaluation-only Energy Score: computed on the selected GPU, without
    # splitting test covariates. The U-statistic excludes fitted self-pairs.
    escore_mean = expected_energy_score(Ys_p_hat, Ys_p_true)

    return {
        "mu_mse": mu_mse,
        "sigma_mse": sigma_mse,
        "mu_medse": mu_medse,
        "sigma_medse": sigma_medse,
        "mu_true": mu_true,
        "mu_hat": mu_hat,
        "sigma_true": sigma_true,
        "sigma_hat": sigma_hat,
        "w_mean": w_mean,
        "escore_mean": escore_mean,
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
    Ys_p_hat = data["Y_hat"]  # [N, S_hat, d], samples from p_hat(. | X_i)
    X = data["X"]  # [N, q]

    # Compute analytical ground truth
    data_class = get_synthetic_data_class(data_name=args.data_name)
    model = data_class(seed=args.seed)
    Ys_p_true, _ = model.generate(n_samples=Ys_p_hat.shape[1], X=X)
    Ys_p_true = Ys_p_true.reshape(
        Ys_p_hat.shape
    )  # [N, S_true, d], true samples
    mu_true, sigma_true = model.mean_std(X)

    # Read the method-independent oracle saved at data-generation time.
    # It was computed using the SAME test X_i but fresh independent true Y.
    test_path = os.path.join(
        "datasets",
        "synthetic",
        args.data_name,
        f"seed_{args.seed}",
        "test.npz",
    )
    with np.load(test_path) as test_data:
        if "ES_oracle" not in test_data:
            raise ValueError(
                f"Missing ES_oracle in {test_path}; regenerate the dataset."
            )
        ES_oracle = float(test_data["ES_oracle"])
    if ES_oracle <= 0:
        raise ValueError("Oracle Energy Score must be positive for rEEScore")

    # Evaluate metrics
    metrics = evaluate(Ys_p_hat, mu_true, sigma_true, Ys_p_true)
    reescore = (metrics["escore_mean"] - ES_oracle) / ES_oracle
    prefix = f"{args.algorithm.upper()}-{args.data_name}"
    print(
        f"[{prefix}] response_dim={metrics['response_dim']}  multivariate={metrics['multivariate_response']}"
    )
    print(f"[{prefix}] Mu MSE:     {metrics['mu_mse']:.6f}")
    print(f"[{prefix}] Sigma MSE:  {metrics['sigma_mse']:.6f}")
    print(f"[{prefix}] Mu MedSE:   {metrics['mu_medse']:.6f}")
    print(f"[{prefix}] Sigma MedSE:{metrics['sigma_medse']:.6f}")
    print(f"[{prefix}] W dist:     {metrics['w_mean']:.6f}")
    print(f"[{prefix}] EScore:     {metrics['escore_mean']:.6f}")
    print(f"[{prefix}] Oracle:     {ES_oracle:.6f}")
    print(f"[{prefix}] rEEScore:   {reescore:.6f}")

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
        mu_mse=metrics["mu_mse"],
        sigma_mse=metrics["sigma_mse"],
        mu_medse=metrics["mu_medse"],
        sigma_medse=metrics["sigma_medse"],
        mu_true=metrics["mu_true"],
        sigma_true=metrics["sigma_true"],
        mu_hat=metrics["mu_hat"],
        sigma_hat=metrics["sigma_hat"],
        w_mean=metrics["w_mean"],
        escore_mean=metrics["escore_mean"],
        ES_oracle=ES_oracle,
        reescore=reescore,
        multivariate_response=metrics["multivariate_response"],
        response_dim=metrics["response_dim"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate conditional generative model outputs"
    )
    parser.add_argument("--seed", type=int, default=230, help="Dataset seed")
    parser.add_argument(
        "--data-name", type=str, default="U1", help="Synthetic data model"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ddpm",
        help="Algorithm name (e.g., ddpm, hy)",
    )
    args = parser.parse_args()
    main(args)

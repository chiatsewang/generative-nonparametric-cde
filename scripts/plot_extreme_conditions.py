import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from condgen_benchmark.data.synthetic import get_synthetic_data_class


def collect_metric_conditions(mode, workspace_dir="workspaces/ddpm"):
    mode_dir = os.path.join(workspace_dir, mode)
    seed_dirs = sorted(glob.glob(os.path.join(mode_dir, "seed_*")))

    metric_records = []

    for seed_dir in seed_dirs:
        seed = int(seed_dir.split("_")[-1])
        eval_path = os.path.join(seed_dir, "eval.npz")
        sample_path = os.path.join(seed_dir, "sampled.npz")

        if not (os.path.exists(eval_path) and os.path.exists(sample_path)):
            continue

        eval_data = np.load(eval_path)
        sample_data = np.load(sample_path)

        emp_mean = eval_data["emp_mean"]
        true_mean = eval_data["true_mean"]
        emp_std = eval_data["emp_std"]
        true_std = eval_data["true_std"]
        w1 = eval_data["w1_distances"]
        X = sample_data["X"]  # Predictor variables
        Y_hat = sample_data["Y_hat"]  # Predicted response variables

        mean_error = np.abs(emp_mean - true_mean)
        std_error = np.abs(emp_std - true_std)

        for i in range(len(X)):
            metric_records.append(
                {
                    "seed": seed,
                    "idx": i,
                    "mean_error": mean_error[i],
                    "std_error": std_error[i],
                    "w1": w1[i],
                    "X": X[i],  # Predictor
                    "Y_hat": Y_hat[i],  # Predicted response
                    "true_mean": true_mean[i],
                    "emp_mean": emp_mean[i],
                }
            )

    return metric_records


def select_best_middle_worst(records, metric, percentile=0.1):
    sorted_records = sorted(records, key=lambda r: r[metric])
    n = len(sorted_records)
    sel_n = int(n * percentile)
    return [
        sorted_records[-sel_n],
        sorted_records[n // 2],
        sorted_records[sel_n],
    ]


def plot_conditions(
    mode, metric_records, save_dir="plots/extreme-results", num_samples=10000
):
    data_class = get_synthetic_data_class(data_name=mode)()

    selected = []
    metric_labels = [r"$\Delta$ mean", r"$\Delta$ std", "W1"]
    for metric in ["mean_error", "std_error", "w1"]:
        selected += select_best_middle_worst(metric_records, metric)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Expand ylim over all predicted and true samples
    y_all = []
    for rec in selected:
        y_all.append(rec["Y_hat"].flatten())
        true_Y_hat, _ = data_class.generate(num_samples, X=rec["X"])
        y_all.append(true_Y_hat.flatten())
    y_all = np.concatenate(y_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    ylim = (y_min, y_max)

    for i, rec in enumerate(selected):
        ax = axes[i]
        X = rec["X"]
        Y_hat = rec["Y_hat"].flatten()
        true_Y_hat, _ = data_class.generate(num_samples, X=X)
        true_Y_hat = true_Y_hat.flatten()

        # Histogram
        ax.hist(true_Y_hat, bins=50, density=True, alpha=0.7, color="skyblue")
        ax.hist(Y_hat, bins=50, density=True, alpha=0.5, color="orange")

        # KDE curves
        kde_pred = gaussian_kde(Y_hat)
        kde_true = gaussian_kde(true_Y_hat)
        y_grid = np.linspace(ylim[0], ylim[1], 300)
        ax.plot(y_grid, kde_pred(y_grid), color="orange", linewidth=1.5)
        ax.plot(y_grid, kde_true(y_grid), color="skyblue", linewidth=1.5)

        # Mean lines
        ax.axvline(rec["true_mean"], color="red", linestyle="--", linewidth=2)
        ax.axvline(rec["emp_mean"], color="green", linestyle="-.", linewidth=2)

        ax.set_xlim(ylim)
        ax.set_title(
            f"Seed {rec['seed']} | Ex {rec['idx']}\n"
            f"{metric_labels[0]}={rec['mean_error']:.4f}, "
            f"{metric_labels[1]}={rec['std_error']:.4f}, W1={rec['w1']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Y")
        ax.set_ylabel("Density")

    # Add unified legend below suptitle
    handles = [
        plt.Line2D([], [], color="skyblue", lw=6, label="True Histogram"),
        plt.Line2D([], [], color="orange", lw=6, label="Predicted Histogram"),
        plt.Line2D([], [], color="skyblue", lw=1.5, label="True KDE"),
        plt.Line2D([], [], color="orange", lw=1.5, label="Predicted KDE"),
        plt.Line2D(
            [], [], color="red", linestyle="--", lw=2, label="True Mean"
        ),
        plt.Line2D(
            [], [], color="green", linestyle="-.", lw=2, label="Empirical Mean"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=6,
        fontsize=10,
        frameon=False,
    )

    # Suptitle
    plt.suptitle(
        f"Extreme Conditions — Mode: {mode}",
        fontsize=18,
        fontweight="bold",
    )

    # Adjust layout to make space for legend + title
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    save_path = os.path.join(save_dir, f"{mode}-extreme-results.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        help="Synthetic data model (e.g., M1, M2, ZhouM1)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="workspaces/ddpm",
        help="Base experiment dir",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="plots/extreme-results",
        help="Directory to save output figure",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of ground truth samples to generate",
    )
    args = parser.parse_args()

    records = collect_metric_conditions(args.data_name, args.workspace_dir)
    plot_conditions(args.data_name, records, args.save_dir, args.num_samples)

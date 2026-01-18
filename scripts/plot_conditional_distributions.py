import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from condgen_benchmark.data.synthetic import get_synthetic_data_class


def plot_conditional_examples(
    mode, n_examples=9, n_samples=10000, seed=230, save_dir=None
):
    np.random.seed(seed)
    data_class = get_synthetic_data_class(data_name=mode)(seed=seed)

    # Sample X and corresponding conditional samples
    X = np.random.randn(n_examples, data_class.predictor_dim)
    Y, _ = data_class.generate(n_samples, X=X)
    Y = Y.reshape(n_examples, n_samples, -1)
    true_means, true_stds = data_class.mean_std(X)

    # Compute shared y-axis range with padding
    y_min, y_max = np.min(Y), np.max(Y)
    y_pad = 0.05 * (y_max - y_min)
    ylim_data = (y_min - y_pad, y_max + y_pad)

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_examples):
        ax = axes[i]
        y_vals = Y[i].squeeze()

        # Plot histogram
        ax.hist(
            y_vals,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            label="Histogram",
        )

        # Plot KDE
        kde = gaussian_kde(y_vals)
        y_grid = np.linspace(*ylim_data, 300)
        ax.plot(y_grid, kde(y_grid), color="blue", lw=1.5, label="KDE")

        # Plot true mean line
        ax.axvline(
            true_means[i],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True Mean",
        )

        # Annotate true mean value
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)
        offset_y = 0.05 * (ylim[1] - ylim[0])
        offset_x = 0.1 * (ylim_data[1] - ylim_data[0])
        ax.text(
            true_means[i] + offset_x,
            ylim[1] - offset_y,
            f"({true_means[i]:.3f})",
            color="red",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

        ax.set_xlim(ylim_data)
        ax.set_title(
            f"Example {i + 1}  "
            f"($\\mu$={true_means[i]:.3f}, $\\sigma$={true_stds[i]:.3f})",
            fontsize=12,
        )
        ax.set_xlabel("y")
        ax.set_ylabel("Density")

    # Hide unused subplots
    for j in range(n_examples, len(axes)):
        axes[j].axis("off")

    # Add suptitle and shared legend
    fig.suptitle(
        f"Conditional Distribution Examples — Mode: {mode}",
        fontsize=18,
        fontweight="bold",
    )
    handles = [
        plt.Line2D([], [], color="skyblue", lw=6, label="Histogram"),
        plt.Line2D([], [], color="blue", lw=1.5, label="KDE"),
        plt.Line2D(
            [], [], color="red", linestyle="--", lw=2, label="True Mean"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{mode}-conditional-distributions.png")
        plt.savefig(fname)
        print(f"Saved to {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        help="Synthetic data model (e.g., M1, M2, ZhouT1)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=9,
        help="Number of X examples to visualize",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of Y samples per X",
    )
    parser.add_argument("--seed", type=int, default=230, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="plots/synthetic-examples",
        help="Optional save directory for plot",
    )
    args = parser.parse_args()

    plot_conditional_examples(
        args.data_name,
        args.n_examples,
        args.n_samples,
        args.seed,
        args.save_dir,
    )

"""Energy-score calculations for benchmark evaluation only.

Called by:
    - scripts/eval.py
    - scripts/generate_synthetic.py

SRM training has its own loss in methods/srm/loss.py. This module does not
participate in training. All test points are processed together on the
selected device, without splitting the pairwise-distance calculations.
"""

import numpy as np
import torch


def _device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _samples_tensor(samples, device):
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("Expected conditional samples with shape (N, S, q)")
    if samples.shape[1] < 2:
        raise ValueError(
            "Energy-score calculation requires at least two samples"
        )
    return torch.as_tensor(samples, dtype=torch.float32, device=device)


def _mean_off_diagonal_distance(samples):
    """Mean Euclidean distance over distinct sample pairs, averaged over X."""
    n_samples = samples.shape[1]
    distances = torch.cdist(samples, samples)
    # cdist's matrix-multiplication implementation can give tiny nonzero
    # diagonal entries due to floating-point roundoff: exclude them exactly.
    distances.diagonal(dim1=1, dim2=2).zero_()
    return distances.sum(dim=(1, 2)).mean() / (n_samples * (n_samples - 1))


@torch.no_grad()
def oracle_energy_score(true_samples, device="auto"):
    """Oracle EScore = 1/2 E[|Y-Y'| | X], averaged over test predictors.

    true_samples: (n_test, n_es_oracle_mc, response_dim), independent draws
    at the X values in test.npz (not the stored observed test responses).
    """
    selected_device = _device(device)
    true_samples = _samples_tensor(true_samples, selected_device)
    return float((0.5 * _mean_off_diagonal_distance(true_samples)).item())


@torch.no_grad()
def expected_energy_score(predicted_samples, true_samples, device="auto"):
    """Expected EScore using all test points and the U-statistic correction.

    First term: all fitted/true cross pairs; second term: distinct fitted pairs.
    The two sample counts may differ, but the predictor and response shapes
    must match.
    """
    selected_device = _device(device)
    predicted_samples = _samples_tensor(predicted_samples, selected_device)
    true_samples = _samples_tensor(true_samples, selected_device)
    if (
        predicted_samples.shape[0] != true_samples.shape[0]
        or predicted_samples.shape[2] != true_samples.shape[2]
    ):
        raise ValueError("Predicted/true samples need the same N and q")

    # Compute the cross term first and release its (N, S_hat, S_true) matrix
    # before constructing the fitted within-pair distance matrix.
    cross_distance = torch.cdist(predicted_samples, true_samples)
    cross_mean = cross_distance.mean()
    del cross_distance

    within_mean = _mean_off_diagonal_distance(predicted_samples)
    return float((cross_mean - 0.5 * within_mean).item())

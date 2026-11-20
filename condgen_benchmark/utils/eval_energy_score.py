"""Energy-score calculations for benchmark evaluation only.

Called by:
    - scripts/eval.py
    - scripts/generate_synthetic.py

SRM training has its own loss in methods/srm/loss.py. This module does not
participate in training. Test points are processed in chunks on the
selected device, so peak memory depends on the chunk size and not on the
number of test points.
"""

import numpy as np
import torch


def _device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


DEFAULT_CHUNK_SIZE = 32


def _check_chunk_size(chunk_size):
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    return int(chunk_size)


def _samples_tensor(samples, device):
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("Expected conditional samples with shape (N, S, q)")
    if samples.shape[1] < 2:
        raise ValueError(
            "Energy-score calculation requires at least two samples"
        )
    return torch.as_tensor(samples, dtype=torch.float32, device=device)


def _mean_off_diagonal_distance(samples, chunk_size):
    """Mean Euclidean distance over distinct sample pairs, averaged over X."""
    n_points, n_samples = samples.shape[:2]
    total = 0.0
    for start in range(0, n_points, chunk_size):
        block = samples[start : start + chunk_size]
        distances = torch.cdist(block, block)
        # cdist's matrix-multiplication implementation can give tiny nonzero
        # diagonal entries due to floating-point roundoff: exclude them.
        distances.diagonal(dim1=1, dim2=2).zero_()
        total += distances.sum(dtype=torch.float64).item()
        del distances
    return total / (n_points * n_samples * (n_samples - 1))


@torch.no_grad()
def oracle_energy_score(
    true_samples, device="auto", chunk_size=DEFAULT_CHUNK_SIZE
):
    """Oracle EScore = 1/2 E[|Y-Y'| | X], averaged over test predictors.

    true_samples: (n_test, n_es_oracle_mc, response_dim), independent draws
    at the X values in test.npz (not the stored observed test responses).
    chunk_size is the number of test points per pairwise-distance block.
    """
    chunk_size = _check_chunk_size(chunk_size)
    selected_device = _device(device)
    true_samples = _samples_tensor(true_samples, selected_device)
    return 0.5 * _mean_off_diagonal_distance(true_samples, chunk_size)


@torch.no_grad()
def expected_energy_score(
    predicted_samples,
    true_samples,
    device="auto",
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Expected EScore using all test points and the U-statistic correction.

    First term: all fitted/true cross pairs; second term: distinct fitted pairs.
    The two sample counts may differ, but the predictor and response shapes
    must match. chunk_size is the number of test points per distance block.
    """
    chunk_size = _check_chunk_size(chunk_size)
    selected_device = _device(device)
    predicted_samples = _samples_tensor(predicted_samples, selected_device)
    true_samples = _samples_tensor(true_samples, selected_device)
    if (
        predicted_samples.shape[0] != true_samples.shape[0]
        or predicted_samples.shape[2] != true_samples.shape[2]
    ):
        raise ValueError("Predicted/true samples need the same N and q")

    n_points = predicted_samples.shape[0]
    cross_total = 0.0
    for start in range(0, n_points, chunk_size):
        cross_distance = torch.cdist(
            predicted_samples[start : start + chunk_size],
            true_samples[start : start + chunk_size],
        )
        cross_total += cross_distance.sum(dtype=torch.float64).item()
        del cross_distance
    cross_mean = cross_total / (
        n_points * predicted_samples.shape[1] * true_samples.shape[1]
    )

    within_mean = _mean_off_diagonal_distance(predicted_samples, chunk_size)
    return float(cross_mean - 0.5 * within_mean)

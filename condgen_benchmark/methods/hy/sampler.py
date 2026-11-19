"""Shared conditional sampler for HY tuning and final test sampling.

For each query X, the local-linear weighted CDF is computed in O(n + ny)
following a once-only sort of the training responses, rather than the
legacy O(n * ny) Python loop. The original weights are retained.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def energy_score_from_quantile(probs, responses, observed):
    """Exact 1D Energy Score of the piecewise-linear sampling quantile.

    Inverse interpolation clips U~Uniform(0,1) to [probs[0], probs[-1]].
    Consequently the resulting distribution can have atoms at its endpoints.
    These endpoint masses are included here, with no Monte Carlo sampling.
    """
    k = len(probs)
    p_first, p_last = probs[0], probs[k - 1]
    q_first, q_last = responses[0], responses[k - 1]

    # E|Z-y| for the flat quantile tails.
    a = p_first * abs(q_first - observed) + (1.0 - p_last) * abs(
        q_last - observed
    )
    # (1/2) E|Z-Z'| = integral_0^1 (2u-1) Q(u) du.
    b = q_first * (p_first * p_first - p_first)
    b += q_last * (p_last - p_last * p_last)

    for j in range(k - 1):
        p0, p1 = probs[j], probs[j + 1]
        q0, q1 = responses[j], responses[j + 1]
        delta_p = p1 - p0
        delta_q = q1 - q0
        if observed <= q0:
            a += delta_p * ((q0 + q1) * 0.5 - observed)
        elif observed >= q1:
            a += delta_p * (observed - (q0 + q1) * 0.5)
        elif delta_q > 0.0:
            a += delta_p * (
                ((observed - q0) ** 2 + (q1 - observed) ** 2) / (2.0 * delta_q)
            )
        else:
            a += delta_p * abs(q0 - observed)
        b += delta_p * (
            (p0 + p1 - 1.0) * (q0 + q1) * 0.5 + delta_p * delta_q / 6.0
        )
    return a - b


def response_grid(Y_train, n_grid):
    """Grid spanning the response Y (never predictor X)."""
    Y_train = np.asarray(Y_train).reshape(-1)
    if n_grid < 2:
        raise ValueError("ny must be >= 2")
    return np.linspace(Y_train.min(), Y_train.max(), n_grid)


class PreparedConditionalSampler:
    """Precompute the H-independent steps for one theta and query set.

    Construct once for each estimated theta. Reuse sample() across all
    candidate H values. The final-sampling convenience function below uses
    this same sample() implementation.
    """

    def __init__(self, X_train, Y_train, X_query, theta, y_grid):
        X_train = np.asarray(X_train, dtype=np.float64)
        Y_train = np.asarray(Y_train, dtype=np.float64).reshape(-1)
        X_query = np.asarray(X_query, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        y_grid = np.asarray(y_grid, dtype=np.float64)
        if X_train.ndim != 2 or X_query.ndim != 2:
            raise ValueError("X_train and X_query must have shape (n, p)")
        if (
            X_train.shape[0] != Y_train.size
            or X_train.shape[1] != X_query.shape[1]
            or theta.size != X_train.shape[1]
        ):
            raise ValueError("X, Y, theta dimensions are incompatible")
        if len(Y_train) == 0:
            raise ValueError("Training data must not be empty")
        if y_grid.ndim != 1 or len(y_grid) < 2 or np.any(np.diff(y_grid) < 0):
            raise ValueError(
                "y_grid must be a sorted 1D array of at least two points"
            )
        theta_norm = np.linalg.norm(theta)
        if not np.isfinite(theta_norm) or theta_norm == 0:
            raise ValueError("theta must have positive finite norm")
        theta = theta / theta_norm

        self.train_projections = X_train @ theta
        self.query_projections = X_query @ theta
        self.mean_training_response = Y_train.mean()
        self.response_sort_indices = np.argsort(Y_train, kind="stable")
        y_sorted = Y_train[self.response_sort_indices]
        self.response_grid_right_indices = np.searchsorted(
            y_sorted, y_grid, side="right"
        )
        self.y_grid = y_grid
        self.n = Y_train.size
        self.n_query = X_query.shape[0]

    def _quantile_knots(self, query_index, H, tol):
        """Knots of the same inverse CDF used for both tuning and sampling.

        Returns None when local-linear weights cannot be normalized, matching
        the sampling fallback to the overall response mean.
        """
        u = (self.query_projections[query_index] - self.train_projections) / H
        K = 0.75 * (1 - u * u) * (np.abs(u) <= 1)
        K = np.maximum(K, 0.0)
        t1 = np.dot(K, u) / (self.n * H)
        t2 = np.dot(K, u * u) / (self.n * H)
        weights = K * (t2 - u * t1)
        denom = weights.sum()
        if not np.isfinite(denom) or abs(denom) < tol:
            return None
        prefix = np.empty(self.n + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(weights[self.response_sort_indices], out=prefix[1:])
        cdf = prefix[self.response_grid_right_indices] / (denom + tol)
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        probs, positions = np.unique(cdf, return_index=True)
        return probs, self.y_grid[positions]

    def sample(self, H, num_samples, rng=None, uniforms=None, tol=1e-12):
        """Sample with precomputed projections and the shared inverse-CDF knots."""
        if H <= 0 or num_samples < 1:
            raise ValueError("H must be positive and num_samples >= 1")
        if rng is None:
            rng = np.random.default_rng()
        if uniforms is not None:
            uniforms = np.asarray(uniforms, dtype=np.float64)
            if uniforms.shape != (self.n_query, num_samples):
                raise ValueError("uniforms must be [n_query, num_samples]")

        sampled_responses = np.empty(
            (self.n_query, num_samples), dtype=np.float64
        )
        for i in range(self.n_query):
            knots = self._quantile_knots(i, H, tol)
            if knots is None:
                sampled_responses[i] = self.mean_training_response
                continue
            probs, responses = knots
            draw_u = (
                uniforms[i]
                if uniforms is not None
                else rng.random(num_samples)
            )
            sampled_responses[i] = np.interp(
                np.clip(draw_u, probs[0], probs[-1]),
                probs,
                responses,
            )
        return sampled_responses

    def energy_score(self, H, observed, tol=1e-12):
        """Mean exact Energy Score (1D CRPS), without random draws.

        Scores the *same* distribution produced by sample(): the CDF after
        monotonicity correction, uniqueness filtering, and inverse linear
        interpolation, including any probability at the support endpoints.
        """
        if not np.isfinite(H) or H <= 0:
            raise ValueError("H must be positive and finite")
        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        if observed.size != self.n_query or not np.isfinite(observed).all():
            raise ValueError("observed must contain one finite Y per query X")
        if self.n_query == 0:
            raise ValueError("Energy Score requires at least one query")
        total_energy_score = 0.0
        for i in range(self.n_query):
            knots = self._quantile_knots(i, H, tol)
            if knots is None:
                total_energy_score += abs(
                    self.mean_training_response - observed[i]
                )
            else:
                probs, responses = knots
                total_energy_score += energy_score_from_quantile(
                    probs, responses, observed[i]
                )
        return total_energy_score / self.n_query


def sample_conditional(
    X_train,
    Y_train,
    X_query,
    theta,
    H,
    num_samples,
    y_grid,
    rng=None,
    uniforms=None,
    tol=1e-12,
):
    """Convenience entry point for final sampling; shares tuning's core."""
    prepared_sampler = PreparedConditionalSampler(
        X_train, Y_train, X_query, theta, y_grid
    )
    return prepared_sampler.sample(
        H, num_samples, rng=rng, uniforms=uniforms, tol=tol
    )

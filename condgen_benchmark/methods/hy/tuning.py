"""Per-seed (h, H) selection by the 1D Energy Score."""

import time

import numpy as np

from condgen_benchmark.methods.hy.fit import fit_thetas
from condgen_benchmark.methods.hy.loss import HallYaoLoss
from condgen_benchmark.methods.hy.sampler import (
    PreparedConditionalSampler,
    response_grid,
)


def bandwidth_grid():
    """21 (h,H) pairs: h=.1,.3,...,1.1; H=h+.1,h+.3,...,<=1.2."""
    bandwidth_pairs = []
    for h_i in range(1, 12, 2):
        for H_i in range(h_i + 1, 13, 2):
            bandwidth_pairs.append((h_i / 10.0, H_i / 10.0))
    assert len(bandwidth_pairs) == 21
    return bandwidth_pairs


def _theta_fit_message(theta_diagnostic):
    """Summarize the selected h optimization for config.yaml."""
    return (
        f"Adam epochs={theta_diagnostic['epochs']} "
        f"best_S_theta={theta_diagnostic['best_S_theta']:.8g}"
    )


def fit_and_tune(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    radius,
    seed,
    h=None,
    H=None,
    n_sub_val=200,
    n_grid_tuning=50,
    adam_lr=1e-2,
    adam_max_epochs=300,
    adam_patience=30,
):
    """Tune per seed; manual (h,H) skips validation scoring entirely."""
    fit_and_tune_start = time.perf_counter()
    if (h is None) != (H is None):
        raise ValueError("Specify both --h and --H, or neither")
    if n_grid_tuning < 2 or n_sub_val < 1:
        raise ValueError("Invalid tuning sizes: need grid>=2, subset>=1")
    theta_objective = HallYaoLoss(X_train, Y_train, radius=radius)
    if h is not None:
        if h <= 0 or H <= 0:
            raise ValueError("h and H must be positive")
        fitted_theta_for_each_h, theta_fit_seconds, theta_fit_diagnostics = (
            fit_thetas(
                theta_objective,
                [h],
                seed,
                lr=adam_lr,
                max_epochs=adam_max_epochs,
                patience=adam_patience,
            )
        )
        return {
            "theta": fitted_theta_for_each_h[float(h)],
            "h": float(h),
            "H": float(H),
            "theta_fit_seconds": theta_fit_seconds,
            "optimization_success": True,
            "optimization_message": _theta_fit_message(
                theta_fit_diagnostics[0]
            ),
            "mode": "manual",
            "bandwidth_energy_scores": [],
            "total_fit_seconds": time.perf_counter() - fit_and_tune_start,
            "theta_fit_diagnostics": theta_fit_diagnostics,
        }

    X_valid = np.asarray(X_valid)
    Y_valid = np.asarray(Y_valid).reshape(-1)
    if len(X_valid) != len(Y_valid) or len(X_valid) == 0:
        raise ValueError(
            "Validation X/Y must have matching, positive row counts"
        )
    validation_rng = np.random.default_rng(seed)
    n_valid_used = min(n_sub_val, len(X_valid))
    validation_indices = validation_rng.choice(
        len(X_valid), n_valid_used, replace=False
    )
    X_valid_subset = X_valid[validation_indices]
    Y_valid_subset = Y_valid[validation_indices]
    y_grid = response_grid(Y_train, n_grid_tuning)
    # Estimate six theta vectors in a single batched Torch training run.
    theta_bandwidth_candidates = list(
        dict.fromkeys(h_candidate for h_candidate, _ in bandwidth_grid())
    )
    fitted_theta_for_each_h, theta_fit_seconds, theta_fit_diagnostics = (
        fit_thetas(
            theta_objective,
            theta_bandwidth_candidates,
            seed,
            lr=adam_lr,
            max_epochs=adam_max_epochs,
            patience=adam_patience,
        )
    )
    for theta_diagnostic in theta_fit_diagnostics:
        print(
            f"[HY theta] h={theta_diagnostic['h']:.1f} "
            f"epochs={theta_diagnostic['epochs']} "
            f"best_S_theta={theta_diagnostic['best_S_theta']:.6g}",
            flush=True,
        )
    prepared_sampler_for_each_h = {}
    bandwidth_energy_scores = []
    best_bandwidth_result = None

    for h_candidate, H_candidate in bandwidth_grid():
        if h_candidate not in prepared_sampler_for_each_h:
            prepared_sampler_for_each_h[h_candidate] = (
                PreparedConditionalSampler(
                    X_train,
                    Y_train,
                    X_valid_subset,
                    fitted_theta_for_each_h[h_candidate],
                    y_grid,
                )
            )
        energy_score = prepared_sampler_for_each_h[h_candidate].energy_score(
            H_candidate, Y_valid_subset
        )
        bandwidth_energy_scores.append(
            (h_candidate, H_candidate, energy_score)
        )
        print(
            f"[HY tune] h={h_candidate:.1f} H={H_candidate:.1f} "
            f"EnergyScore={energy_score:.6f}",
            flush=True,
        )
        if (
            best_bandwidth_result is None
            or energy_score < best_bandwidth_result[2]
        ):
            best_bandwidth_result = (h_candidate, H_candidate, energy_score)

    h_best, H_best, best_energy_score = best_bandwidth_result
    chosen_theta_diagnostic = next(
        entry for entry in theta_fit_diagnostics if entry["h"] == h_best
    )
    total_fit_seconds = time.perf_counter() - fit_and_tune_start
    # theta_fit_seconds is wall-clock time, not the sum of six h-specific runs.
    print(
        f"[HY tune] selected h={h_best:.1f}, H={H_best:.1f}; "
        f"EnergyScore={best_energy_score:.6f}; theta_fit={theta_fit_seconds:.2f}s, "
        f"total_fit={total_fit_seconds:.2f}s"
    )
    return {
        "theta": fitted_theta_for_each_h[h_best],
        "h": h_best,
        "H": H_best,
        "theta_fit_seconds": theta_fit_seconds,
        "total_fit_seconds": total_fit_seconds,
        "optimization_success": True,
        "optimization_message": _theta_fit_message(chosen_theta_diagnostic),
        "mode": "tuned",
        "bandwidth_energy_scores": bandwidth_energy_scores,
        "n_val_used": n_valid_used,
        "n_grid_tuning": n_grid_tuning,
        "theta_fit_diagnostics": theta_fit_diagnostics,
    }

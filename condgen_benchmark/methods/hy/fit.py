"""Adam fitting of Hall--Yao theta for independent bandwidths."""

import time

import numpy as np
import torch


def fit_thetas(
    theta_objective,
    bandwidths,
    seed,
    *,
    lr=1e-2,
    max_epochs=300,
    patience=30,
    normalization_epsilon=1e-5,
):
    """Estimate theta for each h, with per-h early stopping.

    Returns fitted_theta_for_each_h (h -> theta), theta_fit_seconds, and
    theta_fit_diagnostics (one entry per h). Training stays on one Torch device;
    only the final estimates and diagnostics are transferred to CPU.
    """
    if lr <= 0 or max_epochs < 1 or patience < 1 or normalization_epsilon <= 0:
        raise ValueError(
            "Adam requires lr>0, max_epochs>=1, patience>=1 and "
            "normalization_epsilon>0"
        )
    theta_bandwidths = np.asarray(bandwidths, dtype=np.float64)
    if (
        theta_bandwidths.ndim != 1
        or theta_bandwidths.size == 0
        or not np.all(theta_bandwidths > 0)
    ):
        raise ValueError(
            "bandwidths must be a nonempty list of positive numbers"
        )
    rng = np.random.RandomState(seed)
    initial_theta = rng.randn(theta_objective.d)
    initial_theta /= np.linalg.norm(initial_theta)
    # Use the same initial theta for all h.
    initial_w = np.repeat(
        initial_theta[None, :], len(theta_bandwidths), axis=0
    )
    w = torch.nn.Parameter(
        torch.as_tensor(
            initial_w.copy(),
            dtype=theta_objective.X.dtype,
            device=theta_objective.device,
        )
    )
    theta_bandwidths_tensor = torch.as_tensor(
        theta_bandwidths,
        dtype=theta_objective.X.dtype,
        device=theta_objective.device,
    )
    optimizer = torch.optim.Adam([w], lr=lr)
    best_S_theta_values = torch.full(
        (len(theta_bandwidths),),
        float("inf"),
        dtype=w.dtype,
        device=theta_objective.device,
    )
    best_w = w.detach().clone()
    epochs_without_improvement = torch.zeros(
        len(theta_bandwidths), dtype=torch.long, device=theta_objective.device
    )
    epochs_completed = torch.zeros(
        len(theta_bandwidths), dtype=torch.long, device=theta_objective.device
    )
    training_active = torch.ones(
        len(theta_bandwidths), dtype=torch.bool, device=theta_objective.device
    )
    theta_fit_start = time.perf_counter()
    for _ in range(max_epochs):
        active_indices = torch.where(training_active)[0]
        if active_indices.numel() == 0:
            break
        optimizer.zero_grad(set_to_none=True)
        w_active = w.index_select(0, active_indices)
        theta = w_active / torch.sqrt(
            w_active.square().sum(dim=1, keepdim=True) + normalization_epsilon
        )
        current_S_theta_values = theta_objective.objectives(
            theta, theta_bandwidths_tensor.index_select(0, active_indices)
        )
        with torch.no_grad():
            previous_best_S_theta = best_S_theta_values.index_select(
                0, active_indices
            )
            improved_best_S_theta = torch.isfinite(
                current_S_theta_values.detach()
            ) & (current_S_theta_values.detach() < previous_best_S_theta)
            best_S_theta_values[active_indices] = torch.where(
                improved_best_S_theta,
                current_S_theta_values.detach(),
                previous_best_S_theta,
            )
            best_w[active_indices] = torch.where(
                improved_best_S_theta[:, None],
                w_active.detach(),
                best_w.index_select(0, active_indices),
            )
            epochs_completed[active_indices] += 1
            epochs_without_improvement[active_indices] = torch.where(
                improved_best_S_theta,
                0,
                epochs_without_improvement.index_select(0, active_indices) + 1,
            )
            training_active[active_indices] = (
                epochs_without_improvement.index_select(0, active_indices)
                < patience
            )
        if not torch.isfinite(current_S_theta_values).all():
            raise RuntimeError(
                "HY Adam loss became NaN/Inf; consider a smaller learning rate"
            )
        current_S_theta_values.sum().backward()
        optimizer.step()

    # Include the final Adam update among the candidates for the best w.
    # (The ordinary loop evaluates the parameters BEFORE each update.)
    remaining_active_indices = torch.where(training_active)[0]
    if remaining_active_indices.numel() > 0:
        with torch.no_grad():
            final_w = w.index_select(0, remaining_active_indices)
            final_theta = final_w / torch.sqrt(
                final_w.square().sum(dim=1, keepdim=True)
                + normalization_epsilon
            )
            final_S_theta_values = theta_objective.objectives(
                final_theta,
                theta_bandwidths_tensor.index_select(
                    0, remaining_active_indices
                ),
            )
            previous_best_S_theta = best_S_theta_values.index_select(
                0, remaining_active_indices
            )
            improved_final_S_theta = torch.isfinite(final_S_theta_values) & (
                final_S_theta_values < previous_best_S_theta
            )
            best_S_theta_values[remaining_active_indices] = torch.where(
                improved_final_S_theta,
                final_S_theta_values,
                previous_best_S_theta,
            )
            best_w[remaining_active_indices] = torch.where(
                improved_final_S_theta[:, None],
                final_w,
                best_w.index_select(0, remaining_active_indices),
            )
    if theta_objective.device.type == "cuda":
        torch.cuda.synchronize(theta_objective.device)
    theta_fit_seconds = time.perf_counter() - theta_fit_start
    if not torch.isfinite(best_S_theta_values).all():
        raise RuntimeError("HY Adam failed to obtain a finite objective")
    with torch.no_grad():
        best_thetas_tensor = best_w / torch.sqrt(
            best_w.square().sum(dim=1, keepdim=True) + normalization_epsilon
        )
        fitted_thetas = best_thetas_tensor.cpu().numpy().astype(np.float64)
        best_S_theta_numpy = (
            best_S_theta_values.cpu().numpy().astype(np.float64)
        )
        epochs_completed_numpy = epochs_completed.cpu().numpy().astype(int)
    # Stable sign convention, consistent with the CPU implementation.
    for k in range(len(fitted_thetas)):
        nz = np.flatnonzero(fitted_thetas[k])
        if len(nz) and fitted_thetas[k, nz[0]] < 0:
            fitted_thetas[k] *= -1
    fitted_theta_for_each_h = {
        float(theta_bandwidths[k]): fitted_thetas[k]
        for k in range(len(theta_bandwidths))
    }
    theta_fit_diagnostics = [
        {
            "h": float(theta_bandwidths[k]),
            "epochs": int(epochs_completed_numpy[k]),
            "best_S_theta": float(best_S_theta_numpy[k]),
        }
        for k in range(len(theta_bandwidths))
    ]
    return fitted_theta_for_each_h, theta_fit_seconds, theta_fit_diagnostics

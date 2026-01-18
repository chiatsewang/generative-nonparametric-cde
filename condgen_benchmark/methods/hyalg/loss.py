import torch

from .kernel import epanechnikov_kernel
from .utils import (
    enforce_positive_first_nonzero,
    indicator_matrix,
    leave_two_out_sum,
)


def S_theta(X, Y, A_centers, radius, h, tol=1e-12):
    n = X.shape[0]
    Y_ind = indicator_matrix(Y)  # (n, n)

    with torch.no_grad():
        dist = torch.cdist(A_centers, X)  # (A, n)
        in_sphere = dist <= radius  # (A, n)
        mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        Y_ind_masked = Y_ind * mask
        pi_empirical = torch.matmul(in_sphere.float(), Y_ind_masked) / (n - 1)

    def _S_theta(theta_np):
        theta_np = enforce_positive_first_nonzero(theta_np)
        theta = torch.tensor(theta_np, dtype=X.dtype, device=X.device)
        theta = theta / torch.norm(theta)

        proj_X = X @ theta
        u = (proj_X[:, None] - proj_X[None, :]) / h
        K = epanechnikov_kernel(u)

        T1 = leave_two_out_sum(K * u) / ((n - 2) * h)
        T2 = leave_two_out_sum(K * u**2) / ((n - 2) * h)
        norm = leave_two_out_sum(K) * T2 - leave_two_out_sum(K * u) * T1
        wsum = (K @ Y_ind - K) * T2 - ((K * u) @ Y_ind - (K * u)) * T1

        F = torch.empty_like(wsum)
        fallback_matrix = leave_two_out_sum(Y_ind.clone(), dim=0) / (n - 2)
        mask = norm.abs() < tol
        F[mask] = fallback_matrix[mask]
        F[~mask] = (wsum / (norm + tol))[~mask]
        F.fill_diagonal_(0.0)

        pi_hat = torch.matmul(in_sphere.float(), F[None, :]) / (n - 1)

        loss = ((pi_empirical - pi_hat) ** 2).sum().item()
        return loss

    return _S_theta

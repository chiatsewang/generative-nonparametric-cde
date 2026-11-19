"""Differentiable Hall--Yao objective for theta estimation."""

import numpy as np
import torch
from scipy.sparse import csr_matrix


class HallYaoLoss:
    """Batched exact leave-two-out criterion for theta vectors, shape (b,d)."""

    def __init__(
        self, X, Y, radius=1.0, device=None, dtype=torch.float32, tol=1e-12
    ):
        # Device is chosen automatically for training. An explicit device is
        # accepted here only for low-level unit tests and diagnostics.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is not available"
            )
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must have shape (n, p)")
        n, d = X.shape
        if n < 3 or len(Y) != n:
            raise ValueError("X and scalar Y must have matching n >= 3")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.n, self.d = n, d
        self.tol = tol
        response_sort_indices = np.argsort(Y, kind="stable").astype(np.int64)
        response_right_counts = np.searchsorted(
            Y[response_sort_indices], Y, side="right"
        ).astype(np.int64)

        # Fixed sphere membership and empirical CDF terms. No optimizer or
        # theta-dependent quantity is computed here.
        ball_axis_grid = np.linspace(-1.5, 1.5, 3)
        ball_centers = np.stack(
            np.meshgrid(*([ball_axis_grid] * d), indexing="ij"), axis=-1
        ).reshape(-1, d)
        distances_sq = ((ball_centers[:, None, :] - X[None, :, :]) ** 2).sum(
            axis=2
        )
        ball_membership = np.ascontiguousarray(
            (distances_sq <= radius * radius).astype(np.float64)
        )
        cumulative_ball_membership = np.cumsum(
            ball_membership[:, response_sort_indices], axis=1
        )
        empirical_ball_cdf = (
            cumulative_ball_membership[:, response_right_counts - 1]
            - ball_membership
        ) / (n - 1)
        sparse_ball_membership = csr_matrix(ball_membership)

        self.X = torch.as_tensor(X.copy(), dtype=dtype, device=self.device)
        self.Y = torch.as_tensor(Y.copy(), dtype=dtype, device=self.device)
        self.response_sort_indices = torch.as_tensor(
            response_sort_indices.copy(), dtype=torch.long, device=self.device
        )
        self.response_right_indices = torch.as_tensor(
            response_right_counts - 1, dtype=torch.long, device=self.device
        )
        self.leave_one_out_mask = ~torch.eye(
            n, dtype=torch.bool, device=self.device
        )
        self.fallback_cdf = (
            (
                self.response_right_indices[None, :]
                - (self.Y[:, None] <= self.Y[None, :]).to(torch.long)
            )
            / (n - 2)
        ).to(dtype)
        self.empirical_ball_cdf = torch.as_tensor(
            empirical_ball_cdf.copy(), dtype=dtype, device=self.device
        )
        self.ball_membership_matrix = torch.sparse_csr_tensor(
            torch.as_tensor(
                sparse_ball_membership.indptr.astype(np.int64),
                device=self.device,
            ),
            torch.as_tensor(
                sparse_ball_membership.indices.astype(np.int64),
                device=self.device,
            ),
            torch.as_tensor(
                sparse_ball_membership.data, dtype=dtype, device=self.device
            ),
            size=sparse_ball_membership.shape,
            device=self.device,
        )
        self.num_balls = sparse_ball_membership.shape[0]

    def objectives(self, theta, bandwidths):
        """One independent S_theta for each row of theta and bandwidths."""
        if theta.ndim != 2 or theta.shape[1] != self.d:
            raise ValueError("theta must have shape (number_of_bandwidths, p)")
        bandwidths = torch.as_tensor(
            bandwidths, dtype=theta.dtype, device=theta.device
        )
        if bandwidths.ndim != 1 or bandwidths.numel() != theta.shape[0]:
            raise ValueError("bandwidths must have one value per theta row")
        b = theta.shape[0]
        n = self.n
        # All six bands processed in one batched tensor operation.
        train_projections = torch.matmul(theta, self.X.T)  # (b,n)
        h = bandwidths[:, None, None]
        u = (train_projections[:, :, None] - train_projections[:, None, :]) / h
        K = 0.75 * (1 - u.square()) * (u.abs() <= 1).to(u.dtype)
        K = K * self.leave_one_out_mask[None, :, :]
        Ku = K * u
        Ku2 = Ku * u
        s0 = K.sum(dim=2, keepdim=True)
        s1 = Ku.sum(dim=2, keepdim=True)
        s2 = Ku2.sum(dim=2, keepdim=True)
        t1 = (s1 - Ku) / ((n - 2) * h)
        t2 = (s2 - Ku2) / ((n - 2) * h)
        norm = (s0 - K) * t2 - (s1 - Ku) * t1
        # All Y thresholds use cumulative sums in sorted Y order.
        cK = K.index_select(2, self.response_sort_indices).cumsum(dim=2)
        cKu = Ku.index_select(2, self.response_sort_indices).cumsum(dim=2)
        c0 = cK.index_select(2, self.response_right_indices) - K
        c1 = cKu.index_select(2, self.response_right_indices) - Ku
        numerator = c0 * t2 - c1 * t1
        # No in-place operations on tensors required for autograd.
        # In float32, cancellation at nearly singular local-linear fits may
        # produce an unstable denominator.  Use a numerical safety threshold
        # appropriate to float32 (the float64 reference keeps its tolerance).
        threshold = (
            max(self.tol, 1e-8) if theta.dtype == torch.float32 else self.tol
        )
        singular = norm.abs() < threshold
        safe_denom = torch.where(
            singular, torch.ones_like(norm), norm + self.tol
        )
        F = torch.where(
            singular, self.fallback_cdf[None, :, :], numerator / safe_denom
        )
        F = F * self.leave_one_out_mask[None, :, :]
        # Fixed ball-incidence matrix is sparse.  Torch autograd differentiates
        # the dense right-hand side through sparse.mm on CPU/CUDA.
        # Stack all bandwidths into the dense right-hand operand so the
        # fixed CSR membership matrix is multiplied only once per epoch.
        # The independent objectives and their autograd paths are preserved.
        all_bandwidth_cdfs = F.permute(1, 0, 2).reshape(n, b * n)
        predicted_ball_cdf = torch.sparse.mm(
            self.ball_membership_matrix, all_bandwidth_cdfs
        ).reshape(self.num_balls, b, n).permute(1, 0, 2) / (n - 1)
        return (
            (predicted_ball_cdf - self.empirical_ball_cdf[None, :, :])
            .square()
            .sum(dim=(1, 2))
        )

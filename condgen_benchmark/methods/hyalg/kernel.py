import torch


def epanechnikov_kernel(u: torch.Tensor) -> torch.Tensor:
    return 0.75 * (1 - u**2) * (u.abs() <= 1)

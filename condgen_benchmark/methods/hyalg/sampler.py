import torch

from .kernel import epanechnikov_kernel


@torch.no_grad()
def sample_conditional(
    X, Y, x_test, theta_hat, yGrid, H, num_samples_Y, tol=1e-12
):
    n_test = x_test.shape[0]
    n = X.shape[0]
    proj_X = X @ theta_hat

    samples = []

    for i in range(n_test):
        x = x_test[i]
        proj_x = x @ theta_hat
        u = (proj_x - proj_X) / H
        K = epanechnikov_kernel(u)
        T1 = (K * u).sum() / (n * H)
        T2 = (K * u**2).sum() / (n * H)
        w = K * (T2 - u * T1)
        denom = w.sum() + tol

        conF = torch.tensor(
            [(w * (Y <= y)).sum() / denom for y in yGrid], device=X.device
        )
        F_unique = torch.unique(conF, sorted=True)
        uniqIdx = torch.tensor(
            [torch.where(conF == val)[0][0].item() for val in F_unique],
            device=conF.device,
        )

        if len(F_unique) < 10 or torch.all(torch.diff(F_unique) == 0):
            y_samples = torch.full(
                (num_samples_Y,), Y.mean().item(), device=Y.device
            )
        else:
            u_rand = torch.rand(num_samples_Y, device=Y.device).clamp(
                min=F_unique.min().item(), max=F_unique.max().item()
            )
            idx = torch.searchsorted(F_unique, u_rand, right=False).clamp(
                max=F_unique.numel() - 1
            )
            idx0 = (idx - 1).clamp(min=0)
            idx1 = idx

            x0 = F_unique[idx0]
            x1 = F_unique[idx1]
            y0 = yGrid[uniqIdx][idx0]
            y1 = yGrid[uniqIdx][idx1]

            weight = (u_rand - x0) / (x1 - x0 + tol)
            y_samples = y0 + weight * (y1 - y0)

        samples.append(y_samples.unsqueeze(0))

    return torch.cat(samples, dim=0)  # [n_test, num_samples_Y]

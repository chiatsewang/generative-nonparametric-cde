import torch


def energy_score_loss(y_samples, y_true, beta=1.0):
    """Monte Carlo estimate of the energy score.

    y_samples: (batch_size, n_samples, y_dim)
    y_true:    (batch_size, y_dim)
    """
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)

    n_samples = y_samples.shape[1]
    if n_samples < 2:
        raise ValueError(
            "energy_score_loss requires at least two generated samples"
        )

    term1 = torch.norm(y_samples - y_true[:, None, :], dim=-1).pow(beta).mean()

    # U-statistic: exclude self-pairs, using S(S-1) rather than S^2.
    # All diagonal distances are zero, so summing the full matrix is equivalent
    # to summing only the off-diagonal entries.
    diff = y_samples[:, :, None, :] - y_samples[:, None, :, :]
    term2 = torch.norm(diff, dim=-1).pow(beta).sum(dim=(1, 2)).mean()
    term2 = term2 / (n_samples * (n_samples - 1))

    return term1 - 0.5 * term2

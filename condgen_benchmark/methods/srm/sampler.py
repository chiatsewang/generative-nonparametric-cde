import torch
from tqdm import tqdm


@torch.no_grad()
def sample_srm(generator, x, n_samples=1000, batch_size=256, device="cpu"):
    """Generate samples from the fitted SRM conditional generator."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.detach().float()

    if x.dim() == 1:
        x = x.unsqueeze(1)

    device = torch.device(device)
    generator.to(device)
    generator.eval()

    # Test X is small in this benchmark; transfer it once instead of once/batch.
    x = x.to(device)
    samples = []
    for start in tqdm(range(0, x.shape[0], batch_size), desc="Sampling SRM"):
        xb = x[start : start + batch_size]
        yb = generator(xb, n_samples=n_samples)
        samples.append(yb.cpu())

    return torch.cat(samples, dim=0)


def denormalize_y(y, mean, std):
    """Undo coordinate-wise normalization of generated Y samples."""
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=torch.float32)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=torch.float32)

    return y * std.view(1, 1, -1) + mean.view(1, 1, -1)

import time

import torch
from torch.optim import Adam

from condgen_benchmark.utils.gpu_batching import batch_indices

from .loss import gcds_loss


def train_gcds(
    generator,
    discriminator,
    x_train,
    y_train,
    batch_size,
    noise_dim,
    num_epochs=100,
    lr_g=1e-3,
    lr_d=1e-3,
):
    device = x_train.device
    generator.to(device)
    discriminator.to(device)

    opt_g = Adam(generator.parameters(), lr=lr_g)
    opt_d = Adam(discriminator.parameters(), lr=lr_d)

    history_g = []
    history_d = []
    epoch_times = []
    n_train = x_train.shape[0]

    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()

        generator.train()
        discriminator.train()
        total_loss_g = torch.zeros((), device=device)
        total_loss_d = torch.zeros((), device=device)
        n_obs = 0

        for idx in batch_indices(n_train, batch_size, device, shuffle=True):
            x = x_train[idx]
            y = y_train[idx]
            noise = torch.randn(x.size(0), noise_dim, device=device)

            # Anomaly detection is intentionally OFF here; enabling it in the
            # normal training loop is extremely expensive and is only useful
            # for debugging autograd failures.
            loss_g, loss_d = gcds_loss(generator, discriminator, x, y, noise)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            bs = x.shape[0]
            total_loss_g += loss_g.detach() * bs
            total_loss_d += loss_d.detach() * bs
            n_obs += bs

        avg_loss_g = (total_loss_g / n_obs).item()
        avg_loss_d = (total_loss_d / n_obs).item()

        history_g.append(avg_loss_g)
        history_d.append(avg_loss_d)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start
        epoch_times.append(elapsed)
        print(
            f"Epoch {epoch:3d} | Loss G: {avg_loss_g:.4f} | "
            f"Loss D: {avg_loss_d:.4f} | time: {elapsed:.2f}s"
        )

    return history_g, history_d, epoch_times

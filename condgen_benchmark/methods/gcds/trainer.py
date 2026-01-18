# condgen_benchmark/algorithms/gcds/trainer.py
import time

import torch
from torch.optim import Adam

from .loss import gcds_loss


def train_gcds(
    generator,
    discriminator,
    dataloader,
    noise_dim,
    num_epochs=100,
    lr_g=1e-3,
    lr_d=1e-3,
    device="cpu",
):
    generator.to(device)
    discriminator.to(device)

    opt_g = Adam(generator.parameters(), lr=lr_g)
    opt_d = Adam(discriminator.parameters(), lr=lr_d)

    history_g = []
    history_d = []

    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        generator.train()
        discriminator.train()
        total_loss_g = 0.0
        total_loss_d = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            noise = torch.randn(x.size(0), noise_dim, device=device)

            # --- Compute losses ---
            with torch.autograd.set_detect_anomaly(True):
                loss_g, loss_d = gcds_loss(
                    generator, discriminator, x, y, noise
                )

                # --- Update Generator ---
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                # --- Update Discriminator ---
                opt_d.zero_grad()
                # loss_d.backward(retain_graph=True)
                loss_d.backward()
                opt_d.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            n_batches += 1

        avg_loss_g = total_loss_g / n_batches
        avg_loss_d = total_loss_d / n_batches

        history_g.append(avg_loss_g)
        history_d.append(avg_loss_d)

        elapsed = time.time() - start
        epoch_times.append(elapsed)
        print(
            f"Epoch {epoch:3d} | Loss G: {avg_loss_g:.4f} | Loss D: {avg_loss_d:.4f} | time: {elapsed:.2f}s"
        )
    # end of the for loop

    # return the average loss and elapse time
    return history_g, history_d, epoch_times

import copy
import time

import torch
from torch.optim import AdamW

from condgen_benchmark.methods.srm.loss import energy_score_loss
from condgen_benchmark.utils.gpu_batching import batch_indices


@torch.no_grad()
def evaluate_srm(generator, x_val, y_val, batch_size, n_samples=30, beta=1.0):
    generator.eval()
    n_obs = x_val.shape[0]
    total_loss = torch.zeros((), device=x_val.device)

    for idx in batch_indices(n_obs, batch_size, x_val.device, shuffle=False):
        x = x_val[idx]
        y = y_val[idx]
        y_samples = generator(x, n_samples=n_samples)
        loss = energy_score_loss(y_samples, y, beta=beta)
        total_loss += loss.detach() * x.shape[0]

    return (total_loss / max(n_obs, 1)).item()


def train_srm(
    generator,
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    batch_size=128,
    n_samples=30,
    num_epochs=500,
    lr=1e-4,
    weight_decay=0.0,
    beta=1.0,
    early_stop=False,
    patience=50,
    print_every=1,
):
    device = x_train.device
    generator.to(device)
    optimizer = AdamW(generator.parameters(), lr=lr, weight_decay=weight_decay)

    history_train = []
    history_val = []
    epoch_times = []

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    n_train = x_train.shape[0]

    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        generator.train()

        total_loss = torch.zeros((), device=device)

        for idx in batch_indices(n_train, batch_size, device, shuffle=True):
            x = x_train[idx]
            y = y_train[idx]

            y_samples = generator(x, n_samples=n_samples)
            loss = energy_score_loss(y_samples, y, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * x.shape[0]

        avg_train = (total_loss / max(n_train, 1)).item()
        history_train.append(avg_train)

        if x_val is not None and y_val is not None:
            avg_val = evaluate_srm(
                generator,
                x_val,
                y_val,
                batch_size=batch_size,
                n_samples=n_samples,
                beta=beta,
            )
            history_val.append(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                best_state = copy.deepcopy(generator.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
        else:
            avg_val = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start
        epoch_times.append(elapsed)

        if print_every > 0 and epoch % print_every == 0:
            msg = f"Epoch {epoch:4d} | train SRM loss: {avg_train:.6f}"
            if avg_val is not None:
                msg += f" | val SRM loss: {avg_val:.6f}"
            msg += f" | time: {elapsed:.2f}s"
            print(msg)

        if early_stop and x_val is not None and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        generator.load_state_dict(best_state)

    return history_train, history_val, epoch_times

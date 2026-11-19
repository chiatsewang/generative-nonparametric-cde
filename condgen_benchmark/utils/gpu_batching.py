import torch


def batch_indices(n, batch_size, device, shuffle):
    """Yield index tensors directly on *device* without a DataLoader."""
    if shuffle:
        order = torch.randperm(n, device=device)
    else:
        order = torch.arange(n, device=device)

    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]

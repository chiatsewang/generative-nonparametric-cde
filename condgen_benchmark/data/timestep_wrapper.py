import torch
from torch.utils.data import Dataset


class TimestepWrapper(Dataset):
    def __init__(self, base_dataset, T: int, require_norm: bool = False):
        self.base_dataset = base_dataset
        self.T = T
        self.require_norm = require_norm

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        t = torch.randint(0, self.T, ()).long()
        if self.require_norm:
            t_norm = (t.float() / self.T).view(1)
            return x, y, t, t_norm
        return x, y, t


# Old implementation (now commented out):
# The batch consisted of all possible pairs (i, t),
#  i.e., (i, t) ∈ {1,...,N} × {1,...,T}.
# Current implementation samples a single t for each i.

# class TimestepWrapper(Dataset):
#     def __init__(self, base_dataset, T: int, require_norm: bool = False):
#         """
#         Args:
#             base_dataset: a dataset returning (x, y)
#             T: number of diffusion timesteps
#         """
#         self.base_dataset = base_dataset
#         self.T = T
#         self.require_norm = require_norm

#     def __len__(self):
#         return len(self.base_dataset) * self.T

#     def __getitem__(self, idx):
#         data_idx = idx // self.T
#         t = idx % self.T
#         t = torch.tensor(t, dtype=torch.long)

#         x, y = self.base_dataset[data_idx]
#         if self.require_norm:
#             t_norm = torch.tensor([t / float(self.T)], dtype=torch.float32)
#             return x, y, t, t_norm
#         return x, y, t

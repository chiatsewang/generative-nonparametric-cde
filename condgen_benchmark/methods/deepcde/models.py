import torch.nn as nn
import torch.nn.functional as F

from .deepcde_pytorch import cde_layer

# class DeepCDEModel(nn.Module):
#     def __init__(self, input_dim: int, n_basis: int):
#         super().__init__()
#         self.l1 = nn.Linear(input_dim, 32)
#         self.l2 = nn.Linear(32, 64)
#         self.l3 = nn.Linear(64, 32)
#         self.cde = cde_layer(32, n_basis)

#     def forward(self, x):
#         x = F.leaky_relu(self.l1(x))
#         x = F.leaky_relu(self.l2(x))
#         x = F.leaky_relu(self.l3(x))
#         return self.cde(x)  # beta-coefficients


class DeepCDEModel(nn.Module):
    def __init__(self, input_dim: int, n_basis: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 32)
        self.cde = cde_layer(32, n_basis)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        x = F.gelu(self.l3(x))
        return self.cde(x)

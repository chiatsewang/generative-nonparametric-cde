# condgen_benchmark/algorithms/gcds/core.py
import torch
import torch.nn as nn


class GCDSGenerator(nn.Module):
    def __init__(self, x_dim, noise_dim, y_dim, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, x, noise):
        inp = torch.cat([x, noise], dim=1)
        return self.net(inp)


class GCDSDiscriminator(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims=(50, 25)):
        super().__init__()
        layers = [nn.Linear(x_dim + y_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
            ]
        layers += [nn.Linear(hidden_dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        if out.shape[-1] == 1:
            out = out.view(-1)
        # return self.net(inp).squeeze(-1)
        return out

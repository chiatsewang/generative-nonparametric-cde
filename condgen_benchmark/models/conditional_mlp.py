import torch
import torch.nn as nn


class ConditionalMLPScalarT(nn.Module):
    def __init__(self, predictor_dim, response_dim, hidden_dim=50):
        super().__init__()
        input_dim = predictor_dim + response_dim + 1  # scalar t
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, response_dim),
        )

    def forward(self, predictor, response_noise, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B] → [B, 1]
        inp = torch.cat([predictor, response_noise, t], dim=1)
        return self.net(inp)

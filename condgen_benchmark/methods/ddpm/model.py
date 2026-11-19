import torch
import torch.nn as nn


def parse_hidden_dims(hidden_dims):
    """Convert a hidden-layer specification to a list of integers."""
    if hidden_dims is None:
        return [1024, 512, 256]
    if isinstance(hidden_dims, str):
        dims = [int(h.strip()) for h in hidden_dims.split(",") if h.strip()]
    else:
        dims = [int(h) for h in hidden_dims]
    if not dims:
        raise ValueError("hidden_dims must contain at least one hidden layer")
    return dims


class ConditionalMLPScalarT(nn.Module):
    def __init__(
        self,
        predictor_dim,
        response_dim,
        hidden_dims=(1024, 512, 256),
    ):
        super().__init__()

        self.predictor_dim = int(predictor_dim)
        self.response_dim = int(response_dim)
        self.hidden_dims = parse_hidden_dims(hidden_dims)

        layers = []
        in_dim = self.predictor_dim + self.response_dim + 1
        for hidden_dim in self.hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.response_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, predictor, response_noise, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)

        inp = torch.cat([predictor, response_noise, t], dim=1)
        return self.net(inp)

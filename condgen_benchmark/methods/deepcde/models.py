import torch.nn as nn
import torch.nn.functional as F

from condgen_benchmark.methods.deepcde.deepcde_pytorch import cde_layer


def parse_hidden_dims(hidden_dims):
    """Parse comma-separated hidden layer widths or a sequence of widths."""
    if hidden_dims is None:
        return [32, 64, 32]
    if isinstance(hidden_dims, str):
        dims = [int(h.strip()) for h in hidden_dims.split(",") if h.strip()]
    else:
        dims = [int(h) for h in hidden_dims]
    if not dims or any(h <= 0 for h in dims):
        raise ValueError(
            "hidden_dims must contain positive hidden layer sizes"
        )
    return dims


class DeepCDEModel(nn.Module):
    def __init__(self, input_dim: int, n_basis: int, hidden_dims=(32, 64, 32)):
        super().__init__()
        self.hidden_dims = parse_hidden_dims(hidden_dims)
        in_dim = input_dim
        for i, width in enumerate(self.hidden_dims, start=1):
            # Preserve l1/l2/l3 keys when using the legacy default architecture.
            setattr(self, f"l{i}", nn.Linear(in_dim, width))
            in_dim = width
        self.cde = cde_layer(in_dim, n_basis)

    def forward(self, x):
        for i in range(1, len(self.hidden_dims) + 1):
            x = F.gelu(getattr(self, f"l{i}")(x))
        return self.cde(x)

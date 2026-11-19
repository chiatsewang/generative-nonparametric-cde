import torch
import torch.nn as nn


def parse_hidden_dims(hidden_dims):
    """Convert a hidden-layer specification to a list of integers.

    The training script accepts hidden dimensions as a comma-separated string
    such as "1024,512,256". The saved config stores the parsed list so that the
    sampling script can reconstruct exactly the same architecture before
    loading generator.pth.
    """
    if hidden_dims is None:
        return [1024, 512, 256]
    if isinstance(hidden_dims, str):
        dims = [int(h.strip()) for h in hidden_dims.split(",") if h.strip()]
    else:
        dims = [int(h) for h in hidden_dims]
    if not dims:
        raise ValueError("hidden_dims must contain at least one hidden layer")
    return dims


class SRMGenerator(nn.Module):
    """Conditional generator for scoring rule minimization.

    Given covariates x and auxiliary noise epsilon, this model generates
    samples from an estimated conditional distribution of Y | X=x.
    """

    def __init__(
        self,
        x_dim,
        noise_dim,
        y_dim,
        hidden_dims=(1024, 512, 256),
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.noise_dim = int(noise_dim)
        self.y_dim = int(y_dim)
        self.hidden_dims = parse_hidden_dims(hidden_dims)

        layers = []
        in_dim = self.x_dim + self.noise_dim
        for hidden_dim in self.hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, n_samples=1):
        """Generate conditional samples.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, x_dim).
        n_samples : int
            Number of generated samples per observation.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, n_samples, y_dim).
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        epsilon = torch.randn(
            batch_size,
            n_samples,
            self.noise_dim,
            device=x.device,
            dtype=x.dtype,
        )
        x_rep = x[:, None, :].expand(batch_size, n_samples, x.shape[1])
        inp = torch.cat([x_rep, epsilon], dim=-1)
        return self.net(inp)

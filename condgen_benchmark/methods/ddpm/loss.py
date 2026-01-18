import torch
import torch.nn.functional as F


def ddpm_loss(model, diffusion, x0, cond, t, t_norm):
    # device = x0.device
    # t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise)
    noise_pred = model(cond, xt, t_norm)
    return F.mse_loss(noise_pred, noise)

import torch


@torch.no_grad()
def sample_ddpm(model, diffusion, cond, shape, device="cpu"):
    x = torch.randn(shape, device=device)
    T = diffusion.timesteps

    for t in reversed(range(T)):
        t_batch = torch.full(
            (x.shape[0],), t / T, device=device, dtype=torch.float32
        )
        noise_pred = model(cond, x, t_batch)
        alpha = diffusion.alphas[t]
        alpha_bar = diffusion.alpha_bars[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        # Compute posterior variance σ²(t) = (1-αₜ)(1-ᾱₜ₋₁)/(1-ᾱₜ)
        if t > 0:
            alpha_bar_prev = diffusion.alpha_bars[t - 1]
            sigma_t = (
                (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            ).sqrt()
        else:
            sigma_t = torch.tensor(0.0, device=device)

        x = (
            1
            / alpha.sqrt()
            * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred)
            + sigma_t * noise
        )
    return x

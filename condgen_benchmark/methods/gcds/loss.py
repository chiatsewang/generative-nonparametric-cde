import torch

"""
def gcds_loss(generator, discriminator, x_real, y_real, noise):
    y_fake = generator(x_real, noise)
    d_real = discriminator(x_real, y_real)
    d_fake = discriminator(x_real, y_fake.detach())

    loss_d = -torch.mean(d_fake) + torch.mean(torch.exp(d_real))
    loss_g = torch.mean(discriminator(x_real, generator(x_real, noise)))
    return loss_g, loss_d
"""


def gcds_loss(generator, discriminator, x_real, y_real, noise):
    if x_real.dim() == 1:
        x_real = x_real.unsqueeze(1)
    if y_real.dim() == 1:
        y_real = y_real.unsqueeze(1)

    x_fake = generator(x_real, noise)
    d_real = discriminator(x_real, y_real)
    d_fake = discriminator(x_real, x_fake.detach())

    loss_d = -torch.mean(d_fake) + torch.mean(torch.exp(d_real))
    # loss_g = torch.mean(discriminator(x_real, generator(x_real, noise)))
    loss_g = torch.mean(discriminator(x_real, x_fake))
    return loss_g, loss_d

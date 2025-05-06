import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from torchvision.utils import save_image


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.lr_mul = lr_mul
        self.scale = (1 / np.sqrt(in_dim)) * lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        else:
            return F.linear(x, self.weight * self.scale)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, dlatent_dim, num_layers):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(EqualLinear(latent_dim if i == 0 else dlatent_dim, dlatent_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)
        self.pixel_norm = PixelNorm()

    def forward(self, z):
        z = self.pixel_norm(z)
        return self.mapping(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, padding=1, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.style = EqualLinear(style_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv(x)
        x = self.noise(x, noise=noise)
        x = x + self.bias
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        x = x * style
        return self.activation(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, style_dim, channels):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim, 8)

        self.input = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.convs.append(StyledConvBlock(channels[i], channels[i+1], style_dim, upsample=(i != 0)))

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[-1], 3, 1),
            nn.Tanh()
        )

    def forward(self, z):
        style = self.mapping(z)
        x = self.input.repeat(z.size(0), 1, 1, 1)

        for conv in self.convs:
            x = conv(x, style)

        return self.to_rgb(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.blocks.append(DiscriminatorBlock(channels[i], channels[i+1]))

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 2 * 2, 1)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)


# Example training loop structure (simplified)
def train_step(generator, discriminator, g_optimizer, d_optimizer, real_images, latent_dim, device):
    batch_size = real_images.size(0)
    real_images = real_images.to(device)

    # Train discriminator
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z).detach()

    d_real = discriminator(real_images)
    d_fake = discriminator(fake_images)

    d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Train generator
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    g_loss = F.softplus(-discriminator(fake_images)).mean()

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item(), fake_images.detach()


# Utilities for saving checkpoints and samples
def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'step': step
    }, os.path.join(checkpoint_dir, f'stylegan_checkpoint_step{step}.pt'))


def save_samples(fake_images, step, sample_dir, nrow=4):
    os.makedirs(sample_dir, exist_ok=True)
    save_image(fake_images[:nrow**2], os.path.join(sample_dir, f'sample_step{step}.png'), nrow=nrow, normalize=True)


# Training config
class TrainConfig:
    def __init__(self):
        self.latent_dim = 512
        self.style_dim = 512
        self.generator_channels =[512, 512, 256, 128, 64, 32]
        self.discriminator_channels = [3, 64, 128, 256, 512, 512]  # Starts with 3 (RGB)
        self.image_size = 64
        self.batch_size = 32
        self.num_epochs = 50
        self.lr = 2e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'D:/My_Computer/Work projects/Research_Coding_Interview/NST/results'
        self.data_path = 'D:/My_Computer/Work projects/Research_Coding_Interview/NST/womens-clothing-dataset'



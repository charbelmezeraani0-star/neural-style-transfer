"""
Transformation network — the model that actually gets trained.

Architecture: Johnson et al. 2016 "Perceptual Losses for Real-Time Style
Transfer and Super-Resolution" (https://arxiv.org/abs/1603.08155)

  Input image  →  3 downsampling conv blocks
               →  N residual blocks (feature transformation)
               →  3 upsampling conv blocks
               →  Output image  (tanh, range [-1, 1])

Instance Normalization is used instead of Batch Normalization —
it works better for style transfer because it normalises per-image.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → InstanceNorm → ReLU (optionally skip activation)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, activate=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, padding_mode="reflect"),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two conv blocks with a residual (skip) connection — no downsampling."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, kernel_size=3, padding=1, activate=False),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsample (2×) followed by a conv — avoids checkerboard artefacts."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(self.up(x))


class TransformNet(nn.Module):
    """
    Lightweight image-to-image network.

    Parameters
    ----------
    base_channels : int
        Width of the network. 32 is fast; 64 is higher quality.
    n_residual : int
        Number of residual blocks. 5 (default) balances quality / speed.
    """

    def __init__(self, base_channels: int = 32, n_residual: int = 5):
        super().__init__()

        c = base_channels

        # Downsampling encoder
        self.encoder = nn.Sequential(
            ConvBlock(3,      c,     kernel_size=9, stride=1, padding=4),
            ConvBlock(c,      c * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(c * 2,  c * 4, kernel_size=3, stride=2, padding=1),
        )

        # Residual bottleneck
        self.residuals = nn.Sequential(
            *[ResidualBlock(c * 4) for _ in range(n_residual)]
        )

        # Upsampling decoder
        self.decoder = nn.Sequential(
            UpsampleBlock(c * 4, c * 2),
            UpsampleBlock(c * 2, c),
        )

        # Final output conv — tanh maps to (-1, 1); we rescale to (0, 1)
        self.output_conv = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=9, stride=1, padding=4, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        x = self.output_conv(x)
        # Tanh output is in (-1, 1) — shift to (0, 1)
        return (x + 1) / 2

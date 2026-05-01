import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.pad  = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x):
        return self.norm(self.conv(self.pad(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, upsample):
        super().__init__()
        self.upsample = upsample
        self.pad  = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return self.norm(self.conv(self.pad(x)))


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (Isola et al. 2017).

    Classifies overlapping 70×70 patches as real (from the style image) or
    fake (from the generator). This forces the generator to produce textures
    that are indistinguishable from the real style at the patch level —
    something gram-matrix losses alone cannot enforce.

    Uses spectral normalisation on every conv layer for training stability.
    """
    def __init__(self):
        super().__init__()
        SN = nn.utils.spectral_norm

        def block(in_ch, out_ch, stride, norm=True):
            layers = [SN(nn.Conv2d(in_ch, out_ch, 4, stride, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3,   64,  2, norm=False),   # no norm on first layer
            *block(64,  128, 2),
            *block(128, 256, 2),
            *block(256, 512, 1),
            SN(nn.Conv2d(512, 1, 4, 1, 1)),    # output: spatial real/fake map
        )

    def forward(self, x):
        return self.model(x)


class TransformerNet(nn.Module):
    """
    Feed-forward stylization network (Johnson et al. 2016).
    Input/Output: [B, 3, H, W] in [0, 1].

    n_residual=9 gives better quality; use n_residual=5 only for faster prototyping.
    """
    def __init__(self, n_residual: int = 9):
        super().__init__()
        self.enc1 = ConvLayer(3,   32,  9, 1)
        self.enc2 = ConvLayer(32,  64,  3, 2)
        self.enc3 = ConvLayer(64,  128, 3, 2)
        self.res  = nn.Sequential(*[ResBlock(128) for _ in range(n_residual)])
        self.dec1 = UpsampleConv(128, 64, 3, 2)
        self.dec2 = UpsampleConv(64,  32, 3, 2)
        self.dec3 = ConvLayer(32,  3,  9, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.res(x)
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        return torch.sigmoid(self.dec3(x))

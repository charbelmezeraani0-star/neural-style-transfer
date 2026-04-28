"""
Perceptual loss network built on top of frozen VGG-19.

Two components:
  • Content loss  — MSE between VGG feature maps of output vs. content image
  • Style loss    — MSE between Gram matrices of VGG features vs. style image
  • TV loss       — Total-variation regulariser for spatial smoothness

VGG layer naming follows the convention used in the original paper:
  relu1_2, relu2_2, relu3_3, relu4_3
"""

import torch
import torch.nn as nn
import torchvision.models as models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which VGG ReLU layers to use for style and content
STYLE_LAYERS   = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
CONTENT_LAYERS = ["relu3_3"]


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Normalised Gram matrix for a (B, C, H, W) feature map."""
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (c * h * w)


class VGGFeatures(nn.Module):
    """
    Extracts intermediate feature maps from a frozen VGG-19.
    Only runs up to the deepest required layer for efficiency.
    """

    # Map readable names → VGG Sequential indices
    LAYER_MAP = {
        "relu1_1":  1,  "relu1_2":  3,
        "relu2_1":  6,  "relu2_2":  8,
        "relu3_1": 11,  "relu3_2": 13,  "relu3_3": 15,  "relu3_4": 17,
        "relu4_1": 20,  "relu4_2": 22,  "relu4_3": 24,  "relu4_4": 26,
        "relu5_1": 29,
    }

    def __init__(self, layers: list[str]):
        super().__init__()
        self.target_layers = set(layers)
        max_idx = max(self.LAYER_MAP[l] for l in layers)

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[: max_idx + 1])

        # Freeze — we never update VGG weights
        for p in self.slice.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        h = x
        for idx, layer in enumerate(self.slice):
            h = layer(h)
            for name, layer_idx in self.LAYER_MAP.items():
                if layer_idx == idx and name in self.target_layers:
                    features[name] = h
        return features


class PerceptualLoss(nn.Module):
    """
    Combined content + style + TV loss.

    Usage
    -----
    loss_fn = PerceptualLoss(style_image_tensor).to(DEVICE)
    total_loss, info = loss_fn(output_tensor, content_tensor)
    """

    def __init__(
        self,
        style_img: torch.Tensor,
        style_layers:   list[str] = STYLE_LAYERS,
        content_layers: list[str] = CONTENT_LAYERS,
        style_weight:   float = 1e5,
        content_weight: float = 1.0,
        tv_weight:      float = 1e-6,
    ):
        super().__init__()
        self.style_weight   = style_weight
        self.content_weight = content_weight
        self.tv_weight      = tv_weight

        all_layers = list(set(style_layers + content_layers))
        self.vgg   = VGGFeatures(all_layers).to(DEVICE)

        self.style_layers   = style_layers
        self.content_layers = content_layers

        # Pre-compute style Gram matrices (fixed for the whole training run)
        with torch.no_grad():
            style_feats = self.vgg(style_img.to(DEVICE))
        self.style_grams = {
            l: gram_matrix(style_feats[l]).detach()
            for l in style_layers
        }

    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation — penalises high-frequency noise."""
        return (
            torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )

    def forward(
        self,
        output: torch.Tensor,
        content: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        out_feats     = self.vgg(output)
        content_feats = self.vgg(content)

        # Content loss
        c_loss = sum(
            nn.functional.mse_loss(out_feats[l], content_feats[l].detach())
            for l in self.content_layers
        )

        # Style loss — compare Gram matrices to the precomputed style grams
        s_loss = torch.tensor(0.0, device=DEVICE)
        for l in self.style_layers:
            out_gram = gram_matrix(out_feats[l])
            # style_grams[l] has batch size 1; expand to match output batch
            target   = self.style_grams[l].expand(out_gram.size(0), -1, -1)
            s_loss  += nn.functional.mse_loss(out_gram, target)

        tv_loss = self._tv_loss(output)

        total = (
            self.content_weight * c_loss +
            self.style_weight   * s_loss +
            self.tv_weight      * tv_loss
        )

        return total, {
            "content": c_loss.item(),
            "style":   s_loss.item(),
            "tv":      tv_loss.item(),
        }

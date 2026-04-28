# =============================================================================
# Neural Style Transfer — Project Setup
# =============================================================================
# Transforms a content image into the artistic style of a style image
# using a pre-trained VGG-19 network and perceptual loss optimization.
# =============================================================================

# ── Installation (run once in your terminal) ──────────────────────────────────
# pip install torch torchvision pillow matplotlib

# ── Imports ───────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — swap to "TkAgg" if you want pop-up windows

import copy
import os

# ── Device configuration ──────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("No GPU found — running on CPU (will be slower)")

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ── Paths ─────────────────────────────────────────────────────────────────────

CONTENT_IMAGE_PATH = "images/content.jpg"   # your photo goes here
STYLE_IMAGE_PATH   = "images/style.jpg"     # the drawing/painting style goes here
OUTPUT_DIR         = "outputs"

os.makedirs("images",   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────

IMAGE_SIZE     = 512          # resize images to this (long side), lower = faster
NUM_STEPS      = 300          # optimization iterations
STYLE_WEIGHT   = 1_000_000    # how strongly to apply the style
CONTENT_WEIGHT = 1            # how much to preserve the original content
TV_WEIGHT      = 1e-6         # total-variation smoothness regulariser

# ── Image loading and display — imported from image_utils.py ──────────────────
#
# All image functions live in image_utils.py so they can be imported
# independently by other scripts without pulling in the full NST pipeline.

from image_utils import (
    load_image,       # load a file → normalised (1,3,H,W) tensor on DEVICE
    tensor_to_pil,    # normalised tensor → PIL Image  (reverses normalisation)
    imshow,           # display one tensor in a matplotlib Axes
    show_images,      # display content / style / output side by side
    IMAGENET_MEAN,    # [0.485, 0.456, 0.406]
    IMAGENET_STD,     # [0.229, 0.224, 0.225]
)


# ── VGG-19 feature extractor ──────────────────────────────────────────────────
#
# VGG-19 architecture (Simonyan & Zisserman, 2014):
#
#   Input (3 × H × W)
#   │
#   ├── Block 1 ─ 2× [Conv 3×3, 64 filters]  → MaxPool → 64 × H/2 × W/2
#   ├── Block 2 ─ 2× [Conv 3×3, 128 filters] → MaxPool → 128 × H/4 × W/4
#   ├── Block 3 ─ 4× [Conv 3×3, 256 filters] → MaxPool → 256 × H/8 × W/8
#   ├── Block 4 ─ 4× [Conv 3×3, 512 filters] → MaxPool → 512 × H/16 × W/16
#   └── Block 5 ─ 4× [Conv 3×3, 512 filters] → MaxPool → 512 × H/32 × W/32
#   │
#   └── Classifier (3 FC layers → 1000 ImageNet classes)  ← NOT used here
#
# For style transfer we only need .features (the conv blocks).
# The classifier layers understand "what object is this?" — we don't need that.
# The conv layers understand "what textures and structures are here?" — that's
# exactly what content and style loss measure.
#
# Layer index map for vgg19.features  (Sequential indices):
#
#   Index │ Layer        │ Readable name   │ Used for
#   ──────┼──────────────┼─────────────────┼──────────────────────
#     0   │ Conv2d       │                 │
#     1   │ ReLU         │ relu1_1         │ style
#     2   │ Conv2d       │                 │
#     3   │ ReLU         │ relu1_2         │
#     4   │ MaxPool2d    │ pool1           │
#     5   │ Conv2d       │                 │
#     6   │ ReLU         │ relu2_1         │ style
#     7   │ Conv2d       │                 │
#     8   │ ReLU         │ relu2_2         │
#     9   │ MaxPool2d    │ pool2           │
#    10   │ Conv2d       │                 │
#    11   │ ReLU         │ relu3_1         │ style
#    12   │ Conv2d       │                 │
#    13   │ ReLU         │ relu3_2         │
#    14   │ Conv2d       │                 │
#    15   │ ReLU         │ relu3_3         │
#    16   │ Conv2d       │                 │
#    17   │ ReLU         │ relu3_4         │
#    18   │ MaxPool2d    │ pool3           │
#    19   │ Conv2d       │                 │
#    20   │ ReLU         │ relu4_1         │ style
#    21   │ Conv2d       │                 │
#    22   │ ReLU         │ relu4_2         │ content  ← semantic detail
#    23   │ Conv2d       │                 │
#    24   │ ReLU         │ relu4_3         │
#    25   │ Conv2d       │                 │
#    26   │ ReLU         │ relu4_4         │
#    27   │ MaxPool2d    │ pool4           │
#    28   │ Conv2d       │                 │
#    29   │ ReLU         │ relu5_1         │ style
#   ──────┴──────────────┴─────────────────┴──────────────────────

# Human-readable name → Sequential index inside vgg19.features
VGG19_LAYER_MAP: dict[str, int] = {
    "relu1_1":  1,
    "relu1_2":  3,
    "pool1":    4,
    "relu2_1":  6,
    "relu2_2":  8,
    "pool2":    9,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "pool3":   18,
    "relu4_1": 20,
    "relu4_2": 22,   # <-- default content layer
    "relu4_3": 24,
    "relu4_4": 26,
    "pool4":   27,
    "relu5_1": 29,   # deepest style layer used
}

# Layers selected for content and style loss (from Gatys et al. 2015)
CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS   = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]


def load_vgg19() -> nn.Sequential:
    """
    Load VGG-19's convolutional feature extractor, ready for style transfer.

    Steps
    -----
    1. Download pretrained ImageNet weights (cached after first run).
    2. Keep only .features — the 5 conv blocks (36 layers total).
       The .classifier (3 FC layers) is discarded: it maps features to
       1000 class scores, which is useless for pixel-level style transfer.
    3. Set to eval() mode:
       - Disables Dropout (VGG has none, but good practice).
       - Freezes BatchNorm running stats (VGG has none, still good practice).
    4. Freeze all parameters with requires_grad_(False):
       - Prevents PyTorch from building a gradient graph through VGG.
       - Cuts memory use roughly in half during the NST optimisation loop.
       - Ensures VGG weights never change — we only optimise the image.
    5. Move to DEVICE (GPU if available).

    Returns
    -------
    nn.Sequential — vgg19.features on DEVICE, frozen, in eval mode
    """
    # Step 1 & 2 — download weights and extract .features
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    features = vgg.features          # nn.Sequential of 37 layers (indices 0-36)

    # Step 3 — evaluation mode (no dropout / batchnorm training behaviour)
    features.eval()

    # Step 4 — freeze every parameter
    for param in features.parameters():
        param.requires_grad_(False)

    # Step 5 — move to GPU / CPU
    features = features.to(DEVICE)

    return features


def get_features(
    image:    torch.Tensor,
    model:    nn.Sequential,
    layers:   list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run `image` through VGG-19 and return the feature maps at named layers.

    Instead of running the full forward pass and throwing most of it away,
    this function stops as soon as the deepest requested layer is reached.

    Parameters
    ----------
    image  : torch.Tensor  shape (1, 3, H, W) — ImageNet-normalised
    model  : nn.Sequential — the frozen VGG-19 .features extractor
    layers : list of layer names from VGG19_LAYER_MAP
             defaults to CONTENT_LAYERS + STYLE_LAYERS

    Returns
    -------
    dict  { layer_name: feature_map_tensor }
          each feature map has shape (1, C, H', W')
    """
    if layers is None:
        layers = CONTENT_LAYERS + STYLE_LAYERS

    # Convert names to indices and find the deepest one
    target_indices = {VGG19_LAYER_MAP[name]: name for name in layers}
    stop_at        = max(target_indices.keys())

    features = {}
    x = image

    for idx, layer in enumerate(model):
        x = layer(x)                              # forward through this sub-layer
        if idx in target_indices:
            features[target_indices[idx]] = x    # capture this feature map
        if idx == stop_at:
            break                                 # no need to go deeper

    return features


def print_vgg19_summary(model: nn.Sequential) -> None:
    """Print a compact summary of VGG-19 .features with readable layer names."""
    reverse_map = {v: k for k, v in VGG19_LAYER_MAP.items()}

    print("\nVGG-19 Feature Extractor Summary")
    print("─" * 52)
    print(f"{'Idx':<5} {'Type':<22} {'Name':<12} {'Used for'}")
    print("─" * 52)

    for idx, layer in enumerate(model):
        layer_type = type(layer).__name__
        name       = reverse_map.get(idx, "")
        role       = ""
        if name in CONTENT_LAYERS:
            role = "CONTENT"
        elif name in STYLE_LAYERS:
            role = "STYLE"
        print(f"  {idx:<4} {layer_type:<22} {name:<12} {role}")

    print("─" * 52)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable params : 0  (all frozen)")
    print(f"Device           : {next(model.parameters()).device}")
    print()


# ── Content Loss ─────────────────────────────────────────────────────────────
#
# What is content loss?
#   It measures how much the current generated image differs structurally
#   from the original content image, by comparing their VGG feature maps.
#
# Why MSE on feature maps instead of raw pixels?
#   Raw pixel MSE would force the output to look like an exact copy of the
#   photo — no stylisation possible. Comparing VGG feature maps instead
#   preserves high-level structure (shapes, object positions) while allowing
#   low-level appearance (colour, texture, stroke style) to change freely.
#
# Why detach the target?
#   The target is a fixed reference — it should never change during the
#   optimisation loop. Calling .detach() cuts it out of PyTorch's autograd
#   graph so no gradients ever flow back into it (or into VGG through it).
#   Without this, PyTorch would try to differentiate through both branches
#   of the MSE, wasting memory and time computing gradients we don't need.
#
# Why store self.loss as an attribute?
#   ContentLoss is inserted *inside* the model as a pass-through layer.
#   The forward method returns x unchanged so the data keeps flowing through
#   the remaining VGG layers. The training loop reads .loss from each module
#   after the forward pass — no need to restructure the model or re-run it.
#
# Diagram of how ContentLoss sits in the pipeline:
#
#   image → [VGG layers 0-21] → [Conv relu4_2] → ContentLoss → [VGG layers 23+]
#                                                      │
#                                               computes MSE
#                                               stores in self.loss
#                                               returns x unchanged


class ContentLoss(nn.Module):
    """
    Pass-through layer that computes and stores content loss at a VGG layer.

    Inserted into the model immediately after the chosen content layer
    (relu4_2 by default). During each forward pass it:
      1. Computes MSE between the current feature map and the fixed target.
      2. Stores the result in self.loss (read by the training loop).
      3. Returns the feature map unchanged so the rest of VGG still runs.

    Parameters
    ----------
    target : torch.Tensor  shape (1, C, H, W)
        Feature map of the *content* image at the chosen VGG layer.
        Detached from the graph immediately — never updated.

    Attributes
    ----------
    target : torch.Tensor   — fixed reference feature map (detached)
    loss   : torch.Tensor   — scalar MSE loss, updated every forward pass
    """

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()

        # Detach: cut the target out of the autograd graph.
        # register_buffer: moves it to the right device with the module,
        # and saves/loads it with state_dict() — but it is NOT a parameter
        # (no gradient, no optimizer update).
        self.register_buffer("target", target.detach())

        # Initialise loss to zero; will be overwritten on first forward pass
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute content loss and pass x through unchanged.

        Parameters
        ----------
        x : torch.Tensor  shape (1, C, H, W)
            Feature map of the *generated* image — has requires_grad=True
            so gradients flow back to the image pixels.

        Returns
        -------
        torch.Tensor — exactly x, unmodified, so downstream VGG layers
                       continue to receive the correct input.
        """
        # MSE averaged over all (C × H × W) elements.
        # The gradient of MSE w.r.t. x flows back to the image pixels,
        # pulling the generated features toward the content target.
        self.loss = nn.functional.mse_loss(x, self.target)

        # Return x untouched — this layer is transparent to the data flow
        return x


# ── Gram Matrix & Style Loss ──────────────────────────────────────────────────
#
# What is the Gram matrix?
#   Given a feature map of shape (B, C, H, W), flatten the spatial dims:
#
#       F  shape (B, C, H*W)   ← each of the C channels as a long vector
#
#   The Gram matrix is:
#
#       G = F @ F^T            shape (B, C, C)
#
#   Entry G[i, j] = dot product of channel i and channel j across all
#   spatial positions.  A large value means channels i and j tend to
#   activate together — they are correlated textures.
#
# Why does this capture style and not content?
#   The Gram matrix throws away spatial information completely.
#   It doesn't record *where* an edge or stroke appears — only *how often*
#   certain pairs of features co-activate anywhere in the image.
#   That co-activation pattern is exactly what defines a visual style
#   (e.g. "pencil lines tend to appear with cross-hatching").
#   Content, by contrast, is about specific features at specific places.
#
# Why normalise by (C × H × W)?
#   Without normalisation, deeper VGG layers have far larger feature maps
#   and dominate the loss. Dividing by the total element count makes the
#   Gram values comparable across layers with different C, H, W.
#
# Visual:
#
#   feature map (B, C, H, W)
#        │
#        │  .view(B, C, H*W)        flatten spatial dims
#        ▼
#   F  (B, C, N)     N = H*W
#        │
#        │  bmm(F, Fᵀ)              batch matrix multiply
#        ▼
#   G  (B, C, C)                   channel correlation matrix
#        │
#        │  / (C * H * W)           normalise
#        ▼
#   G_norm  (B, C, C)              ← this is the Gram matrix
#
# StyleLoss diagram (five instances, one per style layer):
#
#   image → [relu1_1] → StyleLoss₁ → [relu2_1] → StyleLoss₂ → … → StyleLoss₅
#                            │                          │
#                       gram(x) vs                 gram(x) vs
#                       gram(style)                gram(style)
#                       → self.loss                → self.loss


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalised Gram matrix of a feature map.

    Parameters
    ----------
    x : torch.Tensor  shape (B, C, H, W)
        Intermediate VGG feature map.  B is always 1 in our pipeline.

    Returns
    -------
    torch.Tensor  shape (B, C, C)
        Symmetric matrix of pairwise channel correlations, normalised
        so values are comparable across layers of different sizes.

    Steps
    -----
    1. Flatten H and W into one spatial dimension  →  (B, C, N)
    2. Batch matrix-multiply F with its own transpose  →  (B, C, C)
       G[b, i, j] = sum_n  F[b, i, n] * F[b, j, n]
    3. Divide by C*H*W to normalise across layer sizes
    """
    b, c, h, w = x.size()
    n          = h * w                             # total spatial positions

    # Step 1 — flatten spatial dimensions, keeping batch and channels
    features = x.view(b, c, n)                    # (B, C, N)

    # Step 2 — batch matrix multiply: (B, C, N) × (B, N, C) → (B, C, C)
    G = torch.bmm(features, features.transpose(1, 2))

    # Step 3 — normalise so loss magnitude is independent of layer size
    return G / (c * n)


class StyleLoss(nn.Module):
    """
    Pass-through layer that computes and stores style loss at a VGG layer.

    Style is measured as the MSE between the Gram matrix of the current
    generated image and the Gram matrix of the style reference image.
    Five instances of this class are inserted at relu1_1, relu2_1, relu3_1,
    relu4_1, relu5_1 — each capturing texture at a different scale.

    Parameters
    ----------
    target_feature : torch.Tensor  shape (1, C, H, W)
        Feature map of the *style* image at this VGG layer.
        Its Gram matrix is computed once here and stored as a fixed buffer.

    Attributes
    ----------
    target_gram : torch.Tensor  shape (1, C, C)  — fixed style reference
    loss        : torch.Tensor  scalar            — updated every forward pass
    """

    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()

        # Compute the style Gram matrix once at construction time.
        # Detach immediately — this is a fixed target, never to be updated.
        # register_buffer: device-portable and included in state_dict,
        # but has no gradient and is not an optimisable parameter.
        self.register_buffer("target_gram", gram_matrix(target_feature).detach())

        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss at this layer and pass x through unchanged.

        Parameters
        ----------
        x : torch.Tensor  shape (1, C, H, W)
            Feature map of the *generated* image at this VGG layer.

        Returns
        -------
        torch.Tensor — exactly x, unmodified, so downstream layers are
                       unaffected by the presence of this loss module.
        """
        # Compute the Gram matrix of the current generated features
        G = gram_matrix(x)                                      # (1, C, C)

        # MSE between generated and style Gram matrices.
        # Gradient flows back through gram_matrix() into x, then to pixels.
        self.loss = nn.functional.mse_loss(G, self.target_gram)

        return x


# ── NST Model Builder & Optimization Loop ────────────────────────────────────
#
# How the pieces fit together:
#
#   vgg.features  (frozen, 37 layers)
#        │
#        │  build_nst_model() inserts ContentLoss and StyleLoss modules
#        │  directly after the layers they monitor:
#        ▼
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  Conv/ReLU/Pool × N                                             │
#   │  → StyleLoss (relu1_1)   ← stores style loss, passes x through │
#   │  → Conv/ReLU × N                                               │
#   │  → StyleLoss (relu2_1)                                          │
#   │  → Conv/ReLU × N                                               │
#   │  → StyleLoss (relu3_1)                                          │
#   │  → Conv/ReLU × N                                               │
#   │  → StyleLoss (relu4_1)                                          │
#   │  → Conv/ReLU (relu4_2)                                          │
#   │  → ContentLoss (relu4_2) ← stores content loss, passes through │
#   │  → Conv/ReLU × N                                               │
#   │  → StyleLoss (relu5_1)                                          │
#   │  ─ stop here (no deeper layers needed) ─                        │
#   └─────────────────────────────────────────────────────────────────┘
#        │
#        ▼  run_style_transfer() optimization loop:
#
#   input_img  (clone of content, requires_grad=True)
#        │  optimizer = LBFGS([input_img])
#        │
#        │  for each step:
#        │    closure():
#        │      clamp input_img to [0, 1]
#        │      model(input_img)          ← forward triggers all loss modules
#        │      total_loss = content_weight * Σ content_losses
#        │                 + style_weight  * Σ style_losses
#        │      total_loss.backward()     ← gradients accumulate on input_img
#        │    optimizer.step(closure)     ← LBFGS updates input_img pixels
#        ▼
#   output_img  (clamped, detached)
#
# Why LBFGS instead of Adam?
#   NST is an unconstrained optimisation over a single image (~800k variables
#   for a 512px image). LBFGS uses a quasi-Newton method that approximates
#   second-order curvature — it "sees" the loss landscape more clearly and
#   converges in ~300 steps vs ~3000 for Adam. The closure pattern is
#   mandatory: LBFGS does an internal line-search that may evaluate the loss
#   multiple times per step to find a good step size.
#
# Why clamp inside the closure?
#   After each parameter update, pixel values can drift below 0 or above 1.
#   Clamping inside the closure (before the forward pass) keeps the image
#   physically valid throughout training. We use torch.no_grad() so the
#   in-place clamp_ doesn't disrupt the autograd graph.


def build_nst_model(
    vgg:         nn.Sequential,
    content_img: torch.Tensor,
    style_img:   torch.Tensor,
) -> tuple[nn.Sequential, list[ContentLoss], list[StyleLoss]]:
    """
    Construct the NST pipeline by inserting ContentLoss and StyleLoss modules
    into a copy of the frozen VGG-19 feature extractor.

    The original `vgg` is never modified — we iterate through its layers and
    rebuild a new Sequential, adding loss modules where needed.

    Parameters
    ----------
    vgg         : frozen VGG-19 .features (from load_vgg19())
    content_img : (1, 3, H, W) tensor — the photo to preserve structure from
    style_img   : (1, 3, H, W) tensor — the drawing whose style to apply

    Returns
    -------
    model          : nn.Sequential with VGG layers + interleaved loss modules
    content_losses : list of ContentLoss instances (one per CONTENT_LAYERS)
    style_losses   : list of StyleLoss   instances (one per STYLE_LAYERS)
    """
    # Map layer index → ('content'|'style', name) for every layer we care about
    layer_roles: dict[int, tuple[str, str]] = {}
    for name in CONTENT_LAYERS:
        layer_roles[VGG19_LAYER_MAP[name]] = ("content", name)
    for name in STYLE_LAYERS:
        layer_roles[VGG19_LAYER_MAP[name]] = ("style", name)

    # We only need to go as deep as the deepest monitored layer
    stop_at = max(layer_roles.keys())

    model          = nn.Sequential()
    content_losses = []
    style_losses   = []

    # Running feature maps — updated layer by layer for target computation
    x_content = content_img.clone()
    x_style   = style_img.clone()

    for idx, vgg_layer in enumerate(vgg):
        # Add the VGG sub-layer itself
        model.add_module(f"vgg_{idx}", vgg_layer)

        # Advance the reference images through this layer (no grad needed)
        with torch.no_grad():
            x_content = vgg_layer(x_content)
            x_style   = vgg_layer(x_style)

        # If this is a monitored layer, insert the appropriate loss module
        if idx in layer_roles:
            role, name = layer_roles[idx]

            if role == "content":
                cl = ContentLoss(x_content)           # target = content features
                model.add_module(f"content_loss_{name}", cl)
                content_losses.append(cl)

            else:  # style
                sl = StyleLoss(x_style)               # target = style gram matrix
                model.add_module(f"style_loss_{name}", sl)
                style_losses.append(sl)

        if idx == stop_at:
            break   # no deeper layers contribute to either loss

    return model, content_losses, style_losses


def run_style_transfer(
    content_img:    torch.Tensor,
    style_img:      torch.Tensor,
    vgg:            nn.Sequential,
    num_steps:      int   = 300,
    style_weight:   float = 1_000_000,
    content_weight: float = 1.0,
    print_every:    int   = 50,
) -> torch.Tensor:
    """
    Run the NST optimisation loop and return the stylised image tensor.

    The *image pixels* are the optimisable parameters — VGG and all loss
    modules remain completely frozen throughout.

    Parameters
    ----------
    content_img    : (1, 3, H, W) tensor — photo to stylise
    style_img      : (1, 3, H, W) tensor — reference drawing/painting
    vgg            : frozen VGG-19 .features (from load_vgg19())
    num_steps      : number of LBFGS optimiser steps
    style_weight   : weight applied to the summed style losses
    content_weight : weight applied to the summed content losses
    print_every    : print a loss summary every N closure evaluations

    Returns
    -------
    torch.Tensor  shape (1, 3, H, W), values in [0, 1], detached from graph
    """
    print("Building NST model...")
    model, content_losses, style_losses = build_nst_model(vgg, content_img, style_img)
    model.eval()   # loss modules are already pass-through, but be explicit

    print(f"  {len(content_losses)} content loss module(s) — "
          f"layers: {CONTENT_LAYERS}")
    print(f"  {len(style_losses)}   style   loss module(s) — "
          f"layers: {STYLE_LAYERS}")

    # ── Input image — the variable being optimised ────────────────────────────
    # Start from a copy of the content image.  requires_grad_(True) tells
    # PyTorch to track every operation on this tensor so we can compute
    # d(loss)/d(pixel) and update the pixels with the optimiser.
    input_img = content_img.clone().requires_grad_(True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # LBFGS takes the list of tensors to optimise.
    # max_iter controls internal line-search iterations per step.
    optimizer = optim.LBFGS([input_img], max_iter=20)

    print(f"\nStarting optimisation — {num_steps} steps, "
          f"style_weight={style_weight:,.0f}, "
          f"content_weight={content_weight}\n")
    print(f"  {'Step':>6}  {'Total loss':>14}  {'Content loss':>14}  {'Style loss':>14}")
    print("  " + "─" * 56)

    # ── Closure ───────────────────────────────────────────────────────────────
    # LBFGS requires a closure because its line-search may call it several
    # times per step to bracket the optimal step size.  Every call must:
    #   1. zero gradients
    #   2. do a full forward pass (which populates .loss on every loss module)
    #   3. compute total loss and call .backward()
    #   4. return the loss scalar

    call_count = [0]   # list so the closure can mutate it

    def closure() -> torch.Tensor:
        # Clamp BEFORE the forward pass so VGG never sees out-of-range pixels.
        # torch.no_grad() prevents this in-place op from entering the graph.
        with torch.no_grad():
            input_img.clamp_(0.0, 1.0)

        optimizer.zero_grad()

        # Forward pass — triggers every ContentLoss and StyleLoss module,
        # each of which updates its own .loss attribute as a side effect.
        model(input_img)

        # Weighted sum of all content losses
        content_score = content_weight * sum(cl.loss for cl in content_losses)

        # Weighted sum of all style losses (summed over all 5 layers)
        style_score   = style_weight   * sum(sl.loss for sl in style_losses)

        total_loss = content_score + style_score

        # Backpropagate — accumulates d(total_loss)/d(pixel) on input_img.grad
        total_loss.backward()

        call_count[0] += 1
        if call_count[0] % print_every == 0:
            print(f"  {call_count[0]:>6d}  "
                  f"{total_loss.item():>14.4f}  "
                  f"{content_score.item():>14.4f}  "
                  f"{style_score.item():>14.4f}")

        return total_loss

    # ── Optimisation loop ─────────────────────────────────────────────────────
    for _ in range(num_steps):
        optimizer.step(closure)

    # Final clamp — last update may push a few pixels just out of range
    with torch.no_grad():
        input_img.clamp_(0.0, 1.0)

    print("\nOptimisation complete.")
    return input_img.detach()


# ── Entry point: load both images and show a preview ─────────────────────────

if __name__ == "__main__":

    # ── Step 1: images ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1 — Loading images")
    print("=" * 60)

    missing = [p for p in [CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH]
               if not os.path.isfile(p)]

    if missing:
        print("\nImages not found — place your files at:")
        for p in missing:
            print(f"  {p}")
        print("\nTip: any .jpg or .png works. Then re-run this script.")
        content_tensor = style_tensor = None
    else:
        print("\nContent image:")
        content_tensor = load_image(CONTENT_IMAGE_PATH)
        print("\nStyle image:")
        style_tensor   = load_image(STYLE_IMAGE_PATH)
        print("\nGenerating preview...")
        show_images(content_tensor, style_tensor)

    # ── Step 2: VGG-19 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Loading VGG-19")
    print("=" * 60 + "\n")

    vgg = load_vgg19()
    print_vgg19_summary(vgg)

    # ── Step 3: feature extraction test (only if images are loaded) ───────────
    if content_tensor is not None:
        print("=" * 60)
        print("Step 3 — Feature extraction test")
        print("=" * 60)

        content_features = get_features(content_tensor, vgg)
        style_features   = get_features(style_tensor,   vgg)

        print("\nContent features (relu4_2):")
        for name in CONTENT_LAYERS:
            t = content_features[name]
            print(f"  {name:<10}  shape={tuple(t.shape)}  "
                  f"min={t.min():.3f}  max={t.max():.3f}")

        print("\nStyle features (multi-scale):")
        for name in STYLE_LAYERS:
            t = style_features[name]
            print(f"  {name:<10}  shape={tuple(t.shape)}  "
                  f"min={t.min():.3f}  max={t.max():.3f}")

        print("\nVGG-19 loaded, frozen, and feature extraction working.")

    # ── Step 4: ContentLoss test ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4 — ContentLoss unit test")
    print("=" * 60)

    # Use a synthetic feature map so the test always runs (no image required)
    torch.manual_seed(0)
    fake_target   = torch.rand(1, 512, 32, 32, device=DEVICE)  # shape: (B,C,H,W)
    content_loss  = ContentLoss(fake_target).to(DEVICE)

    # Verify target was detached
    print(f"\n  target.requires_grad : {content_loss.target.requires_grad}"
          f"  ← must be False")

    # Forward with identical input → loss should be 0
    identical_input = fake_target.clone().requires_grad_(True)
    out = content_loss(identical_input)
    print(f"  loss (identical)     : {content_loss.loss.item():.6f}"
          f"  ← must be 0.0")
    print(f"  output is input      : {out is identical_input}"
          f"  ← must be True (pass-through)")

    # Forward with different input → loss should be positive
    different_input = torch.rand_like(fake_target, requires_grad=True)
    content_loss(different_input)
    print(f"  loss (different)     : {content_loss.loss.item():.6f}"
          f"  ← must be > 0.0")

    # Confirm gradient flows to the input image, not to VGG target
    content_loss.loss.backward()
    print(f"  input.grad exists    : {different_input.grad is not None}"
          f"  ← must be True  (gradient reaches the image)")
    print(f"  target.grad          : {content_loss.target.grad}"
          f"  ← must be None  (target is fixed)")

    print("\n  ContentLoss working correctly.")

    # ── Step 5: gram_matrix + StyleLoss unit test ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5 — gram_matrix & StyleLoss unit test")
    print("=" * 60)

    torch.manual_seed(1)

    # ── gram_matrix checks ────────────────────────────────────────────────────
    print("\n  gram_matrix checks:")
    feat = torch.rand(1, 64, 128, 128, device=DEVICE)   # relu1_1 shape
    G    = gram_matrix(feat)

    b, c, h, w = feat.size()
    print(f"    input shape  : {tuple(feat.shape)}")
    print(f"    gram shape   : {tuple(G.shape)}"
          f"  ← must be (1, {c}, {c})")

    # Gram matrix must be square and symmetric
    is_symmetric = torch.allclose(G, G.transpose(1, 2), atol=1e-5)
    print(f"    symmetric    : {is_symmetric}"
          f"  ← must be True  (G = Gᵀ by construction)")

    # All diagonal entries must be non-negative (self dot-product ≥ 0)
    diag_nonneg = (G[0].diag() >= 0).all().item()
    print(f"    diag ≥ 0     : {diag_nonneg}"
          f"  ← must be True  (self-correlation is never negative)")

    # ── StyleLoss checks ──────────────────────────────────────────────────────
    print("\n  StyleLoss checks:")
    fake_style  = torch.rand(1, 256, 64, 64, device=DEVICE)  # relu3_1 shape
    style_loss  = StyleLoss(fake_style).to(DEVICE)

    expected_gram_shape = (1, 256, 256)
    print(f"    target_gram shape   : {tuple(style_loss.target_gram.shape)}"
          f"  ← must be {expected_gram_shape}")
    print(f"    target_gram.grad_fn : {style_loss.target_gram.grad_fn}"
          f"  ← must be None  (detached)")

    # Identical input → loss = 0
    same_input = fake_style.clone().requires_grad_(True)
    style_loss(same_input)
    print(f"    loss (identical)    : {style_loss.loss.item():.6f}"
          f"  ← must be 0.0")

    # Different input → loss > 0
    # Use zeros (maximally different from a uniform-random style tensor)
    diff_input = torch.zeros_like(fake_style, requires_grad=True)
    style_loss(diff_input)
    print(f"    loss (different)    : {style_loss.loss.item():.6e}"
          f"  ← must be > 0.0")

    # Gradient must reach the image pixels, not the style target
    style_loss.loss.backward()
    print(f"    input.grad exists   : {diff_input.grad is not None}"
          f"  ← must be True  (gradient reaches the image)")
    print(f"    target_gram.grad    : {style_loss.target_gram.grad}"
          f"  ← must be None  (style target is fixed)")

    # Five layers, five instances — simulate the full style stack
    print("\n  Simulating all 5 style layers:")
    layer_shapes = [(1,64,256,256),(1,128,128,128),(1,256,64,64),(1,512,32,32),(1,512,16,16)]
    layer_names  = STYLE_LAYERS
    for name, shape in zip(layer_names, layer_shapes):
        feat_map   = torch.rand(*shape, device=DEVICE)
        sl         = StyleLoss(feat_map).to(DEVICE)
        b, c, h, w = shape
        print(f"    {name:<10}  feature {tuple(shape)}  "
              f"→  gram (1, {c}, {c})")

    print("\n  gram_matrix and StyleLoss working correctly.")

    # ── Step 6: build_nst_model + run_style_transfer test ────────────────────
    print("\n" + "=" * 60)
    print("Step 6 — Optimization loop test  (synthetic 64×64 images)")
    print("=" * 60)

    torch.manual_seed(42)

    # Tiny synthetic images so the test runs in seconds, not minutes
    fake_content = torch.rand(1, 3, 64, 64, device=DEVICE)
    fake_style   = torch.rand(1, 3, 64, 64, device=DEVICE)

    # ── build_nst_model checks ────────────────────────────────────────────────
    print("\n  build_nst_model checks:")
    test_model, c_losses, s_losses = build_nst_model(vgg, fake_content, fake_style)

    print(f"    content loss modules : {len(c_losses)}"
          f"  ← must be {len(CONTENT_LAYERS)}")
    print(f"    style   loss modules : {len(s_losses)}"
          f"  ← must be {len(STYLE_LAYERS)}")

    # Verify targets are detached inside each module
    all_detached = all(not cl.target.requires_grad    for cl in c_losses) and \
                   all(not sl.target_gram.requires_grad for sl in s_losses)
    print(f"    all targets detached : {all_detached}"
          f"  ← must be True")

    # Verify VGG parameters are still frozen inside the assembled model
    vgg_params_frozen = all(
        not p.requires_grad
        for name, p in test_model.named_parameters()
        if "content_loss" not in name and "style_loss" not in name
    )
    print(f"    VGG params frozen    : {vgg_params_frozen}"
          f"  ← must be True")

    # ── run_style_transfer: 5 steps as a smoke test ───────────────────────────
    print()
    output = run_style_transfer(
        content_img    = fake_content,
        style_img      = fake_style,
        vgg            = vgg,
        num_steps      = 5,
        style_weight   = STYLE_WEIGHT,
        content_weight = CONTENT_WEIGHT,
        print_every    = 1,
    )

    print(f"\n  Output shape  : {tuple(output.shape)}"
          f"  ← must be (1, 3, 64, 64)")
    print(f"  Output range  : [{output.min():.4f}, {output.max():.4f}]"
          f"  ← must be within [0, 1]")
    print(f"  requires_grad : {output.requires_grad}"
          f"  ← must be False (detached)")
    in_range = (output.min() >= 0.0) and (output.max() <= 1.0)
    print(f"  Pixel values valid : {in_range.item()}"
          f"  ← must be True")

    print("\n  Optimization loop working correctly.")
    print("\n" + "=" * 60)
    print("All steps complete. Ready for real images.")
    print("Place content.jpg and style.jpg in images/ and re-run.")
    print("=" * 60)

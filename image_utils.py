# =============================================================================
# image_utils.py — Image loading, conversion, and display utilities
# =============================================================================
# Standalone module used by neural_style_transfer.py.
# Import with:  from image_utils import load_image, tensor_to_pil, show_images
# =============================================================================

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalisation constants ──────────────────────────────────────────
#
#   VGG-19 was pre-trained on ImageNet.  Every image passed through it must
#   be normalised with the same per-channel statistics used during that
#   training — otherwise the feature activations are completely off.
#
#   Formula applied to each pixel:
#       normalised = (pixel_0_to_1 - mean) / std
#
#   Inverse (to recover a viewable image):
#       pixel_0_to_1 = normalised * std + mean

IMAGENET_MEAN = [0.485, 0.456, 0.406]   # mean for R, G, B channels
IMAGENET_STD  = [0.229, 0.224, 0.225]   # std  for R, G, B channels

# ── Reusable transform pipeline ───────────────────────────────────────────────

def _make_transform(max_size: int) -> transforms.Compose:
    """Build a transform: resize → tensor → ImageNet normalise."""
    return transforms.Compose([
        transforms.Resize(max_size),          # resize shortest side to max_size
        transforms.CenterCrop(max_size),      # square-crop so H == W == max_size
        transforms.ToTensor(),                # PIL uint8 [0,255] → float [0,1]
        transforms.Normalize(                 # subtract mean, divide by std
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

# =============================================================================
# load_image
# =============================================================================

def load_image(path: str, max_size: int = 512) -> torch.Tensor:
    """
    Load a content or style image from disk.

    Parameters
    ----------
    path     : str   — absolute or relative path to a .jpg / .png file
    max_size : int   — the largest dimension after resizing (default 512)

    Returns
    -------
    torch.Tensor  shape (1, 3, max_size, max_size)
                  dtype float32, on DEVICE, ImageNet-normalised

    Pipeline
    --------
    1. Open with PIL, convert to RGB
       → handles grayscale, RGBA, palette-mode images transparently
    2. Resize the shortest side to max_size (aspect ratio kept)
    3. Centre-crop to max_size × max_size
       → guarantees both images have identical spatial dimensions
    4. ToTensor: converts uint8 [0, 255]  →  float32 [0.0, 1.0]
    5. Normalise with ImageNet mean & std  (required for VGG-19)
    6. unsqueeze(0): adds the batch dimension  (3,H,W) → (1,3,H,W)
    7. .to(DEVICE): moves to GPU if available, CPU otherwise
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find image at '{path}'.\n"
            f"Current working directory: {os.getcwd()}"
        )

    # Step 1: open and ensure RGB
    img = Image.open(path).convert("RGB")
    original_size = img.size          # (width, height) BEFORE any transforms

    # Steps 2–5: apply the transform pipeline
    transform = _make_transform(max_size)
    tensor    = transform(img)        # shape: (3, max_size, max_size)

    # Steps 6–7: batch dim + device
    tensor = tensor.unsqueeze(0).to(DEVICE)   # shape: (1, 3, max_size, max_size)

    print(f"Loaded  '{os.path.basename(path)}'")
    print(f"  Original size : {original_size[0]} × {original_size[1]} px")
    print(f"  Tensor shape  : {tuple(tensor.shape)}")
    print(f"  Value range   : [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  Device        : {tensor.device}")

    return tensor

# =============================================================================
# tensor_to_pil
# =============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalised image tensor back to a PIL Image.

    Reverses the ImageNet normalisation:
        pixel = (tensor_value × std) + mean

    Parameters
    ----------
    tensor : torch.Tensor  shape (1, 3, H, W) or (3, H, W)

    Returns
    -------
    PIL.Image.Image  — RGB, uint8, range [0, 255]
    """
    # Remove batch dim if present, detach from autograd graph
    img = tensor.squeeze(0).detach().cpu()    # shape: (3, H, W)

    # Undo ImageNet normalisation channel by channel
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
    img  = img * std + mean                    # back to [0, 1] approximately

    # Clamp: optimisation can push values slightly outside [0, 1]
    img = img.clamp(0.0, 1.0)

    return transforms.ToPILImage()(img)        # float [0,1] → uint8 PIL

# =============================================================================
# imshow  (single tensor)
# =============================================================================

def imshow(tensor: torch.Tensor, title: str = "", ax=None) -> None:
    """
    Display one image tensor using matplotlib.

    Parameters
    ----------
    tensor : torch.Tensor  — normalised (1, 3, H, W) tensor
    title  : str           — label shown above the image
    ax     : plt.Axes      — existing Axes to draw into; creates a new figure if None
    """
    pil_img = tensor_to_pil(tensor)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(pil_img)
    ax.set_title(title, fontsize=13, pad=8)
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        plt.show()

# =============================================================================
# show_images  (content + style + optional output)
# =============================================================================

def show_images(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    output_tensor:  torch.Tensor | None = None,
    save_path:      str = "outputs/preview.png",
) -> None:
    """
    Plot content, style, and (optionally) the stylised result side by side.

    Saves the figure to `save_path` and closes it — works in both Jupyter
    notebooks (inline) and plain Python scripts (saved file only).

    Parameters
    ----------
    content_tensor : (1, 3, H, W) — the original photo tensor
    style_tensor   : (1, 3, H, W) — the drawing / painting style tensor
    output_tensor  : (1, 3, H, W) — the NST result (pass None while training)
    save_path      : file path to save the figure (PNG)
    """
    panels = [
        (content_tensor, "Content Image"),
        (style_tensor,   "Style Image"),
    ]
    if output_tensor is not None:
        panels.append((output_tensor, "Stylised Output"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # If only one panel, axes is a single object — wrap it for uniform handling
    if n_panels == 1:
        axes = [axes]

    for ax, (tensor, label) in zip(axes, panels):
        pil_img = tensor_to_pil(tensor)
        w, h    = pil_img.size
        ax.imshow(pil_img)
        ax.set_title(f"{label}\n{w} × {h} px", fontsize=12)
        ax.axis("off")

    plt.suptitle("Neural Style Transfer", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Preview saved → {save_path}")

# =============================================================================
# Quick self-test — run:  python image_utils.py
# =============================================================================

if __name__ == "__main__":
    import sys

    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print()

    if len(sys.argv) == 3:
        content_path = sys.argv[1]
        style_path   = sys.argv[2]
    else:
        content_path = "images/content.jpg"
        style_path   = "images/style.jpg"

    missing = [p for p in [content_path, style_path] if not os.path.isfile(p)]
    if missing:
        print("Images not found:")
        for p in missing:
            print(f"  {p}")
        print("\nUsage:  python image_utils.py <content.jpg> <style.jpg>")
        sys.exit(0)

    print("── Loading content image ─────────────────────")
    content = load_image(content_path)
    print()
    print("── Loading style image ───────────────────────")
    style   = load_image(style_path)
    print()
    print("── Generating preview ────────────────────────")
    show_images(content, style)

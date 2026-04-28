# =============================================================================
# main.py — Neural Style Transfer: end-to-end runner
# =============================================================================
# Usage:
#   python main.py                              # uses default images
#   python main.py --content img.jpg --style sketch.jpg
#   python main.py --steps 300 --style-weight 1000000
# =============================================================================

import os
import argparse
import time
import datetime
import urllib.request

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL
from PIL import Image

from image_utils import load_image, tensor_to_pil, show_images, DEVICE
from neural_style_transfer import (
    load_vgg19,
    run_style_transfer,
    IMAGE_SIZE,
)

# ── Directories ───────────────────────────────────────────────────────────────

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sample image sources (public domain, Wikimedia Commons) ──────────────────
#
# Content: "Tubingen Neckarfront" — the classic NST benchmark photo
# Style  : "The Starry Night" by Vincent van Gogh (1889)
# Each entry lists several mirror URLs tried in order; PIL fallback if all fail.

SAMPLE_IMAGES = {
    "content": {
        "path": os.path.join(IMAGES_DIR, "content.jpg"),
        # Direct file URL (no /thumb/ prefix) — avoids Wikimedia's thumbnail restrictions
        "urls": [
            "https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
        ],
    },
    "style": {
        "path": os.path.join(IMAGES_DIR, "style.jpg"),
        # Direct file URL for Starry Night (~30 MB); synthetic fallback if slow
        "urls": [
            "https://upload.wikimedia.org/wikipedia/commons/e/ea/"
            "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        ],
    },
}


# ── Synthetic image fallback ──────────────────────────────────────────────────

def _make_synthetic_content(path: str, size: int = 512) -> None:
    """Generate a simple gradient landscape as a stand-in content image."""
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Sky: light blue gradient top → mid
    for r in range(size // 2):
        t = r / (size // 2)
        img[r, :] = [int(135 + t * 50), int(206 - t * 50), int(235 - t * 30)]
    # Ground: green gradient mid → bottom
    for r in range(size // 2, size):
        t = (r - size // 2) / (size // 2)
        img[r, :] = [int(60 - t * 20), int(140 - t * 40), int(60 - t * 20)]
    # Simple "sun"
    cy, cx, rad = size // 4, size // 2, size // 10
    Y, X = np.ogrid[:size, :size]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= rad ** 2
    img[mask] = [255, 220, 80]
    Image.fromarray(img).save(path)
    print(f"  content: synthetic landscape saved to {path}")


def _make_synthetic_style(path: str, size: int = 512) -> None:
    """Generate a swirling Van-Gogh-like pattern as a stand-in style image."""
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(size):
        for c in range(0, size, 2):          # stride for speed
            wave = int(127 + 127 * np.sin((r + c) / 20.0))
            img[r, c]     = [wave // 2, wave // 3, min(wave + 50, 255)]
            img[r, c + 1] = img[r, c]
    Image.fromarray(img).save(path)
    print(f"  style:   synthetic swirl    saved to {path}")


# ── Image downloader ──────────────────────────────────────────────────────────

def download_sample_images() -> None:
    """
    Ensure both sample images exist on disk.

    Strategy for each image:
      1. Skip if the file already exists.
      2. Try each URL in order (first successful download wins).
      3. Fall back to a PIL-generated synthetic image if all URLs fail.
         The synthetic images are simple enough to run NST on and prove
         the pipeline works, even without internet access.
    """
    headers = {"User-Agent": "Mozilla/5.0 (NST-demo/1.0)"}

    for label, info in SAMPLE_IMAGES.items():
        path = info["path"]

        if os.path.isfile(path):
            size_kb = os.path.getsize(path) // 1024
            print(f"  {label}: already exists ({size_kb} KB) — {os.path.basename(path)}")
            continue

        downloaded = False
        for url in info["urls"]:
            try:
                print(f"  {label}: trying {url[:60]}...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                with open(path, "wb") as f:
                    f.write(data)
                _resize_downloaded(path, max_px=1024)   # shrink if too large for PIL
                size_kb = os.path.getsize(path) // 1024
                print(f"  {label}: downloaded ({size_kb} KB) → {os.path.basename(path)}")
                downloaded = True
                break
            except Exception as exc:
                print(f"  {label}: URL failed ({exc}), trying next...")

        if not downloaded:
            print(f"  {label}: all URLs failed — generating synthetic image...")
            if label == "content":
                _make_synthetic_content(path)
            else:
                _make_synthetic_style(path)


def _resize_downloaded(path: str, max_px: int = 1024) -> None:
    """
    Shrink a just-downloaded image to at most max_px on the long side.
    Overwrites the file in-place. Prevents PIL DecompressionBombError on
    very high-resolution source files (e.g. the full Starry Night is 1.5 Gpx).
    """
    PIL.Image.MAX_IMAGE_PIXELS = None          # lift limit temporarily for resize
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(path, quality=92)
        print(f"    resized to {img.size[0]}×{img.size[1]} px")


# ── Result display & save ─────────────────────────────────────────────────────

def save_result(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    output_tensor:  torch.Tensor,
    output_path:    str,
) -> None:
    """
    Save the stylised image on its own AND a side-by-side comparison figure.

    Files written
    -------------
    <output_path>                     — the stylised image  (PNG)
    <output_path stem>_comparison.png — content | style | output side-by-side
    """
    # ── Standalone output image ───────────────────────────────────────────────
    output_pil = tensor_to_pil(output_tensor)
    output_pil.save(output_path)
    print(f"\n  Stylised image  → {output_path}")

    # ── Side-by-side comparison ───────────────────────────────────────────────
    content_pil = tensor_to_pil(content_tensor)
    style_pil   = tensor_to_pil(style_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes,
        [content_pil, style_pil, output_pil],
        ["Content (original photo)", "Style (reference)", "Output (stylised)"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, pad=8)
        ax.axis("off")

    plt.suptitle("Neural Style Transfer Result", fontsize=15, fontweight="bold")
    plt.tight_layout()

    stem, _ = os.path.splitext(output_path)
    comparison_path = f"{stem}_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison      → {comparison_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    print("\n" + "=" * 62)
    print("  Neural Style Transfer")
    print("=" * 62)

    # ── 1. Ensure images exist ────────────────────────────────────────────────
    print("\n[1/5] Checking images...")

    content_path = args.content or SAMPLE_IMAGES["content"]["path"]
    style_path   = args.style   or SAMPLE_IMAGES["style"]["path"]

    # For default paths that don't exist yet, run the downloader
    missing_defaults = any([
        args.content is None and not os.path.isfile(SAMPLE_IMAGES["content"]["path"]),
        args.style   is None and not os.path.isfile(SAMPLE_IMAGES["style"]["path"]),
    ])
    if missing_defaults:
        print("  Sample images not found — fetching...")
        download_sample_images()

    for label, path in [("content", content_path), ("style", style_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} image not found: {path}")
        size_kb = os.path.getsize(path) // 1024
        print(f"  {label}: {os.path.basename(path)}  ({size_kb} KB)")

    # ── 2. Load images ────────────────────────────────────────────────────────
    print("\n[2/5] Loading images...")

    content_tensor = load_image(content_path, max_size=args.image_size)
    style_tensor   = load_image(style_path,   max_size=args.image_size)

    # ── 3. Load VGG-19 ───────────────────────────────────────────────────────
    print("\n[3/5] Loading VGG-19 (frozen feature extractor)...")
    vgg = load_vgg19()
    param_count = sum(p.numel() for p in vgg.parameters())
    print(f"  Parameters : {param_count:,}  (all frozen — only the image is optimised)")
    print(f"  Device     : {DEVICE}")

    # ── 4. Run style transfer ─────────────────────────────────────────────────
    print("\n[4/5] Running style transfer...")
    print(f"  Steps         : {args.steps}")
    print(f"  Style weight  : {args.style_weight:,}")
    print(f"  Content weight: {args.content_weight}")
    print(f"  Image size    : {args.image_size} px")

    t_start = time.time()

    output_tensor = run_style_transfer(
        content_img    = content_tensor,
        style_img      = style_tensor,
        vgg            = vgg,
        num_steps      = args.steps,
        style_weight   = args.style_weight,
        content_weight = args.content_weight,
        print_every    = args.print_every,
    )

    elapsed = time.time() - t_start
    print(f"\n  Time elapsed  : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── 5. Save results ───────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"output_{timestamp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    save_result(content_tensor, style_tensor, output_tensor, output_path)

    print("\n" + "=" * 62)
    print(f"  Done.  Results saved to  outputs/")
    print("=" * 62 + "\n")


# ── CLI argument parser ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural Style Transfer — turn a photo into a drawing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Image paths
    p.add_argument("--content", type=str, default=None,
                   help="Path to content image (your photo). "
                        "Downloads Neckarfront if omitted.")
    p.add_argument("--style", type=str, default=None,
                   help="Path to style image (drawing/painting). "
                        "Downloads Starry Night if omitted.")

    # Optimisation
    p.add_argument("--steps", type=int, default=300,
                   help="Number of LBFGS optimisation steps.")
    p.add_argument("--style-weight", type=float, default=1_000_000,
                   help="Weight for style loss (higher = more stylised).")
    p.add_argument("--content-weight", type=float, default=1.0,
                   help="Weight for content loss (higher = closer to original).")

    # Misc
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help="Resize images so the long side = this value (px).")
    p.add_argument("--print-every", type=int, default=50,
                   help="Print loss every N optimiser closure calls.")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

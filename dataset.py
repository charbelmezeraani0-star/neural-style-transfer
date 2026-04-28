"""
Dataset utilities for training the transformation network.

Content images can be any large collection of photos.
Recommended: MS-COCO train2017 (~80k images, ~18 GB) or any folder of JPGs.

Quick-start with a small synthetic set (for testing the training loop):
    python dataset.py --generate 500 --out data/train
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ImageFolderDataset(Dataset):
    """
    Loads all .jpg / .jpeg / .png images from a directory (non-recursive).

    Parameters
    ----------
    root : str
        Path to the folder containing training images.
    transform : callable, optional
        Torchvision transform applied to each image. Defaults to TRAIN_TRANSFORM.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform or TRAIN_TRANSFORM
        self.paths     = sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in '{root}'.\n"
                "Run:  python dataset.py --generate 500 --out data/train\n"
                "Or download MS-COCO train2017 and point --data-dir there."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation — returns a (B, 3, H, W) tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ── Quick synthetic dataset generator for testing ─────────────────────────────

def generate_synthetic_images(n: int, out_dir: str, size: int = 256):
    """
    Generate N random colour-gradient images for a quick smoke-test of training.
    Not useful for actual quality — only for verifying the training loop runs.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        # Random linear gradient
        c1 = np.random.randint(0, 256, 3)
        c2 = np.random.randint(0, 256, 3)
        for row in range(size):
            t = row / size
            arr[row, :] = (1 - t) * c1 + t * c2
        # Add random noise blobs
        for _ in range(np.random.randint(3, 10)):
            cx, cy = np.random.randint(0, size, 2)
            r = np.random.randint(10, 60)
            Y, X = np.ogrid[:size, :size]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
            arr[mask] = np.random.randint(0, 256, 3)
        Image.fromarray(arr).save(os.path.join(out_dir, f"synth_{i:05d}.png"))
    print(f"Generated {n} synthetic images in '{out_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=int, default=0,
                        help="Number of synthetic images to generate")
    parser.add_argument("--out", type=str, default="data/train",
                        help="Output directory")
    args = parser.parse_args()
    if args.generate > 0:
        generate_synthetic_images(args.generate, args.out)

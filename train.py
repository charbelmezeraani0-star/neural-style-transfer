"""
Train the Fast Neural Style Transfer transformation network.

Usage
-----
# 1. Put your style image in styles/ (e.g. styles/sketch.jpg)
# 2. Put training images in data/train/  (or use --generate for a quick test)
# 3. Run:

    python train.py \\
        --style      styles/sketch.jpg \\
        --data-dir   data/train \\
        --epochs     2 \\
        --batch-size 4 \\
        --checkpoint checkpoints/sketch_model.pth

# To resume from a checkpoint:
    python train.py --resume checkpoints/sketch_model.pth ...

# Quick smoke-test (generates synthetic data automatically):
    python train.py --style styles/sketch.jpg --generate 200 --epochs 1
"""

import os
import time
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transform_net import TransformNet
from perceptual_loss import PerceptualLoss, DEVICE
from dataset import ImageFolderDataset, denormalize, generate_synthetic_images

import torchvision.transforms as transforms
from PIL import Image


# ── Style image loading ───────────────────────────────────────────────────────

def load_style_tensor(path: str, size: int = 256) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(DEVICE)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    if args.generate:
        generate_synthetic_images(args.generate, args.data_dir)

    dataset    = ImageFolderDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images  |  "
          f"{len(dataloader)} batches/epoch  |  "
          f"Device: {DEVICE}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformNet(base_channels=args.base_channels,
                         n_residual=args.n_residual).to(DEVICE)

    start_epoch  = 0
    global_step  = 0
    history: list[dict] = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step",  0)
        history     = ckpt.get("history", [])
        print(f"Resumed from checkpoint  (epoch {start_epoch}, step {global_step})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Loss ──────────────────────────────────────────────────────────────────
    style_tensor = load_style_tensor(args.style, size=args.style_size)
    loss_fn = PerceptualLoss(
        style_img      = style_tensor,
        style_weight   = args.style_weight,
        content_weight = args.content_weight,
        tv_weight      = args.tv_weight,
    ).to(DEVICE)

    # ── Training ──────────────────────────────────────────────────────────────
    total_batches = args.epochs * len(dataloader)
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, content in enumerate(dataloader):
            content = content.to(DEVICE)

            output = model(denormalize(content))  # transform net expects [0,1] input

            # Re-normalise output for VGG
            mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            output_norm = (output - mean) / std

            loss, info = loss_fn(output_norm, content)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                elapsed   = time.time() - t0
                steps_done = epoch * len(dataloader) + batch_idx + 1
                eta        = elapsed / steps_done * (total_batches - steps_done)
                print(
                    f"Epoch {epoch+1}/{start_epoch + args.epochs}  "
                    f"Step {global_step}  "
                    f"Loss {loss.item():.4f}  "
                    f"(content {info['content']:.3f}  "
                    f"style {info['style']:.3f}  "
                    f"tv {info['tv']:.3f})  "
                    f"ETA {eta/60:.1f} min"
                )
                history.append({
                    "step":    global_step,
                    "epoch":   epoch,
                    "loss":    loss.item(),
                    "content": info["content"],
                    "style":   info["style"],
                    "tv":      info["tv"],
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"\n── Epoch {epoch+1} complete  avg_loss={avg_loss:.4f} ──\n")

        # Save checkpoint after every epoch
        _save_checkpoint(model, optimizer, epoch + 1, global_step,
                         history, args.style, args.checkpoint)

    print(f"Training complete in {(time.time()-t0)/60:.1f} min")
    print(f"Model saved to: {args.checkpoint}")

    # Save loss history as JSON for later plotting
    history_path = args.checkpoint.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Loss history saved to: {history_path}")


def _save_checkpoint(model, optimizer, epoch, step, history, style_path, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "epoch":      epoch,
        "step":       step,
        "history":    history,
        "style_path": style_path,
    }, path)
    print(f"Checkpoint saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Fast Neural Style Transfer")

    # Required
    p.add_argument("--style",      required=True, help="Path to style image")

    # Data
    p.add_argument("--data-dir",   default="data/train",
                   help="Folder of training content images")
    p.add_argument("--generate",   type=int, default=0,
                   help="Generate N synthetic training images (for testing)")

    # Training
    p.add_argument("--epochs",         type=int,   default=2)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--workers",        type=int,   default=4)

    # Loss weights
    p.add_argument("--style-weight",   type=float, default=1e5)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--tv-weight",      type=float, default=1e-6)
    p.add_argument("--style-size",     type=int,   default=256,
                   help="Resize style image to this size before computing Gram matrices")

    # Model architecture
    p.add_argument("--base-channels",  type=int,   default=32)
    p.add_argument("--n-residual",     type=int,   default=5)

    # Checkpointing
    p.add_argument("--checkpoint",     default="checkpoints/model.pth")
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint to resume training from")

    # Logging
    p.add_argument("--log-interval",   type=int,   default=50)

    args = p.parse_args()
    train(args)

<div align="center">

# Neural Style Transfer Studio

**Transform any photograph into a work of art using deep learning.**

Built on the algorithm introduced by Gatys et al. (2015), this project uses a frozen VGG-19 convolutional network to extract content and style representations from images, then optimizes a generated image to simultaneously match both — producing stunning artistic renderings directly in the browser.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## Overview

Neural Style Transfer (NST) treats artistic style as a learnable signal. Rather than training a model, it runs iterative gradient descent **on the pixel values of a generated image**, minimizing two competing losses:

| Loss | What it measures | VGG layer |
|---|---|---|
| **Content loss** | MSE between feature maps of the generated and content images | `conv_4` |
| **Style loss** | MSE between Gram matrices of the generated and style images | `conv_1`, `conv_2`, `conv_3` |

The Gram matrix captures **channel-wise feature correlations** — texture and style — independently of spatial structure. Using only the early convolutional layers for style preserves low-level artistic texture without imposing large-scale composition from the style image.

---

## Features

- **Web UI** — drag-and-drop upload, live progress bar, before/after comparison, one-click download
- **Real-time feedback** — Server-Sent Events stream step count and loss values as optimization runs
- **Tunable parameters** — iterations, style strength, and content fidelity controllable from the UI
- **GPU-accelerated** — automatically uses CUDA when available; falls back to CPU
- **Clean separation** — `nst_core.py` is a self-contained engine importable independently of Flask

---

## Architecture

```
ml project/
├── nst_core.py          # NST engine (VGG-19, losses, optimizer loop)
├── app.py               # Flask server (upload, SSE progress, result serving)
├── templates/
│   └── index.html       # Single-page web UI
├── static/
│   ├── css/style.css    # Dark glassmorphism theme
│   └── js/app.js        # Drag-drop, SSE client, UI state
├── NST/
│   ├── NST.ipynb                # Original research notebook
│   ├── NST (tuned).ipynb        # Tuned hyperparameters notebook
│   └── images/
│       ├── content/             # Sample content images
│       └── styles/              # Sample style images
├── uploads/             # Temporary upload storage (git-ignored)
└── outputs/             # Generated results (git-ignored)
```

### How the model is built

```
Input image (pixels, requires_grad=True)
    │
    ▼
nn.Sequential(
  Normalization          ← ImageNet mean/std applied inside the model
  conv_1  →  StyleLoss   ← Gram matrix target frozen at init
  relu_1
  conv_2  →  StyleLoss
  relu_2
  conv_3  →  StyleLoss
  relu_3
  conv_4  →  ContentLoss ← Feature map target frozen at init
  ...      (trimmed here)
)
```

All VGG-19 parameters are frozen. Only `input_img` pixels are optimized via **L-BFGS** — a quasi-Newton method well-suited to this problem because it accounts for second-order curvature, converging significantly faster than Adam or SGD.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow)

### Installation

```bash
git clone https://github.com/charbelmezeraani0-star/neural-style-transfer.git
cd neural-style-transfer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flask pillow
```

### Run the web app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

1. Drop your **content image** (the photo to stylize) on the left zone
2. Drop your **style image** (the artwork to draw from) on the right zone
3. Optionally expand **Advanced Settings** to tune iterations and weights
4. Click **Generate** — a live progress bar tracks each optimization step
5. Download the result when complete

### Use the core engine directly

```python
from nst_core import run_style_transfer

def on_progress(step, total, style_loss, content_loss):
    print(f"[{step}/{total}]  style={style_loss:.2f}  content={content_loss:.4f}")

output_pil = run_style_transfer(
    content_path="photo.jpg",
    style_path="artwork.jpg",
    num_steps=300,
    style_weight=30_000_000,
    content_weight=3,
    progress_callback=on_progress,
)
output_pil.save("result.png")
```

---

## Hyperparameter Guide

| Parameter | Default | Effect |
|---|---|---|
| `num_steps` | `300` | More steps → stronger stylization, longer runtime |
| `style_weight` | `30,000,000` | Higher → style dominates, content structure fades |
| `content_weight` | `3` | Higher → content structure preserved, less stylized |
| Style layers | `conv_1–3` | Earlier layers capture finer textures; adding `conv_4/5` makes style coarser |

**Ratio** `style_weight / content_weight` is what matters. The defaults (`30M / 3 = 10M`) are tuned for pencil-sketch styles. For painting styles, try `style_weight=50_000_000, content_weight=1`.

---

## Technical Reference

**Paper:** Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). *arXiv:1508.06576*

**Key implementation choices vs. the paper:**

- Normalization lives **inside** the model (not pre-applied to the image tensor). This keeps pixel values in `[0, 1]` throughout, which is required for correct L-BFGS clamping and also makes the engine portable.
- Style layers reduced to `conv_1–3` (vs. `conv_1–5` in the paper) — empirically produces cleaner texture transfer for sketch/line-art styles while running faster.
- `torch.set_default_device` is **not** used in `nst_core.py` to keep it importable without side effects.

---

## Dependencies

| Package | Version |
|---|---|
| `torch` | 2.10.0+cu128 |
| `torchvision` | 0.25.0+cu128 |
| `flask` | 3.1.3 |
| `pillow` | 10.2.0 |

---

## License

MIT — see [LICENSE](LICENSE) for details.

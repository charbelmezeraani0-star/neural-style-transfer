# NST Studio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack Neural Style Transfer application with a modern web UI. Supports both **Classic NST** (Gatys et al. 2015) for any style image, and **Fast NST** (Johnson et al. 2016) for instant inference using a pre-trained feed-forward network.

---

## Features

**Classic NST**
- Iterative pixel optimisation using frozen VGG-19
- Adjustable iterations, style strength, and content fidelity
- Real-time progress via Server-Sent Events

**Fast NST**
- Feed-forward TransformerNet — inference in under a second
- 8 quality improvements over the baseline Johnson 2016 model:
  - 5-layer VGG style loss (relu1_2 → relu5_3)
  - Multi-scale gram matrices (3 scales averaged)
  - Feature statistics loss (per-channel mean + std matching)
  - Identity loss (net(style) ≈ style)
  - Per-layer style weights
  - LR warmup + cosine annealing
  - Mixed precision (AMP) + gradient clipping
  - **PatchGAN adversarial loss** — 70×70 discriminator enforces micro-texture realism

**Web UI**
- Drag-and-drop image upload
- Before/After comparison slider on results
- Results gallery with fullscreen viewer
- Training sample previews per model
- Toast notifications

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (HTML + CSS + JS)                          │
│  Drag-drop upload · Comparison slider · Gallery     │
└─────────────────────────┬───────────────────────────┘
                          │ HTTP / SSE
┌─────────────────────────▼───────────────────────────┐
│  Flask  app.py                                      │
│  /upload   /progress/<id>   /fast-stylize           │
│  /models   /gallery         /sample-img/<m>/<f>     │
└────────────┬────────────────────────┬───────────────┘
             │                        │
┌────────────▼──────────┐  ┌──────────▼──────────────┐
│  nst_core.py          │  │  fast_nst.py             │
│  Classic NST          │  │  Feed-forward inference  │
│  VGG-19 optimisation  │  │  TransformerNet          │
└───────────────────────┘  └──────────────────────────┘
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web app
python app.py

# Open browser
http://localhost:5000
```

---

## Training a Fast NST Model

### 1. Download MS-COCO train2014

Get the dataset from [cocodataset.org](https://cocodataset.org/#download) (~13 GB) and place the zip in `~/Downloads/`.

### 2. Train

```bash
bash start_training.sh
```

This will:
1. Unzip the dataset (first run only)
2. Validate all images and write a clean manifest (first run only, ~2 min)
3. Train a `pencil_sketch` model for 4 epochs (~8–10 hrs on RTX 4050)
4. Save sample previews every 500 steps to `samples/pencil_sketch/`
5. Log loss curves to TensorBoard under `runs/`

**Monitor training:**
```bash
# Live loss curves
tensorboard --logdir runs

# Sample previews
samples/pencil_sketch/step_000500.jpg
samples/pencil_sketch/step_001000.jpg
...
```

**Resume after interruption:**
```bash
bash start_training.sh --resume
```

### 3. Train multiple styles

Add style images to `NST/images/styles/`, then:

```bash
bash train_all_styles.sh
```

Each `.jpg` or `.png` in that folder becomes its own model. Already-trained models are skipped.

---

## Training Hyperparameters

| Flag | Default | Effect |
|------|---------|--------|
| `--epochs` | 4 | More epochs = better texture, diminishing returns after 6 |
| `--batch-size` | 4 | Increase if VRAM allows; 8 gives ~10% faster training |
| `--image-size` | 256 | 288–320 for higher quality; slower per step |
| `--style-weight` | 5e10 | Higher = more stylized, less content |
| `--content-weight` | 1e5 | Higher = more faithful to content |
| `--adv-weight` | 1e4 | PatchGAN loss; raise to 5e4 for sharper textures |
| `--sample-every` | 500 | Steps between saved preview images |

---

## Project Structure

```
ml project/
├── app.py               Flask web server
├── nst_core.py          Classic NST engine (VGG-19 optimisation)
├── fast_nst.py          Fast NST inference helper
├── train.py             Fast NST training script
├── transformer_net.py   TransformerNet + PatchDiscriminator
├── preprocess.py        Dataset validation & manifest generation
├── start_training.sh    One-command training launcher
├── train_all_styles.sh  Train all styles in NST/images/styles/
├── requirements.txt
├── models/              Trained .pth model files
├── samples/             Per-step training preview images
├── runs/                TensorBoard logs
├── NST/images/styles/   Style images for training
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Results

*Training in progress — results will be added here after the first run completes.*

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
- Isola, P., et al. (2017). [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (PatchGAN)
- Ulyanov, D., et al. (2017). [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

---

## License

MIT

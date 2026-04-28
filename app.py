"""
DrawAI — Turn any photo into a drawing.
Run with:  streamlit run app.py
"""

import io
import os
import torch
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

from sketch import EFFECTS
from style_transfer import run_style_transfer, DEVICE
from transform_net import TransformNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="DrawAI", page_icon="✏️", layout="wide")
st.title("✏️ DrawAI — Turn your photo into a drawing")
st.caption(f"Running on **{DEVICE}**")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    [
        "Classical Sketch (instant)",
        "Trained Model (fast inference)",
        "Neural Style Transfer — Gatys (slow, no training needed)",
    ],
)

# ── Mode-specific controls ────────────────────────────────────────────────────

if mode == "Classical Sketch (instant)":
    effect_name = st.sidebar.selectbox("Effect", list(EFFECTS.keys()))

elif mode == "Trained Model (fast inference)":
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    ckpt_files = [
        f for f in os.listdir(ckpt_dir) if f.endswith(".pth")
    ] if os.path.isdir(ckpt_dir) else []

    if ckpt_files:
        ckpt_name = st.sidebar.selectbox("Checkpoint", ckpt_files)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    else:
        st.sidebar.warning(
            "No trained checkpoints found.\n\n"
            "Train a model first:\n"
            "```\npython train.py \\\n"
            "  --style styles/sketch.jpg \\\n"
            "  --generate 500 \\\n"
            "  --epochs 2\n```"
        )
        ckpt_path = None

    infer_size = st.sidebar.slider("Max image size (px)", 256, 1024, 512, 64)

else:  # Gatys NST
    style_source = st.sidebar.radio("Style source", ["Upload your own", "Use a preset"])

    if style_source == "Use a preset":
        preset_dir = os.path.join(os.path.dirname(__file__), "styles")
        presets = [
            f for f in os.listdir(preset_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ] if os.path.isdir(preset_dir) else []
        if presets:
            preset_name = st.sidebar.selectbox("Preset", presets)
            style_path  = os.path.join(preset_dir, preset_name)
        else:
            st.sidebar.warning("Drop images into the `styles/` folder.")
            style_path = None
    else:
        uploaded_style = st.sidebar.file_uploader(
            "Style image", type=["jpg", "jpeg", "png"]
        )
        style_path = uploaded_style

    st.sidebar.markdown("### NST parameters")
    max_size     = st.sidebar.slider("Image size (px)", 256, 768, 512, 64)
    num_steps    = st.sidebar.slider("Iterations", 50, 600, 300, 50)
    style_weight = st.sidebar.select_slider(
        "Style strength",
        options=[1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000],
        value=1_000_000,
        format_func=lambda x: f"{x:,}",
    )

# ── Content image upload ──────────────────────────────────────────────────────

st.subheader("Upload your photo")
uploaded_content = st.file_uploader("Content image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_content:
    content_img = Image.open(uploaded_content).convert("RGB")
    col1.image(content_img, caption="Original photo", use_container_width=True)

    run_btn = st.button("🎨 Generate drawing", type="primary")

    if run_btn:
        result = None

        # ── Classical ─────────────────────────────────────────────────────────
        if mode == "Classical Sketch (instant)":
            with st.spinner(f"Applying {effect_name}…"):
                result = EFFECTS[effect_name](content_img)

        # ── Trained model ─────────────────────────────────────────────────────
        elif mode == "Trained Model (fast inference)":
            if ckpt_path is None:
                st.error("No checkpoint selected.")
            else:
                with st.spinner("Loading model and running inference…"):
                    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
                    model = TransformNet().to(DEVICE)
                    model.load_state_dict(ckpt["model"])
                    model.eval()

                    # Resize keeping aspect ratio
                    w, h = content_img.size
                    scale = infer_size / max(w, h)
                    img_r = content_img.resize(
                        (int(w * scale), int(h * scale)), Image.LANCZOS
                    )

                    tf = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ])
                    tensor = tf(img_r).unsqueeze(0).to(DEVICE)

                    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
                    std  = torch.tensor(IMAGENET_STD,  device=DEVICE).view(1, 3, 1, 1)

                    with torch.no_grad():
                        inp    = (tensor * std + mean).clamp(0, 1)
                        output = model(inp).squeeze(0).cpu().clamp(0, 1)

                    result = transforms.ToPILImage()(output)

        # ── Gatys NST ─────────────────────────────────────────────────────────
        else:
            if style_path is None:
                st.error("Please provide a style image.")
            else:
                progress_bar = st.progress(0)
                status_text  = st.empty()

                def update_progress(step, total, loss):
                    progress_bar.progress(step / total)
                    status_text.text(f"Step {step}/{total} — loss: {loss:.2f}")

                with st.spinner("Running Neural Style Transfer…"):
                    result = run_style_transfer(
                        content_img_input=content_img,
                        style_img_input=style_path,
                        max_size=max_size,
                        num_steps=num_steps,
                        style_weight=style_weight,
                        content_weight=1,
                        progress_callback=update_progress,
                    )

                progress_bar.progress(1.0)
                status_text.text("Done!")

        # ── Show and download ─────────────────────────────────────────────────
        if result is not None:
            col2.image(result, caption="Drawing output", use_container_width=True)

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download result",
                data=buf.getvalue(),
                file_name="drawing.png",
                mime="image/png",
            )
            result.save(os.path.join(os.path.dirname(__file__), "outputs", "drawing.png"))

else:
    st.info("Upload a photo on the left to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("How to train your own model"):
    st.code(
        "# 1. Put a style image (any drawing/sketch) in styles/\n"
        "# 2. Add training images to data/train/  (or use --generate for a quick test)\n"
        "# 3. Run:\n\n"
        "python train.py \\\n"
        "    --style      styles/sketch.jpg \\\n"
        "    --data-dir   data/train \\\n"
        "    --generate   500 \\\n"
        "    --epochs     2 \\\n"
        "    --batch-size 4 \\\n"
        "    --checkpoint checkpoints/sketch_model.pth",
        language="bash",
    )
    st.markdown(
        "After training, select **Trained Model** mode and pick your checkpoint "
        "to run instant inference on any photo."
    )

st.caption(
    "Classical effects use OpenCV. "
    "Trained Model uses Johnson et al. 2016 perceptual loss + VGG-19. "
    "Gatys NST uses direct optimisation (Gatys et al. 2015)."
)

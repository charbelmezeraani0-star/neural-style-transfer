"""
Neural Style Transfer engine using VGG-19.
Based on Gatys et al. 2015 — "A Neural Algorithm of Artistic Style".
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG-19 layer names for content and style extraction
CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS   = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(image_input, max_size=512):
    """Load a PIL image or file path and return a (1, 3, H, W) tensor."""
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    # Resize keeping aspect ratio so the long side ≤ max_size
    w, h = img.size
    scale = max_size / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(tensor):
    """Convert a (1, 3, H, W) normalized tensor back to a PIL image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(3, 1, 1)
    img = tensor.squeeze(0).detach()
    img = img * std + mean
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img.cpu())


# ── Loss modules ──────────────────────────────────────────────────────────────

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G / (b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(-1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(-1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model_and_losses(cnn, content_img, style_img,
                            content_layers=CONTENT_LAYERS,
                            style_layers=STYLE_LAYERS):
    """
    Insert ContentLoss and StyleLoss modules after relevant VGG conv layers.
    Returns the trimmed sequential model plus lists of loss modules.
    """
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential(Normalization())

    content_losses = []
    style_losses   = []
    conv_count = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            name = f"conv_{conv_count}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_count}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_count}"
            # Average pooling gives smoother gradients for style transfer
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_count}"
        else:
            name = f"unknown_{conv_count}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_count}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_count}", sl)
            style_losses.append(sl)

        # Stop after the last needed layer to save memory / time
        if conv_count >= 5 and name.startswith("conv_"):
            break

    return model, content_losses, style_losses


# ── Main transfer function ────────────────────────────────────────────────────

def run_style_transfer(
    content_img_input,
    style_img_input,
    max_size=512,
    num_steps=300,
    style_weight=1_000_000,
    content_weight=1,
    progress_callback=None,
):
    """
    Run NST and return the stylised PIL image.

    progress_callback(step, total, loss) is called every 10 steps if provided.
    """
    content_img = load_image(content_img_input, max_size)
    style_img   = load_image(style_img_input,   max_size)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    model, content_losses, style_losses = build_model_and_losses(
        cnn, content_img, style_img
    )

    # Optimise the input image (initialised from content)
    input_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.01)

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            input_img.clamp_(-3, 3)

        optimizer.zero_grad()
        model(input_img)

        style_score   = sum(sl.loss for sl in style_losses)   * style_weight
        content_score = sum(cl.loss for cl in content_losses) * content_weight
        loss = style_score + content_score
        loss.backward()
        optimizer.step()

        if progress_callback and step % 10 == 0:
            progress_callback(step, num_steps, loss.item())

    with torch.no_grad():
        input_img.clamp_(-3, 3)

    return tensor_to_pil(input_img)

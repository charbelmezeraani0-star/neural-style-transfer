import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T
from transformer_net import TransformerNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_cache: dict[str, TransformerNet] = {}


def load_model(path: str) -> TransformerNet:
    if path not in _cache:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Support both plain state_dict and the new {'model_state', 'n_residual'} format
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            n_residual = ckpt.get('n_residual', 9)
            state_dict = ckpt['model_state']
        else:
            # Legacy: raw state_dict — infer n_residual from key count
            res_keys = [k for k in ckpt if k.startswith('res.') and k.endswith('.weight')]
            n_residual = len(res_keys) // 2 or 9
            state_dict = ckpt

        net = TransformerNet(n_residual=n_residual).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        _cache[path] = net
    return _cache[path]


def stylize(model_path: str, image_path: str) -> Image.Image:
    """Apply a trained fast-NST model to an image. Returns PIL Image at original resolution."""
    model = load_model(model_path)
    img   = Image.open(image_path).convert('RGB')

    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze(0).clamp(0, 1).cpu()

    result = T.ToPILImage()(out).resize(img.size, Image.LANCZOS)
    return _remove_dark_artifact(result)


def _remove_dark_artifact(img: Image.Image) -> Image.Image:
    """Remove isolated dark pixel clusters (stuck-neuron training artifact)."""
    arr = np.array(img, dtype=np.float32)
    # Pixels that are anomalously dark relative to their local neighborhood
    gray = arr.mean(axis=2)
    local_median = np.array(img.filter(ImageFilter.MedianFilter(size=9)).convert('L'), dtype=np.float32)
    # Artifact: pixel is very dark AND much darker than its surroundings
    mask = (gray < 30) & (local_median - gray > 40)
    if mask.sum() == 0:
        return img
    # Replace artifact pixels with the local median value
    med = np.array(img.filter(ImageFilter.MedianFilter(size=9)), dtype=np.uint8)
    result = arr.copy().astype(np.uint8)
    result[mask] = med[mask]
    return Image.fromarray(result)

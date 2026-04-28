"""
Classical (non-neural) drawing / sketch effects using OpenCV + PIL.
Fast — no GPU needed.
"""

import cv2
import numpy as np
from PIL import Image


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pencil_sketch(img: Image.Image, blur_sigma: float = 21.0) -> Image.Image:
    """Classic dodge-blend pencil sketch (grayscale)."""
    gray = cv2.cvtColor(_pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    inv  = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=blur_sigma)
    # Colour dodge: gray / (1 - blur/255)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return _cv_to_pil(sketch)


def color_pencil_sketch(img: Image.Image, blur_sigma: float = 21.0) -> Image.Image:
    """Same as pencil_sketch but preserves colour."""
    bgr  = _pil_to_cv(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv  = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=blur_sigma)
    sketch_gray = cv2.divide(gray, 255 - blur, scale=256.0)

    # Blend colour image with sketch
    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(bgr, 0.4, sketch_bgr, 0.6, 0)
    return _cv_to_pil(result)


def charcoal(img: Image.Image, ksize: int = 5, blur_sigma: float = 1.5) -> Image.Image:
    """Heavy edge sketch that looks like charcoal on paper."""
    gray   = cv2.cvtColor(_pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)
    edges  = cv2.Laplacian(blur, cv2.CV_64F)
    edges  = np.uint8(np.clip(np.abs(edges) * 3, 0, 255))
    # Invert so edges are dark on white background
    charcoal_img = 255 - edges
    # Slightly darken for charcoal feel
    charcoal_img = np.clip(charcoal_img.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    return _cv_to_pil(charcoal_img)


def edge_sketch(img: Image.Image, low: int = 50, high: int = 150) -> Image.Image:
    """Clean Canny edge sketch."""
    gray  = cv2.cvtColor(_pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    return _cv_to_pil(255 - edges)


EFFECTS = {
    "Pencil Sketch (B&W)":    pencil_sketch,
    "Color Pencil Sketch":    color_pencil_sketch,
    "Charcoal":               charcoal,
    "Edge Drawing":           edge_sketch,
}

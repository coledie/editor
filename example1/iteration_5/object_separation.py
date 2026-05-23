"""
object_separation.py — iteration 5
------------------------------------
Same segmentation as iteration 4 (chromaticity + texture energy + bilateral filter).
New: soft/blurred mask rendering.

Instead of hard binary masks (pixel belongs to exactly one cluster),
each cluster produces a float weight map that is heavily Gaussian-blurred.
All K weight maps are then normalized to sum to 1.0 at every pixel.
Filters are blended using those weights, so boundaries between segments
fade smoothly into each other rather than cutting hard.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter

DIR_IN  = "C:/Users/coles/Desktop/editor"
DIR_OUT = "C:/Users/coles/Desktop/editor/iteration_5"

K             = 5
SPATIAL_W     = 0.35
TEXTURE_W     = 0.8
TEXTURE_WIN   = 21
BILATERAL_D   = 15
BILATERAL_SC  = 40
BILATERAL_SS  = 40
LABEL_MED     = 21
BLEND_BLUR    = 101   # heavy Gaussian blur on each cluster weight map

# ── I/O ───────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

# ── features (same as iteration 4) ───────────────────────────────────────────

def chromaticity(img):
    f = img.astype(np.float32)
    total = f[..., 0] + f[..., 1] + f[..., 2] + 1e-6
    return f[..., 0] / total, f[..., 1] / total

def local_texture(img):
    gray = np.mean(img.astype(np.float32) / 255.0, axis=2)
    k = TEXTURE_WIN
    mean    = uniform_filter(gray,     size=k)
    mean_sq = uniform_filter(gray**2,  size=k)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    mx = std.max()
    return (std / mx if mx > 0 else std) * TEXTURE_W

def build_features(img):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
    r_c, g_c = chromaticity(smooth)
    tex      = local_texture(smooth)
    ys, xs   = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w - 1) * SPATIAL_W
    yn = ys.astype(np.float32) / (h - 1) * SPATIAL_W
    return np.stack([r_c, g_c, tex, xn, yn], axis=2).reshape(-1, 5).astype(np.float32)

def segment(img):
    h, w = img.shape[:2]
    feats = build_features(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _, labels_flat, _ = cv2.kmeans(feats, K, None, criteria, 12, cv2.KMEANS_PP_CENTERS)
    labels = median_filter(labels_flat.reshape(h, w), size=LABEL_MED)
    return labels

# ── soft weight maps ──────────────────────────────────────────────────────────

def soft_weights(labels, blur_size=BLEND_BLUR):
    """
    For each cluster, create a binary mask then apply heavy Gaussian blur.
    Normalize across clusters so weights sum to 1.0 at every pixel.
    Returns float32 array (H, W, K).
    """
    h, w = labels.shape
    raw = np.zeros((h, w, K), dtype=np.float32)
    for i in range(K):
        mask = (labels == i).astype(np.float32)
        # blur_size must be odd
        bs = blur_size if blur_size % 2 == 1 else blur_size + 1
        raw[..., i] = cv2.GaussianBlur(mask, (bs, bs), 0)
    total = raw.sum(axis=2, keepdims=True) + 1e-6
    return raw / total

# ── visualization: hard segments map ─────────────────────────────────────────

PALETTE = np.array([
    [231,  76,  60],
    [ 52, 152, 219],
    [ 46, 204, 113],
    [241, 196,  15],
    [155,  89, 182],
], dtype=np.uint8)

def colorize(labels):
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in range(K):
        out[labels == i] = PALETTE[i % len(PALETTE)]
    return out

# ── per-cluster filters ───────────────────────────────────────────────────────

def bw(r):
    g = np.mean(r, axis=2, keepdims=True).astype(np.uint8)
    return np.repeat(g, 3, axis=2)

def sepia(r):
    f = r.astype(np.float32)
    nr = np.clip(f[...,0]*0.393 + f[...,1]*0.769 + f[...,2]*0.189, 0, 255)
    ng = np.clip(f[...,0]*0.349 + f[...,1]*0.686 + f[...,2]*0.168, 0, 255)
    nb = np.clip(f[...,0]*0.272 + f[...,1]*0.534 + f[...,2]*0.131, 0, 255)
    return np.stack([nr, ng, nb], axis=2).astype(np.uint8)

def saturate(r):
    hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 2.2, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def cool(r):
    out = r.astype(np.float32).copy()
    out[..., 0] = np.clip(out[..., 0] * 0.65, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * 1.5,  0, 255)
    return out.astype(np.uint8)

def warm(r):
    out = r.astype(np.float32).copy()
    out[..., 0] = np.clip(out[..., 0] * 1.3,  0, 255)
    out[..., 1] = np.clip(out[..., 1] * 1.1,  0, 255)
    out[..., 2] = np.clip(out[..., 2] * 0.7,  0, 255)
    return out.astype(np.uint8)

FILTERS = [bw, sepia, saturate, cool, warm]

# ── soft blended rendering ────────────────────────────────────────────────────

def apply_filters_soft(img, labels):
    """
    Apply each filter to the full image, then blend all K filtered versions
    using the soft per-cluster weight maps.
    Boundaries between segments fade smoothly instead of hard-cutting.
    """
    weights = soft_weights(labels)           # (H, W, K)
    filtered_stack = np.stack([
        FILTERS[i % len(FILTERS)](img).astype(np.float32)
        for i in range(K)
    ], axis=3)                               # (H, W, 3, K)
    # weights: (H, W, K) → expand to (H, W, 1, K) for broadcast over channels
    w = weights[:, :, np.newaxis, :]        # (H, W, 1, K)
    blended = (filtered_stack * w).sum(axis=3)
    return np.clip(blended, 0, 255).astype(np.uint8)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ["a", "b", "c", "d"]:
        print(f"\n{name}.jpg")
        img    = load(name)
        labels = segment(img)
        save(colorize(labels),             f"{name}_segments")
        save(apply_filters_soft(img, labels), f"{name}_filtered")
    print("\nDone.")

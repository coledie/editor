"""
object_separation.py — iteration 4
------------------------------------
Core insight: objects have two properties that are consistent under varying lighting:
  1. Chromaticity  — the HUE of a surface, independent of how bright the light is
  2. Texture energy — how rough/smooth the surface is (sky=flat, trees=rough, road=medium)

Feature vector per pixel (all normalized to [0,1]):
  r_chroma  = R / (R+G+B)   ← lighting-invariant redness
  g_chroma  = G / (R+G+B)   ← lighting-invariant greenness
  texture   = local std-dev of grayscale in a 21x21 window
  x, y      = spatial position (weighted lower)

NO luminance, no L channel, no brightness whatsoever.
Pre-processing: bilateral filter (edge-preserving) rather than Gaussian,
so object boundaries are kept sharp before feature extraction.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter

DIR_IN  = "C:/Users/coles/Desktop/editor"
DIR_OUT = "C:/Users/coles/Desktop/editor/iteration_4"

K             = 5
SPATIAL_W     = 0.35
TEXTURE_W     = 0.8    # texture energy weight relative to chromaticity (=1.0)
TEXTURE_WIN   = 21     # local window for std-dev computation
BILATERAL_D   = 15     # bilateral filter diameter
BILATERAL_SC  = 40     # sigma color
BILATERAL_SS  = 40     # sigma space
LABEL_MED     = 21

# ── I/O ───────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

# ── feature extraction ────────────────────────────────────────────────────────

def chromaticity(img):
    """R/(R+G+B) and G/(R+G+B) — lighting invariant, range [0,1]."""
    f = img.astype(np.float32)
    total = f[..., 0] + f[..., 1] + f[..., 2] + 1e-6
    r = f[..., 0] / total
    g = f[..., 1] / total
    return r, g

def local_texture(img):
    """
    Local texture energy = std-dev of grayscale in a TEXTURE_WIN x TEXTURE_WIN window.
    Computed efficiently via: std = sqrt(E[x^2] - E[x]^2) using uniform_filter.
    Result normalized to [0, 1].
    """
    gray = np.mean(img.astype(np.float32) / 255.0, axis=2)
    k = TEXTURE_WIN
    # uniform_filter gives local mean
    mean   = uniform_filter(gray,    size=k)
    mean_sq = uniform_filter(gray**2, size=k)
    var    = np.maximum(mean_sq - mean**2, 0)
    std    = np.sqrt(var)
    mx = std.max()
    return std / mx if mx > 0 else std

def build_features(img):
    h, w = img.shape[:2]

    # Edge-preserving smooth: bilateral removes within-object lighting gradients
    # but preserves sharp object boundaries
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)

    r_c, g_c = chromaticity(smooth)
    tex      = local_texture(smooth) * TEXTURE_W

    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w - 1) * SPATIAL_W
    yn = ys.astype(np.float32) / (h - 1) * SPATIAL_W

    feats = np.stack([r_c, g_c, tex, xn, yn], axis=2).reshape(-1, 5)
    return feats.astype(np.float32)

# ── segmentation ──────────────────────────────────────────────────────────────

def segment(img):
    feats = build_features(img)
    h, w  = img.shape[:2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _, labels_flat, _ = cv2.kmeans(
        feats, K, None, criteria, 12, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_flat.reshape(h, w)
    labels = median_filter(labels, size=LABEL_MED)
    return labels

# ── visualization ─────────────────────────────────────────────────────────────

PALETTE = np.array([
    [231,  76,  60],
    [ 52, 152, 219],
    [ 46, 204, 113],
    [241, 196,  15],
    [155,  89, 182],
    [ 26, 188, 156],
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

def apply_filters(img, labels):
    out = np.zeros_like(img)
    for i in range(K):
        mask = labels == i
        region = img.copy(); region[~mask] = 0
        out[mask] = FILTERS[i % len(FILTERS)](region)[mask]
    return out

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ["a", "b", "c", "d"]:
        print(f"\n{name}.jpg")
        img = load(name)
        labels = segment(img)
        save(colorize(labels),          f"{name}_segments")
        save(apply_filters(img, labels), f"{name}_filtered")
    print("\nDone.")

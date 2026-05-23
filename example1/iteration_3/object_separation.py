"""
object_separation.py — iteration 3
------------------------------------
Key change: switch from RGB to LAB color space and weight L (lightness) very low.
L = brightness/lighting — almost irrelevant for identifying what object a pixel belongs to.
a = green-red axis  }  these two encode actual object color, lighting-invariant
b = blue-yellow axis}

Features per pixel: (L*0.1, a*1.0, b*1.0, X*0.5, Y*0.5)
This means two pixels of the same object color will cluster together
even if one is in shadow and the other is in bright light.

Pre-blur still applied to flatten fine texture within objects.
Median filter still applied to clean up remaining speckles.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter

DIR_IN  = "C:/Users/coles/Desktop/editor"
DIR_OUT = "C:/Users/coles/Desktop/editor/iteration_3"

K         = 5
L_WEIGHT  = 0.1   # nearly ignore brightness
AB_WEIGHT = 1.0   # full weight on color-opponent channels
SPATIAL_W = 0.5
BLUR_K    = 31
LABEL_MED = 25

# ── I/O ───────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

# ── segmentation ──────────────────────────────────────────────────────────────

def segment(img, k=K):
    h, w = img.shape[:2]

    # Pre-blur to flatten texture
    blurred = cv2.GaussianBlur(img, (BLUR_K, BLUR_K), 0)

    # Convert to LAB — L in [0,100], a and b in [-127,127]
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 100.0  * L_WEIGHT   # normalize + downweight
    a = lab[..., 1] / 127.0  * AB_WEIGHT  # normalize
    b = lab[..., 2] / 127.0  * AB_WEIGHT  # normalize

    # Spatial coords
    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w - 1) * SPATIAL_W
    yn = ys.astype(np.float32) / (h - 1) * SPATIAL_W

    feats = np.stack([L, a, b, xn, yn], axis=2).reshape(-1, 5)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    _, labels_flat, centers = cv2.kmeans(
        feats, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
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

def colorize_labels(labels, k):
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(k):
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
    out[..., 2] = np.clip(out[..., 2] * 1.5, 0, 255)
    return out.astype(np.uint8)

def warm(r):
    out = r.astype(np.float32).copy()
    out[..., 0] = np.clip(out[..., 0] * 1.3, 0, 255)
    out[..., 1] = np.clip(out[..., 1] * 1.1, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * 0.7, 0, 255)
    return out.astype(np.uint8)

FILTERS = [bw, sepia, saturate, cool, warm]

def apply_filters(img, labels, k):
    out = np.zeros_like(img)
    for i in range(k):
        mask = labels == i
        region = img.copy()
        region[~mask] = 0
        out[mask] = FILTERS[i % len(FILTERS)](region)[mask]
    return out

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ["a", "b", "c", "d"]:
        print(f"\n{name}.jpg")
        img = load(name)
        labels = segment(img, k=K)
        save(colorize_labels(labels, K), f"{name}_segments")
        save(apply_filters(img, labels, K), f"{name}_filtered")
    print("\nDone.")

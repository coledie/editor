"""
object_separation.py — iteration 2
------------------------------------
Key changes from iteration 1:
  - Heavy Gaussian pre-blur before building features: textures inside an object
    become uniform color, so K-means groups the whole region not individual pixels.
  - Higher spatial weight (0.7) so position pulls clusters into contiguous zones.
  - Median filter on the label map (size=31) to dissolve salt-and-pepper speckles
    and merge small isolated islands into their neighbors.
  - K reduced to 4 — fewer clusters = larger coherent regions.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter

DIR_IN  = "C:/Users/coles/Desktop/editor"
DIR_OUT = "C:/Users/coles/Desktop/editor/iteration_2"

K         = 4
SPATIAL_W = 0.7
BLUR_K    = 41    # large Gaussian kernel — smooths texture within objects
LABEL_MED = 31    # median filter size on label map — kills speckles

# ── I/O ───────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

# ── segmentation ──────────────────────────────────────────────────────────────

def segment(img, k=K, spatial_w=SPATIAL_W, blur_k=BLUR_K, label_med=LABEL_MED):
    h, w = img.shape[:2]

    # 1. Pre-blur: collapse fine texture into flat color regions
    blurred = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    # 2. Build (R,G,B,X,Y) feature matrix
    rgb = blurred.astype(np.float32) / 255.0
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float32) / (w - 1) * spatial_w
    ys = ys.astype(np.float32) / (h - 1) * spatial_w
    feats = np.concatenate([
        rgb.reshape(-1, 3),
        xs.reshape(-1, 1),
        ys.reshape(-1, 1),
    ], axis=1)

    # 3. K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.1)
    _, labels_flat, centers = cv2.kmeans(
        feats, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_flat.reshape(h, w)

    # 4. Median filter on label map — merges isolated specks into neighbors
    labels = median_filter(labels, size=label_med)

    rgb_centers = np.clip(centers[:, :3] * 255, 0, 255).astype(np.uint8)
    return labels, rgb_centers

# ── visualization ─────────────────────────────────────────────────────────────

PALETTE = np.array([
    [231,  76,  60],   # red
    [ 52, 152, 219],   # blue
    [ 46, 204, 113],   # green
    [241, 196,  15],   # yellow
    [155,  89, 182],   # purple
    [ 26, 188, 156],   # teal
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

FILTERS = [bw, sepia, saturate, cool]

def apply_filters(img, labels, k):
    out = np.zeros_like(img)
    for i in range(k):
        mask = labels == i
        region = img.copy()
        region[~mask] = 0
        filtered = FILTERS[i % len(FILTERS)](region)
        out[mask] = filtered[mask]
    return out

# ── main ──────────────────────────────────────────────────────────────────────

IMAGES = ["a", "b", "c", "d"]

if __name__ == "__main__":
    for name in IMAGES:
        print(f"\n{name}.jpg  (K={K}, spatial_w={SPATIAL_W}, blur={BLUR_K}, label_med={LABEL_MED})")
        img = load(name)
        labels, centers = segment(img)

        save(colorize_labels(labels, K), f"{name}_segments")
        save(apply_filters(img, labels, K), f"{name}_filtered")

    print("\nDone.")

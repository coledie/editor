"""
object_separation.py — iteration 6
------------------------------------
Key insight: one-hot encode cluster labels before blurring.

Previous approach blurred float weight maps — the blur radius was limited
because small K-means patches got washed out entirely.

New approach:
  1. One-hot encode: pixel in cluster i → vector [0,...,1,...,0] of length K
  2. Apply a very wide Gaussian blur to EACH channel independently
     The blurred value at pixel (x,y) for channel i now means:
     "what fraction of pixels within blur_radius of (x,y) belong to cluster i?"
  3. argmax across channels → new label map
     Each pixel is assigned to whichever cluster DOMINATES ITS NEIGHBORHOOD
     not just which cluster it originally belonged to

This is neighborhood-consensus voting. With a large blur (e.g. 301px),
a pixel only keeps its cluster if that cluster wins a 300px radius majority vote.
Isolated specks and thin tendrils disappear. Large solid regions survive and expand.
Boundaries fall naturally where two clusters have equal neighborhood density.

Result: clean organic blob shapes with smooth curved edges, then hard filters
applied to those clean shapes — no soft blending needed.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter

DIR_IN  = "C:/Users/coles/Desktop/editor"
DIR_OUT = "C:/Users/coles/Desktop/editor/iteration_6"

K           = 5
SPATIAL_W   = 0.35
TEXTURE_W   = 0.8
TEXTURE_WIN = 21
BILATERAL_D = 15
BILATERAL_SC = 40
BILATERAL_SS = 40
LABEL_MED   = 21
ONEHOT_BLUR = 301   # wide neighborhood consensus radius

# ── I/O ───────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

# ── features (same as iterations 4 & 5) ──────────────────────────────────────

def chromaticity(img):
    f = img.astype(np.float32)
    total = f[...,0] + f[...,1] + f[...,2] + 1e-6
    return f[...,0]/total, f[...,1]/total

def local_texture(img):
    gray = np.mean(img.astype(np.float32)/255.0, axis=2)
    mean    = uniform_filter(gray,    size=TEXTURE_WIN)
    mean_sq = uniform_filter(gray**2, size=TEXTURE_WIN)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    mx = std.max()
    return (std/mx if mx > 0 else std) * TEXTURE_W

def build_features(img):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
    r_c, g_c = chromaticity(smooth)
    tex      = local_texture(smooth)
    ys, xs   = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w-1) * SPATIAL_W
    yn = ys.astype(np.float32) / (h-1) * SPATIAL_W
    return np.stack([r_c, g_c, tex, xn, yn], axis=2).reshape(-1,5).astype(np.float32)

def segment_raw(img):
    """K-means → raw label map with median cleanup."""
    h, w  = img.shape[:2]
    feats = build_features(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _, labels_flat, _ = cv2.kmeans(feats, K, None, criteria, 12, cv2.KMEANS_PP_CENTERS)
    labels = median_filter(labels_flat.reshape(h, w), size=LABEL_MED)
    return labels

# ── one-hot blur → argmax label smoothing ────────────────────────────────────

def smooth_labels_onehot(labels, blur_size=ONEHOT_BLUR):
    """
    One-hot encode each pixel, blur each channel independently with a wide
    Gaussian, then take argmax. Each pixel is reassigned to the cluster that
    dominates its neighborhood — large blobs grow, isolated pixels vanish.
    """
    h, w = labels.shape
    bs = blur_size if blur_size % 2 == 1 else blur_size + 1

    # Build one-hot stack: (H, W, K)
    onehot = np.zeros((h, w, K), dtype=np.float32)
    for i in range(K):
        onehot[..., i] = (labels == i).astype(np.float32)

    # Blur each channel — now encodes neighborhood cluster density
    blurred = np.zeros_like(onehot)
    for i in range(K):
        blurred[..., i] = cv2.GaussianBlur(onehot[..., i], (bs, bs), 0)

    # Argmax: assign each pixel to the cluster that wins its neighborhood vote
    return np.argmax(blurred, axis=2).astype(np.int32)

# ── visualization ─────────────────────────────────────────────────────────────

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

def apply_filters_hard(img, labels):
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
        img        = load(name)
        raw_labels = segment_raw(img)
        labels     = smooth_labels_onehot(raw_labels)
        save(colorize(labels),              f"{name}_segments")
        save(apply_filters_hard(img, labels), f"{name}_filtered")
    print("\nDone.")

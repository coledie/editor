"""
Recreates the SLIC K=5 segmentation from perm_032 on d.jpg but replaces
scanlines (zone 0) and static (zone 2) with pass-through (original pixels).
Keeps: halftone (zone 1), glitch/rainbow (zone 3), ripple (zone 4).
Output saved to editor root.
"""
import numpy as np
import cv2
from scipy.ndimage import median_filter
import os

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor"

K = 5
SLIC_N_SEGMENTS  = 300
SLIC_COMPACTNESS = 12
SLIC_SIGMA       = 1
MERGE_SMOOTH     = 15

def load_img(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved {DIR_OUT}/{name}.jpg")

def merge_segs_to_k(img, segs, k):
    lab    = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    unique = np.unique(segs)
    means  = np.array([lab[segs == s].mean(axis=0) for s in unique], dtype=np.float32)
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, asgn, _ = cv2.kmeans(means, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = np.zeros(segs.shape, dtype=np.int32)
    for i, s in enumerate(unique):
        labels[segs == s] = int(asgn[i])
    return median_filter(labels, size=MERGE_SMOOTH)

img = load_img("d")
h, w = img.shape[:2]
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

# ── SLIC segmentation ─────────────────────────────────────────────────────────
from skimage.segmentation import slic as sk_slic
segs   = sk_slic(img, n_segments=SLIC_N_SEGMENTS, compactness=SLIC_COMPACTNESS,
                 sigma=SLIC_SIGMA, start_label=0, channel_axis=2)
labels = merge_segs_to_k(img, segs, K)

# ── textures ──────────────────────────────────────────────────────────────────

# halftone dots
sp  = 13
cy_ = (yy // sp) * sp + sp / 2
cx_ = (xx // sp) * sp + sp / 2
dot = (np.sqrt((yy - cy_) ** 2 + (xx - cx_) ** 2) < sp * 0.4).astype(np.float32)
halftone = (dot * 0.85 + 0.15)[..., np.newaxis]

# TV glitch / rainbow
np.random.seed(0)
glitch = img.astype(np.float32).copy()
for y in np.random.choice(h, max(1, h // 8), replace=False):
    s = int(np.random.choice([-1, 1]) * np.random.randint(8, 30))
    glitch[y] = np.roll(glitch[y], s, axis=0)
for y in np.random.choice(h, max(1, h // 6), replace=False):
    s = np.random.randint(4, 14)
    glitch[y, :, 0] = np.roll(glitch[y, :, 0],  s)
    glitch[y, :, 2] = np.roll(glitch[y, :, 2], -s)
glitch = np.clip(glitch * np.random.uniform(0.6, 1.4, (h, w, 1)), 0, 255).astype(np.float32)

# ripple rings
cy_r, cx_r = int(h * 0.55), int(w * 0.5)
d   = np.sqrt((yy - cy_r) ** 2 + (xx - cx_r) ** 2)
ripple = (1.0 + 0.5 * np.sin(d / 16 * 2 * np.pi))[..., np.newaxis].astype(np.float32)

# ── apply — matching perm_032 zone order but scanlines→clean, static→clean ───
# original perm_032: [scanlines, halftone, static, glitch, ripple]
# this version:      [original,  halftone, original, glitch, ripple]

orig = img.astype(np.float32)
out  = np.zeros_like(orig)

for zone in range(K):
    mask = labels == zone
    if zone == 0:    # was scanlines → pass-through clean
        out[mask] = orig[mask]
    elif zone == 1:  # halftone — keep
        out[mask] = (orig * halftone)[mask]
    elif zone == 2:  # was static → pass-through clean
        out[mask] = orig[mask]
    elif zone == 3:  # glitch/rainbow — keep
        out[mask] = glitch[mask]
    elif zone == 4:  # was ripple → pass-through clean
        out[mask] = orig[mask]

result = np.clip(out, 0, 255).astype(np.uint8)
save_img(result, "d_slic_rainbow_nocirlces")

"""
example3/iteration_4/run.py

Same K=5 segmentation as iter_3, but sweep a gallery of elementary CA rules
over the YELLOW zone only (label index 3). One output image per rule.

Rules sampled across Wolfram's four classes for visual variety:
  Class 2 (periodic/nested):  90, 150, 22
  Class 3 (chaotic):          30, 45, 75, 86, 105
  Class 4 (complex):          110, 54, 193, 137
  Other notable:              60 (Sierpinski-ish), 184 (traffic), 73, 126
"""

import os
import numpy as np
import cv2
from scipy.ndimage import median_filter

# ── paths ─────────────────────────────────────────────────────────────────────

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.normpath(os.path.join(HERE, "..", "d.png"))

K            = 5
TARGET_ZONE  = 3                      # yellow in PALETTE order

# ── segmentation tunables (matches iter_3) ────────────────────────────────────

BILATERAL_D     = 15
BILATERAL_SIGMA = 60
MEDIAN_SIZE     = 25
KM_ATTEMPTS     = 12

L_WEIGHT   = 0.3
AB_WEIGHT  = 1.0
SPATIAL_W  = 1.2

# ── CA gallery ────────────────────────────────────────────────────────────────

# (rule_number, init_mode)
GALLERY = [
    ( 22, "single"),
    ( 30, "single"),
    ( 30, "random"),
    ( 45, "single"),
    ( 54, "single"),
    ( 60, "single"),
    ( 73, "random"),
    ( 75, "single"),
    ( 86, "random"),
    ( 90, "single"),
    (105, "single"),
    (110, "random"),
    (126, "single"),
    (137, "random"),
    (150, "single"),
    (184, "random"),
    (193, "random"),
]

CA_ALPHA = 0.5
CA_SEED  = 7

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

# ── segmentation ──────────────────────────────────────────────────────────────

def zscore(a):
    a = a.astype(np.float32)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-6 else 1.0)

def segment(img, K):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    lab    = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)

    Lz = zscore(lab[..., 0]) * L_WEIGHT
    az = zscore(lab[..., 1]) * AB_WEIGHT
    bz = zscore(lab[..., 2]) * AB_WEIGHT

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xz = zscore(xs) * SPATIAL_W
    yz = zscore(ys) * SPATIAL_W

    feats = np.stack([Lz, az, bz, xz, yz], axis=2).reshape(-1, 5).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.5)
    _, labels_flat, _ = cv2.kmeans(feats, K, None, crit, KM_ATTEMPTS,
                                   cv2.KMEANS_PP_CENTERS)
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return median_filter(labels, size=MEDIAN_SIZE)

# ── CA ────────────────────────────────────────────────────────────────────────

def ca_pattern(rule, height, width, init="single", seed=0):
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)

    row = np.zeros(width, dtype=np.uint8)
    if init == "single":
        row[width // 2] = 1
    else:
        rng = np.random.default_rng(seed)
        row = (rng.random(width) < 0.5).astype(np.uint8)

    grid = np.empty((height, width), dtype=np.uint8)
    grid[0] = row
    for t in range(1, height):
        left  = np.roll(row,  1)
        right = np.roll(row, -1)
        idx   = (left << 2) | (row << 1) | right
        row   = rule_bits[idx]
        grid[t] = row
    return (grid * 255).astype(np.uint8)

# ── render single overlay ─────────────────────────────────────────────────────

def render(img, mask, rule, init):
    h, w = img.shape[:2]
    ca   = ca_pattern(rule, h, w, init=init, seed=CA_SEED).astype(np.float32)
    ca3  = np.stack([ca, ca, ca], axis=2)
    out  = img.astype(np.float32).copy()
    blend = (1 - CA_ALPHA) * out + CA_ALPHA * ca3
    out[mask] = blend[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    img = load_img(SRC_PATH)
    print(f"loaded {SRC_PATH}  shape={img.shape}")

    print(f"segmenting K={K}…")
    labels = segment(img, K)
    mask   = labels == TARGET_ZONE
    print(f"yellow zone covers {mask.mean()*100:.1f}% of pixels")

    out_dir = os.path.join(HERE, "rules")
    for rule, init in GALLERY:
        out = render(img, mask, rule, init)
        fname = f"rule_{rule:03d}_{init}.jpg"
        save_img(out, os.path.join(out_dir, fname))
        print(f"  rule {rule:3d}  init={init:<6}  → {fname}")

    print(f"\nwrote {len(GALLERY)} variants to {out_dir}")

"""
example3/iteration_1/run.py

Segment d.png with k-means (K=4 and K=5), then overlay a Wolfram 1D elementary
cellular automaton pattern in each zone, blended 50/50 with the source pixels.

Pipeline per K:
  1. Bilateral filter → LAB → flat color features (no luminance dominance)
  2. K-means + median filter cleanup → label map
  3. For each zone i (i = 0..K-1):
       - generate a 1D CA of shape (H, W) using a distinct rule + initial cond
       - render CA as a black/white image
       - blend with d.png at alpha=0.5, masked to that zone
  4. Save segments.jpg + overlay.jpg in k{K}/

Output: example3/iteration_1/k4/, example3/iteration_1/k5/
"""

import os
import numpy as np
import cv2
from scipy.ndimage import median_filter

# ── paths ─────────────────────────────────────────────────────────────────────

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.normpath(os.path.join(HERE, "..", "d.png"))
K_VALUES = [4, 5]

# ── segmentation tunables ─────────────────────────────────────────────────────

BILATERAL_D     = 15
BILATERAL_SIGMA = 60
MEDIAN_SIZE     = 25
KM_ATTEMPTS     = 12

# ── CA tunables ───────────────────────────────────────────────────────────────

# One rule per zone, picked for visual variety. Extra rules used only if K > len.
CA_RULES = [30, 90, 110, 150, 184, 54, 60, 22]

# Initial-condition style per zone:
#   "single"  – one live cell in the middle
#   "random"  – random bits at density 0.5 (seeded)
CA_INIT  = ["single", "random", "single", "random", "single", "random", "single", "random"]

CA_ALPHA = 0.5     # 50/50 blend
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

def segment(img, K):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    lab    = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Downweight L (lightness) so brightness gradients don't dominate clustering
    lab[..., 0] *= 0.3
    feats  = lab.reshape(-1, 3)
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.5)
    _, labels_flat, _ = cv2.kmeans(feats, K, None, crit, KM_ATTEMPTS,
                                   cv2.KMEANS_PP_CENTERS)
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return median_filter(labels, size=MEDIAN_SIZE)

# ── colorize for segments preview ────────────────────────────────────────────

PALETTE = np.array([
    [231,  76,  60],
    [ 52, 152, 219],
    [ 46, 204, 113],
    [241, 196,  15],
    [155,  89, 182],
    [ 26, 188, 156],
    [230, 126,  34],
    [149, 165, 166],
], np.uint8)

def colorize(labels, K):
    out = np.zeros((*labels.shape, 3), np.uint8)
    for i in range(K):
        out[labels == i] = PALETTE[i % len(PALETTE)]
    return out

# ── 1D Wolfram cellular automaton ─────────────────────────────────────────────

def ca_pattern(rule, height, width, init="single", seed=0):
    """Run rule `rule` for `height` steps on a row of `width` cells.
    Returns a (height, width) uint8 array of 0/255."""
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
        left   = np.roll(row,  1)
        right  = np.roll(row, -1)
        idx    = (left << 2) | (row << 1) | right     # 0..7
        row    = rule_bits[idx]
        grid[t] = row
    return (grid * 255).astype(np.uint8)

# ── overlay ───────────────────────────────────────────────────────────────────

def apply_ca_overlay(img, labels, K):
    h, w = img.shape[:2]
    out  = img.astype(np.float32).copy()

    for i in range(K):
        mask = labels == i
        if not mask.any():
            continue
        rule = CA_RULES[i % len(CA_RULES)]
        init = CA_INIT [i % len(CA_INIT)]
        ca   = ca_pattern(rule, h, w, init=init, seed=CA_SEED + i).astype(np.float32)
        ca3  = np.stack([ca, ca, ca], axis=2)
        blend = (1 - CA_ALPHA) * out + CA_ALPHA * ca3
        out[mask] = blend[mask]

    return np.clip(out, 0, 255).astype(np.uint8)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    img = load_img(SRC_PATH)
    print(f"loaded {SRC_PATH}  shape={img.shape}")

    for K in K_VALUES:
        print(f"  K={K}  segmenting…", end=" ", flush=True)
        labels = segment(img, K)
        print("overlaying…", end=" ", flush=True)
        overlay = apply_ca_overlay(img, labels, K)
        seg_vis = colorize(labels, K)

        out_dir = os.path.join(HERE, f"k{K}")
        save_img(seg_vis,  os.path.join(out_dir, "segments.jpg"))
        save_img(overlay,  os.path.join(out_dir, "overlay.jpg"))

        rules_used = [CA_RULES[i % len(CA_RULES)] for i in range(K)]
        print(f"done  rules={rules_used}")

    print("\nwrote:", os.path.join(HERE, "k4/"), "and", os.path.join(HERE, "k5/"))

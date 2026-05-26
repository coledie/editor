"""
example3/iteration_3/run.py

Iteration 2's spatial weight `SPATIAL_W=40` (in raw LAB units) wasn't enough
because LAB chroma still dominated the distance metric. Fix: standardize every
feature to unit standard deviation, THEN apply explicit per-feature weights.
That way `SPATIAL_W` is a real multiplier of "one σ of color variation".

Final feature vector per pixel (after z-score):
    [Lz * 0.3, az * 1.0, bz * 1.0, xz * SPATIAL_W, yz * SPATIAL_W]
"""

import os
import numpy as np
import cv2
from scipy.ndimage import median_filter

# ── paths ─────────────────────────────────────────────────────────────────────

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.normpath(os.path.join(HERE, "..", "d.png"))
K_VALUES = [4, 5, 6, 7]

# ── segmentation tunables ─────────────────────────────────────────────────────

BILATERAL_D     = 15
BILATERAL_SIGMA = 60
MEDIAN_SIZE     = 25
KM_ATTEMPTS     = 12

# Weights applied AFTER each feature is normalized to zero-mean / unit-std.
# So these numbers express "how many σ of color variation equals 1σ of position".
L_WEIGHT   = 0.3
AB_WEIGHT  = 1.0
SPATIAL_W  = 1.2     # in σ-units of color; balanced — spatial nudges, not dominates

# ── CA tunables ───────────────────────────────────────────────────────────────

CA_RULES = [30, 90, 110, 150, 184, 54, 60, 22]
CA_INIT  = ["single", "random", "single", "random", "single", "random", "single", "random"]
CA_ALPHA = 0.5
CA_SEED  = 7

# Per-K override: only overlay these zone indices. Missing key → all zones.
# Zone index maps to PALETTE row: 0=red, 1=blue, 2=green, 3=yellow, 4=purple…
OVERLAY_ZONES = {
    5: [3],   # only the yellow segment
}

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

    crit  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.5)
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
        idx    = (left << 2) | (row << 1) | right
        row    = rule_bits[idx]
        grid[t] = row
    return (grid * 255).astype(np.uint8)

# ── overlay ───────────────────────────────────────────────────────────────────

def apply_ca_overlay(img, labels, K, zones=None):
    h, w = img.shape[:2]
    out  = img.astype(np.float32).copy()
    allowed = set(range(K)) if zones is None else set(zones)

    for i in range(K):
        if i not in allowed:
            continue
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
        overlay = apply_ca_overlay(img, labels, K, zones=OVERLAY_ZONES.get(K))
        seg_vis = colorize(labels, K)

        out_dir = os.path.join(HERE, f"k{K}")
        save_img(seg_vis,  os.path.join(out_dir, "segments.jpg"))
        save_img(overlay,  os.path.join(out_dir, "overlay.jpg"))

        rules_used = [CA_RULES[i % len(CA_RULES)] for i in range(K)]
        print(f"done  rules={rules_used}")

    print("\nwrote:", os.path.join(HERE, "k4/"), "and", os.path.join(HERE, "k5/"))

"""
iteration_9/run.py
K-Means only. Feature vector per pixel: [L, A, B, x, y]
  - LAB color channels (perceptually uniform)
  - x, y pixel coordinates (normalized 0-1, scaled by SPATIAL_W)
Runs K = 3, 4, 5. All P(5, k) texture permutations per image per K.

Output: iteration_9/k{k}/{image}/segments.jpg
                              permutations/perm_NNN__t0_t1...jpg
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter
from itertools import permutations
import os

# ── tunables ──────────────────────────────────────────────────────────────────

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_9"

IMAGES   = ["a", "c", "d"]
K_VALUES = [3, 4, 5]

# Feature weights — x/y scaled to this range before clustering
# (LAB L channel is 0-100, A/B are roughly -128 to 127)
SPATIAL_W = 40      # increase to make location matter more, decrease for pure colour

# K-Means
KM_ITERATIONS = 100
KM_ATTEMPTS   = 15   # number of random restarts; more = more stable result
KM_EPS        = 0.05

# Post-processing: smooth label map then soft-blend via per-class gaussian blur
MEDIAN_SIZE = 19
BLUR_SIZE   = 151    # gaussian kernel size (odd pixels)

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_img(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

# ── segmentation ──────────────────────────────────────────────────────────────

def segment(img, k):
    """
    K-Means on a 5-column feature matrix: [L, A, B, x_scaled, y_scaled].
    Each pixel contributes one row. x and y run 0..SPATIAL_W across the image.
    """
    h, w = img.shape[:2]
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # pixel coordinate grids, scaled so spatial spread ~ SPATIAL_W
    ys, xs = np.mgrid[0:h, 0:w]
    x_col = xs.astype(np.float32) / (w - 1) * SPATIAL_W   # 0 .. SPATIAL_W
    y_col = ys.astype(np.float32) / (h - 1) * SPATIAL_W   # 0 .. SPATIAL_W

    # (H*W, 5)  columns: L  A  B  x  y
    feats = np.stack(
        [lab[..., 0], lab[..., 1], lab[..., 2], x_col, y_col],
        axis=2
    ).reshape(-1, 5).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KM_ITERATIONS, KM_EPS)
    _, labels_flat, _ = cv2.kmeans(
        feats, k, None, crit, KM_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_flat.reshape(h, w)

    # smooth hard labels, then soft-blend via per-class gaussian → argmax
    labels = median_filter(labels, size=MEDIAN_SIZE)
    bs = BLUR_SIZE if BLUR_SIZE % 2 == 1 else BLUR_SIZE + 1
    onehot  = np.zeros((h, w, k), np.float32)
    for i in range(k):
        onehot[..., i] = (labels == i).astype(np.float32)
    blurred = np.stack(
        [cv2.GaussianBlur(onehot[..., i], (bs, bs), 0) for i in range(k)], axis=2
    )
    return np.argmax(blurred, axis=2).astype(np.int32)

# ── colorize ──────────────────────────────────────────────────────────────────

PALETTE = np.array([
    [231,  76,  60],   # red
    [ 52, 152, 219],   # blue
    [ 46, 204, 113],   # green
    [241, 196,  15],   # yellow
    [155,  89, 182],   # purple
], np.uint8)

def colorize(labels, k):
    out = np.zeros((*labels.shape, 3), np.uint8)
    for i in range(k):
        out[labels == i] = PALETTE[i]
    return out

# ── texture cache ─────────────────────────────────────────────────────────────

TEX_NAMES = ['glitch', 'scanlines', 'halftone', 'static', 'ripple']

def build_cache(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 0: TV glitch
    np.random.seed(0)
    g = img.astype(np.float32).copy()
    for y in np.random.choice(h, max(1, h // 8), replace=False):
        s = int(np.random.choice([-1, 1]) * np.random.randint(8, 30))
        g[y] = np.roll(g[y], s, axis=0)
    for y in np.random.choice(h, max(1, h // 6), replace=False):
        s = np.random.randint(4, 14)
        g[y, :, 0] = np.roll(g[y, :, 0],  s)
        g[y, :, 2] = np.roll(g[y, :, 2], -s)
    g = np.clip(g * np.random.uniform(0.6, 1.4, (h, w, 1)), 0, 255).astype(np.float32)

    # 1: scanlines
    sl = np.ones((h, w, 1), np.float32); sl[::3, :, :] = 0.2

    # 2: halftone dots
    sp  = 13
    cy_ = (yy // sp) * sp + sp / 2
    cx_ = (xx // sp) * sp + sp / 2
    dot = (np.sqrt((yy - cy_) ** 2 + (xx - cx_) ** 2) < sp * 0.4).astype(np.float32)
    ht  = (dot * 0.85 + 0.15)[..., np.newaxis]

    # 3: TV static
    np.random.seed(42)
    ns = np.random.uniform(0.4, 1.6, (h, w)).astype(np.float32)
    gr = np.random.choice(h, int(h * 0.08), replace=False)
    ns[gr] = np.random.uniform(0.1, 2.2, (len(gr), w))
    st = ns[..., np.newaxis]

    # 4: ripple rings
    cy_r, cx_r = int(h * 0.55), int(w * 0.5)
    d  = np.sqrt((yy - cy_r) ** 2 + (xx - cx_r) ** 2)
    rp = (1.0 + 0.5 * np.sin(d / 16 * 2 * np.pi))[..., np.newaxis].astype(np.float32)

    return [('direct', g), ('mult', sl), ('mult', ht), ('mult', st), ('mult', rp)]

# ── apply permutation ─────────────────────────────────────────────────────────

def apply_perm(img, labels, perm, cache):
    orig = img.astype(np.float32)
    out  = np.zeros_like(orig)
    for zone, tex_idx in enumerate(perm):
        mask = labels == zone
        kind, data = cache[tex_idx]
        out[mask] = data[mask] if kind == 'direct' else (orig * data)[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for k in K_VALUES:
        all_perms = list(permutations(range(len(TEX_NAMES)), k))
        print(f"\n{'='*52}")
        print(f"  K={k}   features=[L,A,B,x,y]   {len(all_perms)} perms/image")
        print(f"{'='*52}")

        for img_name in IMAGES:
            print(f"  {img_name}.jpg  ", end="", flush=True)
            img   = load_img(img_name)
            cache = build_cache(img)
            labels = segment(img, k)

            out_dir = f"{DIR_OUT}/k{k}/{img_name}"
            os.makedirs(f"{out_dir}/permutations", exist_ok=True)
            save_img(colorize(labels, k), f"{out_dir}/segments.jpg")

            for i, perm in enumerate(all_perms):
                result = apply_perm(img, labels, perm, cache)
                fname  = f"perm_{i:03d}__{'_'.join(TEX_NAMES[t] for t in perm)}.jpg"
                save_img(result, f"{out_dir}/permutations/{fname}")

            print(f"done ({len(all_perms)} perms)")

    total = sum(len(list(permutations(range(len(TEX_NAMES)), k)))
                for k in K_VALUES) * len(IMAGES)
    print(f"\nDone. {total} images total.")

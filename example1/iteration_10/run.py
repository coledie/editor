"""
iteration_10/run.py
K-Means with a 6-column feature vector: [L, A, B, x, y, edge]

The edge column is the Scharr gradient magnitude on the L channel,
normalized 0-1 and scaled by EDGE_W. This pulls edge pixels into their own
clusters and gives K-means a structural signal beyond color and position.

Scharr kernels (more accurate gradient estimation than Sobel):
  Kx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
  Ky = Kx.T
  magnitude = sqrt(gx^2 + gy^2), normalized to [0, 1]

K_VALUES = [3, 4, 5]
Output: iteration_10/k{k}/{image}/segments.jpg + permutations/
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter
from itertools import permutations
import os

# ── tunables ──────────────────────────────────────────────────────────────────

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_10"

IMAGES   = ["a", "c", "d"]
K_VALUES = [3, 4, 5]

SPATIAL_W = 40      # x/y scale (same range as LAB channels)
EDGE_W    = 30      # edge magnitude scale — raise to make edges cluster harder

KM_ITERATIONS = 100
KM_ATTEMPTS   = 15
KM_EPS        = 0.05

MEDIAN_SIZE = 19
BLUR_SIZE   = 151

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_img(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

# ── edge feature ──────────────────────────────────────────────────────────────

def scharr_magnitude(img):
    """
    Scharr gradient magnitude on the LAB L channel, normalized to [0, 1].
    Scharr is a 3x3 kernel optimised for rotational accuracy — more precise
    than Sobel at detecting edge orientation and magnitude.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L   = lab[..., 0]                              # 0-100 range
    gx  = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gy  = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = mag / (mag.max() + 1e-9)                 # normalize to [0, 1]
    return mag

# ── segmentation ──────────────────────────────────────────────────────────────

def segment(img, k):
    """
    K-Means on [L, A, B, x_scaled, y_scaled, edge_scaled].
    Edge column is Scharr magnitude * EDGE_W — pixels on strong edges
    cluster separately from flat interior regions.
    """
    h, w = img.shape[:2]
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    ys, xs = np.mgrid[0:h, 0:w]
    x_col = xs.astype(np.float32) / (w - 1) * SPATIAL_W
    y_col = ys.astype(np.float32) / (h - 1) * SPATIAL_W

    edge_col = scharr_magnitude(img) * EDGE_W

    # (H*W, 6) — columns: L  A  B  x  y  edge
    feats = np.stack(
        [lab[..., 0], lab[..., 1], lab[..., 2], x_col, y_col, edge_col],
        axis=2
    ).reshape(-1, 6).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KM_ITERATIONS, KM_EPS)
    _, labels_flat, _ = cv2.kmeans(
        feats, k, None, crit, KM_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_flat.reshape(h, w)

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
    [231,  76,  60],
    [ 52, 152, 219],
    [ 46, 204, 113],
    [241, 196,  15],
    [155,  89, 182],
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

    sl = np.ones((h, w, 1), np.float32); sl[::3, :, :] = 0.2

    sp  = 13
    cy_ = (yy // sp) * sp + sp / 2
    cx_ = (xx // sp) * sp + sp / 2
    dot = (np.sqrt((yy - cy_) ** 2 + (xx - cx_) ** 2) < sp * 0.4).astype(np.float32)
    ht  = (dot * 0.85 + 0.15)[..., np.newaxis]

    np.random.seed(42)
    ns = np.random.uniform(0.4, 1.6, (h, w)).astype(np.float32)
    gr = np.random.choice(h, int(h * 0.08), replace=False)
    ns[gr] = np.random.uniform(0.1, 2.2, (len(gr), w))
    st = ns[..., np.newaxis]

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
        print(f"\n{'='*56}")
        print(f"  K={k}   features=[L,A,B,x,y,edge]   {len(all_perms)} perms/image")
        print(f"{'='*56}")

        for img_name in IMAGES:
            print(f"  {img_name}.jpg  ", end="", flush=True)
            img    = load_img(img_name)
            cache  = build_cache(img)
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

"""
iteration_8/run.py
Three clustering algorithms x K_VALUES x IMAGES, all texture permutations each.

Algorithms
----------
  kmeans       : K-Means on LAB color + normalized spatial coords
  slic         : SLIC superpixels merged to k zones via LAB k-means
  felzenszwalb : Felzenszwalb graph segmentation merged to k zones

Output layout
-------------
  iteration_8/k{k}/{image}/{algo}/segments.jpg
  iteration_8/k{k}/{image}/{algo}/permutations/perm_NNN__t0_t1_...jpg
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter
from itertools import permutations
import os

# ── tunables ──────────────────────────────────────────────────────────────────

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_8"

IMAGES   = ["a", "c"]
K_VALUES = [3, 4]

# K-Means params
KM_SPATIAL_W   = 40
KM_ITERATIONS  = 80
KM_ATTEMPTS    = 12
KM_SMOOTH_MED  = 21
KM_SMOOTH_BLUR = 201

# SLIC params
SLIC_N_SEGMENTS  = 300
SLIC_COMPACTNESS = 12
SLIC_SIGMA       = 1

# Felzenszwalb params
FZ_SCALE    = 200
FZ_SIGMA    = 0.8
FZ_MIN_SIZE = 200

MERGE_SMOOTH = 15

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_img(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

# ── segmentation ──────────────────────────────────────────────────────────────

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

def segment_kmeans(img, k):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w - 1) * KM_SPATIAL_W
    yn = ys.astype(np.float32) / (h - 1) * KM_SPATIAL_W
    feats = np.stack([lab[..., 0], lab[..., 1], lab[..., 2], xn, yn],
                     axis=2).reshape(-1, 5).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KM_ITERATIONS, 0.1)
    _, lf, _ = cv2.kmeans(feats, k, None, crit, KM_ATTEMPTS, cv2.KMEANS_PP_CENTERS)
    labels = median_filter(lf.reshape(h, w), size=KM_SMOOTH_MED)
    bs = KM_SMOOTH_BLUR if KM_SMOOTH_BLUR % 2 == 1 else KM_SMOOTH_BLUR + 1
    onehot  = np.zeros((h, w, k), np.float32)
    for i in range(k):
        onehot[..., i] = (labels == i).astype(np.float32)
    blurred = np.stack(
        [cv2.GaussianBlur(onehot[..., i], (bs, bs), 0) for i in range(k)], axis=2)
    return np.argmax(blurred, axis=2).astype(np.int32)

def segment_slic(img, k):
    from skimage.segmentation import slic as sk_slic
    segs = sk_slic(img, n_segments=SLIC_N_SEGMENTS, compactness=SLIC_COMPACTNESS,
                   sigma=SLIC_SIGMA, start_label=0, channel_axis=2)
    return merge_segs_to_k(img, segs, k)

def segment_felzenszwalb(img, k):
    from skimage.segmentation import felzenszwalb as sk_fz
    segs = sk_fz(img, scale=FZ_SCALE, sigma=FZ_SIGMA, min_size=FZ_MIN_SIZE)
    return merge_segs_to_k(img, segs, k)

ALGOS = {
    "kmeans":        segment_kmeans,
    "slic":          segment_slic,
    "felzenszwalb":  segment_felzenszwalb,
}

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
        print(f"\n{'='*50}")
        print(f"K={k}  ({len(all_perms)} permutations per algo)")
        print(f"{'='*50}")

        for img_name in IMAGES:
            print(f"\n  image {img_name}.jpg")
            img   = load_img(img_name)
            cache = build_cache(img)

            for algo_name, algo_fn in ALGOS.items():
                print(f"    [{algo_name}] ", end="", flush=True)
                labels  = algo_fn(img, k)
                out_dir = f"{DIR_OUT}/k{k}/{img_name}/{algo_name}"
                os.makedirs(f"{out_dir}/permutations", exist_ok=True)

                save_img(colorize(labels, k), f"{out_dir}/segments.jpg")

                for i, perm in enumerate(all_perms):
                    result = apply_perm(img, labels, perm, cache)
                    fname  = f"perm_{i:03d}__{'_'.join(TEX_NAMES[t] for t in perm)}.jpg"
                    save_img(result, f"{out_dir}/permutations/{fname}")
                print(f"{len(all_perms)} perms saved")

    total = sum(len(list(permutations(range(len(TEX_NAMES)), k))) for k in K_VALUES) \
            * len(IMAGES) * len(ALGOS)
    print(f"\nDone. {total} images total.")

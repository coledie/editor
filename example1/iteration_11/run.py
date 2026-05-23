"""
iteration_11/run.py
Watershed segmentation + all P(5,k) texture permutations.

Pipeline:
  1. Scharr gradient magnitude on the LAB L channel → watershed elevation surface
  2. Seeds: distance transform of flat (low-gradient) regions, then local maxima
     spaced at least SEED_SPACING px apart
  3. skimage.watershed floods from seeds uphill along gradient
  4. Many resulting segments merged to k zones via k-means on mean LAB color
  5. Smooth label map → per-class gaussian blur + argmax (soft boundaries)

Watershed follows edges natively — boundaries land on gradient ridges
(object contours) rather than being driven purely by color similarity.

Output: iteration_11/k{k}/{image}/segments.jpg + permutations/
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from itertools import permutations
from skimage.segmentation import watershed
import os

# ── tunables ──────────────────────────────────────────────────────────────────

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_11"

IMAGES   = ["a", "c", "d"]
K_VALUES = [3, 4, 5]

# Watershed seed detection
FLAT_PERCENTILE = 55    # gradient percentile below which a pixel counts as "flat"
SEED_SPACING    = 35    # min px between seeds — lower = more seeds = finer init segments

# Watershed flood
COMPACTNESS = 0.0005    # 0 = pure gradient-following; raise for rounder regions

# Merge many watershed segments → k zones
MERGE_SMOOTH = 15

# Final label smoothing
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

# ── segmentation ──────────────────────────────────────────────────────────────

def merge_segs_to_k(img, segs, k):
    """Collapse many fine segments into k zones by k-means on mean LAB color."""
    lab    = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    unique = np.unique(segs)
    unique = unique[unique > 0]   # watershed marks boundaries as -1 or 0
    means  = np.array([lab[segs == s].mean(axis=0) for s in unique], dtype=np.float32)
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, asgn, _ = cv2.kmeans(means, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = np.zeros(segs.shape, dtype=np.int32)
    for i, s in enumerate(unique):
        labels[segs == s] = int(asgn[i])
    return median_filter(labels, size=MERGE_SMOOTH)


def segment(img, k):
    """
    Watershed on Scharr gradient. Seeds come from local maxima in the
    distance transform of flat (low-gradient) image regions.
    """
    h, w = img.shape[:2]

    # Smooth before gradient to reduce noise seeds
    smooth = cv2.GaussianBlur(img, (7, 7), 2.0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    L   = lab[..., 0] / 100.0                     # normalize L to [0, 1]

    # Scharr gradient magnitude — watershed elevation surface
    gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    gradient /= gradient.max() + 1e-9             # [0, 1]

    # Flat regions: pixels below the FLAT_PERCENTILE gradient threshold
    flat = (gradient < np.percentile(gradient, FLAT_PERCENTILE)).astype(np.uint8)

    # Distance transform → local maxima spaced >= SEED_SPACING px apart
    distance  = ndi.distance_transform_edt(flat)
    local_max = (distance == ndi.maximum_filter(distance, size=SEED_SPACING))
    local_max &= distance > 0
    markers, _ = ndi.label(local_max)

    # Flood from seeds along gradient surface
    segs = watershed(gradient, markers, compactness=COMPACTNESS)

    # Merge fine segments → k zones, then soft-blend boundaries
    labels = merge_segs_to_k(img, segs, k)

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
        print(f"\n{'='*58}")
        print(f"  K={k}   watershed + merge   {len(all_perms)} perms/image")
        print(f"{'='*58}")

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

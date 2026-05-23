"""
all_permutations.py — generate all 5! = 120 texture permutations
Computes segmentation labels once, then applies every permutation of the
5 textures across the 5 zones, saving each result to permutations/.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter
from itertools import permutations
import os

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_7/permutations"

K=5; SPATIAL_W=0.35; TEXTURE_W=0.8; TEXTURE_WIN=21
BILATERAL_D=15; BILATERAL_SC=40; BILATERAL_SS=40
LABEL_MED=21; ONEHOT_BLUR=301

os.makedirs(DIR_OUT, exist_ok=True)

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])

def get_labels(img):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
    f = smooth.astype(np.float32)
    total = f[...,0]+f[...,1]+f[...,2]+1e-6
    r_c, g_c = f[...,0]/total, f[...,1]/total
    gray = np.mean(smooth.astype(np.float32)/255., axis=2)
    mean = uniform_filter(gray, size=TEXTURE_WIN)
    mean_sq = uniform_filter(gray**2, size=TEXTURE_WIN)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    tex = std / (std.max() + 1e-9) * TEXTURE_W
    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32)/(w-1)*SPATIAL_W
    yn = ys.astype(np.float32)/(h-1)*SPATIAL_W
    feats = np.stack([r_c, g_c, tex, xn, yn], axis=2).reshape(-1,5).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _, lf, _ = cv2.kmeans(feats, K, None, criteria, 12, cv2.KMEANS_PP_CENTERS)
    labels = median_filter(lf.reshape(h,w), size=LABEL_MED)
    bs = ONEHOT_BLUR if ONEHOT_BLUR%2==1 else ONEHOT_BLUR+1
    onehot = np.zeros((h,w,K), np.float32)
    for i in range(K):
        onehot[...,i] = (labels==i).astype(np.float32)
    blurred = np.stack([cv2.GaussianBlur(onehot[...,i],(bs,bs),0) for i in range(K)], axis=2)
    return np.argmax(blurred, axis=2).astype(np.int32)

# ── pre-compute deterministic textures (cached, reused across permutations) ──

def build_texture_cache(img, labels):
    h, w = img.shape[:2]
    orig = img.astype(np.float32)
    cache = {}

    # 0: glitch — direct pixel replacement, seeded
    np.random.seed(0)
    glitched = img.astype(np.float32).copy()
    shift_rows = np.random.choice(h, max(1, h//8), replace=False)
    for y in shift_rows:
        s = int(np.random.choice([-1,1]) * np.random.randint(8, 30))
        glitched[y] = np.roll(glitched[y], s, axis=0)
    aber_rows = np.random.choice(h, max(1, h//6), replace=False)
    for y in aber_rows:
        s = np.random.randint(4, 14)
        glitched[y,:,0] = np.roll(glitched[y,:,0], s)
        glitched[y,:,2] = np.roll(glitched[y,:,2], -s)
    noise = np.random.uniform(0.6, 1.4, (h, w, 1))
    cache[0] = ('direct', np.clip(glitched * noise, 0, 255).astype(np.float32))

    # 1: scanlines
    pat = np.ones((h, w), np.float32); pat[::3] = 0.2
    cache[1] = ('mult', pat[..., np.newaxis])

    # 2: halftone
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sp = 13
    cy_ = (yy // sp) * sp + sp / 2
    cx_ = (xx // sp) * sp + sp / 2
    dist = np.sqrt((yy - cy_)**2 + (xx - cx_)**2)
    pat = (dist < sp * 0.4).astype(np.float32) * 0.85 + 0.15
    cache[2] = ('mult', pat[..., np.newaxis])

    # 3: TV static
    np.random.seed(42)
    noise = np.random.uniform(0.4, 1.6, (h, w)).astype(np.float32)
    glitch_rows = np.random.choice(h, int(h * 0.08), replace=False)
    noise[glitch_rows] = np.random.uniform(0.1, 2.2, (len(glitch_rows), w))
    cache[3] = ('mult', noise[..., np.newaxis])

    # 4: ripple rings
    cy_r, cx_r = int(h * 0.55), int(w * 0.5)
    yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = np.sqrt((yy2 - cy_r)**2 + (xx2 - cx_r)**2)
    pat2 = 1.0 + 0.5 * np.sin(d2 / 16 * 2 * np.pi).astype(np.float32)
    cache[4] = ('mult', pat2[..., np.newaxis])

    return cache, orig

def apply_permutation(img, labels, perm, cache):
    orig = img.astype(np.float32)
    out = np.zeros_like(orig)
    for zone, tex_idx in enumerate(perm):
        mask = labels == zone
        kind, data = cache[tex_idx]
        if kind == 'direct':
            out[mask] = data[mask]
        else:
            out[mask] = (orig * data)[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

TEX_NAMES = ['glitch', 'scanlines', 'halftone', 'static', 'ripple']

if __name__ == "__main__":
    print("Loading d.jpg and computing segmentation labels...")
    img    = load("d")
    labels = get_labels(img)

    print("Pre-computing texture arrays...")
    cache, orig = build_texture_cache(img, labels)

    all_perms = list(permutations(range(5)))
    total = len(all_perms)
    print(f"Generating {total} permutations -> {DIR_OUT}")

    for i, perm in enumerate(all_perms):
        result = apply_permutation(img, labels, perm, cache)
        fname  = f"perm_{i:03d}__" + "_".join(TEX_NAMES[t] for t in perm)
        save(result, fname)
        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"  {i+1}/{total}")

    print("Done.")

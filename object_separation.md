# Object Separation Skill

Segments an image into contiguous regions corresponding to real objects using K-means on derived per-pixel features, then applies per-cluster artistic filters with optional soft blended boundaries.

When invoked, this skill generates a ready-to-run `object_separation.py` for the current iteration folder.

---

## Core Concept

Standard K-means on RGB or LAB fails because **brightness is the dominant axis** — shadows and highlights of the same object land in different clusters. The fix is to compute per-pixel features that describe *what surface type* a pixel belongs to, independent of lighting.

### Feature vector (per pixel)

| Feature | How computed | Why it helps |
|---|---|---|
| `r_chroma` | `R / (R+G+B)` | Lighting-invariant redness — same in shadow or sunlight |
| `g_chroma` | `G / (R+G+B)` | Lighting-invariant greenness |
| `texture_energy` | `sqrt(E[gray²] - E[gray]²)` in local window | Sky≈0, trees=high, road=medium — identifies surface type |
| `x`, `y` | pixel position / image size × weight | Spatial coherence — nearby pixels prefer same cluster |

No luminance, no L channel, no brightness anywhere in the feature vector.

### Pre-processing

Use `cv2.bilateralFilter` (NOT Gaussian blur). Bilateral is edge-preserving: it smooths within objects but keeps sharp boundaries between them, so features don't bleed across object edges.

### Post-processing

After K-means, apply `scipy.ndimage.median_filter` on the label map to dissolve isolated speckles and merge small islands into their neighbors.

---

## Sample Scripts

### Feature extraction

```python
import numpy as np, cv2
from scipy.ndimage import uniform_filter

def chromaticity(img):
    f = img.astype(np.float32)
    total = f[...,0] + f[...,1] + f[...,2] + 1e-6
    return f[...,0]/total, f[...,1]/total   # r_chroma, g_chroma

def local_texture(img, win=21, weight=0.8):
    gray = np.mean(img.astype(np.float32)/255.0, axis=2)
    mean    = uniform_filter(gray,    size=win)
    mean_sq = uniform_filter(gray**2, size=win)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    mx = std.max()
    return (std/mx if mx > 0 else std) * weight

def build_features(img, spatial_w=0.35):
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(img, 15, 40, 40)
    r_c, g_c = chromaticity(smooth)
    tex      = local_texture(smooth)
    ys, xs   = np.mgrid[0:h, 0:w]
    xn = xs.astype(np.float32) / (w-1) * spatial_w
    yn = ys.astype(np.float32) / (h-1) * spatial_w
    return np.stack([r_c, g_c, tex, xn, yn], axis=2).reshape(-1,5).astype(np.float32)
```

### Segmentation

```python
from scipy.ndimage import median_filter

def segment(img, K=5, label_med=21):
    h, w  = img.shape[:2]
    feats = build_features(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _, labels_flat, _ = cv2.kmeans(feats, K, None, criteria, 12, cv2.KMEANS_PP_CENTERS)
    return median_filter(labels_flat.reshape(h, w), size=label_med)
```

### Hard mask rendering (sharp boundaries)

```python
FILTERS = [bw, sepia, saturate, cool, warm]  # one function per cluster

def apply_filters_hard(img, labels, K):
    out = np.zeros_like(img)
    for i in range(K):
        mask = labels == i
        region = img.copy(); region[~mask] = 0
        out[mask] = FILTERS[i % len(FILTERS)](region)[mask]
    return out
```

### Soft blended rendering (feathered boundaries)

```python
def soft_weights(labels, K, blur_size=101):
    h, w = labels.shape
    raw = np.zeros((h, w, K), dtype=np.float32)
    for i in range(K):
        mask = (labels == i).astype(np.float32)
        bs = blur_size if blur_size % 2 == 1 else blur_size + 1
        raw[..., i] = cv2.GaussianBlur(mask, (bs, bs), 0)
    return raw / (raw.sum(axis=2, keepdims=True) + 1e-6)

def apply_filters_soft(img, labels, K):
    weights = soft_weights(labels, K)                   # (H, W, K)
    filtered = np.stack([
        FILTERS[i % len(FILTERS)](img).astype(np.float32)
        for i in range(K)
    ], axis=3)                                          # (H, W, 3, K)
    w = weights[:, :, np.newaxis, :]                   # (H, W, 1, K)
    return np.clip((filtered * w).sum(axis=3), 0, 255).astype(np.uint8)
```

---

## Parameter Guide

| Parameter | Default | Effect |
|---|---|---|
| `K` | 5 | Number of segments. Lower = bigger blobs. Raise for more detail. |
| `SPATIAL_W` | 0.35 | How much position matters vs color+texture. Raise = more geographically split. |
| `TEXTURE_W` | 0.8 | Texture feature weight. Lower if texture is splitting objects that should be together. |
| `TEXTURE_WIN` | 21 | Window size for local std-dev. Larger = captures broader texture patterns. |
| `BILATERAL_D` | 15 | Bilateral filter diameter. Larger = more smoothing, slower. |
| `LABEL_MED` | 21 | Median filter size on label map. Larger = fewer small islands. |
| `BLEND_BLUR` | 101 | Soft rendering feather width. Larger = wider fade zone at boundaries. |

---

## Iteration History Summary

| Iteration | Key Change | Problem it solved | Remaining issue |
|---|---|---|---|
| 1 | Per-image global artistic effects | Baseline | No object awareness |
| 2 | Gaussian blur + spatial K-means + median filter | Speckled pixel clusters | Still lighting-biased |
| 3 | LAB color space, L weight = 0.1 | Reduced brightness dominance | Gaussian blur bleeds across edges |
| 4 | Chromaticity + texture features + bilateral filter | Lighting invariance; texture separates surface types | Hard boundaries visible |
| 5 | Soft blended weight maps (blur=101) | Sharp seams between filter zones | TBD |

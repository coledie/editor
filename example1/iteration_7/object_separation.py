"""
object_separation.py — iteration 7
------------------------------------
Applies per-zone procedural textures to image d using iteration 6 segmentation.
Each texture is computed as a float multiplier array then applied to the original pixels.

Zone → Texture:
  0 (red)    right forest/clouds  → TV glitch: row shifts + chromatic aberration + noise
  1 (blue)   sky                  → horizontal scanlines (CRT screen feel)
  2 (green)  left forest/cliff    → halftone dot grid (newspaper/offset print)
  3 (yellow) foreground rocks     → heavy TV static noise multiplier
  4 (purple) waterfall/mist       → concentric ripple rings centered on the falls
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter

DIR_IN  = "C:/Users/coles/Desktop/editor/example1"
DIR_OUT = "C:/Users/coles/Desktop/editor/example1/iteration_7"

# ── segmentation (same as iteration 6) ───────────────────────────────────────

K=5; SPATIAL_W=0.35; TEXTURE_W=0.8; TEXTURE_WIN=21
BILATERAL_D=15; BILATERAL_SC=40; BILATERAL_SS=40
LABEL_MED=21; ONEHOT_BLUR=301

def load(name):
    img = cv2.imread(f"{DIR_IN}/{name}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save(arr, name):
    out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{DIR_OUT}/{name}.jpg", out)
    print(f"  saved {name}.jpg")

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
    # one-hot → wide blur → argmax
    bs = ONEHOT_BLUR if ONEHOT_BLUR%2==1 else ONEHOT_BLUR+1
    onehot = np.zeros((h,w,K), np.float32)
    for i in range(K):
        onehot[...,i] = (labels==i).astype(np.float32)
    blurred = np.stack([cv2.GaussianBlur(onehot[...,i],(bs,bs),0) for i in range(K)], axis=2)
    return np.argmax(blurred, axis=2).astype(np.int32)

# ── texture generators ────────────────────────────────────────────────────────

def tex_scanlines(h, w, line_gap=4, dim=0.25):
    """Horizontal CRT scanlines — alternating bright/dim rows."""
    pat = np.ones((h, w), np.float32)
    pat[::line_gap] = dim
    return pat[..., np.newaxis]   # (H,W,1) for broadcast over RGB

def tex_halftone(h, w, spacing=14, radius_frac=0.42):
    """Regular grid of circular dots — newspaper offset print feel."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # nearest dot centre
    cy = (yy // spacing) * spacing + spacing / 2
    cx = (xx // spacing) * spacing + spacing / 2
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    dot = (dist < spacing * radius_frac).astype(np.float32)
    # dots = 1.0 (show original), gaps = 0.15 (nearly black)
    pat = dot * 0.85 + 0.15
    return pat[..., np.newaxis]

def tex_tv_static(h, w, base_noise=0.55, glitch_frac=0.06):
    """
    Per-pixel random noise multiplier + horizontal glitch bands.
    glitch_frac of rows get a high-intensity shift of the noise floor.
    """
    noise = np.random.uniform(1 - base_noise, 1 + base_noise, (h, w)).astype(np.float32)
    # glitch rows: random fraction of rows get blown-out noise
    glitch_rows = np.random.choice(h, int(h * glitch_frac), replace=False)
    noise[glitch_rows] = np.random.uniform(0.1, 2.2, (len(glitch_rows), w))
    return noise[..., np.newaxis]

def tex_glitch_rgb(img_region):
    """
    TV glitch applied directly (not a multiplier — replaces pixels).
    Row-level horizontal shifts + per-channel chromatic aberration.
    """
    out = img_region.astype(np.float32).copy()
    h, w = out.shape[:2]
    # shift ~12% of rows randomly
    shift_rows = np.random.choice(h, max(1, h//8), replace=False)
    for y in shift_rows:
        s = int(np.random.choice([-1,1]) * np.random.randint(8, 30))
        out[y] = np.roll(out[y], s, axis=0)
    # chromatic aberration on another set of rows
    aber_rows = np.random.choice(h, max(1, h//6), replace=False)
    for y in aber_rows:
        s = np.random.randint(4, 14)
        out[y,:,0] = np.roll(out[y,:,0], s)    # red channel shifts right
        out[y,:,2] = np.roll(out[y,:,2], -s)   # blue channel shifts left
    # global noise
    noise = np.random.uniform(0.6, 1.4, (h, w, 1))
    out = out * noise
    return np.clip(out, 0, 255).astype(np.uint8)

def tex_mosaic(img_region, block=22):
    """Pixelation: downsample then nearest-neighbour upsample."""
    h, w = img_region.shape[:2]
    small = cv2.resize(img_region, (max(1, w//block), max(1, h//block)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def tex_ripple_rings(h, w, cy=None, cx=None, freq=18, depth=0.45):
    """
    Concentric sinusoidal rings emanating from (cy, cx).
    Multiplier oscillates between (1-depth) and (1+depth).
    """
    if cy is None: cy = h // 2
    if cx is None: cx = w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    pat = 1.0 + depth * np.sin(dist / freq * 2 * np.pi).astype(np.float32)
    return pat[..., np.newaxis]

# ── apply textures per zone ───────────────────────────────────────────────────

def apply_textures(img, labels):
    h, w = img.shape[:2]
    out = np.zeros_like(img, dtype=np.float32)

    for zone in range(K):
        mask = labels == zone
        region = img.copy()
        region[~mask] = 0
        orig = img.astype(np.float32)

        if zone == 0:   # right forest/clouds → halftone dots (swapped from green)
            tex = tex_halftone(h, w, spacing=13, radius_frac=0.4)
            out[mask] = (orig * tex)[mask]

        elif zone == 1: # sky → scanlines
            tex = tex_scanlines(h, w, line_gap=3, dim=0.2)
            out[mask] = (orig * tex)[mask]

        elif zone == 2: # left forest → TV static noise (swapped from red)
            np.random.seed(42)
            tex = tex_tv_static(h, w, base_noise=0.6, glitch_frac=0.08)
            out[mask] = (orig * tex)[mask]

        elif zone == 3: # foreground rocks → TV glitch (swapped from red)
            glitched = tex_glitch_rgb(img)
            out[mask] = glitched.astype(np.float32)[mask]

        elif zone == 4: # waterfall/mist → concentric ripple rings
            # centre rings on approximate waterfall position
            cy, cx = int(h * 0.55), int(w * 0.5)
            tex = tex_ripple_rings(h, w, cy=cy, cx=cx, freq=16, depth=0.5)
            out[mask] = (orig * tex)[mask]

    return np.clip(out, 0, 255).astype(np.uint8)

# ── segment colorisation for reference ───────────────────────────────────────

PALETTE = np.array([[231,76,60],[52,152,219],[46,204,113],[241,196,15],[155,89,182]], np.uint8)

def colorize(labels):
    out = np.zeros((*labels.shape, 3), np.uint8)
    for i in range(K):
        out[labels==i] = PALETTE[i]
    return out

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading d.jpg and computing labels...")
    img    = load("d")
    labels = get_labels(img)
    save(colorize(labels), "d_segments")

    print("Applying per-zone textures...")
    result = apply_textures(img, labels)
    save(result, "d_textured")
    print("Done.")

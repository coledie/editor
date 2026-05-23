# Texture Skill

Procedural per-pixel texture patterns applied to image regions at configurable opacity.
Textures are generated as float arrays and composited onto the source image — ranging from
a subtle multiplier overlay all the way to complete pixel replacement.

When invoked, this skill generates texture application code for the current iteration.

---

## Opacity Model

Every texture has two application modes controlled by a single `opacity` float in `[0.0, 1.0]`:

```python
def apply_texture(original, texture_rgb, opacity=1.0):
    """
    opacity=0.0  → original image unchanged
    opacity=0.5  → 50/50 blend between original and textured result
    opacity=1.0  → full texture replacement / multiplier
    """
    orig_f = original.astype(np.float32)
    tex_f  = texture_rgb.astype(np.float32)
    blended = orig_f * (1.0 - opacity) + tex_f * opacity
    return np.clip(blended, 0, 255).astype(np.uint8)
```

**Multiplier textures** generate a float pattern array `(H, W, 1)` in roughly `[0, 2]`,
then produce the textured image as `original * pattern`. Pass that result into `apply_texture`
to control how strongly it's blended back over the untouched original.

**Replacement textures** (glitch, mosaic) produce a full RGB result directly.
Pass into `apply_texture` the same way.

### Opacity presets

| Opacity | Feel |
|---|---|
| 0.15 – 0.3 | Subtle hint — barely perceptible, adds micro-detail |
| 0.4 – 0.6 | Balanced — texture reads clearly, original still dominant |
| 0.7 – 0.85 | Strong — texture is the dominant visual element |
| 1.0 | Full replacement — original only visible through the texture pattern itself |

---

## Texture Catalogue

### 1. Scanlines
Horizontal CRT/TV scanlines — alternating bright and dimmed rows.

```python
def tex_scanlines(h, w, line_gap=3, dim=0.2):
    """
    line_gap  : rows between each dark line (3 = very dense, 8 = sparse)
    dim       : brightness of the dark row (0.0 = black bars, 0.5 = subtle)
    """
    pat = np.ones((h, w), np.float32)
    pat[::line_gap] = dim
    textured = original.astype(np.float32) * pat[..., np.newaxis]
    return np.clip(textured, 0, 255).astype(np.uint8)
```

**Iteration 7 params**: `line_gap=3, dim=0.2, opacity=1.0` → sky zone  
**Variation ideas**: `line_gap=2` for ultra-dense interlacing; `dim=0.0` for full black bars;
combine with a slight horizontal offset per row for interlace shimmer.

---

### 2. Halftone Dots
Regular grid of circular dots — newspaper offset-print / pop-art feel.

```python
def tex_halftone(h, w, spacing=13, radius_frac=0.42):
    """
    spacing     : pixels between dot centres (smaller = finer print)
    radius_frac : dot radius as fraction of spacing (0.5 = touching dots)
    inside dots : multiplied by 0.85 (preserve original colour)
    between dots: multiplied by 0.15 (near-black gaps)
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (yy // spacing) * spacing + spacing / 2
    cx = (xx // spacing) * spacing + spacing / 2
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    dot = (dist < spacing * radius_frac).astype(np.float32)
    pat = dot * 0.85 + 0.15          # dots bright, gaps near-black
    textured = original.astype(np.float32) * pat[..., np.newaxis]
    return np.clip(textured, 0, 255).astype(np.uint8)
```

**Iteration 7 params**: `spacing=13, radius_frac=0.4, opacity=1.0` → left forest zone  
**Variation ideas**:
- `spacing=6` for very fine halftone (photo reproduction feel)
- `spacing=30, radius_frac=0.48` for large Ben-Day dots (Lichtenstein pop-art)
- Vary `radius_frac` per pixel based on local luminance → proper halftone screening

---

### 3. TV Static Noise
Per-pixel random brightness noise + blown-out glitch rows.

```python
def tex_tv_static(h, w, base_noise=0.6, glitch_frac=0.08, seed=None):
    """
    base_noise  : half-width of uniform noise range — 0.6 → pixels vary ±60%
    glitch_frac : fraction of rows randomly blown to extreme values [0.1, 2.2]
    seed        : set for reproducible static pattern
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(1 - base_noise, 1 + base_noise, (h, w)).astype(np.float32)
    glitch_rows = np.random.choice(h, int(h * glitch_frac), replace=False)
    noise[glitch_rows] = np.random.uniform(0.1, 2.2, (len(glitch_rows), w))
    textured = original.astype(np.float32) * noise[..., np.newaxis]
    return np.clip(textured, 0, 255).astype(np.uint8)
```

**Iteration 7 params**: `base_noise=0.6, glitch_frac=0.08, seed=42, opacity=1.0` → foreground zone  
**Variation ideas**:
- `base_noise=0.2` for gentle film grain
- `base_noise=0.9, glitch_frac=0.2` for aggressive total signal breakdown
- Apply only to luminance channel (convert to YCbCr) for colour-preserving grain

---

### 4. TV Glitch / Chromatic Aberration
Row-level horizontal pixel shifts + RGB channel split. Direct replacement.

```python
def tex_glitch_rgb(img, shift_frac=0.12, aber_frac=0.16,
                   shift_range=(8,30), aber_range=(4,14)):
    """
    shift_frac  : fraction of rows that get a full horizontal row-shift
    aber_frac   : fraction of rows that get R/B channel split
    shift_range : (min, max) pixels to shift a glitched row
    aber_range  : (min, max) pixels for chromatic aberration
    """
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    shift_rows = np.random.choice(h, max(1, int(h * shift_frac)), replace=False)
    for y in shift_rows:
        s = int(np.random.choice([-1,1]) * np.random.randint(*shift_range))
        out[y] = np.roll(out[y], s, axis=0)
    aber_rows = np.random.choice(h, max(1, int(h * aber_frac)), replace=False)
    for y in aber_rows:
        s = np.random.randint(*aber_range)
        out[y,:,0] = np.roll(out[y,:,0],  s)   # red shifts right
        out[y,:,2] = np.roll(out[y,:,2], -s)   # blue shifts left
    noise = np.random.uniform(0.6, 1.4, (h, w, 1))
    return np.clip(out * noise, 0, 255).astype(np.uint8)
```

**Iteration 7 params**: defaults above, `opacity=1.0` → right forest zone  
**Variation ideas**:
- `shift_frac=0.3, shift_range=(20,80)` for heavy VHS tape damage
- `aber_frac=1.0, aber_range=(2,4)` for subtle full-frame colour fringing
- Apply glitch in vertical bands instead of rows for a different breakdown feel

---

### 5. Concentric Ripple Rings
Sinusoidal brightness rings radiating from a centre point.

```python
def tex_ripple_rings(h, w, cy=None, cx=None, freq=16, depth=0.5):
    """
    cy, cx : ring centre in pixels (defaults to image centre)
    freq   : pixel distance between ring peaks (smaller = tighter rings)
    depth  : oscillation amplitude — 0.5 → multiplier oscillates [0.5, 1.5]
    """
    if cy is None: cy = h // 2
    if cx is None: cx = w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    pat = 1.0 + depth * np.sin(dist / freq * 2 * np.pi).astype(np.float32)
    textured = original.astype(np.float32) * pat[..., np.newaxis]
    return np.clip(textured, 0, 255).astype(np.uint8)
```

**Iteration 7 params**: `cy=h*0.55, cx=w*0.5, freq=16, depth=0.5, opacity=1.0` → waterfall zone  
**Variation ideas**:
- `depth=0.9` for near-black troughs between rings (very dramatic)
- `freq=6` for extremely tight interference pattern
- Use elliptical distance `sqrt((dy/ry)² + (dx/rx)²)` for oval rings
- Animate by shifting phase: `sin(dist/freq * 2π + time_offset)`

---

## Applying Textures to K-Means Zones

Standard pattern for applying different textures per segmentation zone:

```python
def apply_zone_textures(img, labels, zone_textures):
    """
    zone_textures : dict mapping cluster_id → (textured_img_fn, opacity)
                    textured_img_fn takes (img) and returns (H,W,3) uint8
    """
    h, w = img.shape[:2]
    out = np.zeros((h, w, 3), np.float32)
    for zone_id, (tex_fn, opacity) in zone_textures.items():
        mask = labels == zone_id
        textured = tex_fn(img).astype(np.float32)
        orig     = img.astype(np.float32)
        blended  = orig * (1.0 - opacity) + textured * opacity
        out[mask] = blended[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# Example usage (iteration 7 d.jpg assignments):
zone_textures = {
    0: (lambda img: tex_glitch_rgb(img),                          1.0),
    1: (lambda img: apply_mul(img, tex_scanlines(h,w,3,0.2)),     1.0),
    2: (lambda img: apply_mul(img, tex_halftone(h,w,13,0.4)),     1.0),
    3: (lambda img: apply_mul(img, tex_tv_static(h,w,0.6,0.08)), 1.0),
    4: (lambda img: apply_mul(img, tex_ripple_rings(h,w,cy,cx)),  1.0),
}
```

---

## Reproducibility Note

K-means is non-deterministic — cluster indices shift between runs, so textures land on
different zones. To lock assignments: save labels after first run and reload:

```python
# Save
np.save(f"{DIR_OUT}/d_labels.npy", labels)

# Load (skip re-segmentation)
labels = np.load(f"{DIR_OUT}/d_labels.npy")
```

---

## Iteration History

| Iteration | Change | Result |
|---|---|---|
| 7 | First texture pass on d — 5 procedural textures per zone at opacity=1.0 | Concentric rings on waterfall outstanding; halftone on forest clean; glitch on right forest; static on foreground |

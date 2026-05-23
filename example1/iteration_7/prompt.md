## User Prompt
Take d from iteration 6 and apply very interesting textures to each section — applied as a multiplier. Ideas: grayscale, crazy TV pixelation, concentric rings, etc.

## Problem Being Solved
Previous iterations applied colour filters (sepia, B&W, saturate). This iteration moves to procedural texture patterns — structural overlays that modulate the original pixel brightness rather than replacing the colour palette.

## Approach
Re-run iteration 6 segmentation on image d to get the 5 zone labels.
For each zone, generate a float texture array (H×W or H×W×1) and multiply against the original RGB image. Zones:

| Zone | Scene region | Texture | Effect |
|---|---|---|---|
| 0 (red) | Right forest / building | `tex_glitch_rgb` | Row-level horizontal shifts + per-row chromatic aberration (R/B channel split) + random noise. Applied directly (replaces pixels). |
| 1 (blue) | Sky / lower cliff | `tex_scanlines` | Horizontal CRT scanlines — every 3rd row dimmed to 20% brightness. Pure multiplier. |
| 2 (green) | Left forest / cliff | `tex_halftone` | Regular 13px-spaced dot grid. Inside dots: ×0.85. Between dots: ×0.15 (near-black). Newspaper/offset-print look. |
| 3 (yellow) | Foreground | `tex_tv_static` | Per-pixel random uniform noise in [0.4, 1.6]. 8% of rows randomly blown out to [0.1, 2.2]. Heavy grain. |
| 4 (purple) | Waterfall / mist | `tex_ripple_rings` | Sinusoidal concentric rings: `1 + 0.5 * sin(dist/16 * 2π)` centred at (55% height, 50% width). Rings radiate from the falls. |

## Key Parameters
- Halftone spacing=13, radius_frac=0.4
- Scanlines line_gap=3, dim=0.2
- TV static base_noise=0.6, glitch_frac=0.08
- Glitch: ~12% rows shifted ±8–30px, ~16% rows get R/B channel aberration ±4–14px
- Ripple rings freq=16, depth=0.5, centre=(55%h, 50%w)

## Result Assessment
Concentric rings on the waterfall are the standout effect — highly dramatic, radiating from the falls through the mist.
Halftone dots cleanly overlay the left forest without destroying the foliage detail.
TV glitch on the right forest adds digital corruption over the trees.
Note: K-means cluster order is non-deterministic — texture-to-zone assignments may shift between runs. Consider saving/loading labels as .npy for reproducibility.

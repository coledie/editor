## User Prompt
Apply different interesting artistic/design effects to images a, b, c, d using NumPy-based computer vision. Each image should get a unique effect.

## Problem Being Solved
No object separation — this was pure per-image artistic effect, not per-object. Baseline iteration.

## Approach
Each image got a hand-picked global filter:
- **a** (highway): Scharr gradient orientation mapped to HSV rainbow — edge angles become hue
- **b** (sushi): K-means posterization (6 color clusters) with boosted saturation
- **c** (industrial): Heavy sharpen kernel + dilated Canny edges coloured cyan
- **d** (waterfall): Green/blue channel boost + Laplacian glow overlay

## Key Parameters
- K=6 color-only clusters for b
- Sharpen kernel: `[0,-1,0],[-1,6,-1],[0,-1,0]`
- Canny thresholds: 60/140

## Result Assessment
Visually interesting but no awareness of image regions or objects. The effects are applied uniformly to the whole image.
Next step: separate image into spatial regions before applying filters.

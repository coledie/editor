## User Prompt
Iteration 2 and 3 suck — they don't encapsulate objects well at all. Calculate more per-pixel values based on some heuristic and use that in K-means to really capture objects well.

## Problem Being Solved
Previous iterations used RGB or LAB color values which encode *how bright* a pixel is. Lighting variation across a scene makes pixels of the same object look different. We need features that describe *what kind of surface* a pixel belongs to, not how much light is hitting it.

## Approach
Two new derived per-pixel features added alongside chromaticity:

**1. Chromaticity (r_chroma, g_chroma)**
Instead of raw R, G, B, normalize: `r = R/(R+G+B)`, `g = G/(R+G+B)`.
This removes all brightness information. A rock in shadow and a rock in sunlight have identical chromaticity. No L channel, no luminance at all.

**2. Local texture energy**
For every pixel, compute the standard deviation of grayscale values in a 21×21 local window:
`std = sqrt(E[x²] - E[x]²)` via uniform_filter (fast, no loops).
Sky = near-zero texture. Trees/foliage = high texture. Road/rock = medium.
This captures *what material/surface type* the pixel is part of — completely independent of lighting.

**Pre-processing: bilateral filter** instead of Gaussian. Edge-preserving: smooths within objects but keeps sharp boundaries between them, so texture and chromaticity features don't bleed across object edges.

## Key Parameters
- K=5
- Features: (r_chroma ×1.0, g_chroma ×1.0, texture ×0.8, x ×0.35, y ×0.35)
- BILATERAL_D=15, SC=40, SS=40
- LABEL_MED=21

## Result Assessment
Significantly better on a and c:
- **a**: Sky cleanly separated. Individual tree branches isolated from sky and from the cliff band — texture feature doing its job (branches = high texture, sky = zero).
- **c**: Window frame, sky zones, container mid-ground, and floor all in distinct large regions.
- **d**: Sky, forest, and waterfall zone distinguishable.
- **b**: Still fragmented — sushi pieces are small, have complex overlapping colors, and are close together, limiting what spatial features can do.

Remaining issue: still some small islands in b and c mid-section. Could increase LABEL_MED or raise spatial weight specifically for those images.

## User Prompt
Go back to K-means only. Add x, y pixel coordinates as explicit feature columns. Run K = 3, 4, 5.

## Problem Being Solved
Iteration 8 explored three algorithms but SLIC and Felzenszwalb produced noisy, fragmented zone maps when merged. Simpler K-means with well-designed features is more controllable. Explicitly documenting the spatial features (x, y as columns) makes the feature engineering transparent and easier to tune.

## Approach
Feature vector per pixel: `[L, A, B, x_scaled, y_scaled]`
- L, A, B from OpenCV LAB conversion (perceptually uniform color space)
- x_scaled = (col / (w-1)) × SPATIAL_W  — runs 0 → SPATIAL_W across image width
- y_scaled = (row / (h-1)) × SPATIAL_W  — runs 0 → SPATIAL_W across image height

SPATIAL_W=40 puts spatial spread comparable to LAB channel ranges, giving location meaningful but not dominant weight. Post-processing: median filter (size=19) then per-class gaussian blur (kernel=151) + argmax for soft zone boundaries.

K_VALUES = [3, 4, 5] each get their own `k{k}/` output folder.
Permutations: P(5, k) — all ordered selections of k textures from 5.

Output layout: `iteration_9/k{k}/{image}/segments.jpg` + `permutations/`.

## Key Parameters
- SPATIAL_W = 40
- KM_ITERATIONS = 100, KM_ATTEMPTS = 15, KM_EPS = 0.05
- MEDIAN_SIZE = 19, BLUR_SIZE = 151
- K=3 → 60 perms, K=4 → 120 perms, K=5 → 120 perms

## Result Assessment
Explicit [L,A,B,x,y] features produce clean spatially-coherent zones that respect both color similarity and proximity.
K=3 gives the broadest, most dramatic texture regions.
K=5 starts to fragment — especially on images a and c which have complex mid-ground transitions.
SPATIAL_W is the key tuning lever: increasing it makes zones more stripe-like (spatial dominates), decreasing it makes them more color-blob-like.

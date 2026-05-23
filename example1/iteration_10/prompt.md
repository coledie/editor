## User Prompt
Run another iteration — add an edge-detecting convolution as a feature column for K-means.

## Problem Being Solved
Iteration 9 used [L, A, B, x, y] — purely color and position. Edges (boundaries between objects) had no explicit representation, so K-means boundaries were driven only by color similarity and spatial proximity. Adding a structural edge signal should help zones align with real object contours.

## Approach
Added a 6th feature column: Scharr gradient magnitude on the LAB L channel, normalized to [0, 1] and scaled by EDGE_W=30.

Feature vector per pixel: `[L, A, B, x_scaled, y_scaled, edge_scaled]`

Scharr kernels are used (not Sobel) — they are optimised for rotational accuracy and give a more precise gradient estimate from 3×3 support:
  Kx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
  magnitude = sqrt(Scharr_x² + Scharr_y²), normalized to [0, 1]

The edge feature means pixels on strong edges form their own attractor in feature space, separate from flat-color interior regions of the same color. This encourages zone boundaries to track real contours rather than cutting through uniform surfaces.

K_VALUES = [3, 4, 5], images a, c, d. All P(5, k) texture permutations per result.

## Key Parameters
- EDGE_W = 30 (edge column scale — raise to force harder edge-based splitting)
- SPATIAL_W = 40
- KM_ITERATIONS = 100, KM_ATTEMPTS = 15
- MEDIAN_SIZE = 19, BLUR_SIZE = 151

## Result Assessment
K=3 on d.jpg: sky / forest / waterfall+mist separate cleanly, with the waterfall column
isolated as its own zone — edge signal helped pull the falls boundary from the surrounding cliff.
K=4 adds a distinct rock/cliff zone.
Edge column adds structural coherence over iteration 9; zones follow contours more tightly
rather than bleeding across high-contrast boundaries.
EDGE_W is the key new lever: increase to bias toward edge-following, decrease to fall back
toward color-region behavior.

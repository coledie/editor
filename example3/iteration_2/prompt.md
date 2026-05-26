## User Prompt
iter 2 include x y in the columns

## Problem Being Solved
Iteration 1 clustered on LAB color only, so spatially disconnected pixels with
similar color landed in the same zone (e.g. mist patches and sky scattered
together). Adding x,y to the feature vector encourages contiguous regions.

## Approach
Append normalized spatial coordinates as two extra k-means feature columns:

    feats = [L*0.3, a, b, x_norm*SPATIAL_W, y_norm*SPATIAL_W]

`x_norm`, `y_norm` are in [0,1] and scaled by `SPATIAL_W=40` so they sit on a
magnitude comparable to LAB chroma (a,b ∈ roughly [-60, 60] for this image).
Everything else (bilateral pre-filter, median post-filter, CA overlay, α=0.5)
is identical to iteration_1.

## Key Parameters
- `SPATIAL_W = 40.0` — spatial weight in LAB-comparable units. Higher → more
  spatial coherence, less color fidelity.
- `L_WEIGHT = 0.3` — unchanged, keeps chroma driving color decisions.
- CA rules / α / seed — unchanged from iter_1 for direct comparison.

## Result Assessment
TBD. Compare `k4/segments.jpg` and `k5/segments.jpg` against iteration_1: zones
should now be larger, more connected blobs. If they over-fragment vertically/
horizontally instead, lower `SPATIAL_W`. If color separation is lost (e.g.
forest and waterfall merge), raise `L_WEIGHT` slightly or lower `SPATIAL_W`.

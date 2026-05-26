## User Prompt
Its not taking the x and y seriously

## Problem Being Solved
Iteration 2 added x, y to the feature vector but `SPATIAL_W=40` was in raw LAB
units. LAB chroma (a, b) easily ranges 60+ units in a real photo, so spatial
features were drowned out and segmentation looked nearly identical to color-
only k-means.

## Approach
**Z-score normalize every feature first**, then apply explicit per-feature
weights. After standardization, all five axes have unit variance, so the
weights become direct importance multipliers in σ-units:

    Lz * 0.3     a_z * 1.0     b_z * 1.0     x_z * 3.0     y_z * 3.0

With `SPATIAL_W=3.0`, position is roughly 3× as influential per axis as color
chroma — strong enough to force contiguous blobs while still respecting color
where the gradient is sharp.

## Key Parameters
- `SPATIAL_W = 3.0` — in σ-units, so this is a real "spatial dominates color"
  setting. Lower (1.0–2.0) for more color-driven; higher (5–10) for a near-grid.
- `L_WEIGHT = 0.3`, `AB_WEIGHT = 1.0` — unchanged intent: chroma over lightness.
- Bilateral, median, CA settings unchanged from iter_2.

## Result Assessment
TBD. If clusters now look like horizontal/vertical strips (a sign spatial is
too strong), drop `SPATIAL_W` to 2.0. If they still feel color-only, raise to
5.0 or 8.0.

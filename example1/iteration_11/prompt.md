## User Prompt
New iteration — use watershed segmentation.

## Problem Being Solved
Previous iterations (9, 10) used K-means as the sole clustering mechanism, which partitions pixels by feature distance but doesn't natively follow image edges. Watershed is a topographic algorithm: it floods from seed points along a gradient surface, so boundaries naturally land on edge ridges rather than cutting across uniform regions.

## Approach
1. **Gradient surface** — Scharr magnitude on the LAB L channel, normalized [0, 1]. This is the "elevation" the flood climbs.
2. **Seed detection** — pixels below the FLAT_PERCENTILE gradient threshold are "flat" regions. A distance transform on these flat pixels gives depth-into-flatness; local maxima spaced >= SEED_SPACING px apart become watershed seeds (one per basin).
3. **Flood** — `skimage.segmentation.watershed(gradient, markers, compactness=COMPACTNESS)`. Low compactness = pure gradient-following; higher = rounder regions.
4. **Merge** — many fine watershed segments collapsed to k zones via k-means on mean LAB color per segment (same merge step as iteration 8 SLIC/Felzenszwalb).
5. **Smooth** — median filter + per-class gaussian blur + argmax for soft boundaries.

K_VALUES = [3, 4, 5]. All P(5, k) texture permutations per image.

## Key Parameters
- FLAT_PERCENTILE = 55  (gradient threshold for seed region detection)
- SEED_SPACING    = 35  (min px between seeds — lower = more initial segments)
- COMPACTNESS     = 0.0005  (0 = pure gradient; raise for rounder zones)
- MERGE_SMOOTH    = 15, MEDIAN_SIZE = 19, BLUR_SIZE = 151

## Result Assessment
Watershed produces distinctly organic, amoeba-like zone shapes compared to k-means blobs.
The sky separates cleanly as one zone. The waterfall column is isolated by gradient ridges on its
edges and appears as its own zone in K=3. K=4 breaks the forest into separate near/far regions.
Zone boundaries are noticeably more contour-hugging than previous iterations.
SEED_SPACING and FLAT_PERCENTILE are the key levers: wider spacing = fewer, larger initial
segments; lower flat_percentile = seeds only in very flat regions, yielding crisper boundaries.

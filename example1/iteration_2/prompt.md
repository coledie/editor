## User Prompt
I want larger contiguous spaces. There are tons of green and blue pixels scattered next to each other — it's not K-means enough.

## Problem Being Solved
Iteration 1 K-means used only color features (R,G,B) with a low spatial weight (0.4). This produced fragmented speckles: similar colors scattered across the image all got the same label regardless of location.

## Approach
Three changes to force larger, cleaner blobs:
1. **Heavy Gaussian pre-blur** (41×41 kernel) — collapses fine texture and noise within objects so K-means sees uniform color patches rather than individual pixel variations
2. **Higher spatial weight** (0.7 vs 0.4) — position pulls clusters into spatially coherent zones
3. **Median filter on label map** (size=31) — a post-processing pass that replaces every label with the most common label in its 31×31 neighborhood, dissolving isolated islands

Also reduced K from 5 to 4 — fewer clusters means each one is forced to cover a larger region.

## Key Parameters
- K=4
- BLUR_K=41 (Gaussian kernel size)
- SPATIAL_W=0.7
- LABEL_MED=31

## Result Assessment
Much larger contiguous blobs with smooth edges. Sky, ground, mid-ground separated cleanly on most images.
Remaining problem: lighting is still the primary clustering axis. The left half of image a (darker, in shadow) separates from the right half even though it's the same scene layer. K-means in RGB is treating brightness as a primary feature.

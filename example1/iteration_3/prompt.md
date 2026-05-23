## User Prompt
This is still incorrect — likely too lighting-based as well. The segmentation seems to be following brightness/lighting rather than the actual objects in the images.

## Problem Being Solved
RGB K-means treats brightness as a primary feature. Even with spatial weighting, dark shadows and bright highlights of the same object type land in different clusters. The previous result split image a vertically by exposure, not by scene content.

## Approach
Switched from RGB to **LAB color space** and aggressively downweighted the L (lightness) channel:
- L encodes brightness — nearly irrelevant for object identity → weight 0.1
- a encodes green↔red color opposition → weight 1.0
- b encodes blue↔yellow color opposition → weight 1.0

Two pixels of the same object color (e.g. a rock in shadow vs in sun) now have nearly identical feature vectors and land in the same cluster, because L contributes only ~5% of the distance.

## Key Parameters
- K=5
- L_WEIGHT=0.1, AB_WEIGHT=1.0, SPATIAL_W=0.5
- BLUR_K=31, LABEL_MED=25

## Result Assessment
Still over-segmenting by lighting. The LAB approach helped but didn't fully solve the problem because:
1. The Gaussian pre-blur blends across object edges, merging lighting gradients INTO object color estimates
2. When an image has a strong brightness gradient (e.g. blown-out sky in c, uneven exposure in a), the `a` and `b` channels still shift subtly with lighting — downweighting L alone isn't sufficient

Root cause: the pre-processing (Gaussian blur) is not edge-preserving. It spreads lighting information across boundaries.
Next: replace Gaussian with bilateral filter (edge-preserving) and add pyrMeanShiftFiltering to find perceptual density modes before K-means runs.

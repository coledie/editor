## User Prompt
c and d are good for iter 5 — let's blur the K-means results before rendering on the image, like a heavy blur. Also convert object_separation.py to a .md skill file and update editor.md on how Python files are managed.

## Problem Being Solved
Hard binary cluster masks produce visible sharp seams where one filter cuts to another. The boundary looks artificial. We want the filter transitions to feel organic — fade from one treatment into the next over a wide soft zone.

## Approach
**Soft weight map blending:**
1. For each cluster i, create a binary float mask (1.0 where label==i, 0.0 elsewhere)
2. Apply a heavy Gaussian blur (101×101) to each mask → soft 0–1 weight field
3. Stack all K weight maps and normalize at every pixel so weights sum to 1.0
4. Apply every filter to the *full* image (not masked), producing K full-size filtered images
5. Weighted sum: `output = Σ filtered_i * weight_i`

Result: at the center of a cluster, you get nearly 100% of that filter. At boundaries, you get a smooth interpolation between adjacent filters over ~50–100px.

Segmentation is identical to iteration 4 (chromaticity + texture + bilateral + median filter).

## Key Parameters
- BLEND_BLUR=101 (Gaussian kernel on weight maps — controls feather width)
- All other params same as iteration 4

## Result Assessment
TBD after run.

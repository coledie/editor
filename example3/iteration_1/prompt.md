## User Prompt
Make example3/ include d.png and do K-means K=4 / K=5 segmentation, then
overlay a Wolfram 1D elementary cellular automaton over each segment as a 50/50
mean blend with the underlying image.

## Problem Being Solved
First iteration in `example3/` — no prior baseline. Goal is to see how
elementary CA patterns (rules 30/90/110/150/184/…) interact with semantically
distinct image regions when keyed to a k-means segmentation of `d.png`.

## Approach
1. **Segment** with bilateral-filtered LAB k-means. L is downweighted ×0.3
   so brightness gradients don't dominate the clusters. `median_filter` cleans
   speckle. K = 4 and K = 5.
2. **CA per zone** — each cluster gets a distinct elementary CA rule from
   `CA_RULES = [30, 90, 110, 150, 184, …]`, alternating between a single-cell
   seed and a random 50% seed. Grid shape matches the image (H rows × W cols),
   so each scanline of the image is one CA time step.
3. **Blend** at `alpha = 0.5` (true mean) per zone, masked to that zone only.

## Key Parameters
- `BILATERAL_D=15`, `BILATERAL_SIGMA=60` — strong edge-preserving smoothing.
- `L *= 0.3` in LAB feature vector — keeps chroma in charge of clustering.
- `MEDIAN_SIZE=25` — dissolves small islands without erasing thin structures.
- `CA_ALPHA=0.5` — 50/50 mean exactly as requested.
- `CA_SEED=7` — reproducible random initial conditions.

## Result Assessment
TBD after viewing `k4/overlay.jpg` and `k5/overlay.jpg`. Likely next moves:
- Tune `CA_ALPHA` per zone (e.g. light alpha on sky, heavy on rocks).
- Render CA in zone-tinted color instead of B/W for richer composites.
- Try different rule sets per K (e.g. all chaotic, or all class-4).
- Resize CA so a single CA "cell" maps to N×N image pixels (chunkier look).

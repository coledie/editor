## User Prompt
Need a much wider blur then threshold — may need to reformat how you encode which group stuff belongs to, e.g. one-hot encoding.

## Problem Being Solved
Iteration 5 blurred float weight maps derived from K-means labels. The problem: the blur was operating on already-noisy data. Isolated speckles and thin tendrils in the raw label map got blurred but never fully suppressed — they bled into nearby regions and created lumpy, uncertain boundaries. The blur radius was also limited because blurring raw float masks doesn't have a meaningful "winning" threshold.

## Approach
Reformatted cluster membership as one-hot vectors before blurring.

1. **One-hot encode**: pixel belonging to cluster i → float vector [0,...,1,...,0] of length K (one channel per cluster, each channel is a clean binary image)
2. **Wide Gaussian blur per channel** (kernel=301): each blurred channel now encodes "what fraction of the ~300px neighborhood belongs to cluster i" — a local density estimate, not a raw label
3. **Argmax across channels**: assign each pixel to whichever cluster has the highest neighborhood density

This is fundamentally different: it's a majority vote across a 300px radius. Isolated pixels and small patches that are outvoted by their surroundings get reassigned. Only large dominant regions survive. Boundaries fall where two cluster densities are equal — which produces smooth, curved Voronoi-like edges.

## Key Parameters
- ONEHOT_BLUR=301 (neighborhood voting radius in pixels)
- All segmentation params identical to iteration 4/5

## Result Assessment
Segment maps now look like smooth organic blobs — large, clean, no speckling, natural curved boundaries. Like a stylized map or stained glass.
- a: 5 large regions covering sky, treeline band, cliff, road, foreground rocks
- b: Very simplified (3-4 dominant zones) — sushi plate merges to a few large color areas
- c: Sky, window frame, containers, mid-ground, floor clearly delineated in smooth large zones
- d: Sky, left forest, right forest/cliff, waterfall zone — clean curved boundaries

Filtered outputs to be evaluated.

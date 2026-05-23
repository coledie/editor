## User Prompt
Run 3 different clustering algorithms on images a, c, d. Use separate folders per image and per algorithm. Also adjust K — try 3 and 4 groups instead of 5. Run all texture permutations on each result.

## Problem Being Solved
Iteration 7 used a single fixed pipeline (bilateral + texture-variance K-means, K=5) only on image d. K=5 produced over-segmented, fragmented zones. Wanted to explore structurally different segmentation approaches and apply them across multiple source images.

## Approach
Three algorithms, each producing K zones, then P(5,K) texture permutations saved per image:

| Algorithm | Method | Zone boundary character |
|---|---|---|
| `kmeans` | K-Means on LAB + normalized x/y spatial features | Soft blobs, spatially coherent |
| `slic` | SLIC superpixels (300 segments) → merge to K via LAB k-means | Tight edge-following, mosaic-like |
| `felzenszwalb` | Felzenszwalb graph segmentation → merge to K via LAB k-means | Large contrast-driven regions |

SLIC and Felzenszwalb both produce many fine segments first; these are merged down to K zones by running a second k-means on mean LAB colors per superpixel.

Output layout: `iteration_8/k{k}/{image}/{algo}/segments.jpg` + `permutations/`.

## Key Parameters
- K = 3 (60 perms) and K = 4 (120 perms)
- KM_SPATIAL_W = 40
- SLIC: n_segments=300, compactness=12, sigma=1
- Felzenszwalb: scale=200, sigma=0.8, min_size=200
- Post-merge median smooth: size=15

## Result Assessment
K=3 produces clean, broad zones well suited to dramatic texture contrast.
SLIC segments follow image edges most precisely; Felzenszwalb pulls the sky cleanly as one zone.
K-means blobs are smoother and more spatially regular.
Reducing K from 5 to 3 was the right call — fewer zones make individual texture effects more readable.

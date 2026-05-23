## User Prompt
Generate some mandelbrot and julia sets — give me a wide range to choose a starting point from.

## Problem Being Solved
No starting imagery exists for example2. Need a broad gallery spanning different
zoom levels, regions, Julia constants, and colormaps so the user can pick a
visual direction for subsequent iterations.

## Approach
Single `generate.py` script that produces two galleries:

1. **Mandelbrot gallery** — 12 views covering the classic full set plus zooms
   into well-known features (seahorse valley, elephant valley, triple spiral,
   mini-mandelbrot, dendrite, etc.) at varied zoom depths.
2. **Julia gallery** — 12 Julia sets generated from a curated list of `c`
   constants known to produce distinct topologies (dendrite, douady rabbit,
   san marco, siegel disk, spirals, dust).

Each image is rendered at 800×800 with smooth (continuous) iteration coloring
and a varied set of matplotlib colormaps so palette choice is also a sampleable
axis. A contact-sheet `mandelbrot_grid.jpg` and `julia_grid.jpg` are written
alongside the individual full-resolution images so the user can scan and pick.

## Key Parameters
- `WIDTH=HEIGHT=800` — large enough to evaluate detail, small enough to render fast.
- `MAX_ITER=400` for Mandelbrot, `300` for Julia — enough for smooth boundaries at
  the chosen zoom levels without being slow.
- Vectorized numpy escape-time loop — no per-pixel Python.
- Smooth coloring: `n + 1 - log(log|z|)/log2` to avoid banding.

## Result Assessment
First-pass gallery. User picks favorites → next iteration deep-dives on a
chosen region / constant / palette.

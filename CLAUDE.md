# Editor Skill

Classical computer vision design experiments on a set of source images.
Each iteration lives in its own numbered folder. Each folder contains the generated images and a `prompt.md` explaining what the user wanted and how the implementation addresses it.

---

## Folder Structure

```
editor/
  a.jpg, b.jpg, c.jpg, d.jpg       ← original source images (never modified)
  editor.md                         ← this file (the skill)
  object_separation.md              ← object separation sub-skill (see below)
  iteration_1/
    object_separation.py            ← GENERATED artifact, do not edit directly
    prompt.md
    *_segments.jpg
    *_filtered.jpg
  iteration_2/
    object_separation.py            ← GENERATED artifact
    prompt.md
    *_segments.jpg
    *_filtered.jpg
  ...
```

### How Python files work

**The `.md` skill files are the source of truth. The `.py` files are generated artifacts.**

- `object_separation.md` contains the algorithm documentation, sample scripts, and parameter guide.
- When starting a new iteration, the AI reads `object_separation.md` and generates a fresh `object_separation.py` in the new `iteration_N/` folder — it does NOT copy or edit the previous iteration's `.py`.
- Never edit a `.py` file in an iteration folder directly. Instead: describe the change in conversation → AI updates the generated script for the new iteration.
- `.py` files can always be regenerated from the `.md` skill + the iteration's `prompt.md`. If a script is lost or broken, regenerate it.

### Rules
- **Never overwrite originals** (`a.jpg` – `d.jpg`).
- Every iteration gets its own folder: `iteration_N/`.
- Every iteration folder contains a `prompt.md` (see format below) and a generated `object_separation.py`.
- The script in each folder is self-contained and re-runnable: `cd iteration_N && python object_separation.py`.

---

## prompt.md Format

```markdown
## User Prompt
<exact or paraphrased request from the user>

## Problem Being Solved
<what was wrong or missing in the previous iteration>

## Approach
<what changed technically and why it addresses the problem>

## Key Parameters
<the values that matter most and why they were chosen>

## Result Assessment
<what worked, what didn't, what to try next>
```

---

## Skills

### texture.md
Generates per-zone procedural texture code. Contains:
- Opacity model (`apply_texture` blend function, 0.0–1.0 scale with preset table)
- Full catalogue of 5 texture designs with exact code + all parameters
- Zone texture dispatch pattern (`apply_zone_textures`)
- Reproducibility note (saving/loading `.npy` labels)

**When to use:** any time the user wants to overlay a pattern, grain, glitch, halftone, scanlines, or other procedural texture on an image region. Read `texture.md`, pick or design the textures, generate code for the new `iteration_N/` folder.

**Textures available:** scanlines, halftone dots, TV static noise, TV glitch/chromatic aberration, concentric ripple rings.

---

### object_separation.md
Generates per-iteration `object_separation.py` scripts. Contains:
- Full algorithm explanation (feature engineering, pre/post-processing)
- Sample code snippets for every major function
- Parameter reference table
- Iteration history summary

**When to use:** any time the user asks to change how objects are segmented, how filters are applied, or what features K-means uses. Read `object_separation.md`, make the change, generate a new script in the next `iteration_N/` folder.

---

## Object Separation — Research Notes

Goal: separate an image into contiguous regions corresponding to real objects, not lighting zones.

### Why naive K-means fails

K-means on RGB treats brightness as a primary feature. A shadow on a wall and a dark tree both cluster together because their RGB values are similar. Even in LAB space, if L (lightness) is not aggressively downweighted, bright/dark halves of the same object split apart.

Gaussian blur as pre-processing makes this worse for objects with uneven lighting: it blends across edges and homogenizes large light-gradient regions into one K-means attractor.

### Solution ranking (researched)

| Approach | Lighting robust? | Edge aware? | Ease | Notes |
|---|---|---|---|---|
| Bilateral filter + LAB K-means | ✓ moderate | ✓ yes | easy | Best drop-in upgrade from Gaussian |
| pyrMeanShiftFiltering → K-means | ✓ strong | ✓ yes | medium | Internally uses LUV + joint color-space density estimation |
| Hue-only (H from HSV) + spatial | ✓ strong | ✗ no | trivial | Fails on achromatic scenes (grays, whites) |
| SLIC superpixels → cluster | ✓ moderate | ✓✓ best | hard | Overkill for most cases |
| CLAHE on L before LAB K-means | ~ partial | ✗ no | easy | Helps, doesn't solve root cause |

### Recommended pipeline (iteration 4+)

1. `cv2.bilateralFilter` — edge-preserving smoothing; removes within-object lighting gradients without blurring object boundaries
2. `cv2.pyrMeanShiftFiltering` on the bilateral result — moves each pixel toward the mode of its perceptual color+space neighborhood (uses LUV internally); produces flat-color regions that track real objects
3. K-means on the mean-shifted image with pure color features (no spatial needed — mean-shift already enforces spatial coherence)
4. Median filter on label map — cleans any remaining speckles

This stack removes lighting as a clustering factor at two independent stages (bilateral removes gradient, mean-shift finds density modes) before K-means ever runs.

---

## Image Notes

| Image | Scene | Segmentation challenge |
|---|---|---|
| a.jpg | Highway overlook, winter | Sky vs treeline vs road vs rocks all have similar desaturated tones |
| c.jpg | Industrial yard from train | Blown-out sky dominates; mid-ground very desaturated |
| d.jpg | Snoqualmie Falls | Rich greens + misty whites make forest and waterfall hard to separate |

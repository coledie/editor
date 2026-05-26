## User Prompt
Okay now show some other wolfram agent rules on it

## Problem Being Solved
Iter_3 used a fixed rule per zone (rule 150 on yellow for K=5). We want to
explore how different elementary CA rules look when overlaid on the yellow
segment of `d.png`.

## Approach
Keep the iter_3 K=5 segmentation pipeline verbatim. Mask everything except the
yellow zone (label index 3). For each rule in a curated gallery, render one
overlay image at α=0.5. Gallery spans Wolfram's four classes plus famous rules:

- Class 2 (nested / Sierpinski-ish): 22, 60, 90, 150
- Class 3 (chaotic):                 30, 45, 75, 86, 105
- Class 4 (complex / localized):     54, 110, 137, 193
- Other notable:                     73, 126, 184

Each rule rendered with either `single` (one center cell live) or `random`
(50% density) initial condition to highlight its character.

## Key Parameters
- `TARGET_ZONE = 3` — yellow only.
- `CA_ALPHA = 0.5` — unchanged 50/50 blend.
- `CA_SEED = 7` — reproducible random ICs.

## Result Assessment
TBD. Look at the gallery and pick favorites; promote interesting ones into a
multi-zone composition in a later iteration.

"""Replace bottom-left corner of a textured image with the original (untextured) pixels.

Generates several outputs with increasing corner sizes, written to the workspace root.
"""
import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(ROOT, "example1", "a.jpg")
TEXTURED = os.path.join(
    ROOT,
    "example1",
    "iteration_11",
    "k4",
    "a",
    "permutations",
    "perm_040__scanlines_static_ripple_glitch.jpg",
)

orig = cv2.imread(ORIGINAL)
tex = cv2.imread(TEXTURED)
if orig.shape != tex.shape:
    orig = cv2.resize(orig, (tex.shape[1], tex.shape[0]))

H, W = tex.shape[:2]

# Fractions of the image (width, height) the clean corner spans.
# Listed from smallest to largest.
sizes = [
    (0.25, 0.20),
    (0.40, 0.32),
    (0.55, 0.45),
    (0.70, 0.58),
]

for i, (fw, fh) in enumerate(sizes, start=1):
    cw = int(W * fw)
    ch = int(H * fh)

    # Build a soft mask: 1.0 = original, 0.0 = textured.
    # Rectangle anchored at bottom-left, with a feathered edge along
    # the top and right boundary so the transition isn't a hard line.
    mask = np.zeros((H, W), dtype=np.float32)
    y0 = H - ch
    x1 = cw
    mask[y0:H, 0:x1] = 1.0

    # Feather using a Gaussian blur. Feather width scales with corner size.
    feather = max(15, int(min(cw, ch) * 0.18))
    if feather % 2 == 0:
        feather += 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask3 = mask[:, :, None]

    out = (orig.astype(np.float32) * mask3 + tex.astype(np.float32) * (1.0 - mask3))
    out = np.clip(out, 0, 255).astype(np.uint8)

    pct_w = int(fw * 100)
    pct_h = int(fh * 100)
    name = f"perm_040_cleancorner_{i}_w{pct_w}_h{pct_h}.jpg"
    cv2.imwrite(os.path.join(ROOT, name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {name}  (corner {cw}x{ch}, feather {feather})")

print("done")

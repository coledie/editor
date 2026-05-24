"""Reveal whole segmentation blobs (not rectangles) from the original image.

For each seed rectangle anchored in the bottom-left corner, we:
  1. Quantize segments.jpg into a 4-class label map.
  2. Per class, label connected components (8-connectivity).
  3. Any component with >= MIN_OVERLAP pixels inside the seed rectangle is
     marked as "to reveal".
  4. The union of those components forms a binary mask. The mask is feathered
     and used to alpha-composite the original over the textured image.

Outputs are written to the workspace root as
``perm_040_blobreveal_<i>_w<W>_h<H>.jpg``.
"""

import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITER = os.path.join(ROOT, "example1", "iteration_11")

ORIGINAL = os.path.join(ROOT, "example1", "a.jpg")
TEXTURED = os.path.join(ITER, "k4", "perm_040__scanlines_static_ripple_glitch.jpg")
SEGMENTS = os.path.join(ITER, "k4", "a", "segments.jpg")

# Seed rectangles anchored at bottom-left, as (width_frac, height_frac).
SEEDS = [
    (0.25, 0.20),
    (0.40, 0.32),
    (0.55, 0.45),
    (0.70, 0.58),
]

MIN_OVERLAP_FRAC = 0.002   # component must cover at least this fraction of image to count
FEATHER_FRAC = 0.012       # gaussian feather width as fraction of min(H, W)


def quantize_segments(seg_bgr: np.ndarray, k: int = 4) -> np.ndarray:
    """Snap segments image to its k dominant colors -> integer label map."""
    Z = seg_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(
        Z, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    return labels.reshape(seg_bgr.shape[:2]).astype(np.int32)


def reveal_mask(labels: np.ndarray, rect: tuple[int, int, int, int],
                min_overlap_px: int) -> np.ndarray:
    """Return binary mask of all connected components overlapping `rect`.

    rect = (x0, y0, x1, y1) inclusive-exclusive."""
    H, W = labels.shape
    x0, y0, x1, y1 = rect
    seed = np.zeros((H, W), dtype=np.uint8)
    seed[y0:y1, x0:x1] = 1

    out = np.zeros((H, W), dtype=np.uint8)
    for cls in np.unique(labels):
        binary = (labels == cls).astype(np.uint8)
        n, comp = cv2.connectedComponents(binary, connectivity=8)
        # For each component, count how many of its pixels lie inside seed rect.
        # Histogram component ids restricted to seed area.
        ids_in_seed = comp[seed.astype(bool)]
        counts = np.bincount(ids_in_seed, minlength=n)
        # Component 0 in `comp` is background (label != cls), ignore.
        for cid in range(1, n):
            if counts[cid] >= min_overlap_px:
                out[comp == cid] = 1
    return out


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.float32)
    r = radius if radius % 2 == 1 else radius + 1
    m = (mask * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(m, (r, r), 0).astype(np.float32) / 255.0
    return blurred


def main() -> None:
    orig = cv2.imread(ORIGINAL)
    tex = cv2.imread(TEXTURED)
    seg = cv2.imread(SEGMENTS)
    if orig is None or tex is None or seg is None:
        raise SystemExit("missing input image(s)")

    # Conform shapes to the textured image.
    H, W = tex.shape[:2]
    if orig.shape[:2] != (H, W):
        orig = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
    if seg.shape[:2] != (H, W):
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

    labels = quantize_segments(seg, k=4)
    min_overlap_px = max(50, int(MIN_OVERLAP_FRAC * H * W))
    feather_r = max(9, int(FEATHER_FRAC * min(H, W)))

    for i, (fw, fh) in enumerate(SEEDS, start=1):
        cw = int(W * fw)
        ch = int(H * fh)
        rect = (0, H - ch, cw, H)  # x0, y0, x1, y1
        mask = reveal_mask(labels, rect, min_overlap_px)

        # Quick stats so the user can see what was revealed.
        revealed_frac = mask.mean()
        soft = feather(mask, feather_r)[:, :, None]
        out = orig.astype(np.float32) * soft + tex.astype(np.float32) * (1.0 - soft)
        out = np.clip(out, 0, 255).astype(np.uint8)

        pct_w = int(fw * 100)
        pct_h = int(fh * 100)
        name = f"perm_040_blobreveal_{i}_w{pct_w}_h{pct_h}.jpg"
        cv2.imwrite(os.path.join(ROOT, name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {name}  (seed {cw}x{ch}, "
              f"revealed {revealed_frac*100:.1f}% of image, feather {feather_r})")

    print("done")


if __name__ == "__main__":
    main()

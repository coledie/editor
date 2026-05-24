"""Grayscale the RED and YELLOW k=4 zones, EXCEPT inside the bottom-left blob reveal.

Pipeline:
  1. Quantize segments.jpg into 4 kmeans classes.
  2. Identify the red and yellow class indices (closest centers to pure R / pure Y).
  3. Compute a blob-reveal mask: every connected component (any class) with
     >= MIN_OVERLAP_FRAC * H*W pixels inside the bottom-left seed rect.
  4. Grayscale mask = (red OR yellow class) AND NOT blob-reveal mask.
  5. In the grayscale mask: replace with RGB-averaged grayscale. Everywhere
     else: untouched original (so the bottom-left blob reveal stays in color).

Output to workspace root:
    perm_bw_redyellow.jpg
"""

import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITER = os.path.join(ROOT, "example1", "iteration_11")

ORIGINAL = os.path.join(ROOT, "example1", "a.jpg")
SEGMENTS = os.path.join(ITER, "k4", "a", "segments.jpg")

FEATHER_FRAC = 0.008

# Bottom-left blob-reveal seed rectangle (width_frac, height_frac).
SEED = (0.40, 0.32)
MIN_OVERLAP_FRAC = 0.002

# Target colors in BGR for the two zones we want to mark.
TARGETS_BGR = {
    "red":    np.array([  0,   0, 255], dtype=np.float32),
    "yellow": np.array([  0, 255, 255], dtype=np.float32),
}


def tex_tv_static(*_args, **_kwargs):  # noqa: D401
    """Removed: this iteration uses pure grayscale, no static."""
    raise NotImplementedError


def quantize_segments(seg_bgr: np.ndarray, k: int = 4):
    Z = seg_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(seg_bgr.shape[:2]).astype(np.int32)
    return labels, centers  # centers shape: (k, 3) BGR float


def closest_class(centers: np.ndarray, target_bgr: np.ndarray) -> int:
    d = np.linalg.norm(centers - target_bgr[None, :], axis=1)
    return int(np.argmin(d))


def reveal_mask(labels: np.ndarray, rect: tuple[int, int, int, int],
                min_overlap_px: int) -> np.ndarray:
    """Binary mask of every connected component (per class) that overlaps `rect`."""
    H, W = labels.shape
    x0, y0, x1, y1 = rect
    seed = np.zeros((H, W), dtype=bool)
    seed[y0:y1, x0:x1] = True
    out = np.zeros((H, W), dtype=np.uint8)
    for cls in np.unique(labels):
        binary = (labels == cls).astype(np.uint8)
        n, comp = cv2.connectedComponents(binary, connectivity=8)
        counts = np.bincount(comp[seed], minlength=n)
        for cid in range(1, n):
            if counts[cid] >= min_overlap_px:
                out[comp == cid] = 1
    return out


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    r = radius if radius % 2 == 1 else radius + 1
    return cv2.GaussianBlur((mask * 255).astype(np.uint8), (r, r), 0).astype(np.float32) / 255.0


def main() -> None:
    orig = cv2.imread(ORIGINAL)
    seg = cv2.imread(SEGMENTS)
    if orig is None or seg is None:
        raise SystemExit("missing input image(s)")

    H, W = orig.shape[:2]
    if seg.shape[:2] != (H, W):
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

    labels, centers = quantize_segments(seg, k=4)
    red_cls = closest_class(centers, TARGETS_BGR["red"])
    yel_cls = closest_class(centers, TARGETS_BGR["yellow"])
    print(f"red class={red_cls} center={centers[red_cls]}")
    print(f"yellow class={yel_cls} center={centers[yel_cls]}")

    # Blob reveal: components touching the bottom-left rect stay color.
    fw, fh = SEED
    cw, ch = int(W * fw), int(H * fh)
    rect = (0, H - ch, cw, H)
    min_overlap_px = max(50, int(MIN_OVERLAP_FRAC * H * W))
    reveal = reveal_mask(labels, rect, min_overlap_px)

    redyel = ((labels == red_cls) | (labels == yel_cls)).astype(np.uint8)
    target_mask = (redyel & (1 - reveal)).astype(np.uint8)

    # Grayscale base via RGB averaging — no static.
    gray = orig.astype(np.float32).mean(axis=2)
    gray_bgr = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)

    soft = feather(target_mask, max(7, int(FEATHER_FRAC * min(H, W))))[:, :, None]
    out = gray_bgr.astype(np.float32) * soft + orig.astype(np.float32) * (1.0 - soft)
    out = np.clip(out, 0, 255).astype(np.uint8)

    name = "perm_bw_redyellow.jpg"
    cv2.imwrite(os.path.join(ROOT, name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {name}  (grayscale covers {target_mask.mean()*100:.1f}% of image, "
          f"blob-reveal protects {reveal.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()

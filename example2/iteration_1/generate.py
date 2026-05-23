"""Generate a wide gallery of Mandelbrot and Julia sets as starting points.

Run from this folder:
    python generate.py

Outputs:
    mandelbrot/*.jpg       individual 800x800 views
    julia/*.jpg            individual 800x800 views
    mandelbrot_grid.jpg    contact sheet
    julia_grid.jpg         contact sheet
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

WIDTH = 800
HEIGHT = 800
MANDEL_MAX_ITER = 400
JULIA_MAX_ITER = 300

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MANDEL_DIR = os.path.join(OUT_DIR, "mandelbrot")
JULIA_DIR = os.path.join(OUT_DIR, "julia")
os.makedirs(MANDEL_DIR, exist_ok=True)
os.makedirs(JULIA_DIR, exist_ok=True)


# ---------- escape-time kernels (vectorized) ----------

def mandelbrot(center: complex, span: float, max_iter: int) -> np.ndarray:
    """Return smooth iteration counts for a square view centered at `center`."""
    half = span / 2.0
    xs = np.linspace(center.real - half, center.real + half, WIDTH)
    ys = np.linspace(center.imag - half, center.imag + half, HEIGHT)
    C = xs[None, :] + 1j * ys[:, None]
    Z = np.zeros_like(C)
    out = np.full(C.shape, max_iter, dtype=np.float64)
    alive = np.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        Z[alive] = Z[alive] * Z[alive] + C[alive]
        escaped = alive & (np.abs(Z) > 2.0)
        # smooth coloring: i + 1 - log(log|z|)/log 2
        zsafe = np.abs(Z[escaped])
        out[escaped] = i + 1 - np.log(np.log(zsafe)) / np.log(2)
        alive &= ~escaped
        if not alive.any():
            break
    return out


def julia(c: complex, center: complex, span: float, max_iter: int) -> np.ndarray:
    half = span / 2.0
    xs = np.linspace(center.real - half, center.real + half, WIDTH)
    ys = np.linspace(center.imag - half, center.imag + half, HEIGHT)
    Z = xs[None, :] + 1j * ys[:, None]
    out = np.full(Z.shape, max_iter, dtype=np.float64)
    alive = np.ones(Z.shape, dtype=bool)
    for i in range(max_iter):
        Z[alive] = Z[alive] * Z[alive] + c
        escaped = alive & (np.abs(Z) > 2.0)
        zsafe = np.abs(Z[escaped])
        out[escaped] = i + 1 - np.log(np.log(zsafe)) / np.log(2)
        alive &= ~escaped
        if not alive.any():
            break
    return out


# ---------- coloring ----------

def colorize(field: np.ndarray, max_iter: int, cmap_name: str) -> np.ndarray:
    """Map smooth iteration field to an RGB image (uint8)."""
    # normalize escaped pixels; interior (== max_iter) -> black
    interior = field >= max_iter
    f = field.copy()
    f[interior] = 0
    # log compress for nicer dynamic range
    f = np.log1p(f)
    if f.max() > 0:
        f = f / f.max()
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(f)[..., :3]
    rgb[interior] = 0.0
    return (rgb * 255).astype(np.uint8)


def save(rgb: np.ndarray, path: str) -> None:
    plt.imsave(path, rgb)


# ---------- gallery definitions ----------

@dataclass
class MView:
    name: str
    center: complex
    span: float
    cmap: str


MANDELBROT_VIEWS: list[MView] = [
    MView("01_classic",          -0.5 + 0.0j,                 3.5,    "twilight_shifted"),
    MView("02_seahorse_valley",  -0.743643887 + 0.131825904j, 0.01,   "magma"),
    MView("03_elephant_valley",   0.275 + 0.0j,               0.15,   "inferno"),
    MView("04_triple_spiral",    -0.088 + 0.654j,             0.05,   "viridis"),
    MView("05_mini_mandelbrot",  -1.7497 + 0.0j,              0.04,   "cubehelix"),
    MView("06_dendrite",         -0.235125 + 0.827215j,       0.004,  "plasma"),
    MView("07_north_antenna",    -1.25 + 0.0j,                0.3,    "twilight"),
    MView("08_west_bulb",        -1.0 + 0.27j,                0.25,   "ocean"),
    MView("09_deep_zoom_a",      -0.7436438870 + 0.1318259042j, 0.0005, "hot"),
    MView("10_deep_zoom_b",      -1.7693831791 + 0.0042368479j, 0.0008, "bone"),
    MView("11_lightning",        -1.940157 + 0.0j,            0.012,  "gist_heat"),
    MView("12_seahorse_zoom",   -0.745428 + 0.113009j,        0.0028, "nipy_spectral"),
]


@dataclass
class JView:
    name: str
    c: complex
    cmap: str
    span: float = 3.0
    center: complex = 0.0 + 0.0j


JULIA_VIEWS: list[JView] = [
    JView("01_douady_rabbit",   -0.123 + 0.745j,    "magma"),
    JView("02_dendrite",         0.0   + 1.0j,      "twilight_shifted"),
    JView("03_san_marco",       -0.75  + 0.0j,      "inferno"),
    JView("04_siegel_disk",     -0.391 - 0.587j,    "plasma"),
    JView("05_spiral",          -0.7269 + 0.1889j,  "viridis"),
    JView("06_basilica",        -1.0   + 0.0j,      "cubehelix"),
    JView("07_airplane",        -1.755 + 0.0j,      "hot"),
    JView("08_fatou_dust",       0.285 + 0.01j,     "bone"),
    JView("09_lightning",        0.285 + 0.013j,    "gist_heat"),
    JView("10_swirls",          -0.8   + 0.156j,    "twilight"),
    JView("11_feather",         -0.4   + 0.6j,      "ocean"),
    JView("12_galaxy",          -0.70176 - 0.3842j, "nipy_spectral"),
]


# ---------- contact sheet ----------

def contact_sheet(images: list[np.ndarray], titles: list[str], cols: int = 4) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.array(axes).reshape(rows, cols)
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------- run ----------

def main() -> None:
    print("Rendering Mandelbrot gallery...")
    m_imgs, m_titles = [], []
    for v in MANDELBROT_VIEWS:
        field = mandelbrot(v.center, v.span, MANDEL_MAX_ITER)
        rgb = colorize(field, MANDEL_MAX_ITER, v.cmap)
        save(rgb, os.path.join(MANDEL_DIR, f"{v.name}.jpg"))
        m_imgs.append(rgb)
        m_titles.append(f"{v.name}\n{v.cmap}  span={v.span:g}")
        print(f"  {v.name}")
    fig = contact_sheet(m_imgs, m_titles)
    fig.savefig(os.path.join(OUT_DIR, "mandelbrot_grid.jpg"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("Rendering Julia gallery...")
    j_imgs, j_titles = [], []
    for v in JULIA_VIEWS:
        field = julia(v.c, v.center, v.span, JULIA_MAX_ITER)
        rgb = colorize(field, JULIA_MAX_ITER, v.cmap)
        save(rgb, os.path.join(JULIA_DIR, f"{v.name}.jpg"))
        j_imgs.append(rgb)
        j_titles.append(f"{v.name}\nc={v.c.real:+.3f}{v.c.imag:+.3f}j")
        print(f"  {v.name}")
    fig = contact_sheet(j_imgs, j_titles)
    fig.savefig(os.path.join(OUT_DIR, "julia_grid.jpg"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()

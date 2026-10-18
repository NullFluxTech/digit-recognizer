"""Turn any image of a single digit into MNIST format: (784,) float32, white ink on black, in [0, 1].

MNIST's conventions, measured on the 55,000 training images (milestone 5 in LEARN.md):
    background 0, ink up to exactly 1.0 (every image's brightest pixel is 1.0)
    strokes are solid: about half of all ink pixels (49%) are >= 0.9
    stroke width is ~14% of the digit's longer side (2.82 px in the 20 px box)
    the ink's bounding box has its longer side exactly 20 px (aspect ratio kept)
    the ink's center of mass is at (row, col) = (14.0, 14.0) in a 28x28 frame (std 0.29 px)

Pillow is used only to load and resize; every other step is NumPy.
"""

from pathlib import Path

import numpy as np
from PIL import Image

BOX = 20                 # longer side of the digit after scaling
SIZE = 28                # output frame
CENTER = 14.0            # target center of mass (row and column), pixel-index coordinates
INK_THRESHOLD = 0.2      # after cleanup: pixels above this count as ink when cropping
NOISE_FLOOR = 0.1        # after cleanup: pixels below this are set to 0 (paper texture, shadows)
STROKE_FRACTION = 0.141  # MNIST median stroke width / digit size (2.82 px / 20 px)
STROKE_MIN = 0.108       # MNIST 10th percentile: thinner strokes than this get thickened


def load_gray(path: Path) -> np.ndarray:
    """Load any image file as (H,W) float32 in [0,1], 0 = black. Transparency is flattened onto white."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    gray = np.asarray(img.convert("L"), dtype=np.float32)  # (H,W) uint8 values 0..255 as float
    return gray / np.float32(255.0)                          # (H,W) in [0,1]


def _resize(g: np.ndarray, height: int, width: int) -> np.ndarray:
    """Antialiased resize of a [0,1] float image with Pillow (Lanczos filter)."""
    img = Image.fromarray(np.round(g * 255).astype(np.uint8))  # 2-D uint8 -> 8-bit grayscale ("L")
    out = img.resize((width, height), resample=Image.Resampling.LANCZOS)  # Pillow takes (W,H)
    return np.asarray(out, dtype=np.float32) / np.float32(255.0)          # (height,width) in [0,1]


def _shift(g: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate by whole pixels, filling with 0 (unlike np.roll, nothing wraps around)."""
    out = np.zeros_like(g)                                   # (28,28)
    h, w = g.shape
    src_r = slice(max(0, -dy), min(h, h - dy))
    dst_r = slice(max(0, dy), min(h, h + dy))
    src_c = slice(max(0, -dx), min(w, w - dx))
    dst_c = slice(max(0, dx), min(w, w + dx))
    out[dst_r, dst_c] = g[src_r, src_c]
    return out


def stroke_width(g: np.ndarray) -> float:
    """Rough stroke width in pixels: 2 * area / perimeter of the ink (pixels > 0.5).

    For a long stroke of length L and width w: area ~ L*w and perimeter ~ 2L, so w ~ 2*area/perimeter.
    """
    ink = g > 0.5                                            # (H,W) bool
    p = np.pad(ink, 1)                                       # (H+2,W+2): every pixel has 4 neighbours
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]  # (H,W) all 4 are ink
    perimeter = int((ink & ~inner).sum())                    # ink pixels touching the background
    return 2.0 * int(ink.sum()) / max(perimeter, 1)


def _dilate(g: np.ndarray, passes: int) -> np.ndarray:
    """Grayscale dilation (max filter): strokes grow by ~1 px per side per pass.

    Passes alternate a 3x3 square and a 3x3 cross; together they approximate a round pen
    (a square alone would thicken diagonal strokes ~1.4x more than straight ones).
    """
    h, w = g.shape
    for i in range(passes):
        p = np.pad(g, 1)                                     # (H+2,W+2)
        out = g.copy()
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]         # the cross
        if i % 2 == 0:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # + the corners = the full square
        for dy, dx in offsets:
            out = np.maximum(out, p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])  # (H,W) max with shifted copy
        g = out
    return g


def _connected_to(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Pixels of `allowed` reachable from `seed` through `allowed` (8-connected flood fill).

    Repeatedly grow the seed by one pixel (3x3 max filter) and keep only allowed pixels, until
    nothing changes. Both arguments and the result are (H,W) bool masks.
    """
    keep = seed & allowed
    while True:
        grown = _dilate(keep.astype(np.float32), 1) > 0      # one 3x3 square pass: one 8-connected step
        grown &= allowed
        if np.array_equal(grown, keep):
            return keep
        keep = grown


def center_of_mass(g: np.ndarray) -> tuple[float, float]:
    """Ink-weighted mean (row, col) of a 2-D image."""
    total = g.sum()                                          # scalar: total ink
    rows = np.arange(g.shape[0])                             # (H,) 0..H-1
    cols = np.arange(g.shape[1])                             # (W,) 0..W-1
    cy = (g.sum(axis=1) @ rows) / total                      # (H,) @ (H,) -> scalar  sum_r r * ink in row r
    cx = (g.sum(axis=0) @ cols) / total                      # (W,) @ (W,) -> scalar  sum_c c * ink in col c
    return float(cy), float(cx)


def to_mnist_steps(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Run the pipeline and return every intermediate image (for explaining / debugging).

    gray: (H,W) float in [0,1], any polarity. Keys, in order:
        gray, inverted, cleaned, cropped, thickened, scaled (h,w with max = 20), final (28,28)
    Raises ValueError if no ink is found.
    """
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {g.shape}")
    steps = {"gray": g}

    # 1. Polarity: MNIST is white ink on black. The border is almost always background,
    #    so if the border is bright (paper), invert.
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])  # (2W+2H,)
    if np.median(border) > 0.5:
        g = 1.0 - g                                          # (H,W)
    steps["inverted"] = g

    # 2. Cleanup: background -> 0, typical stroke -> 1, faint noise -> 0.
    #    Scaling by the MEDIAN stroke pixel (not the single darkest one) makes strokes solid like
    #    MNIST's, where about half the ink pixels are saturated. With pencil or uneven light, scaling
    #    by the darkest pixel leaves most of the stroke a faint gray.
    background = float(np.median(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])))
    g = np.clip(g - background, 0.0, None)                   # (H,W) >= 0
    if g.max() < 0.15:
        raise ValueError("no ink found: the image is (nearly) uniform")
    stroke_level = float(np.median(g[g > 0.3 * g.max()]))    # typical brightness of stroke pixels
    g = np.clip(g / stroke_level, 0.0, 1.0)                  # (H,W) in [0,1]
    g[g < NOISE_FLOOR] = 0.0
    #    Stains and shadows: in MNIST, gray pixels only occur at the edges of solid strokes, never
    #    as separate blobs. Keep a faint pixel only if it connects to solid ink (>= 0.5).
    g = g * _connected_to(g >= 0.5, g > 0)                   # (H,W) * (H,W) bool mask
    steps["cleaned"] = g

    # 3. Crop to the bounding box of the ink.
    ink_rows = np.flatnonzero(g.max(axis=1) > INK_THRESHOLD)  # (H,) -> row indices with ink
    ink_cols = np.flatnonzero(g.max(axis=0) > INK_THRESHOLD)  # (W,) -> column indices with ink
    g = g[ink_rows[0]:ink_rows[-1] + 1, ink_cols[0]:ink_cols[-1] + 1]  # (h,w)
    steps["cropped"] = g

    # 3b. Thicken unusually thin strokes. MNIST's widths range over ~10-17% of the digit's size; a
    #     stroke thinner than MNIST's 10th percentile is grown to the median (14%). Strokes inside
    #     MNIST's normal range are left alone, and nothing is ever thinned.
    size = max(g.shape)
    width = stroke_width(g)
    passes = 0
    if width < STROKE_MIN * size:
        passes = round((STROKE_FRACTION * size - width) / 2)  # each pass adds ~1 px on each side
    if passes:
        g = _dilate(np.pad(g, passes), passes)               # pad first so strokes can grow outward
    steps["thickened"] = g

    # 4. Scale so the longer side is 20 px, keeping the aspect ratio.
    h, w = g.shape
    scale = BOX / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    g = _resize(g, new_h, new_w)                              # (new_h,new_w), max(new_h,new_w) = 20
    g = g / g.max()                                           # brightest pixel back to exactly 1, as in MNIST
    steps["scaled"] = g

    # 5. Place in a 28x28 frame and shift so the center of mass lands on (14, 14).
    frame = np.zeros((SIZE, SIZE), dtype=np.float32)          # (28,28)
    top, left = (SIZE - new_h) // 2, (SIZE - new_w) // 2
    frame[top:top + new_h, left:left + new_w] = g             # box roughly centered first
    cy, cx = center_of_mass(frame)
    dy, dx = round(CENTER - cy), round(CENTER - cx)
    # never push ink off the frame: clamp the shift to the empty margin on each side
    dy = int(np.clip(dy, -top, SIZE - (top + new_h)))
    dx = int(np.clip(dx, -left, SIZE - (left + new_w)))
    steps["final"] = _shift(frame, dy, dx)                    # (28,28)
    return steps


def to_mnist(gray: np.ndarray) -> np.ndarray:
    """(H,W) grayscale image in [0,1] (any polarity) -> (784,) float32 MNIST-style input row."""
    return to_mnist_steps(gray)["final"].reshape(-1)          # (28,28) -> (784,)

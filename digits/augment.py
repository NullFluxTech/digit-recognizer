"""Data augmentation for 28x28 digits: random affine warps and stroke-thickness changes.

Every transform keeps the label valid (a slanted 7 is still a 7) and stays within the variation
of real handwriting: slant, rotation, narrow/wide writing, small shifts, thinner/thicker pens.

Shapes:  B = batch size,  X (B,784) or (B,28,28),  P = 784 pixel positions.
"""

import numpy as np

SIZE = 28
CENTER = 14.0  # MNIST's center of mass; warps pivot around it so digits stay centered


def affine_warp(imgs: np.ndarray, M: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Warp each image by its own 2x2 matrix M[b] (about the center) plus a shift[b], in pixels.

    A point p of the INPUT moves to  q = M (p - c) + c + shift  in the output. To fill the output,
    go backwards: for every output pixel q, read the input at  p = M^-1 (q - c - shift) + c,
    interpolating bilinearly between the 4 surrounding input pixels. Outside the image reads 0.

    imgs (B,28,28), M (B,2,2), shift (B,2) -> (B,28,28)
    """
    B = imgs.shape[0]
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]                          # (28,28) each: output pixel coordinates
    q = np.stack([rr.ravel(), cc.ravel()]).astype(np.float32)  # (2,P) every output pixel as a column
    Minv = np.linalg.inv(M).astype(np.float32)                 # (B,2,2) batched 2x2 inverses
    p = Minv @ (q[None] - CENTER - shift[:, :, None].astype(np.float32)) + CENTER  # (B,2,2)@(B,2,P) -> (B,2,P)
    # Clamp far-away coordinates into the zero border, so every read below stays inside the array.
    r = np.clip(p[:, 0], -1.0, SIZE)                           # (B,P) input row coordinates
    c = np.clip(p[:, 1], -1.0, SIZE)                           # (B,P) input column coordinates
    r0, c0 = np.floor(r), np.floor(c)                          # (B,P) top-left neighbour
    fr, fc = r - r0, c - c0                                    # (B,P) fractional parts in [0,1)

    W = SIZE + 3                                               # padded width 31: 1 zero col left, 2 right
    padded = np.pad(imgs, ((0, 0), (1, 2), (1, 2))).ravel()    # (B*31*31,) flat; outside reads give 0
    # Row-major flat index of the top-left neighbour (same formula as pixels in milestone 1):
    base = (np.arange(B)[:, None] * W * W                      # (B,1) start of each image
            + (r0.astype(np.int64) + 1) * W + (c0.astype(np.int64) + 1))  # (B,P)
    tl, tr = padded[base], padded[base + 1]                    # (B,P) top-left, top-right
    bl, br = padded[base + W], padded[base + W + 1]            # (B,P) bottom-left, bottom-right
    top = tl + fc * (tr - tl)                                  # (B,P) blend along the row
    bottom = bl + fc * (br - bl)                               # (B,P)
    out = top + fr * (bottom - top)                            # (B,P) then between the rows
    return out.reshape(B, SIZE, SIZE).astype(imgs.dtype)


def random_affine_params(B: int, rng: np.random.Generator, max_rotate_deg=12.0, max_shear=0.3,
                         scale_x=(0.7, 1.1), scale_y=(0.9, 1.1), max_shift=2.0):
    """Random M = rotation @ shear @ scale for B images, and random shifts. Returns (B,2,2), (B,2).

    In (row, col) coordinates:
        scale  [[sy, 0], [0, sx]]       narrow / wide writing (sx < 1 = narrower)
        shear  [[1, 0], [-k, 1]]        slant: rows further down move left by k per row (italic)
        rotate [[cos, -sin], [sin, cos]]
    """
    theta = np.deg2rad(rng.uniform(-max_rotate_deg, max_rotate_deg, B))  # (B,)
    k = rng.uniform(-max_shear, max_shear, B)                             # (B,)
    sx = rng.uniform(*scale_x, B)                                         # (B,)
    sy = rng.uniform(*scale_y, B)                                         # (B,)
    cos, sin, zero, one = np.cos(theta), np.sin(theta), np.zeros(B), np.ones(B)
    R = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)  # (B,2,2)
    S = np.stack([np.stack([one, zero], -1), np.stack([-k, one], -1)], -2)   # (B,2,2)
    D = np.stack([np.stack([sy, zero], -1), np.stack([zero, sx], -1)], -2)   # (B,2,2)
    M = R @ S @ D                                                            # (B,2,2) applied right to left
    shift = rng.uniform(-max_shift, max_shift, (B, 2))                       # (B,2) rows, cols
    return M, shift


def _max3(imgs: np.ndarray, op) -> np.ndarray:
    """3x3 max (op=np.maximum, dilation) or min (op=np.minimum, erosion) filter on (B,28,28)."""
    fill = 0.0 if op is np.maximum else 1.0
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), constant_values=fill)       # (B,30,30)
    out = imgs.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = op(out, p[:, dy:dy + SIZE, dx:dx + SIZE])                   # (B,28,28)
    return out


def random_thickness(imgs: np.ndarray, rng: np.random.Generator, p_thick=0.25, p_thin=0.15) -> np.ndarray:
    """Make some strokes ~1 px thicker or thinner (a half-strength 3x3 dilation / erosion)."""
    u = rng.random(imgs.shape[0])                                            # (B,)
    out = imgs.copy()
    thick, thin = u < p_thick, (u >= p_thick) & (u < p_thick + p_thin)        # (B,) bools
    out[thick] = 0.5 * imgs[thick] + 0.5 * _max3(imgs[thick], np.maximum)
    out[thin] = 0.5 * imgs[thin] + 0.5 * _max3(imgs[thin], np.minimum)
    return out


def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(B,784) -> (B,784): random affine warp, then random stroke thickness."""
    imgs = X.reshape(-1, SIZE, SIZE)                                         # (B,28,28) view
    M, shift = random_affine_params(imgs.shape[0], rng)
    out = random_thickness(affine_warp(imgs, M, shift), rng)
    return np.clip(out, 0.0, 1.0).reshape(X.shape[0], -1)                    # (B,784)

"""Tests for digits.preprocess: any image -> MNIST conventions."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from digits.data import DATA_DIR, FILES
from digits.preprocess import center_of_mass, load_gray, stroke_width, to_mnist, to_mnist_steps


def paper_digit_seven(h=300, w=220, top=40, left=50, thickness=18):
    """A '7' drawn as dark ink (0) on white paper (1), far from centered."""
    g = np.ones((h, w), dtype=np.float32)
    g[top:top + thickness, left:left + 120] = 0.0                       # top bar
    for i in range(160):                                                # diagonal stem
        r, c = top + thickness + i, left + 120 - i // 2
        g[r:r + thickness, c - thickness:c] = 0.0
    return g


def ink_bbox(img28):
    rows = np.flatnonzero(img28.max(axis=1) > 0)
    cols = np.flatnonzero(img28.max(axis=0) > 0)
    return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1


def test_output_follows_mnist_conventions():
    x = to_mnist(paper_digit_seven())
    assert x.shape == (784,) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() == pytest.approx(1.0)
    img = x.reshape(28, 28)
    assert img[0, :].max() == 0 and img[:, 0].max() == 0           # background is 0, ink is bright
    h, w = ink_bbox(img)
    assert max(h, w) <= 21 and max(h, w) >= 19                      # longer side ~20 (Lanczos edges)
    cy, cx = center_of_mass(img)
    assert abs(cy - 14.0) <= 0.5 and abs(cx - 14.0) <= 0.5


def test_polarity_does_not_matter():
    paper = paper_digit_seven()
    assert np.allclose(to_mnist(paper), to_mnist(1.0 - paper))


def test_position_and_scale_do_not_matter():
    small = paper_digit_seven()
    big = np.ones((900, 900), dtype=np.float32)
    big[500:800, 100:320] = np.asarray(Image.fromarray(np.round(small * 255).astype(np.uint8)), np.float32) / 255
    a, b = to_mnist(small), to_mnist(big)
    assert np.abs(a - b).max() < 0.05


def test_aspect_ratio_is_kept():
    g = np.zeros((200, 200), dtype=np.float32)
    g[20:180, 95:105] = 1.0                                          # a thin vertical bar, 160 x 10
    h, w = ink_bbox(to_mnist(g).reshape(28, 28))
    assert h >= 19 and w <= 3                                        # stays thin, not stretched to 20x20


def test_blank_image_is_rejected():
    with pytest.raises(ValueError, match="no ink"):
        to_mnist(np.ones((50, 50), dtype=np.float32))


def test_faint_noise_is_removed():
    rng = np.random.default_rng(0)
    g = paper_digit_seven() - rng.uniform(0, 0.08, size=(300, 220)).astype(np.float32)  # gray paper texture
    clean = to_mnist(paper_digit_seven())
    assert np.abs(to_mnist(np.clip(g, 0, 1)) - clean).max() < 0.1


def test_stroke_width_estimate():
    g = np.zeros((100, 100), np.float32)
    g[10:90, 40:46] = 1.0                                            # an 80 x 6 bar
    assert stroke_width(g) == pytest.approx(6, rel=0.15)


def test_faint_thin_pencil_becomes_solid_and_mnist_thick():
    # Pencil on paper: gray ink (0.55) only 4 px wide in a 230 px tall digit (~2% vs MNIST ~14%).
    pencil = 1.0 - 0.45 * (1.0 - paper_digit_seven(thickness=4))
    img = to_mnist(pencil).reshape(28, 28)
    ink = img[img > 0.1]
    assert (ink >= 0.9).mean() > 0.3                                 # solid strokes (MNIST: 49%)
    assert stroke_width(img) / 20 > 0.108                            # at least MNIST's 10th percentile
    assert img.sum() > 50                                            # vs ~100 for a typical MNIST digit


def test_normal_width_strokes_are_not_thickened():
    steps = to_mnist_steps(paper_digit_seven(thickness=30))         # 30/230 = 13%: inside MNIST's range
    assert steps["thickened"].shape == steps["cropped"].shape


def test_stains_away_from_the_strokes_are_removed():
    clean = paper_digit_seven()
    stained = clean.copy()
    stained[250:290, 10:60] = 0.75                                    # a soft gray smudge, not touching the 7
    assert np.abs(to_mnist(stained) - to_mnist(clean)).max() < 1e-6  # smudge gone, digit unchanged


def test_steps_are_all_returned():
    steps = to_mnist_steps(paper_digit_seven())
    assert list(steps) == ["gray", "inverted", "cleaned", "cropped", "thickened", "scaled", "final"]
    assert max(steps["scaled"].shape) == 20
    assert steps["final"].shape == (28, 28)


def test_load_gray_flattens_transparency(tmp_path):
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)                     # fully transparent ...
    rgba[3:7, 3:7] = [0, 0, 0, 255]                                  # ... with an opaque black square
    path = tmp_path / "t.png"
    Image.fromarray(rgba).save(path)
    g = load_gray(path)
    assert g.shape == (10, 10)
    assert g[0, 0] == pytest.approx(1.0) and g[5, 5] == pytest.approx(0.0)


have_mnist = all((DATA_DIR / name).exists() for name, _ in FILES.values())


def best_shifted_correlation(a, b):
    """Max correlation of a with b shifted by up to 1 px in each direction.

    Thin strokes that move by a single pixel after re-scaling barely change the digit, but they
    wreck a plain pixel-by-pixel correlation (measured: 0.72 for a visually identical 3).
    """
    best = -1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)       # border is empty: no wrap issue
            best = max(best, np.corrcoef(a.ravel(), shifted.ravel())[0, 1])
    return best


def uglified_mnist(n=50, seed=0):
    """MNIST test digits printed 6x larger, dark ink on white paper, at a random spot."""
    from digits.data import load_mnist

    d = load_mnist()
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(d.X_test), size=n, replace=False):
        original = d.X_test[i].reshape(28, 28)
        big = np.asarray(Image.fromarray(np.round(original * 255).astype(np.uint8)).resize(
            (168, 168), Image.Resampling.BICUBIC), np.float32) / 255          # (168,168)
        paper = np.ones((400, 400), np.float32)
        r, c = rng.integers(0, 400 - 168, size=2)
        paper[r:r + 168, c:c + 168] = 1.0 - big                             # dark ink on white
        yield original, paper


@pytest.mark.skipif(not have_mnist, reason="MNIST not downloaded; run `uv run prepare_data.py`")
def test_mnist_digits_survive_uglify_and_preprocess():
    """Undoing size, position and polarity gives back (nearly) the original MNIST digit.

    Exception by design: MNIST's own thinnest digits (stroke below its 10th percentile) get
    thickened, so they are not expected to match pixel for pixel. Their predictions are checked
    in the next test.
    """
    thickened = 0
    for original, paper in uglified_mnist():
        steps = to_mnist_steps(paper)
        back = steps["final"]
        cy, cx = center_of_mass(back)
        assert abs(cy - 14) <= 0.5 and abs(cx - 14) <= 0.5
        if steps["thickened"].shape != steps["cropped"].shape:
            thickened += 1
            continue
        assert best_shifted_correlation(original, back) > 0.85
    assert thickened <= 10                     # roughly MNIST's thinnest 10% (measured: 4 of 50)


MODEL = Path(__file__).resolve().parent.parent / "models" / "mlp.npz"


@pytest.mark.skipif(not (have_mnist and MODEL.exists()), reason="needs MNIST and models/mlp.npz")
def test_mnist_digits_keep_their_prediction_after_preprocess():
    from digits.models import MLP

    model = MLP.load(MODEL)
    pairs = list(uglified_mnist())
    before = model.predict(np.stack([o.reshape(-1) for o, _ in pairs]))     # (50,)
    after = model.predict(np.stack([to_mnist(p) for _, p in pairs]))        # (50,)
    assert (before == after).sum() >= 48

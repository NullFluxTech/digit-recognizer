"""Tests for digits.augment and the training loop's augment hook."""

import numpy as np
import pytest

from digits.augment import affine_warp, augment_batch, random_affine_params, random_thickness
from digits.models import MLP
from digits.nn import SGDMomentum
from digits.train import train

EYE = np.eye(2)[None]                                  # (1,2,2) identity, one image


def blob_image():
    img = np.zeros((1, 28, 28), np.float32)
    img[0, 6:22, 12:16] = 1.0                           # a vertical bar, 16 tall x 4 wide, centered
    return img


def ink_cols(img):
    return np.flatnonzero(img.max(axis=0) > 0.5)


def test_identity_and_integer_shift_are_exact():
    imgs = np.random.default_rng(0).random((5, 28, 28)).astype(np.float32)
    eye = np.repeat(EYE, 5, 0)
    assert np.allclose(affine_warp(imgs, eye, np.zeros((5, 2))), imgs, atol=1e-6)
    shift = np.tile([2.0, -3.0], (5, 1))                # 2 rows down, 3 columns left
    out = affine_warp(imgs, eye, shift)
    assert np.allclose(out[:, 2:, :-3], imgs[:, :-2, 3:], atol=1e-6)
    assert np.all(out[:, :2, :] == 0) and np.all(out[:, :, -3:] == 0)   # newly exposed area is empty


def test_narrow_scale_makes_digit_narrower():
    D = np.array([[[1.0, 0.0], [0.0, 0.5]]])            # columns squeezed to half width
    out = affine_warp(blob_image(), D, np.zeros((1, 2)))[0]
    assert len(ink_cols(out)) == 2                      # 4 px wide -> 2 px wide
    assert out.sum() == pytest.approx(blob_image().sum() / 2, rel=0.05)


def test_shear_slants_lower_rows_left():
    S = np.array([[[1.0, 0.0], [-0.5, 1.0]]])           # k = 0.5
    out = affine_warp(blob_image(), S, np.zeros((1, 2)))[0]
    top, bottom = ink_cols(out[6:9]).mean(), ink_cols(out[19:22]).mean()
    assert bottom < top - 4                             # 13 rows apart * 0.5 = ~6.5 px to the left


def test_rotation_keeps_the_ink():
    t = np.deg2rad(10)
    R = np.array([[[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]])
    out = affine_warp(blob_image(), R, np.zeros((1, 2)))
    assert out.sum() == pytest.approx(blob_image().sum(), rel=0.05)


def test_random_params_shapes_and_seed():
    M1, s1 = random_affine_params(7, np.random.default_rng(1))
    M2, s2 = random_affine_params(7, np.random.default_rng(1))
    assert M1.shape == (7, 2, 2) and s1.shape == (7, 2)
    assert np.array_equal(M1, M2) and np.array_equal(s1, s2)
    assert np.all(np.abs(np.linalg.det(M1)) > 0.5)     # invertible, no collapse


def test_thickness_changes_ink_in_the_right_direction():
    img = np.repeat(blob_image(), 3, 0)
    rng = np.random.default_rng(0)
    assert random_thickness(img, rng, p_thick=1.0, p_thin=0.0).sum() > img.sum()
    assert random_thickness(img, rng, p_thick=0.0, p_thin=1.0).sum() < img.sum()
    assert np.array_equal(random_thickness(img, rng, p_thick=0.0, p_thin=0.0), img)


def test_augment_batch_shape_range_and_seed():
    X = np.random.default_rng(2).random((16, 784), dtype=np.float32)
    a = augment_batch(X, np.random.default_rng(3))
    b = augment_batch(X, np.random.default_rng(3))
    assert a.shape == (16, 784) and a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, X)


def test_train_applies_augment_to_training_batches_only():
    rng = np.random.default_rng(4)
    X, y = rng.random((60, 784), dtype=np.float32), rng.integers(0, 10, 60)
    seen = []

    def spy(Xb, r):
        seen.append(Xb.shape)
        return Xb

    train(MLP(n_hidden=8), SGDMomentum(0.01), X[:40], y[:40], X[40:], y[40:], epochs=2,
          batch_size=20, seed=0, augment=spy, verbose=False)
    assert seen == [(20, 784)] * 4                      # 2 batches per epoch x 2 epochs, never evaluation

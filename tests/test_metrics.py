"""Tests for digits.metrics."""

import numpy as np
import pytest

from digits.metrics import confusion_matrix, most_confused_pairs, per_class_metrics


def confusion_by_loop(y_true, y_pred, k):
    C = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        C[t][p] += 1
    return np.array(C)


def test_confusion_matrix_matches_a_plain_loop():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, size=1000)
    y_pred = np.where(rng.random(1000) < 0.8, y_true, rng.integers(0, 10, size=1000))
    C = confusion_matrix(y_true, y_pred)
    assert C.shape == (10, 10) and C.dtype == np.int64
    assert np.array_equal(C, confusion_by_loop(y_true, y_pred, 10))
    assert C.sum() == 1000
    assert np.array_equal(C.sum(axis=1), np.bincount(y_true, minlength=10))  # rows = true counts
    assert np.array_equal(C.sum(axis=0), np.bincount(y_pred, minlength=10))  # cols = predicted counts


def test_repeated_pairs_all_count():
    # The pitfall np.add.at avoids: with C[t, p] += 1, the three (1, 2) pairs would add up to 1.
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([2, 2, 2, 0])
    C = confusion_matrix(y_true, y_pred, n_classes=3)
    assert C[1, 2] == 3 and C[0, 0] == 1

    naive = np.zeros((3, 3), dtype=np.int64)
    naive[y_true, y_pred] += 1      # buffered: each distinct (row, col) written once
    assert naive[1, 2] == 1         # documents WHY confusion_matrix uses np.add.at


def test_per_class_metrics_small_case():
    # true:  0 0 0 0 1 1 2 2      pred:  0 0 0 1 1 1 0 2
    C = confusion_matrix(np.array([0, 0, 0, 0, 1, 1, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 0, 2]), 3)
    precision, recall, support = per_class_metrics(C)
    assert support.tolist() == [4, 2, 2]
    assert recall.tolist() == pytest.approx([3 / 4, 2 / 2, 1 / 2])
    assert precision.tolist() == pytest.approx([3 / 4, 2 / 3, 1 / 1])


def test_per_class_metrics_never_predicted_class_is_nan():
    C = confusion_matrix(np.array([0, 1]), np.array([0, 0]), 2)
    precision, recall, _ = per_class_metrics(C)
    assert np.isnan(precision[1]) and recall[1] == 0.0


def test_most_confused_pairs_directed_and_symmetric():
    C = np.zeros((4, 4), dtype=np.int64)
    C[3, 1] = 5     # 3 -> 1
    C[1, 3] = 4     # 1 -> 3
    C[0, 2] = 6     # 0 -> 2
    C[2, 2] = 99    # diagonal: never a "confusion"
    assert most_confused_pairs(C, k=2) == [(6, 0, 2), (5, 3, 1)]
    assert most_confused_pairs(C, symmetric=True) == [(9, 1, 3), (6, 0, 2)]


def test_most_confused_pairs_tie_order_is_deterministic():
    C = np.zeros((3, 3), dtype=np.int64)
    C[2, 0] = C[0, 1] = C[1, 2] = 1
    assert most_confused_pairs(C) == [(1, 0, 1), (1, 1, 2), (1, 2, 0)]

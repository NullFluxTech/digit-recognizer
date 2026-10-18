"""Tests for digits.data: IDX parsing, download checksums, scaling, split."""

import gzip
import hashlib
import struct

import numpy as np
import pytest

from digits import data
from digits.data import (
    DATA_DIR,
    FILES,
    download_file,
    images_to_matrix,
    load_idx_gz,
    parse_idx,
    train_val_split,
)


def make_idx(arr: np.ndarray, type_code: int) -> bytes:
    """Build IDX bytes by hand with struct (independent of the parser under test).

    struct format: ">" big-endian, "B" unsigned byte, "I" unsigned 4-byte int.
    """
    header = struct.pack(">BBBB", 0, 0, type_code, arr.ndim)
    header += struct.pack(">" + "I" * arr.ndim, *arr.shape)
    return header + arr.tobytes()  # arr must already have the big-endian dtype


# --- IDX parser ------------------------------------------------------------

def test_parse_uint8_images_roundtrip():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(3, 28, 28), dtype=np.uint8)
    out = parse_idx(make_idx(imgs, 0x08))
    assert out.shape == (3, 28, 28)
    assert out.dtype == np.uint8
    assert np.array_equal(out, imgs)


def test_parse_uint8_labels_roundtrip():
    labels = np.array([5, 0, 4, 1, 9], dtype=np.uint8)
    out = parse_idx(make_idx(labels, 0x08))
    assert out.shape == (5,)
    assert np.array_equal(out, labels)


def test_parse_big_endian_int32():
    # 1 stored big-endian is 00 00 00 01; a little-endian misread would give 16777216
    values = np.array([[1, -2], [300, 70000]], dtype=">i4")
    out = parse_idx(make_idx(values, 0x0C))
    assert out.shape == (2, 2)
    assert out.tolist() == [[1, -2], [300, 70000]]


def test_parse_header_dims_are_big_endian():
    raw = bytes([0, 0, 0x08, 1, 0, 0, 1, 0]) + bytes(256)  # dim = 0x00000100 = 256
    assert parse_idx(raw).shape == (256,)


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"\x00\x00", "too short"),
        (bytes([1, 0, 0x08, 1, 0, 0, 0, 1, 7]), "magic"),
        (bytes([0, 0, 0x42, 1, 0, 0, 0, 1, 7]), "type code"),
        (bytes([0, 0, 0x08, 3, 0, 0, 0, 1]), "only 8 bytes"),
        (bytes([0, 0, 0x08, 1, 0, 0, 0, 4, 1, 2, 3]), "payload is 3 bytes"),     # truncated
        (bytes([0, 0, 0x08, 1, 0, 0, 0, 2, 1, 2, 3]), "payload is 3 bytes"),     # trailing junk
    ],
)
def test_parse_rejects_malformed(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_idx(raw)


def test_load_idx_gz(tmp_path):
    labels = np.arange(10, dtype=np.uint8)
    path = tmp_path / "labels.gz"
    path.write_bytes(gzip.compress(make_idx(labels, 0x08)))
    assert np.array_equal(load_idx_gz(path), labels)


# --- download + checksum (file:// URLs, no network) --------------------------

def test_download_file_accepts_matching_md5(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello mnist")
    dest = tmp_path / "dest.bin"
    download_file(src.as_uri(), dest, hashlib.md5(b"hello mnist").hexdigest())
    assert dest.read_bytes() == b"hello mnist"


def test_download_file_rejects_bad_md5_and_leaves_nothing(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"corrupted")
    dest = tmp_path / "dest.bin"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file(src.as_uri(), dest, "0" * 32)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == [src]  # no leftover .part file


def test_download_mnist_falls_back_to_second_mirror(tmp_path, monkeypatch):
    # Fake mirror 2 holds one tiny file; mirror 1 does not exist.
    mirror2 = tmp_path / "mirror2"
    mirror2.mkdir()
    payload = b"fake labels"
    (mirror2 / "labels.gz").write_bytes(payload)
    monkeypatch.setattr(data, "FILES", {"labels": ("labels.gz", hashlib.md5(payload).hexdigest())})

    out_dir = tmp_path / "data"
    mirrors = [(tmp_path / "missing").as_uri() + "/", mirror2.as_uri() + "/"]
    paths = data.download_mnist(out_dir, mirrors=mirrors)
    assert paths["labels"].read_bytes() == payload


# --- scaling + split -----------------------------------------------------------

def test_images_to_matrix_scales_and_flattens():
    imgs = np.zeros((2, 28, 28), dtype=np.uint8)
    imgs[0, 0, 0] = 255
    imgs[1, 27, 27] = 51
    X = images_to_matrix(imgs)
    assert X.shape == (2, 784)
    assert X.dtype == np.float32
    assert X[0, 0] == 1.0
    assert X[1, 783] == pytest.approx(0.2)  # row-major: (27, 27) -> 27*28 + 27 = 783
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_split_sizes_disjoint_and_complete():
    n = 100
    X = np.arange(n, dtype=np.float32).reshape(n, 1)  # row i holds the value i, so rows are traceable
    y = np.arange(n)
    X_tr, y_tr, X_va, y_va = train_val_split(X, y, n_val=20, seed=0)
    assert X_tr.shape == (80, 1) and X_va.shape == (20, 1)
    tr, va = set(y_tr.tolist()), set(y_va.tolist())
    assert tr.isdisjoint(va)
    assert tr | va == set(range(n))
    # labels stayed attached to their rows
    assert np.array_equal(X_tr[:, 0], y_tr) and np.array_equal(X_va[:, 0], y_va)


def test_split_is_reproducible_and_seed_dependent():
    X = np.arange(1000, dtype=np.float32).reshape(1000, 1)
    y = np.arange(1000)
    a = train_val_split(X, y, n_val=100, seed=0)[3]
    b = train_val_split(X, y, n_val=100, seed=0)[3]
    c = train_val_split(X, y, n_val=100, seed=1)[3]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("n_val", [0, 10, -1])
def test_split_rejects_bad_sizes(n_val):
    with pytest.raises(ValueError):
        train_val_split(np.zeros((10, 1)), np.zeros(10), n_val=n_val, seed=0)


# --- real MNIST (only if prepare_data.py has been run) ---------------------------

have_mnist = all((DATA_DIR / name).exists() for name, _ in FILES.values())


@pytest.mark.skipif(not have_mnist, reason="MNIST not downloaded; run `uv run prepare_data.py`")
def test_real_mnist_shapes_and_split():
    d = data.load_mnist()
    assert d.X_train.shape == (55000, 784) and d.y_train.shape == (55000,)
    assert d.X_val.shape == (5000, 784) and d.y_val.shape == (5000,)
    assert d.X_test.shape == (10000, 784) and d.y_test.shape == (10000,)
    assert d.X_train.dtype == np.float32 and d.y_train.dtype == np.int64
    assert 0.0 <= d.X_train.min() and d.X_train.max() <= 1.0
    assert set(np.unique(d.y_train).tolist()) == set(range(10))
    # well-known MNIST statistic: mean training pixel = 0.1307
    assert d.X_train.mean() == pytest.approx(0.1307, abs=1e-3)

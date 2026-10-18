"""MNIST: download + checksum, IDX parsing, scaling to [0, 1], train/validation split.

Shape conventions used everywhere in this project:
    N   number of examples
    784 = 28 * 28 pixels, one image flattened into a row vector
    X   (N, 784) float32, pixel values in [0, 1]   ("design matrix": one example per row)
    y   (N,)     int64,   class labels 0..9
"""

import gzip
import hashlib
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Two independent mirrors that serve byte-identical copies of the original files.
# (The original host, yann.lecun.com, is often down or rate-limited.)
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

# key -> (file name, published MD5 of the .gz file)
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

# IDX type code (3rd byte of the magic number) -> NumPy dtype.
# ">" means big-endian: the most significant byte comes first in the file.
IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


# ---------------------------------------------------------------------------
# Download + checksum
# ---------------------------------------------------------------------------

def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_file(url: str, dest: Path, md5: str, timeout: float = 60.0) -> None:
    """Download url to dest, keeping it only if its MD5 matches. Raises on failure."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            tmp.write_bytes(resp.read())
        actual = md5_of(tmp)
        if actual != md5:
            raise ValueError(f"checksum mismatch for {url}: expected {md5}, got {actual}")
        tmp.replace(dest)  # only a verified file ever appears under its real name
    finally:
        tmp.unlink(missing_ok=True)


def download_mnist(data_dir: Path = DATA_DIR, mirrors: list[str] = MIRRORS) -> dict[str, Path]:
    """Make sure the four MNIST files are in data_dir with correct checksums.

    Files that already exist and verify are not downloaded again.
    Returns {key: path} for the keys in FILES.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, md5) in FILES.items():
        dest = data_dir / name
        if dest.exists() and md5_of(dest) == md5:
            print(f"  ok        {name}")
        else:
            errors = []
            for mirror in mirrors:
                try:
                    print(f"  download  {mirror}{name}")
                    download_file(mirror + name, dest, md5)
                    break
                except Exception as e:  # network error or bad checksum: try the next mirror
                    errors.append(f"{mirror}: {e}")
            else:  # for/else: runs only if the loop never hit `break`
                raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors))
            print(f"  verified  {name} (md5 {md5})")
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# IDX parsing
# ---------------------------------------------------------------------------

def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX file (already decompressed) into an array of its stored shape.

    Layout (all header integers are 4-byte big-endian unsigned):
        bytes 0-1   0x00 0x00
        byte  2     type code (0x08 = uint8, see IDX_DTYPES)
        byte  3     number of dimensions D
        next 4*D    size of each dimension, d_0 ... d_{D-1}
        rest        d_0 * ... * d_{D-1} values, row-major (last index changes fastest)
    """
    if len(raw) < 4:
        raise ValueError(f"IDX data too short for a header: {len(raw)} bytes")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"bad IDX magic number: first two bytes are {raw[0]:#04x} {raw[1]:#04x}, expected 0x00 0x00")
    type_code, ndim = raw[2], raw[3]
    if type_code not in IDX_DTYPES:
        raise ValueError(f"unknown IDX type code {type_code:#04x}")
    dtype = IDX_DTYPES[type_code]

    header_len = 4 + 4 * ndim
    if len(raw) < header_len:
        raise ValueError(f"IDX header says {ndim} dims but file has only {len(raw)} bytes")
    # Read D big-endian uint32 sizes straight out of the byte buffer (no copy).
    dims = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4)  # (D,)
    shape = tuple(int(d) for d in dims)

    payload = len(raw) - header_len
    expected = math.prod(shape) * dtype.itemsize
    if payload != expected:
        raise ValueError(f"IDX payload is {payload} bytes, header shape {shape} needs {expected}")

    data = np.frombuffer(raw, dtype=dtype, offset=header_len)  # (d_0*...*d_{D-1},) flat view
    return data.reshape(shape)                                 # (d_0, ..., d_{D-1}) view


def load_idx_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return parse_idx(f.read())


# ---------------------------------------------------------------------------
# Scaling + split
# ---------------------------------------------------------------------------

def images_to_matrix(images: np.ndarray) -> np.ndarray:
    """uint8 images (N, 28, 28) in 0..255 -> float32 design matrix (N, 784) in [0, 1]."""
    n = images.shape[0]
    flat = images.reshape(n, -1)                    # (N,28,28) -> (N,784), -1 = "infer this size"
    return flat.astype(np.float32) / np.float32(255.0)  # (N,784) uint8 -> (N,784) float32 in [0,1]


def train_val_split(X: np.ndarray, y: np.ndarray, n_val: int, seed: int):
    """Shuffle rows with a seeded RNG, then cut off the last n_val rows as validation.

    Returns X_train, y_train, X_val, y_val. X and y get the SAME permutation,
    so row i of X still matches label i of y.
    """
    n = X.shape[0]
    if not 0 < n_val < n:
        raise ValueError(f"n_val must be between 1 and {n - 1}, got {n_val}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                       # (N,) every index 0..N-1 once, shuffled
    train_idx, val_idx = perm[:-n_val], perm[-n_val:]  # (N-n_val,), (n_val,)
    # Fancy indexing: X[idx] gathers rows idx[0], idx[1], ... into a new array.
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]  # (N-n_val,784) (N-n_val,) (n_val,784) (n_val,)


@dataclass
class MNIST:
    X_train: np.ndarray  # (55000, 784) float32
    y_train: np.ndarray  # (55000,)     int64
    X_val: np.ndarray    # (5000, 784)  float32
    y_val: np.ndarray    # (5000,)      int64
    X_test: np.ndarray   # (10000, 784) float32  -- touch once per finished model
    y_test: np.ndarray   # (10000,)     int64


def load_mnist(data_dir: Path = DATA_DIR, n_val: int = 5000, seed: int = 0) -> MNIST:
    """Download (if needed), verify, parse, scale and split MNIST."""
    paths = download_mnist(data_dir)
    train_images = load_idx_gz(paths["train_images"])  # (60000,28,28) uint8
    train_labels = load_idx_gz(paths["train_labels"])  # (60000,)      uint8
    test_images = load_idx_gz(paths["test_images"])    # (10000,28,28) uint8
    test_labels = load_idx_gz(paths["test_labels"])    # (10000,)      uint8

    for images, labels in [(train_images, train_labels), (test_images, test_labels)]:
        if images.shape[1:] != (28, 28) or images.shape[0] != labels.shape[0]:
            raise ValueError(f"unexpected shapes: images {images.shape}, labels {labels.shape}")

    X_all = images_to_matrix(train_images)       # (60000,784) float32
    y_all = train_labels.astype(np.int64)        # (60000,)    int64
    X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, n_val=n_val, seed=seed)
    X_test = images_to_matrix(test_images)       # (10000,784) float32
    y_test = test_labels.astype(np.int64)        # (10000,)    int64
    return MNIST(X_train, y_train, X_val, y_val, X_test, y_test)

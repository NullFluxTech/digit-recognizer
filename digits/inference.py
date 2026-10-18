"""Shared helpers for predict.py, draw.py and score_samples.py."""

import re
from pathlib import Path

import numpy as np

from digits.models import Sequential, load_any
from digits.nn import softmax
from digits.preprocess import load_gray, to_mnist

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_personal.npz"   # CNN + augmentation + my own digits
SAMPLES_DIR = ROOT / "samples"
SAMPLE_NAME = re.compile(r"^(\d)_(\d+)\.png$")   # "7_03.png" -> label 7, number 3


def load_model(path: Path = MODEL_PATH) -> Sequential:
    if not path.exists():
        hints = {"cnn_personal.npz": "uv run train_cnn.py --personal", "cnn.npz": "uv run train_cnn.py",
                 "mlp_personal.npz": "uv run train_mlp.py --personal", "mlp_aug.npz": "uv run train_mlp.py --augment"}
        hint = hints.get(path.name, "uv run train_mlp.py")
        raise SystemExit(f"{path} not found: run `{hint}` first")
    return load_any(path)


def predict_probs(model, x: np.ndarray) -> np.ndarray:
    """One preprocessed image (784,) -> class probabilities (10,)."""
    # x[None, :] adds a batch axis of size 1: (784,) -> (1,784), because the model expects rows
    return softmax(model.forward(x[None, :]))[0]  # (1,784) -> (1,10) -> (10,)


def top_k(probs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """[(digit, probability), ...] for the k most likely digits, most likely first."""
    order = np.argsort(-probs, kind="stable")[:k]  # (k,) class indices by decreasing probability
    return [(int(d), float(probs[d])) for d in order]


def parse_sample_name(name: str) -> tuple[int, int] | None:
    """'7_03.png' -> (7, 3); anything else -> None."""
    m = SAMPLE_NAME.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_samples(folder: Path, verbose: bool = True):
    """Preprocess every <label>_<nn>.png in folder.

    Returns X (n,784) float32, y (n,) int64 labels, names (n,) file names, sorted by name.
    Files with other names are skipped (and reported if verbose).
    """
    xs, ys, names = [], [], []
    for f in sorted(folder.glob("*.png")) if folder.exists() else []:
        parsed = parse_sample_name(f.name)
        if parsed is None:
            if verbose:
                print(f"  skipping {f.name}: name is not <digit>_<number>.png")
            continue
        xs.append(to_mnist(load_gray(f)))               # (784,)
        ys.append(parsed[0])
        names.append(f.name)
    X = np.stack(xs) if xs else np.zeros((0, 784), np.float32)  # (n,784)
    return X, np.array(ys, dtype=np.int64), names


def next_sample_path(label: int, samples_dir: Path = SAMPLES_DIR) -> Path:
    """First free samples/<label>_<nn>.png, numbering from 01."""
    taken = set()
    for p in samples_dir.glob(f"{label}_*.png"):
        parsed = parse_sample_name(p.name)
        if parsed is not None:
            taken.add(parsed[1])
    n = 1
    while n in taken:
        n += 1
    return samples_dir / f"{label}_{n:02d}.png"

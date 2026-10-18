"""Milestone 1: download + verify MNIST, parse it, split it, save a grid of samples.

    uv run prepare_data.py
"""

from pathlib import Path

import numpy as np

from digits.data import load_mnist
from digits.plots import save_digit_grid

REPORTS = Path(__file__).resolve().parent / "reports"
SEED = 0


def main() -> None:
    print("MNIST files:")
    data = load_mnist(seed=SEED)

    print("\nShapes and dtypes:")
    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        a = getattr(data, name)
        print(f"  {name:8s} {str(a.shape):14s} {a.dtype}")

    print(f"\nPixel range: [{data.X_train.min():.1f}, {data.X_train.max():.1f}]")
    # axis=None (the default) reduces over every element of the matrix
    print(f"Train pixel mean {data.X_train.mean():.4f}, std {data.X_train.std():.4f}")

    print("\nExamples per class:")
    print("  digit  " + " ".join(f"{d:5d}" for d in range(10)))
    for name, y in [("train", data.y_train), ("val", data.y_val), ("test", data.y_test)]:
        counts = np.bincount(y, minlength=10)  # (10,) counts[d] = number of labels equal to d
        print(f"  {name:5s}  " + " ".join(f"{c:5d}" for c in counts))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(data.X_train.shape[0], size=60, replace=False)  # (60,) random row numbers
    path = REPORTS / "sample_digits.png"
    save_digit_grid(data.X_train[idx], [f"label {d}" for d in data.y_train[idx]], path,
                    suptitle="Random training images with their labels")
    print(f"\nSaved {path.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()

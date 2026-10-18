"""Predict the digit in an image file.

    uv run predict.py path/to/digit.png
    uv run predict.py path/to/digit.png --debug     also save reports/predict_steps.png

Works with dark-on-light (paper, scans, photos) and light-on-dark images of a single digit.
"""

import argparse
from pathlib import Path

from digits.inference import MODEL_PATH, ROOT, load_model, predict_probs, top_k
from digits.plots import save_preprocess_steps
from digits.preprocess import load_gray, to_mnist_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=f"default: models/{MODEL_PATH.name}")
    parser.add_argument("--debug", action="store_true",
                        help="save every preprocessing step to reports/predict_steps.png")
    args = parser.parse_args()

    steps = to_mnist_steps(load_gray(args.image))
    probs = predict_probs(load_model(args.model), steps["final"].reshape(-1))   # (28,28) -> (784,) -> (10,)
    best = top_k(probs)
    print(f"{args.image.name}: {best[0][0]}")
    for digit, p in best:
        print(f"  {digit}  {p:7.2%}")
    if args.debug:
        out = ROOT / "reports" / "predict_steps.png"
        save_preprocess_steps(steps, out, f"{args.image.name}: predicted {best[0][0]} ({best[0][1]:.1%})")
        print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

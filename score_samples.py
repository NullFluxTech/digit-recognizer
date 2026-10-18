"""Score every handwritten sample in samples/ (files named <label>_<nn>.png, e.g. 7_03.png).

    uv run score_samples.py

Prints overall and per-digit accuracy, lists the misses, and saves reports/samples_scored.png
(each sample after preprocessing, i.e. exactly what the model saw).
"""

import argparse
from pathlib import Path

import numpy as np

from digits.inference import MODEL_PATH, ROOT, SAMPLES_DIR, load_model, load_samples, top_k
from digits.nn import softmax
from digits.plots import save_digit_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=SAMPLES_DIR, help="folder with <label>_<nn>.png files")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "samples_scored.png",
                        help="where to save the gallery")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=f"default: models/{MODEL_PATH.name}")
    args = parser.parse_args()

    X, labels, names = load_samples(args.dir)          # (n,784), (n,), n names
    if len(names) == 0:
        raise SystemExit(f"No samples found in {args.dir}. Draw some with `uv run draw.py`.")
    model = load_model(args.model)
    print(f"Model: {args.model.name}")

    P = softmax(model.forward(X))                      # (n,784) -> (n,10)
    preds = P.argmax(axis=1)                           # (n,)
    confs = P[np.arange(len(preds)), preds]            # (n,) probability of the predicted digit
    correct = preds == labels                          # (n,) bools
    for i in np.flatnonzero(~correct):
        alts = ", ".join(f"{d} {p:.0%}" for d, p in top_k(P[i]))
        print(f"  MISS {names[i]}: predicted {preds[i]}  (top 3: {alts})")

    print(f"\n{correct.sum()} of {len(correct)} correct ({correct.mean():.1%})")
    print("\n  digit  samples  correct")
    for d in range(10):
        mask = labels == d                             # (n,) bools: samples of digit d
        if mask.any():
            print(f"  {d:5d}  {mask.sum():7d}  {correct[mask].sum():7d}")

    titles = [f"{n.removesuffix('.png')}\n→{p} {c:.0%}" + ("" if ok else "  ✗")
              for n, p, c, ok in zip(names, preds, confs, correct)]
    save_digit_grid(X, titles, args.out, cols=10, title_size=8, row_height=1.45,
                    suptitle=f"My handwriting after preprocessing: {correct.sum()}/{len(correct)} correct "
                             f"(name, →prediction, confidence, ✗ = miss)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

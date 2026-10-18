"""Milestone 2: softmax regression (784 -> 10) trained with mini-batch SGD.

    uv run train_softmax.py            train with the chosen hyperparameters, test once, save
    uv run train_softmax.py --sweep    try a grid of hyperparameters on VALIDATION only

The sweep never looks at the test set. The default values below were picked from it.
"""

import argparse
import itertools
import time
from pathlib import Path

from digits.data import load_mnist
from digits.models import SoftmaxRegression
from digits.nn import SGD
from digits.plots import save_curves, save_weight_images
from digits.train import evaluate, train

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "softmax.npz"
SEED = 0

# Chosen from `--sweep`: the top runs all reach ~92.1-92.2% validation accuracy (differences of
# 2-3 images out of 5000 = noise), so pick the fastest among them: batch 64 needs 4x fewer
# steps than batch 16 for the same accuracy.
LR = 0.1
BATCH_SIZE = 64
EPOCHS = 15


def sweep(data) -> None:
    lrs = [0.01, 0.05, 0.1, 0.5, 1.0]
    batch_sizes = [16, 64, 256]
    epochs = 15
    print(f"Sweep: {len(lrs) * len(batch_sizes)} runs x {epochs} epochs, scored on validation\n")
    print(f"  {'lr':>5s} {'batch':>5s}  {'best val acc':>12s} {'at epoch':>8s}  {'final val acc':>13s}")
    results = []
    for lr, bs in itertools.product(lrs, batch_sizes):
        model = SoftmaxRegression()
        hist = train(model, SGD(lr), data.X_train, data.y_train, data.X_val, data.y_val,
                     epochs=epochs, batch_size=bs, seed=SEED, verbose=False)
        best = max(range(len(hist["val_acc"])), key=hist["val_acc"].__getitem__)
        results.append((hist["val_acc"][-1], lr, bs))
        print(f"  {lr:5.2f} {bs:5d}  {hist['val_acc'][best]:12.4f} {best:8d}  {hist['val_acc'][-1]:13.4f}")
    acc, lr, bs = max(results)
    print(f"\nBest final validation accuracy: {acc:.4f} with lr={lr}, batch={bs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="hyperparameter grid on validation only")
    args = parser.parse_args()

    data = load_mnist(seed=SEED)
    if args.sweep:
        sweep(data)
        return

    print(f"\nSoftmax regression, SGD lr={LR}, batch={BATCH_SIZE}, {EPOCHS} epochs")
    model = SoftmaxRegression()
    t0 = time.perf_counter()
    hist = train(model, SGD(LR), data.X_train, data.y_train, data.X_val, data.y_val,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED)
    print(f"Training time: {time.perf_counter() - t0:.1f}s")

    # The model is finished: look at the test set exactly once.
    test_loss, test_acc = evaluate(model, data.X_test, data.y_test)
    print(f"\nFinal  val acc {hist['val_acc'][-1]:.4f}   test acc {test_acc:.4f} (loss {test_loss:.4f})")

    model.save(MODEL_PATH)
    save_curves(hist, REPORTS / "softmax_curves.png",
                f"Softmax regression: SGD lr={LR}, batch={BATCH_SIZE}")
    save_weight_images(model.fc.W, REPORTS / "softmax_weights.png",
                       [f"digit {d}" for d in range(10)],
                       suptitle="Softmax regression weights W[:, d] (red +, blue -)")
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}, reports/softmax_curves.png, reports/softmax_weights.png")


if __name__ == "__main__":
    main()

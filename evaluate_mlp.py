"""Milestone 4: evaluate the finished MLP (models/mlp.npz) on the test set.

    uv run evaluate_mlp.py

No training or tuning happens here. The model was finished (and its test accuracy reported) in
milestone 3; this script only takes a closer look at those same test predictions.
Writes reports/confusion_matrix.png, reports/misclassified.png, reports/mlp_first_layer.png.
"""

import sys
from pathlib import Path

import numpy as np

from digits.data import load_mnist
from digits.metrics import confusion_matrix, most_confused_pairs, per_class_metrics
from digits.models import MLP
from digits.nn import softmax
from digits.plots import save_confusion_matrix, save_digit_grid, save_weight_images

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "mlp.npz"


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH.relative_to(ROOT)} not found: run `uv run train_mlp.py` first")
    data = load_mnist()
    model = MLP.load(MODEL_PATH)

    P = softmax(model.forward(data.X_test))          # (10000,784) -> (10000,10) probabilities
    pred = P.argmax(axis=1)                          # (10000,)
    y = data.y_test                                  # (10000,)
    wrong = np.flatnonzero(pred != y)                # (n_wrong,) indices of the mistakes
    print(f"\nTest accuracy {1 - len(wrong) / len(y):.4f}  ({len(wrong)} errors out of {len(y)})")

    # --- confusion matrix -------------------------------------------------------------------
    C = confusion_matrix(y, pred)                    # (10,10)
    print("\nConfusion matrix (rows = true digit, columns = predicted digit):")
    print("        " + "".join(f"{j:6d}" for j in range(10)))
    for i in range(10):
        print(f"  {i:4d}  " + "".join(f"{C[i, j]:6d}" for j in range(10)))

    precision, recall, support = per_class_metrics(C)
    errors_per_class = support - np.diag(C)          # (10,) misclassified true-class images
    print("\n  digit  support  errors  recall  precision")
    for d in range(10):
        print(f"  {d:5d}  {support[d]:7d}  {errors_per_class[d]:6d}  {recall[d]:6.2%}  {precision[d]:9.2%}")

    print("\nMost confused (true -> predicted):")
    for count, t, p in most_confused_pairs(C, k=10):
        print(f"  {t} -> {p}: {count}")
    print("\nMost confused pairs, both directions together:")
    for count, a, b in most_confused_pairs(C, k=5, symmetric=True):
        print(f"  {a} <-> {b}: {count}  ({C[a, b]} as {a}->{b}, {C[b, a]} as {b}->{a})")

    save_confusion_matrix(C, REPORTS / "confusion_matrix.png",
                          f"MLP confusion matrix on the 10,000 test images ({len(wrong)} errors)")

    # --- misclassified gallery, most confident mistakes first ----------------------------------
    conf = P[wrong, pred[wrong]]                     # (n_wrong,) probability given to the wrong answer
    order = wrong[np.argsort(-conf, kind="stable")]  # (n_wrong,) indices sorted by confidence, high first
    titles = [f"{y[i]}→{pred[i]}  {P[i, pred[i]]:.0%}" for i in order]
    save_digit_grid(data.X_test[order], titles, REPORTS / "misclassified.png", cols=16,
                    title_size=7.5, row_height=1.2,
                    suptitle=f"All {len(wrong)} misclassified test images: true→predicted, "
                             f"confidence in the prediction (most confident first)")
    n_sure = int((conf > 0.9).sum())
    print(f"\nMistakes made with > 90% confidence: {n_sure} of {len(wrong)}")

    # --- first-layer weights -------------------------------------------------------------------
    W1, b1 = model.fc1.W, model.fc1.b                # (784,512), (512,)
    H = data.X_train @ W1 + b1                       # (55000,784) @ (784,512) -> (55000,512)
    fires = (H > 0).mean(axis=0)                     # (512,) fraction of training images that activate each unit
    by_freq = np.argsort(-fires, kind="stable")      # (512,) units, most often active first
    dead = np.flatnonzero(fires == 0)
    rare = [u for u in by_freq[::-1] if 0 < fires[u]][:5]   # least active units that aren't dead
    chosen = list(by_freq[:30]) + rare + list(dead[:5])
    W2 = model.fc2.W                                 # (512,10): row j = how unit j's output pushes each class

    def unit_title(u: int) -> str:
        n_img = int(round(fires[u] * H.shape[0]))
        active = "dead" if n_img == 0 else (f"{fires[u]:.0%}" if fires[u] >= 0.01 else f"{n_img} image" + "s" * (n_img > 1))
        return f"unit {u}  {active}\n+{W2[u].argmax()}  −{W2[u].argmin()}"

    save_weight_images(W1[:, chosen], REPORTS / "mlp_first_layer.png",
                       [unit_title(u) for u in chosen], cols=10,
                       suptitle="MLP first-layer weights W1[:, j] (red +, blue −). Rows 1-3: most often "
                                "active units; row 4: 5 rarest, then 5 dead.\nTitle: % of training images "
                                "that activate the unit; +d / −d = digit the unit votes most for / against "
                                "(largest / smallest entry of W2[j, :])")
    print(f"Hidden units: {len(dead)} dead, active on {fires.mean():.1%} of images on average, "
          f"most active unit fires on {fires.max():.0%}")
    print("\nSaved reports/confusion_matrix.png, reports/misclassified.png, reports/mlp_first_layer.png")


if __name__ == "__main__":
    main()

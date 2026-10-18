"""Milestone 6: a small CNN with im2col, trained with augmentation.

    uv run train_cnn.py              train, test once, save models/cnn.npz and the reports
    uv run train_cnn.py --sweep      learning-rate comparison, VALIDATION only (short runs)
    uv run train_cnn.py --val-only   train and report validation accuracy only (no test, nothing saved)
    uv run train_cnn.py --shift      MLP vs CNN on shifted validation digits (needs saved models)
    uv run train_cnn.py --personal   also train on my own digits in personal/ -> models/cnn_personal.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from digits.augment import affine_warp, augment_batch
from digits.data import load_mnist
from digits.inference import load_samples
from digits.models import CNN, MLP
from digits.nn import SGDMomentum
from digits.plots import save_curves, save_feature_maps, save_shift_chart, save_weight_images
from digits.train import cosine_schedule, evaluate, train

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "cnn.npz"
PERSONAL_MODEL_PATH = ROOT / "models" / "cnn_personal.npz"
PERSONAL_DIR = ROOT / "personal"
SEED = 0

C1, C2, HIDDEN = 8, 16, 128
LR = 0.05             # chosen with --sweep (see LEARN.md section 6.6)
MOMENTUM = 0.9
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 5
TRAIN_EVAL_SIZE = 5000   # report train accuracy on a fixed 5,000-image subset: a full pass costs ~8 s
EVAL_BATCH = 1000        # smaller evaluation batches keep im2col's patch matrices small
# Each personal digit is repeated this many times per epoch. Taken from the MLP's 5-fold check
# (train_mlp.py --personal-cv); repeating that check for the CNN would take about an hour.
PERSONAL_REPEAT = 20


def run(data, lr, epochs=EPOCHS, patience=PATIENCE, extra=None, verbose=False):
    """Train one CNN. extra = (X, y, repeat): additional training images, each repeated."""
    X_train, y_train = data.X_train, data.y_train                      # (55000,784), (55000,)
    if extra is not None:
        Xe, ye, repeat = extra
        X_train = np.concatenate([X_train, np.repeat(Xe, repeat, axis=0)])  # (55000 + n*repeat, 784)
        y_train = np.concatenate([y_train, np.repeat(ye, repeat)])         # (55000 + n*repeat,)
    model = CNN(C1, C2, HIDDEN, seed=SEED)
    hist = train(model, SGDMomentum(lr, MOMENTUM), X_train, y_train, data.X_val, data.y_val,
                 epochs=epochs, batch_size=BATCH_SIZE, seed=SEED, schedule=cosine_schedule(lr, epochs),
                 patience=patience, restore_best=True, augment=augment_batch,
                 train_eval_size=TRAIN_EVAL_SIZE, eval_batch_size=EVAL_BATCH, verbose=verbose)
    return model, hist


def sweep(data) -> None:
    epochs = 5
    print(f"Learning rate, {epochs}-epoch runs (cosine over {epochs}), validation only:\n")
    for lr in [0.01, 0.03, 0.05, 0.1]:
        _, h = run(data, lr, epochs=epochs, patience=None)
        b = h["best_epoch"]
        print(f"  lr {lr:5.2f}:  best val acc {h['val_acc'][b]:.4f} (epoch {b})  {h['seconds']:.0f}s")


def shift_experiment(data, models: dict) -> None:
    """Accuracy on validation digits moved 0-4 px to the right, for each model."""
    imgs = data.X_val.reshape(-1, 28, 28)
    B = len(imgs)
    eye = np.repeat(np.eye(2)[None], B, axis=0)                  # (B,2,2) identity: pure shift
    shifts = [0, 1, 2, 3, 4]
    results = {}
    for name, model in models.items():
        accs = []
        for s in shifts:
            moved = affine_warp(imgs, eye, np.tile([0.0, float(s)], (B, 1))).reshape(B, -1)  # (B,784)
            accs.append(evaluate(model, moved, data.y_val, EVAL_BATCH)[1])
        results[name] = accs
        print(f"  {name:28s} " + "  ".join(f"{s}px {a:.4f}" for s, a in zip(shifts, accs)))
    save_shift_chart(shifts, results, REPORTS / "shift_robustness.png",
                     "Shifted digits: MLPs vs CNN (MNIST validation)")


def load_models_for_shift(cnn) -> dict:
    models = {}
    for name, path in [("MLP (milestone 3)", ROOT / "models" / "mlp.npz"),
                       ("MLP + augmentation", ROOT / "models" / "mlp_aug.npz")]:
        if path.exists():
            models[name] = MLP.load(path)
    models["CNN + augmentation"] = cnn
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--personal", action="store_true", help="also train on personal/")
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = load_mnist(seed=SEED)
    if args.sweep:
        sweep(data)
        return
    if args.shift:
        print("\nShift experiment (validation only):")
        shift_experiment(data, load_models_for_shift(CNN.load(MODEL_PATH)))
        return

    print(f"\nCNN conv5x5({C1})-pool-conv3x3({C2})-pool-fc({HIDDEN})-fc(10), SGD momentum {MOMENTUM}, "
          f"lr {LR} cosine, batch {BATCH_SIZE}, up to {EPOCHS} epochs, augmentation on")
    extra = None
    if args.personal:
        Xp, yp, _ = load_samples(PERSONAL_DIR)
        extra = (Xp, yp, PERSONAL_REPEAT)
        print(f"Personal digits: {len(yp)} from {PERSONAL_DIR.name}/, each repeated {PERSONAL_REPEAT}x")
    model, hist = run(data, LR, extra=extra, verbose=True)
    b = hist["best_epoch"]
    if args.val_only:
        print(f"\nBest epoch {b}: val acc {hist['val_acc'][b]:.4f}  (test set not used, nothing saved)")
        print(f"Total time: {time.perf_counter() - t_start:.1f}s")
        return

    # The model is finished: look at the test set exactly once.
    test_loss, test_acc = evaluate(model, data.X_test, data.y_test, EVAL_BATCH)
    print(f"\nKept epoch {b}:  val acc {hist['val_acc'][b]:.4f}   test acc {test_acc:.4f} (loss {test_loss:.4f})")
    print(f"Training time {hist['seconds']:.0f}s, total {time.perf_counter() - t_start:.0f}s")

    if args.personal:
        model.save(PERSONAL_MODEL_PATH)
        save_curves(hist, REPORTS / "cnn_personal_curves.png",
                    f"CNN + personal digits: lr {LR} cosine, batch {BATCH_SIZE}, augmented "
                    f"(train curve on a fixed {TRAIN_EVAL_SIZE}-image subset)", show_lr=True, mark_best=True)
        print(f"Saved {PERSONAL_MODEL_PATH.relative_to(ROOT)}, reports/cnn_personal_curves.png")
        return
    model.save(MODEL_PATH)
    save_curves(hist, REPORTS / "cnn_curves.png",
                f"CNN: SGD momentum {MOMENTUM}, lr {LR} cosine, batch {BATCH_SIZE}, augmented "
                f"(train curve on a fixed {TRAIN_EVAL_SIZE}-image subset)", show_lr=True, mark_best=True)
    save_weight_images(model.conv1.W, REPORTS / "cnn_conv1_filters.png",
                       [f"filter {f}" for f in range(C1)], cols=C1, size=5,
                       suptitle="conv1: the 8 learned 5x5 filters (red +, blue -)")
    # Feature maps for the first validation digit: run the layers one by one and keep the outputs.
    x = data.X_val[:1]                                           # (1,784)
    outputs = []
    for layer in model.layers:
        x = layer.forward(x)
        outputs.append(x)
    maps1, maps2 = outputs[2][0], outputs[5][0]                  # after ReLU 1: (8,28,28); after ReLU 2: (16,14,14)
    save_feature_maps(data.X_val[0].reshape(28, 28), maps1, maps2, REPORTS / "cnn_feature_maps.png",
                      f"What the CNN computes for one validation digit (label {data.y_val[0]})")
    print("\nShift experiment (validation only):")
    shift_experiment(data, load_models_for_shift(model))
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and reports/cnn_*.png, reports/shift_robustness.png")


if __name__ == "__main__":
    main()

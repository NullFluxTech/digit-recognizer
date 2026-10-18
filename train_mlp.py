"""Milestone 3: MLP 784 -> H (ReLU) -> 10, SGD + momentum, LR schedule, early stopping.

    uv run train_mlp.py            train from scratch with the chosen settings, test once, save
    uv run train_mlp.py --sweep    hyperparameter search, scored on VALIDATION only
    uv run train_mlp.py --augment  train on randomly warped digits (milestone 5) -> models/mlp_aug.npz
    uv run train_mlp.py --personal also train on my own digits in personal/ -> models/mlp_personal.npz
    uv run train_mlp.py --personal-cv   5-fold check of personalization on personal/ (no test set)
    add --val-only to train and report validation accuracy without touching the test set or saving

The sweep never looks at the test set. The default values below were picked from it.
"""

import argparse
import itertools

import numpy as np
import time
from pathlib import Path

from digits.augment import augment_batch
from digits.data import load_mnist
from digits.inference import load_samples, parse_sample_name
from digits.models import MLP
from digits.nn import SGDMomentum
from digits.plots import save_curves
from digits.train import constant_schedule, cosine_schedule, evaluate, step_schedule, train

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "mlp.npz"
AUG_MODEL_PATH = ROOT / "models" / "mlp_aug.npz"
PERSONAL_MODEL_PATH = ROOT / "models" / "mlp_personal.npz"
PERSONAL_DIR = ROOT / "personal"
SEED = 0

# Chosen from `--sweep` plus a 3-seed check (see LEARN.md section 3.7). All good settings land at
# ~98.2-98.3% validation; seed-to-seed spread is up to 0.4 points. H=512 beat 256 at every lr;
# lr 0.1 and 0.3 tie; cosine at lr 0.3 was the most stable across seeds (98.26-98.38%).
HIDDEN = 512
LR = 0.3
MOMENTUM = 0.9
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
# With augmentation every epoch shows new variants, so learning is slower and noisier:
# twice the epochs, and more patience before early stopping.
AUG_EPOCHS = 30
AUG_PATIENCE = 8
# Personal digits: 50 images next to 55,000 would barely register, so each one is repeated
# PERSONAL_REPEAT times per epoch (augmentation makes every copy a different variant).
PERSONAL_REPEAT = 20
CV_REPEATS = [0, 20, 60]   # compared by --personal-cv
CV_EPOCHS = 15             # shorter runs for the 5-fold comparison


def make_schedule(kind: str, lr: float, epochs: int):
    if kind == "constant":
        return constant_schedule(lr)
    if kind == "step":
        return step_schedule(lr, every=epochs // 3 + 1, gamma=0.1)  # 2 drops: lr, lr/10, lr/100
    return cosine_schedule(lr, epochs)


def run(data, hidden, lr, schedule, epochs=EPOCHS, patience=PATIENCE, augment=None, extra=None,
        verbose=False):
    """Train one MLP. extra = (X, y, repeat): additional training images, each repeated."""
    X_train, y_train = data.X_train, data.y_train                     # (55000,784), (55000,)
    if extra is not None:
        Xe, ye, repeat = extra
        X_train = np.concatenate([X_train, np.repeat(Xe, repeat, axis=0)])  # (55000 + n*repeat, 784)
        y_train = np.concatenate([y_train, np.repeat(ye, repeat)])         # (55000 + n*repeat,)
    model = MLP(n_hidden=hidden, seed=SEED)
    hist = train(model, SGDMomentum(lr, MOMENTUM), X_train, y_train,
                 data.X_val, data.y_val, epochs=epochs, batch_size=BATCH_SIZE, seed=SEED,
                 schedule=make_schedule(schedule, lr, epochs), patience=patience,
                 restore_best=True, augment=augment, verbose=verbose)
    return model, hist


def personal_cv(data) -> None:
    """Does training on my own digits help on my OTHER digits? 5-fold, never touches the test set.

    Fold k holds out every *_0k.png (one sample of each digit); the model trains on MNIST plus the
    other 40 personal digits and is scored on the 10 held out. Repeat 0 = no personal data.
    """
    Xp, yp, names = load_samples(PERSONAL_DIR)                         # (50,784), (50,)
    fold = np.array([parse_sample_name(n)[1] for n in names])          # (50,) 1..5
    print(f"5-fold check on {len(names)} personal digits, {CV_EPOCHS} epochs per run, augmentation on.\n")
    for repeat in CV_REPEATS:
        correct, val_accs = 0, []
        for k in ([None] if repeat == 0 else sorted(set(fold))):
            held = np.ones(len(names), bool) if k is None else fold == k  # (50,) bools
            extra = None if repeat == 0 else (Xp[~held], yp[~held], repeat)
            model, h = run(data, HIDDEN, LR, "cosine", epochs=CV_EPOCHS, patience=None,
                           augment=augment_batch, extra=extra)
            correct += int((model.predict(Xp[held]) == yp[held]).sum())
            val_accs.append(h["val_acc"][h["best_epoch"]])
        print(f"  repeat {repeat:3d}:  held-out personal {correct}/{len(names)}   "
              f"MNIST val acc {np.mean(val_accs):.4f}")


def sweep(data) -> None:
    print(f"Stage 1: hidden size x learning rate (cosine, momentum {MOMENTUM}, batch {BATCH_SIZE}, "
          f"{EPOCHS} epochs, no early stop). Validation only.\n")
    print(f"  {'H':>4s} {'lr':>5s}  {'best val acc':>12s} {'epoch':>5s}  {'train acc':>9s}  {'time':>5s}")
    stage1 = []
    for hidden, lr in itertools.product([128, 256, 512], [0.01, 0.03, 0.1, 0.3]):
        _, h = run(data, hidden, lr, "cosine", patience=None)
        b = h["best_epoch"]
        stage1.append((h["val_acc"][b], hidden, lr))
        print(f"  {hidden:4d} {lr:5.2f}  {h['val_acc'][b]:12.4f} {b:5d}  {h['train_acc'][b]:9.4f}  "
              f"{h['seconds']:4.0f}s")
    _, hidden, lr = max(stage1)

    print(f"\nStage 2: schedule, with H={hidden}, lr={lr}, early stopping patience {PATIENCE}.\n")
    print(f"  {'schedule':>8s}  {'best val acc':>12s} {'epoch':>5s} {'stopped':>7s}")
    for kind in ["constant", "step", "cosine"]:
        _, h = run(data, hidden, lr, kind)
        b = h["best_epoch"]
        print(f"  {kind:>8s}  {h['val_acc'][b]:12.4f} {b:5d} {h['stopped_epoch']:7d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="hyperparameter search on validation only")
    parser.add_argument("--augment", action="store_true", help="train on randomly warped digits")
    parser.add_argument("--personal", action="store_true", help="also train on personal/ (implies --augment)")
    parser.add_argument("--personal-cv", action="store_true", help="5-fold personalization check")
    parser.add_argument("--val-only", action="store_true", help="no test evaluation, nothing saved")
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = load_mnist(seed=SEED)
    if args.sweep:
        sweep(data)
        return
    if args.personal_cv:
        personal_cv(data)
        return
    args.augment = args.augment or args.personal
    extra = None
    if args.personal:
        Xp, yp, _ = load_samples(PERSONAL_DIR)
        extra = (Xp, yp, PERSONAL_REPEAT)
        print(f"Personal digits: {len(yp)} from {PERSONAL_DIR.name}/, each repeated {PERSONAL_REPEAT}x")

    epochs, patience = (AUG_EPOCHS, AUG_PATIENCE) if args.augment else (EPOCHS, PATIENCE)
    print(f"\nMLP 784 -> {HIDDEN} -> 10, SGD momentum {MOMENTUM}, lr {LR} (cosine), batch {BATCH_SIZE}, "
          f"up to {epochs} epochs, early stopping patience {patience}"
          + (", with augmentation" if args.augment else ""))
    model, hist = run(data, HIDDEN, LR, "cosine", epochs=epochs, patience=patience,
                      augment=augment_batch if args.augment else None, extra=extra, verbose=True)
    if args.val_only:
        b = hist["best_epoch"]
        print(f"\nBest epoch {b}: val acc {hist['val_acc'][b]:.4f}  (test set not used, nothing saved)")
        print(f"Total time: {time.perf_counter() - t_start:.1f}s")
        return

    # The model is finished: look at the test set exactly once.
    test_loss, test_acc = evaluate(model, data.X_test, data.y_test)
    b = hist["best_epoch"]
    print(f"\nKept epoch {b}:  val acc {hist['val_acc'][b]:.4f}   "
          f"test acc {test_acc:.4f} (loss {test_loss:.4f})")

    kind = "personal" if args.personal else ("aug" if args.augment else "")
    path = {"personal": PERSONAL_MODEL_PATH, "aug": AUG_MODEL_PATH, "": MODEL_PATH}[kind]
    curves = REPORTS / (f"mlp_{kind}_curves.png" if kind else "mlp_curves.png")
    model.save(path)
    save_curves(hist, curves,
                f"MLP 784-{HIDDEN}-10: SGD momentum {MOMENTUM}, lr {LR} cosine, batch {BATCH_SIZE}"
                + {"personal": ", augmented + personal digits", "aug": ", augmented", "": ""}[kind],
                show_lr=True, mark_best=True)
    print(f"Saved {path.relative_to(ROOT)}, {curves.relative_to(ROOT)}")
    print(f"Total time (load data + train + evaluate + save): {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

"""Mini-batch training loop, learning-rate schedules, early stopping, evaluation."""

import math
import time

import numpy as np

from digits.nn import log_softmax


def minibatches(n: int, batch_size: int, rng: np.random.Generator):
    """Yield index arrays covering 0..n-1 once, in a fresh random order (one epoch)."""
    perm = rng.permutation(n)                      # (n,)
    for start in range(0, n, batch_size):
        yield perm[start:start + batch_size]       # (<=batch_size,) last batch may be smaller


def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int = 5000) -> tuple[float, float]:
    """Mean cross-entropy loss and accuracy of model on (X, y). No gradients."""
    n = X.shape[0]
    total_loss, correct = 0.0, 0
    for s in range(0, n, batch_size):
        Xb, yb = X[s:s + batch_size], y[s:s + batch_size]   # (b,784), (b,)
        Z = model.forward(Xb)                                # (b,10)
        logP = log_softmax(Z)                                # (b,10)
        total_loss -= logP[np.arange(len(yb)), yb].sum()     # sum of -log p_i[y_i]
        correct += int((Z.argmax(axis=1) == yb).sum())       # (b,) bools; True counts as 1
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Learning-rate schedules: functions epoch (1-based) -> learning rate
# ---------------------------------------------------------------------------

def constant_schedule(lr: float):
    return lambda epoch: lr


def cosine_schedule(lr: float, epochs: int, final_frac: float = 0.0):
    """Half a cosine wave from lr (epoch 1) down to lr * final_frac (after the last epoch).

        lr_e = lr_end + (lr - lr_end) * (1 + cos(pi * (e - 1) / epochs)) / 2
    """
    lr_end = lr * final_frac
    return lambda epoch: lr_end + (lr - lr_end) * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / epochs))


def step_schedule(lr: float, every: int, gamma: float = 0.1):
    """Multiply the learning rate by gamma every `every` epochs: lr, lr, ..., lr*gamma, ..."""
    return lambda epoch: lr * gamma ** ((epoch - 1) // every)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, optimizer, X_train, y_train, X_val, y_val, *, epochs: int, batch_size: int,
          seed: int, schedule=None, patience: int | None = None, restore_best: bool = False,
          augment=None, train_eval_size: int | None = None, eval_batch_size: int = 5000,
          verbose: bool = True) -> dict:
    """Train with mini-batch gradient descent. Returns per-epoch history.

    schedule:  epoch -> learning rate, applied at the start of each epoch (None = keep optimizer.lr)
    patience:  early stopping. Stop when validation accuracy hasn't improved for this many
               epochs. None = always run all epochs.
    restore_best: at the end, put back the weights from the epoch with the best validation
               accuracy (otherwise keep the last epoch's weights).
    augment:   optional function (Xb, rng) -> Xb applied to every training batch (e.g. random
               warps). Evaluation always uses the original, unaugmented images.
    train_eval_size: report train loss/accuracy on only the first N training rows (a fixed
               subset), to save time for slow models. None = the whole training set.
    eval_batch_size: images per forward pass during evaluation (smaller = less memory).

    History lists have one entry per epoch, plus the untrained model at index 0:
        train_loss, train_acc, val_loss, val_acc, lr (lr[0] is nan)
    and also: best_epoch, stopped_epoch, seconds (training time, without evaluation).
    """
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [math.nan]}

    def record(epoch: int, seconds: float) -> None:
        n_eval = X_train.shape[0] if train_eval_size is None else train_eval_size
        tl, ta = evaluate(model, X_train[:n_eval], y_train[:n_eval], eval_batch_size)
        vl, va = evaluate(model, X_val, y_val, eval_batch_size)
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)):
            hist[k].append(v)
        if verbose:
            lr = f"  lr {hist['lr'][-1]:.4f}" if epoch else ""
            print(f"  epoch {epoch:3d}  train loss {tl:.4f} acc {ta:.4f}  "
                  f"val loss {vl:.4f} acc {va:.4f}{lr}  ({seconds:.1f}s)")

    record(0, 0.0)
    best_acc, best_epoch = hist["val_acc"][0], 0
    best_params = [p.copy() for p in model.parameters()]
    train_seconds = 0.0
    epoch = 0

    for epoch in range(1, epochs + 1):
        if schedule is not None:
            optimizer.lr = schedule(epoch)
        hist["lr"].append(optimizer.lr)
        t0 = time.perf_counter()
        for idx in minibatches(X_train.shape[0], batch_size, rng):
            Xb = X_train[idx]                                    # (b,784)
            if augment is not None:
                Xb = augment(Xb, rng)                            # (b,784) -> (b,784)
            model.loss_and_backward(Xb, y_train[idx])            # gradients now on the layers
            optimizer.step(model.layers)
        train_seconds += time.perf_counter() - t0
        record(epoch, time.perf_counter() - t0)

        if hist["val_acc"][-1] > best_acc:   # strictly better; ties keep the earlier epoch
            best_acc, best_epoch = hist["val_acc"][-1], epoch
            best_params = [p.copy() for p in model.parameters()]
        elif patience is not None and epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop: no val improvement for {patience} epochs")
            break

    if restore_best:
        for p, best in zip(model.parameters(), best_params):
            p[...] = best                    # restore the best epoch's weights in place
    hist["best_epoch"] = best_epoch
    hist["stopped_epoch"] = epoch
    hist["seconds"] = train_seconds
    if verbose and restore_best and best_epoch != epoch:
        print(f"  restored weights from epoch {best_epoch} (val acc {best_acc:.4f})")
    return hist

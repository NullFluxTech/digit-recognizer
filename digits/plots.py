"""Plot helpers. Everything is saved as PNG (no windows pop up)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only backend: render to PNG without opening a window
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

# Chart palette (validated colorblind-safe pair + recessive chrome)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TRAIN_COLOR = "#2a78d6"  # blue
VAL_COLOR = "#eb6834"    # orange
# Diverging: blue (negative) <- gray (zero) -> red (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def save_digit_grid(images: np.ndarray, titles: list[str], path: Path, cols: int = 10,
                    suptitle: str | None = None, title_size: float = 9,
                    row_height: float = 1.3) -> None:
    """Save images (K, 784) or (K, 28, 28) as a grid with one title per image."""
    k = images.shape[0]
    images = images.reshape(k, 28, 28)  # (K,784) -> (K,28,28); no-op if already (K,28,28)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * row_height), squeeze=False)
    for i, ax in enumerate(axes.flat):  # .flat walks the 2-D grid of axes as one sequence
        ax.axis("off")
        if i < k:
            # gray_r: 0 -> white, 1 -> black, so digits look like ink on paper
            ax.imshow(images[i], cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_title(titles[i], fontsize=title_size)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curves(hist: dict, path: Path, title: str, show_lr: bool = False,
                mark_best: bool = False) -> None:
    """Loss and accuracy per epoch, train vs validation, as side-by-side panels.

    hist[...][0] is the untrained model; it is left out of the plot (it would squash the
    y-axis) and reported in the subtitle instead.
    show_lr:   add a third panel with the learning rate per epoch (hist["lr"]).
    mark_best: draw a dashed line at hist["best_epoch"], the epoch whose weights were kept.
    """
    epochs = np.arange(1, len(hist["train_loss"]))
    panels = [("loss", "Cross-entropy loss"), ("acc", "Accuracy")]
    fig, axes = plt.subplots(1, 3 if show_lr else 2, figsize=(15 if show_lr else 11, 4.2),
                             facecolor=SURFACE)
    best = hist.get("best_epoch")
    # The numbers go in each panel's title and one legend sits above the panels, so no
    # legend box ever covers a curve.
    at = best if mark_best else len(epochs)
    at_text = f"epoch {best}" if mark_best else "final"
    for ax, (key, label) in zip(axes, panels):
        _style_axes(ax)
        for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
            ax.plot(epochs, hist[f"{split}_{key}"][1:], color=color, linewidth=2,
                    label="train" if split == "train" else "validation")
        if mark_best:
            ax.axvline(best, color=MUTED, linewidth=1.2, linestyle="--",
                       label=f"best validation epoch ({best}), weights kept")
        ax.set_xlabel("epoch", color=INK_2)
        ax.xaxis.get_major_locator().set_params(integer=True)  # whole epochs on the x-axis
        ax.set_title(f"{label}   {at_text}: train {hist[f'train_{key}'][at]:.4f}, "
                     f"val {hist[f'val_{key}'][at]:.4f}", color=INK, loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False,
               labelcolor=INK_2)
    if show_lr:
        ax = axes[2]
        _style_axes(ax)
        ax.plot(epochs, hist["lr"][1:], color=TRAIN_COLOR, linewidth=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("epoch", color=INK_2)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title("Learning rate", color=INK, loc="left")
    fig.suptitle(f"{title}\nepoch 0 (untrained): loss {hist['train_loss'][0]:.3f}, "
                 f"accuracy {hist['train_acc'][0]:.3f}", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def save_weight_images(W: np.ndarray, path: Path, titles: list[str], cols: int = 5,
                       suptitle: str | None = None, size: int = 28) -> None:
    """Show each column of W (size*size, K) as a size x size image, red = positive, blue = negative.

    Each image gets its own symmetric color scale (+-max|w| of that column), so gray is
    always exactly zero.
    """
    k = W.shape[1]
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False,
                             facecolor=SURFACE)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < k:
            img = W[:, i].reshape(size, size)      # (size*size,) -> (size,size), same row-major order as X
            v = np.abs(img).max() or 1.0
            ax.imshow(img, cmap=DIVERGING, vmin=-v, vmax=v)
            ax.set_title(titles[i], fontsize=10, color=INK)
    if suptitle:
        fig.suptitle(suptitle, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)

# Sequential single-hue ramp (light -> dark blue) for magnitudes such as error counts
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
NEUTRAL_CELL = "#f0efec"


def save_confusion_matrix(C: np.ndarray, path: Path, title: str) -> None:
    """Heatmap of a (K,K) confusion matrix with the count in every non-zero cell.

    Off-diagonal cells (errors) are colored on a blue scale; the diagonal (correct predictions,
    ~1000x larger) is drawn in neutral gray so it doesn't wash out the scale. Recall per true
    class is written on the right, precision per predicted class along the bottom.
    """
    k = C.shape[0]
    off = C.astype(float)
    np.fill_diagonal(off, np.nan)                       # nan cells are left for the gray overlay
    vmax = np.nanmax(off) or 1.0
    fig, ax = plt.subplots(figsize=(8.2, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    masked = np.ma.masked_where((off == 0) | np.isnan(off), off)  # zeros: plain surface
    ax.imshow(masked, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
    for i in range(k):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=NEUTRAL_CELL))
    for i in range(k):
        for j in range(k):
            if C[i, j] == 0:
                continue
            dark = i != j and C[i, j] > 0.55 * vmax       # white text only on the darkest blues
            ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=9 if i == j else 10,
                    color="white" if dark else (INK_2 if i == j else INK))
    recall = np.diag(C) / C.sum(axis=1)                   # (K,) row-wise
    precision = np.diag(C) / C.sum(axis=0)                # (K,) column-wise
    for i in range(k):
        ax.text(k - 0.3, i, f"{recall[i]:.1%}", ha="left", va="center", fontsize=9, color=INK_2)
        ax.text(i, k - 0.3, f"{precision[i]:.1%}", ha="center", va="top", fontsize=8.5,
                color=INK_2, rotation=90)
    ax.text(k - 0.3, -0.85, "recall", ha="left", va="center", fontsize=9, color=MUTED)
    ax.text(-0.85, k + 0.35, "precision", ha="right", va="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlabel("predicted digit", color=INK_2)
    ax.set_ylabel("true digit", color=INK_2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelcolor=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    fig.suptitle(title, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

def save_preprocess_steps(steps: dict, path: Path, title: str) -> None:
    """One panel per preprocessing step with its shape. Every panel shows the actual values
    (0 = black, 1 = white): the input as it was, then from step 2 on, MNIST's white-on-black."""
    labels = {"gray": "1. input (grayscale)", "inverted": "2. polarity: ink bright",
              "cleaned": "3. background 0, ink 1", "cropped": "4. crop to the ink",
              "thickened": "5. pen width ~14%", "scaled": "6. longer side = 20",
              "final": "7. 28x28, centered by mass"}
    fig, axes = plt.subplots(1, len(steps), figsize=(2.3 * len(steps), 2.9), facecolor=SURFACE)
    for ax, (key, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{labels.get(key, key)}\n{img.shape[0]}x{img.shape[1]}", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(AXIS)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3: blue, orange, aqua


def save_feature_maps(image: np.ndarray, maps1: np.ndarray, maps2: np.ndarray, path: Path,
                      title: str) -> None:
    """One input digit (28,28), its conv1 activations (c1,28,28) and conv2 activations (c2,14,14)."""
    c1, c2 = len(maps1), len(maps2)
    cols = max(c1, c2 // 2, 1)
    fig, axes = plt.subplots(4, cols, figsize=(1.35 * cols, 6.2), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(image, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("input", fontsize=9, color=INK)
    for i, m in enumerate(maps1):
        axes[1, i].imshow(m, cmap="gray_r", vmin=0, vmax=max(m.max(), 1e-6))
        axes[1, i].set_title(f"conv1 #{i}", fontsize=8, color=INK)
    for i, m in enumerate(maps2):
        ax = axes[2 + i // cols, i % cols]
        ax.imshow(m, cmap="gray_r", vmin=0, vmax=max(m.max(), 1e-6))
        ax.set_title(f"conv2 #{i}", fontsize=8, color=INK)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def save_shift_chart(shifts: list[int], results: dict[str, list[float]], path: Path, title: str) -> None:
    """Accuracy vs horizontal shift in pixels, one line per model (at most 3)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    _style_axes(ax)
    for (name, accs), color in zip(results.items(), SERIES):
        ax.plot(shifts, accs, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f"{accs[-1]:.1%}", (shifts[-1], accs[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_xticks(shifts)
    ax.set_xlabel("shift to the right (pixels)", color=INK_2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(shifts[0] - 0.2, shifts[-1] + 0.7)
    ax.set_title("Validation accuracy", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK_2, loc="lower left")
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
# digit-recognizer

A neural network that reads one handwritten digit (0–9), trained on MNIST,
written from scratch with **NumPy only**: no PyTorch, TensorFlow, JAX, Keras
or scikit-learn. This is a learning project, so the explanations in
[`LEARN.md`](LEARN.md) are as important as the code.

- [`LEARN.md`](LEARN.md): math, shapes and intuition for each milestone
- `reports/`: plots saved as PNGs

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync          # creates .venv and installs numpy, matplotlib, pillow, pytest
uv run pytest    # run the test suite
```

## Commands

| command                    | what it does                                                         |
|----------------------------|----------------------------------------------------------------------|
| `uv run prepare_data.py`   | download MNIST (checksum-verified), split 55k/5k/10k, save `reports/sample_digits.png` |
| `uv run train_softmax.py`  | train softmax regression (~1 s), test once, save `models/softmax.npz` + curves and weight images |
| `uv run train_softmax.py --sweep` | learning-rate × batch-size grid, scored on validation only |
| `uv run train_mlp.py`      | train the MLP 784→512→10 from scratch (~30 s), test once, save `models/mlp.npz` + `reports/mlp_curves.png` |
| `uv run train_mlp.py --sweep` | hidden size × learning rate, then schedule comparison, validation only (~5 min) |
| `uv run train_mlp.py --augment` | MLP trained on randomly warped digits (~90 s) → `models/mlp_aug.npz` |
| `uv run train_mlp.py --personal` | augmented MLP that also trains on my own digits in `personal/` → `models/mlp_personal.npz` |
| `uv run train_mlp.py --personal-cv` | 5-fold check that personalization helps on held-out personal digits (no test set) |
| `uv run train_cnn.py`      | small CNN with im2col (~5 min), test once, save `models/cnn.npz` + filters, feature maps, shift chart |
| `uv run train_cnn.py --shift` | MLPs vs CNN on shifted validation digits |
| `uv run train_cnn.py --personal` | CNN that also trains on my digits in `personal/` → `models/cnn_personal.npz` (the tools' default model) |
| `uv run draw.py`           | drawing window with live top-3 predictions; keys 0-9 set the label, Enter saves to `samples/` |
| `uv run predict.py img.png [--debug]` | top-3 digits for an image file; `--debug` saves every preprocessing step |
| `uv run score_samples.py`  | scores all `samples/<label>_<nn>.png`, saves `reports/samples_scored.png` |
| `uv run evaluate_mlp.py`   | confusion matrix, per-class precision/recall, confused pairs, error gallery, first-layer weights (needs `models/mlp.npz`) |

## License and credits

**Code**: Copyright (C) 2026 NullFlux. Licensed under the GNU Affero General Public License,
version 3 or (at your option) any later version (`AGPL-3.0-or-later`); full text in [`LICENSE`](LICENSE).

**My handwriting** (`personal/`, `samples/`): Copyright (C) 2026 NullFlux. Licensed under
Creative Commons Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`); full text in
[`LICENSE-DATA`](LICENSE-DATA) and at https://creativecommons.org/licenses/by-sa/4.0/.
Credit "NullFlux, digit-recognizer handwriting samples" and share adaptations under the same license.

**MNIST**: the MNIST database of handwritten digits is by Yann LeCun, Corinna Cortes and
Christopher J. C. Burges, made available under the Creative Commons Attribution-ShareAlike 3.0
license (http://yann.lecun.com/exdb/mnist/). It is downloaded by `prepare_data.py`, not stored in
this repository. MNIST digits shown in the figures in `reports/` come from that database and remain
under its license.

## Layout

```
*.py        top-level scripts you run with `uv run <script>.py`
digits/     library code (data loading, layers, models, training, preprocessing, augmentation)
tests/      pytest tests
reports/    generated plots (PNG)
personal/   my own handwritten digits used for training (<label>_<nn>.png)
samples/    my own handwritten digits used only for scoring (<label>_<nn>.png)
data/       MNIST files (downloaded, git-ignored)
models/     trained weights (git-ignored)
```

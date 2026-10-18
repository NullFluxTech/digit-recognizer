"""Classification metrics computed by hand: confusion matrix, precision/recall, confused pairs.

Shapes:
    N   number of examples
    K   number of classes (10)
    C   (K,K) confusion matrix, C[i, j] = number of examples with true class i predicted as j
"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(K,K) int64 matrix: rows = true class, columns = predicted class."""
    C = np.zeros((n_classes, n_classes), dtype=np.int64)  # (K,K)
    # Add 1 at (y_true[n], y_pred[n]) for every n. np.add.at is "unbuffered": repeated index pairs
    # each count. The tempting C[y_true, y_pred] += 1 would count each distinct pair only ONCE.
    np.add.at(C, (y_true, y_pred), 1)                     # (N,) and (N,) index pairs -> (K,K)
    return C


def per_class_metrics(C: np.ndarray):
    """Precision, recall and support per class from a confusion matrix.

    recall[k]    = C[k,k] / sum_j C[k,j]   of the true k's, fraction found       (row sums)
    precision[k] = C[k,k] / sum_i C[i,k]   of the predicted k's, fraction right  (column sums)
    support[k]   = sum_j C[k,j]            number of true k's
    A class that is never predicted gets precision nan (0/0).
    """
    tp = np.diag(C).astype(np.float64)      # (K,K) -> (K,)  correct predictions per class
    support = C.sum(axis=1)                 # (K,K) -> (K,)  sum over predicted j: true counts
    predicted = C.sum(axis=0)               # (K,K) -> (K,)  sum over true i: predicted counts
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / support               # (K,) / (K,) -> (K,)
        precision = tp / predicted          # (K,) / (K,) -> (K,)
    return precision, recall, support


def most_confused_pairs(C: np.ndarray, k: int = 10, symmetric: bool = False):
    """The k largest off-diagonal entries, biggest first.

    symmetric=False: [(count, true, predicted), ...]   e.g. "4 predicted as 9"
    symmetric=True:  [(count, a, b), ...] with a < b    counts a->b and b->a together, e.g. "4 <-> 9"
    Ties are broken by (a, b) so the order is deterministic.
    """
    off = C.copy()
    np.fill_diagonal(off, 0)                # (K,K) with zeros on the diagonal
    if symmetric:
        off = off + off.T                   # (K,K) + (K,K)^T -> (K,K), S[a,b] = C[a,b] + C[b,a]
        rows, cols = np.triu_indices_from(off, k=1)  # each unordered pair once (a < b)
    else:
        rows, cols = np.nonzero(~np.eye(len(off), dtype=bool))  # every (i, j) with i != j
    pairs = [(int(off[a, b]), int(a), int(b)) for a, b in zip(rows, cols) if off[a, b] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    return pairs[:k]

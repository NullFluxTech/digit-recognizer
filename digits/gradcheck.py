"""Gradient checking: compare backprop gradients with finite differences.

Central difference for one parameter entry w:
    dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 eps)        error O(eps^2)

Relative error per entry:
    |a - n| / max(|a|, |n|, tiny)      a = analytic (backprop), n = numerical

Run this in float64: float32 has ~7 significant digits, so L(w+eps) - L(w-eps)
would lose most of them to rounding.

Choosing eps (float64): too small and rounding in L(w+eps) - L(w-eps) dominates
(error ~ 1e-16 / eps); too large and the Taylor error dominates (~ eps^2).
Measured on softmax regression: eps=1e-7 -> 2.4e-5, 1e-6 -> 2.8e-6, 1e-5 -> 3.4e-7,
1e-4 -> 4.7e-7, 1e-3 -> 4.9e-5 max relative error. So the default is 1e-5.
"""

import numpy as np

from digits.nn import softmax_cross_entropy


def numerical_grad(loss_fn, p: np.ndarray, indices: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Finite-difference gradient of loss_fn() w.r.t. p at the given flat indices.

    p is modified in place temporarily and restored exactly.
    Returns an array (len(indices),).
    """
    flat = p.reshape(-1)  # view: writing flat[i] writes into p
    out = np.empty(len(indices), dtype=np.float64)
    for j, i in enumerate(indices):
        old = flat[i]
        flat[i] = old + eps
        l_plus = loss_fn()
        flat[i] = old - eps
        l_minus = loss_fn()
        flat[i] = old
        out[j] = (l_plus - l_minus) / (2 * eps)
    return out


def relative_error(a: np.ndarray, n: np.ndarray, tiny: float = 1e-12) -> np.ndarray:
    return np.abs(a - n) / np.maximum(np.maximum(np.abs(a), np.abs(n)), tiny)


def check_model_gradients(model, X: np.ndarray, y: np.ndarray, *, max_entries: int | None = None,
                          eps: float = 1e-5, seed: int = 0) -> dict[str, float]:
    """Max relative error between backprop and numerical gradients, per parameter.

    max_entries: check at most this many random entries per parameter (None = all).
    Returns {"layer0.W": err, "layer0.b": err, ...}.
    """
    params = [p for layer in model.layers for p, _ in layer.params_and_grads()]
    if X.dtype != np.float64 or any(p.dtype != np.float64 for p in params):
        raise ValueError("gradient check needs float64 inputs and parameters")
    rng = np.random.default_rng(seed)
    model.loss_and_backward(X, y)
    # copy: the numerical pass below only runs forward, but keep analytic grads safe anyway
    analytic = [[g.copy() for _, g in layer.params_and_grads()] for layer in model.layers]

    def loss_fn() -> float:
        return softmax_cross_entropy(model.forward(X), y)[0]

    errors = {}
    for li, layer in enumerate(model.layers):
        for (p, _), g, name in zip(layer.params_and_grads(), analytic[li], ("W", "b")):
            n_entries = p.size
            if max_entries is None or n_entries <= max_entries:
                idx = np.arange(n_entries)
            else:
                idx = rng.choice(n_entries, size=max_entries, replace=False)
            num = numerical_grad(loss_fn, p, idx, eps)
            errors[f"layer{li}.{name}"] = float(relative_error(g.reshape(-1)[idx], num).max())
    return errors

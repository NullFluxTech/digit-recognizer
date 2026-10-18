"""Tests for digits.nn / digits.models / digits.gradcheck: shapes, softmax stability, gradients."""

import numpy as np
import pytest

from digits.gradcheck import check_model_gradients, numerical_grad
from digits.models import SoftmaxRegression
from digits.nn import SGD, Linear, log_softmax, softmax, softmax_cross_entropy
from digits.train import evaluate, minibatches


# --- softmax ---------------------------------------------------------------------

def test_softmax_rows_are_probabilities():
    Z = np.random.default_rng(0).normal(size=(5, 10))
    P = softmax(Z)
    assert P.shape == (5, 10)
    assert np.all(P > 0)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_softmax_is_shift_invariant():
    Z = np.random.default_rng(1).normal(size=(3, 4))
    assert np.allclose(softmax(Z), softmax(Z + 123.0))


@pytest.mark.parametrize("scale", [1e3, 1e4, 1e300])
def test_softmax_stable_on_huge_inputs(scale):
    Z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]) * scale
    # Turn overflow, NaN-producing ops and log(0) into exceptions. Underflow (exp(-2000) -> 0)
    # is expected and harmless, so it stays allowed.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        P = softmax(Z)
        logP = log_softmax(Z)
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(logP))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 2] == pytest.approx(1.0) and P[1, 2] == pytest.approx(1.0)


def test_cross_entropy_stable_when_true_class_is_hopeless():
    # true class logit is 1e4 below the others: p ~ exp(-1e4) underflows to 0, but log p is fine
    Z = np.array([[1e4, 0.0]])
    loss, dZ = softmax_cross_entropy(Z, np.array([1]))
    assert loss == pytest.approx(1e4)
    assert np.allclose(dZ, [[1.0, -1.0]])


def test_cross_entropy_of_uniform_logits_is_log_k():
    Z = np.zeros((7, 10))
    loss, dZ = softmax_cross_entropy(Z, np.arange(7))
    assert loss == pytest.approx(np.log(10))
    assert dZ.shape == (7, 10)
    assert np.allclose(dZ.sum(axis=1), 0.0)  # each row of P - Y sums to 1 - 1 = 0


def test_cross_entropy_grad_is_p_minus_y_over_b():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(4, 3))
    y = np.array([0, 2, 1, 2])
    _, dZ = softmax_cross_entropy(Z, y)
    Y = np.eye(3)[y]  # one-hot: row i of the identity matrix is the one-hot vector for class i
    assert np.allclose(dZ, (softmax(Z) - Y) / 4)


# --- Linear layer -----------------------------------------------------------------------

def test_linear_shapes():
    rng = np.random.default_rng(3)
    layer = Linear(rng.normal(size=(784, 10)), np.zeros(10))
    X = rng.normal(size=(32, 784))
    Z = layer.forward(X)
    assert Z.shape == (32, 10)
    dX = layer.backward(np.ones((32, 10)))
    assert dX.shape == (32, 784)
    assert layer.dW.shape == (784, 10) and layer.db.shape == (10,)


def test_linear_bias_broadcasts_to_every_row():
    layer = Linear(np.zeros((2, 3)), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(layer.forward(np.ones((4, 2))), np.tile([1.0, 2.0, 3.0], (4, 1)))


def test_linear_input_gradient_numerically():
    # dL/dX is not a parameter, so check_model_gradients doesn't cover it; check it here.
    rng = np.random.default_rng(4)
    layer = Linear(rng.normal(size=(5, 3)), rng.normal(size=3))
    X = rng.normal(size=(4, 5))
    y = np.array([0, 1, 2, 1])

    def loss_fn():
        return softmax_cross_entropy(layer.forward(X), y)[0]

    _, dZ = softmax_cross_entropy(layer.forward(X), y)
    dX = layer.backward(dZ)
    num = numerical_grad(loss_fn, X, np.arange(X.size))
    assert np.allclose(dX.reshape(-1), num, rtol=1e-6, atol=1e-9)


# --- gradient check of the whole model (float64) ---------------------------------------------

def test_softmax_regression_gradient_check():
    rng = np.random.default_rng(5)
    model = SoftmaxRegression(dtype=np.float64)
    model.fc.W[...] = rng.normal(scale=0.1, size=model.fc.W.shape)   # non-zero point to check at
    model.fc.b[...] = rng.normal(scale=0.1, size=10)
    X = rng.random((8, 784))                                        # (8,784) float64
    y = rng.integers(0, 10, size=8)
    errors = check_model_gradients(model, X, y)                      # every one of the 7850 entries
    assert set(errors) == {"layer0.W", "layer0.b"}
    for name, err in errors.items():
        assert err < 1e-5, f"{name}: max relative error {err:.2e}"


# --- model + training pieces ---------------------------------------------------------------

def test_model_shapes_and_save_load(tmp_path):
    model = SoftmaxRegression()
    model.fc.W[...] = np.random.default_rng(6).normal(size=(784, 10))
    X = np.random.default_rng(7).random((3, 784), dtype=np.float32)
    assert model.forward(X).shape == (3, 10)
    assert model.predict(X).shape == (3,)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = SoftmaxRegression.load(path)
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_minibatches_cover_every_index_once():
    batches = list(minibatches(10, 4, np.random.default_rng(0)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))


def test_sgd_reduces_loss_on_a_small_problem():
    rng = np.random.default_rng(8)
    X = rng.random((64, 784), dtype=np.float32)
    y = rng.integers(0, 10, size=64)
    model = SoftmaxRegression()
    # Random inputs in [0,1] have |x|^2 ~ 260 (vs ~90 for MNIST); lr=0.5 would diverge here
    opt = SGD(lr=0.03)
    first = model.loss_and_backward(X, y)
    for _ in range(50):
        opt.step(model.layers)
        last = model.loss_and_backward(X, y)
    assert first == pytest.approx(np.log(10), rel=1e-5)  # zero weights -> uniform probabilities
    assert last < 0.6 * first                            # observed ~0.5 * first


def test_evaluate_counts_correctly():
    model = SoftmaxRegression(n_in=2, n_classes=2, dtype=np.float64)
    model.fc.W[...] = np.eye(2)            # logits = X, so prediction = argmax of the row
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1, 1, 1])             # 3 of 4 right
    _, acc = evaluate(model, X, y, batch_size=3)  # batch_size 3 also tests the ragged last batch
    assert acc == 0.75

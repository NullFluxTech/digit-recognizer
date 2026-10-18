"""Tests for milestone 3: ReLU, He init, momentum, schedules, early stopping, MLP + gradient check."""

import math

import numpy as np
import pytest

from digits.gradcheck import check_model_gradients
from digits.models import MLP, SoftmaxRegression
from digits.nn import SGDMomentum, Linear, ReLU, he_init
from digits.train import constant_schedule, cosine_schedule, step_schedule, train


# --- ReLU ------------------------------------------------------------------------------

def test_relu_forward_backward():
    relu = ReLU()
    H = np.array([[-2.0, 0.0, 3.0], [1.5, -0.1, 0.2]])
    assert np.array_equal(relu.forward(H), [[0.0, 0.0, 3.0], [1.5, 0.0, 0.2]])
    G = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    # gradient passes only where the input was > 0 (at exactly 0 we use 0)
    assert np.array_equal(relu.backward(G), [[0.0, 0.0, 30.0], [40.0, 0.0, 60.0]])
    assert relu.params_and_grads() == []


# --- He init ---------------------------------------------------------------------------------

def test_he_init_statistics():
    W, b = he_init(784, 512, np.random.default_rng(0))
    assert W.shape == (784, 512) and b.shape == (512,)
    assert W.dtype == np.float32
    assert np.all(b == 0)
    assert abs(W.mean()) < 1e-3
    assert W.std() == pytest.approx(math.sqrt(2 / 784), rel=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_he_init_keeps_activation_scale_through_relu_layers(seed):
    # The point of He init: E[a^2] stays about the same from layer to layer, *on average*.
    # Each finite-width layer multiplies it by a random factor near 1, so after 10 layers of
    # width 256 it drifts (measured over 40 seeds: 0.4x to 2.2x). With std sqrt(1/n) instead of
    # sqrt(2/n) every layer halves it: 2^-10 = 0.001 every time.
    def ratio_after_10_layers(scale):
        rng = np.random.default_rng(seed)
        A = np.abs(rng.standard_normal((1000, 256)))  # (1000,256) positive inputs, like post-ReLU
        start = (A ** 2).mean()
        for _ in range(10):
            W, b = he_init(256, 256, rng, dtype=np.float64)
            A = np.maximum(0, A @ (W * scale) + b)    # (1000,256) @ (256,256) -> (1000,256)
        return (A ** 2).mean() / start

    assert 0.25 < ratio_after_10_layers(1.0) < 4.0
    assert ratio_after_10_layers(math.sqrt(0.5)) < 0.01


# --- momentum ------------------------------------------------------------------------------

def test_momentum_matches_hand_computation():
    layer = Linear(np.array([[1.0]]), np.array([0.0]))
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    layer.dW = np.array([[2.0]])
    opt.step([layer])            # v = -0.1*2 = -0.2;               W = 1 - 0.2 = 0.8
    assert layer.W[0, 0] == pytest.approx(0.8)
    opt.step([layer])            # v = 0.9*(-0.2) - 0.2 = -0.38;    W = 0.8 - 0.38 = 0.42
    assert layer.W[0, 0] == pytest.approx(0.42)
    layer.dW = np.array([[0.0]])
    opt.step([layer])            # no gradient, still moving: v = -0.342; W = 0.078
    assert layer.W[0, 0] == pytest.approx(0.078)


# --- schedules ------------------------------------------------------------------------------

def test_schedules():
    assert constant_schedule(0.3)(7) == 0.3
    cos = cosine_schedule(0.1, epochs=10)
    assert cos(1) == pytest.approx(0.1)
    assert cos(6) == pytest.approx(0.05)                   # halfway
    assert cos(10) == pytest.approx(0.1 * (1 + math.cos(math.pi * 0.9)) / 2)
    assert all(cos(e) > cos(e + 1) for e in range(1, 10))  # strictly decreasing
    step = step_schedule(1.0, every=3, gamma=0.5)
    assert [step(e) for e in range(1, 8)] == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25]


# --- MLP ------------------------------------------------------------------------------------

def test_mlp_shapes():
    model = MLP(n_hidden=32)
    X = np.random.default_rng(2).random((5, 784), dtype=np.float32)
    assert model.forward(X).shape == (5, 10)
    assert model.predict(X).shape == (5,)
    model.loss_and_backward(X, np.arange(5))
    assert model.fc1.dW.shape == (784, 32) and model.fc1.db.shape == (32,)
    assert model.fc2.dW.shape == (32, 10) and model.fc2.db.shape == (10,)
    assert [p.shape for p in model.parameters()] == [(784, 32), (32,), (32, 10), (10,)]


def test_mlp_init_is_seeded():
    a, b, c = MLP(n_hidden=8, seed=0), MLP(n_hidden=8, seed=0), MLP(n_hidden=8, seed=1)
    assert np.array_equal(a.fc1.W, b.fc1.W)
    assert not np.array_equal(a.fc1.W, c.fc1.W)


def test_mlp_gradient_check():
    rng = np.random.default_rng(3)
    model = MLP(n_hidden=16, seed=3, dtype=np.float64)
    model.fc1.b[...] = rng.normal(scale=0.1, size=16)   # non-zero biases, so their gradients are tested too
    model.fc2.b[...] = rng.normal(scale=0.1, size=10)
    X = rng.random((8, 784))                            # (8,784) float64
    y = rng.integers(0, 10, size=8)
    # ReLU has a kink at 0. One nudge changes a pre-activation by at most eps * max(|x|, 1) = 1e-5,
    # so the numerical gradient is only wrong if some |h| < 1e-5. Require a 10x margin.
    h = X @ model.fc1.W + model.fc1.b                   # (8,784) @ (784,16) -> (8,16) pre-activations
    assert np.abs(h).min() > 1e-4, "a pre-activation sits on ReLU's kink; pick another seed"
    errors = check_model_gradients(model, X, y)         # all 12,730 parameters
    assert set(errors) == {"layer0.W", "layer0.b", "layer2.W", "layer2.b"}
    for name, err in errors.items():
        assert err < 1e-5, f"{name}: max relative error {err:.2e}"


def test_gradcheck_rejects_float32():
    model = MLP(n_hidden=4)
    with pytest.raises(ValueError, match="float64"):
        check_model_gradients(model, np.zeros((2, 784), np.float32), np.array([0, 1]))


@pytest.mark.parametrize("model", [MLP(n_hidden=12, seed=5), SoftmaxRegression()])
def test_save_load_roundtrip(model, tmp_path):
    for p in model.parameters():
        p[...] = np.random.default_rng(6).normal(size=p.shape)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.config() == model.config()
    for a, b in zip(loaded.parameters(), model.parameters(), strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)


# --- training loop: schedule + early stopping --------------------------------------------------

def _tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((200, 784), dtype=np.float32)
    y = (X[:, :392].sum(axis=1) > X[:, 392:].sum(axis=1)).astype(np.int64)  # learnable rule
    return X[:150], y[:150], X[150:], y[150:]


def test_train_applies_schedule_and_records_lr():
    Xtr, ytr, Xva, yva = _tiny_problem()
    opt = SGDMomentum(lr=999.0)
    hist = train(MLP(n_hidden=8), opt, Xtr, ytr, Xva, yva, epochs=3, batch_size=50, seed=0,
                 schedule=step_schedule(0.01, every=1, gamma=0.5), verbose=False)
    assert math.isnan(hist["lr"][0])
    assert hist["lr"][1:] == [0.01, 0.005, 0.0025]
    assert len(hist["val_acc"]) == 4


def test_early_stopping_stops_and_restores_best_weights():
    Xtr, ytr, Xva, yva = _tiny_problem()
    model = MLP(n_hidden=8)
    # lr = 0: weights never change, val acc never improves -> stop after `patience` epochs
    hist = train(model, SGDMomentum(lr=0.0), Xtr, ytr, Xva, yva, epochs=50, batch_size=50,
                 seed=0, patience=3, restore_best=True, verbose=False)
    assert hist["best_epoch"] == 0 and hist["stopped_epoch"] == 3


def test_restore_best_puts_back_the_best_epoch(monkeypatch):
    Xtr, ytr, Xva, yva = _tiny_problem()
    model = MLP(n_hidden=8)
    snapshots = []
    real_step = SGDMomentum.step

    def step_and_snapshot(self, layers):
        real_step(self, layers)
        snapshots.append([p.copy() for p in model.parameters()])

    monkeypatch.setattr(SGDMomentum, "step", step_and_snapshot)
    hist = train(model, SGDMomentum(lr=0.05), Xtr, ytr, Xva, yva, epochs=6, batch_size=50,
                 seed=0, restore_best=True, verbose=False)
    best = hist["best_epoch"]
    assert best >= 1
    steps_per_epoch = 3                                   # 150 rows / batch 50
    expected = snapshots[best * steps_per_epoch - 1]      # weights right after the best epoch
    for p, e in zip(model.parameters(), expected):
        assert np.array_equal(p, e)

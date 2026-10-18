"""Tests for digits.conv and the CNN: im2col vs loops, col2im as transpose, pooling, gradient check."""

import numpy as np
import pytest

from digits.conv import Conv2D, MaxPool2, Reshape, col2im, im2col
from digits.gradcheck import check_model_gradients, numerical_grad
from digits.models import CNN


def naive_conv(X, Wf, b):
    """Plain nested loops: X (B,C,H,W), Wf (F,C,k,k) -> (B,F,H,W), zero padding, stride 1."""
    B, C, H, W = X.shape
    F, _, k, _ = Wf.shape
    p = (k - 1) // 2
    Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))
    out = np.zeros((B, F, H, W))
    for n in range(B):
        for f in range(F):
            for i in range(H):
                for j in range(W):
                    out[n, f, i, j] = np.sum(Xp[n, :, i:i + k, j:j + k] * Wf[f]) + b[f]
    return out


def conv_layer(rng, C, F, k):
    Wf = rng.normal(size=(F, C, k, k))                    # filters in the natural (F,C,k,k) layout
    b = rng.normal(size=F)
    W = Wf.reshape(F, -1).T.copy()                        # (C*k*k, F): same unrolling as im2col rows
    return Conv2D(W, b, k), Wf, b


@pytest.mark.parametrize("k", [3, 5])
def test_conv_forward_matches_loops(k):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 3, 7, 6))
    layer, Wf, b = conv_layer(rng, C=3, F=4, k=k)
    Z = layer.forward(X)
    assert Z.shape == (2, 4, 7, 6)
    assert np.allclose(Z, naive_conv(X, Wf, b))


def test_im2col_rows_are_patches():
    X = np.arange(2 * 1 * 4 * 4, dtype=float).reshape(2, 1, 4, 4)
    cols = im2col(X, 3)
    assert cols.shape == (2 * 4 * 4, 9)
    # row for image 1, pixel (2, 1): the 3x3 patch around it
    row = cols[1 * 16 + 2 * 4 + 1]
    assert np.array_equal(row.reshape(3, 3), X[1, 0, 1:4, 0:3])
    # corner pixel (0, 0) of image 0: top row and left column are zero padding
    assert np.array_equal(cols[0].reshape(3, 3), [[0, 0, 0], [0, 0, 1], [0, 4, 5]])


def test_col2im_is_the_transpose_of_im2col():
    # For a linear map A and its transpose A^T:  <A x, y> = <x, A^T y>  for all x, y.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, 5, 6))
    Y = rng.normal(size=(2 * 5 * 6, 3 * 3 * 3))
    lhs = np.sum(im2col(X, 3) * Y)
    rhs = np.sum(X * col2im(Y, X.shape, 3))
    assert lhs == pytest.approx(rhs)


def test_conv_backward_numerically():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2, 2, 5, 5))
    layer, _, _ = conv_layer(rng, C=2, F=3, k=3)
    G = rng.normal(size=(2, 3, 5, 5))                     # pretend upstream gradient dL/dZ

    def loss():                                           # L = sum(G * Z): then dL/dZ = G
        return float(np.sum(G * layer.forward(X)))

    layer.forward(X)
    dX = layer.backward(G)
    dW, db = layer.dW.copy(), layer.db.copy()
    assert np.allclose(dX.ravel(), numerical_grad(loss, X, np.arange(X.size)), atol=1e-6)
    assert np.allclose(dW.ravel(), numerical_grad(loss, layer.W, np.arange(layer.W.size)), atol=1e-6)
    assert np.allclose(db, numerical_grad(loss, layer.b, np.arange(3)), atol=1e-6)


def test_maxpool_forward_and_gradient_routing():
    X = np.array([[[[1., 5., 2., 0.],
                    [3., 4., 8., 7.],
                    [0., 0., 6., 6.],
                    [0., 0., 6., 1.]]]])                  # (1,1,4,4)
    pool = MaxPool2()
    out = pool.forward(X)
    assert out.shape == (1, 1, 2, 2)
    assert out[0, 0].tolist() == [[5., 8.], [0., 6.]]
    dX = pool.backward(np.array([[[[10., 20.], [30., 40.]]]]))
    expected = np.zeros((4, 4))
    expected[0, 1] = 10                                   # max of the top-left window was at (0,1)
    expected[1, 2] = 20
    expected[2, 0] = 30                                   # all-zero window: the first entry wins the tie
    expected[2, 2] = 40                                   # tie between three 6s: the first wins
    assert np.array_equal(dX[0, 0], expected)


def test_reshape_round_trip():
    r = Reshape((1, 28, 28))
    X = np.arange(2 * 784, dtype=float).reshape(2, 784)
    assert r.forward(X).shape == (2, 1, 28, 28)
    assert np.array_equal(r.backward(r.forward(X)), X)


def test_cnn_shapes_and_parameter_count():
    model = CNN()
    X = np.random.default_rng(3).random((4, 784), dtype=np.float32)
    assert model.forward(X).shape == (4, 10)
    assert sum(p.size for p in model.parameters()) == 103_146
    model.loss_and_backward(X, np.arange(4))
    for layer in model.layers:
        for p, g in layer.params_and_grads():
            assert g.shape == p.shape


def test_cnn_gradient_check():
    rng = np.random.default_rng(4)
    model = CNN(c1=2, c2=3, hidden=8, seed=4, dtype=np.float64)       # small: check every parameter
    for layer in (model.conv1, model.conv2, model.fc1, model.fc2):
        layer.b[...] = rng.normal(scale=0.1, size=layer.b.shape)       # non-zero biases get tested too
    X = rng.random((3, 784))
    y = np.array([0, 4, 7])
    errors = check_model_gradients(model, X, y)
    assert set(errors) == {"layer1.W", "layer1.b", "layer4.W", "layer4.b",
                           "layer8.W", "layer8.b", "layer10.W", "layer10.b"}
    for name, err in errors.items():
        assert err < 1e-5, f"{name}: max relative error {err:.2e}"


def test_load_any_picks_the_right_class(tmp_path):
    from digits.models import MLP, load_any

    CNN(c1=2, c2=2, hidden=4).save(tmp_path / "c.npz")
    MLP(n_hidden=4).save(tmp_path / "m.npz")
    assert type(load_any(tmp_path / "c.npz")) is CNN
    assert type(load_any(tmp_path / "m.npz")) is MLP
    # files from milestones 2-5 have no "kind" entry and hold MLPs
    with np.load(tmp_path / "m.npz") as f:
        legacy = {k: f[k] for k in f.files if k != "kind"}
    np.savez(tmp_path / "old.npz", **legacy)
    assert type(load_any(tmp_path / "old.npz")) is MLP


def test_cnn_save_load(tmp_path):
    model = CNN(c1=3, c2=4, hidden=16, seed=5)
    path = tmp_path / "cnn.npz"
    model.save(path)
    loaded = CNN.load(path)
    X = np.random.default_rng(6).random((2, 784), dtype=np.float32)
    assert loaded.config() == model.config()
    assert np.allclose(loaded.forward(X), model.forward(X))

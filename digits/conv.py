"""Convolution layers via im2col: every convolution becomes one matrix multiply.

Shapes (stride 1, "same" padding, so the output has the input's height and width):
    B   batch size          C   input channels       F   output channels (filters)
    H,W image height/width  k   kernel size (odd)    p = (k-1)/2 padding
    X     (B,C,H,W)          input
    cols  (B*H*W, C*k*k)     one row per output position: the k x k patch around it, all channels
    W     (C*k*k, F)         one column per filter (filter "unrolled" the same way as a patch)
    Z     (B,F,H,W)          output: Z = cols @ W + b, reshaped
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col(X: np.ndarray, k: int) -> np.ndarray:
    """(B,C,H,W) -> (B*H*W, C*k*k): the zero-padded k x k patch around every pixel, as rows."""
    B, C, H, W = X.shape
    p = (k - 1) // 2
    Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))           # (B,C,H+2p,W+2p)
    # A strided VIEW of every k x k window (no copy yet): win[b,c,i,j] is the patch whose
    # top-left corner is Xp[b,c,i,j], i.e. the patch centered on X[b,c,i,j].
    win = sliding_window_view(Xp, (k, k), axis=(2, 3))         # (B,C,H,W,k,k)
    # Put the position axes (B,H,W) first and the patch axes (C,k,k) last, then flatten both
    # groups. Overlapping windows share memory in the view, so this reshape must copy.
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(B * H * W, C * k * k)  # (B*H*W, C*k*k)


def col2im(dcols: np.ndarray, shape: tuple, k: int) -> np.ndarray:
    """The transpose of im2col: (B*H*W, C*k*k) -> (B,C,H,W), adding up overlapping patches.

    Every input pixel appears in up to k*k patches (once per kernel offset), so its gradient is
    the sum of the k*k entries that copied it. For one fixed offset (i, j) the patches don't
    overlap, so the k*k shifted slice-adds below are safe with plain +=.
    """
    B, C, H, W = shape
    p = (k - 1) // 2
    d = dcols.reshape(B, H, W, C, k, k).transpose(0, 3, 1, 2, 4, 5)  # (B,C,H,W,k,k)
    dXp = np.zeros((B, C, H + 2 * p, W + 2 * p), dtype=dcols.dtype)   # (B,C,H+2p,W+2p)
    for i in range(k):
        for j in range(k):
            dXp[:, :, i:i + H, j:j + W] += d[:, :, :, :, i, j]       # (B,C,H,W) shifted copy
    return dXp[:, :, p:p + H, p:p + W]                               # (B,C,H,W) drop the padding


class Conv2D:
    """k x k convolution (cross-correlation, like every deep-learning library), stride 1, same size."""

    def __init__(self, W: np.ndarray, b: np.ndarray, k: int):
        self.W = W                       # (C*k*k, F)
        self.b = b                       # (F,)
        self.k = k
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(b)
        self.cols = None                 # cached im2col of the input
        self.shape = None                # cached input shape (B,C,H,W)

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, C, H, W = X.shape
        self.shape = X.shape
        self.cols = im2col(X, self.k)                    # (B*H*W, C*k*k)
        Z = self.cols @ self.W + self.b                  # (B*H*W, C*k*k) @ (C*k*k, F) -> (B*H*W, F), + (F,)
        return Z.reshape(B, H, W, -1).transpose(0, 3, 1, 2)  # (B,H,W,F) -> (B,F,H,W)

    def backward(self, G: np.ndarray) -> np.ndarray:
        F = G.shape[1]
        Gm = G.transpose(0, 2, 3, 1).reshape(-1, F)     # (B,F,H,W) -> (B*H*W, F), same row order as cols
        self.dW = self.cols.T @ Gm                       # (C*k*k, B*H*W) @ (B*H*W, F) -> (C*k*k, F)
        self.db = Gm.sum(axis=0)                         # (B*H*W, F) -> (F,)
        dcols = Gm @ self.W.T                            # (B*H*W, F) @ (F, C*k*k) -> (B*H*W, C*k*k)
        return col2im(dcols, self.shape, self.k)         # (B,C,H,W)

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class MaxPool2:
    """2 x 2 max pooling, stride 2: (B,C,H,W) -> (B,C,H/2,W/2). H and W must be even."""

    def __init__(self):
        self.idx = None                  # which of the 4 window entries was the max
        self.shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, C, H, W = X.shape
        self.shape = X.shape
        # (B,C,H,W) -> (B,C,H/2,2,W/2,2): axes 3 and 5 index the position inside each 2x2 window
        win = X.reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H // 2, W // 2, 4)
        self.idx = win.argmax(axis=-1)                   # (B,C,H/2,W/2) first max wins on ties
        return np.take_along_axis(win, self.idx[..., None], axis=-1)[..., 0]  # (B,C,H/2,W/2)

    def backward(self, G: np.ndarray) -> np.ndarray:
        B, C, H, W = self.shape
        dwin = np.zeros((B, C, H // 2, W // 2, 4), dtype=G.dtype)
        np.put_along_axis(dwin, self.idx[..., None], G[..., None], axis=-1)  # gradient only to the max
        return dwin.reshape(B, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    def params_and_grads(self):
        return []


class Reshape:
    """(B, ...) -> (B, *shape); backward undoes it. Used for (B,784) <-> (B,1,28,28) and flatten."""

    def __init__(self, shape: tuple):
        self.shape = shape
        self.in_shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.in_shape = X.shape
        return X.reshape(X.shape[0], *self.shape)        # a view, no copy

    def backward(self, G: np.ndarray) -> np.ndarray:
        return G.reshape(self.in_shape)

    def params_and_grads(self):
        return []

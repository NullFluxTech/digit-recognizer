"""Neural-network building blocks, written out by hand.

Shapes:
    B   batch size (rows = examples)
    D   input features   (784 for raw MNIST)
    K   output size      (10 classes for the last layer)

Every layer caches what it needs in forward() and uses it in backward().
backward(G) takes G = dL/d(output) and returns dL/d(input), storing the
parameter gradients (dW, db) on the layer.
"""

import numpy as np


class Linear:
    """Affine layer: Z = X @ W + b."""

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W                      # (D,K)
        self.b = b                      # (K,)
        self.dW = np.zeros_like(W)      # (D,K)
        self.db = np.zeros_like(b)      # (K,)
        self.X = None                   # cached input, set by forward()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        # (B,D) @ (D,K) -> (B,K); + (K,) is broadcast to every row -> (B,K)
        return X @ self.W + self.b

    def backward(self, G: np.ndarray) -> np.ndarray:
        # G = dL/dZ: (B,K)
        self.dW = self.X.T @ G          # (D,B) @ (B,K) -> (D,K)   sums over the batch
        self.db = G.sum(axis=0)         # (B,K) -> (K,)            sums over the batch
        return G @ self.W.T             # (B,K) @ (K,D) -> (B,D)   dL/dX

    def params_and_grads(self):
        """[(parameter, gradient), ...] -- read fresh after every backward()."""
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Elementwise max(0, h). No parameters."""

    def __init__(self):
        self.mask = None                # cached (B,H) bools: where the input was > 0

    def forward(self, Hpre: np.ndarray) -> np.ndarray:
        self.mask = Hpre > 0            # (B,H) bool
        return Hpre * self.mask         # (B,H) * (B,H) -> (B,H)  elementwise; False acts as 0

    def backward(self, G: np.ndarray) -> np.ndarray:
        # d relu(h)/dh = 1 where h > 0, else 0 -> pass the gradient through only where active
        return G * self.mask            # (B,H) * (B,H) -> (B,H)  elementwise, NOT a matrix product

    def params_and_grads(self):
        return []


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32):
    """W ~ Normal(0, 2 / fan_in), b = 0. Keeps activation variance steady through ReLU layers."""
    std = np.sqrt(2.0 / fan_in)
    W = rng.standard_normal((fan_in, fan_out)) * std  # (fan_in,fan_out), each entry N(0, std^2)
    return W.astype(dtype), np.zeros(fan_out, dtype=dtype)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stable for huge or tiny logits."""
    # Subtracting each row's max doesn't change the result (it cancels in the ratio)
    # but makes the largest exponent exp(0) = 1, so exp() can never overflow.
    Zs = Z - Z.max(axis=1, keepdims=True)   # (B,K) - (B,1) -> (B,K)
    E = np.exp(Zs)                          # (B,K), every entry in (0, 1]
    return E / E.sum(axis=1, keepdims=True) # (B,K) / (B,1) -> (B,K), rows sum to 1


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """log(softmax(Z)) without ever computing log(0)."""
    Zs = Z - Z.max(axis=1, keepdims=True)                  # (B,K) - (B,1) -> (B,K)
    lse = np.log(np.exp(Zs).sum(axis=1, keepdims=True))    # (B,K) -> (B,1)  log-sum-exp, >= 0
    return Zs - lse                                        # (B,K) - (B,1) -> (B,K)


def softmax_cross_entropy(Z: np.ndarray, y: np.ndarray):
    """Mean cross-entropy loss of logits Z against integer labels y, and dL/dZ.

    Z: (B,K) logits, y: (B,) ints in 0..K-1.
    Returns (loss: float, dZ: (B,K)) where dZ = (P - Y) / B, Y = one-hot(y).
    """
    B = Z.shape[0]
    rows = np.arange(B)                         # (B,) 0, 1, ..., B-1
    logP = log_softmax(Z)                       # (B,K)
    # logP[rows, y] pairs row i with column y[i]: picks log p_i[y_i] for each example
    loss = -logP[rows, y].mean()                # (B,) -> scalar
    dZ = np.exp(logP)                           # (B,K)  P = softmax(Z)
    dZ[rows, y] -= 1.0                          # (B,K)  P - Y, without building Y
    dZ /= B                                     # (B,K)  the loss is a mean over B examples
    return float(loss), dZ


class SGD:
    """Plain stochastic gradient descent: p <- p - lr * grad."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, layers) -> None:
        for layer in layers:
            for p, g in layer.params_and_grads():
                p -= self.lr * g                # in place: same shape as p


class SGDMomentum:
    """SGD with momentum ("heavy ball"):

        v <- mu * v - lr * grad        velocity: a decaying running sum of past steps
        p <- p + v

    One velocity array per parameter, created on first use.
    """

    def __init__(self, lr: float, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # id(parameter array) -> velocity array, like a map keyed by pointer

    def step(self, layers) -> None:
        for layer in layers:
            for p, g in layer.params_and_grads():
                v = self.velocity.get(id(p))
                if v is None:
                    v = self.velocity[id(p)] = np.zeros_like(p)
                v *= self.momentum              # same shape as p, in place
                v -= self.lr * g                # same shape as p, in place
                p += v                          # same shape as p, in place

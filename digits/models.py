"""Models built from the layers in digits.nn.

A model has:
    layers                  list of layers with forward / backward / params_and_grads
    forward(X)              (B,784) -> logits (B,10)
    loss_and_backward(X,y)  forward + loss + backward; leaves gradients on the layers
    predict(X)              (B,784) -> (B,) predicted digits
"""

import json
from pathlib import Path

import numpy as np

from digits.conv import Conv2D, MaxPool2, Reshape
from digits.nn import Linear, ReLU, he_init, softmax_cross_entropy


class Sequential:
    """Layers applied in order; backprop walks them in reverse (the chain rule)."""

    layers: list

    def config(self) -> dict:
        """Constructor arguments, saved next to the weights so load() can rebuild the model."""
        raise NotImplementedError

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)                       # (B,in) -> (B,out) of that layer
        return X                                       # (B,10) logits

    def loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> float:
        Z = self.forward(X)                            # (B,10)
        loss, G = softmax_cross_entropy(Z, y)          # scalar, (B,10) = dL/dZ
        for layer in reversed(self.layers):
            G = layer.backward(G)                      # dL/d(output) -> dL/d(input) of each layer
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)          # (B,10) -> (B,) index of the largest logit

    def parameters(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p, _ in layer.params_and_grads()]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"p{i}": p for i, p in enumerate(self.parameters())}
        np.savez(path, kind=type(self).__name__, config=json.dumps(self.config()), **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as f:
            n_params = sum(1 for name in f.files if name.startswith("p"))
            params = [f[f"p{i}"] for i in range(n_params)]
            model = cls(**json.loads(str(f["config"])), dtype=params[0].dtype)
            for p, saved in zip(model.parameters(), params, strict=True):
                p[...] = saved                         # [...] writes into the existing array
        return model


class SoftmaxRegression(Sequential):
    """784 -> 10, no hidden layer: logits = X @ W + b, probabilities = softmax(logits)."""

    def __init__(self, n_in: int = 784, n_classes: int = 10, dtype=np.float32):
        self.n_in, self.n_classes = n_in, n_classes
        # Zero init is fine here: the loss is convex in (W, b), and every class gets a
        # different gradient from the start because P - Y differs per column.
        self.fc = Linear(np.zeros((n_in, n_classes), dtype=dtype),  # (784,10)
                         np.zeros(n_classes, dtype=dtype))          # (10,)
        self.layers = [self.fc]

    def config(self) -> dict:
        return {"n_in": self.n_in, "n_classes": self.n_classes}


class MLP(Sequential):
    """784 -> H (ReLU) -> 10.

        A1 = relu(X @ W1 + b1)      (B,784) @ (784,H) -> (B,H)
        Z  = A1 @ W2 + b2           (B,H) @ (H,10)    -> (B,10)
    """

    def __init__(self, n_in: int = 784, n_hidden: int = 256, n_classes: int = 10,
                 seed: int = 0, dtype=np.float32):
        self.n_in, self.n_hidden, self.n_classes, self.seed = n_in, n_hidden, n_classes, seed
        rng = np.random.default_rng(seed)
        # Random init is required here (unlike softmax regression): if all hidden units started
        # equal they would get equal gradients and stay copies of each other forever.
        self.fc1 = Linear(*he_init(n_in, n_hidden, rng, dtype))       # W1 (784,H), b1 (H,)
        self.relu = ReLU()
        self.fc2 = Linear(*he_init(n_hidden, n_classes, rng, dtype))  # W2 (H,10),  b2 (10,)
        self.layers = [self.fc1, self.relu, self.fc2]

    def config(self) -> dict:
        return {"n_in": self.n_in, "n_hidden": self.n_hidden, "n_classes": self.n_classes,
                "seed": self.seed}


class CNN(Sequential):
    """conv5x5(c1) -> ReLU -> pool -> conv3x3(c2) -> ReLU -> pool -> fc(hidden) -> ReLU -> fc(10).

        X       (B,784) -> (B,1,28,28)
        conv1   (B,1,28,28)  -> (B,c1,28,28)     pool -> (B,c1,14,14)
        conv2   (B,c1,14,14) -> (B,c2,14,14)     pool -> (B,c2,7,7) -> flatten (B, c2*49)
        fc1     (B, c2*49) @ (c2*49, hidden) -> (B,hidden)
        fc2     (B,hidden) @ (hidden,10)     -> (B,10)
    """

    def __init__(self, c1: int = 8, c2: int = 16, hidden: int = 128, seed: int = 0, dtype=np.float32):
        self.c1, self.c2, self.hidden, self.seed = c1, c2, hidden, seed
        rng = np.random.default_rng(seed)
        # He init with fan_in = inputs seen by one output value: channels * kernel area
        self.conv1 = Conv2D(*he_init(1 * 5 * 5, c1, rng, dtype), k=5)    # W (25,c1)
        self.conv2 = Conv2D(*he_init(c1 * 3 * 3, c2, rng, dtype), k=3)   # W (9*c1,c2)
        self.fc1 = Linear(*he_init(c2 * 7 * 7, hidden, rng, dtype))      # W (49*c2,hidden)
        self.fc2 = Linear(*he_init(hidden, 10, rng, dtype))              # W (hidden,10)
        self.layers = [Reshape((1, 28, 28)),
                       self.conv1, ReLU(), MaxPool2(),
                       self.conv2, ReLU(), MaxPool2(),
                       Reshape((c2 * 7 * 7,)),
                       self.fc1, ReLU(), self.fc2]

    def config(self) -> dict:
        return {"c1": self.c1, "c2": self.c2, "hidden": self.hidden, "seed": self.seed}

def load_any(path: Path) -> Sequential:
    """Load a saved model of any class, using the class name stored in the file.

    Files saved before the name was stored (milestones 2-5) hold MLPs.
    """
    with np.load(path) as f:
        kind = str(f["kind"]) if "kind" in f.files else "MLP"
    classes = {"MLP": MLP, "CNN": CNN, "SoftmaxRegression": SoftmaxRegression}
    return classes[kind].load(path)
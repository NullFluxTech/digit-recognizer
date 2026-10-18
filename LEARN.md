# LEARN.md

Notes for each milestone: equations, shapes, where the code is, and the intuition.
Shapes are written `(rows, cols)`. `N` = number of examples, `B` = batch size.

---

## Milestone 1: Data

Run it: `uv run prepare_data.py`. Code: `digits/data.py`, `prepare_data.py`, tests in `tests/test_data.py`.

### 1.1 The files and why we check them

MNIST is four gzip-compressed files:

| file                         | contents            | stored shape        |
|------------------------------|---------------------|---------------------|
| `train-images-idx3-ubyte.gz` | 60,000 images       | (60000, 28, 28) uint8 |
| `train-labels-idx1-ubyte.gz` | 60,000 labels (0-9) | (60000,) uint8        |
| `t10k-images-idx3-ubyte.gz`  | 10,000 test images  | (10000, 28, 28) uint8 |
| `t10k-labels-idx1-ubyte.gz`  | 10,000 test labels  | (10000,) uint8        |

The original host is unreliable, so we download from a mirror (`MIRRORS`, `digits/data.py:23`)
and verify each file's **MD5 checksum** against the published value (`FILES`, `digits/data.py:29`).
A checksum is a 128-bit fingerprint of the bytes: if a single byte is different (truncated download,
an HTML error page saved as `.gz`, a mirror with altered data), the fingerprint is completely
different. Before hard-coding the values I downloaded the files from two independent mirrors
and both gave byte-identical files matching the published MD5s.

How `download_file` (`digits/data.py:56`) stays safe: it writes to `name.part`, checks the
MD5 (line 62), and only then renames to the real name (line 65). So a file under its real
name in `data/` is always a verified one. If a mirror fails, `download_mnist` tries the
next one (the `for ... else` at line 91 runs only if no mirror succeeded).

(MD5 is fine for catching corruption; it is not secure against someone *deliberately* crafting
a colliding file. For that you would use SHA-256.)

### 1.2 The IDX format, byte by byte

The first bytes of `train-images-idx3-ubyte` (after gunzip) are:

```
00 00 08 03   00 00 EA 60   00 00 00 1C   00 00 00 1C   <47,040,000 pixel bytes...>
 magic        d0 = 60000    d1 = 28       d2 = 28
```

- bytes 0-1: always `00 00`
- byte 2: type code. `0x08` = unsigned byte (see `IDX_DTYPES`, `digits/data.py:38`)
- byte 3: number of dimensions D (3 for images, 1 for labels)
- then D sizes, each a 4-byte **big-endian** unsigned int
- then the values, row-major

**Big-endian** means the most significant byte comes first: `00 00 EA 60` = 0x0000EA60 = 60000.
Most CPUs today (x86, most ARM) are little-endian. If you read those 4 bytes natively you get 0x60EA0000 = 1,625,948,160.
In C++ you would fix that with `ntohl()` or `std::byteswap`. In NumPy you put the byte order
**in the dtype**: `">u4"` = big-endian (`>`) unsigned (`u`) 4-byte int.

Parsing is in `parse_idx` (`digits/data.py:102`):

```python
dims = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4)   # line 125: (D,) sizes
...
expected = math.prod(shape) * dtype.itemsize                    # line 129: sanity check
data = np.frombuffer(raw, dtype=dtype, offset=header_len)       # line 133: flat pixels
return data.reshape(shape)                                      # line 134: (60000, 28, 28)
```

`np.frombuffer` does **not** copy. It wraps the existing bytes in an array object, like this in C++:

```cpp
const uint8_t* pixels = buf + 16;                  // skip the header
auto img = reinterpret_cast<const uint8_t(*)[28][28]>(pixels);
uint8_t p = img[n][r][c];                          // same as pixels[n*784 + r*28 + c]
```

`reshape` doesn't copy either. It only changes how indices map to the flat memory. In row-major
("C order") layout, element `(n, r, c)` of a `(60000, 28, 28)` array lives at flat position

```
n*784 + r*28 + c
```

NumPy stores this as **strides** `(784, 28, 1)`: how many bytes to jump for +1 in each index.

The parser refuses anything inconsistent: a wrong magic number, an unknown type code, a header
longer than the file, or a payload that doesn't have exactly `d0*d1*...*itemsize` bytes. The
tests build little IDX files by hand with `struct.pack` and check each of these cases
(`tests/test_data.py`, `test_parse_rejects_malformed`).

### 1.3 From images to a matrix: the design matrix X

Each 28×28 image is flattened into one row of 784 numbers. Stacking the rows gives the
**design matrix**:

```
X  : (N, 784)     row i = image i, as a vector in R^784
y  : (N,)         y[i] = label of row i, an integer 0..9
```

`images_to_matrix` (`digits/data.py:146`):

```python
flat = images.reshape(n, -1)                          # (N,28,28) -> (N,784)
return flat.astype(np.float32) / np.float32(255.0)    # (N,784) uint8 -> float32 in [0,1]
```

`-1` in `reshape` means "work this size out yourself" (784 = 28·28). Row-major flattening means
pixel `(r, c)` lands in column `r*28 + c`. So the first 28 columns are the top row of the
image, the next 28 the second row, and so on.

**Where linear algebra comes in (preview of milestone 2).** With this layout, a linear model
on *all* N images at once is a single matrix product:

```
scores = X @ W + b        (N,784) @ (784,10) -> (N,10),  then + (10,) broadcast to every row
```

Row i of `scores` is `x_i W + b`, the 10 class scores for image i. We use the **row-vector
convention** (x is a row, multiplied from the left), so the shapes chain left to right like
in a C++ function pipeline. Every model in this project starts with `X @ W`.

### 1.4 Scaling to [0, 1], and two NumPy idioms

Pixels are stored as `uint8` 0..255 (0 = background, 255 = full ink). We convert to
`float32` and divide by 255:

- **Why not keep uint8?** Integer arithmetic wraps around, exactly like `unsigned char` in C++:
  `np.uint8(200) + np.uint8(100)` is `44`. Gradients also need fractions.
- **Why [0, 1]?** The model's first step is a weighted sum of 784 pixels. With inputs up to 255,
  those sums (and their gradients) come out ~255× bigger, and a learning rate that works for one
  scale blows up at the other. Small, consistent input ranges make training stable.
- **Why float32 rather than float64?** Half the memory (55,000×784 floats = 345 MB in float64,
  172 MB in float32) and faster matrix products. For the gradient checks (later milestones) we
  switch to float64, because finite differences need the extra precision.

**Broadcasting, first appearance.** `flat / np.float32(255.0)` divides an `(N,784)` array by a
single number. NumPy "broadcasts" the scalar: it acts as if the scalar were repeated to shape
`(N,784)`, without actually building that array. The C++ equivalent is a double loop
`for i, for j: out[i][j] = flat[i][j] / 255.0f`. Broadcasting gets more interesting in
milestone 2, where a `(10,)` bias is added to every row of an `(N,10)` matrix.

**`axis=`, first appearance.** Reductions such as `mean`, `sum` and `max` take an `axis` that
says which index to collapse. That axis disappears from the shape:

```
X.mean()          -> scalar      every element                   (prepare_data.py prints this: 0.1307)
X.mean(axis=0)    -> (784,)      collapse rows: the "average image"
X.mean(axis=1)    -> (N,)        collapse columns: average ink per image
```

C++ view: `axis=0` is "loop over i inside, keep j": `for j: m[j] = sum_i X[i][j] / N`.
Softmax in milestone 2 will use `axis=1` ("for each example, across its 10 classes").

### 1.5 Train / validation split

We have three roles for data:

| split      | size   | used for                                                       |
|------------|--------|----------------------------------------------------------------|
| train      | 55,000 | computing gradients, i.e. fitting the weights                  |
| validation | 5,000  | choosing hyperparameters (learning rate, hidden size, epochs…) |
| test       | 10,000 | **one** final measurement per finished model                   |

Why not tune on the test set? Each time you look at test accuracy and change something, you
pick the setting that happens to do well on *those* 10,000 images. Do that enough and the test
number stops measuring "how well does this work on new digits" and starts measuring "how well
did I fit the test set". The validation set absorbs that bias, and the test set stays an honest
estimate.

`train_val_split` (`digits/data.py:153`):

```python
perm = rng.permutation(n)                          # line 163: (60000,) indices 0..59999, shuffled
train_idx, val_idx = perm[:-n_val], perm[-n_val:]  # line 164: (55000,), (5000,)
return X[train_idx], y[train_idx], X[val_idx], y[val_idx]   # line 166
```

- **Negative slicing:** `perm[-n_val:]` = the last 5000 entries, `perm[:-n_val]` = everything before.
- **Fancy indexing, first appearance:** indexing with an *array of integers* gathers those rows:
  `X[train_idx]` has shape `(55000, 784)` and row k is `X[train_idx[k]]`. C++ equivalent:
  `for k: out[k] = X[train_idx[k]];`. Unlike `reshape`, this **copies** (the rows are scattered
  in memory, so a view with fixed strides can't describe them).
- Using the **same** `train_idx` for X and y keeps each label attached to its image. The test
  `test_split_sizes_disjoint_and_complete` checks this by storing `i` in row i.

**Seeds.** `np.random.default_rng(seed)` is a seeded random generator, like
`std::mt19937 gen(seed)`. Same seed, same permutation, same split every run. (We use seed 0.)

### 1.6 Results

```
Shapes and dtypes:
  X_train  (55000, 784)   float32
  y_train  (55000,)       int64
  X_val    (5000, 784)    float32
  y_val    (5000,)        int64
  X_test   (10000, 784)   float32
  y_test   (10000,)       int64

Pixel range: [0.0, 1.0]
Train pixel mean 0.1307, std 0.3081      <- the well-known MNIST values: the parse is right

Examples per class:
  digit      0     1     2     3     4     5     6     7     8     9
  train   5423  6138  5473  5636  5370  4956  5441  5742  5347  5474
  val      500   604   485   495   472   465   477   523   504   475
  test     980  1135  1032  1010   982   892   958  1028   974  1009
```

The classes are roughly balanced (5s are rarest, 1s most common), so plain accuracy is a fair
metric. The validation set mirrors the training distribution even though we didn't stratify.

`reports/sample_digits.png`: 60 random training images with their labels. All of them match,
which confirms that images and labels were parsed in the same order.

### Check yourself

1. The test labels file starts with `00 00 08 01 00 00 27 10`. What does each group of bytes
   mean? What number would you get if you read `00 00 27 10` as a *little*-endian uint32, and
   where in `parse_idx` would that mistake be caught?
2. After `X = images_to_matrix(images)`, which element `X[?, ?]` holds pixel (row 5, column 10)
   of image 3? Which of `images.reshape(n, -1)` and `X[train_idx]` makes a copy, and why?
3. Suppose you tried 20 learning rates and kept the one with the best *test* accuracy, 97.9%.
   Why is 97.9% probably an overestimate of accuracy on new digits, and how does the validation
   split fix that?

---

## Milestone 2: Softmax regression (784 -> 10)

Run it: `uv run train_softmax.py` (or `uv run train_softmax.py --sweep` for the validation-only
hyperparameter search). Code: `digits/nn.py`, `digits/models.py`, `digits/train.py`,
`digits/gradcheck.py`, tests in `tests/test_nn.py`.

### 2.1 The model: one matrix multiply

For one image x (a row vector):

```
z = x W + b               (1,784) @ (784,10) -> (1,10),  + (10,)      "logits" (raw scores)
p_k = exp(z_k) / sum_j exp(z_j)                          (1,10)      softmax: probabilities
prediction = argmax_k z_k
```

Write out one logit: `z_k = sum_j x_j W[j,k] + b_k`. That is the **dot product** of the image with
column k of W, plus a bias. So each column `W[:, k]` is a 784-dim "template" for digit k, and the
logit measures how well the image lines up with it. Section 2.8 shows these templates as images.

For a whole batch, stack B images as rows of X and it is the same formula:

```
Z = X W + b               (B,784) @ (784,10) -> (B,10),  + (10,) broadcast to every row
```

Code: `Linear.forward`, `digits/nn.py:29`. `SoftmaxRegression` (`digits/models.py:60`) is just
one `Linear` layer whose output goes into softmax + cross-entropy.

**Broadcasting rules.** NumPy lines shapes up *from the right*. A dimension matches if the sizes
are equal or one of them is 1, and a missing dimension counts as 1:

```
Z     (B,10)
b       (10,)   -> treated as (1,10) -> stretched to (B,10)     OK
max   (B,1)     -> stretched along columns to (B,10)            OK  (used in softmax)
bad   (B,)      vs (B,10): right-most sizes are B and 10        ERROR (unless B == 10!)
```

C++ equivalent of `X @ W + b`: `for i, for k: Z[i][k] = dot(X[i], W[:,k]) + b[k];`. The `b[k]`
not depending on `i` *is* the broadcast. That last row of the table is why the code passes
`keepdims=True` to `max`/`sum` along `axis=1`. It keeps the result as a (B,1) column instead of a
(B,) vector, so it broadcasts against the right axis.

### 2.2 The loss: cross-entropy

For one example with true class y:

```
L = -log p_y
```

- p_y = 1 (certain and right) -> L = 0
- p_y = 0.1 (uniform guess over 10 classes) -> L = log 10 = 2.303. That's the epoch-0 loss in
  the training log, because zero weights give uniform probabilities.
- p_y -> 0 (confident and wrong) -> L -> infinity

Minimizing the average of `-log p_y` is the same as maximizing the probability the model
assigns to the correct labels (maximum likelihood). For a batch we take the mean over B examples.

### 2.3 Derivation: dL/dz = p - y

Let y also mean the one-hot vector: y_k = 1 if k is the true class, else 0.

**Short route (via log-sum-exp).** Take the log of softmax:

```
log p_y = z_y - log( sum_j exp(z_j) )

L = -log p_y = -z_y + log( sum_j exp(z_j) )
```

Differentiate with respect to one logit z_k:

```
d/dz_k [ -z_y ]                     = -1 if k == y else 0         = -y_k
d/dz_k [ log(sum_j exp(z_j)) ]      = exp(z_k) / sum_j exp(z_j)   =  p_k      (chain rule: (1/S) * dS/dz_k)

dL/dz_k = p_k - y_k          for every k, i.e.   dL/dz = p - y     (1,10)
```

**Long route (via the softmax Jacobian),** to see why the messy parts cancel:

```
dp_i/dz_k = p_i (1[i==k] - p_k)                      (quotient rule on exp(z_i)/S)
dL/dp_i   = -y_i / p_i                               (L = -sum_i y_i log p_i)

dL/dz_k = sum_i (dL/dp_i)(dp_i/dz_k)
        = sum_i (-y_i / p_i) * p_i (1[i==k] - p_k)
        = -sum_i y_i 1[i==k]  +  p_k sum_i y_i
        = -y_k + p_k                                 (sum_i y_i = 1)
```

The `p_i` from the Jacobian cancels the `1/p_i` from the log, and that's why the result is so
simple and never divides by a tiny probability.

**Intuition.** The gradient is "prediction minus truth":
- true class: `p_y - 1` is negative, so gradient descent *raises* z_y, by more the less sure we were.
- every other class: `p_k` is positive, so descent *lowers* z_k, in proportion to how much
  probability we wrongly gave it.
- every entry is in [-1, 1] and each row sums to 0, so the gradient can't explode.

**Batch version.** The loss is a mean over B examples, so each row gets a factor 1/B:

```
dL/dZ = (P - Y) / B       (B,10)
```

Code (`digits/nn.py:93-96`):

```python
loss = -logP[rows, y].mean()     # (B,) -> scalar
dZ = np.exp(logP)                # (B,10)  P
dZ[rows, y] -= 1.0               # (B,10)  P - Y  (subtract 1 at each row's true class)
dZ /= B                          # (B,10)
```

**Fancy indexing with two arrays.** `logP[rows, y]` with `rows = [0, 1, ..., B-1]` pairs them up
element by element: it picks `logP[0, y[0]], logP[1, y[1]], ...` -> shape (B,). In C++:
`for i: out[i] = logP[i][y[i]];`. The same trick on the left of `-=` subtracts 1 at exactly those
positions, so we never build the one-hot matrix Y.

### 2.4 Backprop into W and b (and X)

From `z_k = sum_j x_j W[j,k] + b_k` for one example:

```
dL/dW[j,k] = x_j * dL/dz_k       ->  dW = x^T g     (784,1) @ (1,10) -> (784,10)   an outer product
dL/db_k    = dL/dz_k             ->  db = g         (10,)
dL/dx_j    = sum_k W[j,k] dL/dz_k ->  dx = g W^T    (1,10) @ (10,784) -> (1,784)
```

where `g = dL/dz`. Over a batch the parameter gradients add up across examples, and **a sum
of outer products is exactly a matrix product**:

```
dW = sum_i x_i^T g_i = X^T G      (784,B) @ (B,10) -> (784,10)
db = sum_i g_i       = G.sum(axis=0)       (B,10) -> (10,)
dX = G W^T                        (B,10) @ (10,784) -> (B,784)   (one row per example, no sum)
```

Code: `Linear.backward`, `digits/nn.py:33-35`. `dX` isn't used by softmax regression, but the
MLP in milestone 3 needs it to pass the gradient down to the layer below.

**Shape trick to remember it.** A parameter's gradient has the parameter's shape. dW must be
(784,10). From X (B,784) and G (B,10), the *only* product that gives (784,10) is `X^T @ G`. The
shared dimension B is the one that gets summed away, and summing over examples is exactly
what we want.

### 2.5 Numerical stability

Softmax doesn't change if you subtract the same constant c from every logit:

```
exp(z_k - c) / sum_j exp(z_j - c)  =  e^-c exp(z_k) / (e^-c sum_j exp(z_j))  =  p_k
```

Choosing `c = max_j z_j` makes the largest exponent `exp(0) = 1`:
- no overflow: `exp(1000)` is `inf` in float64, but we never compute it.
- the sum is >= 1, so no division by 0 and no `log(0)`.
- tiny terms may *underflow* to 0 (`exp(-2000) = 0`), which is harmless.

For the loss we never compute `log(p)` from p. We use log-softmax directly:

```
log p_k = (z_k - c) - log( sum_j exp(z_j - c) )      ("log-sum-exp trick")
```

So a hopeless prediction (true-class logit 10,000 below the others) gives a loss of 10,000
instead of `log(0) = -inf`. Code: `softmax` and `log_softmax`, `digits/nn.py:71-80`. Tests:
`test_softmax_stable_on_huge_inputs` (logits up to 1e300 with overflow turned into an error)
and `test_cross_entropy_stable_when_true_class_is_hopeless`.

### 2.6 Mini-batch SGD and choosing the learning rate

```
W <- W - lr * dW          (784,10)
b <- b - lr * db          (10,)
```

Code: `SGD.step`, `digits/nn.py:109`. `p -= ...` updates the array *in place*. The layer and the
optimizer hold the same array object, the way two C++ references to one buffer would.

One **epoch** = shuffle the 55,000 training rows, cut them into batches of B, and take one SGD
step per batch (`minibatches`, `digits/train.py:11`; loop at `digits/train.py:97-98`). A small B
gives noisy but cheap steps, and a large B gives accurate but expensive ones.

The sweep (`uv run train_softmax.py --sweep`, 15 epochs each, validation accuracy only):

```
   lr batch  best val acc at epoch  final val acc
 0.01    16        0.9172       15         0.9172
 0.01    64        0.9034       15         0.9034
 0.01   256        0.8820       15         0.8820
 0.05    16        0.9222       11         0.9208
 0.05    64        0.9190       15         0.9190
 0.05   256        0.9054       12         0.9054
 0.10    16        0.9214       15         0.9214
 0.10    64        0.9220       11         0.9208
 0.10   256        0.9132       15         0.9132
 0.50    16        0.9134       11         0.9052
 0.50    64        0.9214       11         0.9198
 0.50   256        0.9220       11         0.9210
 1.00    16        0.8998       11         0.8928
 1.00    64        0.9158        4         0.9052
 1.00   256        0.9218       11         0.9210
```

**The pattern is lr / B.** The diagonal 0.05/16, 0.1/64, 0.5/256 all land around 92.1%. Over one
epoch you take N/B steps of size `lr * (mean gradient)`, so the total distance travelled per
epoch is roughly `lr * N / B * (typical gradient)`. The same lr/B means the same progress per
epoch.
- **Too small lr/B** (0.01/256): still improving at epoch 15, it just hasn't arrived yet.
- **Too large lr/B** (1.0/16): each step overshoots. One example's step changes its own logits by
  about `lr/B * |x|^2 * (p - y)`. MNIST images have `|x|^2` around 88 (from 13 to 295), so
  `1.0/16 * 88` is about 5.5 logit units per step, and the weights keep jumping around.
- The top runs differ by 2-3 images out of 5,000, which is noise. I picked **lr = 0.1, B = 64**
  (it ties with the best and has 4x fewer steps than B = 16).

### 2.7 Gradient check

Backprop can be silently wrong and still train "OK", so we compare it against the definition of a
derivative, one weight at a time (`digits/gradcheck.py`):

```
numerical:  dL/dw ~= ( L(w + eps) - L(w - eps) ) / (2 eps)       error O(eps^2)
compare:    rel_err = |analytic - numerical| / max(|analytic|, |numerical|)
```

`numerical_grad` (`digits/gradcheck.py:23`) nudges one entry through a flat **view** of the
parameter (line 29, `p.reshape(-1)`: writing into the view writes into W). It runs the forward
pass twice and restores the value exactly.

**Why float64, and choosing eps.** With eps = 1e-5 and a typical gradient of 0.05, the two losses
differ by about 2 * eps * 0.05 = 1e-6 on a loss of about 2.3, so only from the 6th-7th significant
digit on. float32 keeps ~7 digits, so the difference would be almost pure rounding noise. Even in float64, eps is a trade-off.
These are measured on this model:

```
eps      1e-7     1e-6     1e-5     1e-4     1e-3
max err  2.4e-5   2.8e-6   3.4e-7   4.7e-7   4.9e-5
         <- rounding error ~1e-16/eps        Taylor error ~eps^2 ->
```

The worst entries are always the ones with tiny true gradients (~7e-5). There, a fixed absolute
error becomes a large *relative* error. eps = 1e-5 sits at the bottom of the U.

**Result** (`test_softmax_regression_gradient_check`, float64, random W, 8 random images, all
7,850 entries): max relative error **W: 3.4e-7, b: 5.5e-9**, both below the 1e-5 requirement.
`test_linear_input_gradient_numerically` checks dX the same way.

### 2.8 Results

```
uv run train_softmax.py
  epoch   0  train loss 2.3026 acc 0.0986  val loss 2.3026 acc 0.1000
  epoch   1  train loss 0.3636 acc 0.9003  val loss 0.3687 acc 0.8974
  ...
  epoch  15  train loss 0.2640 acc 0.9268  val loss 0.2806 acc 0.9208
Training time: 1.1s

Final  val acc 0.9208   test acc 0.9229 (loss 0.2727)
```

- **Validation 92.08%** (target >= 91%). **Test 92.29%**, measured once, on the finished model.
- `reports/softmax_curves.png`: train and validation stay close (gap ~0.6 points) and both
  flatten out around 92%. That's **underfitting**, not overfitting: a single linear layer just
  can't do better on MNIST, whatever the hyperparameters. That's the motivation for the hidden
  layer in milestone 3.

**What the weights show** (`reports/softmax_weights.png`). Column k of W reshaped to 28x28.
Red pixels: ink there *raises* the score for digit k. Blue: ink there *lowers* it.
- **0**: a red ring with a blue center. Ink in the middle is evidence *against* a zero.
- **1**: a red vertical stroke with blue on both sides.
- **2, 3, 6**: you can make out the shapes. The blue parts matter as much as the red: the 3
  is blue on the left, where a 0 or 8 would close its loops.
- **5, 8, 9** are hard to read. Those are the classes one average template captures worst.
- **The border is exactly gray (0).** Those pixels are 0 in every MNIST image, so
  `dW[j,k] = sum_i x_ij g_ik = 0` forever, and weights that start at 0 stay at 0.
- One template per class is the model's limit. It has to blur all the 7s (with or without a
  crossbar, slanted or upright) into a single average pattern.

### Check yourself

1. One example has logits z = (2, 1, 0) over three classes, and its true class is 1. Roughly what
   is p, and what is that example's row of P - Y? Which logits does a gradient step push up,
   and which down?
2. Why is `dW = X.T @ G` and not `G @ X.T` or `X @ G.T`? Use the shapes, and explain what
   "summing over the batch" has to do with the matrix product.
3. The border of every weight image is exactly gray. Why do those weights never change during
   training? Would that still be true if W were initialized with small random numbers? Would it
   matter for the predictions?
---

## Milestone 3: MLP (784 -> 512 ReLU -> 10)

Run it: `uv run train_mlp.py` (from scratch, ~28 s total), or `uv run train_mlp.py --sweep`
(validation only). Code: `digits/nn.py` (ReLU, He init, momentum), `digits/models.py`
(`Sequential`, `MLP`), `digits/train.py` (schedules, early stopping), tests in `tests/test_mlp.py`.

### 3.1 Forward pass: two matrix multiplies with a ReLU between them

```
H  = X W1 + b1        (B,784) @ (784,512) -> (B,512)   + (512,) broadcast    pre-activations
A  = relu(H)          (B,512)                                               hidden activations
Z  = A W2 + b2        (B,512) @ (512,10)  -> (B,10)    + (10,)  broadcast    logits
L  = mean_i -log softmax(Z)_i[y_i]                                          as before
```

407,050 parameters (W1 alone is 401,408), against 7,850 for softmax regression.

**Why the ReLU is essential.** Without it, the two layers collapse into one:

```
(X W1 + b1) W2 + b2  =  X (W1 W2) + (b1 W2 + b2)        W1 W2 : (784,512) @ (512,10) -> (784,10)
```

That's just another softmax regression, and a rank-<=10 one at that. A product of linear maps is a
linear map. The nonlinearity between them is what lets the network represent things a single
template per class can't: "a 7 is either this shape *or* that shape".

**What a hidden unit does.** `H[:, j] = X @ W1[:, j] + b1[j]` is the same template matching as
in milestone 2, but now there are 512 templates, and ReLU keeps only *positive* evidence:
unit j says "this part of the pattern is present, this strongly". The output layer then
combines 512 such detectors into 10 scores. In the trained model, on average only **12% of the
units fire for a given image**: each image is described by a small, specific set of features.
Milestone 4 shows these templates as images.

`Sequential.forward` (`digits/models.py:29`) just calls `layer.forward` in order:
`fc1` (Linear), `relu` (ReLU), `fc2` (Linear). `MLP` builds those three layers at
`digits/models.py:88-90`.

### 3.2 Backward pass: the chain rule, layer by layer

Each layer gets G = dL/d(its output) and returns dL/d(its input). Walk the layers in reverse:

```
dZ  = (P - Y) / B          (B,10)                            softmax + cross-entropy (milestone 2)

fc2:  dW2 = A^T dZ         (512,B) @ (B,10)  -> (512,10)
      db2 = sum_rows dZ    (B,10) -> (10,)
      dA  = dZ W2^T        (B,10) @ (10,512) -> (B,512)      <- dX of fc2, unused in milestone 2

relu: dH  = dA * 1[H > 0]  (B,512) * (B,512) -> (B,512)      elementwise mask

fc1:  dW1 = X^T dH         (784,B) @ (B,512) -> (784,512)
      db1 = sum_rows dH    (B,512) -> (512,)
      (dX = dH W1^T would go further down, but there's nothing below)
```

Code: the loop `for layer in reversed(self.layers): G = layer.backward(G)` at
`digits/models.py:35-36`. Each `Linear.backward` is the same three lines from milestone 2
(`digits/nn.py:33-35`). The only new piece is `ReLU.backward` (`digits/nn.py:54`).

**ReLU's derivative.** relu(h) = max(0, h) has slope 1 for h > 0 and 0 for h < 0 (at h = 0 we
pick 0). So backprop lets the gradient through where the unit was active and blocks it where it
wasn't. The forward pass saves that mask (`self.mask = Hpre > 0`, `digits/nn.py:49`, a (B,512)
array of bools) and backward multiplies by it.

**`*` is not `@`.** `G * mask` multiplies entry by entry (both are (B,512), and so is the result).
`G @ W` is the matrix product. In C++ terms, `*` is
`for i, j: out[i][j] = G[i][j] * mask[i][j];`, and a bool multiplies as 0 or 1.

**Dead units.** A unit whose pre-activation is <= 0 for *every* input always has a zero mask,
so it never gets a gradient again and can't recover. The trained model has **11 dead units out of
512**. That's harmless here, but it's why a too-large learning rate can kill a big part of a ReLU
network in a few steps.

### 3.3 He initialization (why sqrt(2 / fan_in))

Unlike softmax regression, the MLP can't start from zeros. If every column of W1 were equal,
every hidden unit would compute the same thing, get the same gradient, and stay identical forever.
Random init breaks that symmetry. The question is how large the random numbers should be.

Take one pre-activation, with n = fan_in inputs a_i and independent weights of mean 0 and variance s^2:

```
h = sum_{i=1..n} a_i W_i
Var(h) = n * s^2 * E[a^2]                     (independent terms add their variances)
```

ReLU zeroes the negative half of a symmetric h, which halves its second moment:

```
E[relu(h)^2] = Var(h) / 2 = (n * s^2 / 2) * E[a^2]
```

So the signal's size gets multiplied by `n s^2 / 2` at every layer. Choosing **s^2 = 2/n**
makes that factor exactly 1. That's He init (`he_init`, `digits/nn.py:60-63`).

Measured with 10 stacked ReLU layers of width 256 (40 seeds, `tests/test_mlp.py`):

```
init                  E[a^2] after 10 layers / before
He  s = sqrt(2/n)     median 0.83  (range 0.40 .. 2.19; at width 1024: 0.71 .. 1.51)
bad s = sqrt(1/n)     0.001 every time  (= 2^-10: halved at each layer)
```

He init keeps the scale steady *on average*. Each finite layer multiplies it by a random factor
near 1, and wider layers average better. With only two layers this matters less for us, but it
sets a sensible starting point: the untrained MLP has loss 2.469, close to log 10 = 2.303 (not
exactly, because random weights don't give uniform probabilities).

### 3.4 SGD with momentum

```
v <- mu * v - lr * g            (same shape as the parameter)
p <- p + v
```

Code: `SGDMomentum.step`, `digits/nn.py:112-134`, three in-place lines (132-134). The optimizer
keeps one velocity array per parameter in a dict keyed by `id(p)`, the object's identity, like a
`std::unordered_map<const float*, Array>` keyed by the buffer's address.

Unroll the recursion and v is a decaying sum of all past gradients:

```
v_t = -lr * (g_t + mu g_{t-1} + mu^2 g_{t-2} + ...)          weights sum to 1/(1-mu) = 10 for mu = 0.9
```

- In a direction where the gradient keeps the same sign, the steps add up: the effective
  learning rate is `lr / (1 - mu)`, 10x larger (for us 0.3 / 0.1 = 3).
- In a direction where it flips sign from batch to batch (mini-batch noise, or bouncing between
  the walls of a narrow valley), the terms cancel out.

That's the "heavy ball" picture: a ball rolling downhill builds up speed along the valley floor
and doesn't react much to each bump. `test_momentum_matches_hand_computation` steps through three
updates by hand, including one with zero gradient where the ball keeps rolling.

### 3.5 Learning-rate schedule

```
cosine:   lr_e = lr * (1 + cos(pi * (e - 1) / E)) / 2          e = 1..E    (digits/train.py:39-45)
step:     lr_e = lr * 0.1 ^ floor((e - 1) / 7)                 drops at epochs 8 and 15 (digits/train.py:48-50)
```

The schedule is applied at the start of each epoch (`optimizer.lr = schedule(epoch)`,
`digits/train.py:93`). Large steps early cover ground fast. Small steps late let the mini-batch
noise average out, so the weights settle into the bottom of the minimum instead of bouncing around
it. The seed check in 3.7 shows the effect: with lr = 0.3, constant LR gives 98.02-98.44% across
three seeds and cosine gives 98.26-98.38%. The mean is the same, but cosine is much more
predictable.

### 3.6 Early stopping

After every epoch (`digits/train.py:102-112`):
- if validation accuracy beats the best so far, remember the epoch and **copy** all parameters
  (`p.copy()`; a plain reference would keep changing with training)
- if it hasn't improved for `patience = 5` epochs, stop
- at the end, write the best copy back into the model (`p[...] = best`, in place)

This is a form of **regularization**: it caps how long the model can spend memorizing the
training set. In the final run (`reports/mlp_curves.png`):

```
epoch 9:   train acc 0.9998  val loss 0.0862 (lowest)   val acc 0.9822
epoch 11:  train acc 1.0000  val loss 0.0868            val acc 0.9832  <- best, kept
epoch 16:  train acc 1.0000  val loss 0.0884            val acc 0.9830  -> stop (5 epochs without improvement)
```

From epoch 11 on the model gets every training image right (train loss 0.0005) and there's
nothing left to learn from them. Validation loss *rises* while validation accuracy stays flat:
the model grows more and more confident, including on the ~85 validation images it gets wrong,
and confident mistakes cost a lot of cross-entropy. We stop on accuracy because accuracy is the
target.

### 3.7 Choosing the hyperparameters (validation only)

Stage 1, `uv run train_mlp.py --sweep` (cosine, momentum 0.9, batch 128, 20 epochs, no early stop):

```
   H    lr  best val acc epoch  train acc   time
 128  0.01        0.9686    18     0.9780     7s
 128  0.03        0.9754    17     0.9941     7s
 128  0.10        0.9794    11     0.9995     7s
 128  0.30        0.9792    12     0.9997     7s
 256  0.01        0.9686    16     0.9790     9s
 256  0.03        0.9770    12     0.9943     9s
 256  0.10        0.9808     9     0.9991     8s
 256  0.30        0.9808     9     0.9990     9s
 512  0.01        0.9720    17     0.9813    22s
 512  0.03        0.9782    10     0.9937    22s
 512  0.10        0.9826    15     1.0000    21s
 512  0.30        0.9832    11     1.0000    20s
```

- **lr:** 0.01 is too slow. 0.1 and 0.3 tie at every size, so the best value is no longer at
  the edge of the grid (my first grid stopped at 0.1, so I added 0.3 to check).
- **H:** 512 beats 256 by about 0.2 points at *every* lr. That's a consistent pattern, not one
  lucky run. It costs 2.5x the time, which still fits the budget easily.

Stage 2 (H = 512, lr = 0.3, early stopping patience 5):

```
schedule  best val acc epoch stopped
constant        0.9844    18      20
    step        0.9812    11      16
  cosine        0.9832    11      16
```

Constant looks best. **Is that real?** Rerunning with three seeds (the seed changes both the
init and the shuffling):

```
constant lr 0.1: 0.9830 0.9838 0.9830   mean 0.9833
constant lr 0.3: 0.9844 0.9820 0.9802   mean 0.9822
cosine   lr 0.1: 0.9826 0.9832 0.9816   mean 0.9825
cosine   lr 0.3: 0.9832 0.9826 0.9838   mean 0.9832
```

All four means are within 0.11 points (6 validation images), while the seed alone moves a single
run by up to 0.42 points. **So single-run differences under ~0.4 points are noise here.** The stage-2
"win" for constant LR was one lucky seed. I picked **H = 512, lr = 0.3, cosine**: as good as
any, the most stable across seeds, and it's a real schedule (which the milestone requires).

### 3.8 Gradient check (float64)

`test_mlp_gradient_check`: H = 16 so that all 12,730 parameters can be checked, 8 random images,
random biases:

```
layer0.W (W1)  3.5e-6      layer2.W (W2)  7.6e-7
layer0.b (b1)  1.8e-9      layer2.b (b2)  5.9e-10          all < 1e-5
```

(`layer1` is the ReLU, which has no parameters.) W1's 3.5e-6 is a bit close to the limit, so I
looked at what causes it, varying eps:

```
eps        1e-6     1e-5     1e-4     1e-3
max err    4.2e-5   3.5e-6   2.6e-7   1.79   (!)
```

- At eps = 1e-5 only 4 of 12,544 W1 entries exceed 1e-6. The worst has a true gradient of 5.4e-6
  (5,000x smaller than the median) and an absolute error of 1.9e-11: pure rounding noise on a
  tiny number.
- At eps = 1e-3 the check **fails** on 372 entries. Those nudges are big enough to push some
  pre-activations across ReLU's kink at 0. On one side the slope is 1, on the other 0, so the
  finite difference measures neither. This is the classic gradient-check pitfall for ReLU
  networks.
- One nudge of a W1 entry changes a pre-activation by at most eps * |x| <= 1e-5. The test
  asserts that the smallest |pre-activation| is above 1e-4 (it's 5.5e-4), so no nudge can cross
  the kink.

I kept eps = 1e-5. It passes with a well-understood worst case and stays far from the kink. All
four parameters pass.

### 3.9 Results

```
uv run train_mlp.py
  epoch   1  train loss 0.1060 acc 0.9657  val loss 0.1383 acc 0.9584  lr 0.3000
  ...
  epoch  11  train loss 0.0005 acc 1.0000  val loss 0.0868 acc 0.9832  lr 0.1500
  ...
  epoch  16  ...  early stop: no val improvement for 5 epochs
  restored weights from epoch 11 (val acc 0.9832)

Kept epoch 11:  val acc 0.9832   test acc 0.9842 (loss 0.0754)
Total time (load data + train + evaluate + save): 27.6s
```

- **Test 98.42%** (target >= 97.5%), measured once on the finished model.
- **27.6 s** from scratch, including loading data and evaluating every epoch (budget: 2 min).

Compared with softmax regression:

| | softmax regression | MLP |
|---|---|---|
| parameters | 7,850 | 407,050 |
| train accuracy | 92.7% | 100.0% |
| validation accuracy | 92.1% | 98.3% |
| test accuracy | 92.3% (771 errors) | 98.4% (158 errors) |
| diagnosis | underfitting: train ~ val, both capped | overfitting-ish: train perfect, val 1.7 pts lower |

The hidden layer cuts the test errors by about 5x. The remaining gap between train and
validation is the next thing you could work on (weight decay, data augmentation), but the
milestone target is already met with a wide margin.

### Check yourself

1. In the backward pass, why does `dW1` need `dH` (after the ReLU mask) and not `dA`? What would go
   wrong in the gradient if you forgot the mask? Write the shapes of `X^T`, `dH` and `dW1`.
2. With momentum mu = 0.9, suppose the gradient were the same vector g at every step. What does
   the velocity converge to, and what does that say about the "effective" learning rate? What
   happens instead if g alternates between +g and -g?
3. In the final run, validation *loss* is lowest at epoch 9 but validation *accuracy* is best at
   epoch 11. How can loss go up while accuracy stays the same or improves? Which one should early
   stopping watch, and why did we choose accuracy?
---

## Milestone 4: Evaluating the MLP

Run it: `uv run evaluate_mlp.py` (needs `models/mlp.npz` from `uv run train_mlp.py`). Code:
`digits/metrics.py`, `evaluate_mlp.py`, tests in `tests/test_metrics.py`.

No training or tuning happens here. The MLP was finished in milestone 3 and its test accuracy
reported then; this milestone looks more closely at *those same* predictions (still 98.42%, 158
errors). Nothing we learn here goes back into choosing the model, because that would quietly turn
the test set into a validation set.

### 4.1 The confusion matrix

```
C[i, j] = number of test images whose true digit is i and whose predicted digit is j      (10,10)
```

Row i holds every true i (row sums = class counts), the diagonal holds the correct predictions,
and every off-diagonal number is a specific kind of mistake: "an i read as a j".

```
             0     1     2     3     4     5     6     7     8     9      <- predicted
     0     972     1     1     1     1     0     1     1     2     0
     1       0  1128     1     1     0     0     1     1     3     0
     2       1     1  1014     4     3     0     1     4     4     0
     3       1     0     2   998     0     4     0     1     2     2
     4       2     0     3     1   963     0     3     1     0     9
     5       3     0     0     8     0   872     3     0     4     2
     6       5     2     0     1     4     3   943     0     0     0
     7       1     4     7     1     0     0     0  1011     1     3
     8       3     0     2     6     2     0     3     3   951     4
     9       1     2     1     2     6     1     0     5     1   990
  ^ true
```

`reports/confusion_matrix.png` shows the same matrix. Off-diagonal counts are on a blue scale.
The diagonal is gray, because its ~1000s would otherwise make every 1-9 look the same pale blue.

**Building it: the `np.add.at` pitfall** (`digits/metrics.py:17`). We want
`for n: C[y_true[n]][y_pred[n]] += 1`. The obvious NumPy line is wrong:

```python
C[y_true, y_pred] += 1          # WRONG: each distinct (i, j) pair is incremented only once
np.add.at(C, (y_true, y_pred), 1)   # right: every pair counts, duplicates included
```

`C[idx] += 1` with fancy indexing really does three steps: *gather* `tmp = C[idx]`, add
`tmp + 1`, *scatter* `C[idx] = tmp + 1`. If the pair (4, 9) appears 9 times, all 9 scatters write
the same value, `old + 1`. `np.add.at` is "unbuffered": it does the real one-at-a-time loop.
`test_repeated_pairs_all_count` shows both behaviors side by side.
(Another correct trick: `np.bincount(y_true * 10 + y_pred, minlength=100).reshape(10, 10)`. That
flattens the 2-D index to 1-D row-major, the same formula as pixels in milestone 1.)

### 4.2 Precision and recall: what accuracy hides

```
recall[k]    = C[k,k] / sum_j C[k,j]      of all true k's, the fraction we found      (row sums,    axis=1)
precision[k] = C[k,k] / sum_i C[i,k]      of all "k" predictions, the fraction right  (column sums, axis=0)
```

Code: `per_class_metrics`, `digits/metrics.py:29-34`. `C.sum(axis=1)` collapses the column index
(sum across a row), and `C.sum(axis=0)` collapses the row index (sum down a column).

```
digit  support  errors  recall  precision
    0      980       8  99.18%     98.28%
    1     1135       7  99.38%     99.12%
    2     1032      18  98.26%     98.35%
    3     1010      12  98.81%     97.56%    <- lowest precision
    4      982      19  98.07%     98.37%
    5      892      20  97.76%     99.09%
    6      958      15  98.43%     98.74%
    7     1028      17  98.35%     98.44%
    8      974      23  97.64%     98.24%    <- lowest recall
    9     1009      19  98.12%     98.02%
```

- **0 and 1 are easy** (7-8 errors each). **8 is the hardest to find** (23 missed).
- **5 has high precision but low recall**: when the model says "5" it's nearly always right
  (99.1%), but it misses 20 real 5s, 8 of them read as 3.
- **3 is the model's favorite wrong answer**: 25 non-3s get called 3 (column 3 minus its
  diagonal), the most of any column. So "3" has the lowest precision even though real 3s are
  found well.

The single number 98.42% averages all of this away. Per-class numbers tell you *where* to look.

### 4.3 The most confused pairs

`most_confused_pairs`, `digits/metrics.py:38-54`. With `symmetric=True` it adds `C + C^T` so that
"4 read as 9" and "9 read as 4" count together, then keeps each unordered pair once (upper
triangle, line 49).

```
4 <-> 9: 15   (9 as 4->9, 6 as 9->4)
3 <-> 5: 12   (4 as 3->5, 8 as 5->3)
2 <-> 7: 11   (4 as 2->7, 7 as 7->2)
3 <-> 8:  8   (2 as 3->8, 6 as 8->3)
7 <-> 9:  8   (3 as 7->9, 5 as 9->7)
```

Each pair differs by **one small stroke**: whether a 4's top is closed (-> 9), the top-left corner
of a 3 vs a 5, a 7 with a curly stem and a base stroke vs a 2, the closed left side of an 8 vs a
3. As 784-dim vectors, such digits share most of their ink and differ in only a few dozen
pixels. A model that scores images with dot products sees mostly the shared part. The mistakes
are not random: they're the pairs that really are close in pixel space.

### 4.4 The misclassified gallery

`reports/misclassified.png` shows all 158 errors as `true->predicted` with the probability the
model gave its (wrong) answer, sorted most confident first (`evaluate_mlp.py:62-63`: two-array
fancy indexing picks `P[i, pred[i]]` for each mistake, then `argsort` of the negated values gives
descending order).

- **Many are hard even for a person.** 4s with a closed top, 7s with a curly stem, a 6 written
  like an "L" (shown as 6->1 at 100%), broken strokes, digits drawn very thick or very slanted. A few
  are hard to call at all. MNIST is known to contain some likely-mislabeled images.
- **76 of the 158 mistakes are made with > 90% confidence.** Softmax outputs are *not* a reliable
  measure of certainty. Training to 100% train accuracy pushes logits to extremes, which is the
  same effect as the rising validation loss in section 3.6. A confident model is not
  necessarily a correct one.
- **Unusual position and thickness show up repeatedly.** The model only ever saw MNIST's
  conventions (centered, similar stroke width). Keep this in mind for milestone 5: your own
  digits must be preprocessed into exactly these conventions, or the error rate will be much
  higher than 1.6%.

### 4.5 What the first layer learned

`reports/mlp_first_layer.png` shows 40 columns of W1 (each (784,) reshaped to 28x28; red = ink
here raises the unit's activation, blue = lowers it): the 30 most often active units, the 5
least active (each fires on exactly 1 training image), and 5 dead ones. The title also shows
**which digit the unit votes for and against**: the largest and smallest entries of its row
`W2[j, :]` (`evaluate_mlp.py:80`), i.e. how its output feeds the 10 logits.

**Local stroke detectors, not whole digits.** In milestone 2 each class had one smooth template
the size of the whole digit. Here most units respond to a *part*: a short diagonal stroke at a
particular place (unit 279), a horizontal bar (unit 82), a curve. Red and blue regions often sit
side by side. A unit like that says "ink here **and not** right next to it", which makes it
sensitive to the exact position and direction of a stroke (an edge detector).

**Some units read like digit parts, and vote accordingly.** Unit 272 (blue center, red ring) votes
**+0**, unit 243 (a vertical red stroke) votes **+1**, unit 82 (a bar near the top) votes **+7**. The
output layer combines these: a 7 is "top bar present AND diagonal present AND no closed loop",
built from several overlapping detectors, and each image switches on only ~12% of them. That
is how one network can cover many writing styles of the same digit, which one template per class
couldn't.

**The speckle is leftover initialization.** *(This also answers milestone 2's question 3. Skip
this paragraph if you want to work that one out first.)*
- 67 pixels are 0 in every training image. Their gradient `dW1[pixel, :] = sum_i x_i,pixel * dH_i`
  is always 0, so their weights are **bit-for-bit identical to the He initialization** (checked
  by rebuilding the seed-0 init and comparing).
- Even on inked pixels, training changed W1 by only 0.039 on average, less than the initial
  spread (std 0.051). The learned structure sits *on top of* the random start, so every template
  looks noisy.
- On MNIST this noise is harmless: those pixels are always 0, so the random weights get multiplied
  by 0. But a real photo with a stray mark near the border *would* hit those random weights.
  That's one more reason milestone 5's preprocessing crops, scales and centers the ink, so the
  border stays empty, as in MNIST.

**Dead units.** 11 units never fire on any training image, and 5 more fire on just one. They're
not untouched: every dead unit's weights moved away from the init, and its bias drifted from 0
to negative (unit 280: -0.10). So each one was alive early in training, got pushed below zero
for every image, and from then on received zero gradient (the ReLU mask is all zeros) and froze.
Unit 280's blue blob in the center is what it learned before dying. 16 of 512 units (3%) is a
small loss. A larger learning rate would kill more.

### Check yourself

1. Without looking it up: in `C`, is "a 5 predicted as 3" stored at `C[5, 3]` or `C[3, 5]`? Which
   axis do you sum over to get recall, and why does a class that's often *predicted wrongly* have
   low precision but not necessarily low recall?
2. Why does `C[y_true, y_pred] += 1` give the wrong matrix? Describe what NumPy actually does,
   step by step, when the same (i, j) appears three times.
3. Unit 82 votes +7 and -4. In words: what ink pattern raises its activation, and how does that
   activation then change the 10 logits? Which entries of W1 and W2 do you read for each part?
---

## Milestone 5: My handwriting

Commands:
- `uv run draw.py`: draw with the mouse; live top-3; keys 0-9 set the label, Enter saves to `samples/`
- `uv run predict.py image.png [--debug]`: top-3 for any image file; `--debug` saves `reports/predict_steps.png`
- `uv run score_samples.py`: scores every `samples/<label>_<nn>.png`, saves `reports/samples_scored.png`
- `uv run train_mlp.py --augment` / `--personal` / `--personal-cv`: the milestone-5 models (sections 5.5-5.6)

Code: `digits/preprocess.py`, `digits/augment.py`, `digits/inference.py`, `predict.py`, `draw.py`,
`score_samples.py`; tests in `tests/test_preprocess.py`, `tests/test_augment.py`, `tests/test_tools.py`.

### 5.1 What "MNIST format" means, measured

The model has only ever seen images that follow MNIST's conventions exactly. Measured on the
training set:

```
background 0, brightest pixel exactly 1.0             (every image)
ink bounding box: longer side exactly 20 px           (100% of images), aspect ratio kept
center of mass at (row, col) = (14.0, 14.0)           (std 0.29 px)
solid strokes: 49% of ink pixels are >= 0.9
stroke width ~14% of the digit's size                 (2.82 px of 20; 10th-90th pct ~10%-17%)
```

How much does it matter? The milestone-3 MLP on MNIST validation digits shifted sideways:

```
shift   0 px     1 px     2 px     3 px     4 px
acc     98.3%    96.9%    86.4%    62.6%    37.8%
```

A 2-pixel offset costs 12 points. Milestone 4 showed why: every first-layer unit is a template
pinned to fixed pixel positions. The model has no idea that a digit moved; it just sees ink in
the "wrong" pixels. So preprocessing is not cosmetic: it decides whether the model sees anything
it recognizes.

### 5.2 The pipeline, step by step

`to_mnist_steps` (`digits/preprocess.py:113`) returns every intermediate image; `reports/predict_steps.png`
shows them for one of my digits (`uv run predict.py personal/7_01.png --debug`).

1. **Load as grayscale** (`load_gray`, line 27): any file -> (H,W) floats in [0,1]. Transparent
   PNGs are flattened onto white first, otherwise transparent pixels would read as black.
2. **Polarity** (line 125-128): MNIST is white ink on black. The border is almost always
   background, so if the border's median is bright (paper), invert: `g = 1 - g`. A digit with the
   wrong polarity is a photographic negative to the model: all its templates fire on the paper.
3. **Clean up** (lines 132-145):
   - subtract the background level, so paper becomes 0;
   - divide by the **median stroke pixel** (`stroke_level`, line 140), not the darkest pixel, then
     clip at 1. Pencil and uneven light leave most of a stroke gray; MNIST strokes are solid.
     The network is (piecewise) linear in its input, so a stroke at half brightness pushes every
     pre-activation about half as hard: faint ink is weak evidence;
   - zero out faint noise (< 0.1), then drop faint blobs that don't touch solid ink
     (`_connected_to`, line 145): in MNIST, gray only appears at the edges of strokes.
4. **Crop to the ink** (line 148-152): bounding box of pixels > 0.2.
5. **Pen width** (lines 154-164): if the stroke is thinner than MNIST's 10th percentile (10.8% of
   the digit's size), thicken it to the median (14.1%). The width is estimated as
   `2 * area / perimeter` (`stroke_width`, line 57: a stroke of length L and width w has area
   about L*w and perimeter about 2L). Thickening is a grayscale max filter (`_dilate`, line 69),
   alternating 3x3 square and cross passes so diagonal strokes don't grow ~1.4x faster than
   straight ones. Strokes inside MNIST's normal range are left alone.
6. **Longer side = 20 px, aspect kept** (lines 166-171): a 1 stays a thin bar instead of being
   stretched into a block. Pillow's Lanczos filter averages many input pixels into each output
   pixel, which gives MNIST's soft gray edges. Then the brightest pixel is set back to exactly 1.
7. **Center of mass to (14, 14)** (lines 174-183):

```
cy = sum_r r * (ink in row r) / total ink       = (row sums) . (0, 1, ..., 27) / total      (dot product)
cx = sum_c c * (ink in col c) / total ink
shift = round(14 - cy), round(14 - cx), clamped so no ink leaves the frame
```

   Why mass and not the bounding-box center? Because MNIST did it this way, and the model learned
   *that* placement. For a 7, whose ink is mostly in the top bar, the two differ by a pixel or
   two, and from the table above, 2 px is 12 points of accuracy.

**Check that it reproduces MNIST:** MNIST test digits made 6x larger, inverted onto white paper
and placed at random spots come back to (nearly) the original (`tests/test_preprocess.py`):
**497 of 500** keep their prediction.

### 5.3 The tools

- `predict.py`: load -> `to_mnist` -> MLP -> softmax -> top 3. `predict_probs` (`digits/inference.py`)
  turns one image into a batch of one with `x[None, :]`: (784,) -> (1,784), because the model
  always works on rows.
- `draw.py`: a Tkinter canvas. Every mouse movement paints a thick segment into a (280,280) NumPy
  array (`stamp_segment`). A pixel p is ink if its distance to the segment a-b is at most the
  brush radius, and the closest point of the segment comes from a **projection**:

```
t = clip( (p - a) . (b - a) / |b - a|^2 , 0, 1 )        closest point = a + t (b - a)
```

  The 28x28 preview shows the preprocessed image, i.e. exactly what the model sees. Saved files
  are dark ink on white, like a scan, so they go through the same pipeline as photos.
- `score_samples.py`: parses `<label>_<nn>.png`, preprocesses everything, prints per-digit
  results and the misses, and saves a gallery of the preprocessed images with predictions.

### 5.4 My digits scored low: preprocessing or model?

My 50 digits (pencil photos, `personal/`) first scored **31/50**. The rule: find the cause
before changing anything.

**Step 1, look at what the model saw.** Thin, faint, broken strokes and amplified smudges. That
suggested preprocessing, but "looks different" isn't a diagnosis, so:

**Step 2, measure** (median per image, no model involved):

| | MNIST | my digits, first pipeline |
|---|---|---|
| total ink | 99.6 | **31.1** |
| ink pixels >= 0.9 | 49% | **4%** |
| stroke width | 2.8 px | 2.0 px |
| width / height | 0.80 | 0.47 |
| center-of-mass error | 0.40 px | 0.40 px |

Centering was fine; brightness and width were far outside anything in MNIST. The first contrast
step mapped the single darkest pixel to 1.0, so a pencil stroke ended up at 0.3-0.6.

**Step 3, is the model hurt by that?** MNIST *validation* digits made just as faint and thin:

```
unchanged                 98.3%
faint (x0.3)              92.1%
thinner + faint (x0.45)   90.9%
```

So faint/thin strokes alone cost ~7 points, which explains part of the gap, but not all of it
(62% is much lower). **Diagnosis: preprocessing is partly to blame; the rest is probably the model.**

**Step 4, fix preprocessing, aiming at MNIST statistics, not at my score.** Median-stroke contrast
and pen-width normalization (steps 3 and 5 above). Afterwards: ink 86, saturated 45%, width 3.3 px,
close to MNIST. Result: **39/50**, without touching the model (6s went 1/5 -> 5/5, 9s 2/5 -> 5/5).
The stain filter came next (38/50: one digit flipped each way, i.e. no measurable change; I kept
it because it removes the isolated blobs it targets, verified on the images).

**Step 5, the rest is the model.** The remaining misses follow my style: narrow, strongly slanted,
flat-topped 3s, crossed 7s. MNIST validation digits squeezed and slanted the same way:

```
narrower (x0.6)               94.1%
slanted (shear 0.3)           96.8%
narrower (x0.6) + slanted     92.5%
```

A perfectly ordinary MNIST digit, just narrow and slanted, loses 6 points. The model has learned
MNIST's geometry, not "digits".

**A caveat I kept in mind:** these 50 digits now steered several decisions, so they behave more
like a development set than a test set. Every fix was chosen to match an MNIST property, never
to raise the score, and the final number comes from a *fresh* set (section 5.7).

### 5.5 Data augmentation: teaching the model handwriting variation

`digits/augment.py`. During training every batch is randomly deformed, so each epoch sees new
variants of the 55,000 digits. The label is unchanged: a slanted 7 is still a 7.

**Random affine warp.** For each image a 2x2 matrix M and a shift t (in (row, col) coordinates):

```
M = R(theta) @ S(k) @ D(sy, sx)                                         (digits/augment.py:65)

D = [[sy, 0], [0, sx]]          sx in [0.7, 1.1]: narrower or wider writing
S = [[1, 0], [-k, 1]]           k in [-0.3, 0.3]: slant (lower rows move left)
R = [[cos, -sin], [sin, cos]]   theta in [-12, 12] degrees
t in [-2, 2] px
```

Matrices compose right to left: scale first, then slant, then rotate. A point p of the input
moves to `q = M (p - c) + c + t`, pivoting around the center c = (14, 14).

**Warping means going backwards.** Pushing every input pixel forward would leave holes and
overlaps. Instead, for every *output* pixel q, ask where it came from and read the input there
(`affine_warp`, lines 27-28):

```
p = M^-1 (q - c - t) + c          (B,2,2) @ (B,2,784) -> (B,2,784)   all pixels of all images at once
```

`np.linalg.inv` inverts all B matrices in one call. p is usually between pixels, so the value is
blended from the 4 neighbours (**bilinear interpolation**, line 44): first along the row with the
fractional column, then between the two rows with the fractional row.

**Making it fast.** The first version took 6.7 ms per batch (2.6 s per epoch). Two changes cut that
to 2.5 ms: float32 instead of float64, and a single **flat index** into the raveled batch instead
of 3-array fancy indexing (line 38):

```
index = b * 31*31 + row * 31 + col       neighbours: +1 (right), +31 (below), +32 (diagonal)
```

That is the row-major formula from milestone 1, applied to a (B,31,31) padded batch.

**Stroke thickness**: 25% of images get a half-strength 3x3 dilation (thicker pen), 15% a
half-strength erosion (thinner pen) (`random_thickness`, line 81).

**Results** (`uv run train_mlp.py --augment`, same MLP and optimizer, 30 epochs, validation
chosen on clean MNIST validation):

```
                       no augmentation    augmentation
train acc              100.0%             99.2%       (less memorizing)
validation acc         98.32%             98.88%
test acc (once)        98.42%             99.00%
time from scratch      28 s               92 s
my 50 digits           38/50              44/50       (same final preprocessing for both)
```

Augmentation helps MNIST *itself* too: it is a regularizer. The model can no longer memorize exact
pixel patterns, because they never repeat. (40 epochs reached 98.98% validation, but 30 epochs is
within seed noise of that and fits the 2-minute budget.)
### 5.6 Personalization: training on my own digits

Augmentation can deform MNIST's digits, but it can't invent shapes MNIST barely has (flat-topped
3s, crossed 7s). The direct fix: add my own 50 digits (`personal/`) to the training set.

**The imbalance problem.** 50 images next to 55,000 is 0.09% of the data: their gradient would
vanish in the average. So each personal digit is repeated R times per epoch
(`np.repeat(Xe, R, axis=0)`: (50,784) -> (50*R, 784)), and augmentation makes every copy a different
variant, so the model can't just memorize 50 pictures. Validation stays clean MNIST.

**Choosing R without a test set: 5-fold cross-validation** (`uv run train_mlp.py --personal-cv`).
Fold k holds out every `*_0k.png`, one sample of each digit. The model trains on MNIST plus the
other 40 and is scored on the 10 it never saw. After 5 folds, every personal digit has been scored
exactly once by a model that didn't train on it:

```
repeat   held-out personal digits   MNIST validation
   0              43/50                  98.58%
  20              47/50                  98.58%
  60              44/50                  98.44%
```

R = 20 helps on *unseen* personal digits at no cost on MNIST. R = 60 is worse again: 40 images
repeated 60 times start to dominate, and the model fits *those* drawings instead of my style. (With
50 digits, a difference of 3-4 is suggestive, not proof.)

Final personalized model (`uv run train_mlp.py --personal`, R = 20): **MNIST test 98.83%**
(measured once), 86 s from scratch.

### 5.7 The final, honest test: 50 new digits

My first 50 digits were used for diagnosis and then as training data, so they can't measure
anything anymore. I wrote **50 new digits** (`samples/`, 5 per digit) and scored them **once**
with the finished personalized model. Two checks came first, without looking at the images:

- no file is a byte-for-byte copy of a training digit (a first attempt was exactly that: the
  `personal/` files copied back, caught by comparing SHA-256 hashes);
- no new digit is a re-saved near-copy: after preprocessing, each new digit's best correlation
  with any training digit (median 0.845) is the same as between different training digits (0.845).

```
uv run score_samples.py --model models/mlp_personal.npz
43 of 50 correct (86.0%)

digit    0  1  2  3  4  5  6  7  8  9
correct  3  4  5  4  5  5  5  4  5  3     (of 5 each)
```

**43/50: below the 45/50 target.** I did not tune anything after seeing this, because that would
turn this set into a training set too. The misses (`reports/samples_scored.png`):

- **style the model still doesn't know (5):** a 1 with a long flag -> 4, a flat-topped 3 -> 7, a
  crossed 7 -> 3, a 9 with an open top -> 4, a heavily retraced 0 (two overlapping loops) -> 8;
- **preprocessing (2):** an extra stroke attached to a 0 (the stain filter only removes blobs that
  *don't* touch the digit) -> 7; a faint, broken 9 -> 8.

**The whole journey, with the numbers each step was measured on:**

| step | what changed | my first 50 | new 50 |
|---|---|---|---|
| milestone-3 MLP, first preprocessing | | 31/50 | |
| preprocessing matched to MNIST statistics | contrast, pen width, stains | 38/50 | |
| + augmentation | affine warps, thickness | 44/50 | |
| + personalization (5-fold, held out) | my digits in training | 47/50* | |
| **final model, fresh digits, scored once** | | | **43/50** |

\* each digit scored by a model that didn't train on it, but these digits had also guided the
preprocessing fixes, so this number is optimistic. The drop from 47 to 43 on genuinely new digits
is what that optimism looks like. That's why the fresh set exists.

**What would help next (not done here):**
- more personal training data: 5 examples per digit is very little for a style the model has
  never seen;
- a model that is less sensitive to exact pixel positions: a CNN (milestone 6) shares the same
  stroke detectors across positions instead of learning each position separately.

### Check yourself

1. The milestone-3 MLP drops from 98.3% to 86.4% when MNIST digits move 2 px to the right. Using
   what milestone 4 showed about W1, explain why. Which preprocessing step protects against this,
   and why does it use the center of *mass* rather than the center of the bounding box?
2. In `affine_warp`, why does the code compute `p = M^-1 (q - c - t) + c` for every *output* pixel
   q, instead of moving every input pixel forward with `M`? What would go wrong with the forward
   version?
3. The 5-fold check said 47/50 and the fresh set said 43/50. Both measure the same model on digits
   it never trained on. Give two reasons the first number is likely too high.
---

## Milestone 6: A small CNN with im2col

Run it: `uv run train_cnn.py` (also `--sweep`, `--val-only`, `--shift`). Code: `digits/conv.py`
(im2col, col2im, Conv2D, MaxPool2, Reshape), `CNN` in `digits/models.py`, `train_cnn.py`; tests in
`tests/test_conv.py`.

### 6.1 Convolution: one small template, slid everywhere

An MLP unit has one weight per pixel: a template pinned to one position (milestone 4), which is
why a 2-px shift cost the MLP 12 points (section 5.1). A **convolution filter** is a small template,
here 5x5, applied at *every* position:

```
Z[f, i, j] = b[f] + sum_{c, u, v} X[c, i+u, j+v] * W_f[c, u, v]        u, v in -2..2
```

Each output value is the **dot product** of a filter with the 5x5 patch around pixel (i, j), the
same "template matching" as before, just local. The output Z[f] is a **feature map**: "how much
does each place in the image look like filter f".

- **Weight sharing**: the same 25 numbers are reused at all 784 positions. conv1 has 208
  parameters, against 401,408 for the MLP's W1.
- **Translation equivariance**: shift the input by one pixel and the feature map shifts by one
  pixel too, with the same values. A stroke detector works wherever the stroke is.

(Strictly speaking this is *cross-correlation*: true convolution flips the filter. Every
deep-learning library computes it this way and calls it convolution; since the filter is learned,
the flip doesn't matter.)

### 6.2 im2col: turning convolution into one matrix multiply

Looping over positions in Python would be hopelessly slow. The trick: copy every patch into a row
of a matrix, and then *all* dot products of *all* filters with *all* patches are one matrix product.

```
X      (B,C,H,W)                      input batch
cols   (B*H*W, C*k*k)                 row = the k x k patch (all channels) around one output position
W      (C*k*k, F)                     column = one filter, unrolled in the same order
Z      cols @ W + b                   (B*H*W, C*k*k) @ (C*k*k, F) -> (B*H*W, F)
       reshape/transpose              -> (B,F,H,W)
```

For conv1 on a batch of 128: (128*784, 25) @ (25, 8). For conv2: (128*196, 72) @ (72, 16).

`im2col` (`digits/conv.py:16-26`):

```python
Xp = np.pad(X, ((0,0), (0,0), (p,p), (p,p)))             # line 20: zero border, so the output keeps size H x W
win = sliding_window_view(Xp, (k, k), axis=(2, 3))         # line 23: (B,C,H,W,k,k), a VIEW: no copy
return win.transpose(0,2,3,1,4,5).reshape(B*H*W, C*k*k)    # line 26: positions first, patch last
```

`sliding_window_view` is milestone 1's strides idea taken further. It returns an array whose
extra two axes step through a k x k window, by giving those axes the *same* strides as rows and
columns. No data is copied; neighbouring windows simply overlap in memory. The final `reshape` has
to copy (overlapping windows can't be laid out flat without duplicating pixels), and that copy is
im2col's real cost: each pixel is stored k*k times (25x for conv1).

`Conv2D.forward` is then just `self.cols @ self.W + self.b` (line 62), exactly a `Linear` layer
applied to patches.

### 6.3 Backward: col2im is the transpose of im2col

Because forward is "im2col, then a Linear layer", backward is "Linear backward, then undo im2col"
(`Conv2D.backward`, lines 66-71):

```
Gm    = G reshaped (B,F,H,W) -> (B*H*W, F)          same row order as cols
dW    = cols^T @ Gm       (C*k*k, B*H*W) @ (B*H*W, F) -> (C*k*k, F)     sum over every position of every image
db    = Gm.sum(axis=0)    (B*H*W, F) -> (F,)
dcols = Gm @ W^T          (B*H*W, F) @ (F, C*k*k) -> (B*H*W, C*k*k)    gradient for every patch copy
dX    = col2im(dcols)     (B*H*W, C*k*k) -> (B,C,H,W)
```

`dW = cols^T @ Gm` is where weight sharing shows up in the math: one filter's gradient is the sum
of its gradients at all B*H*W positions, which is the matrix product.

**col2im** (lines 29-43): im2col copied each pixel into up to k*k patch rows, so the pixel's gradient
is the **sum** of the k*k gradient entries of its copies. That makes col2im the *transpose*
(adjoint) of im2col: im2col is a linear map, call it A, and for every linear map the backward
pass multiplies by A^T. `test_col2im_is_the_transpose_of_im2col` checks the defining identity

```
<A x, y> = <x, A^T y>         sum(im2col(X) * Y) == sum(X * col2im(Y))   for random X, Y
```

The implementation loops over the k*k kernel offsets (i, j) and adds one shifted slice each time
(line 42). Within one offset no two patches touch the same pixel, so plain `+=` is correct. (The
`np.add.at` trap from milestone 4 only bites when one assignment writes the same index twice.)

### 6.4 Max pooling

`MaxPool2` (lines 77-99) keeps the largest value in each 2x2 window: (B,C,28,28) -> (B,C,14,14).

```python
win = X.reshape(B, C, H//2, 2, W//2, 2)      # axes 3 and 5: position inside the window
          .transpose(0,1,2,4,3,5).reshape(B, C, H//2, W//2, 4)   # line 88: the 4 window entries last
self.idx = win.argmax(axis=-1)               # line 89: which entry was the max
```

The reshape is free (row-major again: splitting an axis of 28 into 14 x 2 doesn't move any
data). Backward sends each gradient only to the entry that was the max, since only that entry
affected the output (line 95, `np.put_along_axis` writes at the stored argmax). This is the same
idea as ReLU's mask. Ties (common after ReLU, where whole windows are 0) go to the first entry,
so the gradient is never counted twice.

Pooling halves the resolution, and it makes the network tolerant to small shifts: moving a stroke
by one pixel often keeps it inside the same 2x2 window, so the pooled output doesn't change at all.

### 6.5 The architecture

```
X (B,784) -> Reshape  (B,1,28,28)
conv1 5x5, 8 filters  (B,8,28,28)     W (25,8)       208 params   + ReLU
maxpool 2x2           (B,8,14,14)
conv2 3x3, 16 filters (B,16,14,14)    W (72,16)    1,168 params   + ReLU
maxpool 2x2           (B,16,7,7)
Reshape (flatten)     (B,784)
fc1                   (B,128)         W (784,128) 100,480 params  + ReLU
fc2                   (B,10)          W (128,10)    1,290 params
                                                  103,146 total (MLP: 407,050)
```

- conv2 looks at a 3x3 neighbourhood of *pooled* conv1 features. conv1 sees 5x5 pixels, a pooled
  unit 6x6 (in steps of 2 px), so a conv2 unit sees 6 + 2*2 = 10x10 pixels of the original image:
  layers stack small, local detectors into larger shapes.
- He init works unchanged: a conv output sums C*k*k inputs, so fan_in = C*k*k (25 for conv1, 72
  for conv2).
- Everything else (softmax cross-entropy, SGD with momentum, cosine schedule, early stopping,
  augmentation, save/load, gradient check) is reused from earlier milestones. `CNN` is just a
  different list of layers in `Sequential`.

**Gradient check** (`test_cnn_gradient_check`, float64, a small CNN with 2 and 3 filters and 8
hidden units, all 1,383 parameters, random biases): max relative error 3.0e-7 (fc1); conv1
8.5e-9, conv2 1.3e-8. The convolution forward pass is also checked against plain nested loops,
and its backward against finite differences.
### 6.6 Training

Same recipe as the augmented MLP: SGD with momentum 0.9, cosine schedule, batch 128, augmentation,
early stopping on clean MNIST validation (`train_cnn.py`). Learning rate from short validation
runs (`uv run train_cnn.py --sweep`, 5 epochs each):

```
lr      0.01     0.03     0.05     0.10
val     98.34%   98.70%   98.94%   98.78%
```

Short runs favour faster learning rates a little, but 0.05 is a clear peak, so lr = 0.05.

**Speed.** One epoch takes ~17 s (the MLP: ~3 s with augmentation). The arithmetic per image is
similar; what's expensive is memory traffic: im2col copies every pixel 25 times for conv1 (a
(100,352 x 25) matrix per batch), and the transposes between (B,F,H,W) and (B*H*W,F) layouts move
data around. To save time, training accuracy is measured on a fixed 5,000-image subset each epoch
(a full pass over 55,000 images would add ~8 s per epoch); validation is always the full 5,000.

### 6.7 Results

```
uv run train_cnn.py
  epoch   1  train loss 0.0848 acc 0.9742  val loss 0.0908 acc 0.9718  lr 0.0500
  ...
  epoch  13  train loss 0.0214 acc 0.9948  val loss 0.0355 acc 0.9904  lr 0.0048
  epoch  15  ...  restored weights from epoch 13 (val acc 0.9904)

Kept epoch 13:  val acc 0.9904   test acc 0.9925 (loss 0.0222)
Training time 260s, total 283s
```

**Test 99.25%** (target >= 99%), measured once.

| | MLP (M3) | MLP + aug (M5) | **CNN + aug** |
|---|---|---|---|
| parameters | 407,050 | 407,050 | **103,146** |
| validation | 98.32% | 98.88% | **99.04%** |
| test (once each) | 98.42% (158 errors) | 99.00% (100) | **99.25% (75)** |
| training time | 28 s | 92 s | 283 s |

A quarter of the parameters and a quarter fewer errors than the augmented MLP. The CNN is told
something true about images (nearby pixels belong together, and a stroke means the same thing
anywhere), so it doesn't have to learn that 784 separate times.

**What the layers learned** (`reports/cnn_conv1_filters.png`, `reports/cnn_feature_maps.png`):
- conv1 filters are small **oriented edge detectors**: a red band next to a blue band means "ink
  here, none there" in a given direction (filter 6: the top edge of a horizontal stroke; filters 2
  and 5: diagonal edges). These are local versions of the red/blue pairs in the MLP's templates
  (milestone 4), but now one detector covers the whole image.
- conv1 feature maps of a 6: each map lights up a different part (the diagonal stem, the bottom
  of the loop, horizontal segments).
- conv2 maps (14x14, after pooling) are sparse: a few units fire where a particular combination
  of edges occurs, i.e. small stroke parts at a coarser scale.

### 6.8 Shifted digits: the experiment that motivated the CNN

`uv run train_cnn.py --shift` (MNIST validation only):

```
shift (px)              0        1        2        3        4
MLP (milestone 3)     98.3%    96.9%    86.4%    62.6%    37.8%
MLP + augmentation    98.9%    98.8%    98.4%    96.3%    86.0%
CNN + augmentation    99.0%    99.0%    98.8%    98.2%    95.4%
```

(`reports/shift_robustness.png`)

- The augmented MLP is robust up to 2 px *because it was trained on shifts up to 2 px*: it has
  seen and memorized those positions. Beyond its training range it falls off (86% at 4 px).
- The CNN degrades much more slowly outside that range (95.4% at 4 px). Its conv filters detect
  strokes at any position by construction (weight sharing), and pooling absorbs small offsets.
  Only the final fully connected layers are position-specific, and they see a 7x7 map of coarse
  features rather than 784 pixels.

### 6.9 My handwriting

The 50 fresh digits (`samples/`), scored **once** with the finished CNN, which was tuned only on
MNIST validation and never saw any of my digits:

```
uv run score_samples.py --model models/cnn.npz --out reports/samples_scored_cnn.png
44 of 50 correct (88.0%)
digit    0  1  2  3  4  5  6  7  8  9
correct  4  4  4  4  5  5  4  5  5  5
```

| model on the fresh 50 | my digits in training? | score |
|---|---|---|
| MLP + augmentation + personalization (M5) | yes, `personal/` | 43/50 |
| **CNN + augmentation** | no | **44/50** |

Without a single example of my handwriting, the CNN does slightly better than the personalized MLP,
and it fixes exactly the style errors the MLP made (the flat-topped 3, the crossed 7, the flagged
1, the open 9). Its misses are different digits: 1_10 -> 4, 2_10 -> 7, 3_10 -> 7, 5_09 -> 1,
6_10 -> 5, plus the same 0 with an attached stroke (0_08 -> 7) that preprocessing can't clean.

**Still one short of 45/50, so that box stays unticked.** And a caution about comparing the two rows:
the fresh 50 have now been used twice, once per finished model, each time without any tuning. That
is fine for reporting each model, but picking the *better* of the two by this score would be
selecting on the test set; with 50 digits, 43 vs 44 is one digit and not a real difference.

### Check yourself

1. conv2 has C = 8 input channels, k = 3 and F = 16 filters, on a batch of 128 images of size 14x14.
   What are the shapes of `cols`, `W` and `cols @ W`? Roughly how many times is each input value
   copied into `cols`, and why does that make im2col memory-hungry?
2. In `Conv2D.backward`, why is `dW = cols.T @ Gm` a sum over every position of every image, and
   what does that have to do with weight sharing? Why must `col2im` *add* the patch gradients
   instead of writing them?
3. In the shift experiment, the augmented MLP is nearly as robust as the CNN at 1-2 px but much
   worse at 4 px. Explain both halves, using what the augmentation covered and what convolution and
   pooling do.
### 6.10 Personalized CNN and a third fresh set

The second handwriting set (`*_06`-`*_10`) had now been scored twice (personalized MLP, CNN), so it
was no longer a clean test set. It joined the first set in `personal/` as training data: 100 of my
digits, 10 per digit. I wrote a **third set** of 50 (`samples/`, `*_11`-`*_15`) for the final test.

**Personalized CNN** (`uv run train_cnn.py --personal`): the same CNN and recipe, with each of the
100 personal digits repeated 20 times per epoch (2,000 extra images next to 55,000). The repeat
factor comes from the MLP's 5-fold check (section 5.6). Repeating that check for the CNN would
take about an hour, so it is reused rather than re-validated.

```
Kept epoch 13:  val acc 0.9908   test acc 0.9923 (loss 0.0235)        (test measured once)
Training time 267s, total 290s
```

MNIST accuracy is unchanged (plain CNN: 99.25% test), so the personal digits cost nothing there.

**Before scoring**, without looking at the images:
- no new photo is a byte copy of a training digit (SHA-256), and none is a re-saved near-copy:
  each new digit's best correlation with any training digit (median 0.857) is no higher than
  between different training digits (0.868);
- the tools' default model was switched to the personalized CNN *before* the score was known,
  so the choice couldn't be influenced by it.

**The score**, measured once (`uv run score_samples.py --out reports/samples_scored_set3.png`):

```
45 of 50 correct (90.0%)
digit    0  1  2  3  4  5  6  7  8  9
correct  5  5  5  5  5  3  4  5  3  5
```

**45/50: the target is met**, exactly at the threshold. The misses (`reports/samples_scored_set3.png`):
- style (2): a 5 whose top bar and stem read as a 7 (5_15 -> 7), an 8 read as 7 (8_11);
- preprocessing (3): 6_15 -> 5, 5_13 -> 1 and 8_14 -> 1 lost faint parts of their strokes during
  preprocessing (most of the 6's stem is gone). The likely cause is the 0.5 "solid ink" threshold
  behind the stain filter (section 5.2, step 3). That's the next thing to improve, but it wasn't
  changed after seeing this set.

**How much should one trust 45/50?** If the true accuracy on my handwriting is about 90%, the
number of correct digits out of 50 behaves like a binomial count with standard deviation
sqrt(50 * 0.9 * 0.1), about 2.1 digits. Another fresh set of 50 could easily give 43 or 47. "At least
45/50" was met on this set; a 45 says the model reads my writing at roughly 85-95%, not exactly 90%.

**All handwriting results, each from a single scoring run:**

| model | trained on my digits | set 1 (first 50) | set 2 | set 3 |
|---|---|---|---|---|
| MLP, first preprocessing (M3) | no | 31 | | |
| MLP, final preprocessing (M3) | no | 38 | | |
| MLP + augmentation | no | 44 | | |
| MLP + aug + personal (set 1) | set 1 | (47 in 5-fold) | 43 | |
| CNN + augmentation | no | | 44 | |
| **CNN + aug + personal (sets 1-2)** | sets 1-2 | | | **45** |

Read across a row, never down a column: set 1 steered the preprocessing fixes, and a set stops
being a test set once it has been used to make a decision.

### Check yourself (section 6.10)

1. Why did set 2 have to move to `personal/` before set 3 was scored, even though it was never used
   to train anything?
2. A friend's model gets 46/50 on their own fresh set and yours gets 45/50. Using the binomial
   standard deviation above, is theirs clearly better?
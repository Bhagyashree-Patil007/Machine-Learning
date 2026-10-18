# 🚀 Types of Gradient Descent — Caveman Style Notes

---

## 📑 Table of Contents

| #   | Topic                                      | One-line idea                             |
| --- | ------------------------------------------ | ----------------------------------------- |
| 1   | Types of Gradient Descent                  | 3 flavours: Batch, Stochastic, Mini-batch |
| 2   | Batch GD — Math (n-dimensions)             | Slope for every β, update all together    |
| 3   | Batch GD — Matrix / Vector form            | Use `np.dot` instead of loops             |
| 4   | Batch GD — Code (`GDRegressor`)            | Build our own regressor                   |
| 5   | Problem with Batch GD                      | Slow on big data + needs lots of RAM      |
| 6   | Stochastic GD                              | 1 row ➜ 1 update                          |
| 7   | SGD — Code                                 | Random row, update, repeat                |
| 8   | Time Comparison                            | Batch vs SGD per epoch                    |
| 9   | Visualization                              | Smooth path vs jumpy path                 |
| 10  | When to use SGD                            | Big data + non-convex loss                |
| 11  | Learning Schedules                         | Shrink `η` as training goes on            |
| 12  | Sklearn — `SGDRegressor`                   | Ready-made SGD                            |
| 13  | Mini-Batch GD                              | A small group of rows ➜ 1 update          |
| 14  | Mini-Batch — Code + Sklearn                | Batch size is the dial                    |
| 15  | Extra Theory + Cheat Sheet + Interview Q&A | Revision                                  |

---

## 1️⃣ Types of Gradient Descent

Gradient Descent has **3 types**. Only one thing differs: **how many rows are used for ONE update**.

```
                    Gradient Descent
             ┌───────────┼───────────────┐
         Batch GD   Stochastic GD    Mini-Batch GD
        (ALL rows)    (1 row)       (small group of rows)
```

| Type                       | Rows used per update               | Updates per epoch  |
| -------------------------- | ---------------------------------- | ------------------ |
| 🟦 **Batch GD**            | **All `n`** rows                   | **1**              |
| 🟥 **Stochastic GD (SGD)** | **1** row                          | **n**              |
| 🟩 **Mini-Batch GD**       | **batch_size** rows (e.g. 30, 100) | **n / batch_size** |

### Common recipe (same for all 3)

```
start:   m = 1, b = 0          (random start)
repeat:  m = m − η × slope_m
         b = b − η × slope_b
```

- Only the **slope** changes (it is computed on all rows / 1 row / a batch).
- Any dataset with `n` columns has parameters `{β₀, β₁, β₂ … βₙ}` → GD works the same way in **n-dimensions**.

🧠 **Mnemonic — "ALL, ONE, SOME":** Batch = **ALL**, Stochastic = **ONE**, Mini-batch = **SOME**.

---

## 2️⃣ Batch Gradient Descent — Mathematical Formulation

Dataset with **3 columns**: `cgpa | iq | lpa`

| x₁ (cgpa) | x₂ (iq) | y (lpa) |
| --------- | ------- | ------- |
| 8.1       | 93      | 3.2     |
| 7.5       | 95      | 3.5     |
| …         | …       | …       |

### Equation

```
ŷ = β₀ + β₁·x₁ + β₂·x₂
```

- Total parameters = **(n + 1)** → `β₀ … βₙ` (n columns + 1 intercept).
- Here: `β₀, β₁, β₂` → 3 parameters.

### Algorithm

```
1) Random values:   β₀ = 0 , β₁ = β₂ = 1
2) epochs = 100 , lr = 0.1 (η)
   for i in range(epochs):
       β₀ = β₀ − η × ∂L/∂β₀
       β₁ = β₁ − η × ∂L/∂β₁
       β₂ = β₂ − η × ∂L/∂β₂
```

⚠️ Update **all β together** (compute all slopes first, then update).

### Loss function (MSE)

```
L = (1/n) Σ (yᵢ − ŷᵢ)²
```

Example with **n = 2 rows**:

```
L = 1/2 [ (y₁ − β₀ − β₁x₁₁ − β₂x₁₂)² + (y₂ − β₀ − β₁x₂₁ − β₂x₂₂)² ]
```

### Slope w.r.t. β₀ (intercept)

```
∂L/∂β₀ = 1/2 [ 2(y₁ − ŷ₁)(−1) + 2(y₂ − ŷ₂)(−1) ]
       = −(2/2) [ (y₁ − ŷ₁) + (y₂ − ŷ₂) ]

For n rows:
∂L/∂β₀ = −(2/n) Σ (yᵢ − ŷᵢ)
```

### Slope w.r.t. β₁ (needs extra × x)

Derivative of `−β₁·x₁₁` w.r.t. β₁ is `−x₁₁`, so that column's value comes out:

```
∂L/∂β₁ = −(2/n) Σ (yᵢ − ŷᵢ) · xᵢ₁
∂L/∂β₂ = −(2/n) Σ (yᵢ − ŷᵢ) · xᵢ₂
```

### 🌟 General formula (any column m)

```
∂L/∂βₘ = −(2/n) Σ (yᵢ − ŷᵢ) · xᵢₘ
```

- `xᵢₘ` = value of **column m** in **row i**.
- `βₘ` ↔ column m (1 β for each column).

| Parameter      | Slope                    | Extra term |
| -------------- | ------------------------ | ---------- |
| β₀ (intercept) | `−(2/n) Σ (yᵢ − ŷᵢ)`     | none       |
| β₁             | `−(2/n) Σ (yᵢ − ŷᵢ)·xᵢ₁` | × column 1 |
| β₂             | `−(2/n) Σ (yᵢ − ŷᵢ)·xᵢ₂` | × column 2 |
| βₘ             | `−(2/n) Σ (yᵢ − ŷᵢ)·xᵢₘ` | × column m |

---

## 3️⃣ Batch GD — Matrix (Vector) Form 🧮

Writing separate loops for 10 columns is slow and ugly. Use **matrix multiplication** (`np.dot`).

### Step A: Prediction for all rows at once

```
ŷ = np.dot(X_train, coef_) + intercept_
```

Shapes (diabetes dataset, 353 train rows, 10 columns):

```
X_train   : (353, 10)
coef_     : (10,)   → treat as (10, 1)
np.dot    : (353, 10) · (10, 1) = (353, 1)
+ β₀      : added to every row   → ŷ shape (353, 1)
```

### Step B: Slopes of ALL coefficients at once

```
∂L/∂β₁ … ∂L/∂β₁₀ = [ (y − ŷ)ᵀ · X_train ] × (−2/n)
```

Shapes:

```
(y − ŷ)ᵀ : (1, 353)
X_train  : (353, 10)
result   : (1, 10)     ← one slope for each of the 10 coefficients ✅
```

### 🧪 Tiny hand example (verify yourself)

```
X₁ = [1, 3]     X₂ = [2, 4]     y = [5, 7]     ŷ = [6, 8]     n = 2
y − ŷ = [−1, −1]

(y−ŷ)·X₁ = (−1×1) + (−1×3) = −4   →  × (−2/2) =  4
(y−ŷ)·X₂ = (−1×2) + (−1×4) = −6   →  × (−2/2) =  6

slopes = [4, 6]
```

> 📝 Class slide arithmetic on this example is a bit scribbled; the numbers above are recalculated cleanly.

🧠 **Rule:** `(y − ŷ)` is a **row vector**, `X` is the **data matrix**, one `np.dot` gives **all slopes**.

---

## 4️⃣ Batch GD — Code (`GDRegressor`)

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

```python
class GDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        # init
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # prediction for ALL rows
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            # slope of intercept
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

            # slope of all coefficients (matrix form)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

```python
gdr = GDRegressor(epochs=1000, learning_rate=0.5)
gdr.fit(X_train, y_train)
y_pred = gdr.predict(X_test)
r2_score(y_test, y_pred)

# compare with sklearn
lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_, lr.intercept_)
```

✅ If our `coef_` and `intercept_` are close to sklearn's → our implementation is correct.

> 💡 Increase `epochs` / `learning_rate` until the R² gets near sklearn's R².

---

## 5️⃣ The Problem with Batch GD 😩

Think: `n = 1,00,000` rows, `5` columns (6 coefficients), `50` epochs.

```
Per epoch  → every update touches ALL 1,00,000 rows
Total work → roughly 10^10 calculations   ➜  VERY SLOW on big data 🐌
```

| #   | Problem               | Meaning                                                                  |
| --- | --------------------- | ------------------------------------------------------------------------ |
| 1   | ⏱️ **Slow**           | Only **1 update per epoch** and each update scans the **entire** dataset |
| 2   | 💾 **Hardware / RAM** | Whole dataset must sit in memory at once. Impossible for huge data       |

Deep learning makes it worse:

- **CNN** → images (millions of pixels)
- **RNN** → text / sequences
- Millions of rows × thousands of parameters ➜ Batch GD is **not practical**.

➡️ **Solution = Stochastic GD** (and Mini-batch GD).

---

## 6️⃣ Stochastic Gradient Descent (SGD) 🎲

**"Stochastic" = random.**

### Idea

```
Batch GD :  all rows ──► 1 update   (1 epoch = 1 update)
SGD      :  1 row    ──► 1 update   (1 epoch = n updates)
```

### Steps (per epoch)

```
for each epoch:
    repeat n times:
        pick 1 RANDOM row
        calculate slope using only that row
        update coef_ and intercept_
```

### Key points

- **1 epoch = n updates** (because n rows).
- Row is picked **randomly** (so path is _noisy / zig-zag_).
- ✅ **Faster convergence** (reaches good values in fewer epochs).
- ⚠️ Does **not** give a perfectly steady solution; it **dances around** the minimum.

```
Batch GD path:           SGD path:
   \                        \
    \                        \ /\
     \                        /  \/\
      \                       \ /\  \/   (jumpy but fast)
       ●                         ●
   smooth & steady           noisy but quick
```

### Formula for 1 row `i`

```
∂L/∂β₀ = −2 (yᵢ − ŷᵢ)
∂L/∂βₘ = −2 (yᵢ − ŷᵢ) · xᵢₘ
```

(no sum and no `1/n`, because only 1 row)

---

## 7️⃣ SGD — Code

```python
class SGDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):                 # epochs
            for j in range(X_train.shape[0]):        # n updates per epoch
                idx = np.random.randint(0, X_train.shape[0])   # random row

                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

```python
sgd = SGDRegressor(learning_rate=0.01, epochs=40)
sgd.fit(X_train, y_train)
```

- Notice the **extra inner loop** → that is the whole difference from Batch GD.
- SGD usually reaches a good result in **far fewer epochs** (e.g., 40 vs 1000).

---

## 8️⃣ Time Comparison ⏱️

Fix **epochs = 100**.

|             | Updates in 100 epochs | Work per update    |
| ----------- | --------------------- | ------------------ |
| 🟦 Batch GD | **100**               | Heavy (all rows)   |
| 🟥 SGD      | **100 × n**           | Very light (1 row) |

- SGD makes **many more** updates, but **each is tiny**.
- In practice, for the **same number of epochs SGD can look slower** in plain Python (many small loops), but it **converges in fewer epochs** and scales much better on **big data**.

🧠 **Remember:** compare **time to reach good accuracy**, not just time per epoch.

---

## 9️⃣ Visualization 📈

What you see when you plot the path of `(m, b)` / the loss:

|            | Batch GD                    | SGD                            |
| ---------- | --------------------------- | ------------------------------ |
| Path       | Smooth, straight to minimum | Zig-zag, noisy                 |
| Loss curve | Smooth decrease             | Jumps up & down, trending down |
| Final      | Settles exactly             | Hovers **around** the minimum  |

```
Loss
 |\
 | \   /\
 |  \_/  \  /\_/\_/\    ← SGD loss is noisy but goes down
 |        \/
 +----------------------- iterations
```

---

## 🔟 When to use Stochastic GD ✅

| #   | Situation                    | Why SGD                                                         |
| --- | ---------------------------- | --------------------------------------------------------------- |
| 1   | **Big data**                 | Cheap updates, no need to load all rows each step               |
| 2   | **Non-convex loss function** | Random noise helps **jump out of local minima / saddle points** |

```
Convex (like linear regression MSE)      Non-convex (deep learning)
     \       /                              /\      /\
      \     /                              /  \    /  \
       \___/   ← 1 global minimum         /    \__/    \  ← many valleys
                                        local   global
```

- **Convex** → only 1 global minimum, Batch GD is fine on small data.
- **Non-convex** → many valleys; Batch GD may get stuck, SGD's randomness helps escape.

---

## 1️⃣1️⃣ Learning Schedules 🗓️

Problem: with a **constant** learning rate, SGD keeps **jumping around** the minimum and never settles.

**Fix:** start with a **big** `η` (fast progress), then **reduce** it slowly (fine settling). This is called a **learning schedule** (learning-rate decay).

```
η
|\
| \
|  \__
|     ‾‾‾──────      ← η shrinks as training goes on
+------------------ time (t)
```

### Formula used in class

```
t0, t1 = 5, 50

def learning_rate(t):
    return t0 / (t + t1)

for i in range(epochs):
    for j in range(X.shape[0]):
        lr = learning_rate(i * X.shape[0] + j)     # t = total updates so far
        ...
```

- `t = i × n + j` = number of updates done till now.
- Example: `t = 0` → `lr = 5/50 = 0.1`; `t = 100` → `lr = 5/150 ≈ 0.033`.

| Time      | Learning rate                          |
| --------- | -------------------------------------- |
| Beginning | **High** (0.1) → fast steps            |
| Later     | **Low** (0.03 …) → tiny steps, settles |

---

## 1️⃣2️⃣ Sklearn Implementation — `SGDRegressor`

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=100, learning_rate='constant', eta0=0.01)
sgd.fit(X_train, y_train)

print(sgd.coef_)
print(sgd.intercept_)

y_pred = sgd.predict(X_test)
r2_score(y_test, y_pred)
```

| Parameter       | Meaning                                                            |
| --------------- | ------------------------------------------------------------------ |
| `max_iter`      | Number of epochs                                                   |
| `learning_rate` | `'constant'`, `'optimal'`, `'invscaling'` (schedule), `'adaptive'` |
| `eta0`          | Starting learning rate                                             |
| `penalty`       | Regularization (default `'l2'`)                                    |
| `tol`           | Stop early if improvement is smaller than this                     |

---

## 1️⃣3️⃣ Mini-Batch Gradient Descent 🧺

**Middle path** between Batch and SGD.

```
Batch GD  : 1 update / epoch   (slow, steady)      batch_size = n
SGD       : n updates / epoch  (fast, noisy)       batch_size = 1
Mini-batch: n / batch_size updates / epoch         1 < batch_size < n
```

### Example

```
n = 1000 rows
batch_size = 100
number of batches = 1000 / 100 = 10
➜ 10 updates per epoch
```

### Steps

```
for each epoch:
    split data into batches (random groups of rows)
    for each batch:
        calculate slope using that batch only
        update coef_ and intercept_
```

### Comparison table

| Feature              | 🟦 Batch           | 🟥 Stochastic        | 🟩 Mini-Batch                            |
| -------------------- | ------------------ | -------------------- | ---------------------------------------- |
| Rows per update      | n                  | 1                    | batch_size                               |
| Updates / epoch      | 1                  | n                    | n / batch_size                           |
| Speed                | Slow               | Fast                 | **Fast**                                 |
| Stability            | Very steady        | Very noisy           | **Fairly steady**                        |
| Best for             | Small, convex data | Big data, non-convex | **Big data + deep learning (most used)** |
| Memory               | All data in RAM    | Tiny                 | Small                                    |
| Escapes local minima | ❌ Hard            | ✅ Yes               | ✅ Yes                                   |

---

## 1️⃣4️⃣ Mini-Batch — Code + Sklearn

```python
import random

class MBGDRegressor:

    def __init__(self, batch_size, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(int(X_train.shape[0] / self.batch_size)):   # n / batch_size updates

                idx = random.sample(range(X_train.shape[0]), self.batch_size)   # random batch

                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx]) / self.batch_size
                self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

```python
mbr = MBGDRegressor(batch_size=int(X_train.shape[0] / 50), learning_rate=0.01, epochs=100)
mbr.fit(X_train, y_train)
```

### Sklearn (mini-batch with `partial_fit`)

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(learning_rate='constant', eta0=0.1)
batch_size = 35

for i in range(100):                               # epochs
    idx = random.sample(range(X_train.shape[0]), batch_size)
    sgd.partial_fit(X_train[idx], y_train[idx])    # 1 update with a batch
```

> 💡 In sklearn, `SGDRegressor` + `partial_fit` on a batch = Mini-batch GD.

### 📊 Visualization

- Path is **less noisy than SGD** and **faster than Batch** → best of both worlds.

---

### 🧠 Why does random sampling still work?

- A single row (or batch) gives a **noisy estimate** of the true gradient.
- On **average**, it points in the **right direction** (unbiased estimate).
- Many noisy-but-correct steps ≈ one accurate step, but far cheaper.

### 🎚️ Batch size — the dial

| Batch size      | Effect                                                    |
| --------------- | --------------------------------------------------------- |
| Small (1–32)    | Noisy, good generalization, many updates                  |
| Medium (64–256) | Common choice, good balance                               |
| Large (1000+)   | Smooth, needs more memory, may converge to sharper minima |

Common sizes: **32, 64, 128, 256** (powers of 2 suit GPU memory).

### 🔁 Epoch vs Iteration vs Batch

| Term          | Meaning                               |
| ------------- | ------------------------------------- |
| **Epoch**     | One full pass over the whole dataset  |
| **Batch**     | Group of rows used for one update     |
| **Iteration** | One update step (one batch processed) |

```
iterations per epoch = n / batch_size
Example: n = 1000, batch_size = 100  → 10 iterations = 1 epoch
```

### 🛑 Always do

- ✅ **Feature scaling** before any GD (StandardScaler / MinMaxScaler)
- ✅ **Shuffle** data each epoch for SGD / Mini-batch
- ✅ Plot **loss vs epoch**; it should trend down
- ✅ Try `η` in `0.001, 0.01, 0.1` and compare
- ✅ Compare final coefficients with `LinearRegression` (sanity check)

### 🚀 Beyond these 3 (next to learn)

- **Momentum**, **RMSProp**, **Adam** → smarter versions of SGD used in deep learning
- **Regularization (Ridge, Lasso)** → reduce overfitting

---

## 🧠 Master Cheat Sheet

```
Prediction  : ŷ = np.dot(X, coef_) + intercept_
Slope β₀    : −(2/n) Σ (y − ŷ)
Slope βₘ    : −(2/n) Σ (y − ŷ)·xₘ          (vector: (y−ŷ)ᵀ · X × −2/n)
Update      : β = β − η × slope

Batch GD    : use ALL rows       → 1 update / epoch
SGD         : use 1 row          → n updates / epoch
Mini-batch  : use batch_size rows → n/batch_size updates / epoch
Schedule    : lr = t0 / (t + t1)   (t0=5, t1=50)
```

🧠 **Mnemonic — "BIG SLOW, ONE FAST, SOME BEST":**
Batch = big & slow, Stochastic = one & fast but noisy, Mini-batch = some & best balance.

### 🎤 Interview Quick Q&A

| Question                          | Short answer                                                                |
| --------------------------------- | --------------------------------------------------------------------------- |
| Batch vs SGD vs Mini-batch?       | All rows vs 1 row vs small group per update                                 |
| Why is Batch GD slow on big data? | Every update scans all rows; only 1 update per epoch; needs all data in RAM |
| Why is SGD noisy?                 | Gradient from 1 random row is only an approximation                         |
| Why can SGD escape local minima?  | Noise pushes it out of shallow valleys                                      |
| What is a learning schedule?      | Decreasing `η` over time so SGD settles                                     |
| What is an epoch?                 | One full pass through the dataset                                           |
| Updates per epoch?                | Batch = 1, SGD = n, Mini-batch = n / batch_size                             |
| Which is used in deep learning?   | **Mini-batch** (usually with Adam etc.)                                     |
| Does SGD always beat Batch?       | No. On small convex data Batch is fine and steadier                         |
| Sklearn class for SGD?            | `SGDRegressor` (and `partial_fit` for mini-batch)                           |

---

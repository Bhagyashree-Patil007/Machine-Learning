# 🏔️ Gradient Descent —

---

## 📑 Table of Contents

| #   | Topic                     | One-line idea                                       |
| --- | ------------------------- | --------------------------------------------------- |
| 1   | What is Gradient Descent? | Walk downhill to find the lowest point              |
| 2   | The Plan                  | Why we need it for Linear Regression                |
| 3   | Intuition                 | Use slope to decide which way to move               |
| 4   | Mathematical Formulation  | `b_new = b_old − η × slope`                         |
| 5   | Example                   | Step by step numbers with `b = −10`                 |
| 6   | Code from Scratch         | Loop: slope → update → repeat                       |
| 7   | Visualization 1           | Watch line, loss, `b` move per epoch                |
| 8   | Few Discussions           | Learning rate + universality                        |
| 9   | Adding `m` into the mix   | Two parameters → partial derivatives                |
| 10  | Code (m and b)            | `GDRegressor` class                                 |
| 11  | Visualization 2           | 3D bowl of loss                                     |
| 12  | Effect of Learning Rate   | Too small / just right / too big                    |
| 13  | Effect of Loss Function   | Convex, local min, plateau, saddle                  |
| 14  | Effect of Data            | Feature scaling matters                             |
| 15  | Extra Theory              | Batch / SGD / Mini-batch, cheat sheet, interview Qs |

---

## 1️⃣ What is Gradient Descent?

- **Gradient Descent = optimization algorithm.**
- It is **first-order** (uses only first derivative = slope).
- Job: find the **local minimum** of a **differentiable function**.
- Idea: take small repeated steps in the **opposite direction of the gradient**.
- Why opposite? Gradient points **uphill** (steepest ascent). Opposite = **steepest descent** = downhill.
- Go **along** the gradient instead ➜ you climb to a **local maximum** (that is called _gradient ascent_).
- History: Cauchy suggested it in **1847**, Hadamard similar in **1907**, Haskell Curry studied non-linear case in **1944**.
- Also called **Steepest Descent**.

```
   Loss
    |  \                /
    |   \              /
    |    \            /
    |     \          /
    |      \___.____/      <-- we want to reach this bottom
    |          ^
    +-------------------------- parameter (b)
```

🧠 **Mnemonic — "BALL IN BOWL":** ball rolls down the bowl until it stops at the bottom.

---

## 2️⃣ The Plan (Why do we need it?)

In Linear Regression we want the best line:

```
ŷᵢ = m·xᵢ + b
```

Two ways to find best `m` and `b`:

| Method                   | How                                | Problem                                                   |
| ------------------------ | ---------------------------------- | --------------------------------------------------------- |
| 🟦 **OLS (Closed form)** | Direct formula, one shot           | Gets very slow / heavy with many columns (matrix inverse) |
| 🟩 **Gradient Descent**  | Start random, improve step by step | Needs learning rate + many epochs                         |

- Example from class: `cgpa → lpa` data, OLS gave **m = 78.35**.
- **Trick for learning:** fix `m = 78.35`, and find only **`b`** with Gradient Descent first (1 unknown = easy to see).
- If GD gives the same `b` as OLS ➜ **GD works** ✅

### 📉 Loss Function (what we minimize)

```
L = Σ (yᵢ − ŷᵢ)²              (sum of squared errors)
L = Σ (yᵢ − m·xᵢ − b)²
```

- Bigger error ➜ bigger `L`.
- Best line = line with **minimum L**.

---

## 3️⃣ Intuition

Fix `m = 78.35`. Now `L` depends only on `b`:

```
L(b) = Σ (yᵢ − 78.35·xᵢ − b)²   ➜  L is like b²  ➜  a PARABOLA (U shape)
```

```
  L
  |   \                 /
  |    \               /      slope = −ve (left side)  ➜ move RIGHT
  |     \             /       slope = +ve (right side) ➜ move LEFT
  |      \___  _____/
  |          \/  <-- minimum (best b)
  +---------------------------- b
     b=−10                b=10
```

### 🔑 Three golden steps

1. **Pick a random `b`** (e.g., `b = −10`).
2. **Find the slope** at that point.
3. **Move opposite to slope.**

| Where you are                | Slope sign         | What to do to `b` |
| ---------------------------- | ------------------ | ----------------- |
| Left of minimum (`b = −10`)  | **Negative (−ve)** | **Increase** `b`  |
| Right of minimum (`b = +10`) | **Positive (+ve)** | **Decrease** `b`  |

### 🧮 One formula covers both cases

```
b_new = b_old − slope
```

- Slope −ve ➜ `− (−)` = `+` ➜ `b` goes **up** ✅
- Slope +ve ➜ `− (+)` = `−` ➜ `b` goes **down** ✅

### 🧪 Slope quick check (calculus)

```
y = x² + 2
dy/dx = 2x
at x = 5  ➜  slope = 10
```

**Slope = derivative.**

### ⚠️ Problem with `b_new = b_old − slope`

- If slope is big (like −50), the jump is big: `−10 − (−50) = 40` ➜ **overshoots** the minimum!
- Next jump goes back too far ➜ zig-zag, never settles.

### ✅ Fix = Learning Rate (η, "eta")

```
b_new = b_old − η × slope
```

- `η` is a small number like `0.1`, `0.01`, `0.001`.
- It **shrinks the step** so we move gently.

🧠 **Mnemonic — "SLOPE tells direction, ETA tells size".**

---

## 4️⃣ Mathematical Formulation

### Loss

```
L = Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)²
```

### Find slope of `L` with respect to `b`

```
dL/db = d/db [ Σ (yᵢ − m·xᵢ − b)² ]
      = 2 Σ (yᵢ − m·xᵢ − b) × (−1)         (chain rule: derivative of −b is −1)

slope = −2 Σ (yᵢ − m·xᵢ − b)
```

### Algorithm

```
1. Start with random b           (we take b = 0)
2. Choose  η = 0.01
3. Choose  epochs = 100 / 1000
4. for i in range(epochs):
       slope = −2 Σ (yᵢ − m·xᵢ − b)
       b     = b − η × slope
```

- **Epoch** = one full round of "find slope → update `b`".
- The part `η × slope` is the **step size**.

```
    b_old ──(slope)──► compute η × slope ──► b_new
      ▲                                        │
      └────────────── repeat (epoch) ◄─────────┘
```

---

## 5️⃣ Example (hand calculation)

Start: `b = −10`, slope = `−50`, `η = 0.01`

```
Epoch 1:
b_new = −10 − (0.01 × −50)
      = −10 + 0.5
      = −9.5

Epoch 2: (slope at −9.5 is −40)
b_new = −9.5 − (0.01 × −40)
      = −9.5 + 0.4
      = −9.1
```

Observe:

- `b` moved **−10 → −9.5 → −9.1** (towards the minimum).
- Steps get **smaller** automatically because **slope gets smaller** near the bottom. 🎯

### 🛑 When to stop?

| #   | Rule                                 | Meaning                   |
| --- | ------------------------------------ | ------------------------- | --------- | ---------------------------- |
| 1   | `                                    | b_new − b_old             | < 0.0001` | Almost no change ➜ converged |
| 2   | **Fixed epochs** (e.g., 1000 or 100) | Just run N times and stop |

- At the exact minimum: `b_new − b_old = 0` (slope = 0, no more movement).
- In practice most people use **epochs** (simple and safe).

---

## 6️⃣ Code from Scratch (only `b`)

```python
import numpy as np

# m is fixed from OLS (m = 78.35), we only learn b
m = 78.35
b = -120          # random start (class example)
lr = 0.01
epochs = 30

for i in range(epochs):
    slope = -2 * np.sum(y - m * X.ravel() - b)   # dL/db
    b = b - lr * slope                           # update step
    print(i, b)
```

✅ Final `b` should be close to the OLS intercept (class example: around **−3.85**).

---

## 7️⃣ Visualization 1 (what the animation shows)

Four plots in class (`b` starts at **−120** ... or **150** in another run):

| Plot                 | What you see                                             |
| -------------------- | -------------------------------------------------------- |
| 📈 **Data + line**   | Red line **moves up** towards the data cloud every epoch |
| 🥣 **Cost vs b**     | Red `+` points slide **down** the parabola               |
| 📉 **Loss vs epoch** | Drops fast first, then **flattens**                      |
| 📊 **b vs epoch**    | Rises fast first, then **flattens** at best `b`          |

💡 **Key lesson:** the flat part of the loss curve = **"when to stop"**. Training further gives almost no improvement.

```
Loss
 |\
 | \
 |  \___
 |      ‾‾‾‾‾‾‾‾‾‾‾   <-- flat = STOP here
 +---------------------- epochs
```

---

## 8️⃣ Few Discussions

### A) Effect of Learning Rate (epochs = 10 in the slide)

| η                | What happens                            | Result                                    |
| ---------------- | --------------------------------------- | ----------------------------------------- |
| **0.02** (small) | Tiny steps, line crawls                 | 🐢 **Too slow**, not reached in 10 epochs |
| **0.1** (good)   | Steady steps, lines bunch near the data | ✅ **Converges nicely**                   |
| **0.5** (large)  | Giant jumps, line flies away            | 💥 **Diverges** / overshoots              |

### B) Universality of Gradient Descent 🌍

Same recipe works for **any differentiable loss**, not only linear regression:

```
parameter = parameter − η × (dLoss / dparameter)
```

- Linear Regression ✔️
- Logistic Regression ✔️
- Neural Networks / Deep Learning ✔️ (the heart of backpropagation)

🧠 **Only two things change:** the loss function and how you find its slope.

---

## 9️⃣ Adding `m` into the Mix (now learn both `m` and `b`)

### Steps

```
1) Init random values:  m = 1 , b = 0
2) epochs = 100 , lr = 0.01
   for i in range(epochs):
       b = b − η × slope_b
       m = m − η × slope_m
```

### Why partial derivatives?

- Now `L = L(m, b)` → depends on **two** variables.
- Graph is no longer a 2D parabola; it is a **3D bowl**: `z = f(x, y)`.
- To find slope for one variable, treat the other as a constant → **partial derivative (∂)**.

```
b_slope = ∂L/∂b
m_slope = ∂L/∂m
```

### Derivation

`L = Σ (yᵢ − m·xᵢ − b)²`

**Slope w.r.t. `b`:**

```
∂L/∂b = −2 Σ (yᵢ − m·xᵢ − b)
```

**Slope w.r.t. `m`:**

```
∂L/∂m = −2 Σ (yᵢ − m·xᵢ − b) · xᵢ      (extra × xᵢ because derivative of −m·xᵢ is −xᵢ)
```

### 📌 Compare both

| Parameter | Slope formula               | Difference      |
| --------- | --------------------------- | --------------- |
| `b`       | `−2 Σ (yᵢ − m xᵢ − b)`      | plain error sum |
| `m`       | `−2 Σ (yᵢ − m xᵢ − b) · xᵢ` | error **× xᵢ**  |

⚠️ **Update both together** using the _old_ values (compute both slopes first, then update).

---

## 🔟 Code (both `m` and `b`) — `GDRegressor`

```python
class GDRegressor:

    def __init__(self, learning_rate, epochs):
        self.m = 100          # random start (class used 29.19 / -120 in a demo)
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        for i in range(self.epochs):
            y_hat = self.m * X.ravel() + self.b
            loss_slope_b = -2 * np.sum(y - y_hat)
            loss_slope_m = -2 * np.sum((y - y_hat) * X.ravel())

            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
        print(self.m, self.b)

    def predict(self, X):
        return self.m * X + self.b
```

### ✅ Check against sklearn

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_)        # ≈ 27.82
print(lr.intercept_)   # ≈ -2.29
```

- Class result: sklearn **m ≈ 27.83**, **b ≈ −2.29**.
- If `GDRegressor` ends near these values ➜ **our from-scratch GD is correct** 🎉
- Also compared with `cross_val_score(..., scoring='r2', cv=10)` ≈ **0.637**.

---

## 1️⃣1️⃣ Visualization 2 (3D loss surface)

- Plot of `L(m, b)` is a smooth **bowl** (colour scale up to millions).
- Purple/blue = **low loss**, yellow = **high loss**.
- Gradient Descent = a ball starting somewhere on the wall and sliding to the **purple bottom**.

```
     high loss (yellow)
        \          /
         \        /
          \      /
           \____/   <-- purple = minimum (best m, b)
```

---

## 1️⃣2️⃣ Effect of Learning Rate (summary)

| Learning rate  | Behaviour                              | Fix                  |
| -------------- | -------------------------------------- | -------------------- |
| Too **small**  | Very slow, needs many epochs           | Increase η or epochs |
| **Just right** | Smooth fast convergence                | 👍                   |
| Too **large**  | Zig-zag, overshoot, loss goes UP / NaN | Decrease η           |

🧠 **Tip:** try `0.001, 0.01, 0.1` and plot loss vs epochs. Loss should **go down every epoch**.

---

## 1️⃣3️⃣ Effect of Loss Function

### Convex function ✅ (our `Σ(y − ŷ)²`)

```
   \         /
    \       /
     \_____/   <-- ONE minimum = global minimum
```

- Any starting point leads to the **same answer**. Safe.

### Non-convex function (deep learning, many columns) ⚠️

```
   Cost
    |\        /\
    | \      /  \        ______ Plateau (flat, almost no slope)
    |  \_/\_/    \_    _/
    |  Local      Global
    |  minimum    minimum
```

| Problem            | What it means                                                          |
| ------------------ | ---------------------------------------------------------------------- |
| **Local minimum**  | A small valley; GD can get stuck here                                  |
| **Global minimum** | The real lowest point                                                  |
| **Plateau**        | Flat area, slope ≈ 0, training crawls                                  |
| **Saddle point**   | Goes down in one direction, up in another; slope ≈ 0 but not a minimum |

- Where you **start** can decide which minimum you reach.
- Fixes: better initialization, momentum, Adam optimizer, SGD noise.

---

## 1️⃣4️⃣ Effect of Data

Two cost contour pictures from class:

```
 θ₂                         θ₂
  |   ◯ circle                |  ╭────────────╮
  |  (similar scale)          |  │  ellipse   │   (very different scales)
  |   direct path to centre   |  ╰────────────╯   zig-zag, slow
  +-------- θ₁               +----------------- θ₁
```

| Data                                                                    | Contour shape     | GD path             |
| ----------------------------------------------------------------------- | ----------------- | ------------------- |
| **Features on same scale**                                              | Circle            | Straight, fast ➜ ✅ |
| **Features on different scales** (e.g., age 0–100 vs salary 0–1,00,000) | Stretched ellipse | Zig-zag, slow ➜ ❌  |

✅ **Always do Feature Scaling (StandardScaler / MinMaxScaler) before Gradient Descent.**

---

## 1️⃣5️⃣ Extra Theory (added by me for deeper understanding)

### 🧩 Types of Gradient Descent

| Type                    | Data used per update        | Pros                                | Cons                                  |
| ----------------------- | --------------------------- | ----------------------------------- | ------------------------------------- |
| **Batch GD**            | **All** rows                | Stable, smooth path                 | Slow for big data (what we did above) |
| **Stochastic GD (SGD)** | **1** random row            | Very fast, can escape local minima  | Noisy, zig-zag                        |
| **Mini-batch GD**       | A small batch (32, 64, 128) | Best balance, used in deep learning | Need to choose batch size             |

### 📐 Sum vs Mean

- In class we used **sum** of squared errors.
- In practice use **Mean Squared Error (MSE)** → divide by `n` → `η` becomes easier to tune.

### 🔁 Vector form (for many columns)

```
w = w − η × ∇L(w)
```

- `∇L` (nabla) = **gradient** = vector of all partial derivatives.

### ✅ Convergence checklist

- [ ] Features scaled
- [ ] Learning rate not too big
- [ ] Loss decreases every epoch
- [ ] Enough epochs
- [ ] Result matches `sklearn` / OLS

### 🎤 Interview Quick Q&A

| Question                    | Short answer                                                                           |
| --------------------------- | -------------------------------------------------------------------------------------- |
| What is Gradient Descent?   | Iterative optimization that moves parameters opposite to the gradient to minimize loss |
| Why subtract the gradient?  | Gradient points uphill; we want downhill                                               |
| What is learning rate?      | Step size multiplier `η`                                                               |
| What if η too big / small?  | Overshoot/diverge vs very slow                                                         |
| What is an epoch?           | One full pass of the update loop (over all data in batch GD)                           |
| OLS vs GD?                  | OLS = direct formula; GD = iterative, scales to big data                               |
| Is the MSE loss convex?     | Yes, for linear regression ➜ one global minimum                                        |
| Why feature scaling?        | Makes contours round ➜ faster, stable convergence                                      |
| Batch vs SGD vs Mini-batch? | All rows vs one row vs small group per update                                          |

### 🧠 Master Cheat Sheet

```
Loss      :  L = Σ (yᵢ − m xᵢ − b)²
Slope b   :  ∂L/∂b = −2 Σ (yᵢ − m xᵢ − b)
Slope m   :  ∂L/∂m = −2 Σ (yᵢ − m xᵢ − b) · xᵢ
Update    :  b = b − η · ∂L/∂b
             m = m − η · ∂L/∂m
Repeat    :  for each epoch until loss stops falling
```

🧠 **Final mnemonic — "S-E-L-R":**
**S**lope tells direction → **E**ta scales step → **L**oss must fall → **R**epeat for epochs.

---

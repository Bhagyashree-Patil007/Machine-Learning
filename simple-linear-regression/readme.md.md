# 📈 02 — Simple Linear Regression

> **Caveman style notes** — simple words, tables, diagrams, examples. No filler.
> Source: Day 48 (Simple Linear Regression) + Day 49 (Regression Metrics).

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [What is Simple Linear Regression?](#2-what-is-simple-linear-regression)
3. [Code Example (scikit-learn)](#3-code-example-scikit-learn)
4. [Intuition — What is the Best Fit Line?](#4-intuition--what-is-the-best-fit-line)
5. [How to find m and b?](#5-how-to-find-m-and-b)
6. [Code from Scratch](#6-code-from-scratch)
7. [Regression Metrics](#7-regression-metrics)
8. [Worked Example (by hand)](#8-worked-example-by-hand)
9. [Quick Revision Sheet](#9-quick-revision-sheet)

---

## 1. Introduction

**Regression = predict a NUMBER.**
(Classification = predict a category. Regression = predict a value.)

| Question | Output | Type |
|---|---|---|
| Will mail be spam? | Yes / No | Classification |
| What salary package will student get? | 3.5 LPA | **Regression** ✅ |
| What is house price? | ₹45,00,000 | **Regression** ✅ |

**Linear** = relation looks like a **straight line**.

---

## 2. What is Simple Linear Regression?

**Simple words:** Draw ONE straight line through the dots. Use that line to predict new values.

**"Simple"** = only **1 input column** (feature). (More than 1 input = Multiple Linear Regression, next topic.)

### Example from class — CGPA → Package

| CGPA (input `x`) | Package in LPA (output `y`) |
|---|---|
| 7.1 | 3.5 |
| 4.7 | 1.2 |
| 8.9 | 4.2 |
| 8.1 | 3.9 |
| ... | ... |

Data of **200 students**. Question: *new student has CGPA 6.5 → what package?*

```
 CGPA ──► [ MODEL ] ──► Package
 7.1                    3.5 LPA
```

### Scatter plot picture

```
 Package
 (LPA)
  4.5 │                          ●  ●  ╱
  4.0 │                    ● ● ●  ●  ╱
  3.5 │               ● ●  ● ●  ● ╱         ╱ = best fit line
  3.0 │           ●  ● ●  ●  ╱●
  2.5 │       ● ● ●  ●  ╱●
  2.0 │     ●  ● ●   ╱
  1.5 │   ●  ●   ╱
      └──────────────────────────────── CGPA
         4    5    6    7    8    9
```

### The line equation

```
        y  =  m · x  +  b
        │     │   │     │
   package    │  CGPA   │
              │         └── b = intercept (y when x = 0)
              └──────────── m = slope (how steep)
```

| Symbol | Name | Meaning | In our example |
|---|---|---|---|
| `y` | Output / target | What we predict | Package |
| `x` | Input / feature | What we know | CGPA |
| `m` | **Slope** | If x goes up by 1, y goes up by `m` | +1 CGPA → +`m` LPA |
| `b` | **Y-intercept** | Value of y when x = 0 (where line cuts y-axis) | Base package |

🎯 **Whole game of Linear Regression = find best `m` and `b`.**

**Why use it?** Real-world data is roughly linear: more study → more marks, more CGPA → more package, more area → more price.

---

## 3. Code Example (scikit-learn)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load data  (columns: cgpa, package)
df = pd.read_csv('placement.csv')

# 2. Look at data
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA'); plt.ylabel('Package (in LPA)')
plt.show()

# 3. Split input (X) and output (y)
X = df.iloc[:, 0:1]      # 2D  -> shape (200, 1)
y = df.iloc[:, -1]       # 1D  -> shape (200,)

# 4. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# 5. Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 6. Predict
print(lr.predict(X_test.iloc[0].values.reshape(1, 1)))

# 7. See the learned m and b
print("m (slope)     =", lr.coef_)
print("b (intercept) =", lr.intercept_)

# 8. Plot best-fit line
plt.scatter(df['cgpa'], df['package'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('CGPA'); plt.ylabel('Package (in LPA)')
plt.show()
```

> 📝 `X` must be **2D** (rows × columns), `y` can be **1D**.

---

## 4. Intuition — What is the Best Fit Line?

Many lines can be drawn. Which one is **best**?

```
 ● dots              Line A (too flat)    ✗ far from dots
                     Line B (too steep)   ✗ far from dots
   ●    ●  ╱ B       Line C (just right)  ✅ close to ALL dots
 ●   ●  ╱─── C
  ●  ╱ ●  ─── A
```

**Error (residual)** = distance between real dot and line.

```
 y
 │        ● ← actual (yᵢ)
 │        ┆
 │        ┆  error = yᵢ − ŷᵢ
 │  ──────●──────  ← line prediction (ŷᵢ)
 └────────────────── x
```

- `yᵢ` = actual value
- `ŷᵢ` (y-hat) = predicted value = `m·xᵢ + b`

**Best fit line = line where TOTAL error is smallest.**

**Why square the error?**
| Reason | Meaning |
|---|---|
| ➕➖ Cancel out | +5 and −5 would add to 0 (looks perfect but isn't). Squaring makes all positive. |
| 🔨 Big mistakes hurt more | Error 10 → 100, Error 1 → 1 |
| 📐 Smooth curve | Easy to differentiate (calculus) |

### Loss / Error function

```
        n
 E  =   Σ  ( yᵢ − m·xᵢ − b )²        ← "Sum of Squared Errors"
       i=1
```

**Goal: find `m` and `b` that make `E` as small as possible (minimum).**

---

## 5. How to find m and b?

Two ways:

| Way | Idea | Used when |
|---|---|---|
| 🧮 **Closed form** (OLS — Ordinary Least Squares) | Use direct formula (calculus) | Small data, **only 1 feature** → this lecture ✅ |
| 🏔️ **Gradient Descent** | Slowly walk down the error hill | Big data, many features → next topics |

### 5.1 Why derivative? (Maxima / Minima idea)

Error `E` vs `b` (or `m`) makes a **U-shaped bowl**. Bottom of bowl = minimum error.

```
  E │\            /
    │ \          /
    │  \        /
    │   \_    _/
    │     \__/   ← minimum  (slope = 0 here)
    └──────────────── b
```

At the bottom, **slope = 0**. So:

```
 dE/db = 0     and     dE/dm = 0       ← solve both → get (m, b)
```

With 2 variables (m and b), the error is a **3D bowl** (surface). Lowest point = best (m, b).

```
        E (height)
        │      ╭─╮
        │   ╭──╯ ╰──╮
        │  ╱   ●min  ╲        ● = lowest point = best (m, b)
        └─────────────────► m, b
```

### 5.2 Step 1 — Find `b`

Start:
```
 E = Σ ( yᵢ − m·xᵢ − b )²
```

Differentiate with respect to **b** and set to 0:

```
 ∂E/∂b = Σ 2 ( yᵢ − m·xᵢ − b ) · (−1) = 0
       ⇒ Σ ( yᵢ − m·xᵢ − b ) = 0
       ⇒ Σyᵢ − m·Σxᵢ − n·b = 0           (b added n times = n·b)
```

Divide by `n`:

```
 ȳ − m·x̄ − b = 0
```

✅ **Result 1:**

```
 ┌─────────────────┐
 │  b = ȳ − m·x̄   │      (ȳ = mean of y, x̄ = mean of x)
 └─────────────────┘
```

**Meaning:** The best line **always passes through the point (x̄, ȳ)** — the average point.

### 5.3 Step 2 — Find `m`

Put `b = ȳ − m·x̄` back into E:

```
 E = Σ ( yᵢ − m·xᵢ − ȳ + m·x̄ )²
   = Σ [ (yᵢ − ȳ) − m (xᵢ − x̄) ]²
```

Differentiate with respect to **m** and set to 0:

```
 ∂E/∂m = Σ 2 [ (yᵢ − ȳ) − m (xᵢ − x̄) ] · ( −(xᵢ − x̄) ) = 0
       ⇒ Σ (yᵢ − ȳ)(xᵢ − x̄) − m · Σ (xᵢ − x̄)² = 0
```

✅ **Result 2:**

```
        Σ (xᵢ − x̄)(yᵢ − ȳ)
 m  =  ─────────────────────
           Σ (xᵢ − x̄)²
```

### 5.4 Final formulas (remember these!)

| Find | Formula |
|---|---|
| **m** (slope) | `m = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²` |
| **b** (intercept) | `b = ȳ − m·x̄` |

🧠 **Mnemonic:** Find **m first**, then **b** from means. *"m from spread, b from mean."*

---

## 6. Code from Scratch

Build own Linear Regression class (no scikit-learn):

```python
import numpy as np

class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train).flatten()

        x_mean = X_train.mean()
        y_mean = y_train.mean()

        num = 0   # numerator   : Σ (xi - x̄)(yi - ȳ)
        den = 0   # denominator : Σ (xi - x̄)²
        for i in range(len(X_train)):
            num += (X_train[i] - x_mean) * (y_train[i] - y_mean)
            den += (X_train[i] - x_mean) ** 2

        self.m = num / den
        self.b = y_mean - self.m * x_mean
        print("m =", self.m, "| b =", self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b
```

**Use it:**
```python
lr = MeraLR()
lr.fit(X_train, y_train)
print(lr.predict(X_test[0]))
```

✅ Answer matches scikit-learn's `LinearRegression` → proof formula is right.

**Faster (vectorized) version:**
```python
m = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean())**2).sum()
b = y.mean() - m * X.mean()
```

---

## 7. Regression Metrics

**Why metrics?** To know **how good** the model is. Compare `y_test` (real) with `y_pred` (predicted).

```
 error = yᵢ − ŷᵢ
```

### 7.1 MAE — Mean Absolute Error

```
        1   n
 MAE = ─── Σ | yᵢ − ŷᵢ |
        n  i=1
```

| ✅ Good | ❌ Bad |
|---|---|
| Same **unit** as y (LPA → LPA) | Graph has a sharp corner at 0 → **not differentiable** there |
| Easy to understand | |
| **Robust to outliers** (big errors not blown up) | |

**Meaning:** "On average, prediction is off by ___ LPA."

### 7.2 MSE — Mean Squared Error

```
        1   n
 MSE = ─── Σ ( yᵢ − ŷᵢ )²
        n  i=1
```

| ✅ Good | ❌ Bad |
|---|---|
| Smooth curve → **differentiable** (used as **loss function**) | Unit is **squared** (LPA²) — hard to understand |
| Punishes big errors strongly | **Not robust to outliers** |

### 7.3 RMSE — Root Mean Squared Error

```
 RMSE = √MSE = √[ (1/n) Σ ( yᵢ − ŷᵢ )² ]
```

| ✅ Good | ❌ Bad |
|---|---|
| Same **unit** as y again (LPA) | Still affected by outliers |
| Smooth, differentiable | |

### 7.4 R² Score (Coefficient of Determination)

**Idea:** Compare **our model** vs a **dumb model** that always predicts the **mean (ȳ)**.

```
 Dumb model: ──────────── ȳ (flat line)         → error = SST
 Our model:  ╱╱╱╱╱╱╱╱╱╱╱  (best-fit line)       → error = SSR
```

```
          Σ ( yᵢ − ŷᵢ )²        SSR  (error of our model)
 R²  = 1 − ─────────────── = 1 − ─────
          Σ ( yᵢ − ȳ )²         SST  (error of mean model)
```

| R² value | Meaning |
|---|---|
| **1** | Perfect model 🏆 |
| **0.8** | Model explains **80%** of variation in output ✅ |
| **0** | Same as the dumb mean model 😐 |
| **< 0** | Worse than mean model ❌ |

**Example:** R² = 0.78 → "CGPA explains 78% of the changes in package."

### 7.5 Adjusted R² Score

**Problem with R²:** when you **add more input columns**, R² **always goes up (or stays same)** — even if the new column is **useless** (like roll number). R² gets fooled! 😵

**Fix:** Adjusted R² **penalizes useless columns**.

```
                  ( 1 − R² ) ( n − 1 )
 Adj R² = 1 − ───────────────────────
                    n − k − 1

   n = number of rows (samples)
   k = number of input columns (features)
```

| Add column | R² | Adjusted R² |
|---|---|---|
| ✅ **Useful** column | ⬆️ up | ⬆️ up |
| ❌ **Useless** column | ⬆️ slightly up | ⬇️ **down** |

**Rule:** Adjusted R² ≤ R². Use it when you have **many features**.

### 7.6 Metrics comparison table

| Metric | Formula (short) | Unit | Outlier robust? | Differentiable? | Best value |
|---|---|---|---|---|---|
| **MAE** | mean(\|error\|) | Same as y | ✅ Yes | ❌ No (at 0) | 0 |
| **MSE** | mean(error²) | y² | ❌ No | ✅ Yes | 0 |
| **RMSE** | √MSE | Same as y | ❌ No | ✅ Yes | 0 |
| **R²** | 1 − SSR/SST | No unit | — | — | 1 |
| **Adj R²** | R² with penalty | No unit | — | — | 1 |

### 7.7 Code for metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = lr.predict(X_test)

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))

r2 = r2_score(y_test, y_pred)
print("R2   :", r2)

# Adjusted R2
n = X_test.shape[0]          # rows
k = X_test.shape[1]          # columns
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print("Adj R2:", adj_r2)
```

---

## 8. Worked Example (by hand)

Tiny data (4 students):

| CGPA `x` | Package `y` |
|---|---|
| 6 | 2.0 |
| 7 | 3.0 |
| 8 | 3.5 |
| 9 | 4.5 |

**Step 1 — means:** `x̄ = 7.5`, `ȳ = 3.25`

**Step 2 — table:**

| x | y | x − x̄ | y − ȳ | (x−x̄)(y−ȳ) | (x−x̄)² |
|---|---|---|---|---|---|
| 6 | 2.0 | −1.5 | −1.25 | 1.875 | 2.25 |
| 7 | 3.0 | −0.5 | −0.25 | 0.125 | 0.25 |
| 8 | 3.5 | 0.5 | 0.25 | 0.125 | 0.25 |
| 9 | 4.5 | 1.5 | 1.25 | 1.875 | 2.25 |
| | | | **Sum** | **4.0** | **5.0** |

**Step 3 — m and b:**
```
 m = 4.0 / 5.0 = 0.8
 b = 3.25 − 0.8 × 7.5 = −2.75
 Line:  y = 0.8x − 2.75
```
*Meaning:* +1 CGPA → +0.8 LPA.

**Step 4 — predict:**

| x | Actual y | Predicted ŷ | Error (y − ŷ) |
|---|---|---|---|
| 6 | 2.0 | 2.05 | −0.05 |
| 7 | 3.0 | 2.85 | 0.15 |
| 8 | 3.5 | 3.65 | −0.15 |
| 9 | 4.5 | 4.45 | 0.05 |

**Step 5 — metrics:**

| Metric | Calculation | Value |
|---|---|---|
| MAE | (0.05+0.15+0.15+0.05) / 4 | **0.10** |
| MSE | (0.0025+0.0225+0.0225+0.0025) / 4 | **0.0125** |
| RMSE | √0.0125 | **≈ 0.112** |
| R² | SSR = 0.05, SST = 3.25 → 1 − 0.05/3.25 | **≈ 0.985** |
| Adj R² | n=4, k=1 → 1 − (0.0154 × 3)/2 | **≈ 0.977** |

✅ R² ≈ 0.98 → CGPA explains ~98% of package here.
New student CGPA 8.5 → `0.8 × 8.5 − 2.75 = 4.05 LPA`.

---

## 9. Quick Revision Sheet

| Topic | One-line memory |
|---|---|
| Regression | Predict a number |
| Simple LR | 1 input column, 1 straight line |
| Equation | `y = m·x + b` |
| m | Slope — steepness |
| b | Intercept — y when x = 0 |
| Best fit line | Line with smallest total squared error |
| Loss | `E = Σ(yᵢ − m·xᵢ − b)²` |
| Find min | Set `∂E/∂m = 0` and `∂E/∂b = 0` |
| b formula | `b = ȳ − m·x̄` |
| m formula | `Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²` |
| Line passes through | The mean point (x̄, ȳ) |
| MAE | Avg absolute error, same unit, outlier-safe |
| MSE | Avg squared error, smooth, unit², outlier-sensitive |
| RMSE | √MSE, same unit as y |
| R² | How much variation model explains (0 to 1) |
| Adj R² | R² minus penalty for useless columns |

🧠 **Mnemonic for metrics:** **M**ean **A**bsolute, **M**ean **S**quared, **R**oot, **R**-squared, **A**djusted → *"MAM-S-R-R-A"* → go from simple → smart.

---

⭐ *Next:* `03_...` — Multiple Linear Regression & Gradient Descent.

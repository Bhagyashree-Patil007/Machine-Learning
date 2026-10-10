# 📊 03 — Multiple Linear Regression

---

## 📁 What is in this folder?

```
03_multiple_linear_regression/
├── README.md                          ← notes (this file)
└── multiple_linear_regression.ipynb   ← Python code (sklearn + 3D plot)
```

---

## 📑 Table of Contents

1. [Recap](#1-recap--simple-linear-regression)
2. [What is Multiple Linear Regression?](#2-what-is-multiple-linear-regression)
3. [Python Code](#3-python-code)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Code From Scratch](#5-code-from-scratch)
6. [Problem with OLS Solution](#6-problem-with-ols-solution)
7. [Quick Revision Sheet](#7-quick-revision-sheet)

---

## 1. Recap — Simple Linear Regression

| Point     | Memory                                     |
| --------- | ------------------------------------------ |
| Inputs    | Only **1** column (CGPA)                   |
| Output    | A number (Package)                         |
| Equation  | `y = m·x + b`                              |
| Picture   | **Line** in 2D                             |
| Find m, b | `m = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²`, `b = ȳ − m·x̄` |
| Metrics   | MAE, MSE, RMSE, R², Adjusted R²            |

**Problem:** real life never has just 1 input! 🏠 House price depends on area **and** bedrooms **and** location **and** age...
👉 So we use **Multiple** Linear Regression.

---

## 2. What is Multiple Linear Regression?

**Simple words:** Same as Simple Linear Regression, but with **2 or more input columns**.

|               | Simple LR         | Multiple LR                                       |
| ------------- | ----------------- | ------------------------------------------------- |
| Input columns | 1                 | 2 or more                                         |
| Equation      | `y = b + m·x`     | `y = b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ`                 |
| Shape         | Straight **line** | Flat **plane** (2 inputs) / **hyperplane** (more) |
| Example       | CGPA → Package    | CGPA + IQ + Skills → Package                      |

### Example — Student package

| CGPA (x₁) | IQ (x₂) | Package (y) |
| --------- | ------- | ----------- |
| 8.0       | 110     | 6.5         |
| 6.5       | 95      | 3.0         |
| 9.0       | 120     | 9.0         |

```
 CGPA ──┐
 IQ   ──┼──► [ MODEL ] ──► Package
 ...  ──┘
```

### Equation parts

```
   y  =  b₀  +  b₁·x₁  +  b₂·x₂  +  …  +  bₙ·xₙ
   │     │      └──────────┬──────────────┘
   │     │          coefficients (weights)
   │     └── intercept (value when all x = 0)
   └──────── output
```

| Symbol     | Name         | Meaning                                                                     |
| ---------- | ------------ | --------------------------------------------------------------------------- |
| `b₀`       | Intercept    | Base value of y                                                             |
| `b₁, b₂ …` | Coefficients | If **only that column** goes up by 1 (others fixed), y changes by that much |
| `x₁, x₂ …` | Features     | Input columns                                                               |

### Picture: Line (1 input) vs Plane (2 inputs)

```
  Simple LR (2D)               Multiple LR with 2 inputs (3D)

   y │     ╱ •                      y │      ╱╱╱╱╱╱  ← flat plane
     │   •╱                           │    ╱╱╱╱╱╱ •
     │  ╱ •                           │  ╱╱╱╱╱╱•
     │╱•                              │╱╱╱╱╱╱
     └────────── x                    └──x₁──────x₂   (dots float around plane)
```

🎯 **Goal:** find the best plane/hyperplane = best `b₀, b₁, b₂…` which makes total error smallest.

**Reading coefficients — example:** `Package = 0.5 + 0.8·CGPA + 0.02·IQ`

- CGPA +1 → package +0.8 LPA (IQ same)
- IQ +1 → package +0.02 LPA (CGPA same)

---

## 3. Python Code

_(Full code in `multiple_linear_regression.ipynb`)_

### Step 1 — Make sample data (100 rows, 2 features)

```python
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

X, y = make_regression(n_samples=100, n_features=2,
                       n_informative=2, n_targets=1, noise=50)

df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})
df.shape      # (100, 3)
df.head()
```

| Parameter         | Meaning                             |
| ----------------- | ----------------------------------- |
| `n_samples=100`   | 100 rows                            |
| `n_features=2`    | 2 input columns                     |
| `n_informative=2` | both columns really matter          |
| `noise=50`        | add random scatter (like real data) |

### Step 2 — 3D scatter plot

```python
import plotly.express as px
fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
fig.show()
```

👀 Dots look like they float around a plane → plane fits!

### Step 3 — Train / Test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3)
```

### Step 4 — Train the model

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```

### Step 5 — Predict & check metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = lr.predict(X_test)
print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
print("R2 score", r2_score(y_test, y_pred))
```

**Output in notebook:**

| Metric | Value     | Meaning                             |
| ------ | --------- | ----------------------------------- |
| MAE    | ≈ 33.09   | On average, off by ~33 units        |
| MSE    | ≈ 1659.91 | Squared error                       |
| R²     | ≈ 0.719   | Model explains ~72% of variation ✅ |

_(Numbers change on every run because the data is random.)_

### Step 6 — Look at coefficients

```python
lr.coef_        # array([59.645, 13.204])   → b₁, b₂
lr.intercept_   # -7.555                     → b₀
```

**Learned plane:**

```
 target = −7.55 + 59.65·feature1 + 13.20·feature2
```

👉 `feature1` matters **more** (bigger coefficient).

### Step 7 — Draw the plane on 3D plot

```python
import plotly.graph_objects as go

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
xGrid, yGrid = np.meshgrid(y, x)

final = np.vstack((xGrid.ravel().reshape(1, 100),
                   yGrid.ravel().reshape(1, 100))).T      # make `final` FIRST
z = lr.predict(final).reshape(10, 10)

fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
fig.add_trace(go.Surface(x=x, y=y, z=z))
fig.show()
```

> ⚠️ **Note on notebook:** in the cell, `final` is used _before_ it is created. If you run cells top to bottom you get a `NameError`. Define `final` first (as above). Also, this cell reuses the names `x` and `y` and **overwrites** your target `y` — use different names (like `x_range`) if you need `y` later.

---

## 4. Mathematical Formulation

### 4.1 Same idea as before — minimize squared error

```
        n
 E  =   Σ  ( yᵢ − ŷᵢ )²       where  ŷᵢ = b₀ + b₁xᵢ₁ + b₂xᵢ₂ + … + bₘxᵢₘ
       i=1
```

Earlier (simple LR) we differentiated for `m` and `b` one by one. With 100 columns that is 100 equations! 😵
👉 Trick: use **matrices** (linear algebra). All equations in one line.

### 4.2 Put data into matrix form

Take **n rows** and **m columns**. Add a column of **1s** (this handles `b₀`).

```
      ┌ 1  x₁₁  x₁₂ ┐        ┌ y₁ ┐         ┌ b₀ ┐
 X =  │ 1  x₂₁  x₂₂ │   y =  │ y₂ │   β =   │ b₁ │
      │ 1  x₃₁  x₃₂ │        │ y₃ │         │ b₂ │
      └ ⋮    ⋮    ⋮ ┘        └ ⋮  ┘         └    ┘
       (n × (m+1))           (n × 1)       ((m+1) × 1)
```

| Symbol | Name                                   | Size      |
| ------ | -------------------------------------- | --------- |
| `X`    | Input matrix (with extra column of 1s) | n × (m+1) |
| `y`    | Output vector                          | n × 1     |
| `β`    | Coefficients vector (b₀, b₁, b₂…)      | (m+1) × 1 |

**Why the column of 1s?** So `b₀ × 1` = `b₀`. Then whole equation is just: **`ŷ = X·β`**.

### 4.3 Error in matrix form

```
 E = (y − Xβ)ᵀ (y − Xβ)          ← same as Σ(yᵢ − ŷᵢ)²
```

(`ᵀ` = transpose = flip rows and columns. Vector × its own transpose = sum of squares.)

### 4.4 Find minimum — differentiate and set to 0

```
 ∂E/∂β = −2 Xᵀ (y − Xβ) = 0
       ⇒ Xᵀy − XᵀXβ = 0
       ⇒ XᵀX β = Xᵀy
```

### 4.5 ✅ Final formula (Normal Equation / OLS)

```
 ┌──────────────────────────┐
 │  β  =  (XᵀX)⁻¹ · Xᵀ · y  │
 └──────────────────────────┘
```

**One line gives ALL coefficients at once!**
`( )⁻¹` = matrix inverse.

### 4.6 Worked example (small, by hand-style)

Data: 4 rows, 2 features.

| x₁ (area) | x₂ (bedrooms) | y (price) |
| --------- | ------------- | --------- |
| 1         | 1             | 6         |
| 2         | 1             | 9         |
| 3         | 2             | 13        |
| 4         | 2             | 16        |

Add 1s column:

```
      ┌ 1 1 1 ┐        ┌  6 ┐
 X =  │ 1 2 1 │   y =  │  9 │
      │ 1 3 2 │        │ 13 │
      └ 1 4 2 ┘        └ 16 ┘
```

Apply `β = (XᵀX)⁻¹ Xᵀ y` → **β = [2, 3, 1]**

```
 price = 2 + 3·area + 1·bedrooms
```

Check row 3: `2 + 3×3 + 1×2 = 13` ✅

### 4.7 Mapping to simple LR (to connect)

|          | Simple LR                                                 | Multiple LR      |
| -------- | --------------------------------------------------------- | ---------------- |
| Method   | Formula for `m` and `b`                                   | `β = (XᵀX)⁻¹Xᵀy` |
| Both are | **OLS** — Ordinary Least Squares (minimize squared error) |                  |

---

## 5. Code From Scratch

Build own `LinearRegression` using the normal-equation formula:

```python
import numpy as np

class MeraLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        # 1) add column of 1s on the left  → handles b0
        X_train = np.insert(X_train, 0, 1, axis=1)

        # 2) β = (XᵀX)⁻¹ Xᵀ y
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # 3) split β into intercept and coefficients
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_
```

**Use it:**

```python
lr = MeraLR()
lr.fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)
y_pred = lr.predict(X_test)
```

✅ **Check:** results match scikit-learn's `LinearRegression` (same `coef_`, same `intercept_`) → formula is right.

**Line-by-line:**
| Line | Job |
|---|---|
| `np.insert(X, 0, 1, axis=1)` | Add column of 1s at position 0 |
| `X.T @ X` | XᵀX |
| `np.linalg.inv(...)` | Matrix inverse |
| `betas[0]` | b₀ (intercept) |
| `betas[1:]` | b₁, b₂, … (coefficients) |

---

## 6. Problem with OLS Solution

OLS formula `β = (XᵀX)⁻¹ Xᵀ y` is neat, but has **problems**:

### Problem 1 — 🐌 Matrix inverse is very slow

```
 Inverse cost  ≈  O(m³)       (m = number of columns)
```

| Columns | Work (≈ m³)       |
| ------- | ----------------- |
| 10      | 1,000             |
| 1,000   | 1,000,000,000     |
| 10,000  | 1,000,000,000,000 |

Real data (images, text, genes) has **thousands of columns** → too slow, too much memory (XᵀX is m × m).

### Problem 2 — 💥 Inverse may not exist

`XᵀX` is **not invertible** when:

- 🔁 **Multicollinearity** — columns repeat information (e.g. `height in cm` and `height in inches`; or `x₂ = 2·x₁`)
- 📉 More columns than rows
- Result: error `Singular matrix` or crazy unstable coefficients

```
 x₁ = [1, 2, 3, 4]
 x₂ = [2, 4, 6, 8]   ← exactly 2·x₁  → XᵀX determinant = 0 → ❌ no inverse
```

### Problem 3 — 🧱 All data at once

Needs all data in memory together. Cannot learn in small chunks (online learning from Lecture 01).

### Problem 4 — ⚠️ Only for linear + squared error

Formula works only for this exact setup. Other models/loss functions have **no direct formula**.

### 🔑 Solution → **Gradient Descent**

```
        Start somewhere on the hill
                 ●
                  ╲
                   ╲  take small step downhill
                    ●
                     ╲
                      ●  ← reach bottom (min error)
```

- Walk down the error bowl **step by step**
- No inverse needed
- Works for huge data and many columns
- Works with other models too

|                 | OLS (Normal Equation)   | Gradient Descent       |
| --------------- | ----------------------- | ---------------------- |
| Idea            | Direct formula          | Small steps to minimum |
| Inverse needed? | ✅ Yes                  | ❌ No                  |
| Many columns    | 🐌 Slow                 | ✅ Fast                |
| Learning rate?  | No                      | ✅ Yes (must choose)   |
| Iterations?     | No                      | ✅ Yes                 |
| Use when        | Small data, few columns | Big data, many columns |

➡️ **Next topic: Gradient Descent** (`04_Gradient_Descent`).

---

## 7. Quick Revision Sheet

| Topic         | One-line memory                                     |
| ------------- | --------------------------------------------------- |
| Multiple LR   | Linear regression with 2+ input columns             |
| Equation      | `y = b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ`                   |
| Shape         | 1 input = line, 2 inputs = plane, more = hyperplane |
| `b₀`          | Intercept                                           |
| `b₁…bₙ`       | Coefficients — effect of that column, others fixed  |
| Matrix form   | `ŷ = Xβ` (X has extra column of 1s)                 |
| Error         | `E = (y − Xβ)ᵀ(y − Xβ)`                             |
| OLS formula   | `β = (XᵀX)⁻¹ Xᵀ y`                                  |
| sklearn       | `LinearRegression().fit()` → `coef_`, `intercept_`  |
| Scratch trick | `np.insert(X, 0, 1, axis=1)`                        |
| Metrics       | MAE, MSE, RMSE, R², Adjusted R² (same as before)    |
| OLS problem 1 | Inverse is slow (≈ m³)                              |
| OLS problem 2 | Inverse may not exist (multicollinearity)           |
| OLS problem 3 | Needs all data in memory                            |
| Fix           | **Gradient Descent**                                |

🧠 **Mnemonic:** **"Add 1s, Transpose, Invert"** → _Add column of 1s → XᵀX → invert → multiply Xᵀy_.

---

⭐ _Previous:_ [`02_Simple_Linear_Regression`](../02_Simple_Linear_Regression.md) | _Next:_ `04_Gradient_Descent`

# 🧠 01 — Introduction to Machine Learning

---

## 📑 Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [History of ML (short)](#2-history-of-ml-short)
3. [AI vs ML vs DL](#3-ai-vs-ml-vs-dl)
4. [Types of ML](#4-types-of-ml)
5. [Batch vs Online Learning](#5-batch-vs-online-learning)
6. [Instance-based vs Model-based Learning](#6-instance-based-vs-model-based-learning)
7. [Challenges in ML](#7-challenges-in-ml)
8. [Applications of ML](#8-applications-of-ml)
9. [ML Development Life Cycle (MLDLC)](#9-ml-development-life-cycle-mldlc)
10. [Jobs & Roles in ML](#10-jobs--roles-in-ml)
11. [Tensors](#11-tensors)
12. [Tools Setup](#12-tools-setup)
13. [Framing an ML Problem](#13-framing-an-ml-problem)
14. [Quick Revision Sheet](#14-quick-revision-sheet)

---

## 1. What is Machine Learning?

**Simple words:** Normal programming = human writes rules. ML = computer **finds rules itself from data**.

```
 NORMAL PROGRAMMING                 MACHINE LEARNING
 ┌────────┐                         ┌────────┐
 │ DATA   │──┐                      │ DATA   │──┐
 └────────┘  ├──► PROGRAM ──► OUT   └────────┘  ├──► ML ALGO ──► RULES (MODEL)
 ┌────────┐  │                      ┌────────┐  │
 │ RULES  │──┘                      │ OUTPUT │──┘
 └────────┘                         └────────┘
 (human writes rules)               (computer learns rules)
```

**Example — Spam filter**
| Way | How |
|---|---|
| 🧱 Normal code | Human writes: `if "lottery" in mail → spam`. Spammer changes word → code breaks. |
| 🤖 ML | Show 10,000 spam + 10,000 normal mails. Computer learns pattern itself. Works on new tricks too. |

**Formal-ish line:** _"A computer program learns from experience (E) for a task (T) if its performance (P) improves with E."_

🧠 **Mnemonic:** **T-E-P** → _Task, Experience, Performance_.

---

## 2. History of ML (short)

| Year      | Event                                                     |
| --------- | --------------------------------------------------------- |
| 1950      | Alan Turing asks "Can machines think?" (Turing Test)      |
| 1957      | Perceptron (first neural network idea) — Frank Rosenblatt |
| 1960s–70s | "AI winter" — hype too big, results small, funding cut    |
| 1980s     | Backpropagation revives neural networks                   |
| 1997      | IBM Deep Blue beats chess champion Kasparov               |
| 2000s     | Big data + faster computers (GPUs)                        |
| 2012      | AlexNet wins ImageNet → Deep Learning boom                |
| 2016      | AlphaGo beats Go champion                                 |
| 2017+     | Transformers → ChatGPT-type models                        |

**Why ML is booming now?** 3 reasons → 📦 **More data**, ⚡ **More compute (GPU)**, 🧩 **Better algorithms**.

---

## 3. AI vs ML vs DL

```
┌───────────────────────────────────────────────┐
│  AI  — machine acts smart                     │
│   ┌───────────────────────────────────────┐   │
│   │  ML — learns from data                │   │
│   │   ┌───────────────────────────────┐   │   │
│   │   │  DL — many-layer neural nets  │   │   │
│   │   └───────────────────────────────┘   │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
        DL ⊂ ML ⊂ AI   (small box inside big box)
```

|              | AI                          | ML                    | DL                            |
| ------------ | --------------------------- | --------------------- | ----------------------------- |
| Meaning      | Any smart machine behaviour | Learn rules from data | ML using deep neural networks |
| Example      | Chess bot with fixed rules  | Spam filter           | Face unlock, ChatGPT          |
| Data needed  | Can be none                 | Medium                | Very large                    |
| Feature work | Human                       | Human picks features  | Network finds features itself |

---

## 4. Types of ML

```
                    MACHINE LEARNING
      ┌──────────────┬──────────────┬─────────────────┐
 SUPERVISED     UNSUPERVISED    SEMI-SUPERVISED   REINFORCEMENT
 (has labels)   (no labels)     (few labels)      (reward/penalty)
```

| Type                   | Idea                                             | Example                                   |
| ---------------------- | ------------------------------------------------ | ----------------------------------------- |
| 🟦 **Supervised**      | Data has answers (labels). Learn input → output. | House size → price. Mail → spam/not spam. |
| 🟩 **Unsupervised**    | No answers. Find hidden groups/patterns.         | Group customers by buying habit.          |
| 🟨 **Semi-supervised** | Few labelled + many unlabelled.                  | Google Photos: name 1 face, it tags all.  |
| 🟥 **Reinforcement**   | Agent tries actions, gets reward/penalty.        | Robot learning to walk, game bot.         |

**Supervised has 2 kinds:**
| Kind | Output | Example |
|---|---|---|
| **Regression** | A number | Predict salary = ₹6.5 LPA |
| **Classification** | A category | Mail = Spam / Not spam |

**Unsupervised kinds:** Clustering, Dimensionality reduction, Anomaly detection, Association rules.

🧠 **Mnemonic:** _"Teacher / No teacher / Little teacher / Reward"_ = Sup / Unsup / Semi / RL.

---

## 5. Batch vs Online Learning

(_How are ML models trained? — all at once, or little by little?_)

### 5.1 Batch / Offline Learning

Model trained on **all data at once**. Then goes to production. **Does not learn anymore** there.

```
 ALL DATA ──► TRAIN (offline) ──► MODEL ──► DEPLOY ──► predictions
                                                  (model is frozen)
 New data comes? → Retrain from scratch on old + new data → redeploy
```

**Problems with Batch Learning**
| # | Problem | Meaning |
|---|---|---|
| 1 | 📦 Lots of data | Retraining on full data every time is slow |
| 2 | 🖥️ Hardware limit | Needs big memory/GPU, costly |
| 3 | ⏰ Availability | Some apps need fresh model every hour — batch can't |

**Disadvantages:** model gets stale, high retrain cost, can't adapt fast.

### 5.2 Online Learning

Model trained **step by step** with small chunks (mini-batches) of data. **Keeps learning in production.**

```
 DATA chunk 1 ─┐
 DATA chunk 2 ─┼─► MODEL (updates itself again & again) ──► predictions
 DATA chunk 3 ─┘            ▲
                            └──── new data keeps arriving
```

**When to use?**

1. 🌊 **Concept drift** — world changes fast (stock market, fashion trends, spam tricks)
2. 💰 **Cost effective** — no full retrain
3. ⚡ **Faster solution** — updates in minutes

**How to implement?** Use tools that learn in chunks: `partial_fit()` in scikit-learn, River, MOA, SAMOA, scikit-multiflow.

**Learning Rate** = how fast model forgets old and learns new.
| Rate | Effect |
|---|---|
| 🔼 High | Adapts fast, but **forgets old data**, sensitive to noise |
| 🔽 Low | Remembers old data, but **adapts slowly** |

**Out-of-Core Learning:** data too big for RAM? Load in small pieces from disk, learn from each piece, throw away, load next. (Usually done offline, but uses online-style learning.)

```
 HARD DISK (huge data) ──► piece 1 ─► learn ─► drop
                       ──► piece 2 ─► learn ─► drop
                       ──► piece 3 ─► learn ─► drop    (RAM never overflows)
```

**Disadvantages of Online ML:** ⚠️ Tricky to use, ⚠️ Risky (bad/poisoned data can ruin the model live).

### 5.3 Comparison Table

| Feature        | 🟦 Offline (Batch)                                    | 🟧 Online                                             |
| -------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| Complexity     | Less — model is constant                              | Dynamic — model keeps changing                        |
| Computation    | Fewer, single-time training                           | Continuous updates                                    |
| Production use | Easier to implement                                   | Hard to implement & manage                            |
| Applications   | Image classification, stable patterns                 | Finance, economics, health — new patterns keep coming |
| Tools          | scikit-learn, TensorFlow, PyTorch, Keras, Spark MLlib | MOA, SAMOA, scikit-multiflow, streamDM                |

**Example:** 🐱 Cat vs Dog classifier = Batch (cat never changes). 📈 Stock price predictor = Online (market changes daily).

---

## 6. Instance-based vs Model-based Learning

(_How does ML **generalize** to new data?_)

### 6.1 Instance-based Learning ("memorize")

Model **keeps training data**. For new input, it **compares with stored examples** and copies the answer of the most similar ones.

```
 NEW POINT ?
      │  find closest stored points
      ▼
   ○ ○ ●●       Closest 3 are ● ● ●  → answer = ●
   ○ ○ ● ?●     (This is k-Nearest Neighbours — KNN)
```

**Example:** Student memorizes all old question papers; new paper comes → matches with most similar old question.

### 6.2 Model-based Learning ("understand the rule")

Model **learns a formula/parameters** from training data. Then **throws data away**. New input → plug in formula.

```
 TRAIN DATA ──► find line  y = m·x + c  ──► keep only (m, c)
 New x ──► plug in ──► y
```

**Example:** Student understands the _concept_ and solves any new problem. (Linear Regression, Logistic Regression, Neural Nets.)

### 6.3 Differences

| Usual / Model-based ML                               | Instance-based Learning                                        |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| Prepare data for training                            | Prepare data — **no difference here**                          |
| Train model, estimate parameters (discover patterns) | **Do not train.** Pattern discovery postponed till query comes |
| Store the model                                      | **No model to store**                                          |
| Generalize rules **before** seeing new instance      | Generalize **only when** new instance is seen                  |
| Predict using the model                              | Predict using **training data directly**                       |
| Can throw away training data after training          | **Must keep** training data                                    |
| Needs a known model form                             | May have no explicit model form                                |
| Storing model needs **less** storage                 | Storing data needs **more** storage                            |

🧠 **Mnemonic:** Instance = **I**s lazy, **I**nspects data every time. Model = **M**akes formula, **M**oves on.

---

## 7. Challenges in ML

```
 ┌── DATA problems ──────────────────┐   ┌── MODEL problems ───┐   ┌── SYSTEM problems ──┐
 │ 1 Data collection                 │   │ 6 Overfitting       │   │ 8 Software integr.  │
 │ 2 Insufficient / unlabelled data  │   │ 7 Underfitting      │   │ 9 Offline deploy    │
 │ 3 Non-representative data         │   └─────────────────────┘   │ 10 Cost involved    │
 │ 4 Poor quality data               │                             └─────────────────────┘
 │ 5 Irrelevant features             │
 └───────────────────────────────────┘
```

| #   | Challenge                         | Simple meaning                                                        | Example                                          |
| --- | --------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| 1   | **Data collection**               | Getting data is hard (APIs, scraping, surveys, cost, permission)      | Hospital data is private                         |
| 2   | **Insufficient / labelled data**  | Too little data, or no labels. ML is data-hungry.                     | Only 50 photos to train face detector            |
| 3   | **Non-representative data**       | Training data ≠ real world                                            | Train on summer sales only → fails in winter     |
| 4   | **Poor quality data**             | Errors, missing values, outliers, noise. **Garbage in = garbage out** | Age = −5, salary blank                           |
| 5   | **Irrelevant features**           | Useless columns confuse model                                         | Predict house price using owner's shoe size      |
| 6   | **Overfitting**                   | Model memorizes training data, fails on new data                      | Student byhearts answers, fails twisted question |
| 7   | **Underfitting**                  | Model too simple, can't learn pattern even on training data           | Drawing a straight line through curved data      |
| 8   | **Software integration**          | Model must plug into app/website/backend                              | Python model inside Java app                     |
| 9   | **Offline learning / deployment** | Batch model gets stale after deploy                                   | Spam model old vs new spam tricks                |
| 10  | **Cost involved**                 | Data + engineers + GPUs + cloud + maintenance                         | Training big model costs lakhs                   |

**Overfit vs Underfit picture:**

```
 UNDERFIT            GOOD FIT             OVERFIT
 (too simple)        (just right)         (too wiggly)

  •  •   •            •  •   •             •  •   •
 ─────────           ╭──╮  ╭──            ╭╮╭╮ ╭─╮╭
  •   •  •          •   ╰╮╭╯  •         • ╯╰╯╰╮╯ ╰╯ •
 straight line       smooth curve        touches every dot
 High error          Low error           Train ✅  Test ❌
```

🧠 **Mnemonic:** **Over** = _Over-memorized_. **Under** = _Under-learned_.

---

## 8. Applications of ML

| Domain                   | Company example     | What ML does                                                                    |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------- |
| 🛒 **Retail**            | Amazon / Big Bazaar | Product recommendations, demand forecasting, "people also bought"               |
| 🏦 **Banking & Finance** | SBI / HDFC          | Fraud detection, loan default prediction, credit scoring                        |
| 🚕 **Transport**         | OLA / Uber          | **Surge pricing**, ETA, driver–rider matching                                   |
| 🏭 **Manufacturing**     | Tesla               | Robots on assembly line, defect detection, predictive maintenance, self-driving |
| 📱 **Consumer Internet** | Twitter             | Feed ranking, spam/hate detection, who-to-follow suggestions                    |

**Others:** Healthcare (disease detection), Agriculture (crop disease), Education (personalised learning), Entertainment (Netflix / YouTube recommendations).

---

## 9. ML Development Life Cycle (MLDLC)

Steps to build a real ML product — **not just model training!**

```
 ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌─────┐
 │1 FRAME   │──►│2 GATHER  │──►│3 PREPROCESS│──►│4 EDA│
 │ PROBLEM  │   │ DATA     │   │ (clean)    │   │     │
 └──────────┘   └──────────┘   └────────────┘   └──┬──┘
                                                    ▼
 ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────┐
 │9 OPTIMIZE│◄──│8 TESTING │◄──│7 DEPLOY    │◄──│6 MODEL TRAIN│◄─ 5 FEATURE
 └────┬─────┘   └──────────┘   └────────────┘   │ EVAL SELECT │   ENGINEERING
      └────────────── repeat the loop ───────────┴─────────────┘
```

| #   | Step                                       | What you do                                                   | Example (Movie recommender)                    |
| --- | ------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------- |
| 1   | **Frame the problem**                      | Define business goal, type of problem, metrics                | "Increase watch time" → recommendation problem |
| 2   | **Gather data**                            | CSV, database, API, web scraping                              | Watch history, ratings                         |
| 3   | **Data preprocessing**                     | Remove duplicates, handle missing values, fix outliers, scale | Remove blank ratings                           |
| 4   | **EDA** (Exploratory Data Analysis)        | Charts, stats, find patterns                                  | Which genre watched most?                      |
| 5   | **Feature engineering & selection**        | Create new useful columns, drop useless ones                  | "Avg. watch time per genre"                    |
| 6   | **Model training, evaluation & selection** | Try many algorithms, compare, pick best                       | Compare KNN vs Matrix Factorization            |
| 7   | **Model deployment**                       | Put model on server/cloud, make API                           | Flask/FastAPI + AWS                            |
| 8   | **Testing**                                | A/B test with real users, beta test                           | 10% users see new recommender                  |
| 9   | **Optimize**                               | Monitor, retrain, tune, fix drift                             | Retrain weekly                                 |

🧠 **Mnemonic (hand-drawn in notes):** **P-D-P-E-M-E-D-O** → _Plan → Data → Process → EDA → Model → Evaluate → Deploy → Optimize_.

---

## 10. Jobs & Roles in ML

```
 RAW DATA ─► Data Engineer ─► Data Analyst ─► Data Scientist ─► ML Engineer ─► PRODUCT
              (collect/store)  (insights)      (build models)    (deploy/scale)
```

### 10.1 Data Engineer 🏗️

- **Job:** Scrape data, move/store in warehouses, build data pipelines/APIs, handle databases.
- **Skills:** DSA, Java/Python/Scala/R, advanced DBMS, **Big Data tools** (Spark, Hadoop, Kafka, Hive), Cloud (AWS, GCP), distributed systems, pipelines.

### 10.2 Data Analyst 📊

- **Job:** Clean & organize raw data, find insights, make **visualizations**, produce reports, collaborate with teams, optimise data collection.
- **Skills:** Statistics, R/SAS/Python, **SQL**, **Advanced Excel**, data visualization, data storytelling, communication, business acumen (medium–high).

### 10.3 Data Scientist 🔬

- Quote: _"Better at statistics than any software engineer, and better at software engineering than any statistician."_
- **Job:** Analyse data + build ML models + give business solutions.
- **Skills:** Stats, maths, ML, programming, storytelling.

### 10.4 ML Engineer ⚙️

- **Job:** **Deploy** models to production, scale & optimise them, monitor & maintain.
- **Skills:** Maths, R/Python/Java/Scala, distributed systems, data modelling & evaluation, ML models, **software engineering & system design**.

### 10.5 Comparison

| Role               | Analytical     | Business Acumen | Data Storytelling | Soft Skills    | Software Skills |
| ------------------ | -------------- | --------------- | ----------------- | -------------- | --------------- |
| **Data Analyst**   | 🟢 High        | 🟡 Medium–High  | 🟢 High           | 🟡 Medium–High | 🟡 Medium       |
| **Data Engineer**  | 🟡 Medium      | 🔴 Low          | 🔴 Low            | 🟡 Medium      | 🟢 High         |
| **Data Scientist** | 🟢 High        | 🟢 High         | 🟢 High           | 🟢 High        | 🟡 Medium       |
| **ML Engineer**    | 🟡 Medium–High | 🟡 Medium       | 🔴 Low            | 🟢 High        | 🟢 High         |

---

## 11. Tensors

**Tensor = container for numbers** (like a box of numbers). Used everywhere in Deep Learning (TensorFlow, PyTorch).

```
 0D            1D              2D                 3D
 Scalar        Vector          Matrix             Cube of numbers
  [ 5 ]      [1, 2, 3]       [1 2 3]            ┌─────┐
                              [4 5 6]           ┌┴────┐│
                                               ┌┴────┐├┘
                                               └─────┘┘
```

| Rank (ND) | Name      | Shape example                       | Real-life example                                                                  |
| --------- | --------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| **0D**    | Scalar    | `()`                                | One number: temperature = 32                                                       |
| **1D**    | Vector    | `(3,)`                              | One student's features `[age, marks, height]`                                      |
| **2D**    | Matrix    | `(1000, 3)`                         | **Whole table / CSV**: 1000 students × 3 columns                                   |
| **3D**    | 3D tensor | `(samples, timesteps, features)`    | **Time series** (stock prices over days), **text** (sentences × words × embedding) |
| **4D**    | 4D tensor | `(samples, H, W, channels)`         | **Images**: 100 colour photos of 28×28 → `(100, 28, 28, 3)`                        |
| **5D**    | 5D tensor | `(samples, frames, H, W, channels)` | **Video**: clips × frames × H × W × RGB                                            |

**Important words:**

- **Rank** = number of axes (dimensions). Scalar = 0, Vector = 1, Matrix = 2.
- **Axes** = directions of the tensor (rows, columns, depth…).
- **Shape** = size along each axis. Matrix of 3 rows × 2 cols → shape `(3, 2)`.

```python
import numpy as np
a = np.array(5)                       # 0D  shape ()
b = np.array([1, 2, 3])               # 1D  shape (3,)
c = np.array([[1, 2], [3, 4], [5, 6]])# 2D  shape (3, 2)
print(c.ndim, c.shape)                # 2 (3, 2)
```

---

## 12. Tools Setup

| #   | Tool                     | What / Why                                                                          |
| --- | ------------------------ | ----------------------------------------------------------------------------------- |
| 1   | **Anaconda**             | Python + libraries (NumPy, Pandas, scikit-learn…) in one install. Has `conda`.      |
| 2   | **Jupyter Notebook**     | Write code in cells, see output + charts + notes together. Best for ML experiments. |
| 3   | **Virtual Environment**  | Separate "room" per project so library versions don't clash.                        |
| 4   | **Kaggle**               | Free datasets, notebooks, competitions.                                             |
| 5   | **Google Colab**         | Free cloud notebook with **free GPU/TPU**. No install.                              |
| 6   | **Kaggle data on Colab** | Download dataset from Kaggle straight into Colab using Kaggle API key.              |

```bash
# Create & use virtual env with conda
conda create -n ml_env python=3.11
conda activate ml_env
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook
```

```python
# Kaggle dataset in Colab (after uploading kaggle.json)
!pip install -q kaggle
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d <owner>/<dataset-name>
```

---

## 13. Framing an ML Problem

**Example used in class:** _Build a recommendation system for a streaming platform (like Netflix)._

```
 1 Business problem → ML problem
 2 Type of problem
 3 Current solution
 4 Getting data
 5 Metrics to measure
 6 Online vs Batch?
 7 Check assumptions
```

| #   | Step                      | Meaning                                                 | Netflix-type example                                                                                              |
| --- | ------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1   | **Business → ML problem** | Convert business goal to ML task                        | Goal: _more users stay_ → "Recommend movies each user will like"                                                  |
| 2   | **Type of problem**       | Supervised / Unsupervised? Regression / Classification? | Supervised, ranking / classification                                                                              |
| 3   | **Current solution**      | What exists today? Use as baseline                      | Manual "Top 10 trending" list                                                                                     |
| 4   | **Getting data**          | Which data signals available?                           | ① Watch time ② Searched but didn't find ③ Content left in the middle ④ Clicked on recommendations (order of recs) |
| 5   | **Metrics to measure**    | How to know model is good?                              | Click-through rate, watch time, retention                                                                         |
| 6   | **Online vs Batch?**      | Retrain daily/weekly (batch) or live (online)?          | Tastes change → maybe online/mini-batch                                                                           |
| 7   | **Check assumptions**     | Verify team's assumptions are true                      | "Do users really watch what is shown first?"                                                                      |

🧠 **Mnemonic:** **B-T-C-D-M-O-A** → _Business, Type, Current, Data, Metrics, Online/batch, Assumptions_.

---

## 14. Quick Revision Sheet

| Topic          | One-line memory                                                               |
| -------------- | ----------------------------------------------------------------------------- |
| ML             | Computer learns rules from data                                               |
| AI ⊃ ML ⊃ DL   | Big box → medium → small                                                      |
| Types          | Supervised, Unsupervised, Semi-supervised, Reinforcement                      |
| Batch          | Train once on all data, frozen after                                          |
| Online         | Learns bit by bit, always updating                                            |
| Learning rate  | Speed of forgetting old / learning new                                        |
| Out-of-core    | Data bigger than RAM → load in pieces                                         |
| Instance-based | Memorize data, compare at query time (KNN)                                    |
| Model-based    | Learn formula, discard data                                                   |
| Overfitting    | Too much memorizing → fails on new data                                       |
| Underfitting   | Too simple → fails even on training data                                      |
| MLDLC          | Frame → Data → Preprocess → EDA → Features → Model → Deploy → Test → Optimize |
| Data Engineer  | Builds pipelines                                                              |
| Data Analyst   | Finds insights, dashboards                                                    |
| Data Scientist | Builds models                                                                 |
| ML Engineer    | Deploys & scales models                                                       |
| Tensor         | Scalar(0D) → Vector(1D) → Matrix(2D) → ND                                     |

---

⭐ _Next:_ `02_...` — Data types, NumPy & Pandas basics.

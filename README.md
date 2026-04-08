# semi_supervised_gmm

Semi-supervised Gaussian mixture classification with a weighted unlabeled likelihood.

Companion to the paper:

> **Semi-Supervised Generative Classification via a Weighted Unlabeled Likelihood**
> *(under review)*

---

## Overview

Many classification problems have cheap features but expensive labels. This package fits a two-component Gaussian mixture model by maximising a *weighted* log-likelihood:

```
J(θ) = ℓ_sup(θ)  +  λ · ℓ_unl(θ)
```

where `ℓ_sup` is the supervised log-likelihood over labeled data and `ℓ_unl` is the unlabeled marginal mixture log-likelihood. The scalar `λ` controls how much the unlabeled corpus influences the fit. EM yields closed-form updates at every iteration.

The central contribution is treating `λ` as an object of study in its own right:

- **λ = 0** recovers purely supervised MLE.
- **λ > 0** borrows geometric structure from the unlabeled distribution.
- **Pre-fitting diagnostic** `A(0)`: a computable score whose sign predicts whether adding unlabeled data will improve or degrade the estimator — before any semi-supervised fitting is done.

---

## Tutorial notebook

An end-to-end walkthrough using `sklearn.datasets.load_breast_cancer()` (569 samples, 30 features):

```bash
jupyter notebook notebooks/tutorial.ipynb
```

Covers all four estimators, the λ path, the pre-fitting diagnostic, and a comparative AUROC summary.
All plots use a dark theme consistent with the paper's visual style.

---

## Installation

```bash
# from repo root
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `scikit-learn` (base classes only)

---

## Quick start

```python
import numpy as np
from semi_supervised_gmm import SemiSupervisedGMM, make_semi_supervised

rng = np.random.default_rng(0)

# Generate data: y=1 (positive), y=0 (negative), y=-1 (unlabeled)
X_pos = rng.multivariate_normal([2, 0], np.eye(2), 30)
X_neg = rng.multivariate_normal([-2, 0], np.eye(2), 30)
X_u   = rng.multivariate_normal([0, 0], np.eye(2), 300)

X, y = make_semi_supervised(
    np.vstack([X_pos, X_neg]),
    np.array([1]*30 + [0]*30),
    X_u,
)

# Fit
model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)

# Predict
proba  = model.predict_proba(X_test)   # shape (n, 2): [P(y=0|x), P(y=1|x)]
labels = model.predict(X_test)
auc    = model.score(X_test, y_test)   # AUROC
```

---

## Benchmark results

AUROC averaged over 10 random seeds ± std. `covariance_type="ledoit_wolf"`, StandardScaler.  
All four semi-supervised variants shown at N_labeled=40 (20 per class); Parkinsons at N_labeled=20 (15-per-class test due to small positive class).

| Dataset | d | N | N_lab | Supervised | Semi (λ=1) | Learned-λ | Local-λ | Δ best |
|---------|--:|--:|------:|:----------:|:----------:|:---------:|:-------:|:------:|
| Breast Cancer | 30 | 569 | 40 | 0.869±0.049 | 0.971±0.018 | 0.960±0.034 | 0.973±0.018 | **+0.105** |
| Ionosphere | 34 | 351 | 80 | 0.825±0.126 | 0.957±0.012 | 0.952±0.022 | 0.956±0.012 | **+0.132** |
| Heart Disease | 13 | 270 | 40 | 0.780±0.106 | 0.874±0.047 | 0.831±0.096 | 0.872±0.049 | **+0.094** |
| Parkinsons | 22 | 195 | 20 | 0.785±0.065 | 0.808±0.046 | 0.815±0.058 | 0.811±0.043 | +0.030 |
| Sonar | 60 | 208 | 40 | 0.926±0.000 | 0.741±0.131 | 0.767±0.139 | 0.740±0.130 | −0.159 |

**When it helps:** In the label-scarce regime (10–40 labeled samples per class), all three semi-supervised variants consistently improve AUROC by **+0.03 to +0.13** on datasets where the Gaussian mixture structure is reasonable (Breast Cancer, Ionosphere, Heart Disease). The choice of variant matters little — `Semi(λ=1)` and `Local-λ` are typically tied for best.

**When it doesn't:** Sonar (d/N_per_class = 6) is the clear failure case. The feature-to-label ratio is far above the threshold identified in the paper's high-dimensional alignment analysis: unlabeled data estimate a 60-dimensional covariance that doesn't align with the labeled decision boundary, and `‖g₀‖` is large, correctly flagging the diagnostic as `"unreliable"`.

**Stability bonus:** At Ionosphere N_labeled=80, the supervised estimator is highly variable (std=0.126); semi-supervised stabilises it to std=0.012 — a variance-reduction effect separate from the bias improvement.

---

## Four estimators

All estimators follow the sklearn interface: `fit` / `predict` / `predict_proba` / `score`.
Use `y = -1` as the unlabeled sentinel, matching `sklearn.semi_supervised.LabelPropagation`.

### `SupervisedGMM`

Purely supervised MLE (λ = 0). Ignores unlabeled observations.

```python
from semi_supervised_gmm import SupervisedGMM

model = SupervisedGMM().fit(X, y)
```

### `SemiSupervisedGMM`

Fixed global λ. Grid-search on a validation set via `fit_cv`.

```python
from semi_supervised_gmm import SemiSupervisedGMM

# Fixed lambda
model = SemiSupervisedGMM(lambda_=2.0).fit(X, y)

# Grid-searched lambda
model = SemiSupervisedGMM().fit_cv(X, y, X_val, y_val,
                                   lam_grid=np.logspace(-2, 2, 20))
print(model.lambda_used_)   # selected value
```

Compatible with `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(SemiSupervisedGMM(),
                  {"lambda_": [0.1, 0.5, 1.0, 2.0, 5.0]},
                  scoring="accuracy")
gs.fit(X_labeled, y_labeled)
```

### `LearnedLambdaGMM`

Learns λ by gradient ascent on validation log-likelihood (IFT-based, as described in the paper).

```python
from semi_supervised_gmm import LearnedLambdaGMM

model = LearnedLambdaGMM(lambda_init=1.0, n_steps=10).fit(
    X, y, X_val=X_val, y_val=y_val
)
print(model.lambda_)   # learned value
```

### `LocalLambdaGMM`

Per-point confidence-weighted λ: `λ(x) = λ · max(γ(x), 1−γ(x))^α`. Downweights ambiguous unlabeled points.

```python
from semi_supervised_gmm import LocalLambdaGMM

model = LocalLambdaGMM(lambda_=1.0, alpha=1.0).fit(X, y)
```

---

## Pre-fitting diagnostic: should you use unlabeled data?

The **alignment coefficient** `A(0)` is computable before any semi-supervised fitting. Its sign predicts whether unlabeled data will help:

```python
from semi_supervised_gmm import SemiSupervisedGMM

result = SemiSupervisedGMM.alignment_score(X_train, y_train, X_val, y_val)
# {
#   "A0":             float,           # alignment coefficient
#   "g0_norm":        float,           # ||g_0|| score residual norm
#   "recommendation": "use"|"discard"|"unreliable",
#   "n_unlabeled":    int
# }

if result["recommendation"] == "use":
    model = SemiSupervisedGMM(lambda_=1.0).fit(X_train, y_train)
else:
    model = SupervisedGMM().fit(X_train, y_train)
```

**Interpretation:**
- `A(0) > 0` → unlabeled geometry is aligned with the classification task → use semi-supervised learning
- `A(0) < 0` → unlabeled geometry is misaligned → stick with supervised MLE
- `g0_norm` large (> 15) → Gaussian assumption likely violated → diagnostic unreliable

In the small-labeled-data regime (where semi-supervised learning is most consequential), the sign of `A(0)` achieves **65–72% decision accuracy** with mean regret below 0.01 AUROC.

---

## Covariance options

All estimators accept `covariance_type`:

| Value | When to use |
|---|---|
| `"full"` (default) | N ≫ d; unrestricted covariance |
| `"diag"` | d > N or when features are approximately independent |
| `"ledoit_wolf"` | d ≈ N; shrinkage toward scaled identity (requires sklearn) |

```python
model = SemiSupervisedGMM(lambda_=1.0, covariance_type="ledoit_wolf").fit(X, y)
```

---

## Convenience: split-data interface

All estimators expose `fit_semi` for callers who already have labeled and unlabeled data in separate arrays:

```python
model = SemiSupervisedGMM(lambda_=1.0).fit_semi(X_labeled, y_labeled, X_unlabeled)
```

---

## Persistence

```python
model.save("model.pkl")
loaded = SemiSupervisedGMM.load("model.pkl")
```

---

## Replicating paper results

All paper figures and tables can be regenerated using the production package:

```bash
# All experiments
python3 run_paper_experiments.py --experiments all

# Individual experiments
python3 run_paper_experiments.py --experiments e1 e3 e10
```

Experiments available: `e1` (diagnostic validity), `e2` (lambda learning), `e3` (misspecification), `e4` (bias-variance), `e5` (local lambda trajectory), `e6` (lambda parameter path), `e7` (decision accuracy), `e8` (confidence-weighting ablation), `e9` (regime grid), `e10` (high-dimensional geometry), `pr` (parameter recovery), `sens` (sensitivity).

---

## Algorithmic notes

The implementation improves on the reference code in three ways:

**1. Cholesky caching.** Each E-step uses a single Cholesky factorisation per class per iteration (the reference performs two O(d³) factorisations: `slogdet` + `solve`). ~1.5× E-step speedup.

**2. Precomputed labeled scatter.** `X_pos^T X_pos` and `X_pos.sum(0)` are computed once outside the EM loop. The centered scatter is recovered via:

```
(X − μ)ᵀ(X − μ) = XᵀX − outer(Σx, μ) − outer(μ, Σx) + N·outer(μ, μ)
```

Note: the simpler form `XᵀX − N·outer(μ,μ)` only holds when μ = x̄. In EM, μ_new includes unlabeled contributions, so the full identity is required.

**3. Analytic Fisher for `A(0)`.** The reference computes the Fisher information matrix via numerical Jacobians — O(N·d⁴). The production implementation uses the closed-form mean-block Fisher:

```
F_μ = Σ⁻¹ · S_sample · Σ⁻¹
```

Cost: O(N·d²). At d=50 this is a ~5000× reduction and eliminates floating-point error that degrades diagnostic accuracy at high dimension.

---

## Package structure

```
semi_supervised_gmm/
    _params.py        GMMParams dataclass
    _em.py            Pure NumPy EM loop, E/M-step, posterior (zero sklearn)
    _lambda.py        grid_search_lambda, gradient_lambda
    _diagnostics.py   alignment_A0, score_residual_g0 — analytic Fisher
    _data.py          encode_labels: y=-1 sentinel → (X_pos, X_neg, X_u)
    _base.py          BaseGMM: sklearn mixins, predict*, score, persistence
    _estimators.py    Four estimator classes
    exceptions.py     ConvergenceWarning, InsufficientLabeledDataError
```

---

## Running tests

```bash
python3 -m pytest tests/ -q
```

61 tests: unit tests for EM numerics, data encoding, and diagnostics; integration tests for all four estimators including numerical agreement with the reference implementation, `GridSearchCV` compatibility, and save/load roundtrip.

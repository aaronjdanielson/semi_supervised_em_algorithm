# CLAUDE.md — Semi-Supervised EM Algorithm Project

## What this project is

A statistics paper + PyPI package called `semi_supervised_em` (or `semi_supervised_em_algorithm`).

**Paper title:** *Semi-Supervised Generative Classification via a Weighted Unlabeled Likelihood*
**Paper source:** `discovered_materials/papers/semi_supervised_em.tex`

The algorithm fits a two-component Gaussian mixture model by maximizing a weighted log-likelihood:

    J(theta) = l_sup(theta) + lambda * l_unl(theta)

where `l_sup` is the labeled log-likelihood (positives D1 + negatives D0) and `l_unl` is the
unlabeled marginal mixture log-likelihood. EM yields closed-form updates. `lambda` is the
conceptual spine of the paper — it is treated as a diagnostic, a tunable parameter, and an
object of study in its own right.

**Applied origins:** ICBC insurance (note-to-claimant matching), NBA RPM sports analytics.

---

## Paper structure (key sections)

- **Method:** Problem setting, weighted objective, EM algorithm, confidence-weighted extension `lambda(x)`, lambda selection and learning
- **Theory:** EM monotonicity, effective sample size, pseudo-true parameter path, restricted mean-subspace path, bounded damage under confidence weighting, high-dimensional alignment
- **Simulations:** Four regimes (R1 unsupervised, R2 small-sample supervised, R3 global-lambda semi-supervised, R4 confidence-weighted), alignment diagnostic, bias-variance tradeoff, lambda selection, misspecification, high-dimensional geometry
- **Empirical:** UCI benchmarks, notes-to-claimants matching, second application (TODO)

---

## Core theoretical objects and notation

| Symbol | Meaning |
|---|---|
| `theta` | Model parameters `(pi, mu0, mu1, Sigma0, Sigma1)` |
| `lambda` | Unlabeled weight scalar (or function `lambda(x)`) |
| `theta*(lambda)` | Pseudo-true parameter path as function of lambda |
| `H` | Hessian of the full weighted objective at `theta*` |
| `g_0` | Unlabeled score residual = gradient of `l_unl` at supervised MLE |
| `d theta*/d lambda` | `-H^{-1} g_0` — IFT characterization of parameter path |
| `A(0)` | Alignment coefficient = `grad l_V(theta_hat)^T (-H^{-1} g_0)` |
| `calV` | Validation set log-likelihood |
| `N_tilde` | Weighted soft counts = `N_obs + lambda * N_u` |

**Key macro definitions** (already in the .tex preamble):
`\lam`, `\Sig`, `\calJ`, `\calU`, `\calD`, `\calV`, `\calL`, `\calA`, `\calN`, `\LSE`, `\todo{}`

---

## The alignment coefficient A(0) — central diagnostic

`A(0)` is computable before fitting. Its sign predicts whether adding unlabeled data helps:
- `A(0) > 0`: unlabeled data will improve the estimator (use lambda > 0)
- `A(0) < 0`: unlabeled data will degrade the estimator (stick with supervised MLE)

Empirical accuracy: 65–72% in the small-label regime; regret < 0.01 AUROC.
Frame as a **decision support tool**, not a universal oracle.

---

## High-dimensional alignment — external reviewer's contribution

An external reviewer proposed a subsection (already partially in the paper at §Theory line ~1109
and §Simulations line ~2020) that sharpens the high-dimensional story:

**Signal/nuisance decomposition:**

    g_0 = g_parallel + g_perp

where `g_parallel` is the component of the unlabeled score aligned with the task-relevant direction
`v*`, and `g_perp` is orthogonal nuisance.

**High-dimensional scaling:** When class separation lives in a k-dimensional subspace of R^d:

    ||g_parallel|| = O(k),   ||g_perp|| = O(d - k)

**Phase transition:**
- `k / d` large → `g_parallel` dominates → `A(0) > 0` → unlabeled data help
- `k / d` small → `g_perp` dominates → `A(0) < 0` → unlabeled data hurt

**Interpretation:** Unlabeled data estimate the geometry of p(x). The semi-supervised estimator
shrinks toward structures in the unlabeled distribution. This is beneficial iff those structures
align with the classification task.

**Key sentence the reviewer flagged as worth adding:**
> "This phenomenon is analogous to classical bias–variance tradeoffs, but with the additional
> feature that the bias induced by unlabeled data is itself structured and dimension-dependent."

When working on the high-dimensional subsections, check that the theory (§Theory) and simulations
(§Simulations: High-Dimensional Geometry) lock together cleanly — the experiment should mirror
the subsection exactly (dimension sweep varying k and d).

---

## Codebase

Two implementations exist:

| | disco version | project version |
|---|---|---|
| File | `discovered_materials/code/semi_supervised_em_code.py` | `em_functions.py`, `run_em_algorithm.py` |
| Interface | `GMMParams` dataclass | separate functions |
| Log-space | yes | no |
| Warm start | supervised MLE | unclear |
| **Use for PyPI** | **YES — this is the production core** | reference only |

Other files:
- `semi_supervised_k_means.py` — COP-Kmeans constrained variant (separate algorithm, related)
- `scrape_rpm_espn.py` — NBA RPM scraper (2017–2024 data in `discovered_materials/data/`)
- `discovered_materials/code/simulations.py`, `real_data_experiments.py` — simulation code

---

## Paper tone and style

- Precise and restrained — write like a statistics journal paper
- Propositions + interpretations rather than full proofs where appropriate
- No inflation: the alignment coefficient is a decision support tool, not an oracle
- The paper's contribution is the analysis of `lambda` as an object of study, not just a tuning knob

---

## Active TODOs (as of April 2026)

### Paper content (missing sections / tables)
- [ ] **§Misspecification** (tex line ~2217): needs a results table — AUROC for R1–R4 under Student-t (ν=3), skew-normal, and 4-component misspecification scenarios, plus `||g_hat_0||` column as pre-fitting risk indicator
- [ ] **§Within-Group Competition Scoring** (tex line ~2282): needs a results table — top-1 accuracy, recall@2, MRR for R2/R3/R4 across K ∈ {2, 5, 10}; R1 excluded
- [ ] **§Notes-to-Claimants Matching** (tex line ~2350): entire section is TODO — dataset description, feature construction, label source, evaluation (AUROC, AUPRC, top-1, recall@k), lambda selection procedure
- [ ] **§Second Application** (tex line ~2359): entirely TODO

### Figures
- [ ] **`figures/fig_real_data_emp.pdf`** is missing — §Empirical Applications (Cross-Dataset Benchmark) now correctly references this separate file (distinct from `fig_real_data.pdf` used in §Simulations). Needs to be generated from the empirical benchmark experiment.

### Theory / alignment
- [ ] Verify high-dimensional theory subsection (§Theory ~line 1109) matches external reviewer's sharper signal/nuisance decomposition version
- [ ] Dimension sweep simulation (§Simulations ~line 2020) should mirror the theory subsection exactly (vary k and d)

### Package
- [ ] PyPI package unification (disco implementation as core)

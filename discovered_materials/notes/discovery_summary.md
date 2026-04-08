# Discovery Summary: Semi-Supervised EM Algorithm Materials

## What Was Found and Where

### 1. disco/semi_supervised_em/ — Core Paper + Clean Implementation

**Paper**: `semi_supervised_em.tex` / `semi_supervised_em.pdf`
- Title: *Semi-Supervised Gaussian Mixture with Weighted Unlabeled Likelihood*
- Full formal derivation of the algorithm
- Partition: labeled positives D1, labeled negatives D0, unlabeled U
- Model: two Gaussian components, z=1 (positive) / z=0 (negative), mixing weight pi
- Weighted objective: J(theta) = supervised log-likelihood + lambda * unlabeled mixture log-likelihood
- lambda=0 recovers purely supervised MLE; large lambda lets unlabeled data dominate
- E-step: update responsibilities gamma_j = Pr(z=1 | x_j^u; theta) via Bayes' rule, computed in log-space
- M-step: weighted sufficient stats (N_tilde = N_obs + lambda * Nu), update pi, mu, Sigma for each class
- Applied context noted: matching notes to claimants (ICBC insurance claim context)

**Code**: `semi_supervised_em_code.py`
- Clean, modern Python implementation using dataclasses
- GMMParams dataclass (pi, mu0, mu1, Sigma0, Sigma1)
- _logpdf_mvn(): numerically stable log-space MVN PDF via slogdet + linear solve
- _soft_counts(): computes weighted N_tilde counts
- em_semisup_gmm(): main EM with supervised MLE warm start, convergence on inf-norm of param delta
- posterior_prob(): inference function returning Pr(z=1|x) in log-space
- This is the more production-ready version of the algorithm

**Notebook**: `experiment.ipynb` — experiments with the above

### 2. semi_supervised_em_algorithm/ (current project) — Earlier Implementation

**em_functions.py**: Lower-level function-based implementation
- est_mult_gaus(): multivariate normal PDF (uses scipy, less numerically stable)
- get_zhats(): E-step posterior probabilities
- get_mu(), get_sigma(), get_pi(): M-step parameter updates with lambda weighting

**semi_supervised_k_means.py**: Constrained k-means (different algorithm)
- COP-Kmeans with must-link and cannot-link constraints
- k-means++ initialization
- Related but distinct from the Gaussian mixture EM

**run_em_algorithm.py**: Orchestrates the EM loop (E-step + M-step + convergence check)

**main.py, example_1.py**: Entry points / examples

**scrape_rpm_espn.py / .ipynb**: NBA Real Plus-Minus data scraper (2017–2024)
- The RPM CSVs in data/ are the empirical dataset for the sports analytics use case

### 3. ICBC_readings/ — Related Prior Work (R)

**collapsed_gibbs_mcem_slda.R**: Monte Carlo EM for supervised LDA
- Related: uses MCEM (a variant of EM) for topic models with supervision
- Reference for the MCEM connection in related work section

**collapsed_gibbs_gaussian_slda.R**: Gaussian supervised LDA
- Gaussian mixture components in a topic model context
- Related work on Gaussian latent variable models

**rtm.em.R**: EM for Relational Topic Model
- Another EM-based topic model reference

### 4. PatternsEmerge/icbc/ — Not Directly Relevant
Presentations about ordinal neural networks, model claim tactics — ICBC work context but not about semi-supervised EM.

---

## Key Insight: Two Code Versions Exist

| Feature | disco version (semi_supervised_em_code.py) | project version (em_functions.py) |
|---|---|---|
| Interface | GMMParams dataclass | separate functions |
| Log-space | yes (slogdet + solve) | no (scipy PDF) |
| Warm start | supervised MLE | unclear |
| Stability | ridge + resp. clipping | sys.float_info.min |
| Convergence | inf-norm on params | cutoff threshold |

**Recommendation**: The disco version is the more polished codebase. The PyPI package should unify these, likely building on the disco implementation as the core.

---

## Subfolder Structure of discovered_materials/

- **papers/**: LaTeX source + PDF of the formal paper
- **code/**: All Python and R implementations
  - semi_supervised_em_code.py — disco clean version
  - em_functions.py, run_em_algorithm.py — project version
  - semi_supervised_k_means.py — constrained k-means variant
  - collapsed_gibbs_mcem_slda.R, collapsed_gibbs_gaussian_slda.R — MCEM/Gaussian LDA references
  - rtm.em.R — RTM EM reference
- **notebooks/**: experiment.ipynb (disco), scrape_rpm_espn.ipynb (NBA data)
- **data/**: NBA RPM databases 2017–2024 (empirical dataset)
- **notes/**: this file

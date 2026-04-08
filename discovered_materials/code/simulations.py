"""
simulations.py  —  Evidence layer for the paper
"Semi-Supervised Generative Classification via a Weighted Unlabeled Likelihood"

Three main experiments:
  E1. Diagnostic validity: A(0) and |g_0| predict whether unlabeled data help
  E2. λ-learning (gradient ascent) vs. grid search
  E3. Misspecification story: |g_0| → degradation, confidence-weighting reduces damage

Outputs (to discovered_materials/papers/figures/):
  fig_diagnostic.pdf   — scatter: A(0) vs AUROC gain
  fig_lambda_learn.pdf — bar/scatter: gradient-λ vs grid-λ vs performance
  fig_misspec.pdf      — scatter: |g_0| vs AUROC degradation (R3 vs R2)

Also prints LaTeX table source for the paper.

Dependencies: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
from numpy.linalg import slogdet, solve, norm
from scipy.stats import multivariate_normal, t as student_t
from dataclasses import dataclass
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGDIR = Path(__file__).parent.parent / "papers" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Core algorithm (from disco: semi_supervised_em_code.py)
# ---------------------------------------------------------------------------

@dataclass
class GMMParams:
    pi: float
    mu0: np.ndarray
    mu1: np.ndarray
    Sigma0: np.ndarray
    Sigma1: np.ndarray


def _logpdf_mvn(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance not PD.")
    Xc = X - mu
    A = solve(Sigma, Xc.T).T
    quad = np.einsum("ij,ij->i", Xc, A)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def _soft_counts(gamma, N1, N0, lam, Nu):
    N1t = N1 + lam * np.sum(gamma)
    N0t = N0 + lam * np.sum(1.0 - gamma)
    Nt  = N1 + N0 + lam * Nu
    return N1t, N0t, Nt

def sample_gaussian_mixture(rng, mu1, mu0, S1, S0, z):
    """Sample from a Gaussian mixture; z[i]=1 → class 1, z[i]=0 → class 0."""
    n, d = len(z), len(mu1)
    X    = np.empty((n, d))
    mask = z.astype(bool)
    n1, n0 = int(mask.sum()), int((~mask).sum())
    if n1:
        X[mask]  = rng.multivariate_normal(mu1, S1, n1)
    if n0:
        X[~mask] = rng.multivariate_normal(mu0, S0, n0)
    return X



def em_semisup(
    X_pos, X_neg, X_u,
    lam=1.0, tol=5e-6, max_iter=300,
    eps_cov=1e-6, resp_clip=1e-6,
) -> GMMParams:
    N1, d = X_pos.shape
    N0 = X_neg.shape[0]
    Nu = X_u.shape[0]

    # supervised MLE warm start
    mu1 = X_pos.mean(0)
    mu0 = X_neg.mean(0)
    S1  = np.cov(X_pos, rowvar=False) + eps_cov * np.eye(d)
    S0  = np.cov(X_neg, rowvar=False) + eps_cov * np.eye(d)
    pi  = N1 / (N1 + N0)

    for _ in range(max_iter):
        # E-step
        lp1 = _logpdf_mvn(X_u, mu1, S1) + np.log(np.clip(pi, 1e-12, 1 - 1e-12))
        lp0 = _logpdf_mvn(X_u, mu0, S0) + np.log(np.clip(1 - pi, 1e-12, 1 - 1e-12))
        m   = np.maximum(lp1, lp0)
        logden = m + np.log(np.exp(lp1 - m) + np.exp(lp0 - m))
        gamma = np.clip(np.exp(lp1 - logden), resp_clip, 1 - resp_clip)

        # M-step
        N1t, N0t, Nt = _soft_counts(gamma, N1, N0, lam, Nu)
        mu1n = (X_pos.sum(0) + lam * (gamma[:, None] * X_u).sum(0)) / N1t
        mu0n = (X_neg.sum(0) + lam * ((1 - gamma)[:, None] * X_u).sum(0)) / N0t

        Xp_c = X_pos - mu1n
        Xu1  = X_u   - mu1n
        S1n  = (Xp_c.T @ Xp_c + lam * (Xu1 * gamma[:, None]).T @ Xu1) / N1t + eps_cov * np.eye(d)

        Xn_c = X_neg - mu0n
        Xu0  = X_u   - mu0n
        S0n  = (Xn_c.T @ Xn_c + lam * (Xu0 * (1 - gamma)[:, None]).T @ Xu0) / N0t + eps_cov * np.eye(d)

        pi_n = N1t / Nt

        delta = max(
            np.max(np.abs(mu1n - mu1)), np.max(np.abs(mu0n - mu0)),
            np.max(np.abs(S1n - S1)),   np.max(np.abs(S0n - S0)),
            abs(pi_n - pi),
        )
        mu1, mu0, S1, S0, pi = mu1n, mu0n, S1n, S0n, pi_n
        if delta < tol:
            break

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)


def em_supervised(X_pos, X_neg, eps_cov=1e-6) -> GMMParams:
    return em_semisup(X_pos, X_neg, np.zeros((0, X_pos.shape[1])), lam=0.0,
                      eps_cov=eps_cov)


def posterior(X: np.ndarray, p: GMMParams) -> np.ndarray:
    lp1 = _logpdf_mvn(X, p.mu1, p.Sigma1) + np.log(np.clip(p.pi, 1e-12, 1 - 1e-12))
    lp0 = _logpdf_mvn(X, p.mu0, p.Sigma0) + np.log(np.clip(1 - p.pi, 1e-12, 1 - 1e-12))
    m   = np.maximum(lp1, lp0)
    den = m + np.log(np.exp(lp1 - m) + np.exp(lp0 - m))
    return np.exp(lp1 - den)


# ---------------------------------------------------------------------------
# Confidence-weighted EM  (λ(x) = λ · max_g(γ_g)^α)
# ---------------------------------------------------------------------------

def em_conf_weighted(
    X_pos, X_neg, X_u,
    lam=1.0, alpha=1.0, tol=5e-6, max_iter=300, eps_cov=1e-6, resp_clip=1e-6,
) -> GMMParams:
    N1, d = X_pos.shape
    N0 = X_neg.shape[0]
    Nu = X_u.shape[0]

    mu1 = X_pos.mean(0)
    mu0 = X_neg.mean(0)
    S1  = np.cov(X_pos, rowvar=False) + eps_cov * np.eye(d)
    S0  = np.cov(X_neg, rowvar=False) + eps_cov * np.eye(d)
    pi  = N1 / (N1 + N0)

    for _ in range(max_iter):
        lp1 = _logpdf_mvn(X_u, mu1, S1) + np.log(np.clip(pi, 1e-12, 1 - 1e-12))
        lp0 = _logpdf_mvn(X_u, mu0, S0) + np.log(np.clip(1 - pi, 1e-12, 1 - 1e-12))
        m   = np.maximum(lp1, lp0)
        logden = m + np.log(np.exp(lp1 - m) + np.exp(lp0 - m))
        gamma = np.clip(np.exp(lp1 - logden), resp_clip, 1 - resp_clip)

        # confidence weight per point
        conf = np.maximum(gamma, 1 - gamma) ** alpha   # shape (Nu,)
        lam_j = lam * conf                              # effective per-point weight

        N1t = N1 + np.sum(lam_j * gamma)
        N0t = N0 + np.sum(lam_j * (1 - gamma))
        Nt  = N1 + N0 + np.sum(lam_j)

        mu1n = (X_pos.sum(0) + ((lam_j * gamma)[:, None] * X_u).sum(0)) / N1t
        mu0n = (X_neg.sum(0) + ((lam_j * (1-gamma))[:, None] * X_u).sum(0)) / N0t

        Xp_c = X_pos - mu1n;  Xu1 = X_u - mu1n
        S1n  = (Xp_c.T @ Xp_c + (Xu1 * (lam_j * gamma)[:, None]).T @ Xu1) / N1t + eps_cov * np.eye(d)

        Xn_c = X_neg - mu0n;  Xu0 = X_u - mu0n
        S0n  = (Xn_c.T @ Xn_c + (Xu0 * (lam_j * (1-gamma))[:, None]).T @ Xu0) / N0t + eps_cov * np.eye(d)

        pi_n = N1t / Nt

        delta = max(
            np.max(np.abs(mu1n - mu1)), np.max(np.abs(mu0n - mu0)),
            np.max(np.abs(S1n - S1)),   np.max(np.abs(S0n - S0)),
            abs(pi_n - pi),
        )
        mu1, mu0, S1, S0, pi = mu1n, mu0n, S1n, S0n, pi_n
        if delta < tol:
            break

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def brier(scores, labels):
    return np.mean((scores - labels) ** 2)


# ---------------------------------------------------------------------------
# Score residual g_0 and alignment coefficient A(0)
# ---------------------------------------------------------------------------

def score_residual_g0(X_pos, X_neg, X_u, eps_cov=1e-6):
    """
    g_0 = ∇_θ ℓ_unl(θ*(0))  evaluated at the supervised MLE.
    Returns a flat vector of the gradient and its norm.

    Parameterisation: θ = (π, μ1, μ0, vec(Σ1), vec(Σ0))
    We compute the gradient numerically via finite differences — cleaner
    and avoids messy matrix-calculus bookkeeping in this script.
    """
    sup = em_supervised(X_pos, X_neg, eps_cov=eps_cov)

    def unl_ll(p: GMMParams):
        lp1 = _logpdf_mvn(X_u, p.mu1, p.Sigma1) + np.log(np.clip(p.pi, 1e-12, 1 - 1e-12))
        lp0 = _logpdf_mvn(X_u, p.mu0, p.Sigma0) + np.log(np.clip(1 - p.pi, 1e-12, 1 - 1e-12))
        m = np.maximum(lp1, lp0)
        # Mean (not sum) → g0 is per-observation, scale-invariant w.r.t. N_u
        return np.mean(m + np.log(np.exp(lp1 - m) + np.exp(lp0 - m)))

    # pack/unpack θ
    d = sup.mu1.shape[0]

    def pack(p):
        return np.concatenate([[p.pi], p.mu1, p.mu0, p.Sigma1.ravel(), p.Sigma0.ravel()])

    def unpack(v):
        idx = 0
        pi  = float(v[idx]); idx += 1
        mu1 = v[idx:idx+d];  idx += d
        mu0 = v[idx:idx+d];  idx += d
        S1  = v[idx:idx+d*d].reshape(d, d); idx += d*d
        S0  = v[idx:idx+d*d].reshape(d, d)
        return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)

    theta0 = pack(sup)
    h = 1e-5
    g0 = np.zeros_like(theta0)
    f0 = unl_ll(sup)
    for i in range(len(theta0)):
        th = theta0.copy(); th[i] += h
        g0[i] = (unl_ll(unpack(th)) - f0) / h

    return g0, norm(g0)


def _make_pack_unpack(d):
    def pack(p):
        return np.concatenate([[p.pi], p.mu1, p.mu0, p.Sigma1.ravel(), p.Sigma0.ravel()])
    def unpack(v):
        idx = 0
        pi  = float(v[idx]); idx += 1
        mu1 = v[idx:idx+d];   idx += d
        mu0 = v[idx:idx+d];   idx += d
        S1  = v[idx:idx+d*d].reshape(d, d); idx += d*d
        S0  = v[idx:idx+d*d].reshape(d, d)
        return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)
    return pack, unpack


def fisher_sup_ll(X_pos, X_neg, eps_cov=1e-6):
    """
    Empirical Fisher information F of ℓ_sup(θ) at θ*(0):
      F = Σ_i (∇_θ log p_i) (∇_θ log p_i)^T

    H_sup ≈ -F  (negative-definite by construction), so H_sup^{-1} = -F^{-1}.

    This avoids numerical positive-eigenvalue artifacts from the flat Σ
    parameterisation in finite-difference Hessians.  The empirical Fisher is
    exact at the MLE under standard regularity (Bartlett identities).
    """
    sup = em_supervised(X_pos, X_neg, eps_cov=eps_cov)
    d   = sup.mu1.shape[0]
    pack, unpack = _make_pack_unpack(d)
    theta0 = pack(sup)
    n = len(theta0)
    h = 1e-5

    def score_i(x, cls):
        """Score vector for one observation."""
        def ll_single(v):
            p = unpack(v)
            if cls == 1:
                return (_logpdf_mvn(x[None], p.mu1, p.Sigma1)[0]
                        + np.log(np.clip(p.pi, 1e-12, 1-1e-12)))
            else:
                return (_logpdf_mvn(x[None], p.mu0, p.Sigma0)[0]
                        + np.log(np.clip(1-p.pi, 1e-12, 1-1e-12)))
        f0 = ll_single(theta0)
        g  = np.zeros(n)
        for k in range(n):
            th = theta0.copy(); th[k] += h
            g[k] = (ll_single(th) - f0) / h
        return g

    F = np.zeros((n, n))
    for x in X_pos:
        s = score_i(x, 1); F += np.outer(s, s)
    for x in X_neg:
        s = score_i(x, 0); F += np.outer(s, s)
    return F  # H_sup = -F


def val_score_gradient(X_pos_val, X_neg_val, theta0_packed, d):
    """∇_θ ℓ_V(θ*(0)) — gradient of supervised log-likelihood on validation set."""
    _, unpack = _make_pack_unpack(d)

    def val_ll(p):
        ll1 = _logpdf_mvn(X_pos_val, p.mu1, p.Sigma1)
        ll0 = _logpdf_mvn(X_neg_val, p.mu0, p.Sigma0)
        return np.sum(ll1) + np.sum(ll0)

    h = 1e-5
    n = len(theta0_packed)
    f0 = val_ll(unpack(theta0_packed))
    grad = np.zeros(n)
    for i in range(n):
        th = theta0_packed.copy(); th[i] += h
        grad[i] = (val_ll(unpack(th)) - f0) / h
    return grad


def alignment_A0(X_pos, X_neg, X_u, X_pos_val, X_neg_val, eps_cov=1e-6):
    """
    A(0) restricted to mean parameters (mu1, mu0).

    The full IFT formula is A(0) = (F^{-1} g_0) · grad_val over all θ.
    In practice the covariance blocks introduce cancelling large-magnitude
    contributions driven by the biased-label geometry rather than the
    boundary-shift mechanism.  We therefore compute A(0) over the 2d-dimensional
    mean subspace:

        A(0) ≈ (F_mu^{-1} g0_mu) · grad_val_mu

    where F_mu is the 2d×2d diagonal mean block of the Fisher matrix,
    g0_mu is the mean component of the unlabeled score residual, and
    grad_val_mu is the mean component of the validation gradient.

    This is the exact formula for the restricted model in which Σ and π are
    fixed at their supervised estimates — i.e. the diagnostic isolates boundary
    shift through the means, which is the mechanism that controls classification.

    Returns A0 (scalar) and g0_norm (full gradient norm).
    """
    g0, g0_norm = score_residual_g0(X_pos, X_neg, X_u, eps_cov=eps_cov)
    F = fisher_sup_ll(X_pos, X_neg, eps_cov=eps_cov)

    sup = em_supervised(X_pos, X_neg, eps_cov=eps_cov)
    d   = sup.mu1.shape[0]
    pack, _ = _make_pack_unpack(d)
    theta0 = pack(sup)

    # Mean parameter indices: theta = [pi(1), mu1(d), mu0(d), Sigma1(d^2), Sigma0(d^2)]
    mu_idx = slice(1, 1 + 2 * d)

    F_mu  = F[mu_idx, mu_idx] + 1e-6 * np.eye(2 * d)
    g0_mu = g0[mu_idx]

    grad_val = val_score_gradient(X_pos_val, X_neg_val, theta0, d)
    gv_mu = grad_val[mu_idx]

    try:
        F_inv_g0_mu = solve(F_mu, g0_mu)
    except np.linalg.LinAlgError:
        F_inv_g0_mu = np.linalg.lstsq(F_mu, g0_mu, rcond=None)[0]

    A0 = float(F_inv_g0_mu @ gv_mu)
    return A0, g0_norm


# Alias for import by real_data_experiments.py
def alignment_coefficient_mu(X_pos, X_neg, X_u, X_val, y_val, eps_cov=1e-6):
    """
    Wrapper around alignment_A0 that accepts a pooled validation set.
    Returns A0 scalar (g0_norm dropped).
    """
    X_pos_val = X_val[y_val == 1]
    X_neg_val = X_val[y_val == 0]
    if len(X_pos_val) < 2 or len(X_neg_val) < 2:
        return np.nan
    A0, _ = alignment_A0(X_pos, X_neg, X_u, X_pos_val, X_neg_val, eps_cov=eps_cov)
    return A0


# ---------------------------------------------------------------------------
# Shared boundary-shift problem generator
# ---------------------------------------------------------------------------

def make_boundary_shift_problem(rng, N1, N0, Nu, N_val, d, b_labeled, delta, pi=0.4):
    """
    Canonical two-axis design used by E1, E5, E6, E7, and the regime grid.

    Labeled data drawn from biased class means (shift b_labeled):
        mu1_lab = e1 + b_labeled*e1,  mu0_lab = -e1 + b_labeled*e1
    Unlabeled data drawn from shifted means (shift delta):
        mu1_u = e1 + delta*e1,         mu0_u = -e1 + delta*e1
    Validation / test drawn from true unbiased distribution.

    Returns (X_pos, X_neg, X_u, X_val_pos, X_val_neg)
    """
    mu1_true = np.zeros(d); mu1_true[0] = 1.0
    mu0_true = np.zeros(d); mu0_true[0] = -1.0
    S = np.eye(d)

    mu1_lab = mu1_true.copy(); mu1_lab[0] += b_labeled
    mu0_lab = mu0_true.copy(); mu0_lab[0] += b_labeled
    X_pos = rng.multivariate_normal(mu1_lab, S, N1)
    X_neg = rng.multivariate_normal(mu0_lab, S, N0)

    X_val_pos = rng.multivariate_normal(mu1_true, S, N_val)
    X_val_neg = rng.multivariate_normal(mu0_true, S, N_val)

    mu1_u = mu1_true.copy(); mu1_u[0] += delta
    mu0_u = mu0_true.copy(); mu0_u[0] += delta
    z_u = rng.binomial(1, pi, Nu)
    X_u = sample_gaussian_mixture(rng, mu1_u, mu0_u, S, S, z_u)

    return X_pos, X_neg, X_u, X_val_pos, X_val_neg


# ---------------------------------------------------------------------------
# Regime-grid experiment: decision accuracy as a response surface
# ---------------------------------------------------------------------------

def experiment_regime_grid(
    B=50,
    N_lab_grid=(10, 20, 50),
    Nu_grid=(100, 300, 1000),
    N_val_grid=(50, 200),
    lam_fixed=0.5,
    d=2,
    b_labeled=1.0,
    deltas=None,
    seed0=900,
):
    """
    E9: Vary the main sources of finite-sample variance and report decision
    accuracy and mean regret for D = 1[A_mu(0) > 0] vs oracle O = 1[gain > 0].

    Returns a list of dicts; each row = one (N_lab, Nu, N_val) regime.
    """
    if deltas is None:
        deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    print("Running E9: regime-grid decision accuracy...")
    rows = []
    total = len(N_lab_grid) * len(Nu_grid) * len(N_val_grid)
    done = 0
    for N_lab in N_lab_grid:
        for Nu in Nu_grid:
            for N_val in N_val_grid:
                done += 1
                acc_list, reg_list = [], []
                for idx_d, delta in enumerate(deltas):
                    for b in range(B):
                        seed = seed0 + done * 10000 + idx_d * 1000 + b
                        rng  = np.random.default_rng(seed)
                        X_pos, X_neg, X_u, X_vp, X_vn = make_boundary_shift_problem(
                            rng, N_lab, N_lab, Nu, N_val, d, b_labeled, delta)
                        try:
                            A0, _ = alignment_A0(X_pos, X_neg, X_u, X_vp, X_vn)
                            p_ssl = em_semisup(X_pos, X_neg, X_u, lam=lam_fixed)
                            p_sup = em_supervised(X_pos, X_neg)
                            # Test on validation
                            X_test = np.vstack([X_vp, X_vn])
                            y_test = np.array([1]*len(X_vp) + [0]*len(X_vn))
                            gain = (auroc(posterior(X_test, p_ssl), y_test)
                                    - auroc(posterior(X_test, p_sup), y_test))
                            D = int(A0 > 0)
                            O = int(gain > 0)
                            correct = int(D == O)
                            regret  = abs(gain) * int(D != O)
                            acc_list.append(correct)
                            reg_list.append(regret)
                        except Exception:
                            pass

                acc  = float(np.mean(acc_list)) if acc_list else np.nan
                reg  = float(np.mean(reg_list)) if reg_list else np.nan
                rows.append(dict(N_lab=N_lab, Nu=Nu, N_val=N_val,
                                  accuracy=acc, regret=reg,
                                  n_reps=len(acc_list)))
                print(f"  N_lab={N_lab:3d}  Nu={Nu:4d}  N_val={N_val:3d}  "
                      f"acc={acc:.2f}  regret={reg:.5f}  ({done}/{total})")
    return rows


def plot_regime_grid(rows):
    """
    Heat-map of decision accuracy over (N_lab, Nu) with facets for N_val.
    """
    import itertools
    N_labs = sorted(set(r["N_lab"] for r in rows))
    Nus    = sorted(set(r["Nu"]    for r in rows))
    N_vals = sorted(set(r["N_val"] for r in rows))
    n_facets = len(N_vals)

    fig, axes = plt.subplots(1, n_facets, figsize=(5 * n_facets + 1, 4.5), squeeze=False)
    fig.suptitle(r"Decision accuracy of $\mathcal{A}_\mu(0)$ sign over finite-sample regimes",
                 fontsize=12, fontweight="bold")

    for col, N_val in enumerate(N_vals):
        ax = axes[0, col]
        mat = np.full((len(N_labs), len(Nus)), np.nan)
        for r in rows:
            if r["N_val"] != N_val:
                continue
            i = N_labs.index(r["N_lab"])
            j = Nus.index(r["Nu"])
            mat[i, j] = r["accuracy"]

        im = ax.imshow(mat, vmin=0.4, vmax=0.9, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(Nus)));    ax.set_xticklabels(Nus)
        ax.set_yticks(range(len(N_labs))); ax.set_yticklabels(N_labs)
        ax.set_xlabel("$N_u$ (unlabeled)", fontsize=10)
        ax.set_ylabel("$N_{lab}$ (labeled per class)", fontsize=10)
        ax.set_title(f"$N_{{\\mathcal{{V}}}}={N_val}$ per class", fontsize=10)
        for i, j in itertools.product(range(len(N_labs)), range(len(Nus))):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if v < 0.6 or v > 0.8 else "black",
                        fontweight="bold")
        plt.colorbar(im, ax=ax, label="Decision accuracy", fraction=0.046)

    plt.tight_layout()
    out = FIGDIR / "fig_regime_grid.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def gaussian_data(N1, N0, Nu, d=5, pi=0.4, seed=None):
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(d); mu1[0] = 1.0
    mu0 = np.zeros(d); mu0[0] = -1.0
    S   = np.eye(d)
    X1  = rng.multivariate_normal(mu1, S, N1)
    X0  = rng.multivariate_normal(mu0, S, N0)
    # unlabeled: mixture
    z   = rng.binomial(1, pi, Nu)
    Xu  = np.where(z[:, None], rng.multivariate_normal(mu1, S, Nu),
                                rng.multivariate_normal(mu0, S, Nu))
    return X1, X0, Xu, mu1, mu0, S, S, pi


def student_t_data(N1, N0, Nu, d=5, pi=0.4, nu=3, seed=None):
    """Class-conditionals are multivariate t (misspecification scenario a)."""
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(d); mu1[0] = 1.0
    mu0 = np.zeros(d); mu0[0] = -1.0

    def mvt(mu, n, df):
        u = rng.chisquare(df, n) / df
        z = rng.multivariate_normal(np.zeros(d), np.eye(d), n)
        return mu + z / np.sqrt(u)[:, None]

    X1 = mvt(mu1, N1, nu)
    X0 = mvt(mu0, N0, nu)
    z  = rng.binomial(1, pi, Nu)
    Xu = np.where(z[:, None], mvt(mu1, Nu, nu), mvt(mu0, Nu, nu))
    return X1, X0, Xu


def subcluster_data(N1, N0, Nu, d=5, pi=0.4, seed=None):
    """Four-component mixture (2 sub-clusters per class, misspecification scenario c)."""
    rng = np.random.default_rng(seed)
    mu1a = np.zeros(d); mu1a[0] = 1.0; mu1a[1] =  1.0
    mu1b = np.zeros(d); mu1b[0] = 1.0; mu1b[1] = -1.0
    mu0a = np.zeros(d); mu0a[0] = -1.0; mu0a[1] =  1.0
    mu0b = np.zeros(d); mu0b[0] = -1.0; mu0b[1] = -1.0
    S = np.eye(d)

    def sample_class(mu_a, mu_b, n):
        sub = rng.binomial(1, 0.5, n)
        return np.where(sub[:, None],
                        rng.multivariate_normal(mu_a, S, n),
                        rng.multivariate_normal(mu_b, S, n))

    X1 = sample_class(mu1a, mu1b, N1)
    X0 = sample_class(mu0a, mu0b, N0)
    z  = rng.binomial(1, pi, Nu)
    X1u = sample_class(mu1a, mu1b, Nu)
    X0u = sample_class(mu0a, mu0b, Nu)
    Xu  = np.where(z[:, None], X1u, X0u)
    return X1, X0, Xu


def test_data_gaussian(Ntest=5000, d=5, pi=0.4, seed=42):
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(d); mu1[0] = 1.0
    mu0 = np.zeros(d); mu0[0] = -1.0
    S   = np.eye(d)
    z   = rng.binomial(1, pi, Ntest)
    X   = np.where(z[:, None],
                   rng.multivariate_normal(mu1, S, Ntest),
                   rng.multivariate_normal(mu0, S, Ntest))
    return X, z


# ---------------------------------------------------------------------------
# Grid search for best λ on a validation set
# ---------------------------------------------------------------------------

def grid_search_lambda(X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
                       lam_grid=None):
    if lam_grid is None:
        lam_grid = np.logspace(-2, 2, 20)
    y_val = np.concatenate([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))])
    X_val = np.vstack([X_pos_val, X_neg_val])
    best_auroc, best_lam, best_params = -np.inf, lam_grid[0], None
    for lam in lam_grid:
        try:
            p = em_semisup(X_pos_tr, X_neg_tr, X_u, lam=lam)
            sc = posterior(X_val, p)
            a  = auroc(sc, y_val)
            if a > best_auroc:
                best_auroc, best_lam, best_params = a, lam, p
        except Exception:
            continue
    return best_lam, best_params


def gradient_lambda(X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
                    lam_init=1.0, lr=0.5, n_steps=10):
    """
    Gradient ascent on validation log-likelihood w.r.t. λ.
    Uses finite difference ∂ℓ_V/∂λ ≈ (ℓ_V(λ+h) - ℓ_V(λ-h)) / 2h
    evaluated at the converged EM solution for each λ.
    """
    def val_ll(lam):
        try:
            p   = em_semisup(X_pos_tr, X_neg_tr, X_u, lam=max(lam, 1e-4))
            lp1 = _logpdf_mvn(X_pos_val, p.mu1, p.Sigma1)
            lp0 = _logpdf_mvn(X_neg_val, p.mu0, p.Sigma0)
            return np.sum(lp1) + np.sum(lp0)
        except Exception:
            return -np.inf

    lam = lam_init
    h   = 0.1
    for _ in range(n_steps):
        grad = (val_ll(lam + h) - val_ll(lam - h)) / (2 * h)
        lam  = np.clip(lam + lr * grad / (abs(grad) + 1e-6), 1e-3, 100.0)
    return lam


# =============================================================================
# EXPERIMENT 1: Diagnostic validity — A(0) predicts performance gain
# =============================================================================

def experiment_diagnostic(B=40, N1=15, N0=15, Nu=300, N_val=200, d=2,
                           b_labeled=1.0, deltas=None, lam_fixed=0.5, seed0=0):
    """
    Two-axis design that guarantees identifiable A(0) and sign transitions.

    Labeled data BIASED right by b_labeled along feature 0:
      mu1_labeled = 1 + b_labeled,  mu0_labeled = -1 + b_labeled
    → theta*(0) is systematically biased; grad_val(theta*(0)) ≠ 0.

    Unlabeled data shifted right by delta along feature 0:
      mu1_unlabeled = 1 + delta,  mu0_unlabeled = -1 + delta

    Validation/test drawn from the TRUE distribution (mu1=1, mu0=-1).

    Sign pattern at fixed λ:
      delta < b_labeled → unlabeled helps correct labeled bias → A(0) > 0, gain > 0
      delta ≈ b_labeled → unlabeled same bias as labeled → A(0) ≈ 0, gain ≈ 0
      delta > b_labeled → unlabeled MORE biased → worsens estimates → A(0) < 0, gain < 0

    Each (delta, replication) pair gives one scatter point (A(0), gain).
    """
    if deltas is None:
        deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    print(f"Running E1: diagnostic validity  "
          f"(b_labeled={b_labeled}, λ={lam_fixed} fixed, d={d})...")
    all_A0, all_gain, all_g0n, all_delta = [], [], [], []
    pi = 0.4

    for off_idx, delta in enumerate(deltas):
        n_ok = 0
        for b in range(B):
            rng_seed = seed0 + off_idx * 1000 + b
            rng = np.random.default_rng(rng_seed)

            mu1_true = np.zeros(d); mu1_true[0] = 1.0
            mu0_true = np.zeros(d); mu0_true[0] = -1.0
            S = np.eye(d)

            # Labeled: biased right by b_labeled
            mu1_lab = mu1_true.copy(); mu1_lab[0] += b_labeled
            mu0_lab = mu0_true.copy(); mu0_lab[0] += b_labeled
            X1_tr  = rng.multivariate_normal(mu1_lab, S, N1)
            X0_tr  = rng.multivariate_normal(mu0_lab, S, N0)

            # Validation: from TRUE distribution
            X1_val = rng.multivariate_normal(mu1_true, S, N_val)
            X0_val = rng.multivariate_normal(mu0_true, S, N_val)

            # Unlabeled: shifted right by delta
            mu1_u = mu1_true.copy(); mu1_u[0] += delta
            mu0_u = mu0_true.copy(); mu0_u[0] += delta
            z_u   = rng.binomial(1, pi, Nu)
            Xu    = np.where(z_u[:, None],
                             rng.multivariate_normal(mu1_u, S, Nu),
                             rng.multivariate_normal(mu0_u, S, Nu))

            try:
                A0, g0n = alignment_A0(X1_tr, X0_tr, Xu, X1_val, X0_val, eps_cov=1e-4)
                p_sup = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                p_r3  = em_semisup(X1_tr, X0_tr, Xu, lam=lam_fixed)
                Xtest_d, ytest_d = test_data_gaussian(3000, d=d, seed=rng_seed + 99999)
                gain = (auroc(posterior(Xtest_d, p_r3),  ytest_d)
                      - auroc(posterior(Xtest_d, p_sup), ytest_d))
                all_A0.append(A0); all_gain.append(gain)
                all_g0n.append(g0n / Nu); all_delta.append(delta)
                n_ok += 1
            except Exception:
                pass

        if n_ok > 0:
            mask = np.array(all_delta) == delta
            print(f"  delta={delta:.1f}  n={n_ok}  "
                  f"|g0|/Nu={np.mean(np.array(all_g0n)[mask]):.4f}  "
                  f"A(0)={np.mean(np.array(all_A0)[mask]):.4f}  "
                  f"gain={np.mean(np.array(all_gain)[mask]):+.5f}")

    return (np.array(all_A0), np.array(all_gain),
            np.array(all_g0n), np.array(all_delta))


# =============================================================================
# EXPERIMENT 2: λ-learning (gradient ascent) vs. grid search
# =============================================================================

def experiment_lambda_learning(B=100, N1=40, N0=40, Nu=600, N_val=40, d=5, seed0=100):
    """
    Compare three λ-selection strategies:
      (a) ESS default: λ = N_obs / Nu
      (b) grid search on val AUROC
      (c) gradient ascent (finite-diff of val log-lik)
    Report: chosen λ, val AUROC, test AUROC.
    """
    print("Running E2: lambda learning vs grid search...")
    results = {"ess": [], "grid": [], "grad": []}
    lam_ess_list, lam_grid_list, lam_grad_list = [], [], []

    for b in range(B):
        seed = seed0 + b
        rng  = np.random.default_rng(seed)
        mu1  = np.zeros(d); mu1[0] = 1.0
        mu0  = np.zeros(d); mu0[0] = -1.0
        S    = np.eye(d)
        pi   = 0.4

        X1_tr  = rng.multivariate_normal(mu1, S, N1)
        X0_tr  = rng.multivariate_normal(mu0, S, N0)
        X1_val = rng.multivariate_normal(mu1, S, N_val)
        X0_val = rng.multivariate_normal(mu0, S, N_val)
        z_u    = rng.binomial(1, pi, Nu)
        Xu     = np.where(z_u[:, None],
                          rng.multivariate_normal(mu1, S, Nu),
                          rng.multivariate_normal(mu0, S, Nu))
        Xtest, ytest = test_data_gaussian(5000, d=d, seed=seed+5000)

        # ESS default
        lam_ess = (N1 + N0) / Nu
        lam_ess_list.append(lam_ess)
        try:
            p_ess = em_semisup(X1_tr, X0_tr, Xu, lam=lam_ess)
            results["ess"].append(auroc(posterior(Xtest, p_ess), ytest))
        except Exception:
            results["ess"].append(np.nan)

        # grid search
        lam_g, p_g = grid_search_lambda(X1_tr, X0_tr, Xu, X1_val, X0_val)
        lam_grid_list.append(lam_g)
        try:
            results["grid"].append(auroc(posterior(Xtest, p_g), ytest))
        except Exception:
            results["grid"].append(np.nan)

        # gradient ascent
        lam_gr = gradient_lambda(X1_tr, X0_tr, Xu, X1_val, X0_val,
                                 lam_init=lam_ess, lr=0.3, n_steps=8)
        lam_grad_list.append(lam_gr)
        try:
            p_gr = em_semisup(X1_tr, X0_tr, Xu, lam=lam_gr)
            results["grad"].append(auroc(posterior(Xtest, p_gr), ytest))
        except Exception:
            results["grad"].append(np.nan)

        if b % 20 == 0:
            print(f"  b={b}  lam_ess={lam_ess:.2f}  lam_grid={lam_g:.2f}  lam_grad={lam_gr:.2f}")

    return results, lam_ess_list, lam_grid_list, lam_grad_list


# =============================================================================
# EXPERIMENT 3: Misspecification — |g_0| predicts degradation
# =============================================================================

def experiment_misspec(B=200, N1=50, N0=50, Nu=1000, N_val=50, d=5, seed0=200):
    """
    Three misspecification scenarios: Gaussian (correct), t(3), sub-cluster.
    For each: compute |g_0|, AUROC of R2 and R3, degradation = AUROC_R2 - AUROC_R3.
    Report mean table + scatter plot.
    """
    print("Running E3: misspecification story...")
    scenarios = {
        "Gaussian\n(correct)": "gaussian",
        "Student-$t$ ($\\nu=3$)": "student",
        "Sub-cluster": "subcluster",
    }
    table = {}
    scatter_g0, scatter_deg, records = [], [], []

    for scen_name, scen_key in scenarios.items():
        g0norms, auroc_r2s, auroc_r3s, auroc_r4s = [], [], [], []

        for b in range(B):
            seed = seed0 + b
            rng  = np.random.default_rng(seed)
            pi   = 0.4

            if scen_key == "gaussian":
                X1_tr, X0_tr, Xu, *_ = gaussian_data(N1, N0, Nu, d=d, seed=seed)
                X1_val, X0_val, Xu_v, *_ = gaussian_data(N_val, N_val, 10, d=d, seed=seed+10000)
            elif scen_key == "student":
                X1_tr, X0_tr, Xu = student_t_data(N1, N0, Nu, d=d, pi=pi, seed=seed)
                X1_val, X0_val, _ = student_t_data(N_val, N_val, 10, d=d, pi=pi, seed=seed+10000)
            else:
                X1_tr, X0_tr, Xu = subcluster_data(N1, N0, Nu, d=d, pi=pi, seed=seed)
                X1_val, X0_val, _ = subcluster_data(N_val, N_val, 10, d=d, pi=pi, seed=seed+10000)

            # test always Gaussian (evaluating classification against true boundary)
            Xtest, ytest = test_data_gaussian(3000, d=d, seed=seed+20000)

            try:
                _, g0n = score_residual_g0(X1_tr, X0_tr, Xu, eps_cov=1e-4)
                g0norms.append(g0n)

                p_sup = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                lam_g, _ = grid_search_lambda(X1_tr, X0_tr, Xu, X1_val, X0_val)
                p_r3  = em_semisup(X1_tr, X0_tr, Xu, lam=lam_g)
                p_r4  = em_conf_weighted(X1_tr, X0_tr, Xu, lam=lam_g, alpha=1.0)

                auroc_r2s.append(auroc(posterior(Xtest, p_sup), ytest))
                auroc_r3s.append(auroc(posterior(Xtest, p_r3),  ytest))
                auroc_r4s.append(auroc(posterior(Xtest, p_r4),  ytest))
            except Exception:
                continue

        table[scen_name] = {
            "|g0|": np.mean(g0norms),
            "R2 AUROC": np.mean(auroc_r2s),
            "R3 AUROC": np.mean(auroc_r3s),
            "R4 AUROC": np.mean(auroc_r4s),
            "Deg R3": np.mean(np.array(auroc_r2s) - np.array(auroc_r3s)),
        }
        gains = np.array(auroc_r3s) - np.array(auroc_r2s)
        scatter_g0.extend(g0norms)
        scatter_deg.extend(-gains)   # degradation = R2 - R3
        for g0n, gain, r2, r3, r4 in zip(
            g0norms, gains,
            auroc_r2s, auroc_r3s, auroc_r4s,
        ):
            records.append({
                "scenario": scen_name,
                "g0_norm": g0n,
                "gain_r3": gain,
                "gain_r4": r4 - r2,
                "auroc_r2": r2, "auroc_r3": r3, "auroc_r4": r4,
            })
        print(f"  {scen_key}: |g0|={table[scen_name]['|g0|']:.3f}  "
              f"R2={table[scen_name]['R2 AUROC']:.4f}  R3={table[scen_name]['R3 AUROC']:.4f}")

    return table, np.array(scatter_g0), np.array(scatter_deg), records


# =============================================================================
# PLOTTING
# =============================================================================

COLORS = {"r1": "#9b59b6", "r2": "#2ecc71", "r3": "#3498db", "r4": "#e74c3c"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def plot_diagnostic(all_A0, all_gain, all_g0n, all_delta, b_labeled=1.0):
    unique_deltas = np.unique(all_delta)
    cmap = plt.cm.plasma
    norm = plt.Normalize(unique_deltas.min(), unique_deltas.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: A(0) vs gain — THE theorem validation figure
    ax = axes[0]
    sc = ax.scatter(all_A0, all_gain,
                    c=all_delta, cmap=cmap, norm=norm,
                    s=35, alpha=0.65, edgecolors="none")
    # per-delta means with connecting line
    means_A0  = [np.mean(np.array(all_A0)[np.array(all_delta) == d]) for d in unique_deltas]
    means_gain = [np.mean(np.array(all_gain)[np.array(all_delta) == d]) for d in unique_deltas]
    ax.plot(means_A0, means_gain, "k-o", linewidth=1.2, markersize=6,
            zorder=4, label="per-δ mean")
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, linestyle=":", alpha=0.5)
    cb = plt.colorbar(sc, ax=ax); cb.set_label(r"Unlabeled shift $\delta$", fontsize=9)
    ax.set_xlabel(r"Alignment coefficient $\mathcal{A}(0)$", fontsize=11)
    ax.set_ylabel(r"AUROC gain (R3 $-$ R2, $\lambda$ fixed)", fontsize=11)
    ax.set_title(r"$\mathcal{A}(0)$ predicts gain sign and magnitude", fontsize=10)
    ax.legend(fontsize=9)

    # Panel B: |g0|/Nu vs gain
    ax = axes[1]
    ax.scatter(all_g0n, all_gain,
               c=all_delta, cmap=cmap, norm=norm,
               s=35, alpha=0.65, edgecolors="none")
    means_g0 = [np.mean(np.array(all_g0n)[np.array(all_delta) == d]) for d in unique_deltas]
    ax.plot(means_g0, means_gain, "k-o", linewidth=1.2, markersize=6, zorder=4)
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axvline(b_labeled, color="#e74c3c", linewidth=1.0, linestyle="--",
               alpha=0.7, label=fr"$\delta = b = {b_labeled}$")
    ax.set_xlabel(r"Score residual $\|\hat{g}_0\|/N_u$", fontsize=11)
    ax.set_ylabel(r"AUROC gain (R3 $-$ R2, $\lambda$ fixed)", fontsize=11)
    ax.set_title(r"$\|\hat{g}_0\|/N_u$ as pre-fitting risk indicator", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "fig_diagnostic.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_lambda_learning(results, lam_ess_list, lam_grid_list, lam_grad_list):
    """
    Panel A: Test AUROC distribution by λ-selection method (boxplot).
    Panel B: Selected λ vs. test AUROC achieved, colored by method.
             Directly answers: which λ values does each method select,
             and do those selections correlate with good AUROC?
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    method_colors = {"ESS default": "#f39c12", "Grid search": "#2ecc71",
                     "Gradient ascent": "#3498db"}

    # Panel A: test AUROC distribution — violin + jitter shows full distribution
    ax = axes[0]
    data = [np.array(results["ess"]),
            np.array(results["grid"]),
            np.array(results["grad"])]
    data = [d[~np.isnan(d)] for d in data]
    rng_jit = np.random.default_rng(42)
    for i, (d, col) in enumerate(zip(data, method_colors.values()), 1):
        parts = ax.violinplot([d], positions=[i], widths=0.55,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.55)
        parts["cmedians"].set_color("k"); parts["cmedians"].set_linewidth(2)
        jitter = rng_jit.normal(0, 0.025, len(d))
        ax.scatter(i + jitter, d, s=12, alpha=0.30, color="k", linewidths=0, zorder=3)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list(method_colors.keys()), fontsize=9)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_title("(a) Test AUROC by $\\lambda$-selection method", fontsize=10)

    # Panel B: selected λ vs. test AUROC, colored by method
    ax = axes[1]
    method_data = [
        ("ESS default",     lam_ess_list,  results["ess"]),
        ("Grid search",     lam_grid_list, results["grid"]),
        ("Gradient ascent", lam_grad_list, results["grad"]),
    ]
    for label, lam_list, auroc_list in method_data:
        lam_arr   = np.array(lam_list)
        auroc_arr = np.array(auroc_list)
        mask = ~np.isnan(auroc_arr)
        ax.scatter(lam_arr[mask], auroc_arr[mask],
                   alpha=0.55, s=28, color=method_colors[label],
                   edgecolors="none", label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"Selected $\lambda$ (log scale)", fontsize=11)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_title(r"(b) Selected $\lambda$ vs. test AUROC", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.18, linestyle=":")

    plt.tight_layout()
    out = FIGDIR / "fig_lambda_learn.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_misspec(scatter_g0, scatter_deg, records=None):
    """
    Two-panel figure:
    (a) Scatter |g₀| vs AUROC gain (R3−R2), colored by scenario.
        Story: each scenario forms a distinct cluster at a different |g₀| level;
        high |g₀| → degradation, low |g₀| → improvement.
    (b) Violin+jitter of gain by scenario for R3 and R4.
        Story: R4 (confidence-weighted) partially recovers from R3's damage
        under the high-|g₀| (Student-t) scenario.
    """
    scen_colors = {
        "Gaussian\n(correct)":        "#2ecc71",
        "Student-$t$ ($\\nu=3$)":     "#e74c3c",
        "Sub-cluster":                 "#3498db",
    }
    scen_labels = {
        "Gaussian\n(correct)":        "Gaussian (correct)",
        "Student-$t$ ($\\nu=3$)":     "Student-$t$ ($\\nu=3$)",
        "Sub-cluster":                 "Sub-cluster",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel (a): |g₀| vs gain, colored by scenario ----
    ax = axes[0]
    if records:
        for scen, col in scen_colors.items():
            recs = [r for r in records if r["scenario"] == scen]
            g0   = [r["g0_norm"]  for r in recs]
            gain = [r["gain_r3"]  for r in recs]
            ax.scatter(g0, gain, s=18, alpha=0.45, color=col,
                       linewidths=0, label=scen_labels[scen])
    else:
        ax.scatter(scatter_g0, -scatter_deg, s=18, alpha=0.4,
                   color="#888", linewidths=0)
    ax.axhline(0, color="k", linewidth=1.0, linestyle="--", alpha=0.6,
               label="No change")
    ax.set_xlabel(r"Score residual $\|\hat{g}_0\|$", fontsize=11)
    ax.set_ylabel(r"AUROC gain (R3 $-$ R2)", fontsize=11)
    ax.set_title("(a) $\\|\\hat{g}_0\\|$ predicts whether\nunlabeled data help or hurt",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # ---- Panel (b): violin+jitter of gain by scenario, R3 vs R4 ----
    ax = axes[1]
    if records:
        scen_order = list(scen_colors.keys())
        rng_jit = np.random.default_rng(42)
        for i, scen in enumerate(scen_order):
            recs = [r for r in records if r["scenario"] == scen]
            col  = scen_colors[scen]
            for j, (key, label, lw) in enumerate([("gain_r3", "R3", 0.35),
                                                   ("gain_r4", "R4", 0.35)]):
                pos = i + (j - 0.5) * 0.38
                vals = np.array([r[key] for r in recs])
                if len(vals) > 2:
                    parts = ax.violinplot([vals], positions=[pos], widths=0.32,
                                          showmeans=False, showmedians=True,
                                          showextrema=False)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(col)
                        pc.set_alpha(0.45 if j == 0 else 0.25)
                        if j == 1:
                            pc.set_hatch("///")
                    parts["cmedians"].set_color("k")
                    parts["cmedians"].set_linewidth(1.5)
                jitter = rng_jit.normal(0, 0.04, len(vals))
                ax.scatter(pos + jitter, vals, s=8, alpha=0.3,
                           color="k", linewidths=0, zorder=3)

        ax.set_xticks(range(len(scen_order)))
        ax.set_xticklabels([scen_labels[s] for s in scen_order], fontsize=9)
        ax.axhline(0, color="k", linewidth=0.9, linestyle="--", alpha=0.6)
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color="#888", alpha=0.6, label="R3: global λ"),
            mpatches.Patch(color="#888", alpha=0.3, hatch="///",
                           label="R4: conf-weighted λ(x)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")
        ax.set_ylabel("AUROC gain over supervised (R2)", fontsize=10)
        ax.set_title("(b) Gain distribution by scenario\n(R4 partially recovers "
                     "under Student-$t$)", fontsize=10)

    plt.tight_layout()
    out = FIGDIR / "fig_misspec.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# LaTeX table printers
# =============================================================================

def print_misspec_table(table):
    print("\n--- LaTeX: Misspecification Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Mean AUROC across misspecification scenarios ($B=200$ replications,")
    print(r"$(N_1,N_0)=(50,50)$, $N_u=1{,}000$). R2 = supervised MLE; R3 = global-$\lambda$")
    print(r"semi-supervised; R4 = confidence-weighted. Score residual $\|\hat{g}_0\|$ is")
    print(r"computed at the supervised MLE before any EM fitting.}")
    print(r"\label{tab:misspec}")
    print(r"\begin{tabular}{lccccr}")
    print(r"\toprule")
    print(r"Scenario & $\|\hat{g}_0\|$ & R2 AUROC & R3 AUROC & R4 AUROC & Degradation \\")
    print(r"\midrule")
    for name, vals in table.items():
        deg_sign = "+" if vals["Deg R3"] >= 0 else ""
        print(f"{name} & {vals['|g0|']:.3f} & {vals['R2 AUROC']:.4f} & "
              f"{vals['R3 AUROC']:.4f} & {vals['R4 AUROC']:.4f} & "
              f"{deg_sign}{vals['Deg R3']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_lambda_table(results, lam_ess_list, lam_grid_list, lam_grad_list):
    print("\n--- LaTeX: Lambda Learning Table ---")
    ess  = np.nanmean(results["ess"])
    grid = np.nanmean(results["grid"])
    grad = np.nanmean(results["grad"])
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Mean test AUROC and mean selected $\lambda$ by selection strategy")
    print(r"($B=100$ replications, $(N_1,N_0)=(40,40)$, $N_u=600$, $d=5$).}")
    print(r"\label{tab:lambda_learn}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Strategy & Mean $\hat\lambda$ & Mean test AUROC \\")
    print(r"\midrule")
    print(f"ESS default & {np.mean(lam_ess_list):.3f} & {ess:.4f} \\\\")
    print(f"Grid search & {np.mean(lam_grid_list):.3f} & {grid:.4f} \\\\")
    print(f"Gradient ascent & {np.mean(lam_grad_list):.3f} & {grad:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# =============================================================================
# EXPERIMENT 4: Bias-variance tradeoff — variance decreases with λ until bias dominates
# =============================================================================

def experiment_bias_variance(B=300, N1=20, N0=20, Nu=500, d=5,
                              lam_grid=None, seed0=400):
    """
    Show the bias-variance tradeoff across λ under two conditions:
      (a) Correct specification: variance ↓ monotonically, AUROC ↑
      (b) Misspecification (student-t): variance ↓ but bias ↑, AUROC peaks then falls

    Per replication we record mu1_hat[0] (the key discriminative dimension)
    so we can estimate Var(mu1_hat[0]) and Bias²(mu1_hat[0]) across reps.
    """
    if lam_grid is None:
        lam_grid = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    print(f"Running E4: bias-variance tradeoff (B={B}, N1={N0}, Nu={Nu})...")
    pi = 0.4
    mu1_true = np.zeros(d); mu1_true[0] = 1.0
    mu0_true = np.zeros(d); mu0_true[0] = -1.0

    results = {scen: {lam: {"mu1_hat": [], "auroc": []} for lam in lam_grid}
               for scen in ["gaussian", "student"]}

    for b in range(B):
        seed = seed0 + b
        rng  = np.random.default_rng(seed)

        # Labeled data always Gaussian
        X1_tr = rng.multivariate_normal(mu1_true, np.eye(d), N1)
        X0_tr = rng.multivariate_normal(mu0_true, np.eye(d), N0)

        # Test: from true Gaussian distribution
        Xtest, ytest = test_data_gaussian(3000, d=d, seed=seed + 50000)

        for scen in ["gaussian", "student"]:
            if scen == "gaussian":
                z_u = rng.binomial(1, pi, Nu)
                Xu  = np.where(z_u[:, None],
                               rng.multivariate_normal(mu1_true, np.eye(d), Nu),
                               rng.multivariate_normal(mu0_true, np.eye(d), Nu))
            else:
                # student-t unlabeled
                def mvt(mu, n, df=3):
                    u = rng.chisquare(df, n) / df
                    z = rng.multivariate_normal(np.zeros(d), np.eye(d), n)
                    return mu + z / np.sqrt(u)[:, None]
                z_u = rng.binomial(1, pi, Nu)
                Xu  = np.where(z_u[:, None], mvt(mu1_true, Nu), mvt(mu0_true, Nu))

            for lam in lam_grid:
                try:
                    if lam == 0.0:
                        p = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                    else:
                        p = em_semisup(X1_tr, X0_tr, Xu, lam=lam, eps_cov=1e-4)
                    results[scen][lam]["mu1_hat"].append(p.mu1[0])
                    results[scen][lam]["auroc"].append(auroc(posterior(Xtest, p), ytest))
                except Exception:
                    pass
        if b % 50 == 0:
            print(f"  b={b}")

    # Summarize: variance, bias², MSE, mean AUROC
    summary = {}
    for scen in ["gaussian", "student"]:
        summary[scen] = {}
        for lam in lam_grid:
            vals = np.array(results[scen][lam]["mu1_hat"])
            arocs = np.array(results[scen][lam]["auroc"])
            var  = np.var(vals)
            bias2 = (np.mean(vals) - mu1_true[0]) ** 2
            summary[scen][lam] = {
                "var": var, "bias2": bias2, "mse": var + bias2,
                "auroc_mean": np.mean(arocs), "auroc_std": np.std(arocs),
            }
    return summary, lam_grid


def plot_bias_variance(summary, lam_grid):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    scen_styles = {
        "gaussian": {"color": "#2ecc71", "label": "Correct specification (Gaussian)",  "ls": "-"},
        "student":  {"color": "#e74c3c", "label": r"Misspecification (Student-$t$, $\nu=3$)", "ls": "--"},
    }

    # Panel A: Variance + Bias² of mu1_hat[0]
    ax = axes[0]
    for scen, st in scen_styles.items():
        vars_  = [summary[scen][l]["var"]   for l in lam_grid]
        bias2s = [summary[scen][l]["bias2"] for l in lam_grid]
        ax.plot(lam_grid, vars_,  color=st["color"], ls=st["ls"],
                linewidth=2.0, label=f"{st['label']} — Var")
        ax.plot(lam_grid, bias2s, color=st["color"], ls=":",
                linewidth=1.5, label=f"{st['label']} — Bias²")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel(r"Unlabeled weight $\lambda$", fontsize=11)
    ax.set_ylabel(r"$\hat\mu_{1,0}$ estimation error", fontsize=11)
    ax.set_title(r"Variance $\downarrow$, Bias$^2$ $\uparrow$ with $\lambda$", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # Panel B: Test AUROC vs lambda
    ax = axes[1]
    for scen, st in scen_styles.items():
        means = [summary[scen][l]["auroc_mean"] for l in lam_grid]
        stds  = [summary[scen][l]["auroc_std"]  for l in lam_grid]
        means, stds = np.array(means), np.array(stds)
        ax.plot(lam_grid, means, color=st["color"], ls=st["ls"],
                linewidth=2.0, label=st["label"])
        ax.fill_between(lam_grid, means - stds, means + stds,
                        color=st["color"], alpha=0.15)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel(r"Unlabeled weight $\lambda$", fontsize=11)
    ax.set_ylabel("Test AUROC (mean ± 1 SD)", fontsize=11)
    ax.set_title(r"AUROC peaks then falls under misspecification", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    out = FIGDIR / "fig_bias_variance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# EXPERIMENT 5: Local-λ — correlation of A(0) vs gain at small λ
# =============================================================================

def experiment_local_lambda(B=60, N1=15, N0=15, Nu=300, N_val=200, d=2,
                             b_labeled=1.0, lam_values=None, seed0=500):
    """
    Same two-axis design as E1, but evaluate gain at multiple λ values
    to show that A(0) is most predictive near λ=0 (local theory) and
    loses precision as λ grows.

    Returns:
        corrs:  dict {lambda: Pearson r between A(0) and gain}
        all_data: dict {lambda: (A0_array, gain_array)}
    """
    if lam_values is None:
        lam_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    pi = 0.4
    print("Running E5: local-λ correlation study...")

    # Collect (A0, gain) at each λ, pooled across δ
    data = {lam: {"A0": [], "gain": []} for lam in lam_values}

    for off_idx, delta in enumerate(deltas):
        for b in range(B):
            rng_seed = seed0 + off_idx * 1000 + b
            rng = np.random.default_rng(rng_seed)

            mu1_true = np.zeros(d); mu1_true[0] = 1.0
            mu0_true = np.zeros(d); mu0_true[0] = -1.0
            S = np.eye(d)

            mu1_lab = mu1_true.copy(); mu1_lab[0] += b_labeled
            mu0_lab = mu0_true.copy(); mu0_lab[0] += b_labeled
            X1_tr  = rng.multivariate_normal(mu1_lab, S, N1)
            X0_tr  = rng.multivariate_normal(mu0_lab, S, N0)
            X1_val = rng.multivariate_normal(mu1_true, S, N_val)
            X0_val = rng.multivariate_normal(mu0_true, S, N_val)

            mu1_u = mu1_true.copy(); mu1_u[0] += delta
            mu0_u = mu0_true.copy(); mu0_u[0] += delta
            z_u   = rng.binomial(1, pi, Nu)
            Xu    = np.where(z_u[:, None],
                             rng.multivariate_normal(mu1_u, S, Nu),
                             rng.multivariate_normal(mu0_u, S, Nu))

            try:
                A0, _ = alignment_A0(X1_tr, X0_tr, Xu, X1_val, X0_val, eps_cov=1e-4)
                p_sup = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                Xtest_d, ytest_d = test_data_gaussian(3000, d=d, seed=rng_seed + 99999)
                auroc_sup = auroc(posterior(Xtest_d, p_sup), ytest_d)

                for lam in lam_values:
                    p_r3  = em_semisup(X1_tr, X0_tr, Xu, lam=lam)
                    gain  = auroc(posterior(Xtest_d, p_r3), ytest_d) - auroc_sup
                    data[lam]["A0"].append(A0)
                    data[lam]["gain"].append(gain)
            except Exception:
                pass

    # Compute per-λ Pearson correlation
    corrs = {}
    for lam in lam_values:
        a0  = np.array(data[lam]["A0"])
        gn  = np.array(data[lam]["gain"])
        if len(a0) > 5:
            corrs[lam] = float(np.corrcoef(a0, gn)[0, 1])
        else:
            corrs[lam] = np.nan

    for lam in lam_values:
        print(f"  λ={lam:.2f}  r={corrs[lam]:.3f}  n={len(data[lam]['A0'])}")

    return corrs, data


def plot_local_lambda(corrs, data, lam_values=None):
    """
    Two-panel figure:
      Left:  correlation r(A(0), gain) vs λ — shows decay away from λ=0
      Right: scatter A(0) vs gain at λ=0.05 (local) and λ=2.0 (far)
    """
    if lam_values is None:
        lam_values = sorted(corrs.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: correlation vs λ
    ax = axes[0]
    lam_arr = np.array(lam_values)
    r_arr   = np.array([corrs[l] for l in lam_values])
    ax.plot(lam_arr, r_arr, "ko-", linewidth=2, markersize=7)
    ax.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"Unlabeled weight $\lambda$", fontsize=12)
    ax.set_ylabel(r"Pearson $r$: $\mathcal{A}(0)$ vs gain", fontsize=12)
    ax.set_title(r"Alignment signal persists beyond the local regime", fontsize=10)
    ax.set_xscale("log")
    for lam, r in zip(lam_arr, r_arr):
        ax.annotate(f"{r:.2f}", (lam, r), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)

    # Panel B: scatter at two representative λ values
    ax = axes[1]
    colors = {"small": "#3498db", "large": "#e74c3c"}
    lam_small = lam_values[0]
    lam_large = lam_values[-1]

    for (lam_key, lam_val, label) in [
        ("small", lam_small, fr"$\lambda={lam_small}$ (local)"),
        ("large", lam_large, fr"$\lambda={lam_large}$ (far)"),
    ]:
        a0_arr = np.array(data[lam_val]["A0"])
        gn_arr = np.array(data[lam_val]["gain"])
        color  = colors[lam_key]
        ax.scatter(a0_arr, gn_arr, c=color, s=20, alpha=0.45,
                   edgecolors="none", label=label)
        # regression line
        if len(a0_arr) > 2:
            m, c = np.polyfit(a0_arr, gn_arr, 1)
            xr = np.linspace(a0_arr.min(), a0_arr.max(), 50)
            ax.plot(xr, m * xr + c, color=color, linewidth=1.5)

    ax.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="k", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"Alignment coefficient $\mathcal{A}(0)$", fontsize=12)
    ax.set_ylabel(r"AUROC gain (R3 $-$ R2)", fontsize=12)
    ax.set_title(r"Scatter at small vs.\ large $\lambda$", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "fig_local_lambda.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# EXPERIMENT 6: λ-trajectory — gain(λ) with first-order tangent
# =============================================================================

def experiment_lam_trajectory(B=80, N1=15, N0=15, Nu=300, N_val=200, d=2,
                               b_labeled=1.0, deltas_show=None,
                               lam_grid=None, seed0=600):
    """
    For three δ values, plot mean gain(λ) and overlay the first-order
    tangent line  gain ≈ λ · A(0).  Illustrates where the local theory
    holds and where it breaks.
    """
    if deltas_show is None:
        deltas_show = [0.0, 1.5, 3.0]   # helpful / transition / harmful
    if lam_grid is None:
        lam_grid = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])

    pi = 0.4
    print("Running E6: λ-trajectory...")

    results = {d_val: {lam: [] for lam in lam_grid} for d_val in deltas_show}
    A0_mean = {d_val: [] for d_val in deltas_show}

    for off_idx, delta in enumerate(deltas_show):
        for b in range(B):
            rng_seed = seed0 + off_idx * 1000 + b
            rng = np.random.default_rng(rng_seed)

            mu1_true = np.zeros(d); mu1_true[0] = 1.0
            mu0_true = np.zeros(d); mu0_true[0] = -1.0
            S = np.eye(d)

            mu1_lab = mu1_true.copy(); mu1_lab[0] += b_labeled
            mu0_lab = mu0_true.copy(); mu0_lab[0] += b_labeled
            X1_tr  = rng.multivariate_normal(mu1_lab, S, N1)
            X0_tr  = rng.multivariate_normal(mu0_lab, S, N0)
            X1_val = rng.multivariate_normal(mu1_true, S, N_val)
            X0_val = rng.multivariate_normal(mu0_true, S, N_val)

            mu1_u = mu1_true.copy(); mu1_u[0] += delta
            mu0_u = mu0_true.copy(); mu0_u[0] += delta
            z_u   = rng.binomial(1, pi, Nu)
            Xu    = np.where(z_u[:, None],
                             rng.multivariate_normal(mu1_u, S, Nu),
                             rng.multivariate_normal(mu0_u, S, Nu))

            try:
                A0, _ = alignment_A0(X1_tr, X0_tr, Xu, X1_val, X0_val, eps_cov=1e-4)
                A0_mean[delta].append(A0)
                p_sup = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                Xtest_d, ytest_d = test_data_gaussian(3000, d=d, seed=rng_seed + 99999)
                auroc_sup = auroc(posterior(Xtest_d, p_sup), ytest_d)

                for lam in lam_grid:
                    p_r3 = em_semisup(X1_tr, X0_tr, Xu, lam=lam)
                    gain = auroc(posterior(Xtest_d, p_r3), ytest_d) - auroc_sup
                    results[delta][lam].append(gain)
            except Exception:
                pass

        print(f"  δ={delta}  mean A(0)={np.mean(A0_mean[delta]):.2f}")

    return results, A0_mean, lam_grid


def plot_lam_trajectory(results, A0_mean, lam_grid, deltas_show=None):
    if deltas_show is None:
        deltas_show = sorted(results.keys())

    delta_colors = {0.0: "#2ecc71", 1.5: "#f39c12", 3.0: "#e74c3c"}
    delta_labels = {0.0: r"$\delta=0$ (helpful)", 1.5: r"$\delta=1.5$ (transition)",
                    3.0: r"$\delta=3$ (harmful)"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    lam_dense = np.linspace(0, lam_grid.max(), 200)
    for delta in deltas_show:
        color = delta_colors.get(delta, "gray")
        label = delta_labels.get(delta, f"δ={delta}")
        mean_A0 = np.mean(A0_mean[delta])

        # Empirical mean gain(λ)
        means = [np.mean(results[delta][lam]) for lam in lam_grid]
        ax.plot(lam_grid, means, "o-", color=color, linewidth=2,
                markersize=6, label=label)

        # First-order tangent: gain ≈ λ · A(0)
        ax.plot(lam_dense, lam_dense * mean_A0, "--", color=color,
                linewidth=1.2, alpha=0.65)

    ax.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"Unlabeled weight $\lambda$", fontsize=12)
    ax.set_ylabel(r"Mean AUROC gain (R3 $-$ R2)", fontsize=12)
    ax.set_title(r"Gain($\lambda$) and first-order tangent $\lambda\mathcal{A}(0)$"
                 "\n(dashed lines)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "fig_lam_trajectory.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# EXPERIMENT 7: Decision accuracy and regret — diagnostic as an operational rule
# =============================================================================

def experiment_decision_accuracy(B=100, N1=15, N0=15, Nu=300, N_val=200, d=2,
                                  b_labeled=1.0, deltas=None, lam_fixed=0.5,
                                  seed0=700):
    """
    Treats A_mu(0) > 0 as a binary decision rule: "use unlabeled data."
    For each replication computes:
      - oracle decision O = 1[gain(lambda_fixed) > 0]
      - diagnostic decision D = 1[A_mu(0) > 0]
    and records:
      - correctness (D == O)
      - regret = |gain| when D != O (AUROC lost from wrong call)
      - gain when D is correct

    Returns a dict with per-replication outcomes.
    """
    if deltas is None:
        deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    pi = 0.4
    print("Running E7: decision accuracy and regret...")

    records = []  # list of dicts, one per replication

    for off_idx, delta in enumerate(deltas):
        for b in range(B):
            rng_seed = seed0 + off_idx * 1000 + b
            rng = np.random.default_rng(rng_seed)

            mu1_true = np.zeros(d); mu1_true[0] = 1.0
            mu0_true = np.zeros(d); mu0_true[0] = -1.0
            S = np.eye(d)

            mu1_lab = mu1_true.copy(); mu1_lab[0] += b_labeled
            mu0_lab = mu0_true.copy(); mu0_lab[0] += b_labeled
            X1_tr  = rng.multivariate_normal(mu1_lab, S, N1)
            X0_tr  = rng.multivariate_normal(mu0_lab, S, N0)
            X1_val = rng.multivariate_normal(mu1_true, S, N_val)
            X0_val = rng.multivariate_normal(mu0_true, S, N_val)

            mu1_u = mu1_true.copy(); mu1_u[0] += delta
            mu0_u = mu0_true.copy(); mu0_u[0] += delta
            z_u   = rng.binomial(1, pi, Nu)
            Xu    = np.where(z_u[:, None],
                             rng.multivariate_normal(mu1_u, S, Nu),
                             rng.multivariate_normal(mu0_u, S, Nu))

            try:
                A0, _ = alignment_A0(X1_tr, X0_tr, Xu, X1_val, X0_val, eps_cov=1e-4)
                p_sup = em_supervised(X1_tr, X0_tr, eps_cov=1e-4)
                p_r3  = em_semisup(X1_tr, X0_tr, Xu, lam=lam_fixed)
                Xtest_d, ytest_d = test_data_gaussian(3000, d=d, seed=rng_seed + 99999)

                auroc_sup = auroc(posterior(Xtest_d, p_sup), ytest_d)
                auroc_r3  = auroc(posterior(Xtest_d, p_r3),  ytest_d)
                gain      = auroc_r3 - auroc_sup

                D = int(A0 > 0)         # diagnostic decision
                O = int(gain > 0)       # oracle decision (ground truth)
                correct = int(D == O)
                regret  = abs(gain) * int(D != O)  # AUROC lost from wrong call

                records.append({
                    "delta": delta,
                    "A0": A0,
                    "gain": gain,
                    "D": D,
                    "O": O,
                    "correct": correct,
                    "regret": regret,
                })
            except Exception:
                pass

    accuracy = np.mean([r["correct"] for r in records])
    mean_regret = np.mean([r["regret"] for r in records])

    # Per-delta breakdown
    print(f"\n  Overall decision accuracy: {accuracy:.3f}  mean regret: {mean_regret:.4f}")
    for delta in deltas:
        sub = [r for r in records if r["delta"] == delta]
        if sub:
            acc_d = np.mean([r["correct"] for r in sub])
            reg_d = np.mean([r["regret"] for r in sub])
            n_use = sum(r["D"] for r in sub)
            n_oracle = sum(r["O"] for r in sub)
            print(f"  δ={delta:.1f}  acc={acc_d:.2f}  regret={reg_d:.4f}"
                  f"  diag_use={n_use}/{len(sub)}  oracle_use={n_oracle}/{len(sub)}")

    return records


def plot_decision_accuracy(records):
    """
    Two-panel figure:
      Left:  Use-rate lines — diagnostic D=1 fraction vs oracle O=1 fraction per δ.
             Gap filled orange (FN: diagnostic under-recommends) or red (FP: over-recommends).
             Secondary bars show mean regret per δ.
             Story: at δ=1.5 the diagnostic drops hard while the oracle stays ~50% → large FN gap.
      Right: scatter gain vs A(0), colored by correct/incorrect decision.
    """
    import matplotlib.patches as mpatches

    deltas = sorted(set(r["delta"] for r in records))
    x = np.arange(len(deltas))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel A: use-rate comparison + regret ─────────────────────────────
    ax  = axes[0]
    ax2 = ax.twinx()

    diag_use, oracle_use, acc_rate, mean_regret = [], [], [], []
    for delta in deltas:
        sub = [r for r in records if r["delta"] == delta]
        n   = len(sub)
        diag_use.append(   sum(r["D"]       for r in sub) / n)
        oracle_use.append( sum(r["O"]       for r in sub) / n)
        acc_rate.append(   sum(r["correct"] for r in sub) / n)
        mean_regret.append(np.mean([r["regret"] for r in sub]))

    du  = np.array(diag_use)
    ou  = np.array(oracle_use)
    reg = np.array(mean_regret)

    # Regret bars on secondary axis (muted, background context)
    ax2.bar(x, reg, width=0.55, color="#e74c3c", alpha=0.18, zorder=1)
    ax2.set_ylabel("Mean regret (AUROC lost)", fontsize=9, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b", labelsize=8)
    ax2.set_ylim(0, reg.max() * 4)  # keep bars short so lines dominate

    # Fill gap between lines by error type
    ax.fill_between(x, ou, du,
                    where=(du < ou),   # FN: diag under-recommends
                    interpolate=True,
                    color="#f39c12", alpha=0.28, label="FN gap (missed benefit)",
                    zorder=2)
    ax.fill_between(x, ou, du,
                    where=(du >= ou),  # FP: diag over-recommends
                    interpolate=True,
                    color="#e74c3c", alpha=0.20, label="FP gap (unwarranted use)",
                    zorder=2)

    # The two lines
    ax.plot(x, ou, "o-", color="#3498db", lw=2.2, ms=7, zorder=4,
            label="Oracle use-rate  (O=1)")
    ax.plot(x, du, "s--", color="#e67e22", lw=2.0, ms=6, zorder=4,
            label="Diagnostic use-rate  (D=1)")

    # Accuracy overlay as thin dotted line
    ax.plot(x, acc_rate, "k:", lw=1.4, zorder=3, label="Accuracy  (D=O)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"δ={d:.1f}" for d in deltas], fontsize=9)
    ax.set_ylabel("Fraction of replications", fontsize=11)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(
        "(a) Diagnostic vs oracle use-rate per δ\n"
        "orange gap = FN (missed benefit)  ·  red bars = regret",
        fontsize=10)

    # Combine legends from ax only (ax2 has no named lines)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles, labels_leg, fontsize=8, loc="center right",
              framealpha=0.85)

    # ── Panel B: scatter gain vs A(0) ─────────────────────────────────────
    ax = axes[1]
    colors = {(True,  True):  "#2ecc71",   # TP
              (False, False): "#3498db",   # TN
              (True,  False): "#e74c3c",   # FP
              (False, True):  "#f39c12"}   # FN
    labels = {(True,  True):  "TP: correct — use",
              (False, False): "TN: correct — skip",
              (True,  False): "FP: wrong — used, hurt",
              (False, True):  "FN: wrong — skipped, missed"}
    plotted = set()

    for r in records:
        key = (bool(r["D"]), bool(r["O"]))
        color = colors[key]
        label = labels[key] if key not in plotted else None
        ax.scatter(r["A0"], r["gain"], c=color, s=18, alpha=0.45,
                   edgecolors="none", label=label)
        plotted.add(key)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

    # Shade error quadrants lightly
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.fill_betweenx([0, max(ylim)],  min(xlim), 0,
                     alpha=0.06, color="#f39c12")   # FN region (A<0, gain>0)
    ax.fill_betweenx([min(ylim), 0],  0, max(xlim),
                     alpha=0.06, color="#e74c3c")   # FP region (A>0, gain<0)

    ax.set_xlabel(r"Alignment coefficient $\mathcal{A}_\mu(0)$", fontsize=11)
    ax.set_ylabel(r"Realized AUROC gain", fontsize=11)
    ax.set_title(r"Decision boundary at $\mathcal{A}_\mu(0)=0$", fontsize=10)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper left")

    # Annotate accuracy
    acc = np.mean([r["correct"] for r in records])
    regret = np.mean([r["regret"] for r in records])
    ax.text(0.98, 0.04,
            f"Overall accuracy: {acc:.0%}\nMean regret: {regret:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    out = FIGDIR / "fig_decision.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# E8: Confidence-Weighting Ablation
#     (a) 2-D visualization of w(x) in feature space (correct spec + misspec)
#     (b) top-30% vs bottom-30% weight ablation under both regimes
# =============================================================================

def _confidence_weights(X_u: np.ndarray, params: GMMParams, alpha: float) -> np.ndarray:
    """Return per-point confidence weights w_j = (max_g γ_g)^α."""
    lp1 = (_logpdf_mvn(X_u, params.mu1, params.Sigma1)
           + np.log(np.clip(params.pi, 1e-12, 1 - 1e-12)))
    lp0 = (_logpdf_mvn(X_u, params.mu0, params.Sigma0)
           + np.log(np.clip(1 - params.pi, 1e-12, 1 - 1e-12)))
    m = np.maximum(lp1, lp0)
    gamma = np.clip(np.exp(lp1 - m - np.log(np.exp(lp1 - m) + np.exp(lp0 - m))),
                    1e-6, 1 - 1e-6)
    return np.maximum(gamma, 1 - gamma) ** alpha


def experiment_conf_weighting_ablation(
    N1=25, N0=25, Nu=600, N_test=3000, B=80,
    lam=0.5, alpha=2.0,
    top_frac=0.30,
    contam_frac=0.20,
    seed0=800,
):
    """
    E8: Confidence-weighting ablation with localized contamination design.

    Unlabeled data are mostly clean Gaussian, but contam_frac of them come
    from a sub-cluster near the class boundary (x1~0), introducing localised
    misspecification that global lambda cannot handle but w(x) should suppress.
    """
    print("Running E8: confidence-weighting ablation (localized contamination)...")
    d = 2
    mu1_true = np.array([1.0, 0.0])
    mu0_true = np.array([-1.0, 0.0])
    Sigma_true = np.eye(d)

    def _gen_clean(n, rng2):
        Xu1 = rng2.multivariate_normal(mu1_true, Sigma_true, n // 2)
        Xu0 = rng2.multivariate_normal(mu0_true, Sigma_true, n - n // 2)
        return np.vstack([Xu1, Xu0]), np.array([1]*(n//2) + [0]*(n - n//2))

    def _gen_contaminated(n, rng2, frac=contam_frac):
        """
        (1-frac) clean Gaussian + frac contamination near the decision boundary.
        Contaminated cluster: centred at (0,0), narrow in x1, wide in x2 --
        these points are near-ambiguous (gamma~0.5) so w(x) -> small.
        Global lambda is deceived; confidence weighting suppresses them.
        """
        n_contam = max(1, int(frac * n))
        n_clean  = n - n_contam
        Xu_clean, y_clean = _gen_clean(n_clean, rng2)
        contam_mu = np.array([0.0, 0.0])
        contam_S  = np.diag([0.15, 4.0])
        Xu_c = rng2.multivariate_normal(contam_mu, contam_S, n_contam)
        y_c  = np.array([i % 2 for i in range(n_contam)])
        is_c = np.array([False]*n_clean + [True]*n_contam)
        return np.vstack([Xu_clean, Xu_c]), np.concatenate([y_clean, y_c]), is_c

    def _gen_lab_test(rng2):
        X_pos  = rng2.multivariate_normal(mu1_true, Sigma_true, N1)
        X_neg  = rng2.multivariate_normal(mu0_true, Sigma_true, N0)
        Xtest  = np.vstack([
            rng2.multivariate_normal(mu1_true, Sigma_true, N_test // 2),
            rng2.multivariate_normal(mu0_true, Sigma_true, N_test - N_test // 2),
        ])
        ytest  = np.array([1]*(N_test//2) + [0]*(N_test - N_test//2))
        return X_pos, X_neg, Xtest, ytest

    # ---- Visualization rep ----
    rng_vis = np.random.default_rng(seed0 + 999)
    vis_data   = {}
    point_data = {}
    for spec in ["clean", "contaminated"]:
        X_pos, X_neg, Xtest, ytest = _gen_lab_test(rng_vis)
        if spec == "clean":
            Xu, y_u = _gen_clean(Nu, rng_vis)
            is_c = np.zeros(Nu, dtype=bool)
        else:
            Xu, y_u, is_c = _gen_contaminated(Nu, rng_vis)
        sup = em_supervised(X_pos, X_neg)
        w   = _confidence_weights(Xu, sup, alpha)
        post_u = posterior(Xu, sup)
        logit_u = np.log(np.clip(post_u, 1e-8, 1-1e-8)) - np.log(np.clip(1-post_u, 1e-8, 1-1e-8))
        vis_data[spec]   = dict(X_pos=X_pos, X_neg=X_neg, Xu=Xu, y_u=y_u,
                                 is_contam=is_c, w=w, sup=sup)
        point_data[spec] = dict(w=w, gamma=post_u, logit=logit_u,
                                 is_contam=is_c, y_u=y_u)

    # ---- Ablation ----
    records = []
    for spec in ["clean", "contaminated"]:
        for b in range(B):
            seed_b = seed0 + b * 19 + (0 if spec == "clean" else 7000)
            rng_b  = np.random.default_rng(seed_b)
            X_pos, X_neg, Xtest, ytest = _gen_lab_test(rng_b)
            if spec == "clean":
                Xu, y_u = _gen_clean(Nu, rng_b)
                is_c = np.zeros(Nu, dtype=bool)
            else:
                Xu, y_u, is_c = _gen_contaminated(Nu, rng_b)

            try:
                sup     = em_supervised(X_pos, X_neg)
                auc_sup = auroc(posterior(Xtest, sup), ytest)
            except Exception:
                continue

            w = _confidence_weights(Xu, sup, alpha)
            idx_sort = np.argsort(w)[::-1]
            n_top    = max(1, int(top_frac * Nu))
            idx_top  = idx_sort[:n_top]
            idx_bot  = idx_sort[-n_top:]

            w_c  = w[is_c].mean()  if is_c.any()  else np.nan
            w_cl = w[~is_c].mean() if (~is_c).any() else np.nan

            def _auc(X_u_sub, use_cw=False):
                try:
                    if use_cw:
                        p = em_conf_weighted(X_pos, X_neg, X_u_sub, lam=lam, alpha=alpha)
                    else:
                        p = em_semisup(X_pos, X_neg, X_u_sub, lam=lam)
                    return auroc(posterior(Xtest, p), ytest)
                except Exception:
                    return auc_sup

            auc_full = _auc(Xu)
            auc_cw   = _auc(Xu, use_cw=True)
            auc_top  = _auc(Xu[idx_top])
            auc_bot  = _auc(Xu[idx_bot])

            records.append(dict(
                spec=spec, b=b,
                auc_sup=auc_sup, auc_full=auc_full,
                auc_cw=auc_cw,  auc_top=auc_top, auc_bot=auc_bot,
                gain_full=auc_full-auc_sup, gain_cw=auc_cw-auc_sup,
                gain_top=auc_top-auc_sup,   gain_bot=auc_bot-auc_sup,
                w_contam=w_c, w_clean=w_cl,
            ))

    for spec in ["clean", "contaminated"]:
        recs  = [r for r in records if r["spec"] == spec]
        label = "Clean Gaussian" if spec == "clean" else "Contaminated (boundary noise)"
        print(f"\n  {label}:")
        for key, nm in [("gain_full","Full unlabeled (R3)"),
                        ("gain_cw",  "Conf-weighted   (R4)"),
                        ("gain_top", f"Top-{top_frac:.0%} by weight"),
                        ("gain_bot", f"Bot-{top_frac:.0%} by weight")]:
            vals = [r[key] for r in recs]
            print(f"    {nm:<35s}  mean gain {np.mean(vals):+.4f}  (sd {np.std(vals):.4f})")
        if spec == "contaminated":
            wc  = np.nanmean([r["w_contam"] for r in recs])
            wcl = np.nanmean([r["w_clean"]  for r in recs])
            print(f"    Mean w(x): clean={wcl:.3f}, contaminated={wc:.3f}  (ratio {wc/wcl:.2f}x)")

    return records, vis_data, point_data




def plot_conf_weighting_ablation(records, vis_data, point_data=None, alpha=2.0, top_frac=0.30):
    """
    Figure: 2 rows x 3 columns.
    Row 1: clean Gaussian    |  Row 2: contaminated (boundary noise)
    Col 1: feature space colored by w(x), contaminants circled
    Col 2: w(x) vs logit-posterior (boundary distance)
    Col 3: ablation bar chart (gain for full / top-k / bottom-k / conf-weighted)
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        "Confidence weighting: w(x) is large far from the boundary, small near it\n"
        "Localized contamination (20% of unlabeled, centred at boundary) is suppressed",
        fontsize=12, fontweight="bold")

    spec_labels = {"clean": "Clean Gaussian (no contamination)",
                   "contaminated": "Contaminated (20% boundary noise)"}
    colors_class = {1: "#2980b9", 0: "#e74c3c"}

    for row, spec in enumerate(["clean", "contaminated"]):
        vd   = vis_data[spec]
        pd_  = (point_data or {}).get(spec, {})
        X_pos, X_neg, Xu = vd["X_pos"], vd["X_neg"], vd["Xu"]
        y_u  = vd["y_u"]
        w    = vd["w"]
        sup  = vd["sup"]
        is_c = vd.get("is_contam", np.zeros(len(Xu), dtype=bool))
        recs = [r for r in records if r["spec"] == spec]

        # --- Col 0: feature space colored by w(x) ---
        ax = axes[row, 0]
        xlo = min(Xu[:, 0].min(), X_pos[:, 0].min(), X_neg[:, 0].min()) - 0.8
        xhi = max(Xu[:, 0].max(), X_pos[:, 0].max(), X_neg[:, 0].max()) + 0.8
        ylo = min(Xu[:, 1].min(), X_pos[:, 1].min(), X_neg[:, 1].min()) - 0.8
        yhi = max(Xu[:, 1].max(), X_pos[:, 1].max(), X_neg[:, 1].max()) + 0.8
        # Clip for readability
        ylo, yhi = max(ylo, -7), min(yhi, 7)
        xx, yy = np.meshgrid(np.linspace(xlo, xhi, 200),
                              np.linspace(ylo, yhi, 200))
        prob_grid = posterior(np.c_[xx.ravel(), yy.ravel()], sup).reshape(xx.shape)
        ax.contour(xx, yy, prob_grid, levels=[0.5], colors="k", linewidths=1.8)

        # Non-contaminated first
        mask_ok = ~is_c
        sc = ax.scatter(Xu[mask_ok, 0], Xu[mask_ok, 1], c=w[mask_ok],
                        cmap="plasma", s=14, alpha=0.55, vmin=0, vmax=1, linewidths=0)
        # Contaminated: circled in orange
        if is_c.any():
            ax.scatter(Xu[is_c, 0], Xu[is_c, 1], c=w[is_c], cmap="plasma",
                       s=20, alpha=0.7, vmin=0, vmax=1, linewidths=0.8,
                       edgecolors="#e67e22", label="Contaminated")
        ax.scatter(X_pos[:, 0], X_pos[:, 1], marker="^", s=70,
                   c=colors_class[1], edgecolors="k", linewidths=0.5, zorder=5,
                   label="Labeled +")
        ax.scatter(X_neg[:, 0], X_neg[:, 1], marker="v", s=70,
                   c=colors_class[0], edgecolors="k", linewidths=0.5, zorder=5,
                   label="Labeled −")
        plt.colorbar(sc, ax=ax, label="w(x)", fraction=0.046, pad=0.04)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_title(f"{spec_labels[spec]}\nw(x) in feature space", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="upper right")

        # --- Col 1: w(x) vs logit-posterior ---
        ax = axes[row, 1]
        post_u  = pd_.get("gamma",  posterior(Xu, sup))
        logit_u = pd_.get("logit",
                    np.log(np.clip(post_u, 1e-8, 1-1e-8))
                  - np.log(np.clip(1-post_u, 1e-8, 1-1e-8)))
        clip_lo, clip_hi = np.percentile(logit_u, 2), np.percentile(logit_u, 98)
        mask_vis = (logit_u > clip_lo) & (logit_u < clip_hi)

        if is_c.any():
            ax.scatter(logit_u[mask_vis & ~is_c], w[mask_vis & ~is_c],
                       c="#3498db", s=10, alpha=0.35, linewidths=0, label="Clean")
            ax.scatter(logit_u[mask_vis & is_c], w[mask_vis & is_c],
                       c="#e67e22", s=18, alpha=0.6, linewidths=0, label="Contaminated")
            ax.legend(fontsize=8, loc="upper center")
        else:
            ax.scatter(logit_u[mask_vis], w[mask_vis],
                       c=[colors_class[int(y)] for y in y_u[mask_vis]],
                       s=10, alpha=0.35, linewidths=0)

        ax.axvline(0, color="k", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel(r"Logit posterior $\log[\hat\gamma/(1-\hat\gamma)]$", fontsize=9)
        ax.set_ylabel(r"Confidence weight $w(x)$", fontsize=9)
        ax.set_title(f"{spec_labels[spec]}\nw(x) vs boundary distance", fontsize=10)
        ax.text(0.5, 0.96, "Ambiguous pts: low w(x)",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        # --- Col 2: ablation violin+jitter — shows full distribution, not just mean±SE ---
        ax = axes[row, 2]
        gain_keys  = ["gain_full", "gain_top", "gain_bot", "gain_cw"]
        gain_names = ["Full\nunlabeled",
                      f"Top {top_frac:.0%}\nby weight",
                      f"Bot {top_frac:.0%}\nby weight",
                      f"Conf-wtd\n(\u03b1={alpha})"]
        bcols = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        rng_jit2 = np.random.default_rng(99)
        for i, (key, name, col) in enumerate(zip(gain_keys, gain_names, bcols)):
            vals = np.array([r[key] for r in recs])
            if len(vals) > 1:
                parts = ax.violinplot([vals], positions=[i], widths=0.55,
                                      showmeans=False, showmedians=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(col); pc.set_alpha(0.55)
                parts["cmedians"].set_color("k"); parts["cmedians"].set_linewidth(1.8)
            jitter = rng_jit2.normal(0, 0.04, len(vals))
            ax.scatter(i + jitter, vals, s=10, alpha=0.30, color="k", linewidths=0, zorder=3)
            ax.text(i, np.mean(vals) + (0.0005 if np.mean(vals) >= 0 else -0.001),
                    f"{np.mean(vals):+.3f}", ha="center",
                    va="bottom" if np.mean(vals) >= 0 else "top", fontsize=8,
                    color=col, fontweight="bold")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(len(gain_names)))
        ax.set_xticklabels(gain_names, fontsize=8)
        ax.set_ylabel("AUROC gain over supervised (R2)", fontsize=9)
        ax.set_title(f"{spec_labels[spec]}\nAblation: which unlabeled points help?", fontsize=10)

    plt.tight_layout()
    out = FIGDIR / "fig_conf_ablation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")





# =============================================================================
# E10: Dimension sweep — geometry, misalignment, and the high-d regime
# =============================================================================

def experiment_dimension_sweep(
    N1=20, N0=20, Nu=500, N_val=200, N_test=3000, B=60,
    lam=0.5, alpha=2.0,
    d_grid=None,
    seed0=1000,
):
    """
    E10: How do R3 (global λ) and R4 (confidence-weighted) behave as
    dimension d grows across three unlabeled-geometry regimes?

    1. aligned_dense  — signal in all d dimensions (normalised): best case
    2. sparse_signal  — signal only in x1; the rest are noise dimensions
    3. misaligned     — unlabeled clusters are in x2 (orthogonal to class
                        boundary): the "topic ≠ sentiment" failure mode

    True labeled and test distributions always have boundary along x1:
        μ1_true = e1,  μ0_true = -e1,  Σ = I_d

    Also computes A(0) and ||g_0|| per replication to validate the
    high-dimensional alignment theory (§4.6): sign(A(0)) should predict
    sign(gain_r3) in each replication.

    Returns list of dicts with keys:
        d, spec, b, gain_r3, gain_r4, A0, g0_norm
    """
    if d_grid is None:
        d_grid = [2, 5, 10, 20, 50]

    print("Running E10: dimension sweep...")
    records = []
    spec_offsets = {"aligned_dense": 0, "sparse_signal": 50_000, "misaligned": 100_000}

    for d in d_grid:
        S = np.eye(d)
        mu_boundary = np.zeros(d); mu_boundary[0] = 1.0   # true class boundary

        # ---- unlabeled generators per regime ----
        mu_dense = np.ones(d) / np.sqrt(d)   # signal in all d dims, unit norm

        mu_misalign = np.zeros(d)
        if d >= 2:
            mu_misalign[1] = 2.0             # strong cluster in x2, neutral in x1
        else:
            mu_misalign[0] = 2.0

        regime_mus = {
            "aligned_dense": (mu_dense,   -mu_dense),   # helpful: aligned
            "sparse_signal":  (mu_boundary, -mu_boundary),  # unlabeled = labeled DGP
            "misaligned":    (mu_misalign, -mu_misalign),   # harmful: orthogonal
        }

        for spec, (mu1_u, mu0_u) in regime_mus.items():
            for b in range(B):
                seed = seed0 + b * 100 + d * 10_000 + spec_offsets[spec]
                rng  = np.random.default_rng(seed)

                # Labeled
                X_pos = rng.multivariate_normal(mu_boundary, S, N1)
                X_neg = rng.multivariate_normal(-mu_boundary, S, N0)

                # Unlabeled
                z_u = rng.binomial(1, 0.5, Nu)
                X_u = sample_gaussian_mixture(rng, mu1_u, mu0_u, S, S, z_u)

                # Validation (from true boundary distribution, for A(0))
                X_val_pos = rng.multivariate_normal(mu_boundary,  S, N_val // 2)
                X_val_neg = rng.multivariate_normal(-mu_boundary, S, N_val // 2)

                # Test (always from true boundary distribution)
                Xt1  = rng.multivariate_normal(mu_boundary,  S, N_test // 2)
                Xt0  = rng.multivariate_normal(-mu_boundary, S, N_test - N_test // 2)
                Xtest = np.vstack([Xt1, Xt0])
                ytest = np.array([1]*(N_test//2) + [0]*(N_test - N_test//2))

                # Scale ridge with d/N to keep covariance invertible in high-d
                eps = max(1e-6, d / N1)
                try:
                    sup    = em_supervised(X_pos, X_neg, eps_cov=eps)
                    auc_s  = auroc(posterior(Xtest, sup), ytest)
                    p_r3   = em_semisup(X_pos, X_neg, X_u, lam=lam, eps_cov=eps)
                    p_r4   = em_conf_weighted(X_pos, X_neg, X_u, lam=lam, alpha=alpha,
                                              eps_cov=eps)
                    A0, g0_norm = alignment_A0(
                        X_pos, X_neg, X_u, X_val_pos, X_val_neg, eps_cov=eps
                    )
                    records.append(dict(
                        d=d, spec=spec, b=b,
                        gain_r3=auroc(posterior(Xtest, p_r3), ytest) - auc_s,
                        gain_r4=auroc(posterior(Xtest, p_r4), ytest) - auc_s,
                        A0=A0,
                        g0_norm=g0_norm,
                    ))
                except Exception:
                    pass

        print(f"  d={d:3d}  done ({3*B} attempts)")

    return records


def plot_dimension_sweep(records):
    """
    Four-panel figure.
    Panels 1-3: mean AUROC gain ± SE per regime (R3 and R4 lines).
    Panel 4: fraction of replications where sign(A(0)) == sign(gain_r3),
             per regime and dimension — validates the high-dimensional
             alignment theory (§4.6).
    """
    specs = ["aligned_dense", "sparse_signal", "misaligned"]
    titles = [
        "Aligned dense signal\n(unlabeled geometry matches labels)",
        "Sparse signal + nuisance dims\n(boundary in $x_1$ only)",
        "Misaligned cluster geometry\n(unlabeled clusters $\\perp$ class boundary)",
    ]
    d_vals = sorted(set(r["d"] for r in records))
    colors = {"gain_r3": "#3498db", "gain_r4": "#9b59b6"}
    labels = {"gain_r3": "R3: global $\\lambda$",
               "gain_r4": "R4: $\\lambda(x)$ (conf-weighted)"}
    spec_colors = {
        "aligned_dense": "#27ae60",
        "sparse_signal": "#e67e22",
        "misaligned":    "#e74c3c",
    }
    spec_labels = {
        "aligned_dense": "Aligned dense",
        "sparse_signal": "Sparse signal",
        "misaligned":    "Misaligned",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        "High-dimensional geometry: when does unlabeled data help?",
        fontsize=12, fontweight="bold")

    # Panels 1-3: AUROC gain per regime
    for ax, spec, title in zip(axes[:3], specs, titles):
        for key in ["gain_r3", "gain_r4"]:
            means, sems = [], []
            for d in d_vals:
                vals = [r[key] for r in records if r["spec"] == spec and r["d"] == d]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan); sems.append(np.nan)
            means, sems = np.array(means), np.array(sems)
            ax.plot(d_vals, means, "o-", color=colors[key], lw=2,
                    markersize=5, label=labels[key])
            ax.fill_between(d_vals, means - sems, means + sems,
                             color=colors[key], alpha=0.18)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Dimension $d$", fontsize=10)
        ax.set_ylabel("AUROC gain over supervised (R2)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    # Panel 4: A(0) diagnostic accuracy — fraction where sign(A0) == sign(gain_r3)
    ax4 = axes[3]
    has_A0 = any("A0" in r for r in records)
    if has_A0:
        for spec in specs:
            accs = []
            for d in d_vals:
                recs = [r for r in records
                        if r["spec"] == spec and r["d"] == d
                        and "A0" in r and not np.isnan(r["A0"])]
                if recs:
                    correct = sum(
                        np.sign(r["A0"]) == np.sign(r["gain_r3"]) for r in recs
                    )
                    accs.append(correct / len(recs))
                else:
                    accs.append(np.nan)
            ax4.plot(d_vals, accs, "o-", color=spec_colors[spec], lw=2,
                     markersize=5, label=spec_labels[spec])
        ax4.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5, label="Chance")
        ax4.set_ylim(0, 1)
        ax4.set_xlabel("Dimension $d$", fontsize=10)
        ax4.set_ylabel("Fraction sign($\\mathcal{A}(0)$) = sign(gain)", fontsize=9)
        ax4.set_title(
            "$\\mathcal{A}(0)$ diagnostic accuracy\n(theory validation)", fontsize=10
        )
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "A(0) not available\n(rerun experiment)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10)
        ax4.set_title("$\\mathcal{A}(0)$ diagnostic accuracy", fontsize=10)

    plt.tight_layout()
    out = FIGDIR / "fig_dimension_sweep.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# EXPERIMENT: Parameter Recovery
# =============================================================================

def experiment_parameter_recovery(
    B=200, N1=50, N0=50, d=5,
    Nu_grid=(100, 500, 2000),
    seed0=3000,
):
    """
    E-PR: RMSE for μ₁, μ₀, Σ₁, Σ₀, π vs N_u ∈ {100,500,2000}
    at fixed (N1,N0)=(50,50), correct Gaussian specification.
    Methods: R2 (supervised), R3 (global λ=ESS), R4 (conf-weighted).

    True params: mu1=ones(d)/sqrt(d), mu0=-ones(d)/sqrt(d), S=I, pi=0.4
    """
    print("Running E-PR: Parameter Recovery...")

    pi_true  = 0.4
    mu1_true = np.ones(d) / np.sqrt(d)
    mu0_true = -mu1_true
    S_true   = np.eye(d)

    rows = []
    for Nu in Nu_grid:
        errs = {"R2": [], "R3": [], "R4": []}
        for b in range(B):
            seed = seed0 + b + Nu * 10
            rng  = np.random.default_rng(seed)

            # Labeled data
            X1 = rng.multivariate_normal(mu1_true, S_true, N1)
            X0 = rng.multivariate_normal(mu0_true, S_true, N0)

            # Unlabeled mixture
            n1u = rng.binomial(Nu, pi_true)
            n0u = Nu - n1u
            Xu  = np.vstack([
                rng.multivariate_normal(mu1_true, S_true, max(n1u, 1)),
                rng.multivariate_normal(mu0_true, S_true, max(n0u, 1)),
            ])

            # Validation (for grid-search λ)
            X1v = rng.multivariate_normal(mu1_true, S_true, N1)
            X0v = rng.multivariate_normal(mu0_true, S_true, N0)

            try:
                p_r2 = em_supervised(X1, X0, eps_cov=1e-4)
                lam_g, _ = grid_search_lambda(X1, X0, Xu, X1v, X0v)
                p_r3 = em_semisup(X1, X0, Xu, lam=lam_g, eps_cov=1e-4)
                p_r4 = em_conf_weighted(X1, X0, Xu, lam=lam_g, alpha=2.0, eps_cov=1e-4)
            except Exception:
                continue

            def _rmse_params(p):
                e_mu1 = np.sqrt(np.mean((p.mu1 - mu1_true) ** 2))
                e_mu0 = np.sqrt(np.mean((p.mu0 - mu0_true) ** 2))
                e_S1  = np.sqrt(np.mean((p.Sigma1 - S_true) ** 2))
                e_S0  = np.sqrt(np.mean((p.Sigma0 - S_true) ** 2))
                e_pi  = abs(p.pi - pi_true)
                return e_mu1, e_mu0, e_S1, e_S0, e_pi

            for label, p in [("R2", p_r2), ("R3", p_r3), ("R4", p_r4)]:
                errs[label].append(_rmse_params(p))

        row = {"Nu": Nu}
        for label in ("R2", "R3", "R4"):
            if errs[label]:
                arr = np.array(errs[label])   # (B, 5)
                means = arr.mean(0)
                row[label] = {
                    "mu1": means[0], "mu0": means[1],
                    "S1": means[2],  "S0": means[3], "pi": means[4],
                    "combined": np.mean(means[:4]),  # avg over mean/cov errors
                }
        rows.append(row)
        print(f"  Nu={Nu}: R2 combined={row['R2']['combined']:.4f}  "
              f"R3 combined={row['R3']['combined']:.4f}  "
              f"R4 combined={row['R4']['combined']:.4f}")

    return rows


def print_param_recovery_table(rows):
    """Print LaTeX table for parameter recovery results."""
    print("\n--- Parameter Recovery Table (LaTeX) ---")
    print(r"\begin{tabular}{lcccc}")
    print(r"  \toprule")
    print(r"  $N_u$ & Method & $\mathrm{RMSE}(\mu)$ & $\mathrm{RMSE}(\Sigma)$ & $|\hat\pi - \pi|$ \\")
    print(r"  \midrule")
    for row in rows:
        Nu = row["Nu"]
        for i, label in enumerate(("R2", "R3", "R4")):
            d = row[label]
            nu_str = f"{Nu}" if i == 0 else ""
            print(f"  {nu_str} & {label} & {(d['mu1']+d['mu0'])/2:.4f} "
                  f"& {(d['S1']+d['S0'])/2:.4f} & {d['pi']:.4f} \\\\")
        print(r"  \addlinespace")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


def plot_param_recovery(rows):
    """Two-panel figure: RMSE(μ) and RMSE(Σ) vs N_u for R2/R3/R4."""
    Nu_vals = [row["Nu"] for row in rows]
    colors  = {"R2": "#2ecc71", "R3": "#3498db", "R4": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, ylabel in zip(
        axes,
        [("mu1", "mu0"), ("S1", "S0")],
        [r"RMSE($\mu$)", r"RMSE($\Sigma$)"],
    ):
        for label in ("R2", "R3", "R4"):
            vals = [(row[label][key[0]] + row[label][key[1]]) / 2 for row in rows]
            ax.plot(Nu_vals, vals, "o-", color=colors[label], lw=2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("$N_u$ (unlabeled sample size)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(Nu_vals)
        ax.set_xticklabels([str(v) for v in Nu_vals])
        ax.legend(fontsize=9)
        ax.set_title(ylabel, fontsize=10)

    plt.tight_layout()
    out = FIGDIR / "fig_param_recovery.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# EXPERIMENT: Sensitivity to λ and α
# =============================================================================

def experiment_sensitivity(
    B=100, N1=40, N0=40, Nu=600, N_val=40, d=5,
    lam_grid=None, alpha_grid=None,
    seed0=4000,
):
    """
    E-Sens: AUROC vs λ (log scale) for R3 and AUROC vs α for R4.
    Two scenarios: correct Gaussian and sub-cluster misspecification.
    Returns dict: scenario → {lam_grid → [auroc_r3], alpha_grid → [auroc_r4], lam_star}
    """
    print("Running E-Sens: Sensitivity to λ and α...")

    if lam_grid is None:
        lam_grid = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    if alpha_grid is None:
        alpha_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

    scenarios = {"Gaussian": "gaussian", "Sub-cluster": "subcluster"}
    results = {}

    for scen_name, scen_key in scenarios.items():
        auroc_by_lam = {lam: [] for lam in lam_grid}
        auroc_r4_by_alpha = {a: [] for a in alpha_grid}
        lam_stars = []

        for b in range(B):
            seed = seed0 + b
            rng  = np.random.default_rng(seed)

            if scen_key == "gaussian":
                X1, X0, Xu, *_ = gaussian_data(N1, N0, Nu, d=d, seed=seed)
                X1v, X0v, *_ = gaussian_data(N_val, N_val, 10, d=d, seed=seed + 50000)
            else:
                X1, X0, Xu = subcluster_data(N1, N0, Nu, d=d, seed=seed)
                X1v, X0v, _ = subcluster_data(N_val, N_val, 10, d=d, seed=seed + 50000)

            Xtest, ytest = test_data_gaussian(3000, d=d, seed=seed + 20000)

            try:
                lam_g, _ = grid_search_lambda(X1, X0, Xu, X1v, X0v)
                lam_stars.append(lam_g)

                for lam in lam_grid:
                    p = em_semisup(X1, X0, Xu, lam=lam, eps_cov=1e-4)
                    auroc_by_lam[lam].append(auroc(posterior(Xtest, p), ytest))

                for a in alpha_grid:
                    p = em_conf_weighted(X1, X0, Xu, lam=lam_g, alpha=a, eps_cov=1e-4)
                    auroc_r4_by_alpha[a].append(auroc(posterior(Xtest, p), ytest))
            except Exception:
                continue

        results[scen_name] = {
            "lam_grid": lam_grid,
            "alpha_grid": alpha_grid,
            "auroc_lam": {lam: np.mean(v) for lam, v in auroc_by_lam.items() if v},
            "auroc_lam_se": {lam: np.std(v)/np.sqrt(len(v)) for lam, v in auroc_by_lam.items() if v},
            "auroc_alpha": {a: np.mean(v) for a, v in auroc_r4_by_alpha.items() if v},
            "auroc_alpha_se": {a: np.std(v)/np.sqrt(len(v)) for a, v in auroc_r4_by_alpha.items() if v},
            "lam_star_mean": np.mean(lam_stars) if lam_stars else 1.0,
            "lam_ess": N1 / Nu,  # ESS default
        }
        print(f"  {scen_name}: mean lam*={results[scen_name]['lam_star_mean']:.3f}  "
              f"ESS default={results[scen_name]['lam_ess']:.3f}")

    return results


def plot_sensitivity(results):
    """Two-panel figure: (a) AUROC vs λ; (b) AUROC vs α."""
    colors  = {"Gaussian": "#3498db", "Sub-cluster": "#e74c3c"}
    linestyles = {"Gaussian": "-", "Sub-cluster": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): AUROC vs λ
    ax = axes[0]
    for scen, res in results.items():
        lam_vals = sorted(res["auroc_lam"].keys())
        means = [res["auroc_lam"][l] for l in lam_vals]
        ses   = [res["auroc_lam_se"][l] for l in lam_vals]
        means, ses = np.array(means), np.array(ses)
        ax.plot(lam_vals, means, "o", color=colors[scen],
                ls=linestyles[scen], lw=2, label=scen)
        ax.fill_between(lam_vals, means - ses, means + ses,
                        color=colors[scen], alpha=0.15)

    # ESS default vertical line (use Gaussian result)
    lam_ess = list(results.values())[0]["lam_ess"]
    ax.axvline(lam_ess, color="gray", ls=":", lw=1.5, label=f"ESS ($\\lambda={lam_ess:.2f}$)")
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda$ (unlabeled weight)", fontsize=10)
    ax.set_ylabel("AUROC (test)", fontsize=10)
    ax.set_title("(a) R3: AUROC vs $\\lambda$", fontsize=10)
    ax.legend(fontsize=9)

    # Panel (b): AUROC vs α
    ax = axes[1]
    for scen, res in results.items():
        alpha_vals = sorted(res["auroc_alpha"].keys())
        means = [res["auroc_alpha"][a] for a in alpha_vals]
        ses   = [res["auroc_alpha_se"][a] for a in alpha_vals]
        means, ses = np.array(means), np.array(ses)
        ax.plot(alpha_vals, means, "o", color=colors[scen],
                ls=linestyles[scen], lw=2, label=scen)
        ax.fill_between(alpha_vals, means - ses, means + ses,
                        color=colors[scen], alpha=0.15)

    ax.set_xlabel("$\\alpha$ (confidence sharpness)", fontsize=10)
    ax.set_ylabel("AUROC (test)", fontsize=10)
    ax.set_title("(b) R4: AUROC vs $\\alpha$ at $\\lambda = \\lambda^*_{R3}$", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "fig_sensitivity.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(0)

    print("=" * 60)
    print("EXPERIMENT 1: Diagnostic validity")
    print("=" * 60)
    b_lab = 1.0
    deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    all_A0, all_gain, all_g0n, all_delta = experiment_diagnostic(
        B=40, N1=15, N0=15, Nu=300, N_val=200, d=2,
        b_labeled=b_lab, deltas=deltas, lam_fixed=0.5
    )
    plot_diagnostic(all_A0, all_gain, all_g0n, all_delta, b_labeled=b_lab)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Lambda learning vs. grid search")
    print("=" * 60)
    results_e2, lam_ess, lam_grid, lam_grad = experiment_lambda_learning(
        B=100, N1=40, N0=40, Nu=600, N_val=40, d=5
    )
    plot_lambda_learning(results_e2, lam_ess, lam_grid, lam_grad)
    print_lambda_table(results_e2, lam_ess, lam_grid, lam_grad)

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Misspecification story")
    print("=" * 60)
    table_e3, sg0, sdeg, records_e3 = experiment_misspec(
        B=200, N1=50, N0=50, Nu=1000, N_val=50, d=5
    )
    plot_misspec(sg0, sdeg, records_e3)
    print_misspec_table(table_e3)

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Bias-variance tradeoff")
    print("=" * 60)
    summary_e4, lam_grid_e4 = experiment_bias_variance(
        B=300, N1=20, N0=20, Nu=500, d=5
    )
    plot_bias_variance(summary_e4, lam_grid_e4)
    # Print summary table
    print("\n--- Bias-Variance Summary ---")
    for scen in ["gaussian", "student"]:
        print(f"\n  {scen}:")
        for lam in lam_grid_e4:
            s = summary_e4[scen][lam]
            print(f"    λ={lam:.2f}  Var={s['var']:.5f}  Bias²={s['bias2']:.5f}  "
                  f"AUROC={s['auroc_mean']:.4f}")

    print("\n" + "=" * 60)
    print("EXPERIMENT PR: Parameter Recovery")
    print("=" * 60)
    rows_pr = experiment_parameter_recovery(B=200, N1=50, N0=50, d=5)
    print_param_recovery_table(rows_pr)
    plot_param_recovery(rows_pr)

    print("\n" + "=" * 60)
    print("EXPERIMENT Sens: Sensitivity to λ and α")
    print("=" * 60)
    results_sens = experiment_sensitivity(B=100, N1=40, N0=40, Nu=600, N_val=40, d=5)
    plot_sensitivity(results_sens)

    print("\n" + "=" * 60)
    print("EXPERIMENT 10: High-dimensional geometry sweep")
    print("=" * 60)
    records_dim = experiment_dimension_sweep(
        N1=20, N0=20, Nu=500, N_val=200, N_test=3000, B=60,
        lam=0.5, alpha=2.0,
    )
    plot_dimension_sweep(records_dim)

    print("\nDone. Figures in:", FIGDIR)

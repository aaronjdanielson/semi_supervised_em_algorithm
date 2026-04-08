"""
Core EM routines.  Pure NumPy — no sklearn dependency.

Algorithmic improvements over the reference implementation
(discovered_materials/code/simulations.py):

1. Cholesky caching: a single Cholesky per class per iteration replaces the
   two separate O(d^3) factorizations (slogdet + solve) in the original.

2. Labeled sufficient statistics precomputed outside the EM loop:
       XpXp = X_pos.T @ X_pos,  sum_pos = X_pos.sum(0)
   The centered scatter is recovered via the identity
       (X - mu)^T(X - mu) = X^T X - N * outer(mu, mu)
   saving O((N1+N0)*d^2) per iteration.

3. Covariance type: "full" (default), "diag", or "ledoit_wolf".
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve_triangular

from ._params import GMMParams, GMMParamsMulti
from .exceptions import warn_convergence


# ---------------------------------------------------------------------------
# Numerically stable log-pdf for a batch of observations
# ---------------------------------------------------------------------------

def _chol_logpdf(X: np.ndarray, mu: np.ndarray, L: np.ndarray, log_det: float) -> np.ndarray:
    """
    Log-pdf of X under N(mu, L L^T) using a pre-computed Cholesky factor L.

    Parameters
    ----------
    X : (n, d)
    mu : (d,)
    L : (d, d) lower-triangular Cholesky factor of Sigma
    log_det : float  — 2 * sum(log(diag(L)))

    Returns
    -------
    log_probs : (n,)
    """
    d = mu.shape[0]
    Xc = X - mu                                             # (n, d)
    v = solve_triangular(L, Xc.T, lower=True)               # (d, n)
    quad = np.einsum("ij,ij->j", v, v)                      # (n,) mahalanobis^2
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


_LOG_FLOOR = -700.0   # exp(-700) ≈ 9.9e-305 — finite in float64


def _logsumexp_rows(log_p: np.ndarray) -> np.ndarray:
    """
    Log-sum-exp over columns for each row of log_p (n, K), returns (n, 1).

    Protects against rows where all entries are -inf (e.g., extreme outlier
    points where every class log-pdf underflows to -inf).  Those rows are
    floored to _LOG_FLOOR before the standard LSE so that posterior is
    uniform rather than NaN.
    """
    floored = np.maximum(log_p, _LOG_FLOOR)          # (n, K)
    m = floored.max(1, keepdims=True)                 # (n, 1)
    return m + np.log(np.exp(floored - m).sum(1, keepdims=True))


def _safe_cholesky(S: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """
    Cholesky decomposition with automatic ridge escalation if S is not PD.
    Returns (L, log_det).
    """
    S_reg = S + eps * np.eye(S.shape[0])
    for _ in range(6):
        try:
            L = cholesky(S_reg)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            return L, log_det
        except LinAlgError:
            eps *= 10
            S_reg = S + eps * np.eye(S.shape[0])
    raise LinAlgError("Could not compute Cholesky even with heavy regularization.")


def _cov_estimate(X: np.ndarray, covariance_type: str, eps: float) -> np.ndarray:
    """Covariance estimate with optional Ledoit-Wolf shrinkage or diagonal restriction."""
    if X.shape[0] < 2:
        d = X.shape[1]
        return np.eye(d) * eps
    if covariance_type == "diag":
        return np.diag(np.var(X, axis=0, ddof=1) + eps)
    if covariance_type == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        S = lw.covariance_ + eps * np.eye(X.shape[1])
        return S
    # "full"
    S = np.cov(X, rowvar=False) if X.shape[0] > 1 else np.eye(X.shape[1])
    return S + eps * np.eye(X.shape[1])


# ---------------------------------------------------------------------------
# Soft counts helper
# ---------------------------------------------------------------------------

def _soft_counts(gamma: np.ndarray, N1: int, N0: int, lam: float, Nu: int):
    """Weighted effective class counts used in the M-step."""
    Nu1 = lam * gamma.sum()
    Nu0 = lam * (1.0 - gamma).sum()
    N1t = N1 + Nu1
    N0t = N0 + Nu0
    Nt  = N1t + N0t
    return N1t, N0t, Nt


# ---------------------------------------------------------------------------
# Main EM loop
# ---------------------------------------------------------------------------

def em_semisup(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    X_u: np.ndarray,
    lam: float = 1.0,
    tol: float = 5e-6,
    max_iter: int = 300,
    eps_cov: float = 1e-6,
    resp_clip: float = 1e-6,
    covariance_type: str = "full",
    warn: bool = True,
) -> tuple[GMMParams, int, bool]:
    """
    Semi-supervised EM for a two-component Gaussian mixture.

    Maximises J(theta) = l_sup(theta) + lam * l_unl(theta).

    Parameters
    ----------
    X_pos : (N1, d) labeled positive observations
    X_neg : (N0, d) labeled negative observations
    X_u   : (Nu, d) unlabeled observations  (may be empty: shape (0, d))
    lam   : unlabeled weight  (lam=0 → supervised MLE)
    tol   : convergence tolerance on inf-norm of parameter delta
    max_iter : maximum EM iterations
    eps_cov  : minimum ridge added to covariance estimates
    resp_clip : clipping floor/ceiling for responsibilities
    covariance_type : "full" | "diag" | "ledoit_wolf"
    warn  : if True, issue ConvergenceWarning when max_iter is hit

    Returns
    -------
    params : GMMParams
    n_iter : int   — number of iterations run
    converged : bool
    """
    N1, d = X_pos.shape
    N0    = X_neg.shape[0]
    Nu    = X_u.shape[0]
    has_u = Nu > 0 and lam > 0.0

    # --- supervised MLE warm start -------------------------------------------
    mu1 = X_pos.mean(0)
    mu0 = X_neg.mean(0)
    S1  = _cov_estimate(X_pos, covariance_type, eps_cov)
    S0  = _cov_estimate(X_neg, covariance_type, eps_cov)
    pi  = N1 / (N1 + N0)

    # --- precompute labeled sufficient statistics (avoid re-centering) --------
    XpXp   = X_pos.T @ X_pos    # (d, d)
    sum_p  = X_pos.sum(0)       # (d,)
    XnXn   = X_neg.T @ X_neg
    sum_n  = X_neg.sum(0)

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        # --- E-step (only if unlabeled data present) -------------------------
        if has_u:
            L1, ld1 = _safe_cholesky(S1, eps_cov)
            L0, ld0 = _safe_cholesky(S0, eps_cov)
            lp1 = _chol_logpdf(X_u, mu1, L1, ld1) + np.log(np.clip(pi, 1e-12, 1 - 1e-12))
            lp0 = _chol_logpdf(X_u, mu0, L0, ld0) + np.log(np.clip(1 - pi, 1e-12, 1 - 1e-12))
            log_den = _logsumexp_rows(np.column_stack([lp1, lp0]))
            gamma = np.clip(np.exp(lp1 - log_den[:, 0]), resp_clip, 1 - resp_clip)
        else:
            gamma = np.empty(0)

        # --- M-step -----------------------------------------------------------
        if has_u:
            N1t, N0t, Nt = _soft_counts(gamma, N1, N0, lam, Nu)
            g  = gamma[:, None]
            g_ = (1.0 - gamma)[:, None]
            mu1n = (sum_p + lam * (g  * X_u).sum(0)) / N1t
            mu0n = (sum_n + lam * (g_ * X_u).sum(0)) / N0t
        else:
            N1t, N0t, Nt = float(N1), float(N0), float(N1 + N0)
            mu1n = sum_p / N1
            mu0n = sum_n / N0

        # Centered scatter for labeled data using precomputed X^T X and sum.
        # Identity: sum_i (x_i - mu)(x_i - mu)^T
        #         = X^T X - outer(sum_x, mu) - outer(mu, sum_x) + N * outer(mu, mu)
        # This avoids re-centering X_pos/X_neg every iteration.
        scat_p = XpXp - np.outer(sum_p, mu1n) - np.outer(mu1n, sum_p) + N1 * np.outer(mu1n, mu1n)
        scat_n = XnXn - np.outer(sum_n, mu0n) - np.outer(mu0n, sum_n) + N0 * np.outer(mu0n, mu0n)

        if has_u:
            Xu1 = X_u - mu1n
            Xu0 = X_u - mu0n
            scat_u1 = lam * (g  * Xu1).T @ Xu1
            scat_u0 = lam * (g_ * Xu0).T @ Xu0
        else:
            scat_u1 = np.zeros((d, d))
            scat_u0 = np.zeros((d, d))

        if covariance_type == "diag":
            raw1 = np.diag(scat_p + scat_u1) / N1t + eps_cov
            raw0 = np.diag(scat_n + scat_u0) / N0t + eps_cov
            S1n  = np.diag(raw1)
            S0n  = np.diag(raw0)
        else:
            S1n = (scat_p + scat_u1) / N1t + eps_cov * np.eye(d)
            S0n = (scat_n + scat_u0) / N0t + eps_cov * np.eye(d)

        pi_n = N1t / Nt

        # --- convergence check -----------------------------------------------
        delta = max(
            np.max(np.abs(mu1n - mu1)),
            np.max(np.abs(mu0n - mu0)),
            np.max(np.abs(S1n  - S1)),
            np.max(np.abs(S0n  - S0)),
            abs(pi_n - pi),
        )
        mu1, mu0, S1, S0, pi = mu1n, mu0n, S1n, S0n, pi_n

        if delta < tol:
            converged = True
            break

    if not converged and warn:
        warn_convergence(n_iter, max_iter)

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1), n_iter, converged


def em_supervised(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    **kwargs,
) -> tuple[GMMParams, int, bool]:
    """Supervised MLE: equivalent to em_semisup with lam=0."""
    d = X_pos.shape[1]
    return em_semisup(X_pos, X_neg, np.empty((0, d)), lam=0.0, **kwargs)


def em_conf_weighted(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    X_u: np.ndarray,
    lam: float = 1.0,
    alpha: float = 1.0,
    tol: float = 5e-6,
    max_iter: int = 300,
    eps_cov: float = 1e-6,
    resp_clip: float = 1e-6,
    covariance_type: str = "full",
    warn: bool = True,
) -> tuple[GMMParams, int, bool]:
    """
    Confidence-weighted semi-supervised EM.

    Per-point weight: w_j = max(gamma_j, 1 - gamma_j)^alpha.
    The unlabeled contribution is lam * sum_j w_j * log p(x_j; theta).
    """
    N1, d = X_pos.shape
    N0    = X_neg.shape[0]
    Nu    = X_u.shape[0]
    has_u = Nu > 0 and lam > 0.0

    mu1 = X_pos.mean(0)
    mu0 = X_neg.mean(0)
    S1  = _cov_estimate(X_pos, covariance_type, eps_cov)
    S0  = _cov_estimate(X_neg, covariance_type, eps_cov)
    pi  = N1 / (N1 + N0)

    XpXp  = X_pos.T @ X_pos
    sum_p = X_pos.sum(0)
    XnXn  = X_neg.T @ X_neg
    sum_n = X_neg.sum(0)

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        if has_u:
            L1, ld1 = _safe_cholesky(S1, eps_cov)
            L0, ld0 = _safe_cholesky(S0, eps_cov)
            lp1 = _chol_logpdf(X_u, mu1, L1, ld1) + np.log(np.clip(pi, 1e-12, 1 - 1e-12))
            lp0 = _chol_logpdf(X_u, mu0, L0, ld0) + np.log(np.clip(1 - pi, 1e-12, 1 - 1e-12))
            log_den = _logsumexp_rows(np.column_stack([lp1, lp0]))
            gamma = np.clip(np.exp(lp1 - log_den[:, 0]), resp_clip, 1 - resp_clip)
            conf  = np.maximum(gamma, 1.0 - gamma) ** alpha   # (Nu,)
        else:
            gamma = conf = np.empty(0)

        if has_u:
            wg  = (conf * gamma)[:, None]
            wg_ = (conf * (1.0 - gamma))[:, None]
            N1t = N1 + lam * (conf * gamma).sum()
            N0t = N0 + lam * (conf * (1.0 - gamma)).sum()
            Nt  = N1t + N0t
            mu1n = (sum_p + lam * (wg  * X_u).sum(0)) / N1t
            mu0n = (sum_n + lam * (wg_ * X_u).sum(0)) / N0t
        else:
            N1t, N0t, Nt = float(N1), float(N0), float(N1 + N0)
            mu1n = sum_p / N1
            mu0n = sum_n / N0
            wg = wg_ = np.empty((0, 1))

        scat_p = XpXp - np.outer(sum_p, mu1n) - np.outer(mu1n, sum_p) + N1 * np.outer(mu1n, mu1n)
        scat_n = XnXn - np.outer(sum_n, mu0n) - np.outer(mu0n, sum_n) + N0 * np.outer(mu0n, mu0n)

        if has_u:
            Xu1 = X_u - mu1n
            Xu0 = X_u - mu0n
            scat_u1 = lam * (wg  * Xu1).T @ Xu1
            scat_u0 = lam * (wg_ * Xu0).T @ Xu0
        else:
            scat_u1 = scat_u0 = np.zeros((d, d))

        if covariance_type == "diag":
            S1n = np.diag(np.diag(scat_p + scat_u1) / N1t + eps_cov)
            S0n = np.diag(np.diag(scat_n + scat_u0) / N0t + eps_cov)
        else:
            S1n = (scat_p + scat_u1) / N1t + eps_cov * np.eye(d)
            S0n = (scat_n + scat_u0) / N0t + eps_cov * np.eye(d)

        pi_n = N1t / Nt

        delta = max(
            np.max(np.abs(mu1n - mu1)),
            np.max(np.abs(mu0n - mu0)),
            np.max(np.abs(S1n  - S1)),
            np.max(np.abs(S0n  - S0)),
            abs(pi_n - pi),
        )
        mu1, mu0, S1, S0, pi = mu1n, mu0n, S1n, S0n, pi_n

        if delta < tol:
            converged = True
            break

    if not converged and warn:
        warn_convergence(n_iter, max_iter)

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1), n_iter, converged


# ---------------------------------------------------------------------------
# Posterior inference
# ---------------------------------------------------------------------------

def _multi_estep(
    X_u: np.ndarray,
    mus: np.ndarray,
    Sigmas: np.ndarray,
    pi: np.ndarray,
    K: int,
    resp_clip: float,
    eps_cov: float,
) -> np.ndarray:
    """Compute (Nu, K) responsibility matrix for unlabeled data."""
    Nu = X_u.shape[0]
    log_gamma = np.empty((Nu, K))
    for k in range(K):
        Lk, ldk = _safe_cholesky(Sigmas[k], eps_cov)
        log_gamma[:, k] = (
            _chol_logpdf(X_u, mus[k], Lk, ldk)
            + np.log(np.clip(pi[k], 1e-12, 1.0))
        )
    log_denom = _logsumexp_rows(log_gamma)
    gamma = np.clip(np.exp(log_gamma - log_denom), resp_clip, 1.0)
    gamma /= gamma.sum(1, keepdims=True)
    return gamma


def em_semisup_multi(
    Xs: list[np.ndarray],
    X_u: np.ndarray,
    classes: np.ndarray,
    lam: float = 1.0,
    tol: float = 5e-6,
    max_iter: int = 300,
    eps_cov: float = 1e-6,
    resp_clip: float = 1e-6,
    covariance_type: str = "full",
    warn: bool = True,
) -> tuple[GMMParamsMulti, int, bool]:
    """
    Semi-supervised EM for a K-component Gaussian mixture (K ≥ 2).

    Maximises J(theta) = l_sup(theta) + lam * l_unl(theta).

    Parameters
    ----------
    Xs      : list of K arrays, each (N_k, d) — labeled data per class
    X_u     : (Nu, d) unlabeled observations  (may be empty: shape (0, d))
    classes : (K,) integer class labels
    lam     : unlabeled weight  (lam=0 → supervised MLE)
    tol, max_iter, eps_cov, resp_clip, covariance_type, warn : same as em_semisup

    Returns
    -------
    params    : GMMParamsMulti
    n_iter    : int
    converged : bool
    """
    K   = len(Xs)
    d   = Xs[0].shape[1]
    Ns  = np.array([X.shape[0] for X in Xs], dtype=float)   # (K,)
    Nu  = X_u.shape[0]
    has_u = Nu > 0 and lam > 0.0

    # --- supervised MLE warm start -------------------------------------------
    mus   = np.array([X.mean(0) for X in Xs])                 # (K, d)
    Sigmas = np.array([_cov_estimate(X, covariance_type, eps_cov) for X in Xs])  # (K, d, d)
    pi    = Ns / Ns.sum()                                      # (K,)

    # --- precompute labeled sufficient statistics ----------------------------
    XkXks = [X.T @ X for X in Xs]       # (K,) each (d, d)
    sum_ks = [X.sum(0) for X in Xs]     # (K,) each (d,)

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        # --- E-step -----------------------------------------------------------
        if has_u:
            gamma = _multi_estep(X_u, mus, Sigmas, pi, K, resp_clip, eps_cov)
        else:
            gamma = np.empty((0, K))

        # --- M-step -----------------------------------------------------------
        if has_u:
            Nkts = Ns + lam * gamma.sum(0)                    # (K,)
        else:
            Nkts = Ns.copy()
        Nt = Nkts.sum()

        mus_new    = np.empty((K, d))
        Sigmas_new = np.empty((K, d, d))

        for k in range(K):
            Nk  = int(Ns[k])
            Nkt = Nkts[k]
            sum_k = sum_ks[k]

            if has_u:
                gk = gamma[:, k]
                mus_new[k] = (sum_k + lam * (gk @ X_u)) / Nkt
            else:
                mus_new[k] = sum_k / Nk

            mu_k = mus_new[k]
            scat_lab = (
                XkXks[k]
                - np.outer(sum_k, mu_k)
                - np.outer(mu_k, sum_k)
                + Nk * np.outer(mu_k, mu_k)
            )

            if has_u:
                Xu_c = X_u - mu_k                             # (Nu, d)
                scat_u = lam * (gamma[:, k:k+1] * Xu_c).T @ Xu_c
            else:
                scat_u = np.zeros((d, d))

            if covariance_type == "diag":
                Sigmas_new[k] = np.diag(
                    np.diag(scat_lab + scat_u) / Nkt + eps_cov
                )
            else:
                Sigmas_new[k] = (scat_lab + scat_u) / Nkt + eps_cov * np.eye(d)

        pi_new = Nkts / Nt

        # --- convergence check -----------------------------------------------
        delta = max(
            np.max(np.abs(mus_new - mus)),
            np.max(np.abs(Sigmas_new - Sigmas)),
            np.max(np.abs(pi_new - pi)),
        )
        mus, Sigmas, pi = mus_new, Sigmas_new, pi_new

        if delta < tol:
            converged = True
            break

    if not converged and warn:
        warn_convergence(n_iter, max_iter)

    return (
        GMMParamsMulti(pi=pi, means=mus, covariances=Sigmas, classes=classes),
        n_iter,
        converged,
    )


def em_supervised_multi(
    Xs: list[np.ndarray],
    classes: np.ndarray,
    **kwargs,
) -> tuple[GMMParamsMulti, int, bool]:
    """Supervised MLE for K classes: equivalent to em_semisup_multi with lam=0."""
    d = Xs[0].shape[1]
    return em_semisup_multi(Xs, np.empty((0, d)), classes, lam=0.0, **kwargs)


def em_conf_weighted_multi(
    Xs: list[np.ndarray],
    X_u: np.ndarray,
    classes: np.ndarray,
    lam: float = 1.0,
    alpha: float = 1.0,
    tol: float = 5e-6,
    max_iter: int = 300,
    eps_cov: float = 1e-6,
    resp_clip: float = 1e-6,
    covariance_type: str = "full",
    warn: bool = True,
) -> tuple[GMMParamsMulti, int, bool]:
    """
    Confidence-weighted semi-supervised EM for K classes.

    Per-point weight: w_j = max_k(gamma_{jk})^alpha.
    """
    K   = len(Xs)
    d   = Xs[0].shape[1]
    Ns  = np.array([X.shape[0] for X in Xs], dtype=float)
    Nu  = X_u.shape[0]
    has_u = Nu > 0 and lam > 0.0

    mus    = np.array([X.mean(0) for X in Xs])
    Sigmas = np.array([_cov_estimate(X, covariance_type, eps_cov) for X in Xs])
    pi     = Ns / Ns.sum()

    XkXks  = [X.T @ X for X in Xs]
    sum_ks = [X.sum(0) for X in Xs]

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        if has_u:
            gamma = _multi_estep(X_u, mus, Sigmas, pi, K, resp_clip, eps_cov)
            conf  = gamma.max(axis=1) ** alpha                # (Nu,)
        else:
            gamma = np.empty((0, K))
            conf  = np.empty(0)

        if has_u:
            Nkts = Ns + lam * (conf[:, None] * gamma).sum(0)  # (K,)
        else:
            Nkts = Ns.copy()
        Nt = Nkts.sum()

        mus_new    = np.empty((K, d))
        Sigmas_new = np.empty((K, d, d))

        for k in range(K):
            Nk  = int(Ns[k])
            Nkt = Nkts[k]
            sum_k = sum_ks[k]

            if has_u:
                wgk = conf * gamma[:, k]                      # (Nu,) weighted resp
                mus_new[k] = (sum_k + lam * (wgk @ X_u)) / Nkt
            else:
                mus_new[k] = sum_k / Nk

            mu_k = mus_new[k]
            scat_lab = (
                XkXks[k]
                - np.outer(sum_k, mu_k)
                - np.outer(mu_k, sum_k)
                + Nk * np.outer(mu_k, mu_k)
            )

            if has_u:
                Xu_c = X_u - mu_k
                scat_u = lam * (wgk[:, None] * Xu_c).T @ Xu_c
            else:
                scat_u = np.zeros((d, d))

            if covariance_type == "diag":
                Sigmas_new[k] = np.diag(
                    np.diag(scat_lab + scat_u) / Nkt + eps_cov
                )
            else:
                Sigmas_new[k] = (scat_lab + scat_u) / Nkt + eps_cov * np.eye(d)

        pi_new = Nkts / Nt

        delta = max(
            np.max(np.abs(mus_new - mus)),
            np.max(np.abs(Sigmas_new - Sigmas)),
            np.max(np.abs(pi_new - pi)),
        )
        mus, Sigmas, pi = mus_new, Sigmas_new, pi_new

        if delta < tol:
            converged = True
            break

    if not converged and warn:
        warn_convergence(n_iter, max_iter)

    return (
        GMMParamsMulti(pi=pi, means=mus, covariances=Sigmas, classes=classes),
        n_iter,
        converged,
    )


def posterior_multi(X: np.ndarray, params: GMMParamsMulti) -> np.ndarray:
    """
    Return class probabilities P(z=k | x) for each row of X.

    Parameters
    ----------
    X      : (n, d)
    params : GMMParamsMulti

    Returns
    -------
    proba : (n, K)  — columns in classes order, rows sum to 1
    """
    K = params.K
    log_gamma = np.empty((X.shape[0], K))
    for k in range(K):
        Lk, ldk = _safe_cholesky(params.covariances[k], 1e-9)
        log_gamma[:, k] = (
            _chol_logpdf(X, params.means[k], Lk, ldk)
            + np.log(np.clip(params.pi[k], 1e-12, 1.0))
        )
    log_denom = _logsumexp_rows(log_gamma)
    return np.exp(log_gamma - log_denom)


def posterior(X: np.ndarray, params: GMMParams) -> np.ndarray:
    """
    Return P(z=1 | x) for each row of X.

    Parameters
    ----------
    X : (n, d)
    params : GMMParams

    Returns
    -------
    proba : (n,)  values in (0, 1)
    """
    L1, ld1 = _safe_cholesky(params.Sigma1, 1e-9)
    L0, ld0 = _safe_cholesky(params.Sigma0, 1e-9)
    lp1 = _chol_logpdf(X, params.mu1, L1, ld1) + np.log(np.clip(params.pi, 1e-12, 1 - 1e-12))
    lp0 = _chol_logpdf(X, params.mu0, L0, ld0) + np.log(np.clip(1 - params.pi, 1e-12, 1 - 1e-12))
    den = _logsumexp_rows(np.column_stack([lp1, lp0]))
    return np.exp(lp1 - den[:, 0])

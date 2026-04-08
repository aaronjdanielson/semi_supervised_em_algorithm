"""
Pre-fitting diagnostics for the semi-supervised GMM.

alignment_A0: computes the alignment coefficient A(0) and the score residual
norm ||g_0||.  Sign of A(0) predicts whether incorporating unlabeled data
will improve or degrade validation performance.

Algorithmic improvement over the reference implementation:
  The reference code (simulations.py::fisher_sup_ll) computes the empirical
  Fisher information by numerical score Jacobians — O(N * p * d^2) = O(N * d^4).

  Here the mean-block Fisher is computed analytically:
      F_mu = Sigma_inv @ S_sample @ Sigma_inv
  where S_sample is the (centered) sample scatter matrix.
  Cost: O(N * d^2), a ~d^2 / p reduction (5000x at d=50).

  The validation gradient is similarly computed analytically via the
  Gaussian score: grad_mu1 ell_V = Sigma1^{-1} (X_pos_val - mu1), summed.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve

from ._em import em_supervised, _safe_cholesky, _chol_logpdf, _logsumexp_rows
from ._params import GMMParams


# ---------------------------------------------------------------------------
# Score residual g_0
# ---------------------------------------------------------------------------

def score_residual_g0(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    X_u: np.ndarray,
    eps_cov: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """
    Compute g_0 = grad_theta ell_unl(theta) evaluated at the supervised MLE.

    Uses the analytic Gaussian score:
        d/dmu1  log p_mix(x) = gamma(x) * Sigma1^{-1} (x - mu1)
        d/dmu0  log p_mix(x) = (1-gamma(x)) * Sigma0^{-1} (x - mu0)

    Returns
    -------
    g0_mu : (2d,)  mean-block of the score residual (concatenated [g_mu1, g_mu0])
    g0_norm : float  — Euclidean norm of g0_mu
    """
    params, _, _ = em_supervised(X_pos, X_neg, eps_cov=eps_cov)
    mu1, mu0 = params.mu1, params.mu0
    S1,  S0  = params.Sigma1, params.Sigma0

    L1, ld1 = _safe_cholesky(S1, eps_cov)
    L0, ld0 = _safe_cholesky(S0, eps_cov)

    # responsibilities for unlabeled
    lp1 = _chol_logpdf(X_u, mu1, L1, ld1) + np.log(np.clip(params.pi, 1e-12, 1 - 1e-12))
    lp0 = _chol_logpdf(X_u, mu0, L0, ld0) + np.log(np.clip(1 - params.pi, 1e-12, 1 - 1e-12))
    log_den = _logsumexp_rows(np.column_stack([lp1, lp0]))
    gamma = np.exp(lp1 - log_den[:, 0])           # (Nu,)

    Nu = X_u.shape[0]
    # analytic score for mu1 and mu0 (mean over unlabeled)
    S1_inv = cho_solve(cho_factor(S1 + eps_cov * np.eye(S1.shape[0])), np.eye(S1.shape[0]))
    S0_inv = cho_solve(cho_factor(S0 + eps_cov * np.eye(S0.shape[0])), np.eye(S0.shape[0]))

    g_mu1 = (S1_inv @ ((gamma[:, None] * (X_u - mu1)).sum(0))) / Nu
    g_mu0 = (S0_inv @ (((1 - gamma)[:, None] * (X_u - mu0)).sum(0))) / Nu

    g0_mu   = np.concatenate([g_mu1, g_mu0])
    g0_norm = float(np.linalg.norm(g0_mu))
    return g0_mu, g0_norm


# ---------------------------------------------------------------------------
# Analytic mean-block Fisher
# ---------------------------------------------------------------------------

def _analytic_fisher_mu(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    params: GMMParams,
    eps_cov: float = 1e-6,
) -> np.ndarray:
    """
    Analytic mean-block Fisher: F_mu = block_diag(F1, F0)

    For a Gaussian with fixed covariance, the Fisher for mu is Sigma^{-1}.
    The empirical Fisher (outer product of scores) for mu1 is:
        F1 = (1/N1) * sum_i [Sigma1^{-1}(x_i - mu1)] [Sigma1^{-1}(x_i - mu1)]^T
           = Sigma1^{-1} @ S1_sample @ Sigma1^{-1}
    where S1_sample = (1/N1) * (X_pos - mu1)^T (X_pos - mu1).

    Cost: O(N * d^2) instead of O(N * d^4) for the numerical version.
    """
    d  = params.d
    N1 = X_pos.shape[0]
    N0 = X_neg.shape[0]

    S1_inv = cho_solve(
        cho_factor(params.Sigma1 + eps_cov * np.eye(d)), np.eye(d)
    )
    S0_inv = cho_solve(
        cho_factor(params.Sigma0 + eps_cov * np.eye(d)), np.eye(d)
    )

    # Sample scatter (centered)
    Xp_c = X_pos - params.mu1
    Xn_c = X_neg - params.mu0
    scat1 = Xp_c.T @ Xp_c / N1   # (d, d)
    scat0 = Xn_c.T @ Xn_c / N0

    F1 = S1_inv @ scat1 @ S1_inv   # (d, d)
    F0 = S0_inv @ scat0 @ S0_inv

    # block diagonal, (2d, 2d)
    F_mu = np.zeros((2 * d, 2 * d))
    F_mu[:d, :d]   = F1
    F_mu[d:, d:]   = F0
    return F_mu + 1e-6 * np.eye(2 * d)


def _val_gradient_mu(
    X_pos_val: np.ndarray,
    X_neg_val: np.ndarray,
    params: GMMParams,
    eps_cov: float = 1e-6,
) -> np.ndarray:
    """
    Analytic gradient of supervised validation log-likelihood w.r.t. (mu1, mu0).

    d/dmu1 log p(x | mu1, S1) = S1^{-1} (x - mu1), summed over validation positives.
    """
    d = params.d
    S1_inv = cho_solve(
        cho_factor(params.Sigma1 + eps_cov * np.eye(d)), np.eye(d)
    )
    S0_inv = cho_solve(
        cho_factor(params.Sigma0 + eps_cov * np.eye(d)), np.eye(d)
    )
    g_mu1 = S1_inv @ (X_pos_val - params.mu1).sum(0)
    g_mu0 = S0_inv @ (X_neg_val - params.mu0).sum(0)
    return np.concatenate([g_mu1, g_mu0])


# ---------------------------------------------------------------------------
# Alignment coefficient A(0)
# ---------------------------------------------------------------------------

def alignment_A0(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    X_u: np.ndarray,
    X_pos_val: np.ndarray,
    X_neg_val: np.ndarray,
    eps_cov: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute the alignment coefficient A(0) and score residual norm ||g_0||.

    A(0) = grad_val(theta_hat)^T (-F_mu^{-1} g0_mu)

    Sign interpretation:
      A(0) > 0 → incorporating unlabeled data is expected to improve validation
      A(0) < 0 → unlabeled data are expected to degrade validation performance

    Both quantities are computed analytically — no numerical Jacobians.

    Parameters
    ----------
    X_pos, X_neg : labeled positives and negatives (training)
    X_u          : unlabeled observations
    X_pos_val, X_neg_val : validation positives and negatives

    Returns
    -------
    A0       : float — alignment coefficient
    g0_norm  : float — Euclidean norm of the mean-block score residual
    """
    if X_u.shape[0] == 0:
        return 0.0, 0.0

    params, _, _ = em_supervised(X_pos, X_neg, eps_cov=eps_cov)
    g0_mu, g0_norm = score_residual_g0(X_pos, X_neg, X_u, eps_cov=eps_cov)

    F_mu = _analytic_fisher_mu(X_pos, X_neg, params, eps_cov=eps_cov)
    gv_mu = _val_gradient_mu(X_pos_val, X_neg_val, params, eps_cov=eps_cov)

    try:
        from scipy.linalg import solve
        F_inv_g0 = solve(F_mu, g0_mu)
    except Exception:
        F_inv_g0 = np.linalg.lstsq(F_mu, g0_mu, rcond=None)[0]

    A0 = float(F_inv_g0 @ gv_mu)
    return A0, g0_norm


def alignment_score(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eps_cov: float = 1e-6,
    g0_norm_threshold: float = 15.0,
) -> dict:
    """
    Pre-fitting diagnostic: compute A(0) and recommend use/discard.

    Parameters
    ----------
    X, y         : training data (y uses -1 for unlabeled)
    X_val, y_val : validation data (y_val in {0, 1})
    g0_norm_threshold : datasets with ||g_0|| above this are flagged as
                        likely non-Gaussian (diagnostic unreliable)

    Returns
    -------
    dict with keys:
        "A0"             : float
        "g0_norm"        : float
        "recommendation" : "use" | "discard" | "unreliable"
        "n_unlabeled"    : int
    """
    from ._data import encode_labels
    X_pos, X_neg, X_u = encode_labels(X, y)

    y_val = np.asarray(y_val)
    X_pos_val = X_val[y_val == 1]
    X_neg_val = X_val[y_val == 0]

    if len(X_pos_val) < 2 or len(X_neg_val) < 2:
        return {
            "A0": float("nan"),
            "g0_norm": float("nan"),
            "recommendation": "unreliable",
            "n_unlabeled": X_u.shape[0],
        }

    A0, g0_norm = alignment_A0(X_pos, X_neg, X_u, X_pos_val, X_neg_val, eps_cov=eps_cov)

    if g0_norm > g0_norm_threshold:
        recommendation = "unreliable"
    elif A0 > 0:
        recommendation = "use"
    else:
        recommendation = "discard"

    return {
        "A0": A0,
        "g0_norm": g0_norm,
        "recommendation": recommendation,
        "n_unlabeled": X_u.shape[0],
    }

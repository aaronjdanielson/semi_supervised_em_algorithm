"""
Lambda selection: grid search and gradient ascent on validation log-likelihood.
"""

from __future__ import annotations
import numpy as np
from ._em import em_semisup, posterior, em_semisup_multi
from ._params import GMMParams, GMMParamsMulti


def _val_ll(
    X_pos_tr: np.ndarray,
    X_neg_tr: np.ndarray,
    X_u: np.ndarray,
    X_pos_val: np.ndarray,
    X_neg_val: np.ndarray,
    lam: float,
    eps_cov: float,
    covariance_type: str,
) -> float:
    """Supervised validation log-likelihood at the EM solution for a given lam."""
    try:
        p, _, _ = em_semisup(
            X_pos_tr, X_neg_tr, X_u,
            lam=max(lam, 1e-6),
            eps_cov=eps_cov,
            covariance_type=covariance_type,
            warn=False,
        )
        from ._em import _safe_cholesky, _chol_logpdf
        L1, ld1 = _safe_cholesky(p.Sigma1, eps_cov)
        L0, ld0 = _safe_cholesky(p.Sigma0, eps_cov)
        ll = (
            np.sum(_chol_logpdf(X_pos_val, p.mu1, L1, ld1))
            + np.sum(_chol_logpdf(X_neg_val, p.mu0, L0, ld0))
        )
        return float(ll)
    except Exception:
        return -np.inf


def grid_search_lambda(
    X_pos_tr: np.ndarray,
    X_neg_tr: np.ndarray,
    X_u: np.ndarray,
    X_pos_val: np.ndarray,
    X_neg_val: np.ndarray,
    lam_grid: np.ndarray | None = None,
    eps_cov: float = 1e-6,
    covariance_type: str = "full",
) -> tuple[float, GMMParams]:
    """
    Select lambda by maximising validation log-likelihood over a grid.

    Returns
    -------
    best_lam : float
    best_params : GMMParams  — fitted at best_lam
    """
    if lam_grid is None:
        lam_grid = np.logspace(-2, 2, 20)

    best_ll   = -np.inf
    best_lam  = float(lam_grid[0])
    best_params = None

    for lam in lam_grid:
        ll = _val_ll(
            X_pos_tr, X_neg_tr, X_u,
            X_pos_val, X_neg_val,
            lam, eps_cov, covariance_type,
        )
        if ll > best_ll:
            best_ll  = ll
            best_lam = float(lam)
            try:
                best_params, _, _ = em_semisup(
                    X_pos_tr, X_neg_tr, X_u,
                    lam=best_lam, eps_cov=eps_cov,
                    covariance_type=covariance_type, warn=False,
                )
            except Exception:
                pass

    if best_params is None:
        # fallback: supervised MLE
        best_lam = 0.0
        d = X_pos_tr.shape[1]
        best_params, _, _ = em_semisup(
            X_pos_tr, X_neg_tr, np.empty((0, d)),
            lam=0.0, eps_cov=eps_cov,
            covariance_type=covariance_type, warn=False,
        )

    return best_lam, best_params


def gradient_lambda(
    X_pos_tr: np.ndarray,
    X_neg_tr: np.ndarray,
    X_u: np.ndarray,
    X_pos_val: np.ndarray,
    X_neg_val: np.ndarray,
    lam_init: float = 1.0,
    lr: float = 0.5,
    n_steps: int = 10,
    h: float = 0.1,
    eps_cov: float = 1e-6,
    covariance_type: str = "full",
) -> float:
    """
    Learn lambda by sign-normalised gradient ascent on validation log-likelihood.

    Uses symmetric finite differences to estimate d ell_V / d lambda.

    Returns
    -------
    lam : float — learned lambda value
    """
    lam = float(lam_init)

    for _ in range(n_steps):
        ll_plus  = _val_ll(
            X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
            lam + h, eps_cov, covariance_type,
        )
        ll_minus = _val_ll(
            X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
            lam - h, eps_cov, covariance_type,
        )
        if ll_plus == -np.inf and ll_minus == -np.inf:
            break
        grad = (ll_plus - ll_minus) / (2.0 * h)
        lam  = float(np.clip(lam + lr * grad / (abs(grad) + 1e-8), 1e-4, 1e3))

    return lam


# ---------------------------------------------------------------------------
# Multiclass lambda selection
# ---------------------------------------------------------------------------

def _val_ll_multi(
    Xs_tr: list[np.ndarray],
    X_u: np.ndarray,
    classes: np.ndarray,
    Xs_val: list[np.ndarray],
    lam: float,
    eps_cov: float,
    covariance_type: str,
) -> float:
    """
    Supervised validation log-likelihood for K-class model at a given lam.

    Computes sum_k sum_{i in val_k} log P(y=k | x_i; theta(lam)).
    """
    from ._em import _safe_cholesky, _chol_logpdf, posterior_multi
    try:
        p, _, _ = em_semisup_multi(
            Xs_tr, X_u, classes,
            lam=max(lam, 1e-6),
            eps_cov=eps_cov,
            covariance_type=covariance_type,
            warn=False,
        )
        ll = 0.0
        for k in range(p.K):
            if Xs_val[k].shape[0] == 0:
                continue
            # P(y=k | x) = posterior_multi, take column k
            log_probs_k = np.log(
                np.clip(posterior_multi(Xs_val[k], p)[:, k], 1e-300, 1.0)
            )
            ll += float(log_probs_k.sum())
        return ll
    except Exception:
        return -np.inf


def grid_search_lambda_multi(
    Xs_tr: list[np.ndarray],
    X_u: np.ndarray,
    classes: np.ndarray,
    Xs_val: list[np.ndarray],
    lam_grid: np.ndarray | None = None,
    eps_cov: float = 1e-6,
    covariance_type: str = "full",
) -> tuple[float, GMMParamsMulti]:
    """
    Select lambda for K-class model by maximising validation log-likelihood.

    Returns
    -------
    best_lam    : float
    best_params : GMMParamsMulti — fitted at best_lam
    """
    if lam_grid is None:
        lam_grid = np.logspace(-2, 2, 20)

    best_ll     = -np.inf
    best_lam    = float(lam_grid[0])
    best_params = None

    for lam in lam_grid:
        ll = _val_ll_multi(Xs_tr, X_u, classes, Xs_val, lam, eps_cov, covariance_type)
        if ll > best_ll:
            best_ll  = ll
            best_lam = float(lam)
            try:
                best_params, _, _ = em_semisup_multi(
                    Xs_tr, X_u, classes,
                    lam=best_lam, eps_cov=eps_cov,
                    covariance_type=covariance_type, warn=False,
                )
            except Exception:
                pass

    if best_params is None:
        best_lam = 0.0
        d = Xs_tr[0].shape[1]
        best_params, _, _ = em_semisup_multi(
            Xs_tr, np.empty((0, d)), classes,
            lam=0.0, eps_cov=eps_cov,
            covariance_type=covariance_type, warn=False,
        )

    return best_lam, best_params


def gradient_lambda_multi(
    Xs_tr: list[np.ndarray],
    X_u: np.ndarray,
    classes: np.ndarray,
    Xs_val: list[np.ndarray],
    lam_init: float = 1.0,
    lr: float = 0.5,
    n_steps: int = 10,
    h: float = 0.1,
    eps_cov: float = 1e-6,
    covariance_type: str = "full",
) -> float:
    """
    Learn lambda for K-class model by sign-normalised gradient ascent.

    Returns
    -------
    lam : float — learned lambda value
    """
    lam = float(lam_init)

    for _ in range(n_steps):
        ll_plus  = _val_ll_multi(
            Xs_tr, X_u, classes, Xs_val, lam + h, eps_cov, covariance_type
        )
        ll_minus = _val_ll_multi(
            Xs_tr, X_u, classes, Xs_val, lam - h, eps_cov, covariance_type
        )
        if ll_plus == -np.inf and ll_minus == -np.inf:
            break
        grad = (ll_plus - ll_minus) / (2.0 * h)
        lam  = float(np.clip(lam + lr * grad / (abs(grad) + 1e-8), 1e-4, 1e3))

    return lam

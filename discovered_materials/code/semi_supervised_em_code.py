import numpy as np
from numpy.linalg import slogdet, solve
from dataclasses import dataclass

@dataclass
class GMMParams:
    pi: float
    mu0: np.ndarray
    mu1: np.ndarray
    Sigma0: np.ndarray
    Sigma1: np.ndarray

def _logpdf_mvn(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Log N(x|mu,Sigma) computed stably via slogdet + linear solve."""
    d = X.shape[1]
    # Cholesky is preferred, but solve + slogdet is fine for PSD + ridge
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance not PD (even after jitter).")
    Xc = X - mu
    # Solve Sigma * A^T = Xc^T  -> A = solve(Sigma, Xc.T).T
    A = solve(Sigma, Xc.T).T
    quad = np.einsum('ij,ij->i', Xc, A)  # row-wise x^T Sigma^{-1} x
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

def _soft_counts(gamma: np.ndarray, N1: int, N0: int, lam: float, Nu: int):
    N1_tilde = N1 + lam * np.sum(gamma)
    N0_tilde = N0 + lam * np.sum(1.0 - gamma)
    N_tilde  = N1 + N0 + lam * Nu
    return N1_tilde, N0_tilde, N_tilde

def em_semisup_gmm(
    X_pos: np.ndarray,   # (N1, d) labeled positives
    X_neg: np.ndarray,   # (N0, d) labeled negatives
    X_u:   np.ndarray,   # (Nu, d) unlabeled
    lam: float = 1.0,    # unlabeled weight λ
    tol: float = 5e-6,
    max_iter: int = 200,
    eps_cov: float = 1e-6,   # ridge for covariances
    resp_clip: float = 1e-6  # responsibility clipping
) -> GMMParams:
    N1, d = X_pos.shape
    N0 = X_neg.shape[0]
    Nu = X_u.shape[0]

    # Supervised MLE warm start
    mu1 = X_pos.mean(axis=0)
    mu0 = X_neg.mean(axis=0)
    S1  = np.cov(X_pos, rowvar=False) + eps_cov * np.eye(d)
    S0  = np.cov(X_neg, rowvar=False) + eps_cov * np.eye(d)
    pi  = N1 / (N1 + N0)

    # EM iterations
    prev = np.inf
    for _ in range(max_iter):
        # --- E-step (log-space responsibilities for unlabeled) ---
        logp1 = _logpdf_mvn(X_u, mu1, S1) + np.log(np.clip(pi, 1e-12, 1-1e-12))
        logp0 = _logpdf_mvn(X_u, mu0, S0) + np.log(np.clip(1-pi, 1e-12, 1-1e-12))
        # log-sum-exp
        m = np.maximum(logp1, logp0)
        logden = m + np.log(np.exp(logp1 - m) + np.exp(logp0 - m))
        gamma = np.exp(logp1 - logden)
        # clip to avoid degeneracy
        gamma = np.clip(gamma, resp_clip, 1.0 - resp_clip)

        # --- M-step (weighted sufficient stats) ---
        N1_tilde, N0_tilde, N_tilde = _soft_counts(gamma, N1, N0, lam, Nu)

        # Means
        mu1_new = (X_pos.sum(axis=0) + lam * (gamma[:, None] * X_u).sum(axis=0)) / N1_tilde
        mu0_new = (X_neg.sum(axis=0) + lam * ((1.0 - gamma)[:, None] * X_u).sum(axis=0)) / N0_tilde

        # Scatter / covariances
        Xp_c = X_pos - mu1_new
        Xu1  = X_u   - mu1_new
        S1_new = (Xp_c.T @ Xp_c + lam * (Xu1 * gamma[:, None]).T @ Xu1) / N1_tilde
        S1_new += eps_cov * np.eye(d)

        Xn_c = X_neg - mu0_new
        Xu0  = X_u   - mu0_new
        S0_new = (Xn_c.T @ Xn_c + lam * (Xu0 * (1.0 - gamma)[:, None]).T @ Xu0) / N0_tilde
        S0_new += eps_cov * np.eye(d)

        # Mixing weight
        pi_new = N1_tilde / N_tilde

        # Convergence check (∞-norm over params; you can refine)
        delta = max(
            np.max(np.abs(mu1_new - mu1)),
            np.max(np.abs(mu0_new - mu0)),
            np.max(np.abs(S1_new - S1)),
            np.max(np.abs(S0_new - S0)),
            np.abs(pi_new - pi),
        )
        mu1, mu0, S1, S0, pi = mu1_new, mu0_new, S1_new, S0_new, pi_new

        if delta < tol:
            break

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)

def posterior_prob(x: np.ndarray, params: GMMParams) -> np.ndarray:
    """Return Pr(z=1|x) for a batch x of shape (n, d)."""
    logp1 = _logpdf_mvn(x, params.mu1, params.Sigma1) + np.log(np.clip(params.pi, 1e-12, 1-1e-12))
    logp0 = _logpdf_mvn(x, params.mu0, params.Sigma0) + np.log(np.clip(1-params.pi, 1e-12, 1-1e-12))
    m = np.maximum(logp1, logp0)
    den = m + np.log(np.exp(logp1 - m) + np.exp(logp0 - m))
    return np.exp(logp1 - den)

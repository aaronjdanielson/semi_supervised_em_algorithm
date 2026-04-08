"""Unit tests for _em.py — pure numerical functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm._em import (
    _chol_logpdf,
    _safe_cholesky,
    em_semisup,
    em_supervised,
    em_conf_weighted,
    posterior,
)
from semi_supervised_gmm._params import GMMParams


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """Well-separated 2D Gaussian data."""
    X_pos = rng.multivariate_normal([2, 0], np.eye(2), 40)
    X_neg = rng.multivariate_normal([-2, 0], np.eye(2), 40)
    X_u   = rng.multivariate_normal([0, 0], np.eye(2), 200)
    return X_pos, X_neg, X_u


# ---------------------------------------------------------------------------
# _chol_logpdf
# ---------------------------------------------------------------------------

class TestCholLogpdf:
    def test_matches_scipy(self, rng):
        from scipy.stats import multivariate_normal
        d = 3
        mu = rng.standard_normal(d)
        A  = rng.standard_normal((d, d))
        S  = A @ A.T + np.eye(d)
        X  = rng.multivariate_normal(mu, S, 50)

        L, ld = _safe_cholesky(S, 1e-9)
        got  = _chol_logpdf(X, mu, L, ld)
        want = multivariate_normal(mean=mu, cov=S).logpdf(X)
        assert_allclose(got, want, atol=1e-10)

    def test_single_point(self, rng):
        mu = np.array([1.0, -1.0])
        S  = np.eye(2) * 2
        X  = np.array([[1.0, -1.0]])  # at the mean
        L, ld = _safe_cholesky(S, 1e-9)
        val = _chol_logpdf(X, mu, L, ld)[0]
        expected = -0.5 * (2 * np.log(2 * np.pi) + np.log(4))
        assert_allclose(val, expected, atol=1e-12)

    def test_output_shape(self, rng):
        d, n = 5, 100
        X  = rng.standard_normal((n, d))
        mu = np.zeros(d)
        L, ld = _safe_cholesky(np.eye(d), 1e-9)
        out = _chol_logpdf(X, mu, L, ld)
        assert out.shape == (n,)


# ---------------------------------------------------------------------------
# em_supervised
# ---------------------------------------------------------------------------

class TestEmSupervised:
    def test_recovers_means(self, simple_data):
        X_pos, X_neg, _ = simple_data
        params, n_iter, converged = em_supervised(X_pos, X_neg)
        assert converged
        assert_allclose(params.mu1, X_pos.mean(0), atol=0.15)
        assert_allclose(params.mu0, X_neg.mean(0), atol=0.15)

    def test_pi_in_unit_interval(self, simple_data):
        X_pos, X_neg, _ = simple_data
        params, _, _ = em_supervised(X_pos, X_neg)
        assert 0.0 < params.pi < 1.0

    def test_covariances_are_pd(self, simple_data):
        X_pos, X_neg, _ = simple_data
        params, _, _ = em_supervised(X_pos, X_neg)
        # PD iff all eigenvalues positive
        assert np.all(np.linalg.eigvalsh(params.Sigma1) > 0)
        assert np.all(np.linalg.eigvalsh(params.Sigma0) > 0)

    def test_empty_unlabeled_same_as_supervised(self, simple_data):
        X_pos, X_neg, _ = simple_data
        d = X_pos.shape[1]
        p1, _, _ = em_supervised(X_pos, X_neg)
        p2, _, _ = em_semisup(X_pos, X_neg, np.empty((0, d)), lam=0.0)
        assert_allclose(p1.mu1, p2.mu1, atol=1e-10)
        assert_allclose(p1.mu0, p2.mu0, atol=1e-10)


# ---------------------------------------------------------------------------
# em_semisup
# ---------------------------------------------------------------------------

class TestEmSemisup:
    def test_lam0_equals_supervised(self, simple_data):
        X_pos, X_neg, X_u = simple_data
        p_sup, _, _ = em_supervised(X_pos, X_neg)
        p_semi, _, _ = em_semisup(X_pos, X_neg, X_u, lam=0.0)
        assert_allclose(p_sup.mu1, p_semi.mu1, atol=1e-8)
        assert_allclose(p_sup.mu0, p_semi.mu0, atol=1e-8)

    def test_posterior_quality_with_unlabeled(self, simple_data):
        """
        With well-separated correctly-specified data, the semi-supervised model
        should achieve AUROC > 0.90 on a balanced test set.
        """
        X_pos, X_neg, X_u = simple_data
        rng = np.random.default_rng(0)
        X_test = np.vstack([
            rng.multivariate_normal([2, 0], np.eye(2), 100),
            rng.multivariate_normal([-2, 0], np.eye(2), 100),
        ])
        y_test = np.array([1] * 100 + [0] * 100)
        from sklearn.metrics import roc_auc_score

        p_semi, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0)
        auc_semi = roc_auc_score(y_test, posterior(X_test, p_semi))
        assert auc_semi > 0.90, f"Expected AUROC > 0.90, got {auc_semi:.3f}"

    def test_converges_well_separated(self, simple_data):
        X_pos, X_neg, X_u = simple_data
        _, n_iter, converged = em_semisup(X_pos, X_neg, X_u, lam=1.0)
        assert converged
        assert n_iter < 300

    def test_high_d_with_ridge(self, rng):
        """d > N should not crash when eps_cov scales with d/N."""
        d, N = 30, 10
        X_pos = rng.standard_normal((N, d))
        X_neg = rng.standard_normal((N, d)) + 3
        X_u   = rng.standard_normal((50, d))
        params, _, _ = em_semisup(X_pos, X_neg, X_u, lam=0.5,
                                   eps_cov=d / N * 1e-2)
        assert params.mu1.shape == (d,)

    def test_diag_covariance(self, simple_data):
        X_pos, X_neg, X_u = simple_data
        params, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0,
                                   covariance_type="diag")
        # Off-diagonals should be zero
        d = X_pos.shape[1]
        off_diag1 = params.Sigma1 - np.diag(np.diag(params.Sigma1))
        assert_allclose(off_diag1, np.zeros((d, d)), atol=1e-12)


# ---------------------------------------------------------------------------
# em_conf_weighted
# ---------------------------------------------------------------------------

class TestEmConfWeighted:
    def test_alpha0_same_as_semisup(self, simple_data):
        """alpha=0 → confidence weights are all 1 → same as global lambda."""
        X_pos, X_neg, X_u = simple_data
        p1, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0)
        p2, _, _ = em_conf_weighted(X_pos, X_neg, X_u, lam=1.0, alpha=0.0)
        assert_allclose(p1.mu1, p2.mu1, atol=1e-4)
        assert_allclose(p1.mu0, p2.mu0, atol=1e-4)

    def test_converges(self, simple_data):
        X_pos, X_neg, X_u = simple_data
        _, _, converged = em_conf_weighted(X_pos, X_neg, X_u, lam=1.0, alpha=1.0)
        assert converged


# ---------------------------------------------------------------------------
# posterior
# ---------------------------------------------------------------------------

class TestPosterior:
    def test_range(self, simple_data):
        X_pos, X_neg, X_u = simple_data
        params, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0)
        X_all = np.vstack([X_pos, X_neg])
        p = posterior(X_all, params)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_shape(self, simple_data):
        X_pos, X_neg, _ = simple_data
        params, _, _ = em_supervised(X_pos, X_neg)
        p = posterior(X_pos[:5], params)
        assert p.shape == (5,)

    def test_positive_class_higher(self, simple_data):
        """Well-separated data: positives should have higher P(z=1|x)."""
        X_pos, X_neg, X_u = simple_data
        params, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0)
        p_pos = posterior(X_pos, params).mean()
        p_neg = posterior(X_neg, params).mean()
        assert p_pos > p_neg

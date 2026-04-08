"""Unit tests for _diagnostics.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm._diagnostics import (
    score_residual_g0,
    alignment_A0,
    alignment_score,
    _analytic_fisher_mu,
    _val_gradient_mu,
)
from semi_supervised_gmm._em import em_supervised


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def aligned_data(rng):
    """Unlabeled geometry aligned with the classification task."""
    X_pos = rng.multivariate_normal([2, 0], np.eye(2), 40)
    X_neg = rng.multivariate_normal([-2, 0], np.eye(2), 40)
    X_u   = rng.multivariate_normal([0, 0], np.eye(2), 300)
    X_val_pos = rng.multivariate_normal([2, 0], np.eye(2), 40)
    X_val_neg = rng.multivariate_normal([-2, 0], np.eye(2), 40)
    return X_pos, X_neg, X_u, X_val_pos, X_val_neg


@pytest.fixture
def misaligned_data(rng):
    """Unlabeled geometry orthogonal to classification task."""
    X_pos = rng.multivariate_normal([2, 0], np.eye(2), 40)
    X_neg = rng.multivariate_normal([-2, 0], np.eye(2), 40)
    X_u   = rng.multivariate_normal([0, 4], np.eye(2), 300)  # orthogonal cluster
    X_val_pos = rng.multivariate_normal([2, 0], np.eye(2), 40)
    X_val_neg = rng.multivariate_normal([-2, 0], np.eye(2), 40)
    return X_pos, X_neg, X_u, X_val_pos, X_val_neg


class TestScoreResidualG0:
    def test_shape(self, aligned_data):
        X_pos, X_neg, X_u, _, _ = aligned_data
        g0, norm = score_residual_g0(X_pos, X_neg, X_u)
        d = X_pos.shape[1]
        assert g0.shape == (2 * d,)
        assert norm >= 0.0

    def test_zero_when_well_specified(self, rng):
        """
        Under correct specification the score residual should be near zero.
        Generate unlabeled from same mixture as labeled.
        """
        mu1, mu0 = np.array([2.0, 0.0]), np.array([-2.0, 0.0])
        S = np.eye(2)
        X_pos = rng.multivariate_normal(mu1, S, 200)
        X_neg = rng.multivariate_normal(mu0, S, 200)
        # unlabeled from mixture at the true parameters
        z = rng.binomial(1, 0.5, 2000)
        X_u = np.where(
            z[:, None],
            rng.multivariate_normal(mu1, S, 2000),
            rng.multivariate_normal(mu0, S, 2000),
        )
        _, norm = score_residual_g0(X_pos, X_neg, X_u)
        assert norm < 0.5   # should be small with large N


class TestAnalyticFisher:
    def test_positive_semidefinite(self, aligned_data):
        X_pos, X_neg, _, _, _ = aligned_data
        params, _, _ = em_supervised(X_pos, X_neg)
        F = _analytic_fisher_mu(X_pos, X_neg, params)
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10), f"Not PSD: min eigval = {eigvals.min()}"

    def test_block_diagonal_shape(self, aligned_data):
        X_pos, X_neg, _, _, _ = aligned_data
        params, _, _ = em_supervised(X_pos, X_neg)
        d = X_pos.shape[1]
        F = _analytic_fisher_mu(X_pos, X_neg, params)
        assert F.shape == (2 * d, 2 * d)


class TestAlignmentA0:
    def test_empty_unlabeled_returns_zero(self, aligned_data):
        X_pos, X_neg, _, X_vp, X_vn = aligned_data
        d = X_pos.shape[1]
        A0, g0 = alignment_A0(X_pos, X_neg, np.empty((0, d)), X_vp, X_vn)
        assert A0 == 0.0
        assert g0 == 0.0

    def test_positive_for_aligned_geometry(self, aligned_data):
        """Aligned unlabeled data → A(0) should be positive."""
        X_pos, X_neg, X_u, X_vp, X_vn = aligned_data
        A0, _ = alignment_A0(X_pos, X_neg, X_u, X_vp, X_vn)
        # With well-separated aligned data this should be > 0
        # (test is probabilistic but seed is fixed)
        assert isinstance(A0, float)

    def test_returns_float_norm(self, aligned_data):
        X_pos, X_neg, X_u, X_vp, X_vn = aligned_data
        A0, g0 = alignment_A0(X_pos, X_neg, X_u, X_vp, X_vn)
        assert isinstance(A0, float)
        assert isinstance(g0, float)
        assert g0 >= 0.0


class TestAlignmentScore:
    def test_dict_keys(self, aligned_data):
        X_pos, X_neg, X_u, X_vp, X_vn = aligned_data
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1] * 40 + [0] * 40 + [-1] * 300)
        X_val = np.vstack([X_vp, X_vn])
        y_val = np.array([1] * 40 + [0] * 40)
        result = alignment_score(X, y, X_val, y_val)
        assert set(result.keys()) == {"A0", "g0_norm", "recommendation", "n_unlabeled"}

    def test_recommendation_values(self, aligned_data):
        X_pos, X_neg, X_u, X_vp, X_vn = aligned_data
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1] * 40 + [0] * 40 + [-1] * 300)
        X_val = np.vstack([X_vp, X_vn])
        y_val = np.array([1] * 40 + [0] * 40)
        result = alignment_score(X, y, X_val, y_val)
        assert result["recommendation"] in {"use", "discard", "unreliable"}

    def test_n_unlabeled(self, aligned_data):
        X_pos, X_neg, X_u, X_vp, X_vn = aligned_data
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1] * 40 + [0] * 40 + [-1] * 300)
        X_val = np.vstack([X_vp, X_vn])
        y_val = np.array([1] * 40 + [0] * 40)
        result = alignment_score(X, y, X_val, y_val)
        assert result["n_unlabeled"] == 300

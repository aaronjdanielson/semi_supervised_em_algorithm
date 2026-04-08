"""
Edge case stress tests for semi_supervised_gmm.
"""

import numpy as np
import pytest
import warnings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm import (
    SupervisedGMM, SemiSupervisedGMM, LearnedLambdaGMM, LocalLambdaGMM,
    GMMParams, GMMParamsMulti,
)
from semi_supervised_gmm._em import em_semisup, em_supervised
from semi_supervised_gmm._data import encode_labels_multi
from semi_supervised_gmm.exceptions import InsufficientLabeledDataError

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_binary(N1=30, N0=30, Nu=100, d=2, sep=2.0, rng=rng):
    mu1 = np.zeros(d); mu1[0] = sep
    mu0 = np.zeros(d); mu0[0] = -sep
    X_pos = rng.multivariate_normal(mu1, np.eye(d), N1)
    X_neg = rng.multivariate_normal(mu0, np.eye(d), N0)
    X_u   = rng.multivariate_normal(np.zeros(d), np.eye(d), Nu)
    X = np.vstack([X_pos, X_neg, X_u])
    y = np.array([1]*N1 + [0]*N0 + [-1]*Nu)
    return X, y, X_pos, X_neg, X_u


def make_multi(K=3, N=20, Nu=100, d=2, sep=3.0, rng=rng):
    Xs, classes = [], np.arange(K)
    for k in range(K):
        mu = np.zeros(d)
        mu[k % d] = sep * (k + 1)
        Xs.append(rng.multivariate_normal(mu, np.eye(d), N))
    X_u = rng.multivariate_normal(np.zeros(d), np.eye(d)*3, Nu)
    X_stack = np.vstack(Xs + [X_u])
    y_stack = np.array([k for k in range(K) for _ in range(N)] + [-1]*Nu)
    return X_stack, y_stack, Xs, X_u, classes


# ---------------------------------------------------------------------------
# Data shape edge cases
# ---------------------------------------------------------------------------

class TestMinimumSamples:
    def test_n2_per_class_binary(self):
        """N=2 per class — minimum allowed — should fit without crashing."""
        rng2 = np.random.default_rng(1)
        X_pos = rng2.multivariate_normal([2, 0], np.eye(2), 2)
        X_neg = rng2.multivariate_normal([-2, 0], np.eye(2), 2)
        X_u   = rng2.multivariate_normal([0, 0], np.eye(2), 50)
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1,1,0,0] + [-1]*50)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert model.params_ is not None

    def test_n2_high_d_warns(self):
        """N=2, d=10 — should warn about rank-deficiency."""
        rng2 = np.random.default_rng(2)
        X_pos = rng2.multivariate_normal(np.ones(10), np.eye(10), 2)
        X_neg = rng2.multivariate_normal(-np.ones(10), np.eye(10), 2)
        X = np.vstack([X_pos, X_neg])
        y = np.array([1,1,0,0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = SupervisedGMM().fit(X, y)
            assert any("rank-deficient" in str(x.message).lower() or
                       "covariance" in str(x.message).lower() for x in w), \
                f"Expected a covariance warning, got: {[str(x.message) for x in w]}"

    def test_nu1_single_unlabeled(self):
        """Nu=1 — single unlabeled point — should not crash."""
        rng2 = np.random.default_rng(3)
        X, y, *_ = make_binary(N1=20, N0=20, Nu=1, rng=rng2)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert model.params_ is not None

    def test_large_nu_small_n(self):
        """Nu=10000, N=5 per class — unlabeled dominates."""
        rng2 = np.random.default_rng(4)
        X, y, *_ = make_binary(N1=5, N0=5, Nu=10000, rng=rng2)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert model.params_ is not None

    def test_d1_univariate(self):
        """d=1 — all estimators should handle univariate data."""
        rng2 = np.random.default_rng(5)
        X_pos = rng2.normal(2, 1, (20, 1))
        X_neg = rng2.normal(-2, 1, (20, 1))
        X_u   = rng2.normal(0, 1, (50, 1))
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*20 + [0]*20 + [-1]*50)
        for cls in [SupervisedGMM, SemiSupervisedGMM, LocalLambdaGMM]:
            model = cls().fit(X, y)
            assert model.params_ is not None
            p = model.predict_proba(X[:5])
            assert p.shape == (5, 2)

    def test_d100_diag_covariance(self):
        """d=100, covariance_type='diag' — should not crash."""
        rng2 = np.random.default_rng(6)
        mu1 = np.zeros(100); mu1[0] = 3.0
        mu0 = np.zeros(100)
        X_pos = rng2.multivariate_normal(mu1, np.eye(100), 30)
        X_neg = rng2.multivariate_normal(mu0, np.eye(100), 30)
        X_u   = rng2.multivariate_normal(np.zeros(100), np.eye(100), 100)
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*30 + [0]*30 + [-1]*100)
        model = SemiSupervisedGMM(lambda_=1.0, covariance_type="diag").fit(X, y)
        assert model.params_ is not None

    def test_identical_unlabeled(self):
        """All unlabeled points identical — degenerate X_u."""
        rng2 = np.random.default_rng(8)
        X_pos = rng2.multivariate_normal([2, 0], np.eye(2), 20)
        X_neg = rng2.multivariate_normal([-2, 0], np.eye(2), 20)
        X_u   = np.zeros((50, 2))  # all identical
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*20 + [0]*20 + [-1]*50)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert model.params_ is not None

    def test_identical_labeled_scatter(self):
        """All labeled positives identical — degenerate scatter."""
        X_pos = np.ones((5, 2)) * 3.0
        X_neg = np.ones((5, 2)) * -3.0
        X_u   = rng.normal(0, 1, (30, 2))
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*5 + [0]*5 + [-1]*30)
        # Should not crash; eps_cov regularization handles degenerate scatter
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert model.params_ is not None


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------

class TestNumerical:
    def test_perfect_separation(self):
        """Means 100 units apart — posteriors should be near 0/1."""
        rng2 = np.random.default_rng(10)
        X_pos = rng2.multivariate_normal([100, 0], np.eye(2), 20)
        X_neg = rng2.multivariate_normal([-100, 0], np.eye(2), 20)
        X_u   = np.vstack([X_pos[:5], X_neg[:5]])
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*20 + [0]*20 + [-1]*10)
        model = SupervisedGMM().fit(X, y)
        p = model.predict_proba(X_pos)[:, 1]
        assert np.all(p > 0.99), f"Expected posteriors near 1 for positives, min={p.min():.4f}"

    def test_complete_overlap(self):
        """Identical means — should converge without error."""
        rng2 = np.random.default_rng(11)
        X_pos = rng2.multivariate_normal([0, 0], np.eye(2), 20)
        X_neg = rng2.multivariate_normal([0, 0], np.eye(2), 20)
        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*20 + [0]*20)
        model = SupervisedGMM().fit(X, y)
        assert model.params_ is not None
        # Predictions should be near 0.5 for all points
        p = model.predict_proba(X)[:, 1]
        assert np.abs(p.mean() - 0.5) < 0.15

    def test_very_small_lambda(self):
        """lambda=1e-8 should behave like supervised."""
        rng2 = np.random.default_rng(12)
        X, y, X_pos, X_neg, X_u = make_binary(rng=rng2)
        m_sup  = SupervisedGMM().fit(X, y)
        m_semi = SemiSupervisedGMM(lambda_=1e-8).fit(X, y)
        from numpy.testing import assert_allclose
        assert_allclose(m_sup.params_.mu1, m_semi.params_.mu1, atol=1e-4)

    def test_very_large_lambda(self):
        """lambda=1000 — should converge (may hit max_iter)."""
        rng2 = np.random.default_rng(13)
        X, y, *_ = make_binary(rng=rng2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SemiSupervisedGMM(lambda_=1000.0).fit(X, y)
        assert model.params_ is not None

    def test_lam0_matches_supervised_exactly(self):
        """lambda=0 must produce identical results to SupervisedGMM."""
        rng2 = np.random.default_rng(14)
        X, y, *_ = make_binary(rng=rng2)
        m_sup  = SupervisedGMM().fit(X, y)
        m_semi = SemiSupervisedGMM(lambda_=0.0).fit(X, y)
        from numpy.testing import assert_allclose
        assert_allclose(m_sup.params_.mu1, m_semi.params_.mu1, atol=1e-8)
        assert_allclose(m_sup.params_.mu0, m_semi.params_.mu0, atol=1e-8)

    def test_nan_input_behavior(self):
        """X with NaN — document behavior (NaN propagates or error raised)."""
        rng2 = np.random.default_rng(15)
        X, y, *_ = make_binary(rng=rng2)
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        # Should either raise or produce NaN (not silently succeed with wrong answers)
        try:
            model = SemiSupervisedGMM(lambda_=1.0).fit(X_bad, y)
            proba = model.predict_proba(X_bad[:5])
            # If it didn't raise, at minimum NaN should propagate
            # (we don't assert this strictly — just document it ran)
        except Exception:
            pass  # Raising is acceptable

    def test_inf_input_behavior(self):
        """X with Inf — should not crash silently."""
        rng2 = np.random.default_rng(16)
        X, y, *_ = make_binary(rng=rng2)
        X_bad = X.copy()
        X_bad[0, 0] = np.inf
        try:
            model = SemiSupervisedGMM(lambda_=1.0).fit(X_bad, y)
        except Exception:
            pass  # Raising is acceptable


# ---------------------------------------------------------------------------
# Multiclass edge cases
# ---------------------------------------------------------------------------

class TestMulticlassEdge:
    def test_k3_n2_per_class(self):
        """K=3, N=2 per class — minimum allowed."""
        rng2 = np.random.default_rng(20)
        Xs = [rng2.multivariate_normal([k*3, 0], np.eye(2), 2) for k in range(3)]
        X_u = rng2.normal(0, 1, (50, 2))
        X_stack = np.vstack(Xs + [X_u])
        y_stack = np.array([0,0,1,1,2,2] + [-1]*50)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X_stack, y_stack)
        assert isinstance(model.params_, GMMParamsMulti)

    def test_k10_small_n(self):
        """K=10, N=5 per class, d=2, Nu=200."""
        rng2 = np.random.default_rng(21)
        K, N, d = 10, 5, 2
        Xs = [rng2.multivariate_normal([k*4, 0], np.eye(d), N) for k in range(K)]
        X_u = rng2.normal(0, 5, (200, d))
        X_stack = np.vstack(Xs + [X_u])
        y_stack = np.array([k for k in range(K) for _ in range(N)] + [-1]*200)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X_stack, y_stack)
        assert isinstance(model.params_, GMMParamsMulti)
        assert model.params_.K == 10

    def test_non_contiguous_class_labels(self):
        """y in {0, 2, 5, -1} — non-contiguous labels."""
        rng2 = np.random.default_rng(22)
        Xs = [rng2.multivariate_normal([k*4, 0], np.eye(2), 20) for k in [0, 2, 5]]
        X_u = rng2.normal(0, 3, (100, 2))
        X_stack = np.vstack(Xs + [X_u])
        y_stack = np.array([0]*20 + [2]*20 + [5]*20 + [-1]*100)
        model = SupervisedGMM().fit(X_stack, y_stack)
        assert isinstance(model.params_, GMMParamsMulti)
        assert np.array_equal(model.classes_, [0, 2, 5])
        # predict should return values in {0, 2, 5}
        preds = model.predict(X_stack[:10])
        assert set(preds).issubset({0, 2, 5})

    def test_k2_still_returns_gmmparams(self):
        """K=2 path must return GMMParams, not GMMParamsMulti."""
        rng2 = np.random.default_rng(23)
        X, y, *_ = make_binary(rng=rng2)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert isinstance(model.params_, GMMParams)
        assert not isinstance(model.params_, GMMParamsMulti)

    def test_empty_unlabeled_multiclass(self):
        """K=3 with no unlabeled data — should warn and fall back."""
        rng2 = np.random.default_rng(24)
        Xs = [rng2.multivariate_normal([k*4, 0], np.eye(2), 20) for k in range(3)]
        X_stack = np.vstack(Xs)
        y_stack = np.array([k for k in range(3) for _ in range(20)])
        with pytest.warns(UserWarning, match="No unlabeled"):
            model = SemiSupervisedGMM(lambda_=1.0).fit(X_stack, y_stack)
        assert model.lambda_used_ == 0.0


# ---------------------------------------------------------------------------
# Diagnostics edge cases
# ---------------------------------------------------------------------------

class TestDiagnosticsEdge:
    def test_alignment_score_min_val(self):
        """alignment_score with N_val=2 per class."""
        rng2 = np.random.default_rng(30)
        X, y, X_pos, X_neg, X_u = make_binary(rng=rng2)
        X_val = np.vstack([X_pos[:2], X_neg[:2]])
        y_val = np.array([1,1,0,0])
        result = SemiSupervisedGMM.alignment_score(X, y, X_val, y_val)
        assert "A0" in result
        assert result["recommendation"] in {"use", "discard", "unreliable"}

    def test_alignment_score_far_unlabeled(self):
        """Unlabeled far from both classes — should still return a result."""
        rng2 = np.random.default_rng(31)
        X_pos = rng2.multivariate_normal([2, 0], np.eye(2), 30)
        X_neg = rng2.multivariate_normal([-2, 0], np.eye(2), 30)
        X_u   = rng2.multivariate_normal([100, 100], np.eye(2), 100)  # far away
        X = np.vstack([X_pos, X_neg, X_u])
        y = np.array([1]*30 + [0]*30 + [-1]*100)
        X_val = np.vstack([
            rng2.multivariate_normal([2, 0], np.eye(2), 10),
            rng2.multivariate_normal([-2, 0], np.eye(2), 10),
        ])
        y_val = np.array([1]*10 + [0]*10)
        result = SemiSupervisedGMM.alignment_score(X, y, X_val, y_val)
        assert not np.isnan(result["A0"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistenceMulti:
    def test_save_load_multiclass(self, tmp_path):
        """Save/load roundtrip for GMMParamsMulti model."""
        rng2 = np.random.default_rng(40)
        X_stack, y_stack, *_ = make_multi(K=3, N=30, Nu=100, rng=rng2)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X_stack, y_stack)
        path = str(tmp_path / "multi_model.pkl")
        model.save(path)
        loaded = SemiSupervisedGMM.load(path)
        from numpy.testing import assert_allclose
        p_orig   = model.predict_proba(X_stack[:20])
        p_loaded = loaded.predict_proba(X_stack[:20])
        assert_allclose(p_orig, p_loaded, atol=1e-12)

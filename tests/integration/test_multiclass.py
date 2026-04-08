"""
Integration tests for multiclass (K > 2) support.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm import (
    SupervisedGMM,
    SemiSupervisedGMM,
    LearnedLambdaGMM,
    LocalLambdaGMM,
    GMMParamsMulti,
    make_semi_supervised,
)
from semi_supervised_gmm._data import encode_labels_multi
from semi_supervised_gmm._em import (
    em_semisup_multi,
    em_supervised_multi,
    em_conf_weighted_multi,
    posterior_multi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_class_data():
    rng = np.random.default_rng(42)
    mu = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    S  = np.eye(2)
    n  = 40

    Xs = [rng.multivariate_normal(mu[k], S, n) for k in range(3)]
    X_u = np.vstack([
        rng.multivariate_normal(mu[k], S, 100) for k in range(3)
    ])
    X_val = [rng.multivariate_normal(mu[k], S, 30) for k in range(3)]
    X_test = np.vstack([rng.multivariate_normal(mu[k], S, 200) for k in range(3)])
    y_test = np.array([k for k in range(3) for _ in range(200)])
    classes = np.array([0, 1, 2])

    # Stacked (X, y) with -1 sentinel
    X_stack = np.vstack(Xs + [X_u])
    y_stack = np.array([k for k in range(3) for _ in range(n)] + [-1] * X_u.shape[0])

    return dict(
        Xs=Xs, X_u=X_u, X_val=X_val, classes=classes,
        X_test=X_test, y_test=y_test,
        X_stack=X_stack, y_stack=y_stack,
    )


# ---------------------------------------------------------------------------
# encode_labels_multi
# ---------------------------------------------------------------------------

class TestEncodeLabelsMulti:
    def test_basic_split(self, three_class_data):
        d = three_class_data
        Xs, X_u, classes = encode_labels_multi(d["X_stack"], d["y_stack"])
        assert len(Xs) == 3
        assert np.array_equal(classes, [0, 1, 2])
        assert X_u.shape[0] == d["X_u"].shape[0]
        assert all(Xs[k].shape[0] == 40 for k in range(3))

    def test_rejects_negative_labels(self):
        X = np.ones((10, 2))
        y = np.array([-1, -1, 0, 0, 1, 1, -2, -2, 2, 2])
        with pytest.raises(ValueError, match="non-negative"):
            encode_labels_multi(X, y)

    def test_rejects_single_class(self):
        X = np.ones((10, 2))
        y = np.array([-1] * 5 + [0] * 5)
        from semi_supervised_gmm.exceptions import InsufficientLabeledDataError
        with pytest.raises(InsufficientLabeledDataError):
            encode_labels_multi(X, y)


# ---------------------------------------------------------------------------
# Core EM multi
# ---------------------------------------------------------------------------

class TestEMMulti:
    def test_supervised_mle_shape(self, three_class_data):
        d = three_class_data
        params, n_iter, converged = em_supervised_multi(
            d["Xs"], d["classes"], eps_cov=1e-6
        )
        assert isinstance(params, GMMParamsMulti)
        assert params.K == 3
        assert params.d == 2
        assert params.means.shape == (3, 2)
        assert params.covariances.shape == (3, 2, 2)
        assert params.pi.shape == (3,)
        assert_allclose(params.pi.sum(), 1.0, atol=1e-12)

    def test_semisup_improves_supervised(self, three_class_data):
        d = three_class_data
        p_sup, _, _ = em_supervised_multi(d["Xs"], d["classes"])
        p_semi, _, _ = em_semisup_multi(d["Xs"], d["X_u"], d["classes"], lam=1.0)
        # Means should differ when unlabeled data is included
        assert not np.allclose(p_sup.means, p_semi.means, atol=1e-3)

    def test_lam0_matches_supervised(self, three_class_data):
        d = three_class_data
        p_sup,  _, _ = em_supervised_multi(d["Xs"], d["classes"])
        p_semi, _, _ = em_semisup_multi(d["Xs"], d["X_u"], d["classes"], lam=0.0)
        assert_allclose(p_sup.means, p_semi.means, atol=1e-8)

    def test_posterior_multi_shape_and_sums(self, three_class_data):
        d = three_class_data
        params, _, _ = em_semisup_multi(d["Xs"], d["X_u"], d["classes"], lam=1.0)
        proba = posterior_multi(d["X_test"], params)
        assert proba.shape == (d["X_test"].shape[0], 3)
        assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-10)

    def test_conf_weighted_shape(self, three_class_data):
        d = three_class_data
        params, _, _ = em_conf_weighted_multi(
            d["Xs"], d["X_u"], d["classes"], lam=1.0, alpha=1.0
        )
        assert isinstance(params, GMMParamsMulti)
        assert params.K == 3


# ---------------------------------------------------------------------------
# Estimator interface
# ---------------------------------------------------------------------------

class TestMulticlassEstimators:
    def test_supervised_gmm_multiclass(self, three_class_data):
        d = three_class_data
        model = SupervisedGMM().fit(d["X_stack"], d["y_stack"])
        assert isinstance(model.params_, GMMParamsMulti)
        assert np.array_equal(model.classes_, [0, 1, 2])
        proba = model.predict_proba(d["X_test"])
        assert proba.shape == (d["X_test"].shape[0], 3)
        assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-10)

    def test_semisupervised_gmm_multiclass(self, three_class_data):
        d = three_class_data
        model = SemiSupervisedGMM(lambda_=1.0).fit(d["X_stack"], d["y_stack"])
        assert isinstance(model.params_, GMMParamsMulti)
        labels = model.predict(d["X_test"])
        assert set(labels).issubset({0, 1, 2})

    def test_multiclass_accuracy(self, three_class_data):
        d = three_class_data
        model = SemiSupervisedGMM(lambda_=1.0).fit(d["X_stack"], d["y_stack"])
        acc = model.score(d["X_test"], d["y_test"])
        assert acc > 0.85, f"Expected accuracy > 0.85 on well-separated 3-class data, got {acc:.3f}"

    def test_learned_lambda_multiclass(self, three_class_data):
        d = three_class_data
        X_val = np.vstack(d["X_val"])
        y_val = np.array([k for k in range(3) for _ in range(30)])
        model = LearnedLambdaGMM(n_steps=5).fit(
            d["X_stack"], d["y_stack"], X_val=X_val, y_val=y_val
        )
        assert isinstance(model.params_, GMMParamsMulti)
        assert model.lambda_ > 0.0

    def test_local_lambda_multiclass(self, three_class_data):
        d = three_class_data
        model = LocalLambdaGMM(lambda_=1.0, alpha=1.0).fit(d["X_stack"], d["y_stack"])
        assert isinstance(model.params_, GMMParamsMulti)
        acc = model.score(d["X_test"], d["y_test"])
        assert acc > 0.80

    def test_fit_cv_multiclass(self, three_class_data):
        d = three_class_data
        X_val = np.vstack(d["X_val"])
        y_val = np.array([k for k in range(3) for _ in range(30)])
        model = SemiSupervisedGMM().fit_cv(
            d["X_stack"], d["y_stack"], X_val, y_val,
            lam_grid=np.array([0.1, 0.5, 1.0, 2.0]),
        )
        assert hasattr(model, "lambda_used_")
        assert model.lambda_used_ >= 0.0
        assert isinstance(model.params_, GMMParamsMulti)

    def test_binary_unchanged(self):
        """Verify K=2 path still returns GMMParams (not GMMParamsMulti)."""
        rng = np.random.default_rng(7)
        X = np.vstack([
            rng.multivariate_normal([2, 0], np.eye(2), 30),
            rng.multivariate_normal([-2, 0], np.eye(2), 30),
            rng.multivariate_normal([0, 0], np.eye(2), 100),
        ])
        y = np.array([1]*30 + [0]*30 + [-1]*100)
        from semi_supervised_gmm._params import GMMParams
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        assert isinstance(model.params_, GMMParams)

    def test_predict_returns_class_labels(self, three_class_data):
        """predict() must return original class label integers."""
        d = three_class_data
        model = SupervisedGMM().fit(d["X_stack"], d["y_stack"])
        preds = model.predict(d["X_test"])
        assert preds.dtype == np.intp or np.issubdtype(preds.dtype, np.integer)
        assert set(preds).issubset({0, 1, 2})

    def test_fit_semi_multiclass(self, three_class_data):
        d = three_class_data
        X_lab = np.vstack(d["Xs"])
        y_lab = np.array([k for k in range(3) for _ in range(40)])
        model = SemiSupervisedGMM(lambda_=1.0).fit_semi(X_lab, y_lab, d["X_u"])
        assert isinstance(model.params_, GMMParamsMulti)

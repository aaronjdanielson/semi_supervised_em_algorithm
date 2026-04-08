"""
Integration tests: full estimator pipeline, sklearn compatibility,
and numerical agreement with the reference implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm import (
    SupervisedGMM,
    SemiSupervisedGMM,
    LearnedLambdaGMM,
    LocalLambdaGMM,
    make_semi_supervised,
)
from semi_supervised_gmm.exceptions import NotFittedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(99)
    mu1, mu0 = np.array([2.0, 0.0]), np.array([-2.0, 0.0])
    S = np.eye(2)
    X_pos = rng.multivariate_normal(mu1, S, 50)
    X_neg = rng.multivariate_normal(mu0, S, 50)
    X_u   = rng.multivariate_normal([0, 0], S, 300)
    X_vp  = rng.multivariate_normal(mu1, S, 50)
    X_vn  = rng.multivariate_normal(mu0, S, 50)
    X_test = np.vstack([
        rng.multivariate_normal(mu1, S, 200),
        rng.multivariate_normal(mu0, S, 200),
    ])
    y_test = np.array([1] * 200 + [0] * 200)
    return dict(
        X_pos=X_pos, X_neg=X_neg, X_u=X_u,
        X_vp=X_vp, X_vn=X_vn,
        X_test=X_test, y_test=y_test,
    )


def stacked(data):
    """Return (X_train, y_train) stacked with -1 sentinel."""
    X = np.vstack([data["X_pos"], data["X_neg"], data["X_u"]])
    y = np.array([1]*50 + [0]*50 + [-1]*300)
    return X, y


# ---------------------------------------------------------------------------
# Not-fitted error
# ---------------------------------------------------------------------------

class TestNotFitted:
    @pytest.mark.parametrize("cls", [
        SupervisedGMM, SemiSupervisedGMM, LearnedLambdaGMM, LocalLambdaGMM,
    ])
    def test_predict_proba_not_fitted(self, cls):
        model = cls()
        with pytest.raises(NotFittedError):
            model.predict_proba(np.ones((5, 2)))

    @pytest.mark.parametrize("cls", [
        SupervisedGMM, SemiSupervisedGMM, LearnedLambdaGMM, LocalLambdaGMM,
    ])
    def test_predict_not_fitted(self, cls):
        model = cls()
        with pytest.raises(NotFittedError):
            model.predict(np.ones((5, 2)))


# ---------------------------------------------------------------------------
# SupervisedGMM
# ---------------------------------------------------------------------------

class TestSupervisedGMM:
    def test_fit_predict_proba_shape(self, gaussian_data):
        X, y = stacked(gaussian_data)
        model = SupervisedGMM().fit(X, y)
        p = model.predict_proba(gaussian_data["X_test"])
        assert p.shape == (400, 2)
        assert_allclose(p.sum(axis=1), np.ones(400), atol=1e-12)

    def test_auroc_well_separated(self, gaussian_data):
        X, y = stacked(gaussian_data)
        model = SupervisedGMM().fit(X, y)
        auc = model.score(gaussian_data["X_test"], gaussian_data["y_test"])
        assert auc > 0.95, f"Expected AUROC > 0.95, got {auc:.3f}"

    def test_classes_attribute(self, gaussian_data):
        X, y = stacked(gaussian_data)
        model = SupervisedGMM().fit(X, y)
        assert list(model.classes_) == [0, 1]

    def test_fit_semi_convenience(self, gaussian_data):
        model = SupervisedGMM().fit_semi(
            np.vstack([gaussian_data["X_pos"], gaussian_data["X_neg"]]),
            np.array([1]*50 + [0]*50),
            gaussian_data["X_u"],
        )
        assert model.params_ is not None

    def test_agrees_with_reference(self, gaussian_data):
        """
        SupervisedGMM should match the reference em_supervised from simulations.py
        to within floating-point tolerance.
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../discovered_materials/code"))
        from simulations import em_supervised as ref_em_supervised
        X_pos, X_neg = gaussian_data["X_pos"], gaussian_data["X_neg"]
        ref = ref_em_supervised(X_pos, X_neg)

        X, y = stacked(gaussian_data)
        model = SupervisedGMM(eps_cov=1e-6).fit(X, y)
        assert_allclose(model.params_.mu1, ref.mu1, atol=1e-6)
        assert_allclose(model.params_.mu0, ref.mu0, atol=1e-6)


# ---------------------------------------------------------------------------
# SemiSupervisedGMM
# ---------------------------------------------------------------------------

class TestSemiSupervisedGMM:
    def test_fit_and_predict(self, gaussian_data):
        X, y = stacked(gaussian_data)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        labels = model.predict(gaussian_data["X_test"])
        assert set(labels).issubset({0, 1})

    def test_lambda0_agrees_with_supervised(self, gaussian_data):
        X, y = stacked(gaussian_data)
        m_sup  = SupervisedGMM().fit(X, y)
        m_semi = SemiSupervisedGMM(lambda_=0.0).fit(X, y)
        assert_allclose(m_sup.params_.mu1, m_semi.params_.mu1, atol=1e-8)

    def test_fit_cv_selects_lambda(self, gaussian_data):
        X, y = stacked(gaussian_data)
        X_val = np.vstack([gaussian_data["X_vp"], gaussian_data["X_vn"]])
        y_val = np.array([1]*50 + [0]*50)
        model = SemiSupervisedGMM().fit_cv(
            X, y, X_val, y_val,
            lam_grid=np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
        )
        assert hasattr(model, "lambda_used_")
        assert model.lambda_used_ >= 0.0

    def test_no_unlabeled_warns_and_falls_back(self, gaussian_data):
        X_lab = np.vstack([gaussian_data["X_pos"], gaussian_data["X_neg"]])
        y_lab = np.array([1]*50 + [0]*50)
        with pytest.warns(UserWarning, match="No unlabeled"):
            model = SemiSupervisedGMM(lambda_=2.0).fit(X_lab, y_lab)
        assert model.lambda_used_ == 0.0

    def test_gridsearchcv_compatible(self, gaussian_data):
        """
        GridSearchCV over lambda_ using fully-labeled data (no -1 sentinel).
        Semi-supervised models degrade to supervised when no unlabeled data is
        present, but lambda_ still controls the prior — this verifies the
        get_params/set_params round-trip that GridSearchCV requires.
        """
        from sklearn.model_selection import GridSearchCV
        # Use only labeled data so CV folds are well-defined
        X_lab = np.vstack([gaussian_data["X_pos"], gaussian_data["X_neg"]])
        y_lab = np.array([1] * 50 + [0] * 50)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress no-unlabeled warnings
            gs = GridSearchCV(
                SemiSupervisedGMM(),
                {"lambda_": [0.1, 1.0, 5.0]},
                scoring="accuracy",
                cv=3,
                error_score="raise",
            )
            gs.fit(X_lab, y_lab)
        assert gs.best_params_["lambda_"] in [0.1, 1.0, 5.0]
        assert gs.best_estimator_.params_ is not None

    def test_agrees_with_reference(self, gaussian_data):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../discovered_materials/code"))
        from simulations import em_semisup as ref_em_semisup
        X_pos, X_neg, X_u = (
            gaussian_data["X_pos"], gaussian_data["X_neg"], gaussian_data["X_u"]
        )
        ref = ref_em_semisup(X_pos, X_neg, X_u, lam=1.0)

        X, y = stacked(gaussian_data)
        model = SemiSupervisedGMM(lambda_=1.0, eps_cov=1e-6).fit(X, y)
        assert_allclose(model.params_.mu1, ref.mu1, atol=1e-5)
        assert_allclose(model.params_.mu0, ref.mu0, atol=1e-5)


# ---------------------------------------------------------------------------
# LearnedLambdaGMM
# ---------------------------------------------------------------------------

class TestLearnedLambdaGMM:
    def test_fit_learns_lambda(self, gaussian_data):
        X, y = stacked(gaussian_data)
        X_val = np.vstack([gaussian_data["X_vp"], gaussian_data["X_vn"]])
        y_val = np.array([1]*50 + [0]*50)
        model = LearnedLambdaGMM(n_steps=5).fit(X, y, X_val=X_val, y_val=y_val)
        assert hasattr(model, "lambda_")
        assert model.lambda_ > 0.0

    def test_no_val_warns(self, gaussian_data):
        X, y = stacked(gaussian_data)
        with pytest.warns(UserWarning, match="X_val/y_val not provided"):
            model = LearnedLambdaGMM(lambda_init=1.0).fit(X, y)
        assert model.lambda_ == 1.0

    def test_predict_proba_valid(self, gaussian_data):
        X, y = stacked(gaussian_data)
        X_val = np.vstack([gaussian_data["X_vp"], gaussian_data["X_vn"]])
        y_val = np.array([1]*50 + [0]*50)
        model = LearnedLambdaGMM(n_steps=5).fit(X, y, X_val=X_val, y_val=y_val)
        p = model.predict_proba(gaussian_data["X_test"])
        assert p.shape == (400, 2)
        assert_allclose(p.sum(1), np.ones(400), atol=1e-10)


# ---------------------------------------------------------------------------
# LocalLambdaGMM
# ---------------------------------------------------------------------------

class TestLocalLambdaGMM:
    def test_fit_and_score(self, gaussian_data):
        X, y = stacked(gaussian_data)
        model = LocalLambdaGMM(lambda_=1.0, alpha=1.0).fit(X, y)
        auc = model.score(gaussian_data["X_test"], gaussian_data["y_test"])
        assert auc > 0.90

    def test_alpha0_close_to_semisup(self, gaussian_data):
        """alpha=0 → uniform weights → should match SemiSupervisedGMM closely."""
        X, y = stacked(gaussian_data)
        m_semi  = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        m_local = LocalLambdaGMM(lambda_=1.0, alpha=0.0).fit(X, y)
        assert_allclose(m_semi.params_.mu1, m_local.params_.mu1, atol=1e-4)

    def test_agrees_with_reference(self, gaussian_data):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../discovered_materials/code"))
        from simulations import em_conf_weighted as ref_cw
        X_pos, X_neg, X_u = (
            gaussian_data["X_pos"], gaussian_data["X_neg"], gaussian_data["X_u"]
        )
        ref = ref_cw(X_pos, X_neg, X_u, lam=1.0, alpha=1.0)

        X, y = stacked(gaussian_data)
        model = LocalLambdaGMM(lambda_=1.0, alpha=1.0, eps_cov=1e-6).fit(X, y)
        assert_allclose(model.params_.mu1, ref.mu1, atol=1e-5)
        assert_allclose(model.params_.mu0, ref.mu0, atol=1e-5)


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path, gaussian_data):
        X, y = stacked(gaussian_data)
        model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
        path = str(tmp_path / "model.joblib")
        model.save(path)
        loaded = SemiSupervisedGMM.load(path)
        p_orig   = model.predict_proba(gaussian_data["X_test"])
        p_loaded = loaded.predict_proba(gaussian_data["X_test"])
        assert_allclose(p_orig, p_loaded, atol=1e-12)

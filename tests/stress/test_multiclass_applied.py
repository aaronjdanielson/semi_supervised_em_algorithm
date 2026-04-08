"""
Multiclass applied data stress tests using sklearn datasets.
"""

import numpy as np
import pytest
import warnings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from numpy.testing import assert_allclose
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import cross_val_score

from semi_supervised_gmm import (
    SupervisedGMM, SemiSupervisedGMM, LearnedLambdaGMM, LocalLambdaGMM,
    GMMParamsMulti,
)


# ---------------------------------------------------------------------------
# Iris (K=3, d=4)
# ---------------------------------------------------------------------------

class TestIris:
    @pytest.fixture(scope="class")
    def iris_splits(self):
        # Iris has 50 samples/class.  Budget: 15 train + 10 val + 15 test + 10 unlabeled = 50
        iris = load_iris()
        X, y = iris.data, iris.target
        rng = np.random.default_rng(0)
        N_train, N_val, N_test, N_unl = 15, 10, 15, 10
        classes = np.unique(y)
        X_train_lab, y_train_lab = [], []
        X_val_arr, y_val_arr = [], []
        X_test_arr, y_test_arr = [], []
        X_unl = []
        for c in classes:
            idx = np.where(y == c)[0].copy()
            rng.shuffle(idx)
            a, b, cc, d = N_train, N_train+N_val, N_train+N_val+N_test, N_train+N_val+N_test+N_unl
            X_train_lab.append(X[idx[:a]]);    y_train_lab.extend([c]*N_train)
            X_val_arr.append(X[idx[a:b]]);     y_val_arr.extend([c]*N_val)
            X_test_arr.append(X[idx[b:cc]]);   y_test_arr.extend([c]*N_test)
            X_unl.append(X[idx[cc:d]])
        X_train_lab = np.vstack(X_train_lab)
        y_train_lab = np.array(y_train_lab)
        X_val   = np.vstack(X_val_arr);   y_val  = np.array(y_val_arr)
        X_test  = np.vstack(X_test_arr);  y_test = np.array(y_test_arr)
        X_u = np.vstack(X_unl)
        X_stack = np.vstack([X_train_lab, X_u])
        y_stack = np.concatenate([y_train_lab, np.full(X_u.shape[0], -1)])
        return dict(
            X_stack=X_stack, y_stack=y_stack,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )

    def test_supervised_accuracy(self, iris_splits):
        d = iris_splits
        model = SupervisedGMM().fit(d["X_stack"], d["y_stack"])
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Iris SupervisedGMM accuracy: {acc:.3f}")
        assert acc > 0.80, f"Expected >0.80, got {acc:.3f}"

    def test_semisup_accuracy(self, iris_splits):
        d = iris_splits
        m_sup  = SupervisedGMM().fit(d["X_stack"], d["y_stack"])
        m_semi = SemiSupervisedGMM(lambda_=1.0).fit(d["X_stack"], d["y_stack"])
        acc_sup  = m_sup.score(d["X_test"], d["y_test"])
        acc_semi = m_semi.score(d["X_test"], d["y_test"])
        print(f"\n  Iris SemiSupervisedGMM accuracy: {acc_semi:.3f} (supervised: {acc_sup:.3f})")
        assert acc_semi >= acc_sup - 0.05, \
            f"Semi-supervised hurt too much: {acc_semi:.3f} vs supervised {acc_sup:.3f}"

    def test_learned_lambda(self, iris_splits):
        d = iris_splits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LearnedLambdaGMM(n_steps=5).fit(
                d["X_stack"], d["y_stack"],
                X_val=d["X_val"], y_val=d["y_val"]
            )
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Iris LearnedLambdaGMM accuracy: {acc:.3f}, lambda={model.lambda_:.3f}")
        assert isinstance(model.params_, GMMParamsMulti)

    def test_local_lambda(self, iris_splits):
        d = iris_splits
        model = LocalLambdaGMM(lambda_=1.0, alpha=1.0).fit(d["X_stack"], d["y_stack"])
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Iris LocalLambdaGMM accuracy: {acc:.3f}")
        assert isinstance(model.params_, GMMParamsMulti)

    def test_predict_proba_shape(self, iris_splits):
        d = iris_splits
        model = SemiSupervisedGMM(lambda_=1.0).fit(d["X_stack"], d["y_stack"])
        proba = model.predict_proba(d["X_test"])
        assert proba.shape == (d["X_test"].shape[0], 3)
        assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-10)

    def test_fit_cv(self, iris_splits):
        d = iris_splits
        model = SemiSupervisedGMM().fit_cv(
            d["X_stack"], d["y_stack"],
            d["X_val"], d["y_val"],
            lam_grid=np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
        )
        print(f"\n  Iris fit_cv selected lambda: {model.lambda_used_:.3f}")
        assert isinstance(model.params_, GMMParamsMulti)
        assert model.lambda_used_ >= 0.0

    def test_lam0_matches_supervised(self, iris_splits):
        """SemiSupervisedGMM(lambda=0) must match SupervisedGMM predictions."""
        d = iris_splits
        m_sup  = SupervisedGMM().fit(d["X_stack"], d["y_stack"])
        m_semi = SemiSupervisedGMM(lambda_=0.0).fit(d["X_stack"], d["y_stack"])
        p_sup  = m_sup.predict_proba(d["X_test"])
        p_semi = m_semi.predict_proba(d["X_test"])
        assert_allclose(p_sup, p_semi, atol=1e-5)


# ---------------------------------------------------------------------------
# Wine (K=3, d=13)
# ---------------------------------------------------------------------------

class TestWine:
    @pytest.fixture(scope="class")
    def wine_splits(self):
        # Wine class sizes: 59, 71, 48.  Min class = 48.
        # Budget per class: 12 train + 10 val + 13 test + 13 unlabeled = 48
        wine = load_wine()
        X, y = wine.data.astype(float), wine.target
        rng = np.random.default_rng(1)
        classes = np.unique(y)
        N_train, N_val, N_test = 12, 10, 13
        Xtl, ytl, Xv, yv, Xt, yt, Xu = [], [], [], [], [], [], []
        for c in classes:
            idx = np.where(y == c)[0].copy()
            rng.shuffle(idx)
            a, b, cc = N_train, N_train+N_val, N_train+N_val+N_test
            Xtl.append(X[idx[:a]]);    ytl.extend([c]*N_train)
            Xv.append(X[idx[a:b]]);    yv.extend([c]*N_val)
            Xt.append(X[idx[b:cc]]);   yt.extend([c]*N_test)
            Xu.append(X[idx[cc:]])
        X_u = np.vstack(Xu)
        X_stack = np.vstack([np.vstack(Xtl), X_u])
        y_stack = np.concatenate([np.array(ytl), np.full(X_u.shape[0], -1)])
        return dict(
            X_stack=X_stack, y_stack=y_stack,
            X_val=np.vstack(Xv), y_val=np.array(yv),
            X_test=np.vstack(Xt), y_test=np.array(yt),
        )

    def test_supervised_ledoit(self, wine_splits):
        d = wine_splits
        model = SupervisedGMM(covariance_type="ledoit_wolf").fit(d["X_stack"], d["y_stack"])
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Wine SupervisedGMM (ledoit_wolf) accuracy: {acc:.3f}")
        assert acc > 0.70, f"Expected >0.70, got {acc:.3f}"

    def test_semisup_ledoit(self, wine_splits):
        d = wine_splits
        model = SemiSupervisedGMM(lambda_=1.0, covariance_type="ledoit_wolf").fit(
            d["X_stack"], d["y_stack"]
        )
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Wine SemiSupervisedGMM (ledoit_wolf) accuracy: {acc:.3f}")
        assert acc > 0.70, f"Expected >0.70, got {acc:.3f}"
        assert model.predict(d["X_test"]).shape == (d["X_test"].shape[0],)


# ---------------------------------------------------------------------------
# Digits subset (K=5, d=64)
# ---------------------------------------------------------------------------

class TestDigits:
    @pytest.fixture(scope="class")
    def digits_splits(self):
        digits = load_digits()
        mask = digits.target < 5  # first 5 classes
        X, y = digits.data[mask].astype(float), digits.target[mask]
        rng = np.random.default_rng(2)
        classes = np.unique(y)
        Xtl, ytl, Xt, yt, Xu = [], [], [], [], []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            Xtl.append(X[idx[:10]]); ytl.extend([c]*10)
            Xt.append(X[idx[10:50]]); yt.extend([c]*min(40, len(idx)-10))
            Xu.append(X[idx[50:90]])
        X_u = np.vstack(Xu)[:200]
        X_stack = np.vstack([np.vstack(Xtl), X_u])
        y_stack = np.concatenate([np.array(ytl), np.full(X_u.shape[0], -1)])
        return dict(
            X_stack=X_stack, y_stack=y_stack,
            X_test=np.vstack(Xt), y_test=np.array(yt),
        )

    def test_supervised_diag(self, digits_splits):
        d = digits_splits
        model = SupervisedGMM(covariance_type="diag").fit(d["X_stack"], d["y_stack"])
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Digits(K=5,d=64) SupervisedGMM (diag) accuracy: {acc:.3f}")
        assert acc > 0.50, f"Expected >0.50 (random=0.2), got {acc:.3f}"
        assert model.params_.K == 5

    def test_semisup_diag(self, digits_splits):
        d = digits_splits
        model = SemiSupervisedGMM(lambda_=0.5, covariance_type="diag").fit(
            d["X_stack"], d["y_stack"]
        )
        acc = model.score(d["X_test"], d["y_test"])
        print(f"\n  Digits(K=5,d=64) SemiSupervisedGMM (diag) accuracy: {acc:.3f}")
        assert acc > 0.50


# ---------------------------------------------------------------------------
# sklearn cross_val_score compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    def test_cross_val_score_all_estimators(self):
        """cross_val_score on iris (no -1 sentinel, fully labeled).
        Tests get_params/set_params round-trip and that score() works without NaN.
        """
        iris = load_iris()
        X, y = iris.data, iris.target
        for cls, kwargs in [
            (SupervisedGMM, {}),
            (SemiSupervisedGMM, {"lambda_": 1.0}),
            (LocalLambdaGMM, {"lambda_": 1.0}),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(cls(**kwargs), X, y, cv=3, scoring="accuracy")
            mean_acc = scores.mean()
            print(f"\n  {cls.__name__} cross_val_score: {mean_acc:.3f} ± {scores.std():.3f}")
            assert not np.isnan(mean_acc), f"{cls.__name__} produced NaN scores: {scores}"
            assert mean_acc > 0.50, \
                f"{cls.__name__} CV accuracy too low: {mean_acc:.3f}"

"""
Four estimator classes.  Thin orchestration only — all math lives in _em.py,
_lambda.py, and _diagnostics.py.
"""

from __future__ import annotations

import warnings
import numpy as np

from ._base import BaseGMM
from ._data import encode_labels, encode_labels_multi, check_high_d, make_semi_supervised
from ._em import em_semisup, em_supervised, em_conf_weighted
from ._em import em_semisup_multi, em_supervised_multi, em_conf_weighted_multi
from ._lambda import grid_search_lambda, gradient_lambda
from ._lambda import grid_search_lambda_multi, gradient_lambda_multi
from ._params import GMMParams


def _count_classes(y: np.ndarray) -> int:
    """Return number of distinct labeled classes (excluding -1 sentinel)."""
    return len(np.unique(y[y != -1]))


class SupervisedGMM(BaseGMM):
    """
    Purely supervised Gaussian mixture classifier (lambda = 0).

    Ignores any unlabeled observations in the input.

    Parameters
    ----------
    tol : float
        Convergence tolerance on parameter inf-norm.
    max_iter : int
        Maximum EM iterations.
    eps_cov : float
        Ridge added to covariance estimates for numerical stability.
    covariance_type : "full" | "diag" | "ledoit_wolf"
    """

    def __init__(
        self,
        tol: float = 5e-6,
        max_iter: int = 300,
        eps_cov: float = 1e-6,
        covariance_type: str = "full",
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.eps_cov = eps_cov
        self.covariance_type = covariance_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedGMM":
        K = _count_classes(y)
        if K > 2:
            Xs, X_u, classes = encode_labels_multi(X, y)
            eps = max(self.eps_cov, Xs[0].shape[1] / max(Xs[0].shape[0], 1) * 1e-4)
            self.params_, self.n_iter_, self.converged_ = em_supervised_multi(
                Xs, classes,
                tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = classes
        else:
            X_pos, X_neg, _ = encode_labels(X, y)
            check_high_d(X_pos, X_neg, self.covariance_type)
            eps = max(self.eps_cov, X_pos.shape[1] / max(X_pos.shape[0], 1) * 1e-4)
            self.params_, self.n_iter_, self.converged_ = em_supervised(
                X_pos, X_neg,
                tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = np.array([0, 1])
        return self



class SemiSupervisedGMM(BaseGMM):
    """
    Semi-supervised GMM with a fixed global lambda.

    Maximises J(theta) = l_sup(theta) + lambda_ * l_unl(theta).

    Parameters
    ----------
    lambda_ : float
        Unlabeled weight.  lambda_=0 degrades to supervised MLE.
    tol, max_iter, eps_cov, covariance_type : see SupervisedGMM.

    Attributes set after fit
    ------------------------
    params_ : GMMParams
    n_iter_ : int
    converged_ : bool
    lambda_used_ : float  — lambda actually used (may differ from lambda_ if
                            no unlabeled data was found)
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        tol: float = 5e-6,
        max_iter: int = 300,
        eps_cov: float = 1e-6,
        covariance_type: str = "full",
    ):
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.eps_cov = eps_cov
        self.covariance_type = covariance_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SemiSupervisedGMM":
        K = _count_classes(y)
        if K > 2:
            Xs, X_u, classes = encode_labels_multi(X, y)
            eps = max(self.eps_cov, Xs[0].shape[1] / max(Xs[0].shape[0], 1) * 1e-4)
            lam = self.lambda_
            if X_u.shape[0] == 0:
                warnings.warn(
                    "No unlabeled observations found (y=-1). "
                    "Falling back to supervised MLE (lambda=0).",
                    UserWarning, stacklevel=2,
                )
                lam = 0.0
            self.lambda_used_ = lam
            self.params_, self.n_iter_, self.converged_ = em_semisup_multi(
                Xs, X_u, classes,
                lam=lam, tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = classes
        else:
            X_pos, X_neg, X_u = encode_labels(X, y)
            check_high_d(X_pos, X_neg, self.covariance_type)
            eps = max(self.eps_cov, X_pos.shape[1] / max(X_pos.shape[0], 1) * 1e-4)
            lam = self.lambda_
            if X_u.shape[0] == 0:
                warnings.warn(
                    "No unlabeled observations found (y=-1). "
                    "Falling back to supervised MLE (lambda=0).",
                    UserWarning, stacklevel=2,
                )
                lam = 0.0
            self.lambda_used_ = lam
            self.params_, self.n_iter_, self.converged_ = em_semisup(
                X_pos, X_neg, X_u,
                lam=lam, tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = np.array([0, 1])
        return self

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lam_grid: np.ndarray | None = None,
    ) -> "SemiSupervisedGMM":
        """
        Select lambda by grid search on validation log-likelihood, then refit.

        Parameters
        ----------
        X, y         : training data (y uses -1 for unlabeled)
        X_val, y_val : validation data (no unlabeled)
        lam_grid     : array of lambda values to try (default: logspace(-2, 2, 20))
        """
        K = _count_classes(y)
        y_val = np.asarray(y_val)

        if K > 2:
            Xs, X_u, classes = encode_labels_multi(X, y)
            Xs_val = [X_val[y_val == c] for c in classes]
            eps = max(self.eps_cov, Xs[0].shape[1] / max(Xs[0].shape[0], 1) * 1e-4)
            best_lam, best_params = grid_search_lambda_multi(
                Xs, X_u, classes, Xs_val,
                lam_grid=lam_grid, eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = classes
        else:
            X_pos, X_neg, X_u = encode_labels(X, y)
            X_pos_val = X_val[y_val == 1]
            X_neg_val = X_val[y_val == 0]
            eps = max(self.eps_cov, X_pos.shape[1] / max(X_pos.shape[0], 1) * 1e-4)
            best_lam, best_params = grid_search_lambda(
                X_pos, X_neg, X_u, X_pos_val, X_neg_val,
                lam_grid=lam_grid, eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = np.array([0, 1])

        self.lambda_used_ = best_lam
        self.params_ = best_params
        return self


class LearnedLambdaGMM(BaseGMM):
    """
    Semi-supervised GMM where lambda is learned by gradient ascent on
    validation log-likelihood.

    Parameters
    ----------
    lambda_init : float
        Initial value for gradient ascent.
    lr : float
        Learning rate (sign-normalised gradient step).
    n_steps : int
        Number of gradient ascent steps.
    h : float
        Finite-difference step size for gradient estimation.
    tol, max_iter, eps_cov, covariance_type : see SupervisedGMM.

    Attributes set after fit
    ------------------------
    lambda_ : float   — learned lambda value
    params_ : GMMParams
    """

    def __init__(
        self,
        lambda_init: float = 1.0,
        lr: float = 0.5,
        n_steps: int = 10,
        h: float = 0.1,
        tol: float = 5e-6,
        max_iter: int = 300,
        eps_cov: float = 1e-6,
        covariance_type: str = "full",
    ):
        self.lambda_init = lambda_init
        self.lr = lr
        self.n_steps = n_steps
        self.h = h
        self.tol = tol
        self.max_iter = max_iter
        self.eps_cov = eps_cov
        self.covariance_type = covariance_type

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LearnedLambdaGMM":
        """
        Fit the model, learning lambda from a validation set.

        Parameters
        ----------
        X, y         : training data (y uses -1 for unlabeled)
        X_val, y_val : validation data required for lambda learning.
                       If omitted, falls back to SemiSupervisedGMM with lambda_init.
        """
        K = _count_classes(y)

        if K > 2:
            Xs, X_u, classes = encode_labels_multi(X, y)
            eps = max(self.eps_cov, Xs[0].shape[1] / max(Xs[0].shape[0], 1) * 1e-4)
            if X_val is None or y_val is None:
                warnings.warn(
                    "X_val/y_val not provided; using lambda_init without learning.",
                    UserWarning, stacklevel=2,
                )
                lam = self.lambda_init
            else:
                y_val = np.asarray(y_val)
                Xs_val = [X_val[y_val == c] for c in classes]
                lam = gradient_lambda_multi(
                    Xs, X_u, classes, Xs_val,
                    lam_init=self.lambda_init, lr=self.lr,
                    n_steps=self.n_steps, h=self.h,
                    eps_cov=eps, covariance_type=self.covariance_type,
                )
            self.lambda_ = float(lam)
            self.params_, self.n_iter_, self.converged_ = em_semisup_multi(
                Xs, X_u, classes,
                lam=lam, tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = classes
        else:
            X_pos, X_neg, X_u = encode_labels(X, y)
            check_high_d(X_pos, X_neg, self.covariance_type)
            eps = max(self.eps_cov, X_pos.shape[1] / max(X_pos.shape[0], 1) * 1e-4)
            if X_val is None or y_val is None:
                warnings.warn(
                    "X_val/y_val not provided; using lambda_init without learning.",
                    UserWarning, stacklevel=2,
                )
                lam = self.lambda_init
            else:
                y_val = np.asarray(y_val)
                X_pos_val = X_val[y_val == 1]
                X_neg_val = X_val[y_val == 0]
                lam = gradient_lambda(
                    X_pos, X_neg, X_u, X_pos_val, X_neg_val,
                    lam_init=self.lambda_init, lr=self.lr,
                    n_steps=self.n_steps, h=self.h,
                    eps_cov=eps, covariance_type=self.covariance_type,
                )
            self.lambda_ = float(lam)
            self.params_, self.n_iter_, self.converged_ = em_semisup(
                X_pos, X_neg, X_u,
                lam=lam, tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = np.array([0, 1])
        return self

    def fit_semi(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LearnedLambdaGMM":
        X, y = make_semi_supervised(X_labeled, y_labeled, X_unlabeled)
        return self.fit(X, y, X_val=X_val, y_val=y_val)


class LocalLambdaGMM(BaseGMM):
    """
    Confidence-weighted semi-supervised GMM.

    Per-point weight: w_j = max(gamma_j, 1 - gamma_j)^alpha.
    Downweights ambiguous unlabeled points; upweights confident ones.

    Parameters
    ----------
    lambda_ : float
        Global unlabeled weight.
    alpha : float
        Confidence sharpening exponent.  alpha=0 → uniform weights.
    tol, max_iter, eps_cov, covariance_type : see SupervisedGMM.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        alpha: float = 1.0,
        tol: float = 5e-6,
        max_iter: int = 300,
        eps_cov: float = 1e-6,
        covariance_type: str = "full",
    ):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.eps_cov = eps_cov
        self.covariance_type = covariance_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocalLambdaGMM":
        K = _count_classes(y)
        if K > 2:
            Xs, X_u, classes = encode_labels_multi(X, y)
            eps = max(self.eps_cov, Xs[0].shape[1] / max(Xs[0].shape[0], 1) * 1e-4)
            lam = self.lambda_
            if X_u.shape[0] == 0:
                warnings.warn(
                    "No unlabeled observations found. Falling back to supervised MLE.",
                    UserWarning, stacklevel=2,
                )
                lam = 0.0
            self.lambda_used_ = lam
            self.params_, self.n_iter_, self.converged_ = em_conf_weighted_multi(
                Xs, X_u, classes,
                lam=lam, alpha=self.alpha,
                tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = classes
        else:
            X_pos, X_neg, X_u = encode_labels(X, y)
            check_high_d(X_pos, X_neg, self.covariance_type)
            eps = max(self.eps_cov, X_pos.shape[1] / max(X_pos.shape[0], 1) * 1e-4)
            lam = self.lambda_
            if X_u.shape[0] == 0:
                warnings.warn(
                    "No unlabeled observations found. Falling back to supervised MLE.",
                    UserWarning, stacklevel=2,
                )
                lam = 0.0
            self.lambda_used_ = lam
            self.params_, self.n_iter_, self.converged_ = em_conf_weighted(
                X_pos, X_neg, X_u,
                lam=lam, alpha=self.alpha,
                tol=self.tol, max_iter=self.max_iter,
                eps_cov=eps, covariance_type=self.covariance_type,
            )
            self.classes_ = np.array([0, 1])
        return self


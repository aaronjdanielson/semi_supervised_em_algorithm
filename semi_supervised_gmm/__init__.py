"""
semi_supervised_gmm
===================
Semi-supervised Gaussian mixture classifier with a weighted unlabeled likelihood.

Four estimators
---------------
SupervisedGMM       — purely supervised MLE (lambda=0)
SemiSupervisedGMM   — fixed global lambda; supports grid-search via fit_cv()
LearnedLambdaGMM    — lambda learned by gradient ascent on validation log-likelihood
LocalLambdaGMM      — per-point confidence-weighted lambda

All estimators follow the sklearn interface (fit/predict/predict_proba/score)
and are compatible with GridSearchCV.  Use y=-1 as the unlabeled sentinel,
matching sklearn's LabelPropagation convention.

Diagnostics
-----------
alignment_score()   — pre-fitting diagnostic: should unlabeled data be used?
alignment_A0()      — raw A(0) and ||g_0||

Example
-------
>>> from semi_supervised_gmm import SemiSupervisedGMM
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X_pos = rng.multivariate_normal([1, 0], np.eye(2), 30)
>>> X_neg = rng.multivariate_normal([-1, 0], np.eye(2), 30)
>>> X_u   = rng.multivariate_normal([0, 0], np.eye(2), 200)
>>> X = np.vstack([X_pos, X_neg, X_u])
>>> y = np.array([1]*30 + [0]*30 + [-1]*200)
>>> model = SemiSupervisedGMM(lambda_=1.0).fit(X, y)
>>> model.predict_proba(X_pos[:5])  # doctest: +SKIP
"""

from ._params import GMMParams, GMMParamsMulti
from ._estimators import (
    SupervisedGMM,
    SemiSupervisedGMM,
    LearnedLambdaGMM,
    LocalLambdaGMM,
)
from ._diagnostics import alignment_A0, alignment_score
from ._data import make_semi_supervised
from .exceptions import NotFittedError, ConvergenceWarning, InsufficientLabeledDataError

__all__ = [
    "GMMParams",
    "GMMParamsMulti",
    "SupervisedGMM",
    "SemiSupervisedGMM",
    "LearnedLambdaGMM",
    "LocalLambdaGMM",
    "alignment_A0",
    "alignment_score",
    "make_semi_supervised",
    "NotFittedError",
    "ConvergenceWarning",
    "InsufficientLabeledDataError",
]

__version__ = "0.1.1"

"""
BaseGMM: sklearn-compatible mixin layer.  No math here — only interface wiring.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .exceptions import NotFittedError
from ._params import GMMParams, GMMParamsMulti
from ._em import posterior, posterior_multi
from ._data import make_semi_supervised


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the ROC curve.  Pure NumPy — no sklearn required.

    Equivalent to sklearn.metrics.roc_auc_score for binary y_true in {0,1}.
    """
    pos = y_true == 1
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U statistic
    order = np.argsort(y_score)
    ranked_pos = np.where(pos[order])[0]  # 0-indexed ranks of positives
    U = ranked_pos.sum() - n_pos * (n_pos - 1) / 2
    return float(U / (n_pos * n_neg))


class BaseGMM(ClassifierMixin, BaseEstimator):
    """
    Abstract base for all GMM estimators.

    Subclasses must implement _fit_core(X_pos, X_neg, X_u) -> GMMParams.
    """

    _estimator_type = "classifier"

    # set by fit()
    params_: GMMParams | None = None
    n_iter_: int = 0
    converged_: bool = False
    classes_: np.ndarray = np.array([0, 1])

    def _check_is_fitted(self):
        if self.params_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        proba : (n, K)  — columns ordered by classes_
                          For K=2: [P(y=0|x), P(y=1|x)]
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if isinstance(self.params_, GMMParamsMulti):
            return posterior_multi(X, self.params_)
        p1 = posterior(X, self.params_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        threshold : float  — decision threshold on P(y=1|x), binary only
        """
        proba = self.predict_proba(X)
        if isinstance(self.params_, GMMParamsMulti):
            return self.classes_[np.argmax(proba, axis=1)]
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return a scalar performance metric on (X, y).

        Binary (K=2): AUROC.  Multiclass (K>2): accuracy.
        y must not contain the unlabeled sentinel (-1).
        """
        y = np.asarray(y)
        if isinstance(self.params_, GMMParamsMulti) and self.params_.K > 2:
            return float(np.mean(self.predict(X) == y))
        proba = self.predict_proba(X)[:, 1]
        return float(_auroc(y, proba))

    def save(self, path: str) -> None:
        """Persist the fitted estimator to disk (pickle)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseGMM":
        """Load a fitted estimator from disk."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def fit_semi(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
    ) -> "BaseGMM":
        """Convenience: accepts split data instead of stacked (X, y)."""
        X, y = make_semi_supervised(X_labeled, y_labeled, X_unlabeled)
        return self.fit(X, y)

    @classmethod
    def alignment_score(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        eps_cov: float = 1e-6,
        g0_norm_threshold: float = 15.0,
    ) -> dict:
        """
        Pre-fitting diagnostic: should unlabeled data be used?

        Parameters
        ----------
        X, y         : training data with y=-1 for unlabeled
        X_val, y_val : validation data (y_val in {0, 1})

        Returns
        -------
        dict with keys "A0", "g0_norm", "recommendation", "n_unlabeled"
        """
        from ._diagnostics import alignment_score
        return alignment_score(
            X, y, X_val, y_val,
            eps_cov=eps_cov,
            g0_norm_threshold=g0_norm_threshold,
        )

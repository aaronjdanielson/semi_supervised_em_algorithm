"""
Label encoding and input validation.

Convention: y = 1 (positive), y = 0 (negative), y = -1 (unlabeled sentinel).
This matches sklearn's LabelPropagation / LabelSpreading convention.
"""

from __future__ import annotations
import warnings
import numpy as np

from .exceptions import InsufficientLabeledDataError


_UNLABELED = -1


def encode_labels(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into (X_pos, X_neg, X_unlabeled) using y=-1 as unlabeled sentinel.

    Parameters
    ----------
    X : (n, d)
    y : (n,)  integer-like array with values in {0, 1, -1}

    Returns
    -------
    X_pos : (N1, d)
    X_neg : (N0, d)
    X_u   : (Nu, d)  — may be empty

    Raises
    ------
    ValueError if y contains values other than {-1, 0, 1}.
    InsufficientLabeledDataError if fewer than 2 samples in either labeled class.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    allowed = {-1, 0, 1}
    unique_y = set(np.unique(y).tolist())
    bad = unique_y - allowed
    if bad:
        raise ValueError(
            f"y contains unexpected values {bad}. "
            "Use 1 (positive), 0 (negative), -1 (unlabeled)."
        )

    mask_pos = y == 1
    mask_neg = y == 0
    mask_unl = y == _UNLABELED

    X_pos = X[mask_pos]
    X_neg = X[mask_neg]
    X_u   = X[mask_unl]

    if X_pos.shape[0] < 2:
        raise InsufficientLabeledDataError(
            f"Need at least 2 labeled positive samples, got {X_pos.shape[0]}."
        )
    if X_neg.shape[0] < 2:
        raise InsufficientLabeledDataError(
            f"Need at least 2 labeled negative samples, got {X_neg.shape[0]}."
        )

    return X_pos, X_neg, X_u


def encode_labels_multi(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Split (X, y) into per-class arrays and an unlabeled pool.

    Generalises encode_labels to K ≥ 2 classes.  y=-1 is the unlabeled sentinel.
    Class labels must be non-negative integers; at least 2 samples per class.

    Parameters
    ----------
    X : (n, d)
    y : (n,)  integer array; -1 marks unlabeled observations

    Returns
    -------
    Xs      : list of K arrays, each (N_k, d), ordered by sorted class label
    X_u     : (Nu, d)  — may be empty
    classes : (K,) ndarray of sorted class labels (integers ≥ 0)

    Raises
    ------
    ValueError if any non-unlabeled label is negative or non-integer.
    InsufficientLabeledDataError if any class has fewer than 2 samples.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    labeled_mask = y != _UNLABELED
    labeled_vals = y[labeled_mask]
    if np.any(labeled_vals < 0):
        raise ValueError(
            "Class labels must be non-negative integers. "
            "Use y=-1 exclusively for unlabeled observations."
        )

    classes = np.sort(np.unique(labeled_vals).astype(int))
    if len(classes) < 2:
        raise InsufficientLabeledDataError(
            f"Need at least 2 distinct classes, found {len(classes)}."
        )

    Xs = []
    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] < 2:
            raise InsufficientLabeledDataError(
                f"Need at least 2 labeled samples for class {c}, got {Xc.shape[0]}."
            )
        Xs.append(Xc)

    X_u = X[y == _UNLABELED]
    return Xs, X_u, classes


def check_high_d(X_pos: np.ndarray, X_neg: np.ndarray, covariance_type: str) -> None:
    """Warn if d > min(N1, N0) and covariance_type='full' (rank-deficient)."""
    d  = X_pos.shape[1]
    N1 = X_pos.shape[0]
    N0 = X_neg.shape[0]
    if d > min(N1, N0) and covariance_type == "full":
        warnings.warn(
            f"d={d} > min(N1={N1}, N0={N0}): full covariance is rank-deficient. "
            "Consider covariance_type='ledoit_wolf' or 'diag'.",
            UserWarning,
            stacklevel=4,
        )


def make_semi_supervised(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: stack labeled and unlabeled into (X, y) with -1 sentinel.
    Useful when data is already split rather than stacked.
    """
    X_unlabeled = np.asarray(X_unlabeled, dtype=float)
    y_unl = np.full(X_unlabeled.shape[0], _UNLABELED, dtype=int)
    X = np.vstack([X_labeled, X_unlabeled])
    y = np.concatenate([y_labeled, y_unl])
    return X, y

"""Unit tests for _data.py."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semi_supervised_gmm._data import encode_labels, make_semi_supervised
from semi_supervised_gmm.exceptions import InsufficientLabeledDataError


@pytest.fixture
def basic_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = np.array([1] * 30 + [0] * 30 + [-1] * 40)
    return X, y


class TestEncodeLabels:
    def test_splits_correctly(self, basic_data):
        X, y = basic_data
        X_pos, X_neg, X_u = encode_labels(X, y)
        assert X_pos.shape == (30, 3)
        assert X_neg.shape == (30, 3)
        assert X_u.shape  == (40, 3)

    def test_no_unlabeled(self):
        X = np.ones((10, 2))
        y = np.array([1] * 5 + [0] * 5)
        X_pos, X_neg, X_u = encode_labels(X, y)
        assert X_u.shape == (0, 2)

    def test_bad_label_raises(self):
        X = np.ones((6, 2))
        y = np.array([1, 1, 0, 0, -1, 2])
        with pytest.raises(ValueError, match="unexpected values"):
            encode_labels(X, y)

    def test_too_few_positives_raises(self):
        X = np.ones((5, 2))
        y = np.array([1, 0, 0, -1, -1])
        with pytest.raises(InsufficientLabeledDataError):
            encode_labels(X, y)

    def test_too_few_negatives_raises(self):
        X = np.ones((5, 2))
        y = np.array([1, 1, 1, 0, -1])
        with pytest.raises(InsufficientLabeledDataError):
            encode_labels(X, y)

    def test_float_x_preserved(self, basic_data):
        X, y = basic_data
        X_pos, _, _ = encode_labels(X.astype(np.float32), y)
        assert X_pos.dtype == float


class TestMakeSemiSupervised:
    def test_stacks_and_assigns_sentinel(self):
        rng = np.random.default_rng(1)
        X_lab = rng.standard_normal((20, 3))
        y_lab = np.array([1] * 10 + [0] * 10)
        X_unl = rng.standard_normal((50, 3))
        X, y = make_semi_supervised(X_lab, y_lab, X_unl)
        assert X.shape == (70, 3)
        assert (y[-50:] == -1).all()
        assert (y[:20] == y_lab).all()

    def test_roundtrip(self):
        rng = np.random.default_rng(2)
        X_lab = rng.standard_normal((20, 2))
        y_lab = np.array([1] * 10 + [0] * 10)
        X_unl = rng.standard_normal((30, 2))
        X, y = make_semi_supervised(X_lab, y_lab, X_unl)
        X_pos, X_neg, X_u = encode_labels(X, y)
        assert X_pos.shape[0] == 10
        assert X_neg.shape[0] == 10
        assert X_u.shape[0]   == 30

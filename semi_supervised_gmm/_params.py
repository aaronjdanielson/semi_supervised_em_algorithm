from dataclasses import dataclass
import numpy as np


@dataclass
class GMMParamsMulti:
    """
    Fitted parameters for a K-component Gaussian mixture (K ≥ 2).

    Attributes
    ----------
    pi : ndarray, shape (K,)
        Mixing weights.  Sums to 1.
    means : ndarray, shape (K, d)
        Class means.
    covariances : ndarray, shape (K, d, d)
        Class covariances.
    classes : ndarray, shape (K,)
        Integer class labels in sorted order.
    """

    pi: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    classes: np.ndarray

    def copy(self) -> "GMMParamsMulti":
        return GMMParamsMulti(
            pi=self.pi.copy(),
            means=self.means.copy(),
            covariances=self.covariances.copy(),
            classes=self.classes.copy(),
        )

    @property
    def d(self) -> int:
        return self.means.shape[1]

    @property
    def K(self) -> int:
        return self.means.shape[0]


@dataclass
class GMMParams:
    """
    Fitted parameters for a two-component Gaussian mixture.

    Attributes
    ----------
    pi : float
        Mixing weight for class 1 (positive class).  pi in (0, 1).
    mu0 : ndarray, shape (d,)
        Mean of class 0 (negative class).
    mu1 : ndarray, shape (d,)
        Mean of class 1 (positive class).
    Sigma0 : ndarray, shape (d, d)
        Covariance of class 0.
    Sigma1 : ndarray, shape (d, d)
        Covariance of class 1.
    """

    pi: float
    mu0: np.ndarray
    mu1: np.ndarray
    Sigma0: np.ndarray
    Sigma1: np.ndarray

    def copy(self) -> "GMMParams":
        return GMMParams(
            pi=float(self.pi),
            mu0=self.mu0.copy(),
            mu1=self.mu1.copy(),
            Sigma0=self.Sigma0.copy(),
            Sigma1=self.Sigma1.copy(),
        )

    @property
    def d(self) -> int:
        return self.mu1.shape[0]

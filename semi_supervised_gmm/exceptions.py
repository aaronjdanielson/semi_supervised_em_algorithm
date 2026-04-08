import warnings


class NotFittedError(ValueError, AttributeError):
    """Raised when predict/transform is called before fit."""


class ConvergenceWarning(UserWarning):
    """Issued when EM does not converge within max_iter."""


class InsufficientLabeledDataError(ValueError):
    """Raised when labeled set is too small to estimate parameters (N < d+2)."""


def warn_convergence(n_iter, max_iter):
    warnings.warn(
        f"EM did not converge after {max_iter} iterations (last delta at iter {n_iter}).",
        ConvergenceWarning,
        stacklevel=3,
    )

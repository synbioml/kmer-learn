"""Base classes for distance metrics and distance-based kernels."""

from __future__ import annotations

import abc
import warnings
from typing import Iterable, Optional, Union

import numpy as np


SeqLike = Union[str, bytes]


def _to_str(seq: SeqLike) -> str:
    if isinstance(seq, str):
        return seq
    if isinstance(seq, bytes):
        return seq.decode("ascii")
    raise TypeError(f"sequence must be str or bytes, got {type(seq).__name__}")


class BaseDistance(abc.ABC):
    """Abstract base class for pairwise sequence distances.

    Subclasses must implement ``_pair(a, b) -> float``. The default
    ``cdist`` and ``pdist`` loop in Python; subclasses with a fast
    vectorized backend should override them.

    Attributes
    ----------
    is_distance : bool
        True for proper metrics (smaller = more similar). False for
        similarity scores (larger = more similar), e.g. raw NW/SW scores.
    """

    is_distance: bool = True

    @abc.abstractmethod
    def _pair(self, a: str, b: str) -> float:
        ...

    def __call__(self, a: SeqLike, b: SeqLike) -> float:
        return self._pair(_to_str(a), _to_str(b))

    def cdist(self, X: Iterable[SeqLike], Y: Optional[Iterable[SeqLike]] = None) -> np.ndarray:
        """Pairwise distance matrix between X and Y (or X and X)."""
        Xs = [_to_str(s) for s in X]
        Ys = Xs if Y is None else [_to_str(s) for s in Y]
        n, m = len(Xs), len(Ys)
        out = np.empty((n, m), dtype=np.float64)
        for i, a in enumerate(Xs):
            for j, b in enumerate(Ys):
                out[i, j] = self._pair(a, b)
        return out

    def pdist(self, X: Iterable[SeqLike]) -> np.ndarray:
        """Condensed pairwise distances (scipy.spatial.distance.squareform compatible)."""
        Xs = [_to_str(s) for s in X]
        n = len(Xs)
        out = np.empty(n * (n - 1) // 2, dtype=np.float64)
        k = 0
        for i in range(n):
            ai = Xs[i]
            for j in range(i + 1, n):
                out[k] = self._pair(ai, Xs[j])
                k += 1
        return out


# Post-transform formulas. d is a distance (>= 0).
_TRANSFORMS = {
    "linear": lambda d, gamma, coef0, degree, radius: -d,  # similarity = -distance
    "rbf": lambda d, gamma, coef0, degree, radius: np.exp(-gamma * d * d),
    "poly": lambda d, gamma, coef0, degree, radius: (gamma * (-d) + coef0) ** degree,
    "sigmoid": lambda d, gamma, coef0, degree, radius: np.tanh(gamma * (-d) + coef0),
    "laplacian": lambda d, gamma, coef0, degree, radius: np.exp(-gamma * d),
    "exponential": lambda d, gamma, coef0, degree, radius: np.exp(-gamma * d),
    "exponential_clipped": lambda d, gamma, coef0, degree, radius: np.exp(-gamma * max(0.0, d - radius)),
    "cauchy": lambda d, gamma, coef0, degree, radius: 1.0 / (1.0 + gamma * d * d),
}


class DistanceKernel(BaseDistance):
    """Kernel derived from a distance via a post-transform.

    Wraps a :class:`BaseDistance` and applies a transform to convert
    distances to kernel values. The resulting Gram matrix is symmetric
    and PSD for the RBF/Laplacian/Cauchy transforms (since the
    underlying distance is a proper metric).

    Parameters
    ----------
    distance : BaseDistance
        The underlying distance metric.
    transform : {"linear", "rbf", "poly", "sigmoid", "laplacian",
                 "exponential", "exponential_clipped", "cauchy"}
        Post-transform applied to the distance.
        - "linear": K(a,b) = -d(a,b)  (similarity = -distance)
        - "rbf":    K(a,b) = exp(-gamma * d^2)
        - "poly":   K(a,b) = (gamma * (-d) + coef0) ^ degree
        - "sigmoid": K(a,b) = tanh(gamma * (-d) + coef0)
        - "laplacian" / "exponential": K(a,b) = exp(-gamma * d)
        - "exponential_clipped": K(a,b) = exp(-gamma * max(0, d - radius))
        - "cauchy": K(a,b) = 1 / (1 + gamma * d^2)
    gamma : float, default 1.0
        Transform scale parameter.
    coef0 : float, default 0.0
        Transform offset (poly, sigmoid).
    degree : int, default 3
        Transform degree (poly).
    radius : float, default 0.0
        Transform radius (exponential_clipped).

    Examples
    --------
    >>> from kmer.distance import Levenshtein, DistanceKernel
    >>> kern = DistanceKernel(Levenshtein(), transform="rbf", gamma=0.1)
    >>> K = kern.kernel_matrix(["ACGT", "ACGTA", "TTTT"])
    """

    is_distance = False  # a kernel is a similarity

    def __init__(
        self,
        distance: BaseDistance,
        transform: str = "rbf",
        *,
        gamma: float = 1.0,
        coef0: float = 0.0,
        degree: int = 3,
        radius: float = 0.0,
    ):
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"transform must be one of {list(_TRANSFORMS)}, got {transform!r}"
            )
        self.distance = distance
        self.transform = transform
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.radius = radius

    def _pair(self, a: str, b: str) -> float:
        d = self.distance._pair(a, b)
        return float(_TRANSFORMS[self.transform](
            d, self.gamma, self.coef0, self.degree, self.radius
        ))

    def kernel_matrix(self, X: Iterable[SeqLike],
                      Y: Optional[Iterable[SeqLike]] = None) -> np.ndarray:
        """Kernel matrix K[i, j] = transform(d(X[i], Y[j])).

        Returns a 2D numpy array of shape (n_X, n_Y) suitable for
        ``sklearn.svm.SVC(kernel='precomputed')``.
        """
        D = self.distance.cdist(X, Y)  # (n_X, n_Y) distance matrix
        return _TRANSFORMS[self.transform](
            D, self.gamma, self.coef0, self.degree, self.radius
        )


def _check_optional_backend(name: str, extra_msg: str = ""):
    """Warn if an optional backend is not installed."""
    import importlib.util
    if importlib.util.find_spec(name) is None:
        warnings.warn(
            f"{name!r} is not installed; falling back to a slower pure-Python "
            f"implementation. Install it with `pip install {name}` for "
            f"10-100x speedup. {extra_msg}",
            stacklevel=3,
        )
        return False
    return True

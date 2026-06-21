"""Gapped k-mer kernel for nucleotide sequences.

A C-backed implementation of the gkmSVM / LS-GKM gapped k-mer kernel
family (Ghandi et al. 2014; Lee 2016).

Class hierarchy
---------------
::

    BaseGKMKernel (ABC)
    ├── GKMKernel              — full-sequence, uniform weights
    ├── WGKMKernel             — full-sequence, positional weighting
    ├── WindowedGKMKernel      — windowed, uniform weights
    └── WindowedWGKMKernel     — windowed, positional weighting

Each concrete class wraps the appropriate C extension object (``Kernel``
or ``WeightedKernel``) and calls the appropriate evaluation methods.

Quick start
-----------
>>> from gapped_kmer_kernel import GKMKernel
>>> kern = GKMKernel(L=10, k=6, d=3, kernel_type="truncated")
>>> kern.set_references(["ACGTACGTACGTACGTACGT", "TTTTAAAAGGGGCCCCAAAA"])
>>> K = kern.kernel(X_query=["ACGTACGTACGTACGTACGT", "TTTTAAAAGGGGCCCCAAAA"])
>>> K.shape
(2, 2)

References
----------
* Ghandi M, Lee D, et al. PLoS Comput Biol 10, e1003711 (2014).
* Lee D. Bioinformatics 32, btw142 (2016).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence as Seq, Union

import numpy as np

from ._native._gkmkern import Kernel, WeightedKernel, Sequence

SeqLike = Union[str, bytes, Seq[Union[str, bytes]]]


# ------------------------------------------------------------------
# Module-level constants and helpers
# ------------------------------------------------------------------

_SCHEME_ID = {
    "full":           0,
    "truncated":      1,
    "estimated_full": 2,
}

_VALID_TRANSFORMS = (None, "linear", "rbf", "poly", "sigmoid",
                      "exponential", "exponential_clipped")

_POS_KERNEL_ID = {
    "triangular":   0,
    "epanechnikov": 1,
    "gaussian":     2,
    "laplacian":    3,
    "cauchy":       4,
}


def _to_str_list(seqs: SeqLike) -> List[str]:
    """Convert a string/bytes or iterable thereof to a list of strings.

    The C extension requires str input. Bytes are decoded as ASCII; both
    str and bytes may be freely mixed in the input iterable.
    """
    if isinstance(seqs, (str, bytes)):
        return [_to_str(seqs)]
    out: List[str] = []
    for s in seqs:
        out.append(_to_str(s))
    return out


def _to_str(s: Union[str, bytes]) -> str:
    """Coerce a single sequence to str (ASCII)."""
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):
        return s.decode("ascii")
    raise TypeError(
        f"sequence elements must be str or bytes, got {type(s).__name__}")


def _parse_padding(padding, window: Optional[int] = None,
                   L: Optional[int] = None) -> tuple:
    """Parse padding argument to (pad_left, pad_right).

    Padding may be negative (skips positions at sequence edges).
    If window and L are provided, validates that the effective window
    size (window - pad_left - pad_right) >= L.
    """
    if padding is None:
        return (0, 0)
    if isinstance(padding, int):
        return (padding, padding)
    if isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError(
                f"padding tuple/list must have 2 elements, got {len(padding)}")
        return (int(padding[0]), int(padding[1]))
    raise TypeError(
        f"padding must be int or (left, right) tuple, got {type(padding).__name__}")


def _validate_padding(pad: tuple, window: int, L: int):
    """Validate that padding doesn't make the effective window too small.

    For positive padding, the edge windows shrink by pad_left (left edge)
    and pad_right (right edge). Each must still be >= L.
    For negative padding, the effective window grows, so no upper bound."""
    pad_left, pad_right = pad
    if pad_left > 0 and window - pad_left < L:
        raise ValueError(
            f"pad_left={pad_left} too large for window={window}, "
            f"L={L}: left edge window size {window - pad_left} < {L}")
    if pad_right > 0 and window - pad_right < L:
        raise ValueError(
            f"pad_right={pad_right} too large for window={window}, "
            f"L={L}: right edge window size {window - pad_right} < {L}")


def _check_same_length(seqs: List[str], label: str = "sequences"):
    """Raise ValueError if sequences don't all have the same length."""
    if not seqs:
        raise ValueError(f"{label} list is empty")
    seqlen = len(seqs[0])
    if any(len(s) != seqlen for s in seqs):
        raise ValueError(f"all {label} must have the same length")


# ------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------

class BaseGKMKernel(ABC):
    """Abstract base class for all gkm kernels.

    Provides shared infrastructure: parameter validation, reference
    management, post-transform application, and the public ``kernel()``
    dispatch. Subclasses implement the C-call hooks.

    Parameters
    ----------
    L, k, d : int
        Kernel parameters: word length, non-gap positions, max mismatches.
    kernel_type : {"full", "estimated_full", "truncated"}
        Per-mismatch weight scheme.
    use_rc : bool, default False
        Whether to index the reverse-complement strand.
    transform : {None, "linear", "rbf", "poly", "sigmoid", "exponential", "exponential_clipped"}
        Post-transform applied after normalisation.
    transform_gamma, transform_coef0, transform_degree : float/int
        Transform parameters.
    """

    def __init__(
        self,
        L: int = 10,
        k: int = 6,
        d: int = 3,
        *,
        kernel_type: str = "full",
        use_rc: bool = False,
        transform: Optional[str] = None,
        transform_gamma: float = 1.0,
        transform_coef0: float = 0.0,
        transform_degree: int = 3,
        transform_radius: float = 0.0,
    ):
        if L < 2 or L > 12:
            raise ValueError(f"L must be in [2, 12], got {L}")
        if k < 1 or k > L:
            raise ValueError(f"k must be in [1, L], got k={k}, L={L}")
        if d < 0 or d > L - k:
            raise ValueError(f"d must be in [0, L-k], got d={d}, L-k={L-k}")
        if kernel_type not in _SCHEME_ID:
            raise ValueError(
                f"kernel_type must be one of {list(_SCHEME_ID)}, "
                f"got {kernel_type!r}")
        if transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}, "
                f"got {transform!r}")

        self.L = L
        self.k = k
        self.d = d
        self.kernel_type = kernel_type
        self.use_rc = bool(use_rc)
        self.transform = transform
        self.transform_gamma = transform_gamma
        self.transform_coef0 = transform_coef0
        self.transform_degree = transform_degree
        self.transform_radius = transform_radius

        self._ref_seqs: List[str] = []
        self._built = False

    # ------------------------------------------------------------------
    # Post-transform
    # ------------------------------------------------------------------

    def _apply_transform(self, K: np.ndarray, apply: bool = True) -> np.ndarray:
        if not apply or self.transform is None or self.transform == "linear":
            return K
        t, g = self.transform, self.transform_gamma
        if t == "rbf":
            # exp(-gamma * x) where x = 1 - K (distance)
            np.exp(g * (K - 1.0), out=K)
        elif t == "poly":
            # (gamma * K + coef0) ^ degree
            np.power(g * K + self.transform_coef0, self.transform_degree, out=K)
        elif t == "sigmoid":
            # tanh(gamma * K + coef0)
            np.tanh(g * K + self.transform_coef0, out=K)
        elif t == "exponential":
            # exp(-gamma * (x - radius)) where x = 1 - K (distance)
            x = 1.0 - K
            np.exp(-g * (x - self.transform_radius), out=K)
        elif t == "exponential_clipped":
            # exp(-gamma * max(0, x - radius)) where x = 1 - K (distance)
            x = 1.0 - K
            np.exp(-g * np.maximum(0.0, x - self.transform_radius), out=K)
        return K

    # ------------------------------------------------------------------
    # Reference set management
    # ------------------------------------------------------------------

    @abstractmethod
    def set_references(self, X: SeqLike) -> "BaseGKMKernel":
        """Set reference sequences and build the internal kmer tree."""
        ...

    def _ensure_built(self):
        if not self._built:
            raise RuntimeError(
                "no reference sequences set; call set_references() "
                "first, or pass X_ref= to this method")

    def _validate_ref_seqs(self, seqs: List[str]):
        if not seqs:
            raise ValueError("reference sequence list is empty")
        for s in seqs:
            if len(s) < self.L:
                raise ValueError(
                    f"sequence of length {len(s)} is shorter than L={self.L}")

    @property
    def n_references(self) -> int:
        return len(self._ref_seqs)

    @property
    def reference_sequences(self) -> List[str]:
        return list(self._ref_seqs)

    # ------------------------------------------------------------------
    # Kernel evaluation — public dispatch
    # ------------------------------------------------------------------

    def kernel(self, *, X_query=None, X_ref=None,
               apply_transform: bool = True):
        """Kernel matrix.

        Dispatches to self-kernel or cross-kernel based on X_query.

        - ``X_query=None``: self-kernel of references.
        - ``X_query=seqs``: cross-kernel (queries vs references).

        Returns 2D for full-sequence kernels, 3D for windowed kernels
        (and for weighted kernels with multiple offsets).
        """
        if X_ref is not None:
            self.set_references(X_ref)
        if X_query is not None:
            return self._cross_kernel(X_query, apply_transform)
        else:
            self._ensure_built()
            return self._self_kernel(self._ref_seqs, apply_transform)

    @abstractmethod
    def _self_kernel(self, seqs, apply_transform=True):
        """Hook: compute self-kernel. Override in subclass."""
        ...

    @abstractmethod
    def _cross_kernel(self, X_query, apply_transform=True):
        """Hook: compute cross-kernel. Override in subclass."""
        ...

    # ------------------------------------------------------------------
    # Sliding window query (available on non-windowed kernels only)
    # ------------------------------------------------------------------

    def sliding_window_query_kernel(self, X_query, *, window: int, shift: int,
                                    padding=None,
                                    apply_transform: bool = True):
        """Cross sliding-window kernel.

        Slides a window over each query sequence and evaluates K(window,
        ref_j) against the fixed reference tree.

        Returns a list of np.ndarrays (one per query), each of shape
        ``(n_windows_i, n_ref)``. Queries may have different lengths.
        """
        self._ensure_built()
        if window < self.L:
            raise ValueError(f"window must be >= L={self.L}, got {window}")
        if shift < 1:
            raise ValueError(f"shift must be >= 1, got {shift}")
        pad = _parse_padding(padding, window, self.L)
        _validate_padding(pad, window, self.L)
        seqs = _to_str_list(X_query)
        if not seqs:
            raise ValueError("X_query is empty")
        results = []
        for s in seqs:
            if len(s) < window:
                raise ValueError(
                    f"sequence length {len(s)} is shorter than window={window}")
            rows = self._sliding_query_c(s, window, shift, pad)
            K = np.asarray(rows, dtype=np.float64)
            results.append(self._apply_transform(K, apply_transform))
        return results

    @abstractmethod
    def _sliding_query_c(self, seq, window, shift, pad):
        """Hook: call the C sliding-query function. Override in subclass."""
        ...

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def _common_repr(self) -> str:
        rc = f", use_rc=True" if self.use_rc else ""
        t = f", transform={self.transform!r}" if self.transform else ""
        return (f"L={self.L}, k={self.k}, d={self.d}, "
                f"kernel_type={self.kernel_type!r}{rc}{t}, "
                f"n_refs={self.n_references}")


# ------------------------------------------------------------------
# Full-sequence regular kernel
# ------------------------------------------------------------------

class GKMKernel(BaseGKMKernel):
    """Full-sequence gapped k-mer kernel (uniform L-mer weights).

    Parameters
    ----------
    L, k, d : int
        Kernel parameters.
    kernel_type : {"full", "estimated_full", "truncated"}
    use_rc : bool, default False
    transform : {None, "linear", "rbf", "poly", "sigmoid", "exponential", "exponential_clipped"}
    transform_gamma, transform_coef0, transform_degree : float/int
    """

    def __init__(
        self,
        L: int = 10,
        k: int = 6,
        d: int = 3,
        *,
        kernel_type: str = "full",
        use_rc: bool = False,
        transform: Optional[str] = None,
        transform_gamma: float = 1.0,
        transform_coef0: float = 0.0,
        transform_degree: int = 3,
        transform_radius: float = 0.0,
    ):
        super().__init__(L=L, k=k, d=d, kernel_type=kernel_type, use_rc=use_rc,
                         transform=transform, transform_gamma=transform_gamma,
                         transform_coef0=transform_coef0,
                         transform_degree=transform_degree)

        scheme_id = _SCHEME_ID[kernel_type]
        nthreads = os.cpu_count() or 1
        self._kernel = Kernel(
            L=L, k=k, d=d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            nthreads=nthreads,
        )

    def set_references(self, X: SeqLike) -> "GKMKernel":
        seqs = _to_str_list(X)
        self._validate_ref_seqs(seqs)
        scheme_id = _SCHEME_ID[self.kernel_type]
        nthreads = os.cpu_count() or 1
        self._kernel = Kernel(
            L=self.L, k=self.k, d=self.d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            nthreads=nthreads,
        )
        self._ref_seqs = []
        for i, s in enumerate(seqs):
            self._kernel.add_sequence(s, f"seq{i}")
            self._ref_seqs.append(s)
        self._kernel.finalize()
        self._built = True
        return self

    # ------------------------------------------------------------------
    # C-call hooks
    # ------------------------------------------------------------------

    def _self_kernel(self, seqs, apply_transform=True):
        n = self._kernel.num_sequences()
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        rows = self._kernel.eval_batch(self._ref_seqs)
        K = np.asarray(rows, dtype=np.float64).reshape(n, n)
        return self._apply_transform(K, apply_transform)

    def _cross_kernel(self, X_query, apply_transform=True):
        self._ensure_built()
        seqs = _to_str_list(X_query)
        for s in seqs:
            if len(s) < self.L:
                raise ValueError(
                    f"query sequence of length {len(s)} is shorter than L={self.L}")
        n_q = len(seqs)
        n_r = self._kernel.num_sequences()
        if n_q == 0:
            return np.zeros((0, n_r), dtype=np.float64)
        rows = self._kernel.eval_batch(seqs)
        K = np.asarray(rows, dtype=np.float64).reshape(n_q, n_r)
        return self._apply_transform(K, apply_transform)

    def _sliding_query_c(self, seq, window, shift, pad):
        return self._kernel.sliding_query(seq, window, shift, pad)

    def __repr__(self):
        return f"GKMKernel({self._common_repr()})"


# ------------------------------------------------------------------
# Full-sequence weighted kernel
# ------------------------------------------------------------------

class WGKMKernel(BaseGKMKernel):
    """Full-sequence weighted gkm kernel (positional weighting).

    Parameters
    ----------
    L, k, d : int
        Kernel parameters.
    kernel_type : {"full", "estimated_full", "truncated"}
    use_rc : bool, default False
    weight_kernel : {"triangular", "epanechnikov", "gaussian", "laplacian", "cauchy"}
        Spatial kernel for positional weighting.
    weight_gamma : float
        Decay rate (laplacian only).
    weight_sigma : float
        Scale (triangular, epanechnikov, gaussian, cauchy).
    weight_peak : float
        Peak positional weight at the center.
    center_offset : float
        Default offset from midpoint (scalar; negative = left, positive = right).
    transform : {None, "linear", "rbf", "poly", "sigmoid", "exponential", "exponential_clipped"}
    transform_gamma, transform_coef0, transform_degree : float/int
    """

    def __init__(
        self,
        L: int = 10,
        k: int = 6,
        d: int = 3,
        *,
        kernel_type: str = "full",
        use_rc: bool = False,
        weight_kernel: str = "triangular",
        weight_gamma: float = 1.0,
        weight_sigma: float = 50.0,
        weight_peak: float = 50.0,
        center_offset: float = 0.0,
        transform: Optional[str] = None,
        transform_gamma: float = 1.0,
        transform_coef0: float = 0.0,
        transform_degree: int = 3,
        transform_radius: float = 0.0,
    ):
        super().__init__(L=L, k=k, d=d, kernel_type=kernel_type, use_rc=use_rc,
                         transform=transform, transform_gamma=transform_gamma,
                         transform_coef0=transform_coef0,
                         transform_degree=transform_degree)

        if weight_kernel not in _POS_KERNEL_ID:
            raise ValueError(
                f"weight_kernel must be one of {list(_POS_KERNEL_ID)}, "
                f"got {weight_kernel!r}")
        if weight_peak <= 0:
            raise ValueError(f"weight_peak must be > 0, got {weight_peak}")

        self.weight_kernel = weight_kernel
        self.weight_gamma = weight_gamma
        self.weight_sigma = weight_sigma
        self.weight_peak = weight_peak
        self.center_offset = center_offset

        scheme_id = _SCHEME_ID[kernel_type]
        pos_id = _POS_KERNEL_ID[weight_kernel]
        nthreads = os.cpu_count() or 1
        self._kernel = WeightedKernel(
            L=L, k=k, d=d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            pos_kernel=pos_id,
            weight_gamma=weight_gamma,
            weight_sigma=weight_sigma,
            weight_peak=weight_peak,
            nthreads=nthreads,
        )

    def set_references(self, X: SeqLike) -> "WGKMKernel":
        seqs = _to_str_list(X)
        self._validate_ref_seqs(seqs)
        scheme_id = _SCHEME_ID[self.kernel_type]
        pos_id = _POS_KERNEL_ID[self.weight_kernel]
        nthreads = os.cpu_count() or 1
        self._kernel = WeightedKernel(
            L=self.L, k=self.k, d=self.d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            pos_kernel=pos_id,
            weight_gamma=self.weight_gamma,
            weight_sigma=self.weight_sigma,
            weight_peak=self.weight_peak,
            nthreads=nthreads,
        )
        self._ref_seqs = []
        for i, s in enumerate(seqs):
            self._kernel.add_sequence(s, f"seq{i}")
            self._ref_seqs.append(s)
        self._kernel.finalize()
        self._built = True
        return self

    # ------------------------------------------------------------------
    # Offset resolution
    # ------------------------------------------------------------------

    def _resolve_offsets(self, offsets, seqlen):
        """Resolve offsets to (centers_list, is_scalar)."""
        nkmers = seqlen - self.L + 1
        midpoint = (seqlen - 1) / 2.0

        if offsets is None:
            return [midpoint + self.center_offset], True

        if isinstance(offsets, (int, float)):
            return [midpoint + offsets], True

        if isinstance(offsets, slice):
            indices = list(range(*offsets.indices(nkmers)))
            centers = [j + (self.L - 1) / 2.0 for j in indices]
            return centers, False

        if isinstance(offsets, (list, tuple, np.ndarray)):
            if len(offsets) == 0:
                raise ValueError("offsets list is empty")
            centers = [midpoint + float(o) for o in offsets]
            return centers, False

        raise TypeError(
            f"offsets must be None, int, float, slice, or list/tuple of int/float, "
            f"got {type(offsets).__name__}")

    # ------------------------------------------------------------------
    # C-call hooks
    # ------------------------------------------------------------------

    def kernel(self, *, X_query=None, X_ref=None, offsets=None,
               apply_transform: bool = True):
        """Weighted kernel matrix.

        Parameters
        ----------
        X_query : seq(s) or None
        X_ref : seq(s) or None
        offsets : None | int | float | list | slice
            Offset(s) from midpoint. Scalar → 2D, list/slice → 3D.
        apply_transform : bool
        """
        if X_ref is not None:
            self.set_references(X_ref)
        if X_query is not None:
            return self._cross_kernel(X_query, offsets, apply_transform)
        else:
            self._ensure_built()
            return self._self_kernel(self._ref_seqs, offsets, apply_transform)

    def _self_kernel(self, seqs, offsets=None, apply_transform=True):
        seq_list = _to_str_list(seqs) if not isinstance(seqs, list) else seqs
        _check_same_length(seq_list, "sequences")
        seqlen = len(seq_list[0])
        if seqlen < self.L:
            raise ValueError(
                f"sequence length {seqlen} is shorter than L={self.L}")

        if offsets is None:
            offsets = self.center_offset
        result_list = self._kernel.self_eval(seq_list, offsets)
        K = np.asarray(result_list, dtype=np.float64)
        return self._apply_transform(K, apply_transform)

    def _cross_kernel(self, X_query, offsets=None, apply_transform=True):
        self._ensure_built()
        query_seqs = _to_str_list(X_query)
        if not query_seqs:
            raise ValueError("X_query is empty")
        _check_same_length(query_seqs, "query sequences")
        _check_same_length(self._ref_seqs, "reference sequences")
        if len(query_seqs[0]) != len(self._ref_seqs[0]):
            raise ValueError(
                f"query length ({len(query_seqs[0])}) must match "
                f"reference length ({len(self._ref_seqs[0])})")

        if offsets is None:
            offsets = self.center_offset
        result_list = self._kernel.cross_eval(
            query_seqs, self._ref_seqs, offsets)
        K = np.asarray(result_list, dtype=np.float64)
        return self._apply_transform(K, apply_transform)

    def _sliding_query_c(self, seq, window, shift, pad):
        raise NotImplementedError(
            "sliding_window_query_kernel for weighted kernels is not yet "
            "supported (the C implementation has a bug with different-length "
            "queries and refs). Use a non-weighted GKMKernel for sliding "
            "window queries, or use WGKMKernel.kernel() with offsets for "
            "full-sequence weighted evaluation.")

    def __repr__(self):
        wk = (f", weight_kernel={self.weight_kernel!r}"
              f", weight_gamma={self.weight_gamma}"
              f", weight_sigma={self.weight_sigma}"
              f", weight_peak={self.weight_peak}")
        co = f", center_offset={self.center_offset}" if self.center_offset != 0.0 else ""
        return f"WGKMKernel({self._common_repr()}{wk}{co})"


# ------------------------------------------------------------------
# Windowed kernels
# ------------------------------------------------------------------

class _BaseWindowedKernel(BaseGKMKernel):
    """Mixin for windowed kernels. Adds window/shift/padding parameters.

    Overrides kernel() to use windowed evaluation. The actual C calls
    are implemented by the concrete subclasses (WindowedGKMKernel,
    WindowedWGKMKernel).
    """

    def __init__(
        self,
        *args,
        window: int,
        shift: int,
        padding=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if window < self.L:
            raise ValueError(f"window must be >= L={self.L}, got {window}")
        if shift < 1:
            raise ValueError(f"shift must be >= 1, got {shift}")
        self.window = window
        self.shift = shift
        self.padding = _parse_padding(padding, window, self.L)
        _validate_padding(self.padding, window, self.L)

    def kernel(self, *, X_query=None, X_ref=None,
               apply_transform: bool = True):
        """Windowed kernel (3D tensor).

        Uses the constructor's window/shift/padding.

        - ``X_query=None``: self → ``(n_windows, n, n)``
        - ``X_query=seqs``: cross → ``(n_windows, n_query, n_ref)``

        All sequences must be the same length.
        """
        if X_ref is not None:
            self.set_references(X_ref)
        if X_query is not None:
            return self._cross_kernel(X_query, apply_transform)
        else:
            self._ensure_built()
            return self._self_kernel(self._ref_seqs, apply_transform)

    @abstractmethod
    def _self_window_c(self, seqs, window, shift, pad):
        """Hook: call the C self-window-eval function."""
        ...

    @abstractmethod
    def _cross_window_c(self, queries, refs, window, shift, pad):
        """Hook: call the C cross-window-eval function."""
        ...

    def _self_kernel(self, seqs, apply_transform=True):
        seq_list = _to_str_list(seqs) if not isinstance(seqs, list) else seqs
        _check_same_length(seq_list, "sequences")
        result_list = self._self_window_c(
            seq_list, self.window, self.shift, self.padding)
        K = np.asarray(result_list, dtype=np.float64)
        return self._apply_transform(K, apply_transform)

    def _cross_kernel(self, X_query, apply_transform=True):
        self._ensure_built()
        query_seqs = _to_str_list(X_query)
        if not query_seqs:
            raise ValueError("X_query is empty")
        _check_same_length(query_seqs, "query sequences")
        _check_same_length(self._ref_seqs, "reference sequences")
        if len(query_seqs[0]) != len(self._ref_seqs[0]):
            raise ValueError(
                f"query length ({len(query_seqs[0])}) must match "
                f"reference length ({len(self._ref_seqs[0])})")
        result_list = self._cross_window_c(
            query_seqs, self._ref_seqs, self.window, self.shift, self.padding)
        K = np.asarray(result_list, dtype=np.float64)
        return self._apply_transform(K, apply_transform)

    def _sliding_query_c(self, seq, window, shift, pad):
        raise NotImplementedError(
            "sliding_window_query_kernel is not available on windowed kernels; "
            "use a non-windowed kernel (GKMKernel or WGKMKernel) instead")

    def _windowed_repr(self) -> str:
        return (f", window={self.window}, shift={self.shift}"
                f", padding={self.padding}")


class WindowedGKMKernel(_BaseWindowedKernel):
    """Windowed gkm kernel (uniform L-mer weights, per-window evaluation).

    Parameters
    ----------
    L, k, d : int
    kernel_type : {"full", "estimated_full", "truncated"}
    use_rc : bool, default False
    window : int (required)
        Window size in nucleotides.
    shift : int (required)
        Step size between windows.
    padding : int or (int, int), default 0
        Padding for windows (may be negative to skip edge positions).
    transform : {None, "linear", "rbf", "poly", "sigmoid", "exponential", "exponential_clipped"}
    transform_gamma, transform_coef0, transform_degree : float/int
    """

    def __init__(
        self,
        L: int = 10,
        k: int = 6,
        d: int = 3,
        *,
        kernel_type: str = "full",
        use_rc: bool = False,
        window: int,
        shift: int,
        padding=0,
        transform: Optional[str] = None,
        transform_gamma: float = 1.0,
        transform_coef0: float = 0.0,
        transform_degree: int = 3,
        transform_radius: float = 0.0,
    ):
        transform_radius: float = 0.0,
        super().__init__(L=L, k=k, d=d, kernel_type=kernel_type, use_rc=use_rc,
                         transform=transform, transform_gamma=transform_gamma,
                         transform_coef0=transform_coef0,
                         transform_degree=transform_degree,
                         transform_radius=transform_radius,
                         window=window, shift=shift, padding=padding)

        scheme_id = _SCHEME_ID[kernel_type]
        nthreads = os.cpu_count() or 1
        self._kernel = Kernel(
            L=L, k=k, d=d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            nthreads=nthreads,
        )

    def set_references(self, X: SeqLike) -> "WindowedGKMKernel":
        seqs = _to_str_list(X)
        self._validate_ref_seqs(seqs)
        scheme_id = _SCHEME_ID[self.kernel_type]
        nthreads = os.cpu_count() or 1
        self._kernel = Kernel(
            L=self.L, k=self.k, d=self.d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            nthreads=nthreads,
        )
        self._ref_seqs = []
        for i, s in enumerate(seqs):
            self._kernel.add_sequence(s, f"seq{i}")
            self._ref_seqs.append(s)
        self._kernel.finalize()
        self._built = True
        return self

    def _self_window_c(self, seqs, window, shift, pad):
        return self._kernel.self_window_eval(seqs, window, shift, pad)

    def _cross_window_c(self, queries, refs, window, shift, pad):
        return self._kernel.cross_window_eval(queries, refs, window, shift, pad)

    def __repr__(self):
        return f"WindowedGKMKernel({self._common_repr()}{self._windowed_repr()})"


class WindowedWGKMKernel(_BaseWindowedKernel):
    """Windowed weighted gkm kernel (positional weighting, per-window evaluation).

    Uses window-midpoint centering. The constructor's ``center_offset``
    is ignored (windowed evaluation always centers on the window midpoint).

    Parameters
    ----------
    L, k, d : int
    kernel_type : {"full", "estimated_full", "truncated"}
    use_rc : bool, default False
    weight_kernel : {"triangular", "epanechnikov", "gaussian", "laplacian", "cauchy"}
    weight_gamma, weight_sigma, weight_peak : float
    window : int (required)
    shift : int (required)
    padding : int or (int, int), default 0
    transform, transform_gamma, transform_coef0, transform_degree
    """

    def __init__(
        self,
        L: int = 10,
        k: int = 6,
        d: int = 3,
        *,
        kernel_type: str = "full",
        use_rc: bool = False,
        weight_kernel: str = "triangular",
        weight_gamma: float = 1.0,
        weight_sigma: float = 50.0,
        weight_peak: float = 50.0,
        window: int,
        shift: int,
        padding=0,
        transform: Optional[str] = None,
        transform_gamma: float = 1.0,
        transform_coef0: float = 0.0,
        transform_degree: int = 3,
        transform_radius: float = 0.0,
    ):
        super().__init__(L=L, k=k, d=d, kernel_type=kernel_type, use_rc=use_rc,
                         transform=transform, transform_gamma=transform_gamma,
                         transform_coef0=transform_coef0,
                         transform_degree=transform_degree,
                         transform_radius=transform_radius,
                         window=window, shift=shift, padding=padding)

        if weight_kernel not in _POS_KERNEL_ID:
            raise ValueError(
                f"weight_kernel must be one of {list(_POS_KERNEL_ID)}, "
                f"got {weight_kernel!r}")
        if weight_peak <= 0:
            raise ValueError(f"weight_peak must be > 0, got {weight_peak}")

        self.weight_kernel = weight_kernel
        self.weight_gamma = weight_gamma
        self.weight_sigma = weight_sigma
        self.weight_peak = weight_peak

        scheme_id = _SCHEME_ID[kernel_type]
        pos_id = _POS_KERNEL_ID[weight_kernel]
        nthreads = os.cpu_count() or 1
        self._kernel = WeightedKernel(
            L=L, k=k, d=d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            pos_kernel=pos_id,
            weight_gamma=weight_gamma,
            weight_sigma=weight_sigma,
            weight_peak=weight_peak,
            nthreads=nthreads,
        )

    def set_references(self, X: SeqLike) -> "WindowedWGKMKernel":
        seqs = _to_str_list(X)
        self._validate_ref_seqs(seqs)
        scheme_id = _SCHEME_ID[self.kernel_type]
        pos_id = _POS_KERNEL_ID[self.weight_kernel]
        nthreads = os.cpu_count() or 1
        self._kernel = WeightedKernel(
            L=self.L, k=self.k, d=self.d,
            weight_scheme=scheme_id,
            use_rc=1 if self.use_rc else 0,
            pos_kernel=pos_id,
            weight_gamma=self.weight_gamma,
            weight_sigma=self.weight_sigma,
            weight_peak=self.weight_peak,
            nthreads=nthreads,
        )
        self._ref_seqs = []
        for i, s in enumerate(seqs):
            self._kernel.add_sequence(s, f"seq{i}")
            self._ref_seqs.append(s)
        self._kernel.finalize()
        self._built = True
        return self

    def _self_window_c(self, seqs, window, shift, pad):
        return self._kernel.self_window_eval(seqs, window, shift, pad)

    def _cross_window_c(self, queries, refs, window, shift, pad):
        raise NotImplementedError(
            "cross windowed evaluation for weighted kernel is not yet supported; "
            "use self windowed evaluation or the non-windowed WGKMKernel with offsets")

    def __repr__(self):
        wk = (f", weight_kernel={self.weight_kernel!r}"
              f", weight_gamma={self.weight_gamma}"
              f", weight_sigma={self.weight_sigma}"
              f", weight_peak={self.weight_peak}")
        return (f"WindowedWGKMKernel({self._common_repr()}{wk}"
                f"{self._windowed_repr()})")

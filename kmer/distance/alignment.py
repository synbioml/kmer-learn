"""Alignment-based metrics: Needleman-Wunsch and Smith-Waterman.

Both prefer the `parasail` SIMD backend when available; otherwise fall
back to pure-Python DP implementations. Both expose their raw alignment
score (a similarity, not a distance) — wrap with :class:`DistanceKernel`
to convert to a kernel, or use the ``as_distance=True`` flag at
construction to get ``max_score - score(a, b)`` as a distance.
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Union

import numpy as np

from ._base import BaseDistance, _check_optional_backend

_HAS_PARASAIL = importlib.util.find_spec("parasail") is not None
if _HAS_PARASAIL:
    import parasail
else:
    _check_optional_backend("parasail")


# Standard substitution matrices loaded lazily.
def _load_substitution_matrix(name: str) -> np.ndarray:
    """Load a substitution matrix by name.

    Supported: "NUC4.4" (DNA, built-in), "BLOSUM62", "BLOSUM45",
    "BLOSUM80", "PAM30", "PAM70", "PAM250" (require parasail).
    """
    if name.upper() in ("NUC4.4", "NUC44"):
        return _NUC44
    if _HAS_PARASAIL:
        # parasail exposes matrices as direct attributes (lowercase):
        # parasail.blosum62, parasail.pam250, parasail.nuc44, etc.
        attr = name.lower().replace(".", "")
        m = getattr(parasail, attr, None)
        if m is not None:
            n = m.size
            return np.array([[m.s[i][j] for j in range(n)] for i in range(n)],
                            dtype=np.int32)
    raise ValueError(
        f"unknown substitution matrix {name!r}; install parasail for "
        f"BLOSUM/PAM matrices"
    )


# NUC4.4 — standard NCBI DNA scoring matrix.
# Rows/cols: A, C, G, T.
_NUC44 = np.array([
    [ 5, -4, -4, -4],
    [-4,  5, -4, -4],
    [-4, -4,  5, -4],
    [-4, -4, -4,  5],
], dtype=np.int32)


class _AlignmentBase(BaseDistance):
    """Common base for NW/SW alignment-based scores."""

    is_distance = False  # raw score is a similarity

    def __init__(
        self,
        match: int = 1,
        mismatch: int = -1,
        gap: int = -1,
        substitution_matrix: Optional[Union[str, np.ndarray]] = None,
        as_distance: bool = False,
    ):
        """
        Parameters
        ----------
        match, mismatch : int
            Diagonal scores when no substitution_matrix is given.
            Ignored if substitution_matrix is set.
        gap : int
            Linear gap penalty.
        substitution_matrix : str or np.ndarray or None
            If str, name of a known matrix ("NUC4.4", "BLOSUM62", etc.).
            If np.ndarray, a 2D score matrix indexed by base code (for
            DNA: A=0, C=1, G=2, T=3).
            If None, use match/mismatch on the diagonal.
        as_distance : bool, default False
            If True, return max_possible_score - actual_score as a
            distance (smaller = more similar).
        """
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.as_distance = as_distance
        self._sub_matrix_name = None
        if substitution_matrix is not None:
            if isinstance(substitution_matrix, str):
                self._sub_matrix_name = substitution_matrix
                self.sub_matrix = _load_substitution_matrix(substitution_matrix)
            else:
                self.sub_matrix = np.asarray(substitution_matrix, dtype=np.int32)
        else:
            self.sub_matrix = None
        # Cached parasail matrix (built lazily on first use).
        self._parasail_matrix = None

    def _get_parasail_matrix(self):
        """Build or return a cached parasail matrix."""
        if self._parasail_matrix is not None:
            return self._parasail_matrix
        if self._sub_matrix_name is not None:
            # Look up by name: parasail.blosum62, parasail.nuc44, etc.
            attr = self._sub_matrix_name.lower().replace(".", "")
            if attr in ("nuc44",):
                self._parasail_matrix = parasail.nuc44
            else:
                m = getattr(parasail, attr, None)
                if m is None:
                    raise ValueError(
                        f"unknown parasail matrix {self._sub_matrix_name!r}"
                    )
                self._parasail_matrix = m
        else:
            # Build a simple match/mismatch matrix for ACGT.
            self._parasail_matrix = parasail.matrix_create(
                "ACGT", self.match, self.mismatch
            )
        return self._parasail_matrix

    def _base_code(self, c: str) -> int:
        """Map ACGT to 0..3 for indexing into the substitution matrix."""
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(c)
        if idx is None:
            raise ValueError(
                f"substitution matrix requires ACGT characters; got {c!r}"
            )
        return idx

    def _sub_score(self, a: str, b: str) -> int:
        if self.sub_matrix is not None:
            return int(self.sub_matrix[self._base_code(a), self._base_code(b)])
        return self.match if a == b else self.mismatch

    def _max_score_per_pos(self) -> int:
        """Maximum possible per-position score."""
        if self.sub_matrix is not None:
            return int(self.sub_matrix.max())
        return self.match


class NeedlemanWunsch(_AlignmentBase):
    """Needleman-Wunsch global alignment score.

    Returns the alignment score (higher = more similar). With
    ``as_distance=True``, returns ``max(len(a), len(b)) * match - score``
    as a distance.

    Examples
    --------
    >>> nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
    >>> nw("ACGT", "ACGT")
    4.0
    >>> nw("ACGT", "ACG")
    2.0
    """

    def _pair(self, a: str, b: str) -> float:
        if _HAS_PARASAIL:
            score = _nw_score_parasail(a, b, self)
        else:
            score = _nw_score(a, b, self)
        if self.as_distance:
            max_score = max(len(a), len(b)) * self._max_score_per_pos()
            return float(max_score - score)
        return float(score)


class SmithWaterman(_AlignmentBase):
    """Smith-Waterman local alignment score.

    Returns the best local alignment score (higher = more similar).
    With ``as_distance=True``, returns ``max_score - score`` as a
    pseudo-distance.

    Examples
    --------
    >>> sw = SmithWaterman(match=1, mismatch=-1, gap=-1)
    >>> sw("ACGT", "TTACGTTT")
    4.0
    """

    def _pair(self, a: str, b: str) -> float:
        if _HAS_PARASAIL:
            score = _sw_score_parasail(a, b, self)
        else:
            score = _sw_score(a, b, self)
        if self.as_distance:
            max_score = max(len(a), len(b)) * self._max_score_per_pos()
            return float(max_score - score)
        return float(score)


# ======================================================================
# Pure-Python DP implementations
# ======================================================================

def _nw_score(a: str, b: str, params: _AlignmentBase) -> int:
    """Pure-Python Needleman-Wunsch score."""
    n, m = len(a), len(b)
    if n == 0:
        return m * params.gap
    if m == 0:
        return n * params.gap
    prev = [j * params.gap for j in range(m + 1)]
    cur = [0] * (m + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i * params.gap
        for j, cb in enumerate(b, 1):
            sub = params._sub_score(ca, cb)
            cur[j] = max(
                prev[j] + params.gap,
                cur[j - 1] + params.gap,
                prev[j - 1] + sub,
            )
        prev, cur = cur, prev
    return prev[-1]


def _sw_score(a: str, b: str, params: _AlignmentBase) -> int:
    """Pure-Python Smith-Waterman score."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    best = 0
    for i, ca in enumerate(a, 1):
        cur[0] = 0
        for j, cb in enumerate(b, 1):
            sub = params._sub_score(ca, cb)
            v = max(
                0,
                prev[j] + params.gap,
                cur[j - 1] + params.gap,
                prev[j - 1] + sub,
            )
            cur[j] = v
            if v > best:
                best = v
        prev, cur = cur, prev
    return best


# ======================================================================
# parasail backends
# ======================================================================

def _nw_score_parasail(a: str, b: str, params: _AlignmentBase) -> int:
    """Needleman-Wunsch score via parasail (SIMD-accelerated).

    parasail's gap convention is the opposite of the standard: it
    expects a non-negative gap penalty (subtracted internally). We
    negate the user's gap (which is conventionally negative).
    """
    matrix = params._get_parasail_matrix()
    gap_penalty = -params.gap if params.gap < 0 else params.gap
    result = parasail.nw_scan(a, b, gap_penalty, gap_penalty, matrix)
    return result.score


def _sw_score_parasail(a: str, b: str, params: _AlignmentBase) -> int:
    """Smith-Waterman score via parasail."""
    matrix = params._get_parasail_matrix()
    gap_penalty = -params.gap if params.gap < 0 else params.gap
    result = parasail.sw_scan(a, b, gap_penalty, gap_penalty, matrix)
    return result.score

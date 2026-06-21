"""Edit-distance metrics: Hamming and Levenshtein.

Both prefer the `rapidfuzz` C++ backend when available; otherwise fall
back to pure-Python implementations.
"""

from __future__ import annotations

import importlib.util

from ._base import BaseDistance, _check_optional_backend

_HAS_RAPIDFUZZ = importlib.util.find_spec("rapidfuzz") is not None
if _HAS_RAPIDFUZZ:
    from rapidfuzz.distance import Hamming as _RF_Hamming
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein
else:
    _check_optional_backend("rapidfuzz")


class Hamming(BaseDistance):
    """Hamming distance (equal-length sequences only).

    Counts positions where a and b differ. Requires len(a) == len(b).

    Examples
    --------
    >>> Hamming()("ACGT", "ACGA")
    1
    >>> Hamming()("ACGT", "ACGT")
    0
    """

    def _pair(self, a: str, b: str) -> float:
        if len(a) != len(b):
            raise ValueError(
                f"Hamming requires equal-length sequences; got {len(a)} vs {len(b)}"
            )
        if _HAS_RAPIDFUZZ:
            return float(_RF_Hamming.distance(a, b))
        return float(sum(1 for x, y in zip(a, b) if x != y))


class Levenshtein(BaseDistance):
    """Levenshtein edit distance (insertions, deletions, substitutions).

    Examples
    --------
    >>> Levenshtein()("ACGT", "ACGTA")
    1.0
    >>> Levenshtein()("ACGT", "ACGT")
    0.0
    >>> Levenshtein()("ACGT", "TGCA")
    4.0
    """

    def _pair(self, a: str, b: str) -> float:
        if _HAS_RAPIDFUZZ:
            return float(_RF_Levenshtein.distance(a, b))
        return float(_levenshtein_python(a, b))

    def cdist(self, X, Y=None):
        if _HAS_RAPIDFUZZ:
            from ._base import _to_str
            Xs = [_to_str(s) for s in X]
            Ys = Xs if Y is None else [_to_str(s) for s in Y]
            import numpy as np
            n, m = len(Xs), len(Ys)
            out = np.empty((n, m), dtype=np.float64)
            for i, a in enumerate(Xs):
                for j, b in enumerate(Ys):
                    out[i, j] = _RF_Levenshtein.distance(a, b)
            return out
        return super().cdist(X, Y)


def _levenshtein_python(a: str, b: str) -> int:
    """Pure-Python Levenshtein via the standard DP."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP for memory efficiency.
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, cur = cur, prev
    return prev[-1]

"""kmer.encoders.gappy — masked-hash gappy k-mer encoder.

Counts gappy k-mers: patterns like ``*--*`` (match ACGT at positions 0
and 3, don't-care at 1, 2). Returns a scipy.sparse.csr_matrix.

Two construction modes:

1. Explicit masks: ``GappyEncoder(masks=["*--*", "*---*"])``
2. Gap range:      ``GappyEncoder(L=6, g_min=1, g_max=2)`` generates
   all C(L-2, g) masks for g in [g_min, g_max] (no leading/trailing gaps).

With ``canonical_rc=True``:
- Masks are canonicalized (RC-equivalent masks are deduplicated).
- For symmetric masks (mask == RC(mask)), gappy k-mer RC pairs are collapsed.
- For asymmetric masks, no gappy k-mer collapsing (mask canonicalization
  already handles the RC issue).
"""

from __future__ import annotations

from typing import Iterable, Optional, Union
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from ._native._gappy import _count_gappy_kmers

SeqLike = Union[str, bytes, Iterable[Union[str, bytes]]]


def generate_masks(L_min: int, L_max: int,
                   g_min: int = 0, g_max: Optional[int] = None) -> list[str]:
    """Generate all valid gappy k-mer masks for a range of lengths and gaps.

    Masks have concrete positions at 0 and L-1 (no leading/trailing gaps).
    Gaps are chosen from interior positions 1..L-2.
    Number of masks with g gaps = C(L-2, g).

    Parameters
    ----------
    L_min, L_max : int
        Range of mask lengths (inclusive).
    g_min : int, default 0
        Minimum number of gaps.
    g_max : int or None
        Maximum number of gaps. If None, generates all masks with at
        least g_min gaps (up to L_max - 2 per length).

    Returns
    -------
    list[str]
        Mask strings, e.g. ``["*-*", "**-*", "*--*", "*-*-*", ...]``.

    Examples
    --------
    >>> generate_masks(3, 4, 1, 1)
    ['*-*', '**-*', '*--*']
    >>> generate_masks(3, 4, 1)  # g_max=None → all gaps >= 1
    ['*-*', '**-*', '*--*']
    """
    masks = []
    for L in range(L_min, L_max + 1):
        if L < 2:
            continue
        interior = L - 2
        upper = g_max if g_max is not None else interior
        for g in range(g_min, min(upper, interior) + 1):
            for gap_positions in combinations(range(1, L - 1), g):
                mask = ["*"] * L
                for gp in gap_positions:
                    mask[gp] = "-"
                masks.append("".join(mask))
    return masks


def _mask_rc(mask: str) -> str:
    """Return the reverse of a mask string (RC of the mask)."""
    return mask[::-1]


def _canonicalize_masks(masks: list[str]) -> list[str]:
    """Canonicalize masks: replace non-canonical with RC, deduplicate."""
    seen = set()
    result = []
    for m in masks:
        rc = _mask_rc(m)
        canon = min(m, rc)
        if canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result


class GappyEncoder:
    """Gappy k-mer count encoder.

    Parameters
    ----------
    masks : str or list[str] or None
        Explicit mask(s) like ``"*--*"``. ``*`` = concrete position,
        ``-`` = don't-care (gap). If None, must provide ``L``,
        ``g_min``, ``g_max``.
    L : int or None
        Total mask length. Required if ``masks`` is None.
    g_min : int or None
        Minimum number of gaps. Required if ``masks`` is None.
    g_max : int or None
        Maximum number of gaps. If None, generates all masks with at
        least g_min gaps.
    canonical_rc : bool, default False
        If True, canonicalize masks (drop RC duplicates) and collapse
        gappy k-mer RC pairs for symmetric masks.

    Attributes
    ----------
    n_features_ : int
        Total number of feature columns across all masks.
    masks_ : list[str]
        The mask strings actually used.

    Examples
    --------
    >>> from kmer.encoders import GappyEncoder
    >>> enc = GappyEncoder(masks=["*--*"])
    >>> X = enc.fit_transform(["ACGTACGT", "AAAACCCC"])
    >>> X.shape
    (2, 16)

    >>> enc = GappyEncoder(L=4, g_min=1, g_max=2)
    >>> enc.masks_
    ['*--*', '*-*-', '**--']
    """

    def __init__(
        self,
        masks: Optional[Union[str, Iterable[str]]] = None,
        *,
        L: Optional[int] = None,
        g_min: Optional[int] = None,
        g_max: Optional[int] = None,
        canonical_rc: bool = False,
    ):
        self.canonical_rc = canonical_rc

        if masks is not None:
            if isinstance(masks, (str, bytes)):
                masks = [masks]
            self.masks_ = [m if isinstance(m, str) else m.decode("ascii")
                           for m in masks]
            self._validate_masks(self.masks_)
            self._mode = "explicit"
            self._L = None
            self._g_min = None
            self._g_max = None
        else:
            if L is None or g_min is None:
                raise ValueError(
                    "either `masks` or (L, g_min) must be provided"
                )
            self._mode = "range"
            self._L = L
            self._g_min = g_min
            self._g_max = g_max  # may be None
            self.masks_ = generate_masks(L, L, g_min, g_max)

        # Canonicalize masks if requested.
        if self.canonical_rc:
            original = self.masks_
            self.masks_ = _canonicalize_masks(self.masks_)
            if len(self.masks_) < len(original):
                # Some masks were dropped as RC duplicates.
                pass

    @staticmethod
    def _validate_masks(masks):
        """Raise ValueError if any mask is malformed."""
        for m in masks:
            if not m or any(c not in "*-" for c in m):
                raise ValueError(f"invalid mask: {m!r}")
            if "*" not in m:
                raise ValueError(f"mask must have at least one '*': {m!r}")
            if len(m) > 32:
                raise ValueError(f"mask too long (max 32): {m!r}")
            n_concrete = m.count("*")
            if n_concrete > 15:
                raise ValueError(
                    f"mask has too many concrete positions (max 15): {m!r}"
                )
            # Reject leading/trailing gaps: positions 0 and L-1 must be '*'.
            if m[0] != '*' or m[-1] != '*':
                raise ValueError(
                    f"mask must start and end with '*' (no leading/trailing gaps): {m!r}"
                )

    @staticmethod
    def _generate_masks(L: int, g_min: int, g_max: int) -> list[str]:
        """Generate all C(L-2, g) masks for g in [g_min, g_max].

        No leading/trailing gaps: positions 0 and L-1 are always concrete.
        """
        return generate_masks(L, L, g_min, g_max)

    def fit(self, seqs: SeqLike) -> "GappyEncoder":
        self.n_features_ = None
        self._feature_names_ = None
        return self

    def transform(self, seqs: SeqLike) -> sp.csr_matrix:
        if not isinstance(seqs, (str, bytes)):
            seqs = list(seqs)
        else:
            seqs = [seqs]
        if not hasattr(self, "masks_"):
            raise RuntimeError("call fit() first")

        rows, cols, vals, n_features = _count_gappy_kmers(
            seqs, self.masks_, self.canonical_rc
        )

        n_seqs = len(seqs)
        X = sp.csr_matrix(
            (np.asarray(vals, dtype=np.int32),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(n_seqs, n_features),
            dtype=np.int32,
        )
        self.n_features_ = n_features
        return X

    def fit_transform(self, seqs: SeqLike) -> sp.csr_matrix:
        return self.fit(seqs).transform(seqs)

    def get_feature_names_out(self) -> np.ndarray:
        """Return feature labels.

        Format: ``<mask>_<pattern>`` where mask is the mask string and
        pattern is the concrete bases at the ``*`` positions.

        With ``canonical_rc=True``, only canonical patterns are included
        for symmetric masks.
        """
        from ..utils import code_to_kmer, reverse_complement, canonical_code
        if getattr(self, "_feature_names_", None) is None:
            names = []
            for mask in self.masks_:
                n_concrete = mask.count("*")
                is_symmetric = (mask == mask[::-1])
                do_collapse = self.canonical_rc and is_symmetric
                for code in range(4 ** n_concrete):
                    if do_collapse:
                        cc = canonical_code(code, n_concrete)
                        if code != cc:
                            continue
                    pattern = code_to_kmer(code, n_concrete)
                    names.append(f"{mask}_{pattern}")
            self._feature_names_ = np.array(names, dtype=object)
        return self._feature_names_

"""kmer.encoders.mismatch — mismatch k-mer count encoder.

For each k-mer in the sequence, counts all k-mers within Hamming
distance ``m`` (including the exact match). This is the mismatch
kernel feature space of Leslie, Eskin, Noble (2004).
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import scipy.sparse as sp

from ..utils import code_to_kmer, reverse_complement, canonical_code
from ._native._mismatch import _count_mismatch_kmers

SeqLike = Union[str, bytes, Iterable[Union[str, bytes]]]


class MismatchEncoder:
    """Mismatch k-mer count encoder.

    For each k-mer in each input sequence, increments counts for ALL
    k-mers within Hamming distance ``m`` (including the exact match).
    Returns a sparse CSR matrix.

    Parameters
    ----------
    k : int
        K-mer length. Must be in [1, 12].
    m : int
        Maximum number of mismatches. Must be in [0, k]. m=0 gives the
        plain spectrum encoder; m=1 allows one substitution per k-mer.
    canonical_rc : bool, default False
        If True, collapse each k-mer with its reverse complement.

    Attributes
    ----------
    n_features_ : int
        Number of feature columns (= 4**k).

    Examples
    --------
    >>> from kmer.encoders import MismatchEncoder
    >>> enc = MismatchEncoder(k=4, m=1)
    >>> X = enc.fit_transform(["ACGTACGT", "TTTTAAAA"])
    >>> X.shape
    (2, 256)
    """

    def __init__(self, k: int, m: int, canonical_rc: bool = False):
        if not 1 <= k <= 12:
            raise ValueError(f"k must be in [1, 12], got {k}")
        if not 0 <= m <= k:
            raise ValueError(f"m must be in [0, k={k}], got {m}")
        self.k = k
        self.m = m
        self.canonical_rc = canonical_rc

    def fit(self, seqs: SeqLike) -> "MismatchEncoder":
        self.n_features_ = 4 ** self.k
        self._feature_names_ = None
        return self

    def transform(self, seqs: SeqLike) -> sp.csr_matrix:
        if not isinstance(seqs, (str, bytes)):
            seqs = list(seqs)
        else:
            seqs = [seqs]
        if not hasattr(self, "n_features_"):
            self.fit(seqs)
        rows, cols, vals, n_features = _count_mismatch_kmers(
            seqs, self.k, self.m, self.canonical_rc
        )
        n_seqs = len(seqs)
        return sp.csr_matrix(
            (np.asarray(vals, dtype=np.int32),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(n_seqs, n_features),
            dtype=np.int32,
        )

    def fit_transform(self, seqs: SeqLike) -> sp.csr_matrix:
        return self.fit(seqs).transform(seqs)

    def get_feature_names_out(self) -> np.ndarray:
        """Return k-mer strings for each feature column.

        With ``canonical_rc=True``, returns only canonical k-mers.
        """
        if getattr(self, "_feature_names_", None) is None:
            names = []
            for code in range(4 ** self.k):
                if self.canonical_rc:
                    cc = canonical_code(code, self.k)
                    if code != cc:
                        continue
                names.append(code_to_kmer(code, self.k))
            self._feature_names_ = np.array(names, dtype=object)
        return self._feature_names_

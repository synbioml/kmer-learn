"""kmer.encoders.spectrum — rolling-hash k-mer count encoder.

Returns a scipy.sparse.csr_matrix of k-mer counts. Backed by the C
extension at kmer.encoders._native._spectrum for speed (~30 ms for
10k sequences × 500 bp × k=6).
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import scipy.sparse as sp

from ..utils import code_to_kmer, reverse_complement
from ._native._spectrum import _count_kmers

SeqLike = Union[str, bytes, Iterable[Union[str, bytes]]]


class SpectrumEncoder:
    """K-mer count encoder.

    Counts occurrences of every k-mer in each input sequence. Returns
    a sparse CSR matrix where rows are sequences and columns are k-mer
    codes (column index = packed 2-bit code, A=0 C=1 G=2 T=3 MSB-first).

    Parameters
    ----------
    k : int, default 6
        K-mer length. Must be in [1, 12]. For k > 12 the feature
        space (4^k) exceeds 16M entries — feature-name generation
        alone would use >1GB of memory.
    canonical_rc : bool, default False
        If True, collapse each k-mer with its reverse complement:
        the column index becomes ``min(code, rc(code))``. This treats
        a sequence and its reverse complement as having identical
        features, which is appropriate for double-stranded DNA.

    Attributes
    ----------
    n_features_ : int
        Number of feature columns. Equals ``4**k`` without canonical_rc,
        or the number of distinct canonical codes with canonical_rc.
        For canonical_rc=True, this is ``4**k / 2`` for odd k and
        ``(4**k + 4**(k//2)) / 2`` for even k (the +term accounts for
        palindromes).
    feature_names_ : np.ndarray
        K-mer strings for each column, populated by ``get_feature_names_out()``.

    Examples
    --------
    >>> from kmer.encoders import SpectrumEncoder
    >>> enc = SpectrumEncoder(k=3)
    >>> X = enc.fit_transform(["ACGTACGT", "TTTTAAAA"])
    >>> X.shape
    (2, 64)
    >>> enc.get_feature_names_out()[:5]
    array(['AAA', 'AAC', 'AAG', 'AAT', 'ACA'], dtype=object)
    """

    def __init__(self, k: int = 6, canonical_rc: bool = False):
        if not 1 <= k <= 12:
            raise ValueError(
                f"k must be in [1, 12], got {k}. "
                f"For k > 12 the feature space (4^k) exceeds 16M entries "
                f"and feature-name generation alone would use >1GB of memory."
            )
        self.k = k
        self.canonical_rc = canonical_rc

    def fit(self, seqs: SeqLike) -> "SpectrumEncoder":
        """Validate parameters. The encoder is stateless — fit is a no-op
        except for parameter validation and feature-name precomputation."""
        self.n_features_ = 4 ** self.k
        # Precompute feature names lazily.
        self._feature_names_ = None
        return self

    def transform(self, seqs: SeqLike) -> sp.csr_matrix:
        """Count k-mers in each sequence.

        Parameters
        ----------
        seqs : str | bytes | iterable of (str | bytes)
            Sequences to encode. A single sequence is treated as a
            1-row matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (n_seqs, 4**k). Each entry is the count of that
            k-mer in that sequence.
        """
        if not isinstance(seqs, (str, bytes)):
            seqs = list(seqs)
        else:
            seqs = [seqs]
        if not hasattr(self, "n_features_"):
            self.fit(seqs)
        rows, cols, vals, n_features = _count_kmers(
            seqs, self.k, self.canonical_rc
        )
        n_seqs = len(seqs)
        X = sp.csr_matrix(
            (np.asarray(vals, dtype=np.int32),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(n_seqs, n_features),
            dtype=np.int32,
        )
        return X

    def fit_transform(self, seqs: SeqLike) -> sp.csr_matrix:
        """Equivalent to ``fit(seqs).transform(seqs)``."""
        return self.fit(seqs).transform(seqs)

    def get_feature_names_out(self) -> np.ndarray:
        """Return k-mer strings for each feature column.

        Without ``canonical_rc``, returns all ``4**k`` k-mer strings.
        With ``canonical_rc=True``, returns only canonical k-mers
        (those where ``code <= rc(code)``), matching the compact
        column layout of the output matrix.
        """
        if getattr(self, "_feature_names_", None) is None:
            from ..utils import canonical_code
            names = []
            for code in range(4 ** self.k):
                if self.canonical_rc:
                    cc = canonical_code(code, self.k)
                    if code != cc:
                        continue  # skip non-canonical
                names.append(code_to_kmer(code, self.k))
            self._feature_names_ = np.array(names, dtype=object)
        return self._feature_names_

"""Ranked-list enrichment (minimum hypergeometric, DRIMust)."""
from __future__ import annotations

from typing import Optional

from ._base import KmerEnrichment


class DrimustEnrichment(KmerEnrichment):
    """K-mer enrichment along a ranked list by the minimum hypergeometric.

    Implements the word-level core of the DRIM/DRIMust line (Eden et
    al. 2007; Leibovich et al., *NAR* 2013): sequences are ordered by a
    continuous target ``y`` (e.g. differential expression), and every
    k-mer is tested for presence enrichment in prefixes of that
    ranking. The minimum hypergeometric (mHG) statistic scans all
    prefixes up to ``max_top`` of the list and keeps the strongest
    enrichment; its significance is evaluated exactly by dynamic
    programming over binary presence vectors.

    The motif-level part of DRIMust (suffix-tree IUPAC enumeration,
    PWM refinement) is out of scope; this class reports words only.

    Parameters
    ----------
    featurizer : object
        Sparse k-mer encoder (see :class:`kmer.enrichment.KmerEnrichment`).
    max_top : float, default 0.25
        Largest prefix fraction of the ranking considered by the mHG
        scan.
    exact : bool, default True
        Compute exact mHG p-values by dynamic programming; when
        ``False``, a fast normal approximation is used.
    fdr, random_state
        See :class:`kmer.enrichment.KmerEnrichment`.
    """

    def __init__(
        self,
        featurizer,
        *,
        max_top: float = 0.25,
        exact: bool = True,
        fdr: Optional[str] = "bh",
        random_state: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("fdr", fdr)
        kwargs.setdefault("random_state", random_state)
        super().__init__(featurizer, **kwargs)
        self.max_top = max_top
        self.exact = exact

    def _score(self, X, y):
        """mHG statistics per k-mer column (not implemented)."""
        raise NotImplementedError("DrimustEnrichment statistics are not implemented yet")

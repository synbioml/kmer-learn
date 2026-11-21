"""Regression enrichment (REDUCE)."""
from __future__ import annotations

from typing import Optional

from ._base import KmerEnrichment


class ReduceEnrichment(KmerEnrichment):
    """K-mer enrichment by regression against a continuous target.

    Implements the word-level core of REDUCE (Bussemaker, Li & Siggia,
    *Nat Genet* 2001): the target ``y`` is regressed on per-sequence
    k-mer counts. In ``stepwise`` mode k-mers are accepted greedily,
    most significant first, with residualization after each acceptance
    and a P-value entry threshold — the protocol that makes REDUCE's
    coefficients interpretable as motif affinities. With
    ``stepwise=False`` all k-mers are scored in one pass by Pearson or
    Spearman correlation with their parametric t-test p-values.

    Reverse-complement collapsing is handled by the featurizer
    (``canonical_rc=True``); REDUCE keeps one word per
    reverse-complement pair exactly this way.

    The matrix-level generalization (MatrixREDUCE PSAM refinement,
    Foat et al. 2006) is out of scope; this class reports words only.

    Parameters
    ----------
    featurizer : object
        Sparse k-mer encoder (see :class:`kmer.enrichment.KmerEnrichment`).
    stepwise : bool, default True
        Greedy acceptance protocol (REDUCE) instead of a single
        correlation scan.
    p_in : float, default 1e-3
        Two-sided P-value threshold for accepting a k-mer into the
        stepwise model (the original method used ~1e-3).
    max_terms : int or None
        Safety cap on the number of accepted k-mers.
    correlation : {"pearson", "spearman"}, default "pearson"
        Association measure for the non-stepwise scan.
    fdr, random_state
        See :class:`kmer.enrichment.KmerEnrichment`.
    """

    def __init__(
        self,
        featurizer,
        *,
        stepwise: bool = True,
        p_in: float = 1e-3,
        max_terms: Optional[int] = None,
        correlation: str = "pearson",
        fdr: Optional[str] = "bh",
        random_state: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("fdr", fdr)
        kwargs.setdefault("random_state", random_state)
        super().__init__(featurizer, **kwargs)
        self.stepwise = stepwise
        self.p_in = p_in
        self.max_terms = max_terms
        self.correlation = correlation

    def _score(self, X, y):
        """Correlation/t-statistics per k-mer column (not implemented)."""
        raise NotImplementedError("ReduceEnrichment statistics are not implemented yet")

    def _analyze(self, X, y, names):
        """Stepwise protocol overrides the batch template (not implemented)."""
        if self.stepwise:
            raise NotImplementedError(
                "ReduceEnrichment stepwise protocol is not implemented yet"
            )
        super()._analyze(X, y, names)

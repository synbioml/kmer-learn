"""Two-set k-mer enrichment (Fisher exact testing)."""
from __future__ import annotations

from typing import Optional

from ._base import Background, KmerEnrichment


class FisherEnrichment(KmerEnrichment):
    """K-mer enrichment between two sequence sets by Fisher's exact test.

    Implements the word-level core shared by HOMER (Heinz et al., *NAR*
    2010) and DREME (Bailey 2011): for every k-mer, presence counts in
    positives versus negatives form a 2x2 table tested with Fisher's
    exact test; the effect size is the log-odds ratio. This is the
    statistic behind "is this word overrepresented in my set versus a
    matched control?".

    Negatives follow the :class:`kmer.models.KmerMultinomialNB`
    convention: passed explicitly, or generated from ``background``
    (typically :class:`kmer.background.KmerShuffler`, preserving k-mer
    content while destroying positional information) when ``negatives``
    is ``None``.

    The motif-level generalizations (DREME's regular-expression
    refinement, HOMER's positional oligo analysis, known-motif
    scanning) are out of scope.

    Parameters
    ----------
    featurizer : object
        Sparse k-mer encoder (see :class:`kmer.enrichment.KmerEnrichment`).
    background : BaseBackgroundModel, callable, or None
        Negative-set generator used when ``negatives=None``. A
        background model must be configured for auto-negatives.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Enrichment direction tested.
    presence : bool, default True
        Test presence/absence per sequence rather than counts.
    fdr, random_state
        See :class:`kmer.enrichment.KmerEnrichment`.
    """

    def __init__(
        self,
        featurizer,
        *,
        background: Optional[Background] = None,
        alternative: str = "two-sided",
        presence: bool = True,
        fdr: Optional[str] = "bh",
        random_state: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("fdr", fdr)
        kwargs.setdefault("random_state", random_state)
        super().__init__(featurizer, background=background, **kwargs)
        self.alternative = alternative
        self.presence = presence

    def fit(self, positives, negatives=None):
        """Score all k-mers for enrichment in ``positives``.

        Parameters
        ----------
        positives : list of str
            Sequences of the positive set.
        negatives : list of str or None
            Control sequences. When ``None``, one matched negative is
            generated per positive through ``background``.

        Returns
        -------
        self
        """
        raise NotImplementedError("FisherEnrichment is not implemented yet")

    def _score(self, X, y):
        """Fisher exact p-values per k-mer column (not implemented)."""
        raise NotImplementedError("FisherEnrichment statistics are not implemented yet")

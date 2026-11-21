"""Mutual-information enrichment (FIRE)."""
from __future__ import annotations

from typing import Optional

from ._base import KmerEnrichment


class FireEnrichment(KmerEnrichment):
    """K-mer enrichment by mutual information with the target.

    Implements the scoring core of FIRE (Elemento, Slonim & Tavazoie,
    *Mol Cell* 2007): every k-mer's pattern of presence/absence across
    sequences is scored by the mutual information it shares with the
    target ``y``. Significance is assessed by permuting ``y``
    (breaking the sequence-target association), *not* by modeling the
    sequences; a background model is therefore optional and unused by
    default.

    Targets may be discrete labels (e.g. expression clusters, binary
    classes) or continuous values discretized with ``y_bins``
    equal-frequency bins (e.g. single-condition log-ratios).

    The motif-level part of FIRE (greedy IUPAC generalization of
    informative seeds, motif redundancy filtering, PWM logos) is out of
    scope; this class reports informative k-mers only.

    Parameters
    ----------
    featurizer : object
        Sparse k-mer encoder (see :class:`kmer.enrichment.KmerEnrichment`).
    y_bins : int or None
        Number of equal-frequency bins for continuous targets; ``None``
        treats ``y`` as discrete labels.
    presence : bool, default True
        Score presence/absence (FIRE) rather than counts.
    n_permutations : int, default 1000
        Label permutations for p-value calibration (the original method
        used 10,000; lower values trade calibration resolution for
        speed).
    fdr, random_state
        See :class:`kmer.enrichment.KmerEnrichment`.
    """

    def __init__(
        self,
        featurizer,
        *,
        y_bins: Optional[int] = None,
        presence: bool = True,
        n_permutations: int = 1000,
        fdr: Optional[str] = "bh",
        random_state: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("n_permutations", n_permutations)
        kwargs.setdefault("fdr", fdr)
        kwargs.setdefault("random_state", random_state)
        super().__init__(featurizer, **kwargs)
        self.y_bins = y_bins
        self.presence = presence

    def _score(self, X, y):
        """Plug-in mutual information per k-mer column (not implemented)."""
        raise NotImplementedError("FireEnrichment statistics are not implemented yet")

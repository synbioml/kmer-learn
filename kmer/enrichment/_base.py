"""Shared machinery for k-mer enrichment analysis.

This module defines the abstract interface implemented by every
enrichment method in :mod:`kmer.enrichment`. The base class owns the
parts that are method independent:

1. featurization through any sparse k-mer encoder;
2. optional label-permutation calibration of per-k-mer scores;
3. multiple-testing correction (Benjamini-Hochberg or Bonferroni);
4. result assembly, ranking, and selection.

Subclasses supply only the statistical core: raw scores, and analytic
p-values where the underlying method provides them.

The module deliberately stays at the k-mer level. Motif-level
instruments (degenerate-code generalization, PWM/PSAM refinement,
motif scanning and comparison) are out of scope by design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ..background import BaseBackgroundModel

#: Background: a background model instance (Shuffler, Chunker, ...) or any
#: callable mapping a single sequence to a disrupted version of it.
Background = Union[BaseBackgroundModel, Callable[[Union[str, bytes]], Union[str, bytes]]]


@dataclass(frozen=True)
class EnrichmentResult:
    """Per-k-mer enrichment outcome.

    Attributes
    ----------
    kmer : str
        Feature name produced by the featurizer.
    score : float
        Primary statistic (units depend on the method: bits for mutual
        information, correlation coefficient for regression, ``-log10`` of
        the hypergeometric tail for mHG, ``-log10`` of the Fisher p-value
        for enrichment testing).
    pvalue : float
        Uncorrected p-value; analytic where the method provides one,
        label-permutation based otherwise. ``nan`` when no calibration is
        configured.
    qvalue : float
        Multiple-testing corrected p-value; equals ``pvalue`` when
        ``fdr=None``.
    effect : float
        Method-specific effect size (e.g. log-odds ratio for Fisher,
        regression coefficient for REDUCE).
    direction : int
        ``+1`` or ``-1``: whether the k-mer is enriched in the positive
        direction of the target or depleted.
    extra : dict
        Optional method-specific annotations (e.g. stepwise acceptance
        order for REDUCE, robustness scores for FIRE).
    """

    kmer: str
    score: float
    pvalue: float
    qvalue: float
    effect: float
    direction: int
    extra: dict = field(default_factory=dict)


class KmerEnrichment(ABC):
    """Abstract base class for k-mer enrichment analysis.

    The estimator scores every k-mer produced by ``featurizer`` for
    association with a target ``y`` (labels, ranks, or continuous
    values, depending on the subclass), calibrates the scores, and
    returns a ranked table.

    Two distinct null models are supported and must not be conflated:

    * ``background`` reshapes *sequences* (k-mer-preserving shuffles via
      :mod:`kmer.background`). It answers "more frequent than
      composition-matched chance?" and, for two-set methods such as
      :class:`kmer.enrichment.FisherEnrichment`, generates matched
      negative sequences exactly like
      :class:`kmer.models.KmerMultinomialNB` does.
    * ``n_permutations`` breaks the sequence-to-target association by
      permuting ``y``. It answers "informative about the target?" without
      modeling the sequences at all; this is the null used by FIRE
      (Elemento, Slonim & Tavazoie 2007), which therefore accepts
      ``background=None``.

    Parameters
    ----------
    featurizer : object
        Encoder with ``fit_transform(seqs)`` and ``transform(seqs)``
        methods returning a scipy sparse matrix, plus
        ``get_feature_names_out()``. Typically
        :class:`kmer.encoders.SpectrumEncoder` or
        :class:`kmer.encoders.GappyEncoder`; k, gap masks, mismatch
        neighborhoods, and reverse-complement collapsing are configured
        there, not here.
    background : BaseBackgroundModel, callable, or None
        Sequence-space background for null generation or negative-set
        synthesis. See the note above; ignored by subclasses whose
        statistics never touch sequence-space nulls.
    n_permutations : int, default 0
        Number of label permutations used for empirical calibration of
        per-k-mer scores. Applied only when the subclass provides no
        analytic p-values, or as an additional empirical check when
        ``force_permutation=True``.
    force_permutation : bool, default False
        Compute permutation p-values even when analytic p-values exist.
    fdr : {None, "bh", "bonferroni"}, default "bh"
        Multiple-testing correction applied across all scored k-mers.
    random_state : int, default 0
        Seed for permutation and background sampling.

    Attributes
    ----------
    results_ : list of EnrichmentResult
        All scored k-mers sorted by descending score.
    scores_ : dict
        Mapping ``kmer -> score``, mirroring
        ``KmerMultinomialNB.kmer_scores_``.
    """

    def __init__(
        self,
        featurizer,
        *,
        background: Optional[Background] = None,
        n_permutations: int = 0,
        force_permutation: bool = False,
        fdr: Optional[str] = "bh",
        random_state: int = 0,
    ):
        self.featurizer = featurizer
        self.background = background
        self.n_permutations = n_permutations
        self.force_permutation = force_permutation
        self.fdr = fdr
        self.random_state = random_state

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, seqs, y=None):
        """Score all k-mers for association with the target.

        Parameters
        ----------
        seqs : list of str
            Sequences to featurize and score.
        y : array-like or None
            Per-sequence target: labels, ranks, or continuous values
            depending on the subclass. Two-set subclasses (e.g.
            :class:`kmer.enrichment.FisherEnrichment`) override this
            signature with ``fit(positives, negatives=None)``.

        Returns
        -------
        self
        """
        X = self.featurizer.fit_transform(seqs)
        names = list(self.featurizer.get_feature_names_out())
        self._analyze(X, y, names)
        return self

    def top(self, n: int = 20) -> list[EnrichmentResult]:
        """Return the ``n`` highest-scoring k-mers."""
        self._check_fitted()
        return self.results_[:n]

    def select(self, threshold_q: float = 0.05) -> list[str]:
        """Return k-mer names whose q-value does not exceed ``threshold_q``."""
        self._check_fitted()
        return [r.kmer for r in self.results_ if r.qvalue <= threshold_q]

    # ------------------------------------------------------------------
    # template hooks
    # ------------------------------------------------------------------
    def _analyze(self, X, y, names):
        """Template method: score, calibrate, correct, assemble."""
        scores, pvalues, effects, directions = self._score(X, y)

        if pvalues is None or self.force_permutation:
            if self.n_permutations > 0:
                permuted = self._permutation_pvalues(X, y, scores)
                pvalues = permuted if pvalues is None else pvalues
            elif pvalues is None:
                pvalues = np.full(scores.shape, np.nan)

        # K-mers with non-finite scores were never actually tested (e.g.
        # absent everywhere); they are excluded from the multiple-testing
        # family before correction.
        tested = np.isfinite(scores)
        scores = scores[tested]
        effects = effects[tested]
        directions = directions[tested]
        pvalues = pvalues[tested] if pvalues is not None else None
        names = [name for name, keep in zip(names, tested) if keep]

        if pvalues is None:
            pvalues = np.full(scores.shape, np.nan)
        qvalues = self._correct_fdr(pvalues, self.fdr)
        self.results_ = [
            EnrichmentResult(
                kmer=name,
                score=float(score),
                pvalue=float(pval),
                qvalue=float(qval),
                effect=float(effect),
                direction=int(direction),
            )
            for name, score, pval, qval, effect, direction in zip(
                names, scores, pvalues, qvalues, effects, directions
            )
        ]
        self.results_.sort(key=lambda r: r.score, reverse=True)
        self.scores_ = {r.kmer: r.score for r in self.results_}

    @abstractmethod
    def _score(self, X, y):
        """Compute per-k-mer statistics.

        Returns
        -------
        scores : np.ndarray
            Primary statistic per k-mer (column of ``X``).
        pvalues : np.ndarray or None
            Analytic p-values, or ``None`` if the method has none and
            relies on permutation calibration.
        effects : np.ndarray
            Effect size per k-mer.
        directions : np.ndarray
            ``+1``/``-1`` per k-mer.
        """

    # ------------------------------------------------------------------
    # shared statistics
    # ------------------------------------------------------------------
    def _permutation_pvalues(self, X, y, scores):
        """Empirical upper-tail p-values from label permutations.

        Repeatedly permutes ``y``, recomputes the scores through the
        subclass ``_score``, and counts exceedances. Generic: it works
        for every statistic without knowing its distribution.
        """
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        exceedances = np.zeros(scores.shape[0])
        for _ in range(self.n_permutations):
            permuted_scores = self._score(X, rng.permutation(y))[0]
            exceedances += permuted_scores >= scores - 1e-12
        return (exceedances + 1.0) / (self.n_permutations + 1.0)

    @staticmethod
    def _correct_fdr(pvalues, method):
        """Correct p-values across k-mers for multiple testing.

        ``None``/``"none"`` leaves p-values untouched; ``"bonferroni"``
        multiplies by the number of tests; ``"bh"`` applies
        Benjamini-Hochberg with enforced monotonicity.
        """
        p = np.asarray(pvalues, dtype=float)
        if method in (None, "none"):
            return p.copy()
        if method == "bonferroni":
            return np.minimum(p * p.shape[0], 1.0)
        if method == "bh":
            order = np.argsort(p, kind="stable")
            ranked = p[order] * p.shape[0] / (np.arange(p.shape[0]) + 1.0)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            out = np.empty_like(ranked)
            out[order] = np.minimum(ranked, 1.0)
            return out
        raise ValueError(f"unknown fdr method: {method!r}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "results_"):
            raise RuntimeError("this KmerEnrichment instance is not fitted yet")

"""Tests for the DifferentialKmerScorer."""

import pytest
import numpy as np

from kmer.models import DifferentialKmerScorer
from kmer.encoders import SpectrumEncoder
from kmer.perturb import KmerShuffler, Chunker


# ======================================================================
# Construction
# ======================================================================

def test_construction_with_shuffler_background():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=KmerShuffler(k=1, seed=42),
    )
    assert scorer.featurizer.k == 4
    assert scorer.background is not None
    assert scorer.alpha == 1.0
    assert scorer.beta is None  # defaults to alpha at fit time


def test_construction_with_chunker_background():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=Chunker(min_size=2, max_size=4, seed=42),
    )
    assert scorer.background is not None


def test_construction_with_no_background():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
    )
    assert scorer.background is None


def test_construction_asymmetric_smoothing():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=KmerShuffler(k=1, seed=42),
        alpha=0.5,
        beta=2.0,
    )
    assert scorer.alpha == 0.5
    assert scorer.beta == 2.0


# ======================================================================
# Fit
# ======================================================================

def test_fit_with_explicit_negatives():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
    )
    positives = ["ACGTACGTACGT", "ACGTACGTACGT", "ACGTACGTACGT"]
    negatives = ["TTTTAAAATTTT", "GGGGCCCCGGGG", "AAAATTTTCCCC"]
    scorer.fit(positives, negatives)
    assert hasattr(scorer, "kmer_scores_")
    assert scorer.n_positives_ == 3
    assert scorer.n_negatives_ == 3
    assert len(scorer.kmer_scores_) == 64


def test_fit_with_auto_negatives():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=KmerShuffler(k=1, seed=42),
    )
    positives = ["ACGTACGTACGT"] * 5
    scorer.fit(positives)
    assert scorer.n_positives_ == 5
    assert scorer.n_negatives_ == 5  # one negative per positive


def test_fit_without_background_or_negatives_raises():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
    )
    with pytest.raises(ValueError, match="background"):
        scorer.fit(["ACGTACGT"] * 3)


# ======================================================================
# Decision function
# ======================================================================

def test_decision_function_returns_float_array():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=KmerShuffler(k=1, seed=42),
    )
    positives = ["ACGTACGTACGT"] * 20
    scorer.fit(positives)
    scores = scorer.decision_function(["ACGTACGTACGT", "TTTTAAAATTTT"])
    assert scores.shape == (2,)
    assert scores.dtype == np.float64


def test_predict_returns_int_labels():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=KmerShuffler(k=1, seed=42),
    )
    scorer.fit(["ACGTACGTACGT"] * 20)
    preds = scorer.predict(["ACGTACGTACGT", "TTTTAAAATTTT"])
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_returns_probabilities():
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=KmerShuffler(k=1, seed=42),
    )
    scorer.fit(["ACGTACGTACGT"] * 20)
    proba = scorer.predict_proba(["ACGTACGTACGT", "TTTTAAAATTTT"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ======================================================================
# Enriched k-mer detection
# ======================================================================

def test_enriched_kmers_identified():
    """When positives are ACGT-repeats and negatives are mono-shuffled,
    ACGT-family 4-mers should be enriched."""
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=KmerShuffler(k=1, seed=42),
    )
    positives = ["ACGTACGTACGT"] * 50
    scorer.fit(positives)
    top_kmers = [k for k, v in sorted(scorer.kmer_scores_.items(), key=lambda x: -x[1])[:4]]
    # The 4-mers of "ACGTACGT..." are ACGT, CGTA, GTAC, TACG.
    assert "ACGT" in top_kmers
    assert "CGTA" in top_kmers or "GTAC" in top_kmers or "TACG" in top_kmers


def test_positive_sequences_score_higher_than_negatives():
    """On average, positives should score higher than negatives."""
    np.random.seed(0)
    positives = ["ACGTACGT" + "ACGT" * np.random.randint(0, 3) for _ in range(30)]
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=KmerShuffler(k=1, seed=42),
    )
    scorer.fit(positives)

    # New positives should score high, random ACGT should score ~0.
    new_pos = ["ACGTACGTACGTACGT", "ACGTACGTACGTACGT"]
    new_neg = ["TTTTAAAAGGGGCCCC", "AAAACCCCGGGGTTTT"]
    pos_scores = scorer.decision_function(new_pos)
    neg_scores = scorer.decision_function(new_neg)
    assert pos_scores.mean() > neg_scores.mean()


# ======================================================================
# RC invariance with canonical_rc
# ======================================================================

def test_rc_invariance_with_canonical_rc():
    """A sequence and its RC should produce the same score when
    featurizer uses canonical_rc=True."""
    from kmer.utils import reverse_complement
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4, canonical_rc=True),
        background=KmerShuffler(k=1, seed=42),
    )
    scorer.fit(["ACGTACGTACGTACGT"] * 20)
    seq = "ACGTACGTACGTACGTAA"
    rc = reverse_complement(seq)
    scores = scorer.decision_function([seq, rc])
    assert np.isclose(scores[0], scores[1], atol=1e-10), (
        f"RC sequences scored differently: {scores[0]} vs {scores[1]}"
    )


# ======================================================================
# Chunker background
# ======================================================================

def test_chunker_background_works():
    """The scorer should accept a Chunker as background."""
    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=Chunker(min_size=2, max_size=4, seed=42),
    )
    scorer.fit(["ACGTACGTACGTACGT"] * 10)
    # Just verify it ran.
    assert scorer.n_positives_ == 10

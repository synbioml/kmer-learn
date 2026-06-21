"""Tests for the BaseBackgroundModel ABC and its concrete implementations."""

import pytest
from kmer.perturb import BaseBackgroundModel, KmerShuffler, Chunker


def test_base_background_model_is_abstract():
    with pytest.raises(TypeError):
        BaseBackgroundModel()


def test_kmer_shuffler_is_base_background_model():
    """KmerShuffler instances are recognized as BaseBackgroundModel."""
    sh = KmerShuffler(k=2, seed=42)
    assert isinstance(sh, BaseBackgroundModel)


def test_chunker_is_base_background_model():
    """Chunker instances are recognized as BaseBackgroundModel."""
    ch = Chunker(min_size=2, max_size=4, seed=42)
    assert isinstance(ch, BaseBackgroundModel)


def test_shuffler_perturb():
    """KmerShuffler.perturb produces the same output as KmerShuffler.shuffle
    when called on a fresh instance with the same seed."""
    sh1 = KmerShuffler(k=2, seed=42)
    sh2 = KmerShuffler(k=2, seed=42)
    seq = "ACGTACGTACGT"
    assert sh1.perturb(seq) == sh2.shuffle(seq)


def test_shuffler_perturb_many():
    """KmerShuffler.perturb_many matches shuffle_many."""
    sh1 = KmerShuffler(k=2, seed=42)
    sh2 = KmerShuffler(k=2, seed=42)
    seqs = ["ACGTACGT", "TTTTAAAA"]
    assert sh1.perturb_many(seqs) == sh2.shuffle_many(seqs)


def test_chunker_perturb():
    """Chunker.perturb produces the same output as Chunker.chunk when
    called on a fresh instance with the same seed."""
    ch1 = Chunker(min_size=2, max_size=4, seed=42)
    ch2 = Chunker(min_size=2, max_size=4, seed=42)
    seq = "ACGTACGTACGT"
    assert ch1.perturb(seq) == ch2.chunk(seq)


def test_chunker_perturb_many():
    """Chunker.perturb_many matches chunk_many."""
    ch1 = Chunker(min_size=2, max_size=4, seed=42)
    ch2 = Chunker(min_size=2, max_size=4, seed=42)
    seqs = ["ACGTACGT", "TTTTAAAA"]
    assert ch1.perturb_many(seqs) == ch2.chunk_many(seqs)


def test_custom_background_model():
    """User-defined BaseBackgroundModel subclass works with DifferentialKmerScorer."""
    from kmer.models import DifferentialKmerScorer
    from kmer.encoders import SpectrumEncoder

    class ReverseBackground(BaseBackgroundModel):
        """Trivial background: just reverse the sequence (no compositional change)."""
        def perturb(self, seq):
            return seq[::-1]

    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=ReverseBackground(),
    )
    scorer.fit(["ACGTACGT"] * 5)
    assert scorer.n_positives_ == 5
    assert scorer.n_negatives_ == 5


def test_callable_background_still_works():
    """Plain callables (not BaseBackgroundModel) still work as backgrounds."""
    from kmer.models import DifferentialKmerScorer
    from kmer.encoders import SpectrumEncoder

    scorer = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=3),
        background=lambda s: s[::-1],  # plain callable
    )
    scorer.fit(["ACGTACGT"] * 5)
    assert scorer.n_positives_ == 5

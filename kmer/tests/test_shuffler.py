"""Invariant-based tests for the k-mer shuffler.

Tests:
- k-mer composition is preserved (preserve and free modes)
- length is preserved (preserve and free modes)
- length is reduced by 2(k-1) in crop mode
- output is deterministic given the same seed
- RNG state advances between calls
- getstate/setstate round-trip
- non-ACGT characters raise ValueError
- bytes and str inputs both work
- shuffle_many matches serial shuffle calls
- shuffle_many is reproducible across n_jobs
"""

import pytest
import numpy as np
from collections import Counter

from kmer.perturb import KmerShuffler


# Diverse test sequences covering common patterns.
TEST_SEQUENCES = [
    "ACGTACGTACGTACGT",                                # repeating
    "AAACCCGGGTTTAAACCCGGGTTT",                        # multi-repeat
    "ATGCATGCATGCAATGCATGCATGCATGC",                   # mixed
    "GATTACAGATTACA" * 3,                              # spaced repeat
    "AAGCTTGGATCCGAATTC" * 4,                          # restriction sites
    "AGCTTAGCTTAGCTTAGCTT" * 2,                        # 2-mer repeat
    "AAAGCTAAAAGCTTAAAGCTTAAAGCTTAAAGCTT",             # uneven
    "GCGCGCGCGCATATATATATGCGCGCGCGCATATAT",            # GC + AT blocks
    "T" * 30,                                          # homopolymer
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGT",            # long repeat
]


def kmer_counts(seq, k):
    return Counter(seq[i:i+k] for i in range(len(seq) - k + 1))


# ======================================================================
# k-mer preservation
# ======================================================================

@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_preserve_mode_kmer_composition(seq, k):
    """In preserve mode, k-mer composition must be exactly preserved."""
    sh = KmerShuffler(k=k, seed=42, endpoints="preserve")
    out = sh.shuffle(seq)
    assert kmer_counts(seq, k) == kmer_counts(out, k), (
        f"k={k}: composition not preserved for seq len {len(seq)}"
    )


@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_free_mode_kmer_composition(seq, k):
    """In free mode, k-mer composition must be preserved up to at most
    one wrap-around k-mer gained and one k-mer lost (total L1 diff <= 2)."""
    sh = KmerShuffler(k=k, seed=42, endpoints="free")
    out = sh.shuffle(seq)
    in_counts = kmer_counts(seq, k)
    out_counts = kmer_counts(out, k)
    all_kmers = set(in_counts) | set(out_counts)
    diff = sum(abs(in_counts[km] - out_counts.get(km, 0)) for km in all_kmers)
    assert diff <= 2, (
        f"k={k}: L1 diff = {diff} (expected <= 2) for seq len {len(seq)}"
    )


@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_crop_mode_kmer_composition(seq, k):
    """In crop mode, k-mer composition of the interior must be preserved
    up to at most one wrap-around k-mer gained/lost (same as free mode)."""
    drop = k - 1
    if len(seq) < 2 * drop + 1:
        pytest.skip("sequence too short to crop")
    interior = seq[drop : len(seq) - drop]
    sh = KmerShuffler(k=k, seed=42, endpoints="crop")
    out = sh.shuffle(seq)
    in_counts = kmer_counts(interior, k)
    out_counts = kmer_counts(out, k)
    all_kmers = set(in_counts) | set(out_counts)
    diff = sum(abs(in_counts[km] - out_counts.get(km, 0)) for km in all_kmers)
    assert diff <= 2, (
        f"k={k}: cropped interior L1 diff = {diff} (expected <= 2)"
    )


# ======================================================================
# Length preservation
# ======================================================================

@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_length_preservation(seq, k):
    """Length must be preserved in preserve and free modes."""
    for mode in ["preserve", "free"]:
        sh = KmerShuffler(k=k, seed=42, endpoints=mode)
        out = sh.shuffle(seq)
        assert len(out) == len(seq), (
            f"k={k} mode={mode}: len(out)={len(out)} != len(seq)={len(seq)}"
        )


@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_crop_mode_length(seq, k):
    """Crop mode reduces length by 2(k-1)."""
    drop = k - 1
    if len(seq) < 2 * drop + 1:
        pytest.skip("sequence too short to crop")
    sh = KmerShuffler(k=k, seed=42, endpoints="crop")
    out = sh.shuffle(seq)
    assert len(out) == len(seq) - 2 * drop


# ======================================================================
# Determinism
# ======================================================================

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_same_seed_same_output(k):
    """Two shufflers with the same seed produce identical first outputs."""
    seq = "AAACCCGGGTTTAAACCCGGGTTT"
    sh1 = KmerShuffler(k=k, seed=123)
    sh2 = KmerShuffler(k=k, seed=123)
    assert sh1.shuffle(seq) == sh2.shuffle(seq)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_different_calls_differ(k):
    """Successive calls on the same shuffler should usually differ."""
    seq = "AAACCCGGGTTTAAACCCGGGTTT" * 3
    sh = KmerShuffler(k=k, seed=42)
    outputs = {sh.shuffle(seq) for _ in range(5)}
    assert len(outputs) > 1, "successive shuffles all identical"


# ======================================================================
# RNG state
# ======================================================================

def test_getstate_setstate_roundtrip():
    """Saving and restoring RNG state produces identical subsequent outputs."""
    seq = "AAACCCGGGTTTAAACCCGGGTTT"
    sh1 = KmerShuffler(k=3, seed=42)
    sh1.shuffle(seq)  # advance state
    state = sh1.getstate()
    next1 = sh1.shuffle(seq)

    sh2 = KmerShuffler(k=3, seed=42)
    sh2.shuffle(seq)
    sh2.setstate(state)
    next2 = sh2.shuffle(seq)

    assert next1 == next2


# ======================================================================
# Edge cases
# ======================================================================

def test_short_sequence_returns_input():
    """Sequences shorter than k are returned unchanged."""
    sh = KmerShuffler(k=5, seed=42)
    assert sh.shuffle("AC") == "AC"
    assert sh.shuffle("ACGT") == "ACGT"


def test_exact_k_length_returns_input():
    """A sequence of length exactly k has only one k-mer, so one Eulerian path."""
    sh = KmerShuffler(k=4, seed=42)
    # Sequence has only one 4-mer, so only one Eulerian path exists.
    out = sh.shuffle("ACGT")
    assert kmer_counts("ACGT", 4) == kmer_counts(out, 4)


def test_non_acgt_raises():
    """Non-ACGT characters must raise ValueError."""
    sh = KmerShuffler(k=2, seed=42)
    with pytest.raises(ValueError, match="non-ACGT"):
        sh.shuffle("ACGTNACGT")


def test_bytes_input():
    """bytes input produces bytes output."""
    sh = KmerShuffler(k=3, seed=42)
    out = sh.shuffle(b"ACGTACGTACGTACGT")
    assert isinstance(out, bytes)
    in_counts = kmer_counts(b"ACGTACGTACGTACGT", 3)
    out_counts = kmer_counts(out, 3)
    assert in_counts == out_counts


def test_bytes_and_str_produce_same_output():
    """Same seed → same output regardless of input type."""
    seq_str = "ACGTACGTACGTACGT"
    seq_bytes = seq_str.encode("ascii")
    sh1 = KmerShuffler(k=3, seed=42)
    sh2 = KmerShuffler(k=3, seed=42)
    out_str = sh1.shuffle(seq_str)
    out_bytes = sh2.shuffle(seq_bytes)
    assert out_str == out_bytes.decode("ascii")


def test_invalid_k_raises():
    """k must be in [1, 12]."""
    with pytest.raises(ValueError):
        KmerShuffler(k=0)
    with pytest.raises(ValueError):
        KmerShuffler(k=13)


def test_invalid_endpoints_raises():
    """Invalid endpoints mode raises ValueError."""
    with pytest.raises(ValueError):
        KmerShuffler(k=3, seed=42, endpoints="invalid")


# ======================================================================
# Batch operations
# ======================================================================

def test_shuffle_many_matches_serial():
    """shuffle_many should produce the same output as repeated shuffle calls."""
    seqs = [
        "AAACCCGGGTTT",
        "ACGTACGTACGT",
        "TTTTGGGGCCCC",
        "ACGTACGTAAAA",
        "GCGCGCGCGCGC",
    ]
    sh_serial = KmerShuffler(k=3, seed=42)
    serial = [sh_serial.shuffle(s) for s in seqs]
    sh_batch = KmerShuffler(k=3, seed=42)
    batch = sh_batch.shuffle_many(seqs)
    assert serial == batch


def test_shuffle_many_reproducible_across_n_jobs():
    """shuffle_many with n_jobs>1 must produce identical output regardless
    of the specific n_jobs value (as long as n_jobs > 1). Single-threaded
    mode (n_jobs=1) uses the master RNG directly, so it produces a
    different stream from parallel mode."""
    seqs = [
        "AAACCCGGGTTT",
        "ACGTACGTACGT",
        "TTTTGGGGCCCC",
        "ACGTACGTAAAA",
        "GCGCGCGCGCGC",
        "ATGCATGCATGC",
    ]
    # n_jobs=2 vs n_jobs=4 should match (both use derived per-task keys).
    result_2 = KmerShuffler(k=3, seed=42).shuffle_many(seqs, n_jobs=2)
    result_4 = KmerShuffler(k=3, seed=42).shuffle_many(seqs, n_jobs=4)
    assert result_2 == result_4


def test_shuffle_many_preserves_kmer_composition():
    """Every output of shuffle_many must preserve k-mer composition."""
    seqs = [
        "AAACCCGGGTTT",
        "ACGTACGTACGT",
        "TTTTGGGGCCCC",
    ]
    sh = KmerShuffler(k=3, seed=42)
    outs = sh.shuffle_many(seqs)
    for s, o in zip(seqs, outs):
        assert kmer_counts(s, 3) == kmer_counts(o, 3)


def test_shuffle_many_bytes_input():
    """shuffle_many accepts bytes inputs."""
    seqs = [b"ACGTACGT", b"TTTTAAAA"]
    sh = KmerShuffler(k=2, seed=42)
    outs = sh.shuffle_many(seqs)
    assert all(isinstance(o, bytes) for o in outs)

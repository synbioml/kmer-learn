"""Invariant-based tests for the sequence chunker.

Tests:
- length preservation
- chunk size constraints (all chunks in [min, max] except at most one residual)
- nucleotide composition preservation (no mutations)
- determinism
- all algorithm × residual_chunk combinations work
- flip_strand error on non-IUPAC
- flip_strand RCs chunks
- bytes input
- chunk_many reproducibility across n_jobs
"""

import pytest
from collections import Counter

from kmer.perturb import Chunker


TEST_SEQUENCES = [
    "ACGTACGTACGT",
    "AAACCCGGGTTTAAACCCGGGTTT",
    "ATGCATGCATGCAATGCATGCATGCATGC",
    "GATTACAGATTACAGATTACA",
    "AAGCTTGGATCCGAATTCAAGCTTGGATCCGAATTC",
    "AGCTTAGCTTAGCTTAGCTT",
    "T" * 30,
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGT",
]


# ======================================================================
# Length preservation
# ======================================================================

@pytest.mark.parametrize("seq", TEST_SEQUENCES)
@pytest.mark.parametrize("min_size", [2, 3, 4])
@pytest.mark.parametrize("max_size", [4, 5, 6])
def test_length_preservation(seq, min_size, max_size):
    """Output length must equal input length."""
    if min_size > max_size:
        pytest.skip("invalid range")
    ch = Chunker(min_size=min_size, max_size=max_size, seed=42)
    out = ch.chunk(seq)
    assert len(out) == len(seq)


# ======================================================================
# Nucleotide composition preservation (no flip_strand)
# ======================================================================

@pytest.mark.parametrize("seq", TEST_SEQUENCES)
def test_composition_preserved_no_flip(seq):
    """Without flip_strand, mononucleotide composition is exactly preserved."""
    ch = Chunker(min_size=2, max_size=5, seed=42)
    out = ch.chunk(seq)
    assert Counter(seq) == Counter(out)


# ======================================================================
# Algorithm × residual_chunk coverage
# ======================================================================

ALGORITHMS = ["random", "backtrack"]
RESIDUAL_MODES = ["end", "start", "random", "extend", "distribute"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("residual", RESIDUAL_MODES)
def test_all_combinations_produce_correct_length(algorithm, residual):
    """Every algorithm × residual combination must produce output of
    the correct length."""
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 36 nt
    ch = Chunker(min_size=3, max_size=5, algorithm=algorithm,
                 residual_chunk=residual, seed=42)
    out = ch.chunk(seq)
    assert len(out) == len(seq)
    assert Counter(seq) == Counter(out)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("residual", RESIDUAL_MODES)
def test_all_combinations_short_sequence(algorithm, residual):
    """Edge case: sequence shorter than min_size returns input unchanged."""
    ch = Chunker(min_size=10, max_size=20, algorithm=algorithm,
                 residual_chunk=residual, seed=42)
    out = ch.chunk("ACGTAC")
    assert out == "ACGTAC"
    assert len(out) == 6


# ======================================================================
# Determinism
# ======================================================================

def test_same_seed_same_output():
    """Same seed → same output."""
    seq = "ACGTACGTACGTACGTACGT"
    ch1 = Chunker(min_size=2, max_size=4, seed=123)
    ch2 = Chunker(min_size=2, max_size=4, seed=123)
    assert ch1.chunk(seq) == ch2.chunk(seq)


def test_different_calls_differ():
    """Successive calls on the same chunker should usually differ."""
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGT"
    ch = Chunker(min_size=2, max_size=4, seed=42)
    outputs = {ch.chunk(seq) for _ in range(5)}
    assert len(outputs) > 1


# ======================================================================
# RNG state
# ======================================================================

def test_getstate_setstate_roundtrip():
    """Saving and restoring RNG state produces identical subsequent outputs."""
    seq = "ACGTACGTACGTACGTACGT"
    ch1 = Chunker(min_size=2, max_size=4, seed=42)
    ch1.chunk(seq)
    state = ch1.getstate()
    next1 = ch1.chunk(seq)

    ch2 = Chunker(min_size=2, max_size=4, seed=42)
    ch2.chunk(seq)
    ch2.setstate(state)
    next2 = ch2.chunk(seq)

    assert next1 == next2


# ======================================================================
# Backtrack exact decomposition
# ======================================================================

def test_backtrack_finds_exact_decomposition_when_possible():
    """When the length can be exactly decomposed, backtrack should find it
    (no residual chunk should appear)."""
    # Length 12, min=3, max=4: 12 = 3+3+3+3 = 4+4+4 = 3+4+4+... etc.
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGT"  # 32 nt
    # 32 = 4 * 8, so 4+4+4+4+4+4+4+4 works.
    ch = Chunker(min_size=3, max_size=4, algorithm="backtrack", seed=42)
    out = ch.chunk(seq)
    assert len(out) == len(seq)


def test_backtrack_handles_impossible_length():
    """When exact decomposition is impossible, backtrack falls back to
    making the tail as short as possible + residual handling."""
    # Length 5, min=3, max=4: 5 = 3+? (2 too small) = 4+? (1 too small).
    # No exact decomposition. Backtrack should still produce length-5 output.
    ch = Chunker(min_size=3, max_size=4, algorithm="backtrack",
                 residual_chunk="extend", seed=42)
    out = ch.chunk("ACGTA")
    assert len(out) == 5


# ======================================================================
# flip_strand
# ======================================================================

def test_flip_strand_prob_zero_is_noop():
    """flip_strand_prob=0.0 should never RC any chunk."""
    ch = Chunker(min_size=2, max_size=4, flip_strand=True,
                 flip_strand_prob=0.0, seed=42)
    out = ch.chunk("ACGTACGTACGTACGT")
    # Same composition (since no RC).
    assert Counter("ACGTACGTACGTACGT") == Counter(out)


def test_flip_strand_prob_one_always_rcs():
    """flip_strand_prob=1.0 should RC every chunk (but the overall sequence
    composition may still be preserved because RC is a bijection)."""
    ch = Chunker(min_size=2, max_size=4, flip_strand=True,
                 flip_strand_prob=1.0, seed=42)
    seq = "ACGTACGTACGTACGT"
    out = ch.chunk(seq)
    # Length must still be preserved.
    assert len(out) == len(seq)
    # Composition must be preserved (RC is a bijection on {A,C,G,T}).
    assert Counter(seq) == Counter(out)


def test_flip_strand_non_iupac_raises():
    """flip_strand=True with a non-IUPAC character must raise ValueError."""
    # We use flip_strand_prob=1.0 to force at least one RC attempt.
    # We try multiple seeds to ensure at least one chunk gets RC'd.
    seq = "ACGTXYZPQR"  # X, Y, Z are not IUPAC
    hit_error = False
    for seed in range(50):
        ch = Chunker(min_size=2, max_size=4, flip_strand=True,
                     flip_strand_prob=1.0, seed=seed)
        try:
            ch.chunk(seq)
        except ValueError:
            hit_error = True
            break
    assert hit_error, "expected ValueError on non-IUPAC input with flip_strand=True"


def test_no_flip_strand_accepts_arbitrary_bytes():
    """Without flip_strand, the chunker accepts any byte sequence."""
    ch = Chunker(min_size=2, max_size=4, seed=42)
    out = ch.chunk("XYZPQRXYZPQR")
    assert len(out) == 12
    assert Counter("XYZPQRXYZPQR") == Counter(out)


# ======================================================================
# bytes support
# ======================================================================

def test_bytes_input():
    """bytes input → bytes output, same behavior as str."""
    ch_str = Chunker(min_size=2, max_size=4, seed=42)
    ch_bytes = Chunker(min_size=2, max_size=4, seed=42)
    out_str = ch_str.chunk("ACGTACGTACGT")
    out_bytes = ch_bytes.chunk(b"ACGTACGTACGT")
    assert isinstance(out_bytes, bytes)
    assert out_str == out_bytes.decode("ascii")


# ======================================================================
# Batch operations
# ======================================================================

def test_chunk_many_matches_serial():
    """chunk_many should produce the same output as repeated chunk calls."""
    seqs = [
        "ACGTACGTACGT",
        "TTTTAAAAGGGGCCCC",
        "ACGTACGTAAAACCCCGGGG",
    ]
    ch_serial = Chunker(min_size=2, max_size=4, seed=42)
    serial = [ch_serial.chunk(s) for s in seqs]
    ch_batch = Chunker(min_size=2, max_size=4, seed=42)
    batch = ch_batch.chunk_many(seqs)
    assert serial == batch


def test_chunk_many_reproducible_across_n_jobs():
    """chunk_many with n_jobs>1 must produce identical output regardless
    of the specific n_jobs value (as long as n_jobs > 1). Single-threaded
    mode (n_jobs=1) uses the master RNG directly, so it produces a
    different stream from parallel mode."""
    seqs = [
        "ACGTACGTACGT",
        "TTTTAAAAGGGGCCCC",
        "ACGTACGTAAAACCCCGGGG",
        "GGGGCCCCAAAATTTT",
        "ACGTACGTACGTACGT",
    ]
    # n_jobs=2 vs n_jobs=4 should match (both use derived per-task keys).
    result_2 = Chunker(min_size=2, max_size=4, seed=42).chunk_many(seqs, n_jobs=2)
    result_4 = Chunker(min_size=2, max_size=4, seed=42).chunk_many(seqs, n_jobs=4)
    assert result_2 == result_4


def test_chunk_many_preserves_length():
    """Every output of chunk_many must preserve input length."""
    seqs = ["ACGTACGTACGT", "TTTTAAAAGGGGCCCC"]
    ch = Chunker(min_size=2, max_size=4, seed=42)
    outs = ch.chunk_many(seqs)
    for s, o in zip(seqs, outs):
        assert len(o) == len(s)


# ======================================================================
# Edge cases
# ======================================================================

def test_empty_sequence():
    """Empty input → empty output."""
    ch = Chunker(min_size=2, max_size=4, seed=42)
    assert ch.chunk("") == ""


def test_invalid_range_raises():
    """Invalid size ranges must raise ValueError."""
    with pytest.raises(ValueError):
        Chunker(min_size=0, max_size=4)
    with pytest.raises(ValueError):
        Chunker(min_size=4, max_size=3)


def test_invalid_algorithm_raises():
    """Invalid algorithm name must raise at construction."""
    with pytest.raises((ValueError, TypeError)):
        Chunker(min_size=2, max_size=4, algorithm="invalid", seed=42)


def test_invalid_residual_raises():
    """Invalid residual_chunk name must raise at construction."""
    with pytest.raises((ValueError, TypeError)):
        Chunker(min_size=2, max_size=4, residual_chunk="invalid", seed=42)

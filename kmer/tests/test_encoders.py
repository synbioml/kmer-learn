"""Tests for the SpectrumEncoder and GappyEncoder."""

import pytest
import numpy as np
from collections import Counter
import scipy.sparse as sp

from kmer.encoders import SpectrumEncoder, GappyEncoder
from kmer.utils import kmer_to_code, code_to_kmer, reverse_complement


# ======================================================================
# utils.seq helpers
# ======================================================================

def test_kmer_to_code_roundtrip():
    for kmer in ["A", "C", "G", "T", "AC", "ACGT", "AAAACCCCGGGGTTTT"]:
        k = len(kmer)
        code = kmer_to_code(kmer)
        assert code_to_kmer(code, k) == kmer


def test_kmer_to_code_specific_values():
    assert kmer_to_code("A") == 0
    assert kmer_to_code("C") == 1
    assert kmer_to_code("G") == 2
    assert kmer_to_code("T") == 3
    assert kmer_to_code("AC") == 1   # 0b0001 = 1
    assert kmer_to_code("ACGT") == 0b00011011  # = 27


def test_kmer_to_code_rejects_lowercase():
    with pytest.raises(ValueError):
        kmer_to_code("acgt")


def test_kmer_to_code_rejects_non_acgt():
    with pytest.raises(ValueError):
        kmer_to_code("ACGTN")


def test_reverse_complement_basic():
    assert reverse_complement("ACGT") == "ACGT"  # palindrome
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("ACGTACGT") == "ACGTACGT"


def test_reverse_complement_bytes():
    assert reverse_complement(b"AAAA") == b"TTTT"
    assert reverse_complement(b"ACGT") == b"ACGT"


def test_reverse_complement_iupac():
    assert reverse_complement("R") == "Y"
    assert reverse_complement("N") == "N"
    assert reverse_complement("RY") == "RY"


def test_reverse_complement_rejects_lowercase():
    with pytest.raises(ValueError):
        reverse_complement("acgt")


def test_reverse_complement_rejects_u():
    with pytest.raises(ValueError):
        reverse_complement("ACGU")


def test_reverse_complement_rejects_invalid():
    with pytest.raises(ValueError):
        reverse_complement("ACGTX")


# ======================================================================
# SpectrumEncoder — basic API
# ======================================================================

def test_spectrum_shape():
    enc = SpectrumEncoder(k=3)
    X = enc.fit_transform(["ACGTACGT", "TTTTAAAA"])
    assert X.shape == (2, 64)


def test_spectrum_returns_csr():
    enc = SpectrumEncoder(k=3)
    X = enc.fit_transform(["ACGTACGT"])
    assert sp.issparse(X)
    assert X.format == "csr"


def test_spectrum_counts_match_reference():
    """Counts from the encoder must match a manual Counter."""
    seqs = ["ACGTACGT", "TTTTAAAACCCCGGGG", "AAAACGTAAA"]
    k = 3
    enc = SpectrumEncoder(k=k)
    X = enc.fit_transform(seqs)
    names = enc.get_feature_names_out()

    for i, seq in enumerate(seqs):
        manual = Counter(seq[j:j+k] for j in range(len(seq) - k + 1))
        row = X.getrow(i).toarray().ravel()
        for kmer, count in manual.items():
            col = list(names).index(kmer)
            assert row[col] == count, f"seq {i} kmer {kmer}: {row[col]} != {count}"


def test_spectrum_total_counts():
    """Total count per sequence = len(seq) - k + 1."""
    seqs = ["ACGTACGT", "TTTTAAAA", "ACGTACGTACGTACGT"]
    k = 3
    enc = SpectrumEncoder(k=k)
    X = enc.fit_transform(seqs)
    totals = np.asarray(X.sum(axis=1)).ravel()
    for i, seq in enumerate(seqs):
        expected = max(0, len(seq) - k + 1)
        assert totals[i] == expected, f"seq {i}: {totals[i]} != {expected}"


def test_spectrum_short_sequence():
    """Sequences shorter than k produce a zero row."""
    enc = SpectrumEncoder(k=6)
    X = enc.fit_transform(["ACG", "ACGTACGTACGT"])
    assert X.shape == (2, 4**6)
    assert X.getrow(0).nnz == 0  # no k-mers


def test_spectrum_feature_names():
    enc = SpectrumEncoder(k=2)
    names = enc.get_feature_names_out()
    assert len(names) == 16
    assert "AA" in names
    assert "TT" in names
    assert "AC" in names


def test_spectrum_invalid_k_raises():
    with pytest.raises(ValueError):
        SpectrumEncoder(k=0)
    with pytest.raises(ValueError):
        SpectrumEncoder(k=13)  # k > 12 exceeds memory limit


def test_spectrum_non_acgt_raises():
    enc = SpectrumEncoder(k=3)
    with pytest.raises(ValueError):
        enc.fit_transform(["ACGTN"])


def test_spectrum_bytes_input():
    """Bytes input should work and produce the same result as str."""
    enc_str = SpectrumEncoder(k=3)
    enc_bytes = SpectrumEncoder(k=3)
    X_str = enc_str.fit_transform(["ACGTACGT"])
    X_bytes = enc_bytes.fit_transform([b"ACGTACGT"])
    assert np.allclose(X_str.toarray(), X_bytes.toarray())


# ======================================================================
# SpectrumEncoder — canonical_rc
# ======================================================================

def test_spectrum_canonical_rc_collapses_pairs():
    """With canonical_rc=True, a k-mer and its RC share a column."""
    enc = SpectrumEncoder(k=4, canonical_rc=True)
    X = enc.fit_transform(["AAAATTTT"])  # contains AAAA and TTTT (= RC(AAAA))
    # Without canonical_rc, AAAA and TTTT would be in separate columns.
    # With canonical_rc, both contribute to the canonical column.
    enc_no_rc = SpectrumEncoder(k=4, canonical_rc=False)
    X_no_rc = enc_no_rc.fit_transform(["AAAATTTT"])
    # The canonical version should have fewer non-zero columns.
    assert X.nnz <= X_no_rc.nnz


def test_spectrum_rc_sequences_have_same_features():
    """A sequence and its RC should produce identical feature vectors
    when canonical_rc=True."""
    enc = SpectrumEncoder(k=4, canonical_rc=True)
    enc.fit(["ACGTACGTACGT"])  # populate feature names
    X1 = enc.transform(["ACGTACGTACGT"])
    X2 = enc.transform([reverse_complement("ACGTACGTACGT")])
    # As canonical vectors, they should be identical.
    diff = (X1 - X2).toarray()
    assert np.allclose(diff, 0), f"RC sequences differ: max diff {np.abs(diff).max()}"


def test_spectrum_canonical_rc_no_double_counting():
    """canonical_rc=True must NOT double counts. Total = len-k+1, same
    as without canonical_rc. Only the column assignment changes (k-mer
    and its RC share a column)."""
    enc_no_rc = SpectrumEncoder(k=4, canonical_rc=False)
    enc_rc = SpectrumEncoder(k=4, canonical_rc=True)
    seq = "GCGCAATTGCGCAAAAGCGC"
    X_no = enc_no_rc.fit_transform([seq])
    X_rc = enc_rc.fit_transform([seq])
    # Total counts must be equal (both = len-k+1 = 17-4+1 = 14).
    assert X_no.sum() == X_rc.sum() == 17, (
        f"no_rc={X_no.sum()}, rc={X_rc.sum()}, expected 17"
    )


def test_spectrum_canonical_rc_collapses_pairs():
    """AAAA and TTTT should share a canonical column."""
    enc_rc = SpectrumEncoder(k=4, canonical_rc=True)
    seq = "AAAATTTT"  # AAAA at pos 0, TTTT at pos 4
    X = enc_rc.fit_transform([seq])
    aaaa = list(enc_rc.get_feature_names_out()).index("AAAA")
    # Both AAAA and TTTT map to canonical AAAA, so count = 2.
    assert X[0, aaaa] == 2


def test_spectrum_palindrome_not_doubled():
    """AATT (palindrome) should NOT be counted twice with canonical_rc.
    It appears once on the forward strand, so count = 1."""
    enc_rc = SpectrumEncoder(k=4, canonical_rc=True)
    seq = "GCGCAATTGCGC"
    X = enc_rc.fit_transform([seq])
    aatt = list(enc_rc.get_feature_names_out()).index("AATT")
    assert X[0, aatt] == 1, f"palindrome should not be doubled, got {X[0, aatt]}"


# ======================================================================
# GappyEncoder — explicit masks
# ======================================================================

def test_gappy_explicit_mask_basic():
    """Single mask, single sequence."""
    enc = GappyEncoder(masks=["*--*"])
    X = enc.fit_transform(["ACGTACGT"])
    # Mask *--* matches positions 0..3 in 'ACGTACGT':
    # pos 0: AC, pos 1: CG, pos 2: GT, pos 3: TA, pos 4: AC
    # So AC count=2, others=1. Total patterns observed = 4 distinct.
    assert X.shape == (1, 16)  # 4^2 patterns for a 2-concrete mask
    assert X.sum() == 5  # 5 windows


def test_gappy_multiple_masks():
    enc = GappyEncoder(masks=["*--*", "*---*"])
    X = enc.fit_transform(["ACGTACGTACGT"])
    # 4^2 + 4^2 = 32 features
    assert X.shape[1] == 32


def test_gappy_mask_too_long_raises():
    with pytest.raises(ValueError):
        GappyEncoder(masks=["*" + "-" * 50])


def test_gappy_no_concrete_raises():
    with pytest.raises(ValueError):
        GappyEncoder(masks=["----"])


def test_gappy_invalid_char_in_mask_raises():
    with pytest.raises(ValueError):
        GappyEncoder(masks=["X--*"])


# ======================================================================
# GappyEncoder — gap range
# ======================================================================

def test_gappy_gap_range_generates_correct_masks():
    enc = GappyEncoder(L=4, g_min=1, g_max=2)
    # C(2,1) + C(2,2) = 2 + 1 = 3 masks (no leading/trailing gaps)
    assert len(enc.masks_) == 3  # C(L-2,g): C(2,1)+C(2,2)
    # Each mask should be length 4 with at least one '*'.
    for m in enc.masks_:
        assert len(m) == 4
        assert "*" in m


def test_gappy_gap_range_count():
    """Verify counts match a manual computation for a small case."""
    seq = "ACGTACGT"
    enc = GappyEncoder(L=4, g_min=1, g_max=1)  # only 1 gap
    X = enc.fit_transform([seq])
    # C(2,1)=2 masks with 1 gap, each producing 4^3=64 patterns = 128 features
    assert X.shape == (1, 2 * 64)  # 2 masks × 64 patterns


def test_gappy_requires_masks_or_range():
    with pytest.raises(ValueError):
        GappyEncoder()


def test_gappy_g_max_optional():
    """g_max can be None; generates all masks with >= g_min gaps."""
    enc = GappyEncoder(L=4, g_min=1)
    # L=4, g_min=1, g_max=None → g in {1, 2}: C(2,1)+C(2,2) = 2+1 = 3 masks
    assert len(enc.masks_) == 3


def test_gappy_rejects_leading_trailing_gaps():
    """Masks with leading or trailing gaps should raise."""
    with pytest.raises(ValueError, match="leading/trailing"):
        GappyEncoder(masks=["-*-*"])
    with pytest.raises(ValueError, match="leading/trailing"):
        GappyEncoder(masks=["*-*-"])
    # Valid masks should not raise
    GappyEncoder(masks=["*-*", "*--*", "**-*"])


def test_gappy_bytes_input():
    """Bytes masks should work too."""
    enc_str = GappyEncoder(masks=["*--*"])
    enc_bytes = GappyEncoder(masks=[b"*--*"])
    X1 = enc_str.fit_transform(["ACGTACGT"])
    X2 = enc_bytes.fit_transform(["ACGTACGT"])
    assert np.allclose(X1.toarray(), X2.toarray())


def test_gappy_non_acgt_raises():
    enc = GappyEncoder(masks=["*--*"])
    with pytest.raises(ValueError):
        enc.fit_transform(["ACGTNACGT"])


# ======================================================================
# GappyEncoder — determinism (column order)
# ======================================================================

def test_gappy_column_order_deterministic_across_calls():
    """Column indices must be deterministic: transform() on the same
    sequences with the same encoder must produce identical matrices
    regardless of which sequences were used in fit_transform()."""
    enc = GappyEncoder(masks=["*--*"])
    # Use a "rich" sequence in fit_transform to populate columns.
    enc.fit_transform(["ACGTACGTACGTACGT"])
    # Now transform two different sequences.
    X1 = enc.transform(["ACGTACGT"])
    # Reset and use a different fit sequence.
    enc2 = GappyEncoder(masks=["*--*"])
    enc2.fit_transform(["TTTTAAAACCCCGGGG"])
    X2 = enc2.transform(["ACGTACGT"])
    # Same input → same output (columns are deterministic, ordered by code).
    assert X1.shape == X2.shape
    import numpy as np
    assert np.allclose(X1.toarray(), X2.toarray())


def test_gappy_feature_names_match_column_order():
    """Feature names must correspond to the columns of the output matrix."""
    from collections import Counter
    enc = GappyEncoder(masks=["*--*"])
    seq = "ACGTACGT"
    X = enc.fit_transform([seq])
    names = enc.get_feature_names_out()
    assert len(names) == X.shape[1]
    # Manually compute gappy k-mer counts for "*--*" applied to seq.
    # The mask "*--*" has length 4, with concrete positions [0, 3].
    # At position p, the pattern is (seq[p], seq[p+3]).
    manual = Counter()
    for p in range(len(seq) - 4 + 1):
        pattern = seq[p] + seq[p + 3]
        manual[pattern] += 1
    # Compare with the matrix.
    row = X.getrow(0).toarray().ravel()
    for i, name in enumerate(names):
        # name format: "<mask>_<pattern>", e.g. "*--*_AC"
        pattern = name.split("_")[1]
        expected = manual.get(pattern, 0)
        assert row[i] == expected, (
            f"col {i} ({name}): got {row[i]}, expected {expected}"
        )

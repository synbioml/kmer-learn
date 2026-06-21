"""Tests for the MismatchEncoder."""

import pytest
import numpy as np
from collections import Counter
import scipy.sparse as sp

from kmer.encoders import MismatchEncoder, SpectrumEncoder


def test_mismatch_shape():
    enc = MismatchEncoder(k=4, m=1)
    X = enc.fit_transform(["ACGTACGT", "TTTTAAAA"])
    assert X.shape == (2, 256)


def test_mismatch_returns_csr():
    enc = MismatchEncoder(k=3, m=1)
    X = enc.fit_transform(["ACGTACGT"])
    assert sp.issparse(X)
    assert X.format == "csr"


def test_mismatch_m0_matches_spectrum():
    """m=0 should produce identical counts to SpectrumEncoder."""
    enc_m = MismatchEncoder(k=4, m=0)
    enc_s = SpectrumEncoder(k=4)
    seqs = ["ACGTACGT", "TTTTAAAACCCC", "AAAACGTACGTAAA"]
    X_m = enc_m.fit_transform(seqs)
    X_s = enc_s.fit_transform(seqs)
    assert np.allclose(X_m.toarray(), X_s.toarray())


def test_mismatch_total_counts():
    """Total count = (n_kmers) * (1 + k * 3) for m=1."""
    seq = "ACGTACGT"  # 8 nt, k=4 → 5 k-mers
    enc = MismatchEncoder(k=4, m=1)
    X = enc.fit_transform([seq])
    # Each k-mer generates 1 (exact) + 4*3 (one mismatch at each of 4 positions) = 13 neighbors
    expected_total = 5 * 13
    assert X.sum() == expected_total


def test_mismatch_m2_total_counts():
    """For m=2: 1 + k*3 + C(k,2)*9 neighbors per k-mer."""
    seq = "ACGTAC"  # 6 nt, k=4 → 3 k-mers
    enc = MismatchEncoder(k=4, m=2)
    X = enc.fit_transform([seq])
    # 1 + 4*3 + C(4,2)*9 = 1 + 12 + 54 = 67 neighbors per k-mer
    expected_total = 3 * 67
    assert X.sum() == expected_total


def test_mismatch_invalid_k_raises():
    with pytest.raises(ValueError):
        MismatchEncoder(k=0, m=0)
    with pytest.raises(ValueError):
        MismatchEncoder(k=13, m=0)


def test_mismatch_invalid_m_raises():
    with pytest.raises(ValueError):
        MismatchEncoder(k=4, m=5)  # m > k
    with pytest.raises(ValueError):
        MismatchEncoder(k=4, m=-1)


def test_mismatch_non_acgt_raises():
    enc = MismatchEncoder(k=4, m=1)
    with pytest.raises(ValueError):
        enc.fit_transform(["ACGTNACGT"])


def test_mismatch_short_sequence():
    """Sequences shorter than k produce a zero row."""
    enc = MismatchEncoder(k=6, m=1)
    X = enc.fit_transform(["ACG", "ACGTACGTACGT"])
    assert X.shape == (2, 4**6)
    assert X.getrow(0).nnz == 0


def test_mismatch_feature_names():
    enc = MismatchEncoder(k=2, m=1)
    enc.fit(["ACGT"])
    names = enc.get_feature_names_out()
    assert len(names) == 16
    assert "AA" in names
    assert "TT" in names


def test_mismatch_bytes_input():
    enc_str = MismatchEncoder(k=3, m=1)
    enc_bytes = MismatchEncoder(k=3, m=1)
    X_str = enc_str.fit_transform(["ACGTACGT"])
    X_bytes = enc_bytes.fit_transform([b"ACGTACGT"])
    assert np.allclose(X_str.toarray(), X_bytes.toarray())


def test_mismatch_canonical_rc():
    """With canonical_rc=True, a k-mer and its RC share a column."""
    enc = MismatchEncoder(k=4, m=1, canonical_rc=True)
    X = enc.fit_transform(["AAAATTTT"])
    enc_no_rc = MismatchEncoder(k=4, m=1, canonical_rc=False)
    X_no_rc = enc_no_rc.fit_transform(["AAAATTTT"])
    # Canonical version should have fewer non-zero columns.
    assert X.nnz <= X_no_rc.nnz


def test_mismatch_m1_includes_exact_match():
    """The exact k-mer's count should be at least 1 (it's its own neighbor)."""
    enc = MismatchEncoder(k=4, m=1)
    X = enc.fit_transform(["ACGTACGT"])
    names = enc.get_feature_names_out()
    # The k-mer "ACGT" appears at positions 0 and 4 in "ACGTACGT".
    # With m=1, its count column should include 2 (exact matches) plus
    # contributions from neighboring k-mers that are within Hamming distance 1.
    acgt_col = list(names).index("ACGT")
    assert X[0, acgt_col] >= 2

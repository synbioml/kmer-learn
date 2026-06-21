"""Tests for the kmer.distance module."""

import pytest
import numpy as np
from scipy.spatial.distance import squareform

from kmer.distance import (
    BaseDistance,
    DistanceKernel,
    Hamming,
    Levenshtein,
    NeedlemanWunsch,
    SmithWaterman,
)


# ======================================================================
# BaseDistance ABC
# ======================================================================

def test_base_distance_is_abstract():
    with pytest.raises(TypeError):
        BaseDistance()


def test_custom_distance_subclass():
    """A trivial BaseDistance subclass works."""
    class LengthDiff(BaseDistance):
        def _pair(self, a, b):
            return abs(len(a) - len(b))
    d = LengthDiff()
    assert d("ACGT", "ACGTA") == 1
    assert d("ACGT", "ACGT") == 0


def test_cdist_shape():
    d = Hamming()
    X = ["ACGT", "AAAA", "ACGT"]
    M = d.cdist(X)
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 0)


def test_cdist_cross_shape():
    d = Hamming()
    X = ["ACGT", "AAAA"]
    Y = ["ACGA", "TTTT"]
    M = d.cdist(X, Y)
    assert M.shape == (2, 2)


def test_pdist_condensed_format():
    d = Hamming()
    X = ["ACGT", "ACGA", "AAAA"]
    p = d.pdist(X)
    # 3 sequences → 3 pairwise distances
    assert p.shape == (3,)
    # Should be convertible to a square matrix via scipy.
    S = squareform(p)
    assert S.shape == (3, 3)
    assert S[0, 1] == p[0]


def test_bytes_input():
    d = Hamming()
    assert d(b"ACGT", b"ACGA") == 1


def test_mixed_str_bytes():
    d = Hamming()
    assert d("ACGT", b"ACGA") == 1


# ======================================================================
# Hamming
# ======================================================================

class TestHamming:
    def test_identical(self):
        assert Hamming()("ACGT", "ACGT") == 0

    def test_single_diff(self):
        assert Hamming()("ACGT", "ACGA") == 1

    def test_all_diff(self):
        assert Hamming()("ACGT", "TGCA") == 4

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            Hamming()("ACGT", "ACG")

    def test_returns_float(self):
        assert isinstance(Hamming()("ACGT", "ACGA"), float)


# ======================================================================
# Levenshtein
# ======================================================================

class TestLevenshtein:
    def test_identical(self):
        assert Levenshtein()("ACGT", "ACGT") == 0.0

    def test_one_insertion(self):
        assert Levenshtein()("ACGT", "ACGTA") == 1.0

    def test_one_deletion(self):
        assert Levenshtein()("ACGTA", "ACGT") == 1.0

    def test_one_substitution(self):
        assert Levenshtein()("ACGT", "ACGA") == 1.0

    def test_completely_different(self):
        assert Levenshtein()("AAAA", "TTTT") == 4.0

    def test_empty_string(self):
        assert Levenshtein()("", "ACGT") == 4.0
        assert Levenshtein()("ACGT", "") == 4.0
        assert Levenshtein()("", "") == 0.0

    def test_symmetric(self):
        d = Levenshtein()
        for a, b in [("ACGT", "ACGTA"), ("AAAA", "TTTT"), ("ACGTACGT", "ACGTTCGA")]:
            assert d(a, b) == d(b, a)

    def test_triangle_inequality(self):
        """Levenshtein is a metric: d(a,c) <= d(a,b) + d(b,c)."""
        d = Levenshtein()
        for a, b, c in [("ACGT", "ACGTA", "ACGTAC"), ("AAAA", "AATA", "TTTT")]:
            assert d(a, c) <= d(a, b) + d(b, c) + 1e-9

    def test_matches_pure_python(self):
        """If rapidfuzz is available, its results must match pure Python."""
        import importlib.util
        if not importlib.util.find_spec("rapidfuzz"):
            pytest.skip("rapidfuzz not installed")
        # Force the pure-Python path.
        from kmer.distance.edit import _levenshtein_python
        for a, b in [("ACGT", "ACGTA"), ("AAAA", "TTTT"), ("ACGTACGT", "ACGTTCGA")]:
            assert _levenshtein_python(a, b) == int(Levenshtein()(a, b))


# ======================================================================
# NeedlemanWunsch
# ======================================================================

class TestNeedlemanWunsch:
    def test_identical(self):
        nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
        assert nw("ACGT", "ACGT") == 4.0

    def test_one_gap(self):
        nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
        # "ACGT" vs "ACG-T": 3 matches + 1 gap = 2
        assert nw("ACGT", "ACGT"[:-1] + "T") == 4.0  # same
        assert nw("ACGT", "ACG") == 2.0  # 3 matches - 1 gap

    def test_pure_mismatch(self):
        nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
        # All positions mismatched.
        assert nw("AAAA", "TTTT") == -4.0

    def test_substitution_matrix_nuc44(self):
        nw = NeedlemanWunsch(substitution_matrix="NUC4.4", gap=-4)
        # ACGT vs ACGT: 4 matches × 5 = 20
        assert nw("ACGT", "ACGT") == 20.0

    def test_as_distance(self):
        nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1, as_distance=True)
        # Max score = 4 * 1 = 4. Actual = 4. Distance = 0.
        assert nw("ACGT", "ACGT") == 0.0
        # Actual score = 2 (3 matches - 1 gap). Distance = 4 - 2 = 2.
        assert nw("ACGT", "ACG") == 2.0

    def test_is_similarity(self):
        """Raw NW is a similarity (is_distance = False)."""
        assert NeedlemanWunsch().is_distance is False


# ======================================================================
# SmithWaterman
# ======================================================================

class TestSmithWaterman:
    def test_identical(self):
        sw = SmithWaterman(match=1, mismatch=-1, gap=-1)
        assert sw("ACGT", "ACGT") == 4.0

    def test_local_alignment(self):
        sw = SmithWaterman(match=1, mismatch=-1, gap=-1)
        # "ACGT" is a local match inside "TTACGTTT".
        assert sw("ACGT", "TTACGTTT") == 4.0

    def test_no_match(self):
        sw = SmithWaterman(match=1, mismatch=-1, gap=-1)
        # No positive-scoring local alignment.
        assert sw("AAAA", "TTTT") == 0.0

    def test_substitution_matrix(self):
        sw = SmithWaterman(substitution_matrix="NUC4.4", gap=-4)
        # ACGT vs ACGT: 4 matches × 5 = 20
        assert sw("ACGT", "ACGT") == 20.0

    def test_is_similarity(self):
        assert SmithWaterman().is_distance is False


# ======================================================================
# DistanceKernel
# ======================================================================

class TestDistanceKernel:
    def test_rbf_transform(self):
        kern = DistanceKernel(Hamming(), transform="rbf", gamma=1.0)
        # Identical: K = exp(0) = 1
        assert np.isclose(kern("ACGT", "ACGT"), 1.0)
        # One mismatch: K = exp(-1)
        assert np.isclose(kern("ACGT", "ACGA"), np.exp(-1))

    def test_linear_transform(self):
        kern = DistanceKernel(Hamming(), transform="linear")
        # K = -d
        assert kern("ACGT", "ACGT") == 0.0
        assert kern("ACGT", "ACGA") == -1.0

    def test_laplacian_transform(self):
        kern = DistanceKernel(Levenshtein(), transform="laplacian", gamma=0.5)
        # K = exp(-0.5 * d)
        assert np.isclose(kern("ACGT", "ACGTA"), np.exp(-0.5))

    def test_cauchy_transform(self):
        kern = DistanceKernel(Hamming(), transform="cauchy", gamma=1.0)
        # K = 1 / (1 + d^2)
        assert np.isclose(kern("ACGT", "ACGT"), 1.0)
        assert np.isclose(kern("ACGT", "ACGA"), 1.0 / 2.0)

    def test_exponential_clipped_transform(self):
        kern = DistanceKernel(Hamming(), transform="exponential_clipped",
                              gamma=1.0, radius=2.0)
        # d=1 < radius=2 → max(0, 1-2) = 0 → exp(0) = 1
        assert np.isclose(kern("ACGT", "ACGA"), 1.0)
        # d=4 > radius=2 → max(0, 4-2) = 2 → exp(-2)
        assert np.isclose(kern("ACGT", "TGCA"), np.exp(-2))

    def test_gram_matrix_symmetric(self):
        kern = DistanceKernel(Levenshtein(), transform="rbf", gamma=0.1)
        seqs = ["ACGT", "ACGTA", "TTTT", "ACGTACGT"]
        K = kern.kernel_matrix(seqs)
        assert K.shape == (4, 4)
        assert np.allclose(K, K.T)

    def test_gram_diagonal_one_for_rbf(self):
        """For RBF, K(a, a) = exp(0) = 1."""
        kern = DistanceKernel(Hamming(), transform="rbf", gamma=1.0)
        K = kern.kernel_matrix(["ACGT", "AAAA", "ACGT"])
        assert np.allclose(np.diag(K), 1.0)

    def test_gram_psd_for_rbf(self):
        """The RBF Gram matrix must be positive semi-definite."""
        kern = DistanceKernel(Levenshtein(), transform="rbf", gamma=0.1)
        seqs = ["ACGT", "ACGTA", "TTTT", "ACGTACGT", "AAAATTTT"]
        K = kern.kernel_matrix(seqs)
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() >= -1e-10, (
            f"RBF kernel not PSD; min eigenvalue = {eigvals.min()}"
        )

    def test_gram_cross(self):
        kern = DistanceKernel(Hamming(), transform="rbf", gamma=1.0)
        X = ["ACGT", "AAAA"]
        Y = ["ACGA", "TTTT", "ACGT"]
        K = kern.kernel_matrix(X, Y)
        assert K.shape == (2, 3)

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="transform"):
            DistanceKernel(Hamming(), transform="invalid")

    def test_kernel_matrix_returns_gram(self):
        """DistanceKernel.kernel_matrix returns the kernel matrix."""
        kern = DistanceKernel(Hamming(), transform="rbf", gamma=1.0)
        K = kern.kernel_matrix(["ACGT", "AAAA"], ["ACGT", "AAAA"])
        # Should be all 1s on the diagonal.
        assert np.allclose(np.diag(K), 1.0)

    def test_with_nw_as_distance(self):
        """NW with as_distance=True can be wrapped in a kernel."""
        nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1, as_distance=True)
        kern = DistanceKernel(nw, transform="rbf", gamma=0.1)
        K = kern.kernel_matrix(["ACGT", "ACGTA", "TTTT"])
        assert K.shape == (3, 3)
        assert np.allclose(np.diag(K), 1.0)  # d=0 → exp(0) = 1

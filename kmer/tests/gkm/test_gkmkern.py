"""Unit tests for the gapped_kmer_kernel module (v3 — new class hierarchy).

Run with: `pytest tests/` from the package root.
"""

import os
import sys
import math
import pytest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kmer.kernels import (  # noqa: E402
    GKMKernel, WGKMKernel,
    WindowedGKMKernel, WindowedWGKMKernel,
)
from kmer.kernels._native._gkmkern import Kernel, WeightedKernel, Sequence  # noqa: E402

from brute_force_reference import (  # noqa: E402
    kernel_pair as brute_pair,
    kernel_matrix as brute_matrix,
)

SEQS_6 = [
    "ACGTACGTACGTACGTACGT",
    "TTTTAAAAGGGGCCCCAAAA",
    "ACGTTGCATGCATGCATGCA",
    "CCCCGGGGTTTTAAAACCCC",
    "ATATGCGCATATGCGCATAT",
    "GAATTCGAATTCGAATTCGA",
]

SEQ4 = [
    "ACGTACGTACGTACGTACGT",
    "TTTTAAAAGGGGCCCCAAAA",
    "ACGTTGCATGCATGCATGCA",
    "CCCCGGGGTTTTAAAACCCC",
]

# Same-length sequences for windowed tests
SEQ4_W = [
    "ACGTACGTACGTACGTACGTAC",
    "TTTTAAAAGGGGCCCCAAAATT",
    "ACGTTGCATGCATGCATGCATG",
    "CCCCGGGGTTTTAAAACCCCAG",
]

REF_DIR = os.environ.get(
    "GKM_REF_DIR",
    os.path.join(os.path.dirname(HERE), "data", "gkmsvm_reference"),
)


def _load_ref_matrix(L, k, d):
    path = os.path.join(REF_DIR, f"gkm_L{L}_k{k}_d{d}.txt")
    if not os.path.exists(path):
        pytest.skip(f"reference output not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip("\t")
            if not line:
                continue
            rows.append([float(x) for x in line.split("\t")])
    n = len(rows)
    M = np.zeros((n, n), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            M[i, j] = v
            if i != j:
                M[j, i] = v
    return M


# ==================================================================
# Parameter validation
# ==================================================================

class TestParameterValidation:
    def test_valid_params(self):
        k = GKMKernel(L=6, k=4, d=2)
        assert k.L == 6 and k.k == 4 and k.d == 2

    def test_L_too_small(self):
        with pytest.raises(ValueError):
            GKMKernel(L=1, k=1, d=0)

    def test_L_too_large(self):
        with pytest.raises(ValueError):
            GKMKernel(L=13, k=6, d=3)

    def test_k_greater_than_L(self):
        with pytest.raises(ValueError):
            GKMKernel(L=6, k=7, d=0)

    def test_d_greater_than_L_minus_k(self):
        with pytest.raises(ValueError):
            GKMKernel(L=6, k=4, d=3)

    def test_invalid_kernel_type(self):
        with pytest.raises(ValueError):
            GKMKernel(L=6, k=4, d=2, kernel_type="bogus")

    def test_invalid_transform(self):
        with pytest.raises(ValueError):
            GKMKernel(L=6, k=4, d=2, transform="bogus")

    def test_invalid_weight_kernel(self):
        with pytest.raises(ValueError):
            WGKMKernel(L=6, k=4, d=2, weight_kernel="bogus")

    def test_invalid_weight_peak(self):
        with pytest.raises(ValueError):
            WGKMKernel(L=6, k=4, d=2, weight_peak=0)

    def test_invalid_padding_too_large(self):
        """Padding that makes effective window < L should raise."""
        with pytest.raises(ValueError):
            WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=4, padding=7)

    def test_invalid_padding_tuple_too_large(self):
        with pytest.raises(ValueError):
            WindowedGKMKernel(L=10, k=6, d=3, window=15, shift=5, padding=(6, 6))

    def test_invalid_padding_tuple_wrong_len(self):
        with pytest.raises(ValueError):
            WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=4, padding=(1, 2, 3))

    def test_window_too_small(self):
        with pytest.raises(ValueError):
            WindowedGKMKernel(L=10, k=6, d=3, window=5, shift=2)

    def test_shift_invalid(self):
        with pytest.raises(ValueError):
            WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=0)

    def test_bytes_accepted(self):
        """Bytes and str should both work and produce identical results."""
        k_str = GKMKernel(L=6, k=4, d=2, use_rc=True)
        k_str.set_references(["ACGTACGTACGTACGTACGT", "TTTTAAAAGGGGCCCCAAAA"])
        K_str = np.asarray(k_str.kernel())
        k_bytes = GKMKernel(L=6, k=4, d=2, use_rc=True)
        k_bytes.set_references([b"ACGTACGTACGTACGTACGT", b"TTTTAAAAGGGGCCCCAAAA"])
        K_bytes = np.asarray(k_bytes.kernel())
        np.testing.assert_allclose(K_str, K_bytes, atol=1e-12)


# ==================================================================
# Kernel properties (GKMKernel)
# ==================================================================

class TestGKMKernelProperties:
    @pytest.mark.parametrize("L,k,d", [(6,4,1),(6,4,2),(8,5,2),(10,6,3),(10,6,1)])
    def test_diagonal_is_one(self, L, k, d):
        kern = GKMKernel(L=L, k=k, d=d, kernel_type="truncated", use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(apply_transform=False)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    @pytest.mark.parametrize("L,k,d", [(6,4,2),(10,6,3)])
    def test_symmetric(self, L, k, d):
        kern = GKMKernel(L=L, k=k, d=d, use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(apply_transform=False)
        np.testing.assert_allclose(K, K.T, atol=1e-9)

    def test_use_rc_false_differs_from_true(self):
        k_rc = GKMKernel(L=6, k=4, d=2, use_rc=True)
        k_no = GKMKernel(L=6, k=4, d=2, use_rc=False)
        k_rc.set_references(SEQ4)
        k_no.set_references(SEQ4)
        K_rc = k_rc.kernel(apply_transform=False)
        K_no = k_no.kernel(apply_transform=False)
        assert not np.allclose(K_rc, K_no)


# ==================================================================
# Brute-force comparison (GKMKernel)
# ==================================================================

class TestBruteForce:
    @pytest.mark.parametrize("L,k,d", [(6,4,1),(6,4,2),(8,5,2),(10,6,3),(10,6,1)])
    def test_kernel_matrix_matches_brute_force(self, L, k, d):
        kern = GKMKernel(L=L, k=k, d=d, use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(apply_transform=False)
        K_brute = brute_matrix(SEQS_6, L=L, k=k, d=d)
        np.testing.assert_allclose(K, K_brute, atol=1e-7)


# ==================================================================
# C reference comparison (GKMKernel)
# ==================================================================

class TestCReference:
    @pytest.mark.parametrize("L,k,d", [
        (6,4,1),(6,4,2),(8,4,1),(8,4,2),(8,4,3),
        (8,6,1),(8,6,2),(10,4,1),(10,4,2),(10,4,3),
        (10,6,1),(10,6,2),(10,6,3),
    ])
    def test_kernel_matrix_matches_c_reference(self, L, k, d):
        if d > L - k:
            pytest.skip("d > L-k")
        kern = GKMKernel(L=L, k=k, d=d, kernel_type="full", use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(apply_transform=False)
        K_ref = _load_ref_matrix(L, k, d)
        np.testing.assert_allclose(K, K_ref, atol=1e-6)


# ==================================================================
# Weight scheme
# ==================================================================

class TestWeightScheme:
    def test_full_weights_are_combinatorial(self):
        k = GKMKernel(L=8, k=5, d=2, kernel_type="full")
        assert k._kernel.weight(0) == math.comb(8, 5)
        assert k._kernel.weight(1) == math.comb(7, 5)

    def test_truncated_differs_from_full(self):
        k_full = GKMKernel(L=10, k=6, d=3, kernel_type="full", use_rc=True)
        k_full.set_references(SEQ4)
        K_full = k_full.kernel(apply_transform=False)
        k_trunc = GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True)
        k_trunc.set_references(SEQ4)
        K_trunc = k_trunc.kernel(apply_transform=False)
        assert not np.allclose(K_full, K_trunc)

    def test_set_references_preserves_kernel_type(self):
        for s, sid in [("full", 0), ("truncated", 1), ("estimated_full", 2)]:
            k = GKMKernel(L=6, k=4, d=2, kernel_type=s)
            assert k._kernel.weight_scheme() == sid
            k.set_references(SEQ4)
            assert k._kernel.weight_scheme() == sid


# ==================================================================
# Transforms
# ==================================================================

class TestTransforms:
    @pytest.mark.parametrize("tf", ["rbf", "poly", "sigmoid", "exponential", "exponential_clipped"])
    def test_transforms_produce_valid_kernel(self, tf):
        kern = GKMKernel(L=6, k=4, d=2, kernel_type="truncated",
                         transform=tf, transform_gamma=1.0, transform_radius=0.3)
        kern.set_references(SEQ4)
        K = kern.kernel()
        assert K.shape == (4, 4)
        assert np.all(np.isfinite(K))
        if tf not in ("sigmoid", "exponential"):
            np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    def test_rbf_sharpens_kernel(self):
        k1 = GKMKernel(L=6, k=4, d=2, transform="rbf", transform_gamma=1.0)
        k2 = GKMKernel(L=6, k=4, d=2, transform="rbf", transform_gamma=5.0)
        k1.set_references(SEQ4)
        k2.set_references(SEQ4)
        K1 = k1.kernel()
        K2 = k2.kernel()
        assert np.mean(K1) > np.mean(K2)

    def test_apply_transform_false_returns_raw(self):
        kern = GKMKernel(L=6, k=4, d=2, transform="rbf", transform_gamma=1.0)
        kern.set_references(SEQ4)
        K_raw = kern.kernel(apply_transform=False)
        K_t = kern.kernel(apply_transform=True)
        assert not np.allclose(K_raw, K_t)


# ==================================================================
# WGKMKernel (weighted, full-sequence)
# ==================================================================

class TestWGKMKernel:
    def test_scalar_offset_returns_2d(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=0)
        assert K.shape == (4, 4)

    def test_list_offset_returns_3d(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=[0, 5, 10])
        assert K.shape == (3, 4, 4)

    def test_none_offset_uses_constructor(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        center_offset=3.0)
        wk.set_references(SEQ4)
        K_default = wk.kernel()
        K_explicit = wk.kernel(offsets=3.0)
        np.testing.assert_allclose(K_default, K_explicit)

    def test_diagonal_is_one(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="gaussian",
                        weight_sigma=10, weight_peak=50, use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    def test_symmetric(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="cauchy",
                        weight_sigma=10, weight_peak=50, use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=0)
        np.testing.assert_allclose(K, K.T, atol=1e-9)

    @pytest.mark.parametrize("wk_type", ["triangular", "epanechnikov", "gaussian",
                                           "laplacian", "cauchy"])
    def test_all_spatial_kernels_valid(self, wk_type):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel=wk_type,
                        weight_sigma=10, weight_gamma=0.1,
                        weight_peak=50, use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)
        np.testing.assert_allclose(K, K.T, atol=1e-9)
        assert np.all(np.isfinite(K))

    def test_negative_offset(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        use_rc=True)
        wk.set_references(SEQ4)
        K_left = wk.kernel(offsets=-5)
        K_right = wk.kernel(offsets=5)
        K_center = wk.kernel(offsets=0)
        assert not np.allclose(K_left, K_center)
        assert not np.allclose(K_right, K_center)

    def test_float_offset(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="gaussian",
                        weight_sigma=10, weight_peak=50, use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=2.5)
        assert K.shape == (4, 4)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    def test_slice_offset(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(offsets=slice(0, 10, 3))
        assert K.shape == (4, 4, 4)

    def test_cross_weighted(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(X_query=SEQ4[:2], offsets=0)
        assert K.shape == (2, 4)

    def test_cross_weighted_multiple_offsets(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        use_rc=True)
        wk.set_references(SEQ4)
        K = wk.kernel(X_query=SEQ4[:2], offsets=[0, 5])
        assert K.shape == (2, 2, 4)


# ==================================================================
# WindowedGKMKernel
# ==================================================================

class TestWindowedGKMKernel:
    def test_shape(self):
        kern = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=4)
        kern.set_references(SEQ4_W)
        T = kern.kernel()
        assert T.ndim == 3
        assert T.shape[1] == 4 and T.shape[2] == 4

    def test_diagonal_is_one(self):
        kern = WindowedGKMKernel(L=6, k=4, d=2, use_rc=True, window=12, shift=4)
        kern.set_references(SEQ4_W)
        T = kern.kernel()
        for w in range(T.shape[0]):
            np.testing.assert_allclose(np.diag(T[w]), 1.0, atol=1e-9)

    def test_symmetric(self):
        kern = WindowedGKMKernel(L=6, k=4, d=2, use_rc=True, window=12, shift=4)
        kern.set_references(SEQ4_W)
        T = kern.kernel()
        for w in range(T.shape[0]):
            np.testing.assert_allclose(T[w], T[w].T, atol=1e-9)

    def test_cross_window(self):
        kern = WindowedGKMKernel(L=6, k=4, d=2, use_rc=True, window=12, shift=4)
        kern.set_references(SEQ4_W)
        T = kern.kernel(X_query=SEQ4_W[:2])
        assert T.ndim == 3
        assert T.shape[1] == 2 and T.shape[2] == 4

    def test_padding_increases_window_count(self):
        k1 = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=6, padding=0)
        k1.set_references(SEQ4_W)
        T1 = k1.kernel()
        k2 = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=6, padding=3)
        k2.set_references(SEQ4_W)
        T2 = k2.kernel()
        assert T2.shape[0] >= T1.shape[0]

    def test_negative_padding(self):
        """Negative padding skips edge positions."""
        k1 = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=6, padding=0)
        k1.set_references(SEQ4_W)
        T1 = k1.kernel()
        k2 = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=6, padding=-2)
        k2.set_references(SEQ4_W)
        T2 = k2.kernel()
        # Negative padding makes windows shorter; should still work
        assert T2.ndim == 3

    def test_sliding_query_not_available(self):
        """sliding_window_query_kernel should raise on windowed kernels."""
        kern = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=4)
        kern.set_references(SEQ4_W)
        with pytest.raises(NotImplementedError):
            kern.sliding_window_query_kernel("ACGTACGTACGTACGT", window=12, shift=4)


# ==================================================================
# WindowedWGKMKernel
# ==================================================================

class TestWindowedWGKMKernel:
    def test_shape(self):
        wk = WindowedWGKMKernel(L=6, k=4, d=2, use_rc=True,
                                weight_kernel="gaussian", weight_sigma=5,
                                weight_peak=50, window=12, shift=4)
        wk.set_references(SEQ4_W)
        T = wk.kernel()
        assert T.ndim == 3

    def test_diagonal_is_one(self):
        wk = WindowedWGKMKernel(L=6, k=4, d=2, use_rc=True,
                                weight_kernel="gaussian", weight_sigma=5,
                                weight_peak=50, window=12, shift=4)
        wk.set_references(SEQ4_W)
        T = wk.kernel()
        for w in range(T.shape[0]):
            np.testing.assert_allclose(np.diag(T[w]), 1.0, atol=1e-9)

    def test_symmetric(self):
        wk = WindowedWGKMKernel(L=6, k=4, d=2, use_rc=True,
                                weight_kernel="cauchy", weight_sigma=5,
                                weight_peak=50, window=12, shift=4)
        wk.set_references(SEQ4_W)
        T = wk.kernel()
        for w in range(T.shape[0]):
            np.testing.assert_allclose(T[w], T[w].T, atol=1e-9)

    def test_sliding_query_not_available(self):
        wk = WindowedWGKMKernel(L=6, k=4, d=2, weight_kernel="gaussian",
                                weight_sigma=5, weight_peak=50,
                                window=12, shift=4)
        wk.set_references(SEQ4_W)
        with pytest.raises(NotImplementedError):
            wk.sliding_window_query_kernel("ACGTACGTACGTACGT", window=12, shift=4)


# ==================================================================
# Sliding window query (non-windowed kernels only)
# ==================================================================

class TestSlidingWindowQuery:
    def test_returns_list_for_multiple_queries(self):
        kern = GKMKernel(L=6, k=4, d=2)
        kern.set_references(SEQ4)
        W = kern.sliding_window_query_kernel(
            ["ACGTACGTACGTACGTACGTACGTACGTACGT",
             "ACGTACGTACGTACGTACGT"],
            window=12, shift=6)
        assert isinstance(W, list)
        assert len(W) == 2

    def test_single_query_string(self):
        kern = GKMKernel(L=6, k=4, d=2)
        kern.set_references(SEQ4)
        W = kern.sliding_window_query_kernel(
            "ACGTACGTACGTACGTACGTACGTACGTACGT",
            window=12, shift=6)
        assert isinstance(W, list)
        assert len(W) == 1

    def test_weighted_sliding_query_not_implemented(self):
        """sliding_window_query_kernel for weighted kernels has a known C
        bug with different-length queries/refs. It should raise
        NotImplementedError until fixed."""
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50)
        wk.set_references(SEQ4)
        with pytest.raises(NotImplementedError):
            wk.sliding_window_query_kernel(
                "ACGTACGTACGTACGTACGTACGTACGTACGT",
                window=12, shift=6)


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_short_sequence_raises(self):
        kern = GKMKernel(L=10, k=6, d=3)
        with pytest.raises(ValueError):
            kern.set_references(["ACGT"])

    def test_non_acgt_raises(self):
        kern = GKMKernel(L=6, k=4, d=2)
        with pytest.raises((ValueError, RuntimeError)):
            kern.set_references(["ACGTNACGTACGTACGTACGT"])

    def test_empty_sequence_list(self):
        kern = GKMKernel(L=6, k=4, d=2)
        with pytest.raises((ValueError, RuntimeError)):
            kern.set_references([])

    def test_single_reference(self):
        kern = GKMKernel(L=6, k=4, d=2)
        kern.set_references([SEQ4[0]])
        K = kern.kernel()
        assert K.shape == (1, 1)
        np.testing.assert_allclose(K[0, 0], 1.0, atol=1e-9)

    def test_kernel_not_built_raises(self):
        kern = GKMKernel(L=6, k=4, d=2)
        with pytest.raises(RuntimeError):
            kern.kernel()

    def test_repr(self):
        kern = GKMKernel(L=6, k=4, d=2, kernel_type="truncated", use_rc=True)
        assert "truncated" in repr(kern)
        assert "L=6" in repr(kern)

    def test_repr_with_transform(self):
        kern = GKMKernel(L=6, k=4, d=2, transform="rbf")
        assert "rbf" in repr(kern)

    def test_weighted_repr(self):
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="gaussian",
                        weight_sigma=10, weight_peak=50)
        r = repr(wk)
        assert "gaussian" in r
        assert "L=6" in r

    def test_windowed_repr(self):
        wk = WindowedGKMKernel(L=6, k=4, d=2, window=12, shift=4)
        r = repr(wk)
        assert "window=12" in r
        assert "shift=4" in r


# ==================================================================
# Query API
# ==================================================================

class TestQueryAPI:
    def test_cross_kernel(self):
        kern = GKMKernel(L=10, k=6, d=3, use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(X_query=[SEQS_6[0]])
        assert K.shape == (1, 6)

    def test_cross_kernel_multiple(self):
        kern = GKMKernel(L=10, k=6, d=3, use_rc=True)
        kern.set_references(SEQS_6)
        K = kern.kernel(X_query=SEQS_6[:3])
        assert K.shape == (3, 6)

    def test_lazy_references(self):
        kern = GKMKernel(L=6, k=4, d=2)
        K = kern.kernel(X_ref=SEQ4)
        assert K.shape == (4, 4)
        assert kern.n_references == 4


# ==================================================================
# Kernel type independence
# ==================================================================

class TestKernelTypeIndependence:
    def test_kernel_types_do_not_interfere(self):
        schemes = ["full", "truncated", "estimated_full"]
        kerns = {}
        for s in schemes:
            k = GKMKernel(L=8, k=5, d=2, kernel_type=s, use_rc=True)
            k.set_references(SEQ4)
            kerns[s] = k
        w_full = [kerns["full"]._kernel.weight(m) for m in range(3)]
        assert w_full[0] == math.comb(8, 5)
        Ks = {s: kerns[s].kernel(apply_transform=False) for s in schemes}
        for s in schemes:
            np.testing.assert_allclose(np.diag(Ks[s]), 1.0, atol=1e-9)
        assert not np.allclose(Ks["full"], Ks["truncated"])

    def test_regular_and_weighted_independent(self):
        k = GKMKernel(L=6, k=4, d=2, use_rc=True)
        k.set_references(SEQ4)
        wk = WGKMKernel(L=6, k=4, d=2, weight_kernel="laplacian",
                        weight_gamma=0.1, weight_sigma=50, weight_peak=50,
                        use_rc=True)
        wk.set_references(SEQ4)
        K_reg = k.kernel(apply_transform=False)
        K_wgt = wk.kernel(offsets=0)
        np.testing.assert_allclose(np.diag(K_reg), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.diag(K_wgt), 1.0, atol=1e-9)
        assert not np.allclose(K_reg, K_wgt)

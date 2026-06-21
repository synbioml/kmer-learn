"""Pure-Python brute-force reference implementation of the gkmSVM
"full" gapped k-mer kernel.

Used only by the test suite to validate the C implementation against
an independent implementation that follows the math directly, without
any trie tricks.

The gkmSVM full kernel for sequences a, b is defined as

    K(a, b) = sum over all gapped patterns p (length L, k non-gap
              positions, gaps allowed only between non-gap positions)
              of  (# occurrences of p in a) * (# occurrences of p in b)

where a "gapped pattern" matches a substring of length L if the
non-gap positions agree. Equivalently:

    K(a, b) = sum_{m=0..d} C(L-m, K) * mmcount(a, b, m)

where mmcount(a, b, m) counts the number of (substring-of-a,
substring-of-b) pairs of length L that agree in exactly L - m
positions on the k non-gap positions.

Hmm, actually the cleanest equivalent formulation — and the one that
matches the libgkm.c semantics bit-for-bit — is:

    For each pair of L-mer windows (i in a, j in b, on either strand):
        mm = number of mismatched positions in [0, L)
        if mm <= d:
            K_unnormalized(a, b) += weights[mm]
    K_norm(a, b) = K_unnormalized(a, b) / (sqnorm(a) * sqnorm(b))

with weights[mm] = C(L - mm, K)  and  sqnorm(a) = sqrt(K(a, a)).

This is exactly what `calc_gkm_kernel_wt` computes in libgkm.c, and
it's what we brute-force here.
"""

from __future__ import annotations

import math
from itertools import product
from typing import List

import numpy as np


_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def _encode(seq: str) -> np.ndarray:
    """Encode A=1, C=2, G=3, T=4. Non-ACGT -> 1 (matches libgkm)."""
    out = np.zeros(len(seq), dtype=np.int32)
    for i, c in enumerate(seq.upper()):
        out[i] = {"A": 1, "C": 2, "G": 3, "T": 4}.get(c, 1)
    return out


def _encode_rc(seq: str) -> np.ndarray:
    """Reverse-complement encoding."""
    return _encode("".join(_COMPLEMENT[c] for c in reversed(seq.upper())))


def _weights(L: int, k: int) -> List[float]:
    """weights[m] = C(L - m, K) for m = 0..L."""
    return [float(math.comb(L - m, k)) if (L - m) >= k else 0.0
            for m in range(L + 1)]


def _count_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Number of positions where a != b."""
    return int(np.sum(a != b))


def _sqnorm(seq: str, L: int, k: int, d: int) -> float:
    """sqrt(K(seq, seq)) computed brute-force."""
    weights = _weights(L, k)
    if len(seq) < L:
        return 0.0
    enc_f = _encode(seq)
    enc_r = _encode_rc(seq)
    # both strands contribute
    lmers_f = [enc_f[i:i + L] for i in range(len(seq) - L + 1)]
    lmers_r = [enc_r[i:i + L] for i in range(len(seq) - L + 1)]
    lmers = lmers_f + lmers_r
    # only forward-strand L-mers are the "outer" loop in libgkm's
    # sqnorm calc (which iterates i in 0..nkmers-1, j in 0..2*nkmers-1).
    # We mirror that exactly to avoid double-counting.
    s = 0.0
    for i in range(len(lmers_f)):
        ai = lmers_f[i]
        for j in range(len(lmers)):
            mm = _count_mismatches(ai, lmers[j])
            if mm <= d:
                s += weights[mm]
    return math.sqrt(s)


def _kernel_unnormalized(a: str, b: str, L: int, k: int, d: int) -> float:
    """K(a, b) computed brute-force. Both strands of each sequence."""
    weights = _weights(L, k)
    if len(a) < L or len(b) < L:
        return 0.0
    a_f = [_encode(a)[i:i + L] for i in range(len(a) - L + 1)]
    a_r = [_encode_rc(a)[i:i + L] for i in range(len(a) - L + 1)]
    b_f = [_encode(b)[i:i + L] for i in range(len(b) - L + 1)]
    b_r = [_encode_rc(b)[i:i + L] for i in range(len(b) - L + 1)]
    a_lmers = a_f + a_r
    b_lmers = b_f + b_r
    s = 0.0
    # The libgkm DFS, when evaluating K(a, b) with a as the query,
    # adds all L-mers of both strands of b to the trie, then queries
    # with all L-mers of both strands of a (well, actually only the
    # forward strand of a, per `gkmkernel_kernelfunc_sqnorm_single`
    # which only iterates i in 0..nkmers-1 for the outer loop).
    # But for the kernel_func_batch case (which we use for cross-seq
    # evaluation), the outer loop is over the forward strand of a,
    # and the trie contains both strands of b.
    # Wait - looking at the code more carefully: in
    # `gkmkernel_kernelfunc_batch_single`, only forward-strand L-mers
    # of `a` are pushed into matching_bases (using da->seq + i, not
    # da->seq_rc + i). So the outer loop is forward-only.
    # The trie (kernel->prob_kmertree) is built via
    # `kmer_tree_add_sequence(tree, i, x[i])` which adds BOTH strands.
    # So: outer = forward L-mers of a; trie = both strands of every
    # reference sequence.
    for ai in a_f:
        for bj in b_lmers:
            mm = _count_mismatches(ai, bj)
            if mm <= d:
                s += weights[mm]
    return s


def kernel_pair(a: str, b: str, L: int, k: int, d: int) -> float:
    """Normalised K(a, b) computed brute-force."""
    sa = _sqnorm(a, L, k, d)
    sb = _sqnorm(b, L, k, d)
    if sa == 0 or sb == 0:
        return 0.0
    return _kernel_unnormalized(a, b, L, k, d) / (sa * sb)


def kernel_matrix(seqs: List[str], L: int, k: int, d: int) -> np.ndarray:
    """Full kernel matrix K[i, j] = K_norm(seqs[i], seqs[j])."""
    n = len(seqs)
    sqnorms = [_sqnorm(s, L, k, d) for s in seqs]
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if sqnorms[i] == 0 or sqnorms[j] == 0:
                K[i, j] = 0.0
            else:
                K[i, j] = _kernel_unnormalized(seqs[i], seqs[j], L, k, d) / (
                    sqnorms[i] * sqnorms[j])
    return K

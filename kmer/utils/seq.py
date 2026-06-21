"""k-mer and sequence utility functions.

All functions require uppercase input. Non-IUPAC characters (including
lowercase and U) raise ValueError.
"""

from __future__ import annotations

from typing import Iterable

ALPHABET = "ACGT"
IUPAC_ALPHABET = "ACGTRYSWKMBDHVN"

_ENCODE = {c: i for i, c in enumerate(ALPHABET)}  # A=0, C=1, G=2, T=3
_DECODE = {i: c for i, c in enumerate(ALPHABET)}

_IUPAC_COMPLEMENT = {
    "A": "T", "C": "G", "G": "C", "T": "A",
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D", "N": "N",
}


def kmer_to_code(kmer: str | bytes) -> int:
    """Pack an ACGT k-mer into an integer code (A=0, C=1, G=2, T=3, MSB-first)."""
    if isinstance(kmer, bytes):
        kmer = kmer.decode("ascii")
    code = 0
    for c in kmer:
        if c not in _ENCODE:
            raise ValueError(f"non-ACGT character {c!r} in k-mer {kmer!r}")
        code = (code << 2) | _ENCODE[c]
    return code


def code_to_kmer(code: int, k: int) -> str:
    """Unpack an integer code to a k-mer string."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return "".join(_DECODE[(code >> (2 * i)) & 0b11] for i in range(k - 1, -1, -1))


def reverse_complement(seq: str | bytes) -> str | bytes:
    """Reverse complement of an uppercase IUPAC sequence.

    Raises ValueError on any character not in ACGTRYSWKMBDHVN.
    """
    is_bytes = isinstance(seq, bytes)
    s = seq.decode("ascii") if is_bytes else seq
    try:
        rc = "".join(_IUPAC_COMPLEMENT[c] for c in reversed(s))
    except KeyError as e:
        raise ValueError(f"invalid IUPAC character {e.args[0]!r}") from e
    return rc.encode("ascii") if is_bytes else rc


def canonical_code(code: int, k: int) -> int:
    """Return min(code, rc(code)) for a packed k-mer."""
    rc = _packed_rc(code, k)
    return code if code < rc else rc


def _packed_rc(code: int, k: int) -> int:
    _COMP = (3, 2, 1, 0)  # A<->T, C<->G
    out = 0
    for _ in range(k):
        out = (out << 2) | _COMP[code & 0b11]
        code >>= 2
    return out


def iterate_kmers(seq: str | bytes, k: int) -> Iterable[str]:
    """Yield all k-mers in seq, left to right."""
    if isinstance(seq, bytes):
        seq = seq.decode("ascii")
    for i in range(len(seq) - k + 1):
        yield seq[i : i + k]


def kmer_is_canonical(kmer: str | bytes) -> bool:
    """Return True if kmer <= its reverse complement."""
    rc = reverse_complement(kmer)
    if isinstance(kmer, bytes):
        return kmer <= rc.encode("ascii")
    return kmer <= rc

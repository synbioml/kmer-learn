"""kmer.utils — utility functions for k-mer and sequence operations."""

from .seq import (
    ALPHABET,
    IUPAC_ALPHABET,
    kmer_to_code,
    code_to_kmer,
    reverse_complement,
    canonical_code,
    iterate_kmers,
    kmer_is_canonical,
)

__all__ = [
    "ALPHABET",
    "IUPAC_ALPHABET",
    "kmer_to_code",
    "code_to_kmer",
    "reverse_complement",
    "canonical_code",
    "iterate_kmers",
    "kmer_is_canonical",
]

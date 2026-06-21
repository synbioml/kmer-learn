"""kmer.encoders — k-mer feature encoders.

Each encoder produces a scipy.sparse.csr_matrix of k-mer counts
(rows = sequences, columns = features). Encoders share a common
backend interface and can be used interchangeably downstream.

Classes
-------
- :class:`SpectrumEncoder`   — plain k-mer counts (rolling hash)
- :class:`GappyEncoder`      — gappy k-mer counts (masked hash)
- :class:`MismatchEncoder`   — mismatch-tolerant k-mer counts
"""

from .spectrum import SpectrumEncoder
from .gappy import GappyEncoder, generate_masks
from .mismatch import MismatchEncoder

__all__ = ["SpectrumEncoder", "GappyEncoder", "MismatchEncoder", "generate_masks"]

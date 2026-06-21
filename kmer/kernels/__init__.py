"""Sequence kernels for nucleotide sequences.

Currently exposes the gkmSVM / LS-GKM gapped k-mer kernel family
(Ghandi et al. 2014; Lee 2016) via a C-backed implementation.

Classes
-------
- :class:`GKMKernel`           — full-sequence, uniform weights
- :class:`WGKMKernel`          — full-sequence, positional weighting
- :class:`WindowedGKMKernel`   — windowed, uniform weights
- :class:`WindowedWGKMKernel`  — windowed, positional weighting
"""

from .gkmkernel import (
    BaseGKMKernel,
    GKMKernel,
    WGKMKernel,
    WindowedGKMKernel,
    WindowedWGKMKernel,
)

__all__ = [
    "BaseGKMKernel",
    "GKMKernel",
    "WGKMKernel",
    "WindowedGKMKernel",
    "WindowedWGKMKernel",
]

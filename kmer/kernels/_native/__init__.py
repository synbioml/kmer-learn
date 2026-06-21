"""Compiled C extension for the gkm kernel family.

This subpackage contains only the compiled extension module and its C
source. The Python class hierarchy lives in :mod:`kmer.kernels._core`.
"""

from ._gkmkern import Kernel, WeightedKernel, Sequence

__all__ = ["Kernel", "WeightedKernel", "Sequence"]

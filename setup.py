"""Build script for the kmer CPython extensions.

Compiles the following extensions:
  - kmer.kernels._native._gkmkern       (gkm kernel family)
  - kmer.perturb._native._shuffler      (k-mer Eulerian shuffler)
  - kmer.perturb._native._chunker       (sequence chunker)
  - kmer.perturb._native._local         (position-bounded local shuffler)
  - kmer.encoders._native._spectrum     (rolling-hash k-mer counter)
  - kmer.encoders._native._gappy        (masked-hash gappy k-mer counter)
  - kmer.encoders._native._mismatch     (mismatch k-mer counter)

All paths must be relative to this file (setuptools requirement).
The root kmer/ directory is added to every extension's include path
so that all C code can `#include "_common.h"` (the shared header).
"""
import sys

from setuptools import setup, Extension


# The native code only uses OpenMP pragmas, not direct runtime calls. Apple
# Clang wheels therefore build serially without an external libomp toolchain;
# MSVC and GCC retain their native OpenMP modes.
if sys.platform == "win32":
    _BASE_COMPILE_ARGS = ["/O2", "/W3"]
    _OPENMP_COMPILE_ARGS = ["/openmp"]
    _OPENMP_LINK_ARGS = []
    _CXX_STANDARD_ARGS = ["/std:c++14"]
    _CXX_WARNING_ARGS = []
    _MATH_LIBRARIES = []
elif sys.platform == "darwin":
    _BASE_COMPILE_ARGS = ["-O3", "-Wall"]
    _OPENMP_COMPILE_ARGS = []
    _OPENMP_LINK_ARGS = []
    _CXX_STANDARD_ARGS = ["-std=c++11"]
    _CXX_WARNING_ARGS = ["-Wno-write-strings"]
    _MATH_LIBRARIES = ["m"]
else:
    _BASE_COMPILE_ARGS = ["-O3", "-Wall"]
    _OPENMP_COMPILE_ARGS = ["-fopenmp"]
    _OPENMP_LINK_ARGS = ["-fopenmp"]
    _CXX_STANDARD_ARGS = ["-std=c++11"]
    _CXX_WARNING_ARGS = ["-Wno-write-strings"]
    _MATH_LIBRARIES = ["m"]

_C_OPENMP_ARGS = _BASE_COMPILE_ARGS + _OPENMP_COMPILE_ARGS
_CXX_OPENMP_ARGS = (
    _CXX_STANDARD_ARGS
    + _BASE_COMPILE_ARGS
    + _CXX_WARNING_ARGS
    + _OPENMP_COMPILE_ARGS
)


def _ext(name, sources, include_dirs=None, extra_compile_args=None,
         extra_link_args=None, libraries=None):
    base_includes = ["kmer"]
    return Extension(
        name=name,
        sources=sources,
        include_dirs=base_includes + (include_dirs or []),
        extra_compile_args=(
            _BASE_COMPILE_ARGS
            if extra_compile_args is None
            else extra_compile_args
        ),
        extra_link_args=[] if extra_link_args is None else extra_link_args,
        libraries=[] if libraries is None else libraries,
    )


# --- gkm kernel ---------------------------------------------------------
gkm_ext = _ext(
    name="kmer.kernels._native._gkmkern",
    sources=[
        "kmer/kernels/_native/gkmkern.c",
        "kmer/kernels/_native/_gkmkern_pylib.c",
    ],
    include_dirs=["kmer/kernels/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
    libraries=_MATH_LIBRARIES,
)

# --- perturb: shuffler + chunker + local shuffler ----------------------
shuffler_ext = _ext(
    name="kmer.perturb._native._shuffler",
    sources=["kmer/perturb/_native/_shuffler_pylib.c"],
    include_dirs=["kmer/perturb/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
)
chunker_ext = _ext(
    name="kmer.perturb._native._chunker",
    sources=["kmer/perturb/_native/_chunker_pylib.c"],
    include_dirs=["kmer/perturb/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
)
local_ext = _ext(
    name="kmer.perturb._native._local",
    sources=["kmer/perturb/_native/_local_pylib.c"],
    include_dirs=["kmer/perturb/_native"],
)

# --- encoders: spectrum + gappy + mismatch -----------------------------
spectrum_ext = _ext(
    name="kmer.encoders._native._spectrum",
    sources=["kmer/encoders/_native/_spectrum_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
)
gappy_ext = _ext(
    name="kmer.encoders._native._gappy",
    sources=["kmer/encoders/_native/_gappy_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
)
mismatch_ext = _ext(
    name="kmer.encoders._native._mismatch",
    sources=["kmer/encoders/_native/_mismatch_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=_C_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
)



# --- ann: HNSW index (C++ with vendored hnswlib) ----------------------
ann_ext = Extension(
    name="kmer.ann._native._hnsw",
    sources=["kmer/ann/_native/_gkm_hnsw.cpp"],
    include_dirs=["kmer", "kmer/ann/_native", "kmer/kernels/_native"],
    language="c++",
    extra_compile_args=_CXX_OPENMP_ARGS,
    extra_link_args=_OPENMP_LINK_ARGS,
    libraries=_MATH_LIBRARIES,
)

setup(
    ext_modules=[gkm_ext, shuffler_ext, chunker_ext, local_ext,
                 spectrum_ext, gappy_ext, mismatch_ext, ann_ext],
)

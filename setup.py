"""Build script for the kmer CPython extensions.

Compiles the following extensions:
  - kmer.kernels._native._gkmkern       (gkm kernel family)
  - kmer.perturb._native._shuffler      (k-mer Eulerian shuffler)
  - kmer.perturb._native._chunker       (sequence chunker)
  - kmer.encoders._native._spectrum     (rolling-hash k-mer counter)
  - kmer.encoders._native._gappy        (masked-hash gappy k-mer counter)
  - kmer.encoders._native._mismatch     (mismatch k-mer counter)

All paths must be relative to this file (setuptools requirement).
The root kmer/ directory is added to every extension's include path
so that all C code can `#include "_common.h"` (the shared header).
"""
import os

from setuptools import setup, Extension


def _ext(name, sources, include_dirs=None, extra_compile_args=None,
         extra_link_args=None, libraries=None):
    base_includes = ["kmer"]
    return Extension(
        name=name,
        sources=sources,
        include_dirs=base_includes + (include_dirs or []),
        extra_compile_args=extra_compile_args or ["-O3", "-Wall"],
        extra_link_args=extra_link_args or [],
        libraries=libraries or [],
    )


# --- gkm kernel ---------------------------------------------------------
gkm_ext = _ext(
    name="kmer.kernels._native._gkmkern",
    sources=[
        "kmer/kernels/_native/gkmkern.c",
        "kmer/kernels/_native/_gkmkern_pylib.c",
    ],
    include_dirs=["kmer/kernels/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    libraries=(["m"] if os.name != "nt" else []),
)

# --- perturb: shuffler + chunker ---------------------------------------
shuffler_ext = _ext(
    name="kmer.perturb._native._shuffler",
    sources=["kmer/perturb/_native/_shuffler_pylib.c"],
    include_dirs=["kmer/perturb/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)
chunker_ext = _ext(
    name="kmer.perturb._native._chunker",
    sources=["kmer/perturb/_native/_chunker_pylib.c"],
    include_dirs=["kmer/perturb/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

# --- encoders: spectrum + gappy + mismatch -----------------------------
spectrum_ext = _ext(
    name="kmer.encoders._native._spectrum",
    sources=["kmer/encoders/_native/_spectrum_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)
gappy_ext = _ext(
    name="kmer.encoders._native._gappy",
    sources=["kmer/encoders/_native/_gappy_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)
mismatch_ext = _ext(
    name="kmer.encoders._native._mismatch",
    sources=["kmer/encoders/_native/_mismatch_pylib.c"],
    include_dirs=["kmer/encoders/_native"],
    extra_compile_args=["-O3", "-Wall", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)


setup(
    ext_modules=[gkm_ext, shuffler_ext, chunker_ext,
                 spectrum_ext, gappy_ext, mismatch_ext],
)

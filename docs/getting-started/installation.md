# Installation

## Install from PyPI

```bash
pip install kmer-learn
```

The core installation requires Python 3.9 or newer and installs NumPy, SciPy,
and scikit-learn.

## Optional accelerated backends

```bash
pip install "kmer-learn[fast]"
```

The `fast` extra installs:

- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) for accelerated Hamming
  and Levenshtein distances.
- [parasail](https://github.com/jeffdaily/parasail) for SIMD-accelerated global
  and local alignments.

The package includes Python fallbacks, but these backends are strongly
recommended for larger workloads.

## Install from source

A C/C++ compiler is required to build the native extensions.

```bash
git clone https://github.com/synbioml/kmer-learn.git
cd kmer-learn
pip install -e .
```

## Build this documentation

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The generated site is static. Open `docs/_build/html/index.html` locally; no
server is required.

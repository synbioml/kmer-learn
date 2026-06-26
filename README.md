# kmer-learn

A modern Python toolkit for classical sequence machine learning.

> **⚠ API stability:** This package is in early development (v0.x). The public API is **not yet stable** — breaking changes may be introduced between minor versions until v1.0. Pin your dependency to an exact version (e.g. `kmer-learn==0.1.0`) if reproducibility matters.

## Install

```bash
pip install kmer-learn
```

From source (requires a C compiler):

```bash
git clone https://github.com/synbioml/kmer-learn.git
cd kmer-learn
pip install -e .
```

Optional backends (strongly recommended for speed):

```bash
pip install rapidfuzz parasail   # 10-100x faster edit distances and alignments
pip install pyfastx              # fast FASTA parsing for examples
```

## Features

### Kernels
- **GKMKernel** family — gapped k-mer kernels from gkmSVM / LS-GKM (Ghandi et al. 2014; Lee 2016). C-backed, OpenMP-parallel, with full/estimated/truncated mismatch schemes, 6 post-transforms (RBF, poly, sigmoid, exponential, …), 5 positional weighting kernels (triangular, Epanechnikov, Gaussian, Laplacian, Cauchy), reverse-complement indexing, sliding-window scan, and 3D windowed tensors.
- **DistanceKernel** — turns any distance into a kernel via a post-transform.

### Encoders (CSR output)
- **SpectrumEncoder** — plain k-mer counts via rolling hash (k ≤ 12).
- **GappyEncoder** — gappy k-mer counts with explicit masks (`"*--*"`) or gap ranges (`L=6, g_min=2, g_max=3`).
- **MismatchEncoder** — mismatch-tolerant k-mer counts (Leslie, Eskin, Noble 2004).

All encoders support `canonical_rc=True` for reverse-complement collapsing.

### Distances
- **Hamming**, **Levenshtein** (rapidfuzz backend + Python fallback).
- **NeedlemanWunsch**, **SmithWaterman** (parasail backend + Python fallback), with custom substitution matrices (NUC4.4, BLOSUM62, …).

### Models
- **DifferentialKmerScorer** — Multinomial Naive Bayes on k-mer features, with auto-generated negatives via Shuffler/Chunker.
- **KernelSVM** — SVM with a precomputed kernel (works with GKMKernel and DistanceKernel).
- **LinearSVM** — Linear SVM on encoder features.
- **KNNClassifier** — k-Nearest Neighbors with a sequence distance.

### Sequence perturbation
- **KmerShuffler** — k-mer-preserving shuffle via random Eulerian paths in the De Bruijn graph. Three endpoint modes (preserve / free / crop). Philox4×32-10 RNG, reproducible across `n_jobs`.
- **Chunker** — block-level perturbation: split into chunks of size `[min, max]`, optionally reverse-complement each, shuffle, concatenate. Five residual-handling modes, two algorithms (random / backtrack).
- **BaseBackgroundModel** — ABC for custom background models.

### Utilities
- `kmer.utils` — bit-packed k-mer helpers (`kmer_to_code`, `code_to_kmer`, `reverse_complement`, `canonical_code`).

## Quick start

```python
from kmer.kernels import GKMKernel
from kmer.models import KernelSVM

# Train a gkm-SVM
clf = KernelSVM(GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True), C=1.0)
clf.fit(positives + negatives, [1]*len(positives) + [0]*len(negatives))
preds = clf.predict(test_seqs)
```

```python
from kmer.encoders import SpectrumEncoder
from kmer.models import DifferentialKmerScorer
from kmer.perturb import KmerShuffler

# Differential k-mer scoring with dinucleotide-shuffled background
scorer = DifferentialKmerScorer(
    featurizer=SpectrumEncoder(k=6, canonical_rc=True),
    background=KmerShuffler(k=2, seed=42),
)
scorer.fit(positives)
top_motifs = sorted(scorer.kmer_scores_.items(), key=lambda x: -x[1])[:20]
```

```python
from kmer.distance import Levenshtein, DistanceKernel
from kmer.models import KNNClassifier

# KNN with edit distance
clf = KNNClassifier(Levenshtein(), n_neighbors=5)
clf.fit(train_seqs, y_train)
```

## Examples (Vignettes)

The `examples/` directory (top-level, next to `kmer/`) contains a series of cross-linked Jupyter notebooks. Each notebook starts with a vignette index linking to all others.

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [`01_basic_kernel_matrix.ipynb`](examples/01_basic_kernel_matrix.ipynb) | GKMKernel: build, inspect, verify invariants |
| 02 | [`02_distance_metrics_and_kernels.ipynb`](examples/02_distance_metrics_and_kernels.ipynb) | Distance metrics (Hamming, Levenshtein, NW, SW) + DistanceKernel (RBF, PSD, KernelSVM) |
| 03 | [`03_svc_with_kernel.ipynb`](examples/03_svc_with_kernel.ipynb) | Train a gkm-SVM with KernelSVM |
| 04 | [`04_clustering_sequences.ipynb`](examples/04_clustering_sequences.ipynb) | Hierarchical clustering with kernel distances |
| 05 | [`05_score_long_sequence.ipynb`](examples/05_score_long_sequence.ipynb) | Sliding-window scan of a long sequence |
| 06 | [`06_weighted_kernel.ipynb`](examples/06_weighted_kernel.ipynb) | WGKMKernel positional weighting (centered motif) |
| 07 | [`07_transform_and_comparison.ipynb`](examples/07_transform_and_comparison.ipynb) | All 3 schemes × 6 transforms, GKM vs WGKM |
| 08 | [`08_windowed_3d_tensors.ipynb`](examples/08_windowed_3d_tensors.ipynb) | WindowedGKMKernel 3D output (line plot) |
| 09 | [`09_spectrum_encoder_and_differential.ipynb`](examples/09_spectrum_encoder_and_differential.ipynb) | SpectrumEncoder + DifferentialKmerScorer |
| 10 | [`10_gappy_encoder.ipynb`](examples/10_gappy_encoder.ipynb) | GappyEncoder with masks, gap ranges, RC collapse |
| 11 | [`11_mismatch_encoder.ipynb`](examples/11_mismatch_encoder.ipynb) | MismatchEncoder and comparison to spectrum |
| 12 | [`12_shuffler_and_chunker.ipynb`](examples/12_shuffler_and_chunker.ipynb) | KmerShuffler + Chunker for negative-set generation |

## Citation

An article describing this package is in preparation. Until it is published, please cite the package as:

> Zinkevich A. *kmer-learn: Classical machine learning primitives for nucleotide sequences.* (in preparation).

For the mean time, if you use the package in your research, please cite the relevant foundational works listed below.

## References

The package builds on the following foundational works:

- **gkmSVM** — Ghandi M, Lee D, Mohammad-Noori M, Beer MA. *Enhanced regulatory sequence prediction using gapped k-mer features.* PLoS Comput Biol. 2014;10(7):e1003711.
- **LS-GKM** — Lee D. *LS-GKM: a new gkm-SVM for large-scale datasets.* Bioinformatics. 2016;32(14):2196–8.
- **Mismatch kernel** — Leslie CS, Eskin E, Cohen A, Weston J, Noble WS. *Mismatch string kernels for discriminative protein classification.* Bioinformatics. 2004;20 Suppl 1:i467–76.
- **Needleman-Wunsch** — Needleman SB, Wunsch CD. *A general method applicable to the search for similarities in the amino acid sequence of two proteins.* J Mol Biol. 1970;48(3):443–53.
- **Smith-Waterman** — Smith TF, Waterman MS. *Identification of common molecular subsequences.* J Mol Biol. 1981;147(1):195–7.

Third-party libraries used as optional backends:

- **[rapidfuzz](https://github.com/maxbachmann/RapidFuzz)** — fast Levenshtein and Hamming distances.
- **[parasail](https://github.com/jeffdaily/parasail)** — SIMD-accelerated sequence alignment (Daily, 2016).
- **[scikit-learn](https://scikit-learn.org)** — SVM, Naive Bayes, KNN.
- **[NumPy](https://numpy.org) / [SciPy](https://scipy.org)** — array and sparse-matrix infrastructure.

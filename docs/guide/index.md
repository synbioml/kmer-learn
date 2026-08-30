# Package guide

The public package is divided by the role each component plays in a sequence
analysis.

| Section | Purpose | Key public objects |
|---|---|---|
| `kmer.distance` | Compare sequences directly | `Hamming`, `Levenshtein`, `NeedlemanWunsch`, `SmithWaterman`, `DistanceKernel` |
| `kmer.encoders` | Convert sequences into sparse count matrices | `SpectrumEncoder`, `GappyEncoder`, `MismatchEncoder` |
| `kmer.kernels` | Compare motif content with k-mer kernels | `GKMKernel`, `WGKMKernel`, windowed variants |
| `kmer.background` | Generate null/background sequences | `KmerShuffler`, `Chunker`, `LocalShuffler`, `LocalChunker` |
| `kmer.models` | Fit supervised scikit-learn-style estimators | `KmerMultinomialNB`, `KmerLinearRegression`, `LinearSVM`, `KernelSVM`, `KNNClassifier` |
| `kmer.ann` | Search large sequence collections approximately | `ANNIndex` |
| `kmer.utils` | Encode and manipulate nucleotide strings and k-mers | `reverse_complement`, `kmer_to_code`, `canonical_code` |

Each section combines concepts, API-selection guidance, executable examples,
interpretation, practical caveats, and links to generated reference pages.
Follow the sequence from direct comparison through representations, null
models, supervised learning, and scalable search, or jump to the section
matching your analysis strategy.

```{toctree}
:maxdepth: 1

distances
encoders
kernels
background
models
ann
utils
```

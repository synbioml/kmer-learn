# Package guide

The public package is divided by the role each component plays in a sequence
analysis.

| Section | Purpose | Key public objects |
|---|---|---|
| `kmer.utils` | Encode and manipulate nucleotide strings and k-mers | `reverse_complement`, `kmer_to_code`, `canonical_code` |
| `kmer.encoders` | Convert sequences into sparse count matrices | `SpectrumEncoder`, `GappyEncoder`, `MismatchEncoder` |
| `kmer.distance` | Compare sequences directly | `Hamming`, `Levenshtein`, `NeedlemanWunsch`, `SmithWaterman`, `DistanceKernel` |
| `kmer.kernels` | Compare motif content with gapped k-mer kernels | `GKMKernel`, `WGKMKernel`, windowed variants |
| `kmer.models` | Fit scikit-learn-style estimators to sequences | `LinearSVM`, `KernelSVM`, `KNNClassifier`, `DifferentialKmerScorer` |
| `kmer.background` | Generate null/background sequences | `KmerShuffler`, `Chunker`, `LocalShuffler`, `LocalChunker` |
| `kmer.ann` | Search large sequence collections approximately | `ANNIndex` |

The first complete section is {doc}`encoders`. The remaining sections will be
migrated one at a time using the same structure: concepts, API selection,
executable examples, interpretation, caveats, and links to reference pages.

```{toctree}
:maxdepth: 1

encoders
```

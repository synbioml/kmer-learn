# Choosing a workflow

Start with the kind of comparison or prediction your analysis needs.

| Goal | Representation | Typical model or operation |
|---|---|---|
| Classify by exact or approximate k-mer content | `SpectrumEncoder`, `GappyEncoder`, or `MismatchEncoder` | `LinearSVM` |
| Identify enriched sequence patterns | A k-mer encoder | `DifferentialKmerScorer` |
| Compare regulatory sequences with gapped motifs | `GKMKernel` or `WGKMKernel` | `KernelSVM` |
| Classify using edit or alignment similarity | A class from `kmer.distance` | `KNNClassifier` or `DistanceKernel` |
| Generate matched negative sequences | `KmerShuffler`, `Chunker`, or their local variants | A downstream classifier or scorer |
| Find neighbors in a large sequence collection | An `ANNIndex` metric | HNSW approximate search |
| Locate sequence signal along a long region | A windowed GKM kernel | Sliding-window scoring |

## A useful first path

1. Read {doc}`../guide/encoders` to turn sequences into a sparse feature matrix.
2. Fit a feature-based model such as {class}`kmer.models.LinearSVM`.
3. Compare that workflow with a kernel or distance method when exact k-mer
   positions, mismatches, or indels are biologically important.

## How the documentation is organized

- **Package guide:** concepts, assumptions, parameter choices, and executable
  examples for each public package section.
- **Tutorials and recipes:** complete task-oriented analyses that combine
  package sections.
- **API reference:** signatures and detailed behavior generated from source
  docstrings.

This layering lets new users follow a path while experienced users can jump
straight to a package section or API object.

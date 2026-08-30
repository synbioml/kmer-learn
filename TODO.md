# TODO

## Completed

- [x] Repair Windows wheel compilation with MSVC-native flags.
  - [x] Replace GNU optimization, warning, standard, and OpenMP flags on Windows.
  - [x] Validate Windows, macOS, and Linux extension metadata.
  - [x] Rebuild native extensions and pass 737 full-suite tests.

- [x] Repair macOS ARM64 wheel compilation.
  - [x] Omit unsupported OpenMP flags from macOS extension builds.
  - [x] Remove the unnecessary GCC-only `-fpermissive` C++ flag.
  - [x] Preserve Linux OpenMP flags and validate extension metadata.
  - [x] Rebuild native extensions and pass 737 full-suite tests.

- [x] Repair the reported GitHub build dependency failure.
  - [x] Remove the unused pandas import instead of adding a dependency.
  - [x] Correct the push workflow branch from `main` to `master`.
  - [x] Validate metadata and pass 737 full-suite tests.

- [x] Add `LocalChunker` as a composition layer over `LocalShuffler`.
  - [x] Support list, tuple, and one-dimensional NumPy chunk collections.
  - [x] Preserve chunk objects and strictly bound displacement in chunk units.
  - [x] Delegate seed/state, L/H resolution, permutation, and statistics.
  - [x] Export and document the public API; pass 737 full-suite tests.

- [x] Select and implement the final `kmer.perturb.LocalShuffler` algorithm.
  - [x] Implement and validate the O(nS) H=0 virtual-tail version.
  - [x] Preserve it byte-for-byte as `kmer/perturb/_local_virtual_tail.py`.
  - [x] Prototype and benchmark uniform-kick and free-first collision chains.
  - [x] Select randomized-insertion uniform-kick with soft free-first and hard augmenting fallback.
  - [x] Implement the C extension with `str`, `bytes`, and 1D NumPy array support.
  - [x] Implement adaptive/custom L/H resolution, explicit disabling, and literal aliases.
  - [x] Replace `LocalShuffler`, update tests/documentation, and pass 686 full-suite tests.
  - [x] Validate theoretical marginals, limits/fallbacks, reproducibility, and performance.
- [x] Locate the `kmer-learn` repository and identify `kmer.perturb` as the integration point.
- [x] Confirm the public class name `LocalShuffler`.

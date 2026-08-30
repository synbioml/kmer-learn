# kmer-learn

**Classical machine-learning tools for nucleotide sequences.**

`kmer-learn` provides sparse k-mer encoders, sequence distances, gapped k-mer
kernels, scikit-learn-compatible models, sequence perturbation methods, and
approximate nearest-neighbor search. Performance-critical operations use native
extensions while the public workflows remain Pythonic.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Start with a workflow
:link: getting-started/choosing-a-workflow
:link-type: doc
Choose a representation and model based on your biological question.
:::

:::{grid-item-card} Explore the package
:link: guide/index
:link-type: doc
Understand each package section, its assumptions, and when to use it.
:::

:::{grid-item-card} Run tutorials
:link: guide/encoders
:link-type: doc
Follow executable analyses with interpreted output and practical guidance.
:::

:::{grid-item-card} Look up an API
:link: api/index
:link-type: doc
Find signatures and details generated from the public Python docstrings.
:::

::::

```{note}
`kmer-learn` is in alpha development. Pin an exact package version when
reproducibility matters because public APIs may change before version 1.0.
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Start here

getting-started/installation
getting-started/choosing-a-workflow
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Package guide

guide/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api/index
```

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Sequence statistics

`kmer.stats` summarizes individual sequences: base composition, k-mer
content, repetitiveness, and predictability. Every callable accepts a
single sequence or a list of sequences and returns one row per
sequence, which makes the outputs directly usable as feature matrices
for the models in {doc}`models`.

## Learning goals

By the end of this guide, you will be able to:

- measure monomer fractions and directional skews per sequence;
- summarize a k-mer histogram by entropy, Gini, or duplicate count;
- condense rolling window profiles into per-sequence summaries;
- quantify repetitiveness with homopolymer, tandem-repeat, substring,
  and Lempel–Ziv statistics;
- fit empirical Markov diagnostics up to order three.

## Monomer composition and skew

```{code-cell} ipython3
from kmer.stats import monomer_content, monomer_skew

seqs = ["GCGCGCATATATGCGC", "AAAACCCCGGGGTTTT", "ACGTACGTACGTACGT"]

print(monomer_content(seqs, letters="GC"))
print(monomer_skew(seqs, letters="GC").ravel())
```

`monomer_content` accepts any letter combination (`"GC"`, `"AT"`,
`"CG"`); `monomer_skew` returns directional skews such as GC-skew,
which are zero for strand-symmetric sequences like these.

## K-mer content summaries

{func}`~kmer.stats.kmer_content` reduces a sequence's k-mer histogram
to a single number per statistic:

```{code-cell} ipython3
from kmer.stats import kmer_content

print("entropy :", kmer_content(seqs, L=3, stat="entropy").ravel())
print("gini    :", kmer_content(seqs, L=3, stat="gini").ravel())
print("dups    :", kmer_content(seqs, L=3, stat="duplicates").ravel())
```

The repeating sequence has the lowest entropy, the most uneven
(Gini) histogram, and the most duplicate k-mers — three views of the
same fact that it draws from a small effective vocabulary.

## Windowed summaries

The `windowed_*` callables compute a statistic in complete windows
along each sequence and then aggregate the resulting profile:

```{code-cell} ipython3
import numpy as np
from kmer.stats import windowed_monomer_content

region = "TTTTTTTTTT" + "GCGCGCGCGC" * 6 + "AAAAAACCCC"
summary = windowed_monomer_content(
    [region], window_size=10, step=5, agg=("mean", "std", "max"),
)
print(summary)
```

One row per sequence, one column per requested aggregation. Shared
aggregations across all windowed statistics are `mean`, `std`,
`median`, `min`, `max`, quantiles (`level=...`), and threshold counts
(`count_thresholds=...`). Windowed skew and k-mer content work the
same way through `windowed_monomer_skew` and
`windowed_kmer_content`.

## Repetition and complexity

```{code-cell} ipython3
from kmer.stats import (
    homopolymer_stats,
    tandem_repeat_abundance,
    substring_complexity,
    lempel_ziv_complexity,
)

low = "AAAAACCCCCGGGGGTTTTTAAAAACCCCC"
high = "ACGTTGCAATGCCGTTACGATCGATCGTAC"

print("homopolymer runs :", homopolymer_stats([low, high], L_min=4,
                                              stat=("count", "max")))
print("tandem repeats   :", tandem_repeat_abundance(
    ["ACACACACGCGCGCGT", "ACGTTGCAACGTTGCA"],
    window_size=8, step=4, agg="count").ravel())
print("distinct substrings:", substring_complexity([low, high],
                                                   L_min=1, L_max=8).ravel())
print("Lempel-Ziv        :", lempel_ziv_complexity([low, high]).ravel())
```

`substring_complexity` counts distinct substrings with a native suffix
automaton and scales to long sequences; `L_min`/`L_max` restrict the
length range. `lempel_ziv_complexity` reports the normalized LZ76
factor count, which tends to zero for periodic inputs.
`tandem_repeat_abundance` summarizes rolling k-mer histograms in
complete windows, counting k-mers seen more than once.

## Markov diagnostics

```{code-cell} ipython3
from kmer.stats import markov_entropy, markov_transition_stats

periodic = "ACACACACACACACAC"
random_like = "ACGTTGCAATGCCGTT"

print("order 0:", markov_entropy([periodic, random_like], order=0).ravel())
print("order 1:", markov_entropy([periodic, random_like], order=1).ravel())

summary = markov_transition_stats([periodic, random_like], order=1)
print("transition summary columns:", summary.shape[1])
```

`markov_entropy` returns the empirical conditional entropy of order
0–3 chains (optionally strand-collapsed with `use_rc=True`); a
deterministic sequence such as `ACAC…` drops to zero entropy at order
1. `markov_transition_stats` summarizes the fitted transition table —
predictability and convergence per state.

These are *diagnostics*: per-sequence descriptors for comparison and
feature extraction. They are not a generative background model;
matched null sequences come from {doc}`background`.

## Next steps

- Feed the per-sequence rows to the estimators in {doc}`models`.
- Compare sequences against composition-matched nulls from
  {doc}`background`.
- Consult {doc}`../api/stats` for all signatures and options.

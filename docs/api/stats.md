# Sequence statistics API

`kmer.stats` provides callable sequence-level statistics for filtering,
generation assessment, and optional aggregation with
{class}`kmer.encoders.SequenceStatsEncoder`. Every statistic accepts one string
or an iterable of strings and returns a two-dimensional NumPy array.

## Callable contracts

```{eval-rst}
.. currentmodule:: kmer.stats

.. autoclass:: BaseSequenceStat
   :members:

.. autoclass:: BaseWindowSequenceStat
   :members:

.. autofunction:: sequence_stat

.. autofunction:: windowed_sequence_stat
```

Windowed statistics accept `agg` as one string or a tuple containing `mean`,
`std`, `median`, `min`, `max`, `quantile`, or `count`. Quantiles require
`level`; threshold counts require inclusive
`count_thresholds=(lower, upper)`, where either bound may be `None`.

## Composition

```{eval-rst}
.. py:function:: monomer_content(sequences, letters="GC", *, alphabet=ALPHABET)

   Calculate combined monomer fractions. A tuple such as ``("G", "C", "GC")``
   requests three separate columns.

.. py:function:: windowed_monomer_content(sequences, *, window_size, step=1, agg="mean", level=None, count_thresholds=None, letters="GC", alphabet=ALPHABET)

   Aggregate monomer fractions across complete sliding windows.

.. py:function:: monomer_skew(sequences, letters="GC", *, alphabet=ALPHABET)

   Calculate directional skews such as ``(G-C)/(G+C)``.

.. py:function:: windowed_monomer_skew(sequences, *, window_size, step=1, agg="mean", level=None, count_thresholds=None, letters="GC", alphabet=ALPHABET)

   Aggregate directional monomer skews across complete sliding windows.
```

## K-mer histogram statistics

```{eval-rst}
.. py:function:: kmer_content(sequences, L=3, *, stat="entropy", alphabet=ALPHABET)

   Summarize a k-mer histogram. ``stat`` accepts ``entropy`` (normalized
   Shannon entropy), ``gini`` (normalized Gini-Simpson diversity),
   ``duplicates`` (observations beyond each distinct k-mer's first), or a
   tuple of these values.

.. py:function:: windowed_kmer_content(sequences, *, window_size, step=1, agg="mean", level=None, count_thresholds=None, L=3, stat="entropy", alphabet=ALPHABET)

   Aggregate rolling k-mer histogram statistics. Histogram moments are updated
   incrementally as each window advances.

.. py:data:: tandem_repeat_abundance

   A lightweight local-recurrence proxy equivalent to
   ``windowed_kmer_content.partial(L=3, window_size=9, step=1,
   stat="duplicates", agg="count", count_thresholds=(1, None))``.
```

## Substrings, runs, and compressibility

```{eval-rst}
.. py:function:: substring_complexity(sequences, L_min=1, L_max=None, *, normalize=False, alphabet=ALPHABET)

   Count distinct substrings in the requested length range using a native
   suffix automaton. Normalization divides by the maximum possible count for
   the sequence length and alphabet.

.. py:data:: kmer_complexity

   Alias of :func:`substring_complexity`.

.. py:function:: homopolymer_stats(sequences, L_min=5, *, stat=("count", "fraction", "max"), alphabet=ALPHABET)

   Summarize homopolymer runs meeting the minimum length.

.. py:function:: lempel_ziv_complexity(sequences, *, normalize=True, alphabet=ALPHABET)

   Calculate LZ76 factor counts with optional asymptotic normalization.
```

## Markov statistics

```{eval-rst}
.. py:function:: markov_entropy(sequences, order=1, *, normalize=True, smoothing=0.0, use_rc=False, alphabet=ALPHABET)

   Calculate empirical conditional entropy. ``use_rc=True`` appends separate
   ``fwd`` and ``rc`` columns.

.. py:function:: markov_transition_stats(sequences, order=1, *, smoothing=0.0, use_rc=False, alphabet=ALPHABET)

   Return transition predictability, context coverage, log-likelihood,
   perplexity, spectral gap, and mixing-time diagnostics. A zero spectral gap
   and infinite mixing time indicate a reducible or periodic empirical chain.
```

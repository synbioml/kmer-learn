"""Sequence-level statistics for filtering and generation assessment."""

from ._base import BaseSequenceStat, sequence_stat
from ._window import BaseWindowSequenceStat, windowed_sequence_stat
from .content import (
    kmer_content,
    monomer_content,
    monomer_skew,
    windowed_kmer_content,
    windowed_monomer_content,
    windowed_monomer_skew,
)
from .complexity import (
    homopolymer_stats,
    kmer_complexity,
    lempel_ziv_complexity,
    substring_complexity,
)
from .markov import markov_entropy, markov_transition_stats


tandem_repeat_abundance = windowed_kmer_content.partial(
    L=3,
    window_size=9,
    step=1,
    stat="duplicates",
    agg="count",
    count_thresholds=(1, None),
)

__all__ = [
    "BaseSequenceStat",
    "BaseWindowSequenceStat",
    "sequence_stat",
    "windowed_sequence_stat",
    "monomer_content",
    "windowed_monomer_content",
    "kmer_content",
    "windowed_kmer_content",
    "monomer_skew",
    "windowed_monomer_skew",
    "substring_complexity",
    "kmer_complexity",
    "homopolymer_stats",
    "lempel_ziv_complexity",
    "markov_entropy",
    "markov_transition_stats",
    "tandem_repeat_abundance",
]

"""kmer.perturb._base — abstract base class for background models.

A "background model" generates null-model sequences from real ones:
shuffles, chunk-shuffles, or any other disruption that preserves some
composition property. Background models are used by
:class:`kmer.models.DifferentialKmerScorer` and similar estimators
to generate negatives when none are provided.

All background models implement two methods:
  - ``perturb(seq)``       — perturb one sequence
  - ``perturb_many(seqs)`` — perturb many (may be parallelized)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Union

SeqLike = Union[str, bytes]


class BaseBackgroundModel(ABC):
    """Abstract base class for sequence background models.

    Subclasses must implement :meth:`perturb` (single sequence).
    :meth:`perturb_many` has a default Python-loop implementation that
    subclasses may override for performance (e.g. the C-backed
    :class:`KmerShuffler` and :class:`Chunker` do this).
    """

    @abstractmethod
    def perturb(self, seq: SeqLike) -> SeqLike:
        """Perturb a single sequence.

        Parameters
        ----------
        seq : str or bytes
            Input sequence. Return type matches input type.

        Returns
        -------
        str or bytes
            Perturbed sequence of the same type and (usually) length.
        """
        ...

    def perturb_many(self, seqs: Iterable[SeqLike]) -> list:
        """Perturb many sequences.

        Default implementation loops in Python. Subclasses with a
        C-backed batch API should override this for performance.

        Parameters
        ----------
        seqs : iterable of (str or bytes)
            Input sequences.

        Returns
        -------
        list
            Perturbed sequences, one per input.
        """
        return [self.perturb(s) for s in seqs]

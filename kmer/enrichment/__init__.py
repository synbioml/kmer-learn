"""kmer.enrichment — k-mer enrichment analysis against a target.

Methods score every k-mer produced by a sparse encoder for association
with labels, ranks, or continuous values, calibrate the scores against
label permutations, and correct for multiple testing. Motif-level
instruments (PWM/PSAM representation, degenerate generalization,
scanning) are deliberately out of scope.
"""
from ._base import Background, EnrichmentResult, KmerEnrichment
from .fire import FireEnrichment
from .reduce import ReduceEnrichment
from .fisher import FisherEnrichment
from .drimust import DrimustEnrichment

__all__ = [
    "KmerEnrichment",
    "EnrichmentResult",
    "Background",
    "FireEnrichment",
    "ReduceEnrichment",
    "FisherEnrichment",
    "DrimustEnrichment",
]

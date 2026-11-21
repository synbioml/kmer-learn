#!/usr/bin/env python
"""Generate the enrichment reference dataset with planted k-mers.

Creates a deterministic dummy dataset used to compare kmer.enrichment
implementations against the official tools (DREME, HOMER, REDUCE,
FIRE, DRIMust). The dataset and every reference output live in
``kmer/tests/data/enrichment_reference/``.

Layout
------
seqs.fa          all sequences, IDs S000..S119
clusters.txt     "ID<TAB>cluster" (0/1), FIRE-style partition
expression.txt   "ID<TAB>y", REDUCE/DRIMust-style continuous target
positives.fa     cluster 0 sequences (two-set methods)
negatives.fa     cluster 1 sequences (two-set methods)
truth.json       ground truth: planted k-mers, carriers, target model

Ground truth: motif A occurs in 70% of cluster 0, motif B in 65% of
cluster 1, each planted once at a random position on a random strand.
The continuous target is y = 3*nA - 2*nB + eps, eps ~ N(0, 0.5).

Usage::

    python scripts/generate_enrichment_dataset.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20071005
N_SEQS = 120
SEQ_LEN = 150
CLUSTER_SPLIT = 60
MOTIF_A = "ACGTTGCA"
MOTIF_B = "TTTGGCCA"
FRAC_A = 0.70
FRAC_B = 0.65
BETA_A = 3.0
BETA_B = -2.0
NOISE_SD = 0.5

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def generate(out_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    letters = np.array(list("ACGT"))

    seqs = ["".join(letters[rng.integers(0, 4, SEQ_LEN)]) for _ in range(N_SEQS)]
    clusters = [0] * CLUSTER_SPLIT + [1] * (N_SEQS - CLUSTER_SPLIT)

    carriers = {MOTIF_A: [], MOTIF_B: []}
    for cluster, motif, frac in ((0, MOTIF_A, FRAC_A), (1, MOTIF_B, FRAC_B)):
        members = [i for i, c in enumerate(clusters) if c == cluster]
        chosen = rng.choice(members, size=int(round(frac * len(members))),
                             replace=False)
        for i in chosen:
            planted = motif if rng.random() < 0.5 else reverse_complement(motif)
            pos = int(rng.integers(0, SEQ_LEN - len(motif) + 1))
            seq = seqs[i]
            seqs[i] = seq[:pos] + planted + seq[pos + len(motif):]
            carriers[motif].append({"seq": f"S{i:03d}", "strand":
                                    "+" if planted == motif else "-",
                                    "position": pos})

    y = []
    for i, seq in enumerate(seqs):
        n_a = seq.count(MOTIF_A) + seq.count(reverse_complement(MOTIF_A))
        n_b = seq.count(MOTIF_B) + seq.count(reverse_complement(MOTIF_B))
        y.append(BETA_A * n_a + BETA_B * n_b + rng.normal(0, NOISE_SD))

    truth = {
        "seed": SEED,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "motifs": {
            MOTIF_A: {"cluster": 0, "fraction": FRAC_A, "carriers": carriers[MOTIF_A]},
            MOTIF_B: {"cluster": 1, "fraction": FRAC_B, "carriers": carriers[MOTIF_B]},
        },
        "reverse_complements": {MOTIF_A: reverse_complement(MOTIF_A),
                                MOTIF_B: reverse_complement(MOTIF_B)},
        "target_model": {"beta_a": BETA_A, "beta_b": BETA_B,
                         "noise_sd": NOISE_SD},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "seqs.fa", "w") as fh:
        for i, seq in enumerate(seqs):
            fh.write(f">S{i:03d}\n{seq}\n")
    with open(out_dir / "clusters.txt", "w") as fh:
        for i, c in enumerate(clusters):
            fh.write(f"S{i:03d}\t{c}\n")
    with open(out_dir / "expression.txt", "w") as fh:
        for i, value in enumerate(y):
            fh.write(f"S{i:03d}\t{value:.6f}\n")
    for name, cluster in (("positives.fa", 0), ("negatives.fa", 1)):
        with open(out_dir / name, "w") as fh:
            for i, c in enumerate(clusters):
                if c == cluster:
                    fh.write(f">S{i:03d}\n{seqs[i]}\n")
    with open(out_dir / "truth.json", "w") as fh:
        json.dump(truth, fh, indent=1)

    return truth


def main() -> None:
    default = Path(__file__).resolve().parent.parent / "kmer" / "tests" / "data" / "enrichment_reference"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default)
    args = parser.parse_args()
    truth = generate(args.out_dir)
    print(f"wrote dataset to {args.out_dir}")
    print(f"motif A carriers: {len(truth['motifs'][MOTIF_A]['carriers'])}")
    print(f"motif B carriers: {len(truth['motifs'][MOTIF_B]['carriers'])}")


if __name__ == "__main__":
    main()

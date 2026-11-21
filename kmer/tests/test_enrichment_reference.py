"""Reference tests for kmer.enrichment against official implementations.

The dataset in ``tests/data/enrichment_reference/`` was produced by
``scripts/generate_enrichment_dataset.py`` (seed 20071005): 120 random
150 bp sequences in two clusters, with ACGTTGCA planted in 70% of
cluster 0 (42 sequences, random position and strand) and TTTGGCCA in
65% of cluster 1 (39 sequences); the continuous target is
``y = 3*nA - 2*nB + N(0, 0.5)``.

``official_outputs.json`` records the outputs of the official
implementations of the reimplemented methods on this dataset:

- DREME 5.5.5 (MEME Suite): ``dreme -p positives.fa -n negatives.fa -k 8``
- HOMER v5.1: ``findMotifs.pl positives.fa fasta <dir> -fasta negatives.fa -len 8``
- FIRE (github.com/hanig/FIRE @ e7417852): ``fire.pl --expfiles=clusters.txt
  --exptype=discrete --fastafile_dna=seqs.fa``
- DRIMUST (github.com/YakhiniGroup/DRIMUST @ acacdf62):
  ``MotifSearchTester 5 9 ds 0.05 <ranked-sequences> <out> <clusters>``
- REDUCE Suite: distribution offline; no official output (see JSON).

These tests (1) verify the dataset is deterministic and matches its
ground truth, (2) verify the official tools recovered the planted
motifs, and (3) will compare the kmer.enrichment implementations to
the same ground truth once their statistics are implemented — those
tests activate automatically when NotImplementedError disappears.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
REF_DIR = os.path.join(HERE, "data", "enrichment_reference")
MOTIF_A, RC_A = "ACGTTGCA", "TGCAACGT"
MOTIF_B, RC_B = "TTTGGCCA", "TGGCCAAA"


def _read_fasta(path):
    seqs, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif name is not None:
                seqs[name] = line
    return seqs


def _truth():
    with open(os.path.join(REF_DIR, "truth.json")) as fh:
        return json.load(fh)


def _official():
    with open(os.path.join(REF_DIR, "official_outputs.json")) as fh:
        return json.load(fh)


class TestDataset:
    def test_regeneration_is_deterministic(self, tmp_path):
        script = os.path.join(HERE, "..", "..", "scripts",
                              "generate_enrichment_dataset.py")
        subprocess.run([sys.executable, script, "--out-dir", str(tmp_path)],
                       check=True, capture_output=True)
        for fname in ("seqs.fa", "clusters.txt", "expression.txt",
                      "positives.fa", "negatives.fa", "truth.json"):
            committed = open(os.path.join(REF_DIR, fname), "rb").read()
            regenerated = (tmp_path / fname).read_bytes()
            assert hashlib.md5(committed).hexdigest() == \
                hashlib.md5(regenerated).hexdigest(), fname

    def test_planted_carriers_match_truth(self):
        truth = _truth()
        seqs = _read_fasta(os.path.join(REF_DIR, "seqs.fa"))
        carriers_a = {n for n, s in seqs.items()
                      if MOTIF_A in s or RC_A in s}
        carriers_b = {n for n, s in seqs.items()
                      if MOTIF_B in s or RC_B in s}
        planted_a = {c["seq"] for c in truth["motifs"][MOTIF_A]["carriers"]}
        planted_b = {c["seq"] for c in truth["motifs"][MOTIF_B]["carriers"]}
        # every planted carrier is a carrier; random background noise may
        # add at most a couple of chance occurrences
        assert planted_a <= carriers_a and len(carriers_a) - len(planted_a) <= 2
        assert planted_b <= carriers_b and len(carriers_b) - len(planted_b) <= 2
        assert len(planted_a) == 42
        assert len(planted_b) == 39

    def test_planting_fractions(self):
        truth = _truth()
        assert truth["motifs"][MOTIF_A]["fraction"] == 0.70
        assert truth["motifs"][MOTIF_B]["fraction"] == 0.65
        assert truth["reverse_complements"][MOTIF_A] == RC_A
        assert truth["reverse_complements"][MOTIF_B] == RC_B

    def test_continuous_target_tracks_planted_motifs(self):
        seqs = _read_fasta(os.path.join(REF_DIR, "seqs.fa"))
        rows = [l.split() for l in
                open(os.path.join(REF_DIR, "expression.txt"))]
        y = np.array([float(v) for _, v in rows])
        n_a = np.array([seqs[n].count(MOTIF_A) + seqs[n].count(RC_A)
                        for n in sorted(seqs)])
        n_b = np.array([seqs[n].count(MOTIF_B) + seqs[n].count(RC_B)
                        for n in sorted(seqs)])
        y_sorted = np.array([float(dict(rows)[n]) for n in sorted(seqs)])
        truth = _truth()
        expected = (truth["target_model"]["beta_a"] * n_a +
                    truth["target_model"]["beta_b"] * n_b)
        assert np.corrcoef(y_sorted, expected)[0, 1] > 0.95


class TestOfficialReferences:
    def test_dreme_recovers_motif_a(self):
        motifs = _official()["dreme"]["motifs"]
        assert motifs[0]["seq"] == MOTIF_A
        assert int(motifs[0]["nsites"]) == 42
        assert float(motifs[0]["e_value"]) < 1e-10

    def test_homer_recovers_motif_a(self):
        motifs = _official()["homer"]["motifs"]
        top = motifs[0]
        assert top["consensus"] in (MOTIF_A, RC_A)
        assert float(top["p_value"].split("e")[0]) < 10  # p = 1e-18
        assert float(top["p_value"].split("e")[1]) <= -10
        assert top["targets_pct"] > 70
        assert top["background_pct"] < 5

    def test_fire_recovers_both_motifs(self):
        fire = _official()["fire"]
        signif = {m["motif"].strip("."): m for m in fire["significant_motifs"]}
        # FIRE optimizes 7-mer seeds; the reported motifs are cores of the
        # planted 8-mers
        assert any(key in RC_A or key in MOTIF_A for key in signif)
        assert any(key in MOTIF_B or key in RC_B for key in signif)
        assert all(float(m["z_score"]) > 40 for m in signif.values())
        per_cluster = {(m["motif"].strip("."), m["cluster"]): m
                       for m in fire["per_cluster"]}
        row = per_cluster[("TGCAACG", 0)]
        assert abs(row["fraction"] - 0.70) < 0.02
        row = per_cluster[("TTGGCCA", 1)]
        assert abs(row["fraction"] - 0.65) < 0.02

    def test_drimust_recovers_motif_a_at_top_ranks(self):
        top = _official()["drimust"]["top_motifs"][0]
        assert top["motif"] == RC_A
        assert top["rc"] == MOTIF_A
        assert top["corrected"] < 1e-30
        assert top["enrichment"] > 2.0
        # all motif occurrences concentrate at the top of the ranking
        assert top["b"] == top["B"] and top["n"] <= 43

    def test_reduce_official_unavailable_is_documented(self):
        assert _official()["reduce"]["status"].startswith("unavailable")


def _fit_or_skip(estimator, **fit_kwargs):
    try:
        estimator.fit(**fit_kwargs)
    except NotImplementedError:
        pytest.skip(f"{type(estimator).__name__} statistics not implemented yet")
    return estimator


class TestKmerEnrichmentAgainstGroundTruth:
    """Activate automatically once the statistics are implemented."""

    SEQ_LIST = list(_read_fasta(os.path.join(REF_DIR, "seqs.fa")).values())
    SEQ_NAMES = sorted(_read_fasta(os.path.join(REF_DIR, "seqs.fa")))
    SEQ_BY_NAME = _read_fasta(os.path.join(REF_DIR, "seqs.fa"))
    CLUSTERS = [int(l.split("\t")[1]) for l in
                open(os.path.join(REF_DIR, "clusters.txt"))]
    Y = [float(l.split("\t")[1]) for l in
         open(os.path.join(REF_DIR, "expression.txt"))]

    def _encoder(self):
        from kmer.encoders import SpectrumEncoder
        return SpectrumEncoder(k=8, canonical_rc=True)

    def _top_kmers(self, est, n=25):
        return {r.kmer for r in est.top(n)}

    def test_fire_finds_both_planted_motifs(self):
        from kmer.enrichment import FireEnrichment
        est = _fit_or_skip(FireEnrichment(self._encoder(), n_permutations=200),
                           seqs=self.SEQ_LIST, y=self.CLUSTERS)
        top = self._top_kmers(est)
        assert MOTIF_A in top and MOTIF_B in top

    def test_fisher_finds_motif_a(self):
        from kmer.enrichment import FisherEnrichment
        positives = [s for s, c in zip(self.SEQ_LIST, self.CLUSTERS) if c == 0]
        negatives = [s for s, c in zip(self.SEQ_LIST, self.CLUSTERS) if c == 1]
        est = _fit_or_skip(FisherEnrichment(self._encoder()),
                           positives=positives, negatives=negatives)
        assert est.results_[0].kmer == MOTIF_A
        assert est.results_[0].qvalue < 1e-6

    def test_reduce_finds_motif_a(self):
        from kmer.enrichment import ReduceEnrichment
        est = _fit_or_skip(ReduceEnrichment(self._encoder()),
                           seqs=self.SEQ_LIST, y=self.Y)
        top = self._top_kmers(est)
        assert MOTIF_A in top or RC_A in top

    def test_drimust_finds_motif_a(self):
        from kmer.enrichment import DrimustEnrichment
        est = _fit_or_skip(DrimustEnrichment(self._encoder()),
                           seqs=self.SEQ_LIST, y=self.Y)
        top = self._top_kmers(est)
        assert MOTIF_A in top or RC_A in top

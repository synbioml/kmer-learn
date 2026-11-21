#!/usr/bin/env python
"""Condense official-tool outputs into the enrichment reference fixture.

Parses the raw outputs produced by the official implementations of the
methods reimplemented in ``kmer.enrichment`` when run on the planted
dummy dataset (``scripts/generate_enrichment_dataset.py``) and writes
``official_outputs.json`` next to the dataset.

Official implementations and provenance:

- DREME 5.5.5 (MEME Suite) — dreme.xml
- HOMER v5.1 — homerResults.html + homerMotifs.motifs8
- FIRE (Elemento; github.com/hanig/FIRE @ e7417852, 2016-07-21) — clusters.txt.signif
- DRIMUST (YakhiniGroup/DRIMUST @ acacdf62, 2016-11-26) — mHG table
- REDUCE Suite (Bussemaker lab): distribution site is offline and no
  mirror exists as of 2026-11; no official output. ReduceEnrichment is
  instead validated against independent OLS/statistics references.

Usage::

    python scripts/condense_enrichment_references.py \
        --dreme-xml /tmp/ref_dreme/dreme.xml \
        --homer-dir /tmp/ref_homer \
        --fire-dir /tmp/fire_run/clusters.txt_FIRE/DNA \
        --drimust-tsv /tmp/ref_drimust.tsv
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import xml.etree.ElementTree as ET


def parse_dreme(path: Path) -> dict:
    root = ET.parse(path).getroot()
    motifs = []
    for m in root.iter("motif"):
        motifs.append({
            "id": m.get("id"),
            "seq": m.get("seq"),
            "length": m.get("length"),
            "nsites": m.get("nsites"),
            "e_value": m.get("evalue"),
            "p_value": m.get("pvalue"),
        })
    return {"tool": "DREME", "version": "5.5.5 (MEME Suite)", "motifs": motifs}


def parse_homer(homer_dir: Path) -> dict:
    # HOMER motif files carry everything in their FASTA-like headers:
    # >ACGTTGCA 1-ACGTTGCA,BestGuess:... T:45.0(75.00%),B:0.9(1.60%),P:1e-18
    motifs = []
    seen = set()
    for f in sorted((homer_dir / "homerResults").glob("motif*.motif")):
        header = f.read_text().splitlines()[0]
        consensus = header.split("\t")[0].lstrip(">")
        if consensus in seen:
            continue
        seen.add(consensus)
        stats = re.search(r"T:([\d.]+)\(([\d.]+)%\),B:([\d.]+)\(([\d.]+)%\),P:(\S+)", header)
        motifs.append({
            "consensus": consensus,
            "targets_pct": float(stats.group(2)),
            "background_pct": float(stats.group(4)),
            "p_value": stats.group(5),
        })
    return {"tool": "HOMER", "version": "v5.1", "motifs": motifs}


def parse_fire(fire_dir: Path) -> dict:
    signif = []
    for line in (fire_dir / "clusters.txt.signif").read_text().splitlines():
        a = line.split("\t")
        signif.append({"motif": a[0], "mi": float(a[1]), "unused": a[2],
                       "z_score": float(a[3]), "robustness": a[4], "seed": a[6]})
    per_cluster = []
    for line in (fire_dir / "clusters.txt.signif.motifs.rep").read_text().splitlines():
        a = line.split("\t")
        per_cluster.append({"motif": a[0], "cluster": int(a[1]),
                            "present": int(a[2]), "absent": int(a[3]),
                            "fraction": float(a[4]), "cluster_size": int(a[5])})
    return {"tool": "FIRE",
            "version": "github.com/hanig/FIRE e7417852 (2016-07-21)",
            "significant_motifs": signif, "per_cluster": per_cluster}


def parse_drimust(path: Path) -> dict:
    rows = []
    with open(path) as fh:
        header = fh.readline().split("\t")
        for i, line in enumerate(fh):
            if i >= 10:
                break
            a = line.rstrip("\n").split("\t")
            rows.append({
                "motif": a[0], "rc": a[1], "mhg_score": float(a[2]),
                "corrected": float(a[3]), "N": int(a[4]), "B": int(a[5]),
                "n": int(a[6]), "b": int(a[7]),
                "enrichment": float(a[8]),
            })
    return {"tool": "DRIMUST",
            "version": "github.com/YakhiniGroup/DRIMUST acacdf62 (2016-11-26)",
            "header": header[:9], "top_motifs": rows}


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    default_out = here / "kmer" / "tests" / "data" / "enrichment_reference"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dreme-xml", type=Path, required=True)
    p.add_argument("--homer-dir", type=Path, required=True)
    p.add_argument("--fire-dir", type=Path, required=True)
    p.add_argument("--drimust-tsv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=default_out / "official_outputs.json")
    args = p.parse_args()

    doc = {
        "dataset": "scripts/generate_enrichment_dataset.py (seed 20071005)",
        "generated": datetime.date.today().isoformat(),
        "note": "Planted motifs ACGTTGCA (70% of cluster 0) and TTTGGCCA "
                "(65% of cluster 1); continuous target y = 3*nA - 2*nB + N(0, 0.5).",
        "reduce": {
            "tool": "REDUCE Suite (Bussemaker lab)",
            "version": None,
            "status": "unavailable: official distribution site offline, "
                      "no mirror as of 2026-11; no official output recorded.",
        },
        "dreme": parse_dreme(args.dreme_xml),
        "homer": parse_homer(args.homer_dir),
        "fire": parse_fire(args.fire_dir),
        "drimust": parse_drimust(args.drimust_tsv),
    }
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

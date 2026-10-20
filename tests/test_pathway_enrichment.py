"""Tests for pathway enrichment module."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.pathway_enrichment import (
    IBD_PATHWAYS,
    pathway_enrichment,
    top_gene_pathway_coverage,
    pathway_summary,
)
from hidden_locus.core.validation import ValidationReport


def test_pathways_nonempty():
    assert len(IBD_PATHWAYS) >= 8
    assert "IL23_Th17_signaling" in IBD_PATHWAYS
    assert "NOD2_innate_immune" in IBD_PATHWAYS
    assert "autophagy_xenophagy" in IBD_PATHWAYS
    # Each pathway has at least 20 genes.
    for name, genes in IBD_PATHWAYS.items():
        assert len(genes) >= 20, f"{name} has only {len(genes)} genes"


def test_pathway_enrichment_basic():
    # 200 genes total, put IL23R/STAT3/IL12B in top 10.
    gene_scores = pd.DataFrame(
        {
            "gene_name": ["IL23R", "STAT3", "IL12B"] + [f"DECOY_{i}" for i in range(197)],
            "max_score": np.linspace(1.0, 0.0, 200),
        }
    )
    out = pathway_enrichment(gene_scores, universe_size=200, ks=(10, 20))
    assert not out.empty
    sub = out[out["pathway"] == "IL23_Th17_signaling"]
    assert not sub.empty
    k10 = sub[sub["k"] == 10].iloc[0]
    assert k10["n_hits"] >= 3  # IL23R, STAT3, IL12B all in pathway
    assert k10["enrichment"] > 1.0
    assert "IL23R" in k10["genes_found"]


def test_pathway_enrichment_empty():
    out = pathway_enrichment(pd.DataFrame())
    assert out.empty


def test_top_gene_pathway_coverage():
    top = ["IL23R", "STAT3", "NOD2", "ATG16L1", "UNKNOWN"]
    out = top_gene_pathway_coverage(top)
    assert not out.empty
    # NOD2 should appear in NOD2 pathway.
    nod2_row = out[out["pathway"] == "NOD2_innate_immune"].iloc[0]
    assert nod2_row["n_hits"] >= 1
    assert "NOD2" in nod2_row["genes"]


def test_pathway_summary_smoke():
    ranked = pd.DataFrame(
        {
            "gene_name": ["IL23R", "STAT3", "NOD2", "ATG16L1", "DECOY"],
            "model_score": [0.95, 0.9, 0.85, 0.8, 0.1],
        }
    )
    report = ValidationReport()
    out = pathway_summary(ranked, report, ks=(5,))
    assert not out.empty

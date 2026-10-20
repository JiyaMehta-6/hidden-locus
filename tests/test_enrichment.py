"""Tests for enrichment module (post-hoc truth-set validation)."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.enrichment import (
    IBD_CAUSAL_GENES,
    IBD_TRUTH_SYMBOLS,
    collapse_gene_scores,
    load_curated_ibd_truth,
    load_reported_genes,
    run_enrichment,
)


def test_curated_ibd_truth_nonempty():
    assert len(IBD_TRUTH_SYMBOLS) >= 30
    assert "NOD2" in IBD_TRUTH_SYMBOLS
    assert "IL23R" in IBD_TRUTH_SYMBOLS


def test_load_curated_ibd_truth():
    truth = load_curated_ibd_truth()
    assert truth == set(IBD_TRUTH_SYMBOLS)


def test_load_reported_genes_parses_semicolons():
    assoc = pd.DataFrame(
        {
            "trait": ["Crohn disease"] * 3,
            "pvalue": [1e-10, 1e-9, 0.5],
            "mapped_gene": ["NOD2; ATG16L1", "IL23R", "NOTCH1"],
        }
    )
    genes = load_reported_genes(assoc, r"(?i)crohn")
    assert "NOD2" in genes
    assert "ATG16L1" in genes
    assert "IL23R" in genes
    assert "NOTCH1" not in genes  # p=0.5 not significant


def test_collapse_gene_scores_max():
    df = pd.DataFrame(
        {
            "gene_name": ["A", "A", "B", "B", "C"],
            "model_score": [0.9, 0.5, 0.3, 0.8, 0.1],
            "rank": [1, 5, 3, 2, 10],
        }
    )
    collapsed = collapse_gene_scores(df)
    assert len(collapsed) == 3
    # Max score per gene.
    a_row = collapsed[collapsed["gene_name"] == "A"].iloc[0]
    assert a_row["max_score"] == 0.9
    b_row = collapsed[collapsed["gene_name"] == "B"].iloc[0]
    assert b_row["max_score"] == 0.8
    # Sorted descending by max_score.
    assert list(collapsed["gene_name"]) == ["A", "B", "C"]


def test_run_enrichment_basic():
    # 100 genes total, 10 truth genes, top-10 has 5 truth genes.
    gene_scores = pd.DataFrame(
        {
            "gene_name": [f"TRUTH_{i}" if i < 5 else f"DECOY_{i}" for i in range(100)],
            "max_score": np.linspace(1.0, 0.0, 100),
        }
    )
    truth = {f"TRUTH_{i}" for i in range(10)}
    out = run_enrichment(gene_scores, truth, universe_size=100, ks=(10, 25, 50))
    assert not out.empty
    assert list(out["k"]) == [10, 25, 50]
    top10 = out[out["k"] == 10].iloc[0]
    assert top10["n_truth"] == 5
    assert top10["enrichment"] > 1.0
    assert top10["pvalue"] < 0.05
    assert top10["odds_ratio"] > 1.0


def test_run_enrichment_empty_truth():
    gene_scores = pd.DataFrame({"gene_name": ["A", "B"], "max_score": [0.9, 0.1]})
    out = run_enrichment(gene_scores, set(), universe_size=10)
    assert out.empty


def test_run_enrichment_empty_scores():
    out = run_enrichment(pd.DataFrame(), {"A"}, universe_size=10)
    assert out.empty

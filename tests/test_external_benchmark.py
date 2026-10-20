"""Tests for external benchmark (head-to-head vs L2G/PoPS)."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.external_benchmark import (
    gene_level_auc,
    spearman_correlation,
    topk_jaccard,
    run_external_benchmark,
)
from hidden_locus.core.validation import ValidationReport


def test_spearman_correlation_aligned():
    ours = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5], index=["A", "B", "C", "D", "E"])
    theirs = pd.Series([0.95, 0.85, 0.75, 0.65, 0.55], index=["A", "B", "C", "D", "E"])
    out = spearman_correlation(ours, theirs)
    assert out["n_shared"] == 5
    assert out["spearman_rho"] > 0.9


def test_spearman_correlation_partial_overlap():
    ours = pd.Series([0.9, 0.8, 0.7, 0.6], index=["A", "B", "C", "D"])
    theirs = pd.Series([0.95, 0.85, 0.1, 0.2], index=["A", "B", "X", "Y"])
    out = spearman_correlation(ours, theirs)
    assert out["n_shared"] == 2


def test_topk_jaccard_full_overlap():
    s = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5], index=["A", "B", "C", "D", "E"])
    out = topk_jaccard(s, s, k=3)
    assert out["jaccard"] == 1.0
    assert out["n_overlap"] == 3


def test_topk_jaccard_no_overlap():
    ours = pd.Series([0.9, 0.8, 0.7], index=["A", "B", "C"])
    theirs = pd.Series([0.9, 0.8, 0.7], index=["X", "Y", "Z"])
    out = topk_jaccard(ours, theirs, k=3)
    assert out["jaccard"] == 0.0
    assert out["n_overlap"] == 0


def test_gene_level_auc():
    scores = pd.Series([0.9, 0.8, 0.1, 0.05], index=["NOD2", "IL23R", "GENE_X", "GENE_Y"])
    truth = {"NOD2", "IL23R"}
    out = gene_level_auc(scores, truth)
    assert out["auc"] == 1.0
    assert out["n_positives"] == 2


def test_gene_level_auc_no_positives():
    scores = pd.Series([0.9, 0.8], index=["A", "B"])
    out = gene_level_auc(scores, set())
    assert np.isnan(out["auc"])


def test_run_external_benchmark_basic():
    ranked = pd.DataFrame(
        {
            "gene_name": ["NOD2", "IL23R", "DECOY1", "DECOY2"],
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "model_score": [0.9, 0.8, 0.2, 0.1],
        }
    )
    l2g = pd.DataFrame(
        {
            "chrom": ["1", "1", "1", "1"],
            "pos": [100, 100, 100, 100],
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "l2g_score": [0.95, 0.85, 0.15, 0.05],
        }
    )
    report = ValidationReport()
    out = run_external_benchmark(
        ranked, l2g, pops_scores=None, curated_truth={"NOD2", "IL23R"}, report=report
    )
    assert not out.empty
    assert "comparator" in out.columns
    assert "metric" in out.columns
    assert "opentargets_l2g" in out["comparator"].values
    assert "hidden_locus" in out["comparator"].values


def test_run_external_benchmark_no_data():
    report = ValidationReport()
    out = run_external_benchmark(pd.DataFrame(), None, None, None, report)
    assert out.empty

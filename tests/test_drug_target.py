"""Tests for drug-target overlap module."""

import pandas as pd
import pytest

from hidden_locus.core.drug_target_overlap import (
    IBD_DRUG_TARGETS,
    drug_target_overlap,
    drug_target_summary,
)
from hidden_locus.core.validation import ValidationReport


def test_drug_targets_nonempty():
    assert len(IBD_DRUG_TARGETS) >= 15
    assert "TNF" in IBD_DRUG_TARGETS
    assert "IL23R" in IBD_DRUG_TARGETS
    assert "JAK1" in IBD_DRUG_TARGETS
    # Each target has drugs and mechanism.
    for gene, info in IBD_DRUG_TARGETS.items():
        assert "drugs" in info
        assert "mechanism" in info
        assert len(info["drugs"]) >= 1


def test_drug_target_overlap_finds_matches():
    ranked = pd.DataFrame(
        {
            "gene_name": ["TNF", "IL23R", "JAK1", "UNKNOWN1", "UNKNOWN2"],
            "model_score": [0.95, 0.9, 0.85, 0.1, 0.05],
            "rank": [1, 2, 3, 4, 5],
        }
    )
    out = drug_target_overlap(ranked, top_n=10)
    assert not out.empty
    assert "TNF" in out["gene_name"].values
    assert "IL23R" in out["gene_name"].values
    assert "JAK1" in out["gene_name"].values


def test_drug_target_overlap_no_matches():
    ranked = pd.DataFrame(
        {
            "gene_name": ["FAKE1", "FAKE2"],
            "model_score": [0.9, 0.1],
            "rank": [1, 2],
        }
    )
    out = drug_target_overlap(ranked, top_n=10)
    assert out.empty


def test_drug_target_summary():
    ranked = pd.DataFrame(
        {
            "gene_name": ["TNF", "IL23R", "FAKE"],
            "model_score": [0.95, 0.9, 0.1],
            "rank": [1, 2, 3],
        }
    )
    report = ValidationReport()
    summary = drug_target_summary(ranked, report, top_n=10)
    assert summary["n_drug_targets_total"] >= 15
    assert summary["n_drug_targets_in_top"] == 2
    assert summary["drugs_covered"] >= 2

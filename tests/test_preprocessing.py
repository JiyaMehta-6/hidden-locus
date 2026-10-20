"""Tests for GWAS association loading, locus assignment and user input."""

import zipfile

import pandas as pd
import pytest

from hidden_locus.core.preprocessing import (
    assign_loci,
    deduplicate_variants,
    select_index_variants,
    variants_from_user_frame,
)
from hidden_locus.core.validation import ValidationReport


def _assoc_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_select_index_variants_filters_trait_and_pvalue():
    rep = ValidationReport()
    assoc = _assoc_frame(
        [
            {"trait": "Crohn disease", "chrom_raw": "1", "pos_raw": "1000",
             "pvalue": "1e-10", "risk_raw": "rs1-A", "snp_raw": "rs1",
             "pubmed": "1", "pvalue_num": None},
            {"trait": "Crohn disease", "chrom_raw": "1", "pos_raw": "2000",
             "pvalue": "0.5", "risk_raw": "rs2-A", "snp_raw": "rs2",
             "pubmed": "1"},
            {"trait": "Height", "chrom_raw": "2", "pos_raw": "3000",
             "pvalue": "1e-12", "risk_raw": "rs3-A", "snp_raw": "rs3",
             "pubmed": "2"},
        ]
    )
    out = select_index_variants(assoc, r"(?i)crohn", p_threshold=5e-8, report=rep)
    assert len(out) == 1
    assert out.iloc[0]["chrom"] == "1"
    assert out.iloc[0]["rsid"] == "rs1"


def test_select_index_variants_multi_position_cells():
    rep = ValidationReport()
    assoc = _assoc_frame(
        [{"trait": "Crohn disease", "chrom_raw": "1,2", "pos_raw": "1000, 2000",
          "pvalue": "1e-10", "risk_raw": "rs1-A", "snp_raw": "rs1", "pubmed": "1"}]
    )
    out = select_index_variants(assoc, r"(?i)crohn", report=rep)
    # only first chrom/pos pair kept
    assert len(out) == 1
    assert out.iloc[0]["chrom"] == "1"
    assert int(out.iloc[0]["pos"]) == 1000


def test_deduplicate_variants_keeps_strongest_p():
    rep = ValidationReport()
    df = pd.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "pos": [100, 100, 300],
            "rsid": ["rs1", "rs1", "rs3"],
            "risk_allele": ["A", "A", "G"],
            "pvalue": [1e-6, 1e-9, 1e-8],
            "trait": ["Crohn"] * 3,
            "pubmed": ["111", "222", "333"],
        }
    )
    out = deduplicate_variants(df, rep)
    assert len(out) == 2
    row = out[out["variant_uid"] == "1:100"].iloc[0]
    assert row["pvalue"] == 1e-9
    assert "111" in row["pubmed"] and "222" in row["pubmed"]


def test_deduplicate_variants_allele_follows_strongest_row():
    rep = ValidationReport()
    df = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [100, 100],
            "rsid": ["rs_weak", "rs_strong"],
            "risk_allele": ["A", "T"],
            "risk_strand": ["+", "+"],
            "pvalue": [1e-6, 1e-9],
            "trait": ["Crohn", "Crohn"],
            "pubmed": ["1", "2"],
        }
    )
    out = deduplicate_variants(df, rep)
    row = out[out["variant_uid"] == "1:100"].iloc[0]
    assert row["pvalue"] == 1e-9
    assert row["rsid"] == "rs_strong"
    assert row["risk_allele"] == "T"


def test_assign_loci_gap_clustering():
    df = pd.DataFrame(
        {
            "chrom": ["1", "1", "1", "1"],
            "pos": [1000, 2000, 10_000_000, 10_000_500],
            "pvalue": [1e-9] * 4,
        }
    )
    out = assign_loci(df, gap_bp=500_000)
    assert out["locus_id"].iloc[0] == out["locus_id"].iloc[1]
    assert out["locus_id"].iloc[1] != out["locus_id"].iloc[2]
    assert out["locus_id"].iloc[2] == out["locus_id"].iloc[3]


def test_assign_loci_empty():
    out = assign_loci(pd.DataFrame(columns=["chrom", "pos", "pvalue"]))
    assert "locus_id" in out.columns
    assert out.empty


def test_variants_from_user_frame():
    frame = pd.DataFrame({"chrom": ["1", "X"], "pos": [100, 200], "rsid": ["rs1", "rs2"],
                          "alt": ["A", "G"]})
    out = variants_from_user_frame(frame, pvalue=1e-8)
    assert len(out) == 2
    assert out.iloc[0]["variant_uid"] == "1:100"
    assert out.iloc[1]["chrom"] == "X"
    assert out.iloc[0]["study"] == "user_input"


def test_variants_from_user_frame_drops_invalid_positions():
    frame = pd.DataFrame(
        {
            "chrom": ["1", "2", "3", None],
            "pos": ["100", "not_a_pos", 0, "300"],
            "rsid": ["rs1", "rs2", "rs3", "rs4"],
            "alt": ["A", "C", "G", "T"],
        }
    )
    out = variants_from_user_frame(frame)
    assert out["variant_uid"].tolist() == ["1:100"]

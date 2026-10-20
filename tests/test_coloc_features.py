"""Tests for coloc features integration."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.features import (
    COLOC_FEATURES,
    FeatureContext,
    GeneAnnotation,
    build_feature_table,
    coloc_feature_columns,
    feature_columns,
    finalize_features,
)


@pytest.fixture
def small_genes() -> GeneAnnotation:
    genes = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_name": ["GENE_A", "GENE_B"],
            "chrom": ["1", "1"],
            "start": [100_000, 500_000],
            "end": [200_000, 600_000],
            "strand": ["+", "-"],
            "biotype": ["protein_coding", "protein_coding"],
            "tss": [100_000, 600_000],
        }
    )
    return GeneAnnotation(genes=genes, exons_by_gene={})


@pytest.fixture
def coloc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_uid": ["1:150000", "1:150000", "1:550000"],
            "gene_id": ["ENSG1", "ENSG2", "ENSG2"],
            "coloc_eqtl_h4": [0.95, 0.3, 0.8],
            "coloc_eqtl_clpp": [0.1, 0.01, 0.05],
            "coloc_sqtl_h4": [0.5, 0.1, 0.2],
            "coloc_pqtl_h4": [0.2, 0.05, 0.1],
        }
    )


def test_coloc_features_defined():
    assert "coloc_eqtl_h4" in COLOC_FEATURES
    assert "coloc_missing" in COLOC_FEATURES


def test_coloc_features_merged(small_genes, coloc_df):
    candidates = pd.DataFrame(
        {
            "variant_uid": ["1:150000", "1:150000", "1:550000"],
            "chrom": ["1", "1", "1"],
            "pos": [150_000, 150_000, 550_000],
            "rsid": ["rs1", "rs1", "rs2"],
            "risk_allele": ["T", "T", "G"],
            "pvalue": [1e-10, 1e-10, 1e-9],
            "trait": ["Crohn"] * 3,
            "pubmed": ["123"] * 3,
            "study": ["GWAS"] * 3,
            "locus_id": ["chr1_1", "chr1_1", "chr1_1"],
            "gene_id": ["ENSG1", "ENSG2", "ENSG2"],
            "gene_name": ["GENE_A", "GENE_B", "GENE_B"],
            "gene_biotype": ["protein_coding"] * 3,
            "gene_strand": ["+", "-", "-"],
            "gene_tss": [100_000, 600_000, 600_000],
            "gene_start": [100_000, 500_000, 500_000],
            "gene_end": [200_000, 600_000, 600_000],
            "dist_tss": [50_000, -50_000, -50_000],
        }
    )
    ctx = FeatureContext(genes=small_genes, ccre=None, coloc=coloc_df, coloc_loaded=True)
    ft = build_feature_table(candidates, ctx, tissues=[])
    assert "coloc_eqtl_h4" in ft.columns
    assert "coloc_missing" in ft.columns
    # First candidate should have h4=0.95.
    row0 = ft[(ft["variant_uid"] == "1:150000") & (ft["gene_id"] == "ENSG1")].iloc[0]
    assert row0["coloc_eqtl_h4"] == pytest.approx(0.95)
    assert row0["coloc_missing"] == 0


def test_coloc_missing_when_absent(small_genes):
    candidates = pd.DataFrame(
        {
            "variant_uid": ["1:150000"],
            "chrom": ["1"],
            "pos": [150_000],
            "rsid": ["rs1"],
            "risk_allele": ["T"],
            "pvalue": [1e-10],
            "trait": ["Crohn"],
            "pubmed": ["123"],
            "study": ["GWAS"],
            "locus_id": ["chr1_1"],
            "gene_id": ["ENSG1"],
            "gene_name": ["GENE_A"],
            "gene_biotype": ["protein_coding"],
            "gene_strand": ["+"],
            "gene_tss": [100_000],
            "gene_start": [100_000],
            "gene_end": [200_000],
            "dist_tss": [50_000],
        }
    )
    ctx = FeatureContext(genes=small_genes, ccre=None, coloc_loaded=False)
    ft = build_feature_table(candidates, ctx, tissues=[])
    assert ft["coloc_missing"].iloc[0] == 1
    assert ft["coloc_eqtl_h4"].iloc[0] == 0.0


def test_coloc_feature_columns_included_in_model(small_genes, coloc_df):
    candidates = pd.DataFrame(
        {
            "variant_uid": ["1:150000"],
            "chrom": ["1"],
            "pos": [150_000],
            "rsid": ["rs1"],
            "risk_allele": ["T"],
            "pvalue": [1e-10],
            "trait": ["Crohn"],
            "pubmed": ["123"],
            "study": ["GWAS"],
            "locus_id": ["chr1_1"],
            "gene_id": ["ENSG1"],
            "gene_name": ["GENE_A"],
            "gene_biotype": ["protein_coding"],
            "gene_strand": ["+"],
            "gene_tss": [100_000],
            "gene_start": [100_000],
            "gene_end": [200_000],
            "dist_tss": [50_000],
        }
    )
    ctx = FeatureContext(genes=small_genes, ccre=None, coloc=coloc_df, coloc_loaded=True)
    ft = build_feature_table(candidates, ctx, tissues=[])
    ft = finalize_features(ft)
    cols = coloc_feature_columns(ft)
    assert "coloc_eqtl_h4" in cols
    assert "coloc_missing" in cols
    all_features = feature_columns(ft)
    assert "coloc_eqtl_h4" in all_features

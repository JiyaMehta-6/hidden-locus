"""Shared fixtures for Hidden Locus tests (synthetic, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.features import CcreAnnotation, FeatureContext, GeneAnnotation
from hidden_locus.core.intervals import IntervalIndex


@pytest.fixture
def small_genes() -> GeneAnnotation:
    genes = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "gene_name": ["GENE_A", "GENE_B", "GENE_C"],
            "chrom": ["1", "1", "2"],
            "start": [100_000, 500_000, 300_000],
            "end": [200_000, 600_000, 400_000],
            "strand": ["+", "-", "+"],
            "biotype": ["protein_coding", "lncRNA", "protein_coding"],
            "tss": [100_000, 600_000, 300_000],
        }
    )
    exons = {
        "ENSG1": (np.array([110_000, 150_000]), np.array([111_000, 151_000])),
        "ENSG2": (np.array([520_000]), np.array([521_000])),
    }
    return GeneAnnotation(genes=genes, exons_by_gene=exons)


@pytest.fixture
def small_ccre() -> CcreAnnotation:
    return CcreAnnotation(
        index=IntervalIndex(
            chrom=np.array(["1", "1"], dtype=object),
            start=np.array([100_500, 510_000], dtype=np.int64),
            end=np.array([100_800, 511_000], dtype=np.int64),
        ),
        classes=np.array(["dELS", "pELS"], dtype=object),
        accessions=np.array(["EH1", "EH2"], dtype=object),
    )


@pytest.fixture
def synthetic_variants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_uid": ["1:100600", "2:300100"],
            "chrom": ["1", "2"],
            "pos": [100_600, 300_100],
            "rsid": ["rs1", "rs2"],
            "risk_allele": ["T", "G"],
            "pvalue": [1e-10, 5e-9],
            "trait": ["Crohn disease"] * 2,
            "pubmed": ["123456"] * 2,
            "study": ["GWAS"] * 2,
            "locus_id": ["chr1_1", "chr2_1"],
        }
    )


@pytest.fixture
def labelled_feature_table() -> pd.DataFrame:
    """A small deterministic table with labels for evaluation tests."""
    rng = np.random.RandomState(0)
    n = 400
    loci = [f"chr1_{i % 8 + 1}" for i in range(n)]
    score = rng.rand(n)
    df = pd.DataFrame(
        {
            "variant_uid": [f"1:{1000 + i}" for i in range(n)],
            "chrom": "1",
            "pos": np.arange(1000, 1000 + n),
            "rsid": None,
            "gene_id": [f"ENSG{i:04d}" for i in range(n)],
            "gene_name": [f"GENE{i}" for i in range(n)],
            "locus_id": loci,
            "pvalue": 1e-8,
            "trait": "Crohn disease",
            "consequence": rng.choice(
                ["exon", "intron", "promoter", "proximal", "distal"], n
            ),
            "ccre_class": rng.choice(
                ["none", "dELS", "pELS", "PLS", "CTCF-only", "DNase-H3K4me3"], n
            ),
            "ccre_dist": rng.randint(0, 5000, n).astype(float),
            "dist_tss": rng.randint(-1_000_000, 1_000_000, n),
            "abs_log10_dist_tss": rng.rand(n) * 5,
            "log10_dist_body": rng.rand(n) * 4,
            "is_protein_coding": rng.randint(0, 2, n),
            "gwas_log10_p": rng.rand(n) * 10,
            "gwas_p_missing": 0,
            "pip": rng.rand(n),
            "pip_missing": 0,
            "regulatory_missing": 0,
            "gtex_missing": 0,
            "gtex_n_tissues": rng.randint(0, 4, n),
            "eqtl_direction_conflict": 0,
            "label": (score > 0.7).astype(int),
            "label_pvalue": np.where(score > 0.7, 1e-6, np.nan),
            "label_study": "BLUEPRINT",
        }
    )
    return df

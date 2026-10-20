"""Tests for candidate generation, feature building and leakage guards."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.features import (
    FeatureContext,
    assert_no_leakage,
    attach_labels,
    build_feature_table,
    feature_columns,
    finalize_features,
    generate_candidates,
    parse_variant_ids,
    strip_version,
)
from hidden_locus.core.validation import ValidationReport


def test_strip_version():
    assert strip_version("ENSG00000123456.15") == "ENSG00000123456"
    assert strip_version("ENSG00000123456") == "ENSG00000123456"


def test_parse_variant_ids_gtex_and_catalogue():
    s = pd.Series(["19_226776_C_T_b38", "chr19_226776_C_T", "chr1_100_A_G"])
    out = parse_variant_ids(s)
    assert out["chrom"].tolist() == ["19", "19", "1"]
    assert out["pos"].tolist() == [226776, 226776, 100]
    assert out["ref"].tolist() == ["C", "C", "A"]
    assert out["alt"].tolist() == ["T", "T", "G"]


def test_parse_variant_ids_garbage():
    out = parse_variant_ids(pd.Series(["not_a_variant", None]))
    assert out["chrom"].isna().all()


def test_generate_candidates_window(synthetic_variants, small_genes):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    assert not cands.empty
    # variant 1:100600 near GENE_A (tss 100000, +) and GENE_B (tss 600000, -)
    genes_for_v1 = set(cands[cands["variant_uid"] == "1:100600"]["gene_name"])
    assert "GENE_A" in genes_for_v1
    assert "GENE_B" in genes_for_v1
    # variant 2:300100 near GENE_C only (different chromosome for the others)
    genes_for_v2 = set(cands[cands["variant_uid"] == "2:300100"]["gene_name"])
    assert genes_for_v2 == {"GENE_C"}


def test_generate_candidates_empty_variants(small_genes):
    empty = pd.DataFrame(columns=["variant_uid", "chrom", "pos"])
    out = generate_candidates(empty, small_genes)
    assert out.empty


def test_signed_distance_direction(synthetic_variants, small_genes):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    ga = cands[cands["gene_name"] == "GENE_A"].iloc[0]
    # variant downstream of + strand gene -> positive
    assert ga["dist_tss"] == 100_600 - 100_000
    gb = cands[cands["gene_name"] == "GENE_B"].iloc[0]
    # variant 100600 relative to GENE_B tss 600000 on - strand:
    # delta = 100600 - 600000 < 0, strand flips -> positive (downstream for -)
    assert gb["dist_tss"] == (600_000 - 100_600)


def test_build_feature_table_smoke(synthetic_variants, small_genes, small_ccre):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    ctx = FeatureContext(genes=small_genes, ccre=small_ccre,
                         gtex_loaded=False, regulatory_loaded=True)
    ft = build_feature_table(cands, ctx)
    assert "consequence" in ft.columns
    assert set(ft["consequence"]) <= {
        "exon", "intron", "promoter", "proximal", "distal", "gene_body"
    }
    assert ft["regulatory_missing"].eq(0).all()
    assert ft["gtex_missing"].eq(1).all()
    assert ft["pip_missing"].eq(1).all()
    # promoter consequence for variants near a TSS
    assert (ft["consequence"] == "promoter").sum() >= 1
    # one variant falls inside the first cCRE [100500,100800)
    assert (ft["ccre_class"] != "none").sum() >= 1


def test_build_feature_table_with_gtex(synthetic_variants, small_genes, small_ccre):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    pairs = pd.DataFrame(
        {
            "variant_uid": ["1:100600"],
            "gene_id": ["ENSG1"],
            "pval_nominal": [1e-12],
            "slope": [0.5],
            "ref": ["A"],
            "alt": ["T"],
        }
    )
    egenes = pd.DataFrame({"gene_id": ["ENSG1"], "qval": [1e-8]})
    ctx = FeatureContext(
        genes=small_genes, ccre=small_ccre,
        gtex={"Whole_Blood": pairs}, gtex_egenes={"Whole_Blood": egenes},
        gtex_loaded=True, regulatory_loaded=True,
    )
    ft = build_feature_table(cands, ctx, tissues=["Whole_Blood"])
    hit = ft[(ft["variant_uid"] == "1:100600") & (ft["gene_id"] == "ENSG1")].iloc[0]
    assert hit["gtex_pair_whole_blood"] == 1
    assert hit["gtex_log10p_whole_blood"] > 10
    assert hit["allele_match"] == "allele_verified"
    miss = ft[ft["gene_id"] == "ENSG2"].iloc[0]
    assert miss["gtex_pair_whole_blood"] == 0


def test_finalize_and_leakage_guard(synthetic_variants, small_genes, small_ccre):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    ctx = FeatureContext(genes=small_genes, ccre=small_ccre,
                         gtex_loaded=False, regulatory_loaded=True)
    ft = build_feature_table(cands, ctx)
    ft = finalize_features(ft)
    cols = feature_columns(ft)
    assert_no_leakage(ft, cols)
    # explicit failure when an identifier sneaks in
    with pytest.raises(ValueError, match="leaked"):
        assert_no_leakage(ft, cols + ["gene_name"])
    # category dummies exist
    for level in ("conseq_exon", "conseq_promoter", "ccre_dELS"):
        assert level in cols


def test_attach_labels(synthetic_variants, small_genes, small_ccre):
    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    labels = pd.DataFrame(
        {"variant_uid": ["1:100600"], "gene_id": ["ENSG1"], "pvalue": [1e-7]}
    )
    out = attach_labels(cands, labels, label_study="test_study", p_threshold=1e-5)
    assert out["label"].sum() == 1
    labelled_row = out[out["label"] == 1].iloc[0]
    assert labelled_row["gene_id"] == "ENSG1"
    assert labelled_row["label_study"] == "test_study"
    assert out["label_pvalue"].isna().sum() == len(out) - 1


def test_attach_labels_empty():
    cands = pd.DataFrame({"variant_uid": ["1:1"], "gene_id": ["E1"]})
    out = attach_labels(cands, pd.DataFrame(), label_study="x", p_threshold=1e-5)
    assert out["label"].eq(0).all()


def test_attach_labels_respects_p_threshold():
    cands = pd.DataFrame({"variant_uid": ["1:1", "1:2"], "gene_id": ["E1", "E2"]})
    labels = pd.DataFrame(
        {"variant_uid": ["1:1", "1:2"], "gene_id": ["E1", "E2"], "pvalue": [1e-9, 1e-3]}
    )
    out = attach_labels(cands, labels, label_study="x", p_threshold=1e-5)
    assert out["label"].tolist() == [1, 0]


def test_ccre_dist_finite_when_chrom_absent(synthetic_variants, small_genes, small_ccre):
    # small_ccre only covers chrom 1; variant 2:300100 must not get dist 0.
    from hidden_locus.core.features import CCRE_DIST_UNOBSERVED

    cands = generate_candidates(synthetic_variants, small_genes, window_bp=1_000_000)
    ctx = FeatureContext(genes=small_genes, ccre=small_ccre,
                         gtex_loaded=False, regulatory_loaded=True)
    ft = build_feature_table(cands, ctx)
    chrom2 = ft[ft["chrom"] == "2"]
    assert not chrom2.empty
    assert (chrom2["ccre_dist"] == CCRE_DIST_UNOBSERVED).all()


def test_load_label_eqtl_cache_keyed_by_uid_set(tmp_path, monkeypatch):
    import gzip

    from hidden_locus import config
    from hidden_locus.core.features import load_label_eqtl

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    pairs = tmp_path / "labels.tsv.gz"
    perm = tmp_path / "perm.tsv.gz"
    with gzip.open(pairs, "wt") as fh:
        fh.write("chrom\tposition\tref\talt\tvariant\tpvalue\tgene_id\n")
        fh.write("1\t100\tA\tG\tchr1_100_A_G\t1e-8\tENSG1.5\n")
    with gzip.open(perm, "wt") as fh:
        fh.write("molecular_trait_id\tp_beta\n")
        fh.write("ENSG1\t0.01\n")

    rep1 = ValidationReport()
    p1, _ = load_label_eqtl(pairs, perm, {"1:100"}, 1e-5, rep1)
    assert len(p1) == 1
    # Same uids -> cache hit.
    rep2 = ValidationReport()
    load_label_eqtl(pairs, perm, {"1:100"}, 1e-5, rep2)
    assert any(i.code == "labels_cache" for i in rep2.issues)
    # Different candidate set -> cache must NOT be reused.
    rep3 = ValidationReport()
    p3, _ = load_label_eqtl(pairs, perm, {"1:999"}, 1e-5, rep3)
    assert not any(i.code == "labels_cache" for i in rep3.issues)
    assert p3.empty


def test_load_label_eqtl_warns_when_egenes_unavailable(tmp_path, monkeypatch):
    import gzip

    from hidden_locus import config
    from hidden_locus.core.features import load_label_eqtl

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    pairs = tmp_path / "labels.tsv.gz"
    perm = tmp_path / "perm.tsv.gz"
    with gzip.open(pairs, "wt") as fh:
        fh.write("chrom\tposition\tref\talt\tvariant\tpvalue\tgene_id\n")
        fh.write("1\t100\tA\tG\tchr1_100_A_G\t1e-8\tENSG1\n")
    with gzip.open(perm, "wt") as fh:
        fh.write("something_else\n")  # no gene / p_beta columns

    rep = ValidationReport()
    p, egenes = load_label_eqtl(pairs, perm, {"1:100"}, 1e-5, rep)
    assert egenes == set()
    assert any(i.code == "label_egenes_filter_missing" for i in rep.warnings)


def test_finalize_features_encodes_hyphenated_ccre_classes():
    df = pd.DataFrame(
        {
            "consequence": ["distal", "distal"],
            "ccre_class": ["CTCF-only", "DNase-H3K4me3"],
        }
    )
    out = finalize_features(df)
    assert out["ccre_CTCF_only"].tolist() == [1, 0]
    assert out["ccre_DNase_H3K4me3"].tolist() == [0, 1]
    assert "ccre_CTCF-only" not in out.columns

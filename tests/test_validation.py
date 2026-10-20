"""Tests for validation, allele harmonization and input guards."""

import pandas as pd
import pytest

from hidden_locus.core.validation import (
    ValidationError,
    ValidationReport,
    check_assembly,
    harmonize_allele_pair,
    normalize_chrom,
    parse_strongest_risk_allele,
    validate_coordinates,
)


def test_normalize_chrom_aliases():
    assert normalize_chrom("chr1") == "1"
    assert normalize_chrom("CHR1") == "1"
    assert normalize_chrom("1") == "1"
    assert normalize_chrom("X") == "X"
    assert normalize_chrom("chr23") is None
    assert normalize_chrom(None) is None


def test_check_assembly_supported():
    rep = ValidationReport()
    assert check_assembly("GRCh38", rep) == "GRCh38"
    assert check_assembly("hg38", rep) == "GRCh38"
    assert not rep.errors


def test_check_assembly_unsupported_errors():
    rep = ValidationReport()
    check_assembly("GRCh37", rep)
    assert rep.errors


def test_validate_coordinates_drops_bad_rows():
    rep = ValidationReport()
    df = pd.DataFrame(
        {
            "chrom": ["1", None, "1"],
            "pos": [100, 200, -5],
            "ref": ["A", "A", "A"],
            "alt": ["G", "G", "G"],
        }
    )
    out = validate_coordinates(df, rep)
    assert len(out) == 1
    assert int(out.iloc[0]["pos"]) == 100
    assert rep.warnings


def test_harmonize_allele_verified():
    assert harmonize_allele_pair("G", "A", "G") == "allele_verified"
    assert harmonize_allele_pair("A", "A", "G") == "allele_verified"


def test_harmonize_allele_strand_flip():
    assert harmonize_allele_pair("T", "A", "G") == "strand_flip"
    assert harmonize_allele_pair("C", "A", "G") == "strand_flip"


def test_harmonize_allele_mismatch():
    assert harmonize_allele_pair("A", "C", "G") == "allele_mismatch"


def test_harmonize_allele_missing():
    assert harmonize_allele_pair(None, "A", "G") == "position_only"
    assert harmonize_allele_pair("G", None, None) == "position_only"


def test_parse_strongest_risk_allele_formats():
    # current GWAS Catalog format: rsid-allele, optional -? (allele unstated)
    rsid, allele, strand = parse_strongest_risk_allele("rs10089517-A")
    assert rsid == "rs10089517" and allele == "A" and strand is None
    rsid, allele, strand = parse_strongest_risk_allele("rs999-T")
    assert rsid == "rs999" and allele == "T"
    rsid, allele, strand = parse_strongest_risk_allele("rs1-?")
    assert rsid == "rs1" and allele is None
    # multi-rsid and positional cells are not parsed (position-only fallback)
    assert parse_strongest_risk_allele("rs1; rs2-A") == (None, None, None)
    assert parse_strongest_risk_allele("chr12:111446804-?") == (None, None, None)
    assert parse_strongest_risk_allele(None) == (None, None, None)
    assert parse_strongest_risk_allele("nonsense") == (None, None, None)

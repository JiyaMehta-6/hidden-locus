"""Tests for dataset file bookkeeping (no network)."""

import pytest

from hidden_locus import config
from hidden_locus.core import datasets as ds


def test_local_files_excludes_partial_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path)
    spec = ds.DatasetSpec(
        key="fake_ds",
        name="fake",
        purpose="test",
        source="test",
        license_note="test",
        approx_size="1 B",
        url="https://example.invalid/fake.tsv.gz",
    )
    d = tmp_path / "fake_ds"
    d.mkdir()
    (d / "fake.tsv.gz").write_bytes(b"data")
    (d / "fake.tsv.gz.part").write_bytes(b"partial")
    (d / ".meta.json").write_text("{}")
    files = ds.local_files(spec)
    assert [p.name for p in files] == ["fake.tsv.gz"]


def test_require_ignores_partial_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path)
    monkeypatch.setitem(ds.DATASET_INDEX, "fake_ds", ds.DatasetSpec(
        key="fake_ds",
        name="fake",
        purpose="test",
        source="test",
        license_note="test",
        approx_size="1 B",
        url="https://example.invalid/fake.tsv.gz",
    ))
    d = tmp_path / "fake_ds"
    d.mkdir()
    (d / "fake.tsv.gz.part").write_bytes(b"partial only")
    try:
        raised = False
        ds.require("fake_ds")
    except ds.DatasetError:
        raised = True
    assert raised


def test_dataset_status_zero_byte_files_are_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path)
    spec = ds.DatasetSpec(
        key="fake_ds",
        name="fake",
        purpose="test",
        source="test",
        license_note="test",
        approx_size="1 B",
        url="https://example.invalid/fake.tsv.gz",
    )
    d = tmp_path / "fake_ds"
    d.mkdir()
    (d / "fake.tsv.gz").write_bytes(b"")
    assert ds.dataset_status(spec)["state"] == "incomplete"


def test_extract_gtex_tissues_replaces_truncated_files(tmp_path, monkeypatch):
    import gzip
    import io
    import tarfile

    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path)
    spec_dir = tmp_path / "gtex_eqtl"
    spec_dir.mkdir()
    # Tiny tar with the two Whole_Blood members.
    tar_path = spec_dir / "GTEx_Analysis_v8_eQTL.tar"
    with tarfile.open(tar_path, "w") as tar:
        for member_name, payload in (
            ("GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz",
             b"pairs-bytes"),
            ("GTEx_Analysis_v8_eQTL/Whole_Blood.v8.egenes.txt.gz",
             b"egenes-bytes"),
        ):
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    # Pre-plant a truncated (zero-byte) extracted file: must be re-extracted.
    ex_dir = spec_dir / "extracted"
    ex_dir.mkdir()
    (ex_dir / "Whole_Blood.signif_variant_gene_pairs.txt.gz").write_bytes(b"")
    out = ds.extract_gtex_tissues(["Whole_Blood"])
    assert len(out) == 2
    pairs = ex_dir / "Whole_Blood.signif_variant_gene_pairs.txt.gz"
    egenes = ex_dir / "Whole_Blood.egenes.txt.gz"
    assert pairs.read_bytes() == b"pairs-bytes"
    assert egenes.read_bytes() == b"egenes-bytes"
    # No stray .part files left behind.
    assert not list(ex_dir.glob("*.part"))


def test_load_pips_normalizes_float_chrom_pos(tmp_path, monkeypatch):
    import pandas as pd

    from hidden_locus import config
    from hidden_locus.core.pipeline import _load_pips
    from hidden_locus.core.validation import ValidationReport

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    # CSV round-trip turns integers into floats.
    pd.DataFrame({"chrom": [1.0], "pos": [100.0], "pip": [0.9]}).to_csv(
        tmp_path / "fine_mapping_pips.csv", index=False
    )
    cands = pd.DataFrame({"variant_uid": ["1:100", "1:100", "1:200"]})
    rep = ValidationReport()
    s = _load_pips(cands, rep)
    assert s is not None
    assert float(s["1:100"]) == pytest.approx(0.9)
    # Coverage counts unique variants, not (variant, gene) rows.
    assert any("1 candidate variants covered" in i.message for i in rep.issues)


def test_read_eqtl_pairs_empty_file_raises_clean_error(tmp_path):
    import pandas as pd

    from hidden_locus.core.features import read_eqtl_pairs

    empty = tmp_path / "empty.tsv.gz"
    empty.write_bytes(b"")
    try:
        raised = False
        read_eqtl_pairs(empty)
    except ValueError as exc:
        raised = True
        assert "empty or corrupt" in str(exc)
    assert raised

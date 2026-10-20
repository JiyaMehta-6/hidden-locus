"""Tests for settings persistence and coercion."""

import json

from hidden_locus import config


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    s = config.Settings(window_bp=250_000, seed=7, tissues=["Whole_Blood"])
    s.save()
    loaded = config.Settings.load()
    assert loaded.window_bp == 250_000
    assert loaded.seed == 7
    assert loaded.tissues == ["Whole_Blood"]


def test_settings_load_coerces_bad_types(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    bad = {
        "window_bp": "not-a-number",
        "locus_gap_bp": "500000",   # numeric string -> int
        "seed": 7.9,                # float -> int
        "tissues": "Whole_Blood",   # not a list -> default
        "gene_biotypes": None,      # not a list -> default
        "use_pu": 1,                # truthy -> bool
        "label_p_threshold": "bad",
        "unknown_field": "ignored",
    }
    (tmp_path / "settings.json").write_text(json.dumps(bad), encoding="utf-8")
    s = config.Settings.load()
    defaults = config.Settings()
    assert s.window_bp == defaults.window_bp
    assert s.locus_gap_bp == 500_000
    assert s.seed == 7
    assert s.tissues == list(defaults.tissues)
    assert s.gene_biotypes == list(defaults.gene_biotypes)
    assert s.use_pu is True
    assert s.label_p_threshold == defaults.label_p_threshold


def test_settings_load_non_dict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    (tmp_path / "settings.json").write_text("[1, 2, 3]", encoding="utf-8")
    s = config.Settings.load()
    assert s.window_bp == config.Settings().window_bp

"""Tests for report artifact writers."""

import json

import pandas as pd

from hidden_locus import config
from hidden_locus.core.features import finalize_features
from hidden_locus.core.pipeline import PipelineResult, export_report
from hidden_locus.core.ranking import train_candidate_model
from hidden_locus.core.reporting import write_row_contributions
from hidden_locus.core.validation import ValidationReport


def test_write_row_contributions_long_format(tmp_path, labelled_feature_table):
    df = finalize_features(labelled_feature_table)
    model = train_candidate_model(df, model_name="logreg", use_pu=False)
    ranked = df.copy().sort_values("label", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    path = write_row_contributions(model, ranked, tmp_path / "contrib.csv", top_n=10)
    out = pd.read_csv(path)

    n_features = len(model.features)
    assert len(out) == 10 * n_features
    assert set(out["rank"].unique()) == set(range(1, 11))
    assert out["feature"].nunique() == n_features
    # Row-major alignment: each rank block covers every feature exactly once.
    assert (out.groupby("rank")["feature"].nunique() == n_features).all()
    # Training-standardized contributions must not be all zero.
    assert (out["contribution"].abs() > 0).any()


def test_export_report_provenance_records_settings_hash(tmp_path):
    settings = config.Settings()
    result = PipelineResult(report=ValidationReport(), settings=settings)

    written = export_report(result, out_dir=tmp_path)

    bundle = json.loads(written["provenance"].read_text(encoding="utf-8"))
    assert bundle["extra"]["settings_hash"] == settings.cache_key
    assert bundle["extra"]["settings_hash"]

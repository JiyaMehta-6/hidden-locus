"""Tests for grouped evaluation, metrics and explainability."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.evaluation import (
    compare_baselines,
    feature_group_ablations,
    grouped_cross_validate,
)
from hidden_locus.core.explain import (
    global_importance,
    permutation_importance,
    row_contributions,
    top_features_for_row,
)
from hidden_locus.core.features import finalize_features
from hidden_locus.core.ranking import train_candidate_model


@pytest.fixture
def ready_table(labelled_feature_table):
    return finalize_features(labelled_feature_table)


def test_grouped_cv_by_locus(ready_table):
    result = grouped_cross_validate(ready_table, model_name="logreg", use_pu=False,
                                    n_splits=4, seed=42)
    assert len(result.folds) == 4
    # every locus appears in exactly one test fold
    assert result.oof_prob.notna().all()
    assert 0.0 <= result.mean_roc_auc <= 1.0
    assert np.isfinite(result.mean_average_precision)
    summary = result.summary()
    assert summary["model"] == "logreg"
    assert summary["n_folds"] == 4


def test_grouped_cv_needs_two_loci():
    df = pd.DataFrame(
        {
            "locus_id": ["chr1_1"] * 10,
            "label": [0, 1] * 5,
            "abs_log10_dist_tss": np.arange(10, dtype=float),
        }
    )
    with pytest.raises(ValueError, match="at least 2 loci"):
        grouped_cross_validate(df, model_name="logreg", use_pu=False)


def test_compare_baselines_table(ready_table):
    cv = grouped_cross_validate(ready_table, model_name="logreg", use_pu=False, n_splits=4)
    table = compare_baselines(ready_table, cv, l2g_scores=None)
    models = set(table["model"])
    assert "logreg" in models
    assert "nearest_gene" in models
    assert {"mean_roc_auc", "mean_average_precision"} <= set(table.columns)


@pytest.mark.slow
def test_feature_group_ablations(ready_table):
    table = feature_group_ablations(
        ready_table, model_name="logreg", n_splits=4,
        feature_groups={"structural": ["abs_log10_dist_tss", "gwas_log10_p"]},
    )
    ablations = set(table["ablation"])
    assert "full" in ablations
    assert "minus_structural" in ablations


@pytest.mark.slow
def test_feature_group_ablations_reuses_full_result(ready_table):
    cv = grouped_cross_validate(ready_table, model_name="logreg", use_pu=False, n_splits=4)
    table = feature_group_ablations(
        ready_table, model_name="logreg", n_splits=4,
        feature_groups={"structural": ["abs_log10_dist_tss"]},
        full_result=cv,
    )
    full_row = table[table["ablation"] == "full"].iloc[0]
    assert full_row["mean_roc_auc"] == pytest.approx(cv.mean_roc_auc)
    assert full_row["n_folds"] == len(cv.folds)


def test_compare_baselines_reports_n_scored(ready_table):
    cv = grouped_cross_validate(ready_table, model_name="logreg", use_pu=False, n_splits=4)
    table = compare_baselines(ready_table, cv, use_pu=False)
    assert "n_scored" in table.columns
    assert (table["n_scored"] == len(ready_table)).all()
    # Plain-logreg candidate + skipped lr baseline -> exactly one logreg row.
    assert list(table["model"]).count("logreg") == 1


@pytest.mark.slow
def test_compare_baselines_reports_pu_ablation_row(ready_table):
    cv = grouped_cross_validate(ready_table, model_name="logreg", use_pu=True, n_splits=4)
    table = compare_baselines(ready_table, cv, use_pu=True)
    models = set(table["model"])
    assert "logreg" in models
    assert "logreg_non_pu" in models
    # Plain logreg is computed exactly once (the non_pu row, no lr duplicate).
    assert list(table["model"]).count("logreg_non_pu") == 1
    assert table["model"].is_unique


def test_global_importance_linear(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    imp = global_importance(model, ready_table)
    frame = imp.to_frame()
    assert len(frame) == len(model.features)
    assert frame["feature"].is_unique
    assert imp.method in {"coefficient", "gain", "permutation"}


def test_global_importance_tree(ready_table):
    model = train_candidate_model(ready_table, model_name="xgboost", use_pu=False)
    imp = global_importance(model, ready_table)
    assert imp.method == "gain"
    assert (imp.importances >= 0).all()


def test_permutation_importance(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    imp = permutation_importance(model, ready_table, n_repeats=2, random_state=0)
    assert imp.method == "permutation"
    assert len(imp.importances) == len(model.features)
    assert np.isfinite(imp.importances).all()


def test_row_contributions_shape(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    contrib = row_contributions(model, ready_table.head(5))
    assert len(contrib) == 5
    assert "score_sum" in contrib.columns
    assert np.isfinite(contrib["score_sum"]).all()


def test_top_features_for_row(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    top = top_features_for_row(model, ready_table, row_index=0, k=5)
    assert len(top) <= 5
    assert (top["contribution"].abs().diff().fillna(0) <= 1e-12).all()
    # Single-row explanations must standardize against training statistics;
    # a frame-derived standardization would make every contribution zero.
    assert (top["contribution"].abs() > 0).any()


def test_row_contributions_single_row_nonzero(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    one = row_contributions(model, ready_table.iloc[[0]])
    values = one.drop(columns=["score_sum", "method"]).to_numpy()
    assert np.isfinite(values).all()
    assert (np.abs(values) > 0).any()

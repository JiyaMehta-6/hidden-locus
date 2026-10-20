"""Tests for model training, PU bagging and baselines."""

import numpy as np
import pandas as pd
import pytest

from hidden_locus.core.features import assert_no_leakage, feature_columns, finalize_features
from hidden_locus.core.ranking import (
    PUBaggedClassifier,
    TrainedModel,
    build_feature_matrix,
    logistic_regression_baseline,
    nearest_gene_baseline,
    opentargets_baseline,
    train_candidate_model,
)


@pytest.fixture
def ready_table(labelled_feature_table):
    return finalize_features(labelled_feature_table)


def test_build_feature_matrix_no_leakage(ready_table):
    x, cols = build_feature_matrix(ready_table)
    assert x.shape == (len(ready_table), len(cols))
    assert np.isfinite(x).all()
    assert_no_leakage(ready_table, cols)


def test_train_xgboost_pu(ready_table):
    model = train_candidate_model(ready_table, model_name="xgboost", use_pu=True, n_bags=5)
    assert model.is_pu
    prob = model.predict_proba(ready_table)
    assert prob.shape == (len(ready_table),)
    assert ((prob >= 0) & (prob <= 1)).all()
    # should separate labels better than chance on training data
    y = ready_table["label"].to_numpy()
    assert prob[y == 1].mean() > prob[y == 0].mean()


def test_train_logreg_non_pu(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=False)
    assert not model.is_pu
    pred = model.predict(ready_table, threshold=0.5)
    assert set(np.unique(pred)) <= {0, 1}


def test_train_logreg_pu(ready_table):
    model = train_candidate_model(ready_table, model_name="logreg", use_pu=True, n_bags=5)
    assert isinstance(model.estimator, PUBaggedClassifier)
    prob = model.predict_proba(ready_table)
    assert np.isfinite(prob).all()


def test_pu_bagging_needs_unlabelled_rows(ready_table):
    x, cols = build_feature_matrix(ready_table)
    y = np.ones(len(ready_table), dtype=int)  # all positive
    est = PUBaggedClassifier(lambda random_state: None, n_bags=2)
    with pytest.raises(ValueError, match="positive labels|unlabelled"):
        est.fit(x, y)


def test_nearest_gene_baseline_one_per_variant(ready_table):
    # build a small candidates frame with distances
    cands = pd.DataFrame(
        {
            "variant_uid": ["1:100", "1:100", "1:100", "1:200"],
            "gene_name": ["A", "B", "C", "A"],
            "dist_tss": [5000, -100, 90_000, 10],
        }
    )
    score = nearest_gene_baseline(cands)
    assert score.iloc[1] == 1.0  # |−100| smallest for variant 1:100
    assert score.iloc[0] == 0.0 and score.iloc[2] == 0.0
    assert score.iloc[3] == 1.0  # only gene for 1:200


def test_logistic_regression_baseline(ready_table):
    model = logistic_regression_baseline(ready_table)
    assert model.name == "logreg"
    assert not model.is_pu
    prob = model.predict_proba(ready_table)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_opentargets_baseline_alignment(ready_table):
    l2g = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [ready_table["pos"].iloc[0]],
            "gene_id": [ready_table["gene_id"].iloc[0]],
            "l2g_score": [0.87],
        }
    )
    scores = opentargets_baseline(ready_table, l2g)
    assert scores.iloc[0] == pytest.approx(0.87)
    assert np.isnan(scores.iloc[1:].to_numpy()).all()
    # missing table -> all NaN
    assert opentargets_baseline(ready_table, None).isna().all()


def test_unknown_model_rejected(ready_table):
    with pytest.raises(ValueError, match="Unknown model"):
        train_candidate_model(ready_table, model_name="deep_learning")

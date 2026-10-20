"""Model explainability: linear coefficients, tree importances, per-row attributions.

No external SHAP dependency is required.  For tree ensembles we expose the
built-in feature_importances_ plus a cheap permutation-style importance on a
held-out subset; for logistic regression we expose the fitted coefficients.
Per-row explanations use a simple contribution decomposition (feature value
times standardized coefficient / importance weight) that is deterministic and
documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ranking import TrainedModel


@dataclass
class GlobalImportance:
    """Per-feature importance aggregated over the training set."""

    model_name: str
    features: list[str]
    importances: np.ndarray
    method: str  # "gain" | "coefficient" | "permutation"

    def to_frame(self) -> pd.DataFrame:
        order = np.argsort(-self.importances)
        return pd.DataFrame(
            {
                "feature": [self.features[i] for i in order],
                "importance": self.importances[order],
                "method": self.method,
                "model": self.model_name,
            }
        )


def global_importance(model: TrainedModel, df: pd.DataFrame | None = None) -> GlobalImportance:
    """Extract a global feature-importance vector from a trained model."""
    est = model.estimator
    if hasattr(est, "estimators_") and hasattr(est.estimators_[0], "feature_importances_"):
        # PUBaggedClassifier: average gain importances across bags.
        imp = np.mean(
            [e.feature_importances_ for e in est.estimators_], axis=0
        )
        return GlobalImportance(model.name, model.features, imp, method="gain")
    if hasattr(est, "feature_importances_"):
        return GlobalImportance(
            model.name, model.features, est.feature_importances_, method="gain"
        )
    if hasattr(est, "coef_"):
        coef = np.atleast_1d(est.coef_).ravel()
        return GlobalImportance(model.name, model.features, coef, method="coefficient")
    if df is not None:
        return permutation_importance(model, df)
    raise ValueError(f"Cannot extract importance from {type(est).__name__}")


def permutation_importance(
    model: TrainedModel,
    df: pd.DataFrame,
    n_repeats: int = 5,
    random_state: int = 42,
) -> GlobalImportance:
    """Permutation importance on a labelled subset (drop in ROC-AUC)."""
    from sklearn.metrics import roc_auc_score

    if "label" not in df.columns:
        raise ValueError("permutation_importance needs a 'label' column")
    x = df[model.features].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present for permutation importance")
    baseline = roc_auc_score(y, model.estimator.predict_proba(x)[:, 1])
    rng = np.random.RandomState(random_state)
    importances = np.zeros(len(model.features), dtype=np.float64)
    for j in range(len(model.features)):
        drops = []
        for _ in range(n_repeats):
            perm = x.copy()
            perm[:, j] = rng.permutation(perm[:, j])
            score = roc_auc_score(y, model.estimator.predict_proba(perm)[:, 1])
            drops.append(baseline - score)
        importances[j] = float(np.mean(drops))
    return GlobalImportance(model.name, model.features, importances, method="permutation")


def row_contributions(model: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Per-row additive contributions toward the predicted score.

    For linear models contribution = standardized_value * coefficient.
    For tree models we fall back to standardized_value * global gain weight
    (sign taken from the value's deviation from the training mean).  This is
    an approximation, clearly labelled as such in the output.

    Standardization uses the training-set statistics stored on the model so
    single-row explanations stay meaningful; only when a model carries no
    statistics (e.g. loaded from an old run) are the given rows used.
    """
    est = model.estimator
    x = df[model.features].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if model.feature_mean_ is not None and model.feature_std_ is not None:
        mean = np.asarray(model.feature_mean_, dtype=np.float64)
        std = np.asarray(model.feature_std_, dtype=np.float64)
    else:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
    z = (x - mean) / std

    if hasattr(est, "estimators_"):
        weight = np.mean([e.feature_importances_ for e in est.estimators_], axis=0)
        signed = z * weight  # sign carried by z (above/below mean)
        method = "gain_x_z (approximate)"
    elif hasattr(est, "feature_importances_"):
        weight = est.feature_importances_
        signed = z * weight
        method = "gain_x_z (approximate)"
    elif hasattr(est, "coef_"):
        coef = np.atleast_1d(est.coef_).ravel()
        signed = z * coef
        method = "linear_contribution"
    else:
        raise ValueError(f"Cannot explain {type(est).__name__}")

    out = pd.DataFrame(signed, columns=[f"contrib_{c}" for c in model.features], index=df.index)
    out["score_sum"] = signed.sum(axis=1)
    out["method"] = method
    return out


def top_features_for_row(
    model: TrainedModel, df: pd.DataFrame, row_index: int, k: int = 10
) -> pd.DataFrame:
    """Return the top-k absolute contributions for a single candidate row."""
    contrib = row_contributions(model, df.loc[[row_index]])
    values = contrib.drop(columns=["score_sum", "method"]).iloc[0]
    order = values.abs().sort_values(ascending=False).head(k)
    return pd.DataFrame(
        {
            "feature": order.index,
            "contribution": values.loc[order.index].to_numpy(),
            "row_index": row_index,
            "method": contrib["method"].iloc[0],
        }
    )

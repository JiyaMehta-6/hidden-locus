"""Model training: baselines, PU bagging, XGBoost candidate model.

The candidate model is trained on positive-unlabelled data derived from an
independent eQTL study.  Nearest-gene, logistic regression and (optionally)
Open Targets L2G scores are exposed as explicit comparators.  All feature
transformations are fitted on training folds only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config
from .features import assert_no_leakage, feature_columns, gtex_feature_columns

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


@dataclass
class TrainedModel:
    """Wrapper around a fitted estimator plus its feature list."""

    name: str
    estimator: object
    features: list[str]
    is_pu: bool = False
    # Training-set feature statistics for explainability.  Without these,
    # single-row explanations standardize against the row itself and every
    # contribution becomes zero.
    feature_mean_: "np.ndarray | None" = None
    feature_std_: "np.ndarray | None" = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float64)
        return self.estimator.predict_proba(x)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)


@dataclass
class PUBaggedClassifier:
    """Simple bagged PU learner (positive-unlabelled).

    Repeatedly samples an unlabeled subset the size of the labelled-positive
    set, treats it as pseudo-negatives, trains a base classifier, and
    averages predicted probabilities.  Each bag is perfectly balanced
    (n_pos positives + n_pos sampled negatives), so class weighting on the
    base learner has no effect.  Hyperparameters are fixed for reproducibility.

    Supports weighted sampling of unlabelled instances via `unlabeled_weights`
    in `fit()` - higher weight = more likely to be sampled as pseudo-negative.
    This enables negative-confidence weighting: unlabelled instances with
    features suggesting they are true negatives (e.g., far from TSS, no eQTL
    evidence, no regulatory overlap) get higher sampling probability.
    """

    base_factory: object
    n_bags: int = config.DEFAULT_PU_N_BAGS
    random_state: int = config.DEFAULT_SEED
    estimators_: list = field(default_factory=list, repr=False)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unlabeled_weights: np.ndarray | None = None,
    ) -> "PUBaggedClassifier":
        rng = np.random.RandomState(self.random_state)
        pos_idx = np.flatnonzero(y == 1)
        unl_idx = np.flatnonzero(y == 0)
        n_pos = len(pos_idx)
        if n_pos == 0:
            raise ValueError("No positive labels available for PU training")
        if n_pos >= len(unl_idx):
            raise ValueError(
                f"PU bagging needs more unlabelled than labelled rows "
                f"(pos={n_pos}, unl={len(unl_idx)})"
            )

        # Normalize unlabelled weights to a probability distribution
        if unlabeled_weights is not None:
            w = unlabeled_weights[unl_idx]
            w = np.maximum(w, 0)  # clip negative weights to 0
            w_sum = w.sum()
            if w_sum > 0:
                probs = w / w_sum
            else:
                probs = None  # uniform
        else:
            probs = None

        self.estimators_.clear()
        for bag in range(self.n_bags):
            sample = rng.choice(unl_idx, size=n_pos, replace=False, p=probs)
            train_idx = np.concatenate([pos_idx, sample])
            y_bag = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
            est = self.base_factory(random_state=self.random_state + bag)
            est.fit(x[train_idx], y_bag)
            self.estimators_.append(est)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model not fitted")
        votes = np.column_stack([est.predict_proba(x)[:, 1] for est in self.estimators_])
        mean = votes.mean(axis=1)
        return np.column_stack([1 - mean, mean])

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= threshold).astype(int)


def _logreg_base(random_state: int):
    # Features are unscaled and on very different ranges (distances, -log10
    # p-values, binary flags), so lbfgs needs a generous iteration budget to
    # converge; 2000 was not enough and raised ConvergenceWarning.
    return LogisticRegression(
        max_iter=5000,
        C=1.0,
        random_state=random_state,
    )


def _xgboost_base(random_state: int):
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract the leakage-checked feature matrix from a feature table."""
    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, cols


def train_candidate_model(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    use_pu: bool = True,
    n_bags: int = config.DEFAULT_PU_N_BAGS,
    seed: int = config.DEFAULT_SEED,
) -> TrainedModel:
    """Train the candidate ranking model on a feature table with labels.

    ``df`` must contain a ``label`` column (1 = positive from the independent
    study, 0 = unlabelled).
    """
    x, cols = build_feature_matrix(df)
    y = df["label"].to_numpy(dtype=int)

    # Compute negative-confidence weights for PU bagging.
    # Higher weight = more confident this unlabelled instance is a true negative.
    unlabeled_weights = None
    if use_pu:
        # Base weight: 1.0 for all unlabelled
        weights = np.ones(len(df), dtype=np.float64)
        unl_idx = np.flatnonzero(y == 0)

        # Feature-based negative confidence (only for unlabelled rows)
        if "abs_log10_dist_tss" in df.columns:
            # Farther from TSS -> more likely negative
            dist_score = df["abs_log10_dist_tss"].fillna(0).to_numpy()
            weights[unl_idx] *= 1 + 0.5 * (dist_score[unl_idx] / (dist_score[unl_idx].max() + 1e-6))

        if "regulatory_missing" in df.columns:
            # No regulatory evidence -> more likely negative
            weights[unl_idx] *= 1 + 0.3 * df["regulatory_missing"].fillna(1).to_numpy()[unl_idx]
        elif "ccre_class" in df.columns:
            # No cCRE overlap
            no_ccre = (df["ccre_class"] == "none").astype(float)
            weights[unl_idx] *= 1 + 0.3 * no_ccre[unl_idx]

        if "gtex_missing" in df.columns:
            weights[unl_idx] *= 1 + 0.4 * df["gtex_missing"].fillna(1).to_numpy()[unl_idx]
        elif "gtex_n_tissues" in df.columns:
            # Fewer tissues with eQTL -> more likely negative
            gtex_n = df["gtex_n_tissues"].fillna(0).to_numpy()
            weights[unl_idx] *= 1 + 0.3 * (1 - gtex_n[unl_idx] / (gtex_n[unl_idx].max() + 1))

        if "abc_missing" in df.columns:
            weights[unl_idx] *= 1 + 0.3 * df["abc_missing"].fillna(1).to_numpy()[unl_idx]
        elif "abc_score" in df.columns:
            weights[unl_idx] *= 1 + 0.3 * (1 - df["abc_score"].fillna(0).to_numpy()[unl_idx])

        if "gut_missing" in df.columns:
            weights[unl_idx] *= 1 + 0.4 * df["gut_missing"].fillna(1).to_numpy()[unl_idx]
        elif "gut_n_celltypes" in df.columns:
            gut_n = df["gut_n_celltypes"].fillna(0).to_numpy()
            weights[unl_idx] *= 1 + 0.3 * (1 - gut_n[unl_idx] / (gut_n[unl_idx].max() + 1))

        # Normalize weights for unlabelled
        unlabeled_weights = weights
    else:
        unlabeled_weights = None

    if model_name == "logreg":
        if use_pu:
            est = PUBaggedClassifier(_logreg_base, n_bags=n_bags, random_state=seed)
            est.fit(x, y, unlabeled_weights=unlabeled_weights)
            is_pu = True
        else:
            est = _logreg_base(seed)
            est.fit(x, y)
            is_pu = False
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed; use model_name='logreg'")
        if use_pu:
            est = PUBaggedClassifier(_xgboost_base, n_bags=n_bags, random_state=seed)
            est.fit(x, y, unlabeled_weights=unlabeled_weights)
            is_pu = True
        else:
            est = _xgboost_base(seed)
            est.fit(x, y)
            is_pu = False
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return TrainedModel(
        name=model_name, estimator=est, features=cols, is_pu=is_pu,
        feature_mean_=x.mean(axis=0), feature_std_=std,
    )


def train_tissue_ablation(
    df: pd.DataFrame,
    tissues: list[str],
    model_name: str = "xgboost",
    seed: int = config.DEFAULT_SEED,
) -> TrainedModel:
    """Train a candidate model using only GTEx feature columns (tissue-only)."""
    cols = gtex_feature_columns(df, tissues)
    if not cols:
        raise ValueError("No GTEx feature columns found; run build_feature_table first")
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    if model_name == "xgboost" and HAS_XGBOOST:
        est = _xgboost_base(seed)
    else:
        est = _logreg_base(seed)
    est.fit(x, y)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return TrainedModel(
        name=f"{model_name}_tissue_only", estimator=est, features=cols, is_pu=False,
        feature_mean_=x.mean(axis=0), feature_std_=std,
    )


def nearest_gene_baseline(candidates: pd.DataFrame) -> pd.Series:
    """Return 1 for the nearest-TSS gene in each variant's candidate set.

    Ties are broken by smaller |dist_tss| then gene name; the score is a
    deterministic rank, not a probability.
    """
    df = candidates.copy()
    df["abs_dist"] = df["dist_tss"].abs()
    df = df.sort_values(["variant_uid", "abs_dist", "gene_name"], kind="mergesort")
    df["ng_rank"] = df.groupby("variant_uid").cumcount()
    score = pd.Series(0.0, index=df.index)
    score[df["ng_rank"] == 0] = 1.0
    out = pd.Series(np.nan, index=candidates.index, dtype=float)
    out.loc[df.index] = score
    return out.fillna(0.0)


def logistic_regression_baseline(
    df: pd.DataFrame,
    seed: int = config.DEFAULT_SEED,
) -> TrainedModel:
    """Standard logistic regression on the full feature set (not PU)."""
    return train_candidate_model(df, model_name="logreg", use_pu=False, seed=seed)


def opentargets_baseline(
    df: pd.DataFrame,
    l2g_scores: "pd.DataFrame | None",
) -> pd.Series:
    """Return Open Targets L2G scores aligned to *df* rows, or NaN.

    *l2g_scores* must have ``chrom``, ``pos``, ``gene_id``, ``l2g_score``.
    Rows with no L2G entry are marked missing (score NaN), never zero-filled.
    """
    if l2g_scores is None or l2g_scores.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)
    l2g = l2g_scores.copy()
    # Normalize like candidate uids ("1:123"), guarding against CSV round-trip
    # turning integers into floats ("1.0").
    chrom_s = l2g["chrom"].astype(str).str.replace(r"\.0$", "", regex=True)
    pos_s = pd.to_numeric(l2g["pos"], errors="coerce").astype("Int64").astype(str)
    l2g["variant_uid"] = chrom_s + ":" + pos_s
    l2g = l2g.drop_duplicates(subset=["variant_uid", "gene_id"], keep="first")
    # Align by index to preserve row order even if df has a non-RangeIndex.
    left = df[["variant_uid", "gene_id"]].copy()
    left.index = df.index
    merged = left.merge(
        l2g[["variant_uid", "gene_id", "l2g_score"]],
        on=["variant_uid", "gene_id"],
        how="left",
    )
    return pd.Series(merged["l2g_score"].to_numpy(dtype=float), index=df.index)

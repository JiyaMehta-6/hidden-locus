"""Leakage-aware model evaluation: grouped CV, ranking metrics, ablations.

Every split is grouped by ``locus_id`` so the same genomic region never
appears in both train and test.  Metrics follow a positive-unlabelled
protocol: labelled positives are the positive class and all other rows are
treated as negatives for scoring only (unlabelled rows are not confirmed
negatives; see :func:`grouped_cross_validate`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .. import config
from .ranking import TrainedModel, train_candidate_model


@dataclass
class FoldResult:
    """Metrics and predictions for one cross-validation fold."""

    fold: int
    n_train: int
    n_test: int
    n_test_labelled: int
    y_true: np.ndarray
    y_prob: np.ndarray
    roc_auc: float
    average_precision: float
    brier: float


@dataclass
class CVResult:
    """Aggregated cross-validation output for one model."""

    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    oof_prob: "pd.Series | None" = None  # out-of-fold probability per row
    oof_index: "pd.Index | None" = None

    @property
    def mean_roc_auc(self) -> float:
        vals = [f.roc_auc for f in self.folds if np.isfinite(f.roc_auc)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def std_roc_auc(self) -> float:
        vals = [f.roc_auc for f in self.folds if np.isfinite(f.roc_auc)]
        return float(np.std(vals)) if vals else float("nan")

    @property
    def mean_average_precision(self) -> float:
        vals = [f.average_precision for f in self.folds if np.isfinite(f.average_precision)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_brier(self) -> float:
        vals = [f.brier for f in self.folds if np.isfinite(f.brier)]
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "n_folds": len(self.folds),
            "mean_roc_auc": self.mean_roc_auc,
            "std_roc_auc": self.std_roc_auc,
            "mean_average_precision": self.mean_average_precision,
            "mean_brier": self.mean_brier,
        }


def _metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    """Return (roc_auc, average_precision, brier); NaN when undefined."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = float("nan")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = float("nan")
    return float(auc), float(ap), float(brier)


def grouped_cross_validate(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    use_pu: bool = True,
    n_splits: int = config.DEFAULT_N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    label_col: str = "label",
) -> CVResult:
    """Run GroupKFold by locus_id, retraining the candidate model per fold.

    Positive-unlabelled evaluation protocol: labelled positives (from the
    independent study) are the positive class; every other row in a test
    fold is treated as negative for metric purposes only.  Unlabelled rows
    are *not* confirmed negatives — this is the standard PU assumption and
    the resulting ROC-AUC/AP must be read with that caveat.  Predictions
    are generated for every row so full rankings can be built.

    Splits are grouped by ``locus_id`` so the same genomic region never
    appears in both train and test.  Note: a gene can appear in multiple
    loci (via different variants within ±1 Mb); GroupKFold does not prevent
    gene-level leakage through GTEx eQTL features.  This is a known limitation.
    """
    if "locus_id" not in df.columns:
        raise ValueError("Feature table is missing locus_id; cannot group by locus")
    groups = df["locus_id"].to_numpy()
    y_all = df[label_col].to_numpy(dtype=int)
    n = len(df)
    oof = np.full(n, np.nan, dtype=np.float64)

    unique_groups = np.unique(groups)
    n_splits_eff = min(n_splits, len(unique_groups))
    if n_splits_eff < 2:
        raise ValueError(
            f"Need at least 2 loci for grouped CV, found {len(unique_groups)}"
        )

    gkf = GroupKFold(n_splits=n_splits_eff)
    result = CVResult(model_name=model_name)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(n), groups=groups)):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        try:
            model = train_candidate_model(
                df_train, model_name=model_name, use_pu=use_pu, seed=seed
            )
        except (ValueError, RuntimeError) as exc:
            result.folds.append(
                FoldResult(
                    fold=fold, n_train=len(train_idx), n_test=len(test_idx),
                    n_test_labelled=0, y_true=np.array([]), y_prob=np.array([]),
                    roc_auc=float("nan"), average_precision=float("nan"),
                    brier=float("nan"),
                )
            )
            continue
        prob = model.predict_proba(df_test)
        oof[test_idx] = prob
        mask = df_test[label_col].to_numpy(dtype=int) == 1
        # Metric subset: labelled positives vs the rest of the fold's rows
        # (unlabelled treated as negatives for metric purposes only).
        y_true = df_test[label_col].to_numpy(dtype=int)
        y_prob = prob
        auc, ap, brier = _metric_block(y_true, y_prob)
        result.folds.append(
            FoldResult(
                fold=fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_test_labelled=int(mask.sum()),
                y_true=y_true,
                y_prob=y_prob,
                roc_auc=auc,
                average_precision=ap,
                brier=brier,
            )
        )
    result.oof_prob = pd.Series(oof, index=df.index)
    result.oof_index = df.index
    return result


def compare_baselines(
    df: pd.DataFrame,
    candidate_result: CVResult,
    l2g_scores: "pd.DataFrame | None" = None,
    model_name: "str | None" = None,
    use_pu: bool = True,
    seed: int = config.DEFAULT_SEED,
) -> pd.DataFrame:
    """Assemble a metric comparison table across candidate model + baselines.

    Returns a tidy DataFrame with one row per model: mean ROC-AUC, mean AP,
    mean Brier, and fold-level std where available.  When the candidate model
    uses PU bagging, a ``<model>_non_pu`` row is added so the PU contribution
    is measurable (the PU ablation).
    """
    from .ranking import nearest_gene_baseline, opentargets_baseline

    model_name = model_name or candidate_result.model_name
    rows = [candidate_result.summary()]

    # PU bagging ablation: same model trained without PU bagging.
    if use_pu and candidate_result.folds:
        try:
            non_pu = grouped_cross_validate(
                df, model_name=model_name, use_pu=False,
                n_splits=len(candidate_result.folds), seed=seed,
            )
            row = non_pu.summary()
            row["model"] = f"{model_name}_non_pu"
            rows.append(row)
        except (ValueError, RuntimeError):
            pass

    # PU ablation for logistic regression (when candidate is not logreg).
    if use_pu and model_name != "logreg":
        try:
            lr_pu = grouped_cross_validate(
                df, model_name="logreg", use_pu=True,
                n_splits=len(candidate_result.folds), seed=seed,
            )
            row = lr_pu.summary()
            row["model"] = "logreg_pu"
            rows.append(row)
        except (ValueError, RuntimeError):
            pass

    # Nearest-gene: deterministic score, evaluate against labels directly.
    ng = nearest_gene_baseline(df)
    y = df["label"].to_numpy(dtype=int)
    auc, ap, brier = _metric_block(y, ng.to_numpy(dtype=float))
    rows.append(
        {
            "model": "nearest_gene",
            "n_folds": 0,
            "mean_roc_auc": auc,
            "std_roc_auc": float("nan"),
            "mean_average_precision": ap,
            "mean_brier": brier,
        }
    )

    # Logistic regression (non-PU) as a second learned baseline.  Skipped
    # when the candidate is already plain logreg (duplicate) or when the
    # <model>_non_pu row above already computed plain logreg.
    if model_name != "logreg":
        try:
            lr_result = grouped_cross_validate(
                df, model_name="logreg", use_pu=False,
                n_splits=max(2, len(candidate_result.folds) or config.DEFAULT_N_SPLITS),
                seed=seed,
            )
            rows.append(lr_result.summary())
        except (ValueError, RuntimeError):
            pass

    # Open Targets L2G comparator (external scores, not trained here).
    ot = opentargets_baseline(df, l2g_scores)
    if ot.notna().any():
        mask = ot.notna().to_numpy()
        auc, ap, brier = _metric_block(y[mask], ot.to_numpy(dtype=float)[mask])
        rows.append(
            {
                "model": "opentargets_l2g",
                "n_folds": 0,
                "mean_roc_auc": auc,
                "std_roc_auc": float("nan"),
                "mean_average_precision": ap,
                "mean_brier": brier,
                "n_scored": int(mask.sum()),
            }
        )

    out = pd.DataFrame(rows)
    if "n_scored" not in out.columns:
        out["n_scored"] = len(df)
    else:
        out["n_scored"] = out["n_scored"].fillna(len(df)).astype(int)
    return out


def feature_group_ablations(
    df: pd.DataFrame,
    feature_groups: "dict[str, list[str]] | None" = None,
    model_name: str = "xgboost",
    n_splits: int = config.DEFAULT_N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    use_pu: bool = True,
    full_result: "CVResult | None" = None,
) -> pd.DataFrame:
    """Run grouped CV with subsets of feature groups removed.

    ``feature_groups`` maps a group label to the columns in that group; the
    default uses the built-in structural / consequence / cCRE / GTEx / PIP
    groupings from :mod:`hidden_locus.core.features`.  Pass ``full_result``
    to reuse an already-computed full-model CV instead of retraining it.
    """
    from . import features as F

    if feature_groups is None:
        feature_groups = {
            "structural": [
                c for c in F.STRUCTURAL_FEATURES
                if c in df.columns and c not in ("pip", "pip_missing", "regulatory_missing", "gtex_missing",
                                                  "gtex_n_tissues", "eqtl_direction_conflict")
            ],
            "consequence": [c for c in F.CONSEQUENCE_FEATURES if c in df.columns],
            "ccre": F.regulatory_feature_columns(df),
            "gtex": F.gtex_feature_columns(df, _gtex_tissues(df)),
            "finemapping": F.finemapping_feature_columns(df),
            "coloc": F.coloc_feature_columns(df),
        }
    all_features = F.feature_columns(df)
    rows = []
    if full_result is None:
        full_result = grouped_cross_validate(
            df, model_name=model_name, use_pu=use_pu, seed=seed, n_splits=n_splits
        )
    rows.append({"ablation": "full", **full_result.summary()})
    for group_name, cols in feature_groups.items():
        if not cols:
            continue
        reduced = [c for c in all_features if c not in set(cols)]
        if not reduced:
            continue
        df_sub = df.copy()
        # Drop the removed columns from the frame so build_feature_matrix
        # cannot see them.
        for col in cols:
            if col in df_sub.columns:
                df_sub = df_sub.drop(columns=[col])
        try:
            res = grouped_cross_validate(
                df_sub, model_name=model_name, use_pu=use_pu, seed=seed, n_splits=n_splits
            )
            rows.append({"ablation": f"minus_{group_name}", **res.summary()})
        except (ValueError, RuntimeError):
            rows.append(
                {
                    "ablation": f"minus_{group_name}",
                    "model": model_name,
                    "n_folds": 0,
                    "mean_roc_auc": float("nan"),
                    "std_roc_auc": float("nan"),
                    "mean_average_precision": float("nan"),
                    "mean_brier": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _gtex_tissues(df: pd.DataFrame) -> list[str]:
    """Recover the tissue list from GTEx feature column names in *df*."""
    suffixes = set()
    for col in df.columns:
        if col.startswith("gtex_log10p_"):
            suffixes.add(col[len("gtex_log10p_"):])
    return sorted(suffixes)

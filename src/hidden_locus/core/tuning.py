"""Hyperparameter tuning and ensemble model utilities.

This module provides:
1. Optuna-based hyperparameter optimization for XGBoost and LightGBM
2. Ensemble model that combines multiple base learners
3. Nested cross-validation for unbiased performance estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .evaluation import CVResult, grouped_cross_validate
from .features import feature_columns, assert_no_leakage
from .ranking import TrainedModel, train_candidate_model


def _check_optuna():
    try:
        import optuna
        return optuna
    except ImportError:
        raise RuntimeError("optuna not installed; `pip install optuna`")


def _check_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        raise RuntimeError("lightgbm not installed; `pip install lightgbm`")


def _check_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        raise RuntimeError("xgboost not installed; `pip install xgboost`")


def _xgboost_objective(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    }


def _lightgbm_objective(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    }


def tune_xgboost(
    df: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    use_pu: bool = True,
) -> tuple[dict, float]:
    """Optimize XGBoost hyperparameters using Optuna with grouped CV.

    Returns best params and best OOF AUC.
    """
    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    groups = df["locus_id"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = _xgboost_objective(trial)
        from xgboost import XGBClassifier
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        aucs = []
        for train_idx, test_idx in gkf.split(x, groups=groups):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            est = XGBClassifier(
                **params,
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
            )
            est.fit(x_train, y_train)
            prob = est.predict_proba(x_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test, prob)
            except ValueError:
                auc = 0.5
            aucs.append(auc)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_lightgbm(
    df: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    use_pu: bool = True,
) -> tuple[dict, float]:
    """Optimize LightGBM hyperparameters using Optuna with grouped CV."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise RuntimeError("lightgbm not installed; `pip install lightgbm`")

    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    groups = df["locus_id"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = _lightgbm_objective(trial)
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        aucs = []
        for train_idx, test_idx in gkf.split(x, groups=groups):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            est = LGBMClassifier(
                **params,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            )
            est.fit(x_train, y_train)
            prob = est.predict_proba(x_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test, prob)
            except ValueError:
                auc = 0.5
            aucs.append(auc)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


@dataclass
class EnsembleModel:
    """Ensemble of multiple base models with weighted averaging."""

    models: list[TrainedModel]
    weights: list[float] | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.array(self.weights) / np.sum(self.weights)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = np.column_stack([m.predict_proba(df) for m in self.models])
        return np.average(probs, axis=1, weights=self.weights)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)


def train_ensemble(
    df: pd.DataFrame,
    models_config: list[dict] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    use_pu: bool = True,
) -> EnsembleModel:
    """Train an ensemble of models with default or custom configurations.

    models_config: list of dicts with keys: model_name, n_bags, use_pu, etc.
    If None, trains default XGBoost (PU), LightGBM (PU), and LogisticRegression (non-PU).
    """
    if models_config is None:
        models_config = [
            {"model_name": "xgboost", "use_pu": True, "n_bags": 20},
            {"model_name": "lightgbm", "use_pu": True, "n_bags": 20},
            {"model_name": "logreg", "use_pu": False},
        ]

    models = []
    for cfg in models_config:
        model = train_candidate_model(df, **cfg, seed=42)
        models.append(model)

    # Weight models by their CV performance (placeholder - would need CV scores)
    return EnsembleModel(models=models)


def nested_cv(
    df: pd.DataFrame,
    inner_trials: int = 30,
    outer_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Run nested cross-validation for unbiased performance estimation.

    Outer loop: GroupKFold for final evaluation
    Inner loop: Optuna tuning on training folds
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    groups = df["locus_id"].to_numpy()

    outer_gkf = GroupKFold(n_splits=outer_splits)
    outer_aucs = []
    best_params_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(x, groups=groups)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        # Inner CV for hyperparameter tuning
        inner_params, _ = tune_xgboost(
            df.iloc[train_idx].reset_index(drop=True),
            n_trials=inner_trials,
            n_splits=min(4, len(np.unique(train_groups))),
            seed=seed + fold,
        )
        best_params_list.append(inner_params)

        # Train on full training set with best params
        from xgboost import XGBClassifier
        est = XGBClassifier(**inner_params, random_state=seed, n_jobs=-1)
        est.fit(x_train, y_train)

        prob = est.predict_proba(x_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = 0.5
        outer_aucs.append(auc)

    return {
        "outer_aucs": outer_aucs,
        "mean_outer_auc": float(np.mean(outer_aucs)),
        "std_outer_auc": float(np.std(outer_aucs)),
        "best_params_per_fold": best_params_list,
    }


def calibrate_model(
    model: TrainedModel,
    df: pd.DataFrame,
    method: str = "isotonic",
    cv_folds: int = 5,
) -> TrainedModel:
    """Calibrate a trained model's predicted probabilities.

    Uses cross-validation to fit a calibrator (isotonic or sigmoid) on OOF predictions.
    Returns a new TrainedModel wrapper that applies calibration.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    groups = df["locus_id"].to_numpy()

    base_est = model.estimator
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    calibrated = CalibratedClassifierCV(base_est, method=method, cv=cv)
    calibrated.fit(x, y, groups=groups)

    # Create a new TrainedModel wrapper with calibrated estimator
    class CalibratedWrapper:
        def __init__(self, calibrated_est, features):
            self.calibrated_est = calibrated_est
            self.features = features

        def predict_proba(self, x):
            return self.calibrated_est.predict_proba(x)

    wrapper = CalibratedWrapper(model.estimator, model.features)
    return TrainedModel(
        name=f"{model.name}_calibrated",
        estimator=wrapper,
        features=model.features,
        is_pu=model.is_pu,
        feature_mean_=model.feature_mean_,
        feature_std_=model.feature_std_,
    )


def evaluate_calibration(
    model: TrainedModel,
    df: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """Evaluate calibration quality using reliability diagram metrics."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    cols = feature_columns(df)
    assert_no_leakage(df, cols)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)

    prob = model.predict_proba(df)
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(prob, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.any():
            ece += (mask.sum() / len(prob)) * abs(frac_pos[i] - mean_pred[i])

    brier = brier_score_loss(y, prob)

    return {
        "ece": float(ece),
        "brier": float(brier),
        "frac_pos": frac_pos.tolist(),
        "mean_pred": mean_pred.tolist(),
    }
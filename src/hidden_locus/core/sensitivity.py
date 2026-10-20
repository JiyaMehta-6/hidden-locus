"""Sensitivity analysis utilities for testing robustness to parameter choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .pipeline import run_full_pipeline, PipelineResult
from .validation import ValidationReport


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis sweep."""

    parameter: str
    values: list[Any]
    metrics: dict[str, list[float]]  # metric_name -> list of values per param value
    best_value: Any
    best_metric: float


def sensitivity_sweep(
    base_settings: Settings,
    param_name: str,
    param_values: list[Any],
    metric: str = "mean_roc_auc",
    n_splits: int = 3,
    seed: int = 42,
    report: ValidationReport | None = None,
) -> SensitivityResult:
    """Run sensitivity analysis by sweeping a single parameter.

    Parameters
    ----------
    base_settings : Settings
        Base settings to modify.
    param_name : str
        Name of the parameter to sweep (e.g., "window_bp", "label_p_threshold", "n_splits").
    param_values : list
        Values to test for the parameter.
    metric : str
        Metric to track (e.g., "mean_roc_auc", "mean_average_precision", "mean_brier").
    n_splits : int
        Number of CV splits for each run (reduced for speed).
    seed : int
        Random seed.
    report : ValidationReport, optional
        For logging.

    Returns
    -------
    SensitivityResult with metrics for each parameter value.
    """
    report = report if report is not None else ValidationReport()
    metrics_dict: dict[str, list[float]] = {metric: []}

    for val in param_values:
        # Create modified settings
        settings = Settings(
            disease_query=base_settings.disease_query,
            trait_pattern=base_settings.trait_pattern,
            assembly=base_settings.assembly,
            tissues=list(base_settings.tissues),
            window_bp=base_settings.window_bp,
            locus_gap_bp=base_settings.locus_gap_bp,
            label_p_threshold=base_settings.label_p_threshold,
            gene_biotypes=list(base_settings.gene_biotypes),
            seed=seed,
            n_splits=n_splits,
            model_name=base_settings.model_name,
            use_pu=base_settings.use_pu,
            candidate_variant_p=base_settings.candidate_variant_p,
        )
        # Override the parameter being swept
        setattr(settings, param_name, val)

        rep = ValidationReport()
        result = run_full_pipeline(settings, report=rep)

        if result.cv is not None:
            cv_summary = result.cv.summary()
            metric_val = cv_summary.get(metric, float("nan"))
        else:
            metric_val = float("nan")

        metrics_dict[metric].append(metric_val)
        report.add("info", "sensitivity", f"{param_name}={val}: {metric}={metric_val:.4f}")

    # Find best value
    best_idx = int(np.nanargmax(metrics_dict[metric])) if any(not np.isnan(v) for v in metrics_dict[metric]) else 0
    best_value = param_values[best_idx]
    best_metric = metrics_dict[metric][best_idx]

    return SensitivityResult(
        parameter=param_name,
        values=param_values,
        metrics=metrics_dict,
        best_value=best_value,
        best_metric=best_metric,
    )


def multi_param_sensitivity(
    base_settings: Settings,
    param_grid: dict[str, list[Any]],
    metric: str = "mean_roc_auc",
    n_splits: int = 3,
    seed: int = 42,
    max_runs: int | None = None,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Run sensitivity analysis over a grid of parameters.

    Returns a DataFrame with one row per parameter combination.
    """
    import itertools

    report = report if report is not None else ValidationReport()
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    if max_runs is not None and len(combinations) > max_runs:
        combinations = combinations[:max_runs]

    rows = []
    for combo in combinations:
        settings = Settings(
            disease_query=base_settings.disease_query,
            trait_pattern=base_settings.trait_pattern,
            assembly=base_settings.assembly,
            tissues=list(base_settings.tissues),
            window_bp=base_settings.window_bp,
            locus_gap_bp=base_settings.locus_gap_bp,
            label_p_threshold=base_settings.label_p_threshold,
            gene_biotypes=list(base_settings.gene_biotypes),
            seed=42,
            n_splits=3,
            model_name=base_settings.model_name,
            use_pu=base_settings.use_pu,
            candidate_variant_p=base_settings.candidate_variant_p,
        )

        for name, val in zip(param_grid.keys(), combo):
            setattr(settings, name, val)

        rep = ValidationReport()
        result = run_full_pipeline(settings, report=rep)

        row = {name: val for name, val in zip(param_grid.keys(), combo)}
        if result.cv is not None:
            cv_summary = result.cv.summary()
            row["metric_value"] = cv_summary.get("mean_roc_auc", float("nan"))
            row["n_positive"] = result.n_positives
            row["n_candidates"] = len(result.candidates)
        else:
            row["metric_value"] = float("nan")

        rows.append(row)
        report.add("info", "sensitivity", f"{combo}: AUC={row.get('metric_value', float('nan')):.4f}")

    return pd.DataFrame(rows)


def plot_sensitivity(
    result: SensitivityResult,
    output_path: str | None = None,
) -> Any:
    """Plot sensitivity analysis results."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_name = list(result.metrics.keys())[0]
    values = result.values
    metrics = result.metrics[metric_name]

    ax.plot(values, metrics, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(result.parameter)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Sensitivity Analysis: {result.parameter} vs {metric_name}")
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = values.index(result.best_value)
    ax.plot(values[best_idx], result.best_metric, 'r*', markersize=15,
            label=f'Best: {result.best_value} (AUC={result.best_metric:.4f})')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
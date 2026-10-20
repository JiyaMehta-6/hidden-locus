"""Report and artifact generation: tables, figures, provenance bundles.

Everything written here is reproducible from the on-disk datasets and the
recorded settings; nothing is fabricated.  Figures use a non-interactive
matplotlib backend so they can be generated headlessly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from .datasets import load_manifest  # noqa: E402
from .evaluation import CVResult  # noqa: E402
from .explain import GlobalImportance  # noqa: E402


def write_ranked_table(df: pd.DataFrame, path: Path) -> Path:
    """Write the ranked candidate table (all provenance columns kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_metrics_table(cv_results: "list[CVResult] | pd.DataFrame", path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cv_results, pd.DataFrame):
        cv_results.to_csv(path, index=False)
    else:
        pd.DataFrame([r.summary() for r in cv_results]).to_csv(path, index=False)
    return path


def write_importance_table(imp: GlobalImportance, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_frame().to_csv(path, index=False)
    return path


def write_row_contributions(
    model, ranked: pd.DataFrame, path: Path, top_n: int = 100
) -> Path:
    """Write per-row feature contributions for the top-ranked candidates.

    Long format: one row per (candidate, feature), sorted by rank.  Uses the
    model's training-set standardization so single rows stay meaningful.
    """
    import numpy as np

    from .explain import row_contributions

    path.parent.mkdir(parents=True, exist_ok=True)
    head = ranked.head(top_n)
    contrib = row_contributions(model, head)
    value_cols = [c for c in contrib.columns if c.startswith("contrib_")]
    id_cols = [c for c in ("rank", "locus_id", "variant_uid", "gene_name")
               if c in head.columns]
    values = contrib[value_cols].to_numpy()
    n, f = values.shape
    out = pd.DataFrame(
        {
            **{
                col: np.repeat(head[col].to_numpy(), f)
                for col in id_cols
            },
            "feature": np.tile(
                [c.removeprefix("contrib_") for c in value_cols], n
            ),
            "contribution": values.ravel(),
            "method": contrib["method"].iloc[0],
        }
    )
    out.to_csv(path, index=False)
    return path


def plot_roc_curves(cv_results: "list[CVResult]", path: Path) -> Path:
    """Plot ROC curves per fold for each cross-validated model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for result in cv_results:
        for fold in result.folds:
            if len(fold.y_true) == 0 or len(np.unique(fold.y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(fold.y_true, fold.y_prob)
            ax.plot(fpr, tpr, alpha=0.5, lw=1,
                    label=f"{result.model_name} fold{fold.fold} (AUC={fold.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC by locus-grouped fold")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(imp: GlobalImportance, path: Path, top_n: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = imp.to_frame().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(frame["feature"][::-1], frame["importance"][::-1])
    ax.set_xlabel(f"importance ({imp.method})")
    ax.set_title(f"Top {min(top_n, len(frame))} features ({imp.model_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_provenance(
    settings,
    validation_summary: str,
    extra: "dict | None",
    path: Path,
) -> Path:
    """Write a JSON provenance bundle describing how the run was produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app": {"name": config.APP_NAME, "version": config.APP_VERSION},
        "settings": asdict(settings) if hasattr(settings, "__dataclass_fields__") else dict(settings),
        "datasets": load_manifest(),
        "validation_summary": validation_summary,
        "extra": extra or {},
    }
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def write_validation_report(report, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(issue) for issue in report.issues]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path

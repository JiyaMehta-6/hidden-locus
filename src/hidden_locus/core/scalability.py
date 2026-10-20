"""Scalability benchmark: runtime vs number of loci.

Generates timing data by running pipeline stages on subsets of loci
(10%, 25%, 50%, 75%, 100%) and measuring wall-clock time.  Produces a
figure suitable for conference presentation.

All data is local — no downloads, no logins.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from .validation import ValidationReport  # noqa: E402


def scalability_benchmark(
    settings: config.Settings,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Run pipeline stages on loci subsets and measure runtime.

    Returns DataFrame: n_loci, n_variants, n_candidates, n_features,
                       time_variants, time_candidates, time_features,
                       time_labels, time_total.
    """
    report = report if report is not None else ValidationReport()
    from .pipeline import (
        prepare_variants,
        prepare_annotations,
        build_candidates_stage,
        build_features_stage,
        attach_labels_stage,
    )

    rows = []
    for frac in fractions:
        t0 = time.time()
        sub_settings = config.Settings(
            disease_query=settings.disease_query,
            trait_pattern=settings.trait_pattern,
            tissues=list(settings.tissues),
            window_bp=settings.window_bp,
            locus_gap_bp=settings.locus_gap_bp,
            label_p_threshold=settings.label_p_threshold,
            gene_biotypes=list(settings.gene_biotypes),
            seed=settings.seed,
            n_splits=settings.n_splits,
            model_name=settings.model_name,
            use_pu=settings.use_pu,
            candidate_variant_p=settings.candidate_variant_p,
            use_coloc_features=settings.use_coloc_features,
            coloc_pp4_threshold=settings.coloc_pp4_threshold,
        )
        row = {"fraction": frac}
        try:
            # Stage 1: variants
            t1 = time.time()
            variants = prepare_variants(sub_settings, report)
            row["time_variants"] = time.time() - t1

            # Subsample loci.
            unique_loci = variants["locus_id"].unique()
            n_keep = max(1, int(len(unique_loci) * frac))
            keep_loci = set(unique_loci[:n_keep])
            variants = variants[variants["locus_id"].isin(keep_loci)].copy()
            row["n_loci"] = len(keep_loci)
            row["n_variants"] = len(variants)

            # Stage 2: annotations (cached, so fast)
            t1 = time.time()
            genes, ccre, abc, gut_atlas = prepare_annotations(sub_settings, report)
            row["time_annotations"] = time.time() - t1

            # Stage 3: candidates
            t1 = time.time()
            candidates = build_candidates_stage(variants, genes, sub_settings, report)
            row["time_candidates"] = time.time() - t1
            row["n_candidates"] = len(candidates)

            # Stage 4: features
            t1 = time.time()
            features, _ctx = build_features_stage(
                candidates, genes, ccre, abc, gut_atlas, sub_settings, report
            )
            row["time_features"] = time.time() - t1
            row["n_features"] = len(features)

            # Stage 5: labels
            t1 = time.time()
            features = attach_labels_stage(features, sub_settings, report)
            row["time_labels"] = time.time() - t1
            row["n_positives"] = int(features["label"].sum()) if "label" in features.columns else 0

            row["time_total"] = time.time() - t0
            row["success"] = True
        except Exception as exc:
            report.add("warning", f"scalability_frac_{frac}", f"Failed at {frac:.0%}: {exc}")
            row["time_total"] = time.time() - t0
            row["success"] = False
        rows.append(row)
        report.add(
            "info", f"scalability_{frac:.0%}",
            f"Scalability {frac:.0%}: {row.get('n_loci', 0)} loci, "
            f"{row['time_total']:.1f}s total.",
        )
    return pd.DataFrame(rows)


def plot_scalability(
    bench: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot runtime vs number of loci (log-log scale)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = bench[bench.get("success", True) == True] if "success" in bench.columns else bench  # noqa: E712
    if ok.empty or "n_loci" not in ok.columns:
        ok = bench
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total runtime vs n_loci.
    ax = axes[0]
    x = ok["n_loci"].to_numpy()
    y = ok["time_total"].to_numpy()
    ax.plot(x, y, "o-", lw=2, ms=8, label="Total pipeline")
    if "time_features" in ok.columns:
        ax.plot(x, ok["time_features"].to_numpy(), "s--", lw=1.5, ms=6, label="Feature build")
    if "time_candidates" in ok.columns:
        ax.plot(x, ok["time_candidates"].to_numpy(), "^--", lw=1.5, ms=6, label="Candidate gen")
    ax.set_xlabel("Number of loci")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Scalability: Runtime vs Number of Loci")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: candidates vs runtime (throughput).
    ax = axes[1]
    if "n_candidates" in ok.columns:
        ax.scatter(ok["n_candidates"], ok["time_total"], c="steelblue", s=80, zorder=5)
        for _, r in ok.iterrows():
            ax.annotate(
                f"{r['n_loci']} loci",
                (r["n_candidates"], r["time_total"]),
                fontsize=8, xytext=(5, 5), textcoords="offset points",
            )
        ax.set_xlabel("Number of candidate pairs")
        ax.set_ylabel("Total runtime (seconds)")
        ax.set_title("Throughput: Candidates vs Runtime")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

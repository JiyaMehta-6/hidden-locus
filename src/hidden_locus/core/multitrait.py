"""Multi-trait benchmark: test model generalizability across diseases.

Runs the pipeline on multiple traits from GWAS Catalog and compares
performance metrics.  Labels are trait-agnostic molecular (BLUEPRINT eQTL),
so the model should generalize across traits.

Usage::

    .venv/Scripts/python.exe -m hidden_locus.core.multitrait
    .venv/Scripts/python.exe -m hidden_locus.core.multitrait --traits ibd t2d ra

Everything uses local GWAS Catalog downloads — no logins, no payments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .. import config
from .pipeline import (
    build_candidates_stage,
    build_features_stage,
    prepare_annotations,
    prepare_variants,
    attach_labels_stage,
    train_and_evaluate_stage,
)
from .validation import ValidationReport


# Trait patterns matched against GWAS Catalog MAPPED_TRAIT / DISEASE/TRAIT.
TRAITS: dict[str, str] = {
    "ibd": r"(?i)crohn|inflammatory bowel|ulcerative colitis",
    "t2d": r"(?i)type 2 diabetes|non-insulin-dependent",
    "ra": r"(?i)rheumatoid arthritis",
    "asthma": r"(?i)asthma",
    "cad": r"(?i)coronary artery|myocardial infarction",
    "celiac": r"(?i)celiac",
    "psoriasis": r"(?i)psoriasis",
    "as": r"(?i)ankylosing spondylitis",
}


def run_trait_benchmark(
    trait_name: str,
    trait_pattern: str,
    base_settings: config.Settings,
    report: ValidationReport | None = None,
) -> dict:
    """Run the core pipeline stages for one trait and return CV metrics.

    Skips export/report writing — just computes features, labels, CV.
    """
    report = report if report is not None else ValidationReport()
    settings = config.Settings(
        disease_query=f"Multi-trait: {trait_name}",
        trait_pattern=trait_pattern,
        tissues=list(base_settings.tissues),
        window_bp=base_settings.window_bp,
        locus_gap_bp=base_settings.locus_gap_bp,
        label_p_threshold=base_settings.label_p_threshold,
        gene_biotypes=list(base_settings.gene_biotypes),
        seed=base_settings.seed,
        n_splits=base_settings.n_splits,
        model_name=base_settings.model_name,
        use_pu=base_settings.use_pu,
        candidate_variant_p=base_settings.candidate_variant_p,
        use_coloc_features=base_settings.use_coloc_features,
        coloc_pp4_threshold=base_settings.coloc_pp4_threshold,
    )
    t0 = time.time()
    row = {"trait": trait_name, "trait_pattern": trait_pattern}
    try:
        variants = prepare_variants(settings, report)
        genes, ccre, abc, gut_atlas = prepare_annotations(settings, report)
        candidates = build_candidates_stage(variants, genes, settings, report)
        features, _ctx = build_features_stage(
            candidates, genes, ccre, abc, gut_atlas, settings, report
        )
        features = attach_labels_stage(features, settings, report)
        if features.empty or "label" not in features.columns or int(features["label"].sum()) == 0:
            row.update(
                {
                    "n_variants": int(len(variants)),
                    "n_candidates": int(len(candidates)),
                    "n_positives": 0,
                    "roc_auc": float("nan"),
                    "average_precision": float("nan"),
                    "brier": float("nan"),
                    "nearest_gene_auc": float("nan"),
                    "runtime_s": time.time() - t0,
                }
            )
            return row
        cv, baselines, _ablations, _model, _imp = train_and_evaluate_stage(
            features, settings, report
        )
        # Extract nearest-gene AUC from baselines if present.
        nearest_gene_auc = float("nan")
        if not baselines.empty and "model" in baselines.columns:
            ng = baselines[baselines["model"] == "nearest_gene"]
            if not ng.empty:
                nearest_gene_auc = float(ng["mean_roc_auc"].iloc[0])
        row.update(
            {
                "n_variants": int(len(variants)),
                "n_candidates": int(len(candidates)),
                "n_positives": int(features["label"].sum()),
                "roc_auc": cv.mean_roc_auc,
                "std_roc_auc": cv.std_roc_auc,
                "average_precision": cv.mean_average_precision,
                "brier": cv.mean_brier,
                "nearest_gene_auc": nearest_gene_auc,
                "runtime_s": time.time() - t0,
            }
        )
    except Exception as exc:
        report.add("warning", f"trait_{trait_name}_failed", f"Trait {trait_name} failed: {exc}")
        row.update(
            {
                "n_variants": 0,
                "n_candidates": 0,
                "n_positives": 0,
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "brier": float("nan"),
                "nearest_gene_auc": float("nan"),
                "runtime_s": time.time() - t0,
            }
        )
    return row


def run_multitrait_benchmark(
    traits: "dict[str, str] | None" = None,
    base_settings: "config.Settings | None" = None,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Run benchmark across multiple traits and return a tidy DataFrame."""
    traits = traits if traits is not None else TRAITS
    base_settings = base_settings or config.Settings()
    report = report if report is not None else ValidationReport()
    rows = []
    for name, pattern in traits.items():
        report.add("info", f"trait_{name}_start", f"Running multi-trait benchmark for {name}...")
        row = run_trait_benchmark(name, pattern, base_settings, report)
        rows.append(row)
        report.add(
            "info", f"trait_{name}_done",
            f"{name}: AUC={row.get('roc_auc', float('nan')):.4f} "
            f"(n_pos={row.get('n_positives', 0)})",
        )
    df = pd.DataFrame(rows)
    # Compute mean and std across successful traits.
    successful = df[df["roc_auc"].notna()]
    if not successful.empty:
        summary = {
            "trait": "__mean__",
            "roc_auc": successful["roc_auc"].mean(),
            "std_roc_auc": successful["roc_auc"].std(),
            "average_precision": successful["average_precision"].mean(),
            "nearest_gene_auc": successful["nearest_gene_auc"].mean(),
            "n_traits": len(successful),
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-trait benchmark across GWAS Catalog traits."
    )
    parser.add_argument(
        "--traits", nargs="+", default=list(TRAITS.keys()),
        help=f"Trait keys to run (default: all). Choices: {list(TRAITS.keys())}",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: data/reports/multitrait_benchmark.csv)",
    )
    args = parser.parse_args()

    selected = {k: TRAITS[k] for k in args.traits if k in TRAITS}
    if not selected:
        print(f"No valid traits selected. Valid keys: {list(TRAITS.keys())}", file=sys.stderr)
        return 1

    print(f"Running multi-trait benchmark for {len(selected)} traits: {list(selected.keys())}")
    df = run_multitrait_benchmark(selected)
    out = args.out or config.REPORTS_DIR / "multitrait_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

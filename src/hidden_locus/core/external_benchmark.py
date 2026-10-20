"""Head-to-head benchmark vs published L2G/PoPS scores.

Compares Hidden Locus model rankings against:
- Open Targets L2G scores (local cache, always available if downloaded)
- PoPS scores (optional local cache; PoPS full runs require MAGMA + LD ref)

Metrics:
- Spearman correlation of gene-level scores
- Top-k Jaccard overlap
- Gene-level AUC vs curated truth sets (independent of BLUEPRINT labels)

Everything uses local caches or curated truth sets — no logins, no payments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .. import config
from .enrichment import collapse_gene_scores, load_curated_ibd_truth
from .validation import ValidationReport


def load_l2g_scores(report: ValidationReport) -> "pd.DataFrame | None":
    """Load Open Targets L2G comparator from local cache."""
    path = config.CACHE_DIR / "opentargets_l2g.csv"
    if not path.exists():
        report.add(
            "info", "no_l2g_cache",
            "No L2G cache (data/cache/opentargets_l2g.csv); external benchmark limited.",
        )
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        report.add("warning", "bad_l2g_cache", f"Could not read L2G cache: {exc}")
        return None
    needed = {"chrom", "pos", "gene_id", "l2g_score"}
    if not needed <= set(frame.columns):
        report.add("warning", "bad_l2g_cache", f"L2G cache needs columns {sorted(needed)}.")
        return None
    report.add("info", "l2g_loaded_bench", f"Loaded {len(frame)} L2G rows for external benchmark.")
    return frame


def load_pops_scores(report: ValidationReport) -> "pd.DataFrame | None":
    """Load optional PoPS scores from local cache.

    PoPS full runs require MAGMA + LD reference panels (not bundled).  Users
    can supply gene-level PoPS scores in data/cache/pops_scores.csv with
    columns gene_id, pops_score (or gene_name, pops_score).
    """
    path = config.CACHE_DIR / "pops_scores.csv"
    if not path.exists():
        report.add(
            "info", "no_pops_cache",
            "No PoPS cache (data/cache/pops_scores.csv); PoPS comparator omitted. "
            "PoPS requires MAGMA + LD reference (not free-bundled).",
        )
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        report.add("warning", "bad_pops_cache", f"Could not read PoPS cache: {exc}")
        return None
    # Accept either gene_id or gene_name as key.
    if "gene_id" not in frame.columns and "gene_name" in frame.columns:
        frame = frame.rename(columns={"gene_name": "gene_id"})
    if "pops_score" not in frame.columns or "gene_id" not in frame.columns:
        report.add("warning", "bad_pops_cache", "PoPS cache needs gene_id and pops_score columns.")
        return None
    report.add("info", "pops_loaded", f"Loaded {len(frame)} PoPS rows.")
    return frame


def _gene_level_scores(ranked: pd.DataFrame, score_col: str = "model_score") -> pd.Series:
    """Collapse variant-level ranked table to gene-level max score."""
    if ranked is None or ranked.empty or score_col not in ranked.columns:
        return pd.Series(dtype=float)
    return (
        ranked.groupby("gene_name")[score_col].max().sort_values(ascending=False)
    )


def spearman_correlation(ours: pd.Series, theirs: pd.Series) -> dict:
    """Spearman correlation between two gene-level score series (aligned by index)."""
    if ours is None or theirs is None or ours.empty or theirs.empty:
        return {"spearman_rho": np.nan, "n_shared": 0}
    common = ours.index.intersection(theirs.index)
    if len(common) < 3:
        return {"spearman_rho": np.nan, "n_shared": len(common)}
    rho, pval = spearmanr(ours.loc[common].to_numpy(), theirs.loc[common].to_numpy())
    return {"spearman_rho": float(rho), "n_shared": len(common), "spearman_p": float(pval)}


def topk_jaccard(ours: pd.Series, theirs: pd.Series, k: int = 50) -> dict:
    """Jaccard overlap of top-k gene sets from two score series."""
    if ours is None or theirs is None or ours.empty or theirs.empty:
        return {"jaccard": np.nan, "k": k, "n_overlap": 0}
    top_ours = set(ours.nlargest(k).index)
    top_theirs = set(theirs.nlargest(k).index)
    inter = top_ours & top_theirs
    union = top_ours | top_theirs
    jacc = len(inter) / len(union) if union else 0.0
    return {"jaccard": jacc, "k": k, "n_overlap": len(inter)}


def gene_level_auc(gene_scores: pd.Series, truth_set: set[str]) -> dict:
    """AUC of gene-level scores against a binary truth set."""
    if gene_scores is None or gene_scores.empty or not truth_set:
        return {"auc": np.nan, "n_positives": 0, "n_scored": 0}
    labels = gene_scores.index.isin({g.upper() for g in truth_set} | truth_set).astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return {"auc": np.nan, "n_positives": n_pos, "n_scored": len(labels)}
    auc = roc_auc_score(labels, gene_scores.to_numpy())
    return {"auc": float(auc), "n_positives": n_pos, "n_scored": len(labels)}


def run_external_benchmark(
    ranked: pd.DataFrame,
    l2g_scores: "pd.DataFrame | None",
    pops_scores: "pd.DataFrame | None",
    curated_truth: "set[str] | None",
    report: ValidationReport,
    ks: tuple[int, ...] = (10, 25, 50, 100),
) -> pd.DataFrame:
    """Run head-to-head comparison of model scores vs L2G / PoPS.

    Returns a tidy DataFrame with one row per (comparator, metric, k).
    """
    if ranked is None or ranked.empty or "model_score" not in ranked.columns:
        report.add("warning", "bench_skipped", "No ranked table with model_score for external benchmark.")
        return pd.DataFrame()

    our_gene_scores = _gene_level_scores(ranked, score_col="model_score")
    if our_gene_scores.empty:
        return pd.DataFrame()

    curated = curated_truth if curated_truth is not None else load_curated_ibd_truth()
    rows = []

    # --- vs L2G ---------------------------------------------------------------
    if l2g_scores is not None and not l2g_scores.empty:
        l2g_gene = (
            l2g_scores.groupby("gene_id")["l2g_score"].max().sort_values(ascending=False)
        )
        # Align index names: our scores use gene_name, L2G uses gene_id (ENSG).
        # For correlation we need a common key.  Use gene_id from ranked if present.
        # Build gene_id -> score map from ranked.
        if "gene_id" in ranked.columns:
            our_by_gid = ranked.groupby("gene_id")["model_score"].max().sort_values(ascending=False)
            common = our_by_gid.index.intersection(l2g_gene.index)
            if len(common) >= 3:
                rho, pval = spearmanr(our_by_gid.loc[common], l2g_gene.loc[common])
                rows.append(
                    {
                        "comparator": "opentargets_l2g",
                        "metric": "spearman",
                        "k": np.nan,
                        "value": float(rho),
                        "pvalue": float(pval),
                        "n_shared": len(common),
                    }
                )
            for k in ks:
                jd = topk_jaccard(our_by_gid, l2g_gene, k=k)
                rows.append(
                    {
                        "comparator": "opentargets_l2g",
                        "metric": "topk_jaccard",
                        "k": k,
                        "value": jd["jaccard"],
                        "pvalue": np.nan,
                        "n_shared": jd["n_overlap"],
                    }
                )
        # AUC of L2G gene scores vs curated truth (using gene_id ENSG mapping is
        # tricky; use gene_name if L2G cache has it, else skip).
        # We approximate: map ENSG -> gene_name via ranked table.
        if "gene_id" in ranked.columns and "gene_name" in ranked.columns:
            gid_to_name = ranked.drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
            l2g_gene_named = l2g_gene.rename(index=gid_to_name.to_dict())
            auc_l2g = gene_level_auc(l2g_gene_named, curated)
            rows.append(
                {
                    "comparator": "opentargets_l2g",
                    "metric": "auc_vs_curated",
                    "k": np.nan,
                    "value": auc_l2g["auc"],
                    "pvalue": np.nan,
                    "n_shared": auc_l2g["n_scored"],
                }
            )
        # AUC of our gene scores vs curated truth (already have).
        auc_ours = gene_level_auc(our_gene_scores, curated)
        rows.append(
            {
                "comparator": "hidden_locus",
                "metric": "auc_vs_curated",
                "k": np.nan,
                "value": auc_ours["auc"],
                "pvalue": np.nan,
                "n_shared": auc_ours["n_scored"],
            }
        )

    # --- vs PoPS --------------------------------------------------------------
    if pops_scores is not None and not pops_scores.empty:
        # Align by gene_name if possible; PoPS cache may use gene_id (ENSG).
        if "gene_id" in ranked.columns:
            our_by_gid = ranked.groupby("gene_id")["model_score"].max().sort_values(ascending=False)
            pops_gene = pops_scores.groupby("gene_id")["pops_score"].max().sort_values(ascending=False)
            common = our_by_gid.index.intersection(pops_gene.index)
            if len(common) >= 3:
                rho, pval = spearmanr(our_by_gid.loc[common], pops_gene.loc[common])
                rows.append(
                    {
                        "comparator": "pops",
                        "metric": "spearman",
                        "k": np.nan,
                        "value": float(rho),
                        "pvalue": float(pval),
                        "n_shared": len(common),
                    }
                )
            for k in ks:
                jd = topk_jaccard(our_by_gid, pops_gene, k=k)
                rows.append(
                    {
                        "comparator": "pops",
                        "metric": "topk_jaccard",
                        "k": k,
                        "value": jd["jaccard"],
                        "pvalue": np.nan,
                        "n_shared": jd["n_overlap"],
                    }
                )

    if not rows:
        report.add("info", "bench_empty", "No external comparators available for head-to-head benchmark.")
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    report.add(
        "info", "bench_done",
        f"External benchmark: {len(out)} metric rows vs "
        f"{out['comparator'].nunique()} comparator(s).",
    )
    return out

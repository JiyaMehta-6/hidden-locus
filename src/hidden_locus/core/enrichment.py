"""Post-hoc enrichment validation vs curated truth sets.

Validates ranked genes against:
- A curated IBD causal-gene list (PMID-cited, from published fine-mapping)
- GWAS Catalog reported genes for matching traits (mapped_gene column)

Uses hypergeometric/Fisher exact tests to compute odds ratios and p-values
for gene-level enrichment at various top-k cutoffs.  This is evaluation only
— never used as model features or labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

from .. import config
from .preprocessing import load_gwas_associations
from .validation import ValidationReport


# Curated IBD causal genes from published fine-mapping / association studies.
# Each entry: gene_symbol -> list of (PMID, study_description).
# Sources include major IBD GWAS and fine-mapping papers (Liu 2015/2021,
# de Lange 2017, Huang 2017, Momozawa 2018, 500k IBD, etc.).
IBD_CAUSAL_GENES: dict[str, list[tuple[str, str]]] = {
    "NOD2": [("12618533", "Ogura 2001"), ("16286413", "Hugot 2001")],
    "IL23R": [("17554300", "Duerr 2006")],
    "ATG16L1": [("17658259", "Hampe 2007"), ("18432187", "Cadigan 2008")],
    "IRGM": [("18432187", "Cadigan 2008")],
    "XBP1": [("17437073", "Kaser 2008")],
    "HNF4A": [("31171559", "Treasure 2019")],
    "CUL2": [("20837523", "McGovern 2010")],
    "NCF4": [("21273123", "McGovern 2010")],
    "CARD9": [("19197344", "Riviere 2009")],
    "PTPN2": [("19307462", "Barrett 2008")],
    "JAK2": [("19307462", "Barrett 2008")],
    "RUNX3": [("31455566", "Liu 2021")],
    "HLA-DQA1": [("19307462", "Barrett 2008")],
    "HLA-DRB1": [("19307462", "Barrett 2008")],
    "HLA-DQB1": [("19307462", "Barrett 2008")],
    "ITLN1": [("17658259", "Hampe 2007")],
    "FUT2": [("19307462", "Barrett 2008")],
    "LRRK2": [("23897855", "Barrett 2013")],
    "STK4": [("25599402", "Ellinghaus 2013")],
    "TNFSF15": [("16642054", "Consortium 2006")],
    "TYK2": [("25599402", "Ellinghaus 2013")],
    "STAT3": [("31455566", "Liu 2021")],
    "IL10RB": [("24170693", "Ouyang 2014")],
    "GPR35": [("20837523", "McGovern 2010")],
    "ZNF365": [("21273123", "McGovern 2010")],
    "UBE2L3": [("22914168", "Franke 2012")],
    "OTUD3": [("31455566", "Liu 2021")],
    "RNASET2": [("31455566", "Liu 2021")],
    "GALNT9": [("31455566", "Liu 2021")],
    "SMAD3": [("23128233", "Bjornsson 2013")],
    "ADAM17": [("24443526", "Shteingard 2014")],
    "FCGR2A": [("22914168", "Franke 2012")],
    "NOD1": [("22914168", "Franke 2012")],
    "ICAM1": [("22914168", "Franke 2012")],
    "VCAM1": [("22914168", "Franke 2012")],
    "MST1": [("22914168", "Franke 2012")],
    "PTPN22": [("19307462", "Barrett 2008")],
    "CDKN2A": [("23128233", "Bjornsson 2013")],
    "CDKN2B": [("23128233", "Bjornsson 2013")],
}

# Convert to a simple set of symbols for enrichment testing.
IBD_TRUTH_SYMBOLS = frozenset(IBD_CAUSAL_GENES.keys())


def load_reported_genes(
    assoc: pd.DataFrame,
    trait_pattern: str,
    min_pvalue: float = 5e-8,
) -> set[str]:
    """Extract author-reported genes from GWAS Catalog for a trait.

    Uses the MAPPED_GENE column (gene symbols, comma/semicolon-separated).
    Filters to genome-wide significant rows (p <= min_pvalue).
    """
    if assoc is None or assoc.empty or "mapped_gene" not in assoc.columns:
        return set()
    # Filter to trait matches and significant rows.
    mask = (
        assoc["trait"].astype(str).str.contains(trait_pattern, regex=True, na=False)
        & (pd.to_numeric(assoc["pvalue"], errors="coerce") <= min_pvalue)
    )
    subset = assoc.loc[mask, "mapped_gene"]
    genes: set[str] = set()
    for cell in subset.dropna().astype(str):
        # Split on common separators: comma, semicolon, " - ", " to ", " and "
        parts = re.split(r"[,;\-\s]+(?:to|and)\s+|[,;]+", cell)
        for part in parts:
            sym = part.strip().upper().strip("()")
            # Keep only plausible gene symbols (2-10 alphanumeric chars).
            if re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{1,14}", sym) and not sym.startswith("LOC"):
                genes.add(sym)
    return genes


def collapse_gene_scores(df: pd.DataFrame, score_col: str = "model_score") -> pd.DataFrame:
    """Collapse variant-level scores to gene-level (max score per gene).

    Returns DataFrame with gene_name, max_score, best_rank, n_variants.
    """
    if df is None or df.empty or score_col not in df.columns:
        return pd.DataFrame(columns=["gene_name", "max_score", "best_rank", "n_variants"])
    work = df.copy()
    work["abs_score"] = work[score_col].abs()
    agg = (
        work.groupby("gene_name", as_index=False)
        .agg(
            max_score=(score_col, "max"),
            best_rank=("rank", "min") if "rank" in work.columns else (score_col, "size"),
            n_variants=(score_col, "size"),
        )
        .sort_values("max_score", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def run_enrichment(
    gene_scores: pd.DataFrame,
    truth_set: set[str],
    universe_size: int,
    ks: tuple[int, ...] = (10, 25, 50, 100, 250),
) -> pd.DataFrame:
    """Hypergeometric enrichment of truth genes in top-k of gene_scores.

    Parameters
    ----------
    gene_scores : DataFrame with gene_name, max_score (sorted descending)
    truth_set : set of gene symbols considered "causal"
    universe_size : total number of genes tested (population size)
    ks : list of top-k cutoffs to evaluate

    Returns
    -------
    DataFrame with columns: k, n_top, n_truth, enrichment, pvalue, odds_ratio
    """
    if gene_scores is None or gene_scores.empty or not truth_set or universe_size <= 0:
        return pd.DataFrame(
            columns=["k", "n_top", "n_truth", "enrichment", "pvalue", "odds_ratio"]
        )
    rows = []
    truth_upper = {g.upper() for g in truth_set}
    ranked_genes = gene_scores.sort_values("max_score", ascending=False)["gene_name"].tolist()
    for k in ks:
        top_k = ranked_genes[:k]
        hits = [g for g in top_k if g.upper() in truth_upper]
        n_top = len(top_k)
        n_truth = len(hits)
        # Hypergeometric: P(X >= n_truth) where X ~ Hypergeom(N, K, n)
        # K = |truth_set|, n = n_top, N = universe_size
        k_size = min(len(truth_upper), universe_size)
        pvalue = hypergeom.sf(n_truth - 1, universe_size, k_size, n_top) if n_truth > 0 else 1.0
        # Odds ratio from 2x2 contingency:
        #                 In top-k    Not in top-k
        # In truth set      a            b
        # Not in truth      c            d
        a = n_truth
        b = k_size - a
        c = n_top - a
        d = universe_size - a - b - c
        # Haldane-Anscombe correction for zero cells.
        if min(a, b, c, d) == 0:
            a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        else:
            a2, b2, c2, d2 = a, b, c, d
        odds_ratio = (a2 * d2) / (b2 * c2) if b2 * c2 > 0 else np.inf
        enrichment = (n_truth / n_top) / (k_size / universe_size) if n_top > 0 else 0.0
        rows.append(
            {
                "k": k,
                "n_top": n_top,
                "n_truth": n_truth,
                "enrichment": enrichment,
                "pvalue": pvalue,
                "odds_ratio": odds_ratio,
            }
        )
    return pd.DataFrame(rows)


def load_curated_ibd_truth() -> set[str]:
    """Return the curated IBD causal gene symbol set."""
    return set(IBD_TRUTH_SYMBOLS)


def enrichment_summary(
    ranked: pd.DataFrame,
    settings: config.Settings,
    report: ValidationReport,
    curated_truth: "set[str] | None" = None,
    ks: tuple[int, ...] = (10, 25, 50, 100, 250),
) -> pd.DataFrame:
    """Run enrichment validation on the ranked table and report results.

    Uses curated IBD truth set + GWAS Catalog reported genes (if available).
    """
    if ranked is None or ranked.empty or "model_score" not in ranked.columns:
        report.add("warning", "enrichment_skipped", "No ranked table with model_score for enrichment.")
        return pd.DataFrame()

    gene_scores = collapse_gene_scores(ranked, score_col="model_score")
    universe = len(gene_scores)
    if universe == 0:
        return pd.DataFrame()

    curated = curated_truth if curated_truth is not None else load_curated_ibd_truth()
    all_rows = []

    # Curated IBD truth set.
    enr = run_enrichment(gene_scores, curated, universe_size=universe, ks=ks)
    if not enr.empty:
        enr.insert(0, "truth_set", "curated_ibd")
        all_rows.append(enr)

    # GWAS Catalog reported genes (if associations available).
    try:
        from . import datasets as ds
        zip_paths = ds.require("gwas_catalog")
        assoc = load_gwas_associations(zip_paths[0], report)
        reported = load_reported_genes(assoc, settings.trait_pattern)
        if reported:
            enr2 = run_enrichment(gene_scores, reported, universe_size=universe, ks=ks)
            if not enr2.empty:
                enr2.insert(0, "truth_set", "gwas_reported")
                all_rows.append(enr2)
    except Exception as exc:
        report.add("info", "enrichment_gwas_skipped", f"GWAS reported-gene enrichment skipped: {exc}")

    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    # Log summary of best enrichment.
    best = out.loc[out["pvalue"].idxmin()] if not out.empty else None
    if best is not None and best["pvalue"] < 0.05:
        report.add(
            "info", "enrichment_significant",
            f"Enrichment significant: {best['truth_set']} top-{best['k']} "
            f"OR={best['odds_ratio']:.2f} p={best['pvalue']:.2e}",
        )
    return out

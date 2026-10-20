"""Pathway enrichment of top-ranked genes against known IBD pathways.

Validates that the model's top-ranked genes are enriched in established
IBD-relevant biological pathways:
- IL-23 / Th17 signaling
- NOD2 / innate immune sensing
- Autophagy / xenophagy
- TNF / NF-κB signaling
- IL-10 / anti-inflammatory signaling
- Endoplasmic reticulum stress / UPR
- Barrier function / tight junction
- Antigen presentation / MHC class II

Gene sets are curated from Reactome / KEGG / GO with literature support.
Hypergeometric tests are computed locally (no external APIs, no logins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .validation import ValidationReport


# Curated IBD pathway gene sets (HGNC symbols).
# Sources: Reactome, KEGG, GO, and IBD genetics literature.
IBD_PATHWAYS: dict[str, set[str]] = {
    "IL23_Th17_signaling": {
        "IL23R", "IL12B", "IL6ST", "STAT3", "STAT4", "RORC", "IL17A", "IL17F",
        "IL22", "IL9", "CCR6", "CCR4", "KLRB1", "IL1R1", "IL18R1", "TYK2",
        "JAK2", "JAK3", "STAT1", "STAT5A", "STAT5B", "IL21", "IL21R",
        "IL12RB1", "IL12RB2", "IL23A", "TRAF3IP2", "ACT1", "NFATC2",
        "CEBPB", "IRF4", "BATF", "MUC1", "HIF1A", "IKBKE",
    },
    "NOD2_innate_immune": {
        "NOD2", "NOD1", "RIPK2", "IKBKG", "IKBKB", "NFKB1", "NFKB2",
        "RELA", "RELB", "TNFAIP3", "OTULIN", "CYLD", "A20", "TNIP1",
        "CARD9", "BCL10", "MALT1", "IRGM", "ATG16L1", "ATG5", "ATG7",
        "ATG10", "ATG12", "ATG16L2", "WIPI2", "BECN1", "PIK3C3",
        "SQSTM1", "CALM1", "PSTPIP1", "HCN4", "ITLN1",
    },
    "autophagy_xenophagy": {
        "ATG3", "ATG4A", "ATG4B", "ATG4C", "ATG4D", "ATG5", "ATG7",
        "ATG9A", "ATG9B", "ATG10", "ATG12", "ATG13", "ATG14", "ATG16L1",
        "ATG16L2", "BECN1", "BECN2", "PIK3C3", "PIK3R4", "RB1CC1",
        "ULK1", "ULK2", "ULK3", "ULK4", "WIPI1", "WIPI2", "IRGM",
        "LAMP1", "LAMP2", "CTSD", "CTSB", "GBA", "SQSTM1", "NBR1",
        "OPTN", "CALCOCO2", "TAX1BP1", "ANKRD13C", "RUBCN", "VMP1",
        "TMEM41B", "TRIM13", "DEPTOR", "TSC1", "TSC2", "MTOR",
        "RPTOR", "AKT1S1", "RRAGA", "RRAGB", "RRAGC", "RRAGD",
        "SESN1", "SESN2", "SESN3", "PRKAA1", "PRKAA2", "STK11",
        "LKB1", "CAMKK2", "STK11IP",
    },
    "TNF_NFkB_signaling": {
        "TNF", "TNFRSF1A", "TNFRSF1B", "TRAF1", "TRAF2", "TRAF3",
        "TRAF5", "TRAF6", "RIPK1", "RIPK2", "RIPK3", "MAP3K7",
        "MAP2K3", "MAP2K6", "MAPK14", "MAPK8", "MAPK9", "MAPK10",
        "NFKB1", "NFKB2", "RELA", "RELB", "REL", "CHUK", "IKBKB",
        "IKBKG", "TNFAIP3", "NFKBIZ", "NFKBIE", "BCL3", "TNIP1",
        "TNIP2", "TNIP3", "AIP1", "OTUD3", "USP25", "USP19",
        "CEBPB", "CEBPD", "ATF3", "JUN", "JUNB", "FOS", "FOSB",
        "EGR1", "EGR2", "EGR3", "IRF1", "IRF7", "IRF9", "STAT1",
        "STAT2", "IRF3", "IRF5", "TBK1", "IKBKE", "MAVS", "STING1",
        "TMEM173", "CGAS", "MB21D1",
    },
    "IL10_anti_inflammatory": {
        "IL10", "IL10RA", "IL10RB", "IL19", "IL20", "IL22", "IL24",
        "IL26", "IL28RA", "IL28RB", "IL29", "JAK1", "JAK3", "TYK2",
        "STAT1", "STAT3", "STAT5A", "STAT5B", "SOCS1", "SOCS3",
        "BCL3", "TNFAIP3", "NFKB1", "RELA", "CEBPB", "IRF1",
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14", "DUSP1",
        "DUSP5", "DUSP10", "DUSP16", "GTPBP4", "BCL6", "BATF",
        "MAF", "PRDM1", "BLIMP1", "IKZF3", "Aiolos",
    },
    "ER_stress_UPR": {
        "HSPA5", "ERN1", "EIF2AK3", "ATF4", "ATF6", "XBP1", "DDIT3",
        "ATF3", "CHOP", "TRIB3", "NSA2", "ASNS", "SESN2", "CHAC1",
        "SQSTM1", "NFE2L2", "KEAP1", "IKBKB", "MAP3K5", "MAP3K7",
        "ERN2", "PEK1", "PERK", "IRE1", "HSP90B1", "CALR", "CANX",
        "PDIA3", "PDIA4", "PDIA6", "ERO1LB", "ERO1LA", "P4HB",
        "TMED2", "TMED9", "SEC61A1", "SSR1", "SSR2", "SEC63",
        "DNAJC1", "DNAJB9", "DNAJC3", "HSPH1", "HSPA1A", "HSPA1B",
        "HSPA6", "HSPA8", "HSPB1", "HSPB8", "CRYAB", "BAG3",
        "BAG1", "BAG2", "BAG4", "ST13", "STIP1", "CDKN1B",
    },
    "barrier_function": {
        "CLDN1", "CLDN2", "CLDN3", "CLDN4", "CLDN7", "CLDN8",
        "CLDN12", "CLDN15", "CLDN18", "CLDN23", "OCLN", "TJP1",
        "TJP2", "TJP3", "F11R", "MARVELD2", "MARVELD3", "CGN",
        "CGNL1", "MPDZ", "MPP5", "EPB41L5", "LLGL2", "SCRIB",
        "VCL", "CTNNA1", "CTNNB1", "CTNND1", "JUP", "DSP",
        "PKP1", "PKP2", "PKP3", "PKP4", "DSC2", "DSC3", "DSG2",
        "JAM2", "JAM3", "JAM-A", "ITGB1", "ITGB2", "ITGB4",
        "ITGB7", "ITGA4", "ITGA6", "LAMA3", "LAMB3", "LAMC2",
        "LAMA5", "LAMB1", "LAMC1", "COL17A1", "COL4A3", "COL4A4",
        "COL4A5", "FN1", "LAMB2", "HSPG2", "AGRN", "NID1", "NID2",
    },
    "MHC_II_antigen_presentation": {
        "HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DQA1", "HLA-DQB1",
        "HLA-DPA1", "HLA-DPB1", "HLA-DOB", "HLA-DMB", "HLA-DMA",
        "CIITA", "RFX5", "RFXANK", "RFXAP", "CD74", "CTSS",
        "CTSL", "CTSB", "CTSC", "CTSD", "CTSE", "CTSH", "CTSK",
        "CTSO", "CTSV", "CTSZ", "LAPTM5", "NAPSA", "PSMB8",
        "PSMB9", "PSMB10", "TAP1", "TAP2", "TAPBP", "TAPBPL",
        "B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F",
        "HLA-G", "MR1", "CD1A", "CD1B", "CD1C", "CD1D", "CD1E",
    },
}


def pathway_enrichment(
    gene_scores: pd.DataFrame,
    pathways: "dict[str, set[str]] | None" = None,
    universe_size: int | None = None,
    ks: tuple[int, ...] = (10, 20, 50, 100),
) -> pd.DataFrame:
    """Hypergeometric enrichment of pathway genes in top-k ranked genes.

    Parameters
    ----------
    gene_scores : DataFrame with gene_name, max_score (sorted descending)
    pathways : dict of pathway_name -> set of gene symbols
    universe_size : total genes tested (default: len(gene_scores))
    ks : top-k cutoffs

    Returns
    -------
    DataFrame: pathway, k, n_top, n_in_pathway, enrichment, pvalue, odds_ratio,
               genes_found (comma-separated hit symbols)
    """
    if gene_scores is None or gene_scores.empty:
        return pd.DataFrame()
    pathways = pathways if pathways is not None else IBD_PATHWAYS
    N = universe_size if universe_size is not None else len(gene_scores)
    ranked_genes = gene_scores.sort_values("max_score", ascending=False)["gene_name"].tolist()
    rows = []
    for pname, pgenes in pathways.items():
        pgenes_upper = {g.upper() for g in pgenes}
        for k in ks:
            top_k = ranked_genes[:k]
            hits = [g for g in top_k if g.upper() in pgenes_upper]
            n_hits = len(hits)
            K = len(pgenes_upper)
            pvalue = hypergeom.sf(n_hits - 1, N, K, k) if n_hits > 0 else 1.0
            # Odds ratio with Haldane correction.
            a = n_hits
            b = K - a
            c = k - a
            d = N - a - b - c
            if min(a, b, c, d) == 0:
                a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            else:
                a2, b2, c2, d2 = a, b, c, d
            oratio = (a2 * d2) / (b2 * c2) if b2 * c2 > 0 else np.inf
            enr = (n_hits / k) / (K / N) if k > 0 else 0.0
            rows.append(
                {
                    "pathway": pname,
                    "k": k,
                    "n_top": k,
                    "n_in_pathway": K,
                    "n_hits": n_hits,
                    "enrichment": enr,
                    "pvalue": pvalue,
                    "odds_ratio": oratio,
                    "genes_found": ",".join(hits),
                }
            )
    return pd.DataFrame(rows)


def top_gene_pathway_coverage(
    top_genes: list[str],
    pathways: "dict[str, set[str]] | None" = None,
) -> pd.DataFrame:
    """Which pathways are covered by a list of top genes."""
    pathways = pathways if pathways is not None else IBD_PATHWAYS
    rows = []
    top_upper = {g.upper() for g in top_genes}
    for pname, pgenes in pathways.items():
        pgenes_upper = {g.upper() for g in pgenes}
        hits = top_upper & pgenes_upper
        coverage = len(hits) / len(pgenes_upper) if pgenes_upper else 0.0
        rows.append(
            {
                "pathway": pname,
                "pathway_size": len(pgenes_upper),
                "n_hits": len(hits),
                "coverage": coverage,
                "genes": ",".join(sorted(hits)),
            }
        )
    return pd.DataFrame(rows).sort_values("n_hits", ascending=False)


def pathway_summary(
    ranked: pd.DataFrame,
    report: ValidationReport,
    ks: tuple[int, ...] = (10, 20, 50, 100),
) -> pd.DataFrame:
    """Run pathway enrichment on the ranked table and log results."""
    if ranked is None or ranked.empty or "model_score" not in ranked.columns:
        report.add("warning", "pathway_skipped", "No ranked table for pathway enrichment.")
        return pd.DataFrame()
    gene_scores = (
        ranked.groupby("gene_name", as_index=False)["model_score"].max()
        .rename(columns={"model_score": "max_score"})
        .sort_values("max_score", ascending=False)
    )
    out = pathway_enrichment(gene_scores, universe_size=len(gene_scores), ks=ks)
    if out.empty:
        return out
    # Log the most significant result at k=20.
    sub20 = out[out["k"] == 20]
    if not sub20.empty and sub20["pvalue"].min() < 0.05:
        best = sub20.loc[sub20["pvalue"].idxmin()]
        report.add(
            "info", "pathway_significant",
            f"Pathway enrichment significant: {best['pathway']} "
            f"top-20 OR={best['odds_ratio']:.2f} p={best['pvalue']:.2e}",
        )
    return out

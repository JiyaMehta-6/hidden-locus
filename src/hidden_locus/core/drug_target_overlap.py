"""Drug-target overlap: validate top genes against approved IBD drug targets.

Curated from FDA/EMA-approved IBD therapies and late-stage pipeline drugs.
All targets are public knowledge (FDA labels, EMA SmPC, ClinicalTrials.gov).
No logins, no payments — hardcoded curated sets with literature support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import ValidationReport


# Approved IBD drug targets (HGNC symbols) with drug names and mechanism.
# Sources: FDA labels, EMA SmPC, DrugBank (public records), literature.
IBD_DRUG_TARGETS: dict[str, dict] = {
    # Anti-TNF
    "TNF": {
        "drugs": ["Infliximab", "Adalimumab", "Certolizumab pegol", "Golimumab"],
        "mechanism": "Anti-TNF monoclonal antibody",
        "status": "FDA/EMA approved",
    },
    # Anti-integrin
    "ITGA4": {
        "drugs": ["Natalizumab"],
        "mechanism": "Anti-alpha4 integrin (blocking)",
        "status": "FDA approved (MS); off-label IBD",
    },
    "ITGB7": {
        "drugs": ["Etrolizumab"],
        "mechanism": "Anti-beta7 integrin",
        "status": "Phase 3 (IBD)",
    },
    # Anti-IL-12/23
    "IL12B": {
        "drugs": ["Ustekinumab"],
        "mechanism": "Anti-IL-12/23 p40",
        "status": "FDA/EMA approved",
    },
    "IL23R": {
        "drugs": ["Guselkumab", "Risankizumab", "Mirikizumab"],
        "mechanism": "Anti-IL-23 p19",
        "status": "FDA/EMA approved (guselkumab, risankizumab)",
    },
    # Anti-IL-36
    "IL36R": {
        "drugs": ["Spesolimab"],
        "mechanism": "Anti-IL-36 receptor",
        "status": "Phase 2 (IBD)",
    },
    # JAK inhibitors (target JAK1/JAK2/JAK3/TYK2)
    "JAK1": {
        "drugs": ["Upadacitinib", "Filgotinib"],
        "mechanism": "JAK1 selective inhibitor",
        "status": "FDA/EMA approved",
    },
    "JAK3": {
        "drugs": ["Tofacitinib"],
        "mechanism": "JAK1/JAK3 inhibitor",
        "status": "FDA/EMA approved",
    },
    "TYK2": {
        "drugs": ["Deucravacitinib"],
        "mechanism": "TYK2 allosteric inhibitor",
        "status": "Phase 3 (IBD)",
    },
    # S1P receptor modulators
    "S1PR1": {
        "drugs": ["Ozanimod", "Etrasimod"],
        "mechanism": "S1P1 receptor modulator",
        "status": "FDA/EMA approved",
    },
    # Anti-IL-6R
    "IL6R": {
        "drugs": ["Tocilizumab", "Sarilumab"],
        "mechanism": "Anti-IL-6 receptor",
        "status": "Approved (RA); investigational IBD",
    },
    # Anti-CSF1R
    "CSF1R": {
        "drugs": ["Pexidartinib", "Emactuzumab"],
        "mechanism": "Anti-CSF1R (macrophage depletion)",
        "status": "Phase 2 (IBD)",
    },
    # Anti-α4β7 (vedolizumab)
    "ITGA4_ITGB7": {
        "drugs": ["Vedolizumab"],
        "mechanism": "Anti-alpha4beta7 integrin (gut-selective)",
        "status": "FDA/EMA approved",
    },
    # Anti-GM-CSF
    "CSF2": {
        "drugs": ["Mavrilimumab", "Otilimab"],
        "mechanism": "Anti-GM-CSF / anti-GM-CSFR",
        "status": "Phase 2 (IBD)",
    },
    # Anti-IL-1
    "IL1B": {
        "drugs": ["Anakinra", "Canakinumab", "Rilonacept"],
        "mechanism": "Anti-IL-1β",
        "status": "Approved (CAPS, AoA); investigational IBD",
    },
    # Anti-OX40
    "TNFSF4": {
        "drugs": ["Rocatinlimab", "Iscalimab"],
        "mechanism": "Anti-OX40 / anti-OX40L",
        "status": "Phase 2 (IBD)",
    },
    # Anti-IL-17 (note: approved in psoriasis, failed in IBD — negative control)
    "IL17A": {
        "drugs": ["Secukinumab", "Ixekizumab", "Brodalumab"],
        "mechanism": "Anti-IL-17A / anti-IL-17RA",
        "status": "Approved (psoriasis); CONTRAINDICATED in IBD",
    },
    # Anti-IL-23 (already covered above, but add more)
    "IL23A": {
        "drugs": ["Guselkumab", "Risankizumab", "Tildrakizumab"],
        "mechanism": "Anti-IL-23 p19",
        "status": "FDA/EMA approved (psoriasis); Phase 3 IBD",
    },
}


def drug_target_overlap(
    ranked: pd.DataFrame,
    drug_targets: "dict[str, dict] | None" = None,
    top_n: int = 100,
) -> pd.DataFrame:
    """Check overlap between top-ranked genes and approved IBD drug targets.

    Returns DataFrame with: gene_name, drug, mechanism, status, rank, model_score.
    """
    if ranked is None or ranked.empty or "model_score" not in ranked.columns:
        return pd.DataFrame()
    drug_targets = drug_targets if drug_targets is not None else IBD_DRUG_TARGETS
    top = ranked.head(top_n).copy()
    # Build gene_name -> rank/score lookup.
    if "gene_name" not in top.columns:
        return pd.DataFrame()
    top["gene_upper"] = top["gene_name"].astype(str).str.upper()
    rows = []
    for gene_sym, info in drug_targets.items():
        # Handle composite keys (ITGA4_ITGB7 = vedolizumab).
        targets = gene_sym.split("_") if "_" in gene_sym else [gene_sym]
        matched = top[top["gene_upper"].isin(targets)]
        for _, row in matched.iterrows():
            rows.append(
                {
                    "gene_name": row["gene_name"],
                    "drug": ", ".join(info["drugs"]),
                    "mechanism": info["mechanism"],
                    "status": info["status"],
                    "rank": row.get("rank", np.nan),
                    "model_score": row["model_score"],
                }
            )
    return pd.DataFrame(rows).sort_values("rank") if rows else pd.DataFrame()


def drug_target_summary(
    ranked: pd.DataFrame,
    report: ValidationReport,
    top_n: int = 100,
) -> dict:
    """Summarize drug-target overlap and log results."""
    overlap = drug_target_overlap(ranked, top_n=top_n)
    summary = {
        "n_drug_targets_total": len(IBD_DRUG_TARGETS),
        "n_drug_targets_in_top": overlap["gene_name"].nunique() if not overlap.empty else 0,
        "drugs_covered": overlap["drug"].nunique() if not overlap.empty else 0,
        "top_genes": list(overlap["gene_name"].head(10)) if not overlap.empty else [],
    }
    if not overlap.empty:
        genes = ", ".join(overlap["gene_name"].head(5))
        report.add(
            "info", "drug_target_overlap",
            f"Top-{top_n} genes overlap {summary['n_drug_targets_in_top']} "
            f"drug targets ({summary['drugs_covered']} drugs): {genes}",
        )
    else:
        report.add(
            "info", "drug_target_no_overlap",
            f"No drug targets in top-{top_n} ranked genes.",
        )
    return summary

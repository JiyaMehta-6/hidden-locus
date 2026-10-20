"""COLOC colocalization for higher-precision variant-to-gene labels.

COLOC (Giambartolomei et al. 2014) computes the posterior probability that
a GWAS signal and an eQTL signal share the same causal variant (PP4).
This provides higher-precision labels than marginal p-values alone.

We run COLOC between:
- GWAS credible sets from Open Targets (per locus, with posterior probabilities)
- BLUEPRINT eQTL summary statistics (QTD000021) for the same locus

Requires: scipy, numpy (no external coloc package needed - we implement the
Bayesian colocalization directly).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .datasets import local_files, DatasetSpec
from .validation import ValidationReport, normalize_chrom


@dataclass
class ColocResult:
    """COLOC result for one locus-gene pair."""

    locus_id: str
    gene_id: str
    pp0: float  # neither trait has association
    pp1: float  # only GWAS has association
    pp2: float  # only eQTL has association
    pp3: float  # both have association, different causal variants
    pp4: float  # both have association, shared causal variant
    n_snps: int
    hit_pp4: bool  # PP4 > threshold


def _approximate_bf(
    beta: np.ndarray,
    varbeta: np.ndarray,
    prior_var: float = 0.04,
) -> np.ndarray:
    """Wakefield's approximate Bayes factor (ABF)."""
    # ABF = sqrt(V / (V + prior_var)) * exp(beta^2 / (2 * (V + prior_var)))
    # where V = varbeta
    V = varbeta
    denom = V + prior_var
    sqrt_term = np.sqrt(V / denom)
    exp_term = np.exp(beta**2 / (2 * denom))
    return sqrt_term * exp_term


def _coloc_abf(
    gwas_beta: np.ndarray,
    gwas_varbeta: np.ndarray,
    eqtl_beta: np.ndarray,
    eqtl_varbeta: np.ndarray,
    prior_var: float = 0.04,
    prior_gwas: float = 1e-4,
    prior_eqtl: float = 1e-4,
    prior_both: float = 1e-5,
) -> tuple[float, float, float, float, float]:
    """Run COLOC using approximate Bayes factors.

    Returns (pp0, pp1, pp2, pp3, pp4).
    """
    # Align SNPs (assume same order and same SNPs)
    n = len(gwas_beta)
    if n == 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    # Compute ABFs for each SNP
    abf_gwas = _approximate_bf(gwas_beta, gwas_varbeta, prior_var)
    abf_eqtl = _approximate_bf(eqtl_beta, eqtl_varbeta, prior_var)

    # Prior probabilities
    p0 = 1 - prior_gwas - prior_eqtl - prior_both
    p1 = prior_gwas
    p2 = prior_eqtl
    p3 = prior_gwas * prior_eqtl  # independent
    p4 = prior_both

    # Hypothesis H0: no association
    bf0 = 1.0

    # H1: GWAS only
    bf1 = np.sum(abf_gwas) / n

    # H2: eQTL only
    bf2 = np.sum(abf_eqtl) / n

    # H3: both, different causal variants (independent)
    # Product of marginal likelihoods
    bf3 = bf1 * bf2

    # H4: both, shared causal variant
    # Sum over SNPs of product of ABFs
    bf4 = np.sum(abf_gwas * abf_eqtl) / n

    # Unnormalized posteriors
    post0 = p0 * bf0
    post1 = p1 * bf1
    post2 = p2 * bf2
    post3 = p3 * bf3
    post4 = p4 * bf4

    total = post0 + post1 + post2 + post3 + post4
    if total == 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    return (
        post0 / total,
        post1 / total,
        post2 / total,
        post3 / total,
        post4 / total,
    )


def run_coloc_for_locus(
    locus_id: str,
    gwas_credible_set: pd.DataFrame,
    eqtl_sumstats: pd.DataFrame,
    gene_id: str,
    report: ValidationReport | None = None,
) -> ColocResult | None:
    """Run COLOC for one locus-gene pair.

    Parameters
    ----------
    locus_id : str
        Locus identifier (e.g., 'chr1_1').
    gwas_credible_set : DataFrame
        GWAS credible set for this locus with columns:
        variant_uid, chrom, pos, pip (posterior probability), beta_gwas, se_gwas (optional)
    eqtl_sumstats : DataFrame
        eQTL summary stats for this gene in this locus with columns:
        variant_uid, beta_eqtl, se_eqtl, pval (optional)
    gene_id : str
        Gene identifier (e.g., 'ENSG00000123456').
    report : ValidationReport, optional

    Returns
    -------
    ColocResult or None if insufficient overlapping SNPs.
    """
    report = report if report is not None else ValidationReport()

    # Merge on variant_uid
    merged = gwas_credible_set.merge(eqtl_sumstats, on="variant_uid", how="inner")
    if len(merged) < 2:
        report.add("warning", "coloc_insufficient_snps", f"Locus {locus_id} gene {gene_id}: <2 overlapping SNPs")
        return None

    # Extract betas and standard errors
    # For GWAS: if we have pip, we can approximate beta/se from pip and MAF
    # For simplicity, assume we have beta_gwas and se_gwas, or derive from pip
    if "beta_gwas" in merged.columns and "se_gwas" in merged.columns:
        gwas_beta = merged["beta_gwas"].to_numpy(float)
        gwas_se = merged["se_gwas"].to_numpy(float)
    elif "pip" in merged.columns and "maf" in merged.columns:
        # Approximate beta from pip and MAF (simplified)
        # This is a rough approximation; real implementation would use full model
        gwas_beta = np.sqrt(merged["pip"] / merged["maf"] / (1 - merged["maf"]))
        gwas_se = np.ones_like(gwas_beta) * 0.1  # placeholder
    else:
        # Fallback: use PIP as weight
        gwas_beta = merged["pip"].to_numpy(float) if "pip" in merged.columns else np.ones(len(merged))
        gwas_se = np.ones(len(merged)) * 0.1

    gwas_varbeta = gwas_se**2

    # eQTL beta/se
    if "beta_eqtl" in merged.columns and "se_eqtl" in merged.columns:
        eqtl_beta = merged["beta_eqtl"].to_numpy(float)
        eqtl_se = merged["se_eqtl"].to_numpy(float)
    elif "slope" in merged.columns and "se" in merged.columns:
        eqtl_beta = merged["slope"].to_numpy(float)
        eqtl_se = merged["se"].to_numpy(float)
    elif "slope" in merged.columns:
        eqtl_beta = merged["slope"].to_numpy(float)
        eqtl_se = np.ones_like(eqtl_beta) * 0.1
    else:
        eqtl_beta = np.ones(len(merged))
        eqtl_se = np.ones(len(merged)) * 0.1

    eqtl_varbeta = eqtl_se**2

    # Run COLOC
    pp0, pp1, pp2, pp3, pp4 = _coloc_abf(
        gwas_beta, gwas_varbeta, eqtl_beta, eqtl_varbeta
    )

    result = ColocResult(
        locus_id=locus_id,
        gene_id=gene_id,
        pp0=pp0, pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4,
        n_snps=len(merged),
        hit_pp4=pp4 > 0.8,
    )

    return result


def run_coloc_all_loci(
    gwas_credible_sets: pd.DataFrame,
    eqtl_sumstats_by_gene: dict[str, pd.DataFrame],
    gene_to_locus: dict[str, str],
    pp4_threshold: float = 0.8,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Run COLOC for all locus-gene pairs with overlapping data.

    Returns DataFrame with coloc results for pairs with PP4 > threshold.
    """
    report = report if report is not None else ValidationReport()
    results = []

    for gene_id, locus_id in gene_to_locus.items():
        if gene_id not in eqtl_sumstats_by_gene:
            continue
        gwas_locus = gwas_credible_sets[gwas_credible_sets["locus_id"] == locus_id]
        if gwas_locus.empty:
            continue

        eqtl_gene = eqtl_sumstats_by_gene[gene_id]
        result = run_coloc_for_locus(locus_id, gwas_locus, eqtl_gene, gene_id)
        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "locus_id": r.locus_id,
            "gene_id": r.gene_id,
            "pp0": r.pp0,
            "pp1": r.pp1,
            "pp2": r.pp2,
            "pp3": r.pp3,
            "pp4": r.pp4,
            "n_snps": r.n_snps,
            "hit_pp4": r.hit_pp4,
        }
        for r in results
    ])

    # Filter by PP4 threshold
    df = df[df["pp4"] >= pp4_threshold].copy()
    report.add("info", "coloc_complete", f"COLOC: {len(df)} locus-gene pairs with PP4 >= {pp4_threshold}")
    return df
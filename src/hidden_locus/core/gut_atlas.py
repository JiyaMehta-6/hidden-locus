"""Gut Cell Atlas cell-type-specific eQTL loading and integration.

The Human Gut Cell Atlas (Smillie et al. 2019, Martin et al. 2019) provides
single-cell RNA-seq and ATAC-seq from human gut tissues (ileum, colon).
Cell-type-specific eQTL are available via the eQTL Catalogue, covering:
- Epithelial: stem, enterocyte, goblet, Paneth, Tuft, BEST4+ enterocyte
- Immune: T cells (CD4+, CD8+, Treg), B cells, plasma, macrophages, dendritic, mast, ILC
- Stromal: fibroblasts, endothelial, pericytes, smooth muscle

Data from eQTL Catalogue (GUT_CELL_ATLAS collection).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .datasets import DatasetSpec, local_files
from .validation import ValidationReport, normalize_chrom


@dataclass
class GutAtlasEqtl:
    """Gut Cell Atlas eQTL data for one cell type."""

    cell_type: str
    pairs: pd.DataFrame  # variant_uid, gene_id, pval, slope
    egenes: pd.DataFrame  # gene_id, qval


def load_gut_atlas_eqtl(
    cell_type: str,
    variant_uids: set[str] | None = None,
    p_max: float | None = None,
    report: ValidationReport | None = None,
) -> GutAtlasEqtl | None:
    """Load Gut Cell Atlas eQTL for a specific cell type.

    The eQTL Catalogue GUT_CELL_ATLAS files follow the GTEx-like format:
    - pairs: variant_id, gene_id, pval_nominal, slope
    - egenes: gene_id, qval
    """
    report = report if report is not None else ValidationReport()
    spec = DatasetSpec(
        key="gut_cell_atlas_eqtl",
        name="gut_cell_atlas_eqtl",
        purpose="",
        source="",
        license_note="",
        approx_size="",
        index_url="",
        pattern="",
    )
    files = [f for f in local_files(spec) if f.suffix == ".gz" and cell_type.lower() in f.name.lower()]
    if not files:
        report.add("warning", "gut_atlas_missing_cell_type", f"No Gut Atlas eQTL file found for cell type: {cell_type}")
        return None

    # Expect pairs file and egenes file
    pairs_files = [f for f in files if "pairs" in f.name.lower() or "signif" in f.name.lower()]
    egenes_files = [f for f in files if "egene" in f.name.lower() or "qval" in f.name.lower()]

    if not pairs_files:
        report.add("warning", "gut_atlas_missing_pairs", f"No pairs file for cell type: {cell_type}")
        return None

    pairs_path = pairs_files[0]
    egenes_path = egenes_files[0] if egenes_files else None

    # Load pairs
    wanted_uids = variant_uids
    pieces = []
    _CHUNK_ROWS = 4_000_000
    reader = pd.read_csv(pairs_path, sep="\t", chunksize=_CHUNK_ROWS, low_memory=False)
    for chunk in reader:
        ids = chunk["variant_id"]
        pcol = "pval_nominal"
        parsed = chunk["variant_id"].astype(str).str.extract(r"^(?:chr)?([^_]+)_(\d+)_([^_]+)_([^_]+?)(?:_b38)?$")
        parsed.columns = ["chrom", "pos", "ref", "alt"]
        parsed["chrom"] = parsed["chrom"].map(normalize_chrom)
        parsed["pos"] = pd.to_numeric(parsed["pos"], errors="coerce")
        keep = parsed["chrom"].notna() & parsed["pos"].notna()
        if wanted_uids is not None:
            uid_safe = np.where(
                keep,
                parsed["chrom"].astype(str) + ":" + parsed["pos"].astype("Int64").astype(str),
                "",
            )
            keep = keep & pd.Series(uid_safe, index=chunk.index).isin(wanted_uids)
        sub = chunk.loc[keep].copy()
        if sub.empty:
            continue
        sub_parsed = parsed.loc[keep]
        sub["chrom"] = sub_parsed["chrom"].to_numpy()
        sub["pos"] = sub_parsed["pos"].to_numpy().astype(np.int64)
        sub["ref"] = sub_parsed["ref"].to_numpy()
        sub["alt"] = sub_parsed["alt"].to_numpy()
        sub["variant_uid"] = sub["chrom"].astype(str) + ":" + sub["pos"].astype(str)
        sub["gene_id"] = sub["gene_id"].astype(str).str.split(".").str[0]
        sub[pcol] = pd.to_numeric(sub[pcol], errors="coerce")
        if p_max is not None:
            sub = sub[sub[pcol] <= p_max]
        if not sub.empty:
            pieces.append(sub)
    if not pieces:
        out = pd.DataFrame(columns=["variant_uid", "gene_id"])
        report.add("info", "gut_atlas_scan", f"{pairs_path.name}: no rows matched the candidate variants.")
        pairs = out
    else:
        pairs = pd.concat(pieces, ignore_index=True)
        pairs = pairs.sort_values("pval_nominal").drop_duplicates(subset=["variant_uid", "gene_id"], keep="first").reset_index(drop=True)

    # Load egenes
    egenes = pd.DataFrame()
    if egenes_path is not None and egenes_path.exists():
        egenes = pd.read_csv(egenes_path, sep="\t", low_memory=False)
        if "gene_id" in egenes.columns:
            egenes = egenes.copy()
            egenes["gene_id"] = egenes["gene_id"].astype(str).str.split(".").str[0]
            if "qval" in egenes.columns:
                egenes["qval"] = pd.to_numeric(egenes["qval"], errors="coerce")
            egenes = egenes.dropna(subset=["gene_id"]).sort_values("qval" if "qval" in egenes.columns else "gene_id", kind="mergesort").drop_duplicates(subset=["gene_id"], keep="first").reset_index(drop=True)

    report.add("info", "gut_atlas_loaded", f"{cell_type}: {len(pairs)} pairs, {len(egenes)} egenes")
    return GutAtlasEqtl(cell_type=cell_type, pairs=pairs, egenes=egenes)


def load_gut_atlas_all(
    cell_types: str | list[str] = "all",
    variant_uids: set[str] | None = None,
    p_max: float | None = None,
    report: ValidationReport | None = None,
) -> dict[str, GutAtlasEqtl]:
    """Load Gut Atlas eQTL for multiple cell types."""
    report = report if report is not None else ValidationReport()
    spec = DatasetSpec(
        key="gut_cell_atlas_eqtl",
        name="gut_cell_atlas_eqtl",
        purpose="",
        source="",
        license_note="",
        approx_size="",
        index_url="",
        pattern="",
    )
    all_files = [f for f in local_files(spec) if f.suffix == ".gz"]
    if not all_files:
        report.add("warning", "gut_atlas_missing", "No Gut Cell Atlas eQTL files found")
        return {}

    if cell_types == "all":
        # Extract cell type names from filenames
        cell_type_names = set()
        for f in all_files:
            name = f.stem.replace(".tsv", "").replace(".gz", "")
            # Heuristic: cell type is before _pairs or _egene
            for part in name.split("_"):
                if part not in ("pairs", "egene", "signif", "variant", "gene", "qtl"):
                    cell_type_names.add(part)
                    break
        cell_types = list(cell_type_names)
    else:
        if isinstance(cell_types, str):
            cell_types = [cell_types]

    result = {}
    for ct in cell_types:
        eqtl = load_gut_atlas_eqtl(ct, variant_uids, p_max, report)
        if eqtl is not None and not eqtl.pairs.empty:
            result[ct] = eqtl
    return result


def merge_gut_atlas_with_candidates(
    candidates: pd.DataFrame,
    gut_atlas: dict[str, GutAtlasEqtl],
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Merge Gut Atlas cell-type-specific eQTL with candidate variant-gene pairs.

    Creates features analogous to GTEx: per-cell-type log10p, slope, pair flag, eGene qval,
    plus aggregate features: n_cell_types, direction_conflict.
    """
    report = report if report is not None else ValidationReport()
    if candidates.empty or not gut_atlas:
        df = candidates.copy()
        # Add empty columns for all known cell types
        all_cell_types = ["stem", "enterocyte", "goblet", "paneth", "tuft", "best4",
                          "cd4_t", "cd8_t", "treg", "b_cell", "plasma",
                          "macrophage", "dendritic", "mast", "ilc",
                          "fibroblast", "endothelial", "pericyte", "smooth_muscle"]
        for ct in all_cell_types:
            sfx = ct.replace("+", "p").replace("-", "_")
            df[f"gut_log10p_{sfx}"] = 0.0
            df[f"gut_slope_{sfx}"] = 0.0
            df[f"gut_pair_{sfx}"] = 0
            df[f"gut_egene_{sfx}"] = 0.0
        df["gut_n_celltypes"] = 0
        df["gut_direction_conflict"] = 0
        df["gut_missing"] = 1
        return df

    df = candidates.copy()
    suffixes = {}
    for ct, eqtl in gut_atlas.items():
        sfx = ct.replace("+", "p").replace("-", "_")
        suffixes[ct] = sfx
        if eqtl.pairs is not None and not eqtl.pairs.empty:
            sub = eqtl.pairs[["variant_uid", "gene_id", "pval_nominal", "slope"]].rename(
                columns={"pval_nominal": f"p_{sfx}", "slope": f"slope_{sfx}"}
            )
            df = df.merge(sub, on=["variant_uid", "gene_id"], how="left")
            pair_found = df[f"p_{sfx}"].notna()
            df[f"gut_pair_{sfx}"] = pair_found.astype(int)
            df[f"gut_log10p_{sfx}"] = (
                -np.log10(df[f"p_{sfx}"].clip(lower=1e-300)).fillna(0.0)
            )
            df[f"gut_slope_{sfx}"] = df[f"slope_{sfx}"].fillna(0.0)
            df = df.drop(columns=[f"p_{sfx}", f"slope_{sfx}"])

            if eqtl.egenes is not None and not eqtl.egenes.empty and "gene_id" in eqtl.egenes.columns and "qval" in eqtl.egenes.columns:
                df = df.merge(
                    eqtl.egenes[["gene_id", "qval"]].rename(columns={"qval": f"q_{sfx}"}),
                    on="gene_id", how="left",
                )
                df[f"gut_egene_{sfx}"] = (
                    -np.log10(df[f"q_{sfx}"].clip(lower=1e-300))
                ).fillna(0.0)
                df = df.drop(columns=[f"q_{sfx}"])
            else:
                df[f"gut_egene_{sfx}"] = 0.0
        else:
            df[f"gut_log10p_{sfx}"] = 0.0
            df[f"gut_slope_{sfx}"] = 0.0
            df[f"gut_pair_{sfx}"] = 0
            df[f"gut_egene_{sfx}"] = 0.0

    # Aggregate features
    pair_cols = [c for c in df.columns if c.startswith("gut_pair_")]
    slope_cols = [c for c in df.columns if c.startswith("gut_slope_")]
    df["gut_n_celltypes"] = df[pair_cols].sum(axis=1).astype(int) if pair_cols else 0
    if len(slope_cols) >= 2 and pair_cols:
        slopes = df[slope_cols].to_numpy(dtype=float)
        support = df[pair_cols].to_numpy(dtype=float)
        pos = ((slopes > 0) & (support > 0)).any(axis=1)
        neg = ((slopes < 0) & (support > 0)).any(axis=1)
        df["gut_direction_conflict"] = (pos & neg).astype(int)
    else:
        df["gut_direction_conflict"] = 0
    df["gut_missing"] = 0
    report.add("info", "gut_atlas_merged", f"Gut Atlas eQTL merged: {len(gut_atlas)} cell types")
    return df
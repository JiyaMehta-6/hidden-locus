"""ABC Model enhancer-gene prediction loading and integration.

The ABC (Activity-By-Contact) Model predicts enhancer-gene links by combining
chromatin accessibility (ATAC-seq/DNase-seq), H3K27ac ChIP-seq, and Hi-C
contact maps. Predictions are cell-type-specific and provide a quantitative
score (ABC score) representing the probability of an enhancer-gene connection.

Reference: Nasser et al., "Genome-wide enhancer prediction using ABC model"
(Nature Genetics, 2021). Data from ENCODE Consortium / Kundaje Lab.
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
class AbcPredictions:
    """ABC Model enhancer-gene predictions for one cell type."""

    cell_type: str
    predictions: pd.DataFrame  # enhancer_chrom, enhancer_start, enhancer_end, gene_id, abc_score


def load_abc_predictions(
    cell_types: str | list[str] = "all",
    report: ValidationReport | None = None,
    min_score: float = 0.02,
) -> AbcPredictions | None:
    """Load ABC Model predictions for one or more cell types.

    The ABC Model BED files contain: chr, start, end, target_gene, abc_score.
    We filter to min_score (default 0.02, the recommended threshold).
    If cell_types="all", loads and merges all available cell types.
    """
    report = report if report is not None else ValidationReport()
    spec = DatasetSpec(
        key="abc_model",
        name="abc_model",
        purpose="",
        source="",
        license_note="",
        approx_size="",
        index_url="",
        pattern="",
    )
    all_files = [f for f in local_files(spec) if f.suffix == ".gz"]
    if not all_files:
        report.add("warning", "abc_missing", "No ABC Model files found")
        return None

    if cell_types == "all":
        files = all_files
    else:
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        files = [f for f in all_files if any(ct.lower() in f.name.lower() for ct in cell_types)]

    if not files:
        report.add("warning", "abc_missing_cell_type", f"No ABC Model files found for cell types: {cell_types}")
        return None

    frames = []
    loaded_cells = []
    for path in files:
        cell_name = path.stem.replace(".bed", "")
        cols = ["chrom", "start", "end", "target_gene", "abc_score"]
        try:
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=cols, dtype=str)
        except Exception as e:
            report.add("warning", "abc_load_failed", f"Failed to read ABC file {path}: {e}")
            continue
        if df.empty:
            continue
        df["chrom"] = df["chrom"].map(normalize_chrom)
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
        df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
        df["abc_score"] = pd.to_numeric(df["abc_score"], errors="coerce")
        df = df.dropna(subset=["chrom", "start", "end", "target_gene", "abc_score"])
        df = df[df["abc_score"] >= min_score]
        df = df.rename(columns={"target_gene": "gene_id"})
        df["gene_id"] = df["gene_id"].str.split(".").str[0]  # strip version
        df["abc_cell_type"] = cell_name
        frames.append(df)
        loaded_cells.append(cell_name)

    if not frames:
        report.add("warning", "abc_empty", "No valid ABC predictions after filtering")
        return None

    merged = pd.concat(frames, ignore_index=True)
    # For each (chrom, start, end, gene_id), keep the max score across cell types
    merged = merged.sort_values("abc_score", ascending=False).drop_duplicates(
        subset=["chrom", "start", "end", "gene_id"], keep="first"
    )
    report.add("info", "abc_loaded", f"Loaded {len(merged)} ABC predictions from {len(loaded_cells)} cell types (score>={min_score})")
    return AbcPredictions(cell_type="+".join(loaded_cells), predictions=merged.reset_index(drop=True))


def merge_abc_with_candidates(
    candidates: pd.DataFrame,
    abc: AbcPredictions,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Merge ABC predictions with candidate variant-gene pairs.

    For each candidate (variant, gene), we check if any ABC enhancer for that gene
    overlaps the variant position. If so, we assign the max ABC score.
    """
    report = report if report is not None else ValidationReport()
    if candidates.empty or abc is None or abc.predictions.empty:
        candidates["abc_score"] = 0.0
        candidates["abc_missing"] = 1
        return candidates

    out = candidates.copy()
    # ABC enhancers: chrom, start, end, gene_id, abc_score
    # For each gene, collect its enhancers
    enh_by_gene = {g: grp[["chrom", "start", "end", "abc_score"]].to_numpy()
                   for g, grp in abc.predictions.groupby("gene_id")}

    scores = []
    for row in out.itertuples(index=False):
        gene_enh = enh_by_gene.get(row.gene_id)
        if gene_enh is None:
            scores.append(0.0)
            continue
        # Check overlap: variant pos in [enh_start, enh_end] on same chrom
        same_chrom = gene_enh[:, 0] == row.chrom
        if not same_chrom.any():
            scores.append(0.0)
            continue
        enh = gene_enh[same_chrom]
        in_enh = (enh[:, 1] <= row.pos) & (enh[:, 2] >= row.pos)
        if not in_enh.any():
            scores.append(0.0)
        else:
            scores.append(float(enh[in_enh, 3].max()))
    out["abc_score"] = scores
    out["abc_missing"] = (np.array(scores) == 0.0).astype(int)
    report.add("info", "abc_merged", f"ABC scores added: {(np.array(scores) > 0).sum()} candidates with ABC enhancer overlap")
    return out
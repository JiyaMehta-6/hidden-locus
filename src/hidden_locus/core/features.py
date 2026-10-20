"""Candidate variant-to-gene generation and feature extraction (stages C-E).

Produces a documented feature table in which every row is a candidate
(variant, gene) pair at one disease locus.  Only features backed by the
loaded public datasets are emitted; absent evidence is represented by
explicit missing-evidence indicators rather than fabricated values.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .abc_model import merge_abc_with_candidates
from .gut_atlas import merge_gut_atlas_with_candidates, load_gut_atlas_all
from .intervals import IntervalIndex, point_distances, point_overlaps
from .validation import ValidationReport, harmonize_allele_pair, normalize_chrom

CCRE_CLASSES = ("PLS", "pELS", "dELS", "CTCF-only", "DNase-H3K4me3")
CONSEQUENCE_LEVELS = ("exon", "intron", "promoter", "proximal", "distal")
PROMOTER_UP_BP = 2_000
PROMOTER_DOWN_BP = 500
PROXIMAL_BP = 10_000
# Stand-in for "no cCRE anywhere on this contig" (point_distances -> inf).
# Must stay finite: build_feature_matrix maps non-finite values to 0, which
# would otherwise read as "inside a cCRE".
CCRE_DIST_UNOBSERVED = 10_000_000.0

_CHUNK_ROWS = 4_000_000

_ATTR_RE = {
    "gene_id": re.compile(r'gene_id "([^"]+)"'),
    "gene_name": re.compile(r'gene_name "([^"]+)"'),
    "gene_type": re.compile(r'(?:gene_type|gene_biotype) "([^"]+)"'),
}

# GTEx: 19_226776_C_T_b38 ; eQTL Catalogue: chr19_226776_C_T
_VARIANT_ID_RE = re.compile(r"^(?:chr)?([^_]+)_(\d+)_([^_]+)_([^_]+?)(?:_b38)?$")


def strip_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def parse_variant_ids(series: pd.Series) -> pd.DataFrame:
    """Parse variant identifiers into chrom/pos/ref/alt columns."""
    parsed = series.astype(str).str.extract(_VARIANT_ID_RE)
    parsed.columns = ["chrom", "pos", "ref", "alt"]
    parsed["chrom"] = parsed["chrom"].map(normalize_chrom)
    parsed["pos"] = pd.to_numeric(parsed["pos"], errors="coerce")
    parsed["ref"] = parsed["ref"].str.upper()
    parsed["alt"] = parsed["alt"].str.upper()
    return parsed


@dataclass
class GeneAnnotation:
    """Gene models parsed from a GENCODE GTF."""

    genes: pd.DataFrame  # gene_id, gene_name, chrom, start, end, strand, biotype, tss
    exons_by_gene: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load_gtf(
        cls,
        path: Path,
        biotypes: "list[str] | None" = None,
        report: ValidationReport | None = None,
    ) -> "GeneAnnotation":
        report = report if report is not None else ValidationReport()
        names = [
            "seqname", "source", "feature", "start", "end", "score",
            "strand", "frame", "attribute",
        ]
        raw = pd.read_csv(
            path, sep="\t", comment="#", header=None, names=names,
            usecols=["seqname", "feature", "start", "end", "strand", "attribute"],
            dtype=str, low_memory=False,
        )
        raw = raw[~raw["seqname"].str.contains("_", na=False)].copy()
        raw["chrom"] = raw["seqname"].map(normalize_chrom)
        raw = raw[raw["chrom"].notna()]

        gene_rows = raw[raw["feature"] == "gene"]
        records = []
        for row in gene_rows.itertuples(index=False):
            attrs = row.attribute
            gene_id = _ATTR_RE["gene_id"].search(attrs)
            if not gene_id:
                continue
            gid = strip_version(gene_id.group(1))
            gene_name = _ATTR_RE["gene_name"].search(attrs)
            gene_type = _ATTR_RE["gene_type"].search(attrs)
            start, end = int(row.start), int(row.end)
            tss = start if row.strand == "+" else end
            records.append(
                {
                    "gene_id": gid,
                    "gene_name": gene_name.group(1) if gene_name else gid,
                    "chrom": row.chrom,
                    "start": start,
                    "end": end,
                    "strand": row.strand,
                    "biotype": gene_type.group(1) if gene_type else "unknown",
                    "tss": tss,
                }
            )
        genes = pd.DataFrame.from_records(records)
        if genes.empty:
            raise ValueError(f"No gene records parsed from {path}")
        if biotypes:
            before = len(genes)
            genes = genes[genes["biotype"].isin(biotypes)].copy()
            report.add(
                "info", "biotype_filter",
                f"Kept {len(genes)} of {before} genes with biotypes {biotypes}.",
            )
        genes = genes.drop_duplicates(subset=["gene_id"], keep="first").reset_index(drop=True)

        exons_by_gene: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        exon_rows = raw[raw["feature"] == "exon"]
        if not exon_rows.empty:
            exon_rows = exon_rows[exon_rows["attribute"].str.contains("gene_id", na=False)].copy()

            def _exon_gene(attrs: str):
                match = _ATTR_RE["gene_id"].search(attrs)
                return strip_version(match.group(1)) if match else None

            exon_rows["gene_id"] = exon_rows["attribute"].map(_exon_gene)
            exon_rows = exon_rows[exon_rows["gene_id"].isin(set(genes["gene_id"]))]
            for gid, group in exon_rows.groupby("gene_id"):
                starts = group["start"].to_numpy(dtype=np.int64)
                ends = group["end"].to_numpy(dtype=np.int64)
                order = np.argsort(starts)
                exons_by_gene[gid] = (starts[order], ends[order])
            report.add(
                "info", "exons_loaded",
                f"Exon structure available for {len(exons_by_gene)} genes.",
            )
        else:
            report.add(
                "warning", "no_exons",
                "Gene annotation contains no exon records; intron/exon "
                "consequences collapse to a gene-body label.",
            )
        report.add("info", "genes_loaded", f"Loaded {len(genes)} genes from {Path(path).name}.")
        return cls(genes=genes, exons_by_gene=exons_by_gene)


@dataclass
class CcreAnnotation:
    """Candidate cis-regulatory elements from ENCODE SCREEN."""

    index: IntervalIndex
    classes: np.ndarray
    accessions: np.ndarray

    @classmethod
    def load_bed(cls, path: Path, report: ValidationReport | None = None) -> "CcreAnnotation":
        report = report if report is not None else ValidationReport()
        frame = pd.read_csv(
            path, sep="\t", header=None, dtype=str,
            names=["chrom", "start", "end", "accession", "accession2", "group"],
            low_memory=False,
        )
        frame["chrom"] = frame["chrom"].map(normalize_chrom)
        frame = frame[frame["chrom"].notna()].copy()
        frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
        frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
        frame = frame[
            frame["start"].notna() & frame["end"].notna() & (frame["end"] > frame["start"])
        ]
        classes = (
            frame["group"].fillna("").str.split(",").str[0].fillna("unknown")
            .to_numpy(dtype=object)
        )
        index = IntervalIndex(
            chrom=frame["chrom"].to_numpy(dtype=object),
            start=frame["start"].to_numpy(dtype=np.int64),
            end=frame["end"].to_numpy(dtype=np.int64),
        )
        report.add("info", "ccre_loaded", f"Loaded {len(frame)} regulatory elements.")
        return cls(
            index=index,
            classes=classes,
            accessions=frame["accession"].to_numpy(dtype=object),
        )


@dataclass
class FeatureContext:
    """Everything needed to annotate candidate pairs with evidence."""

    genes: GeneAnnotation
    ccre: "CcreAnnotation | None"
    gtex: dict[str, pd.DataFrame] = field(default_factory=dict)  # tissue -> pairs
    gtex_egenes: dict[str, pd.DataFrame] = field(default_factory=dict)
    pip: "pd.Series | None" = None  # variant_uid -> PIP
    abc: "AbcPredictions | None" = None
    gut_atlas: dict = field(default_factory=dict)  # cell_type -> GutAtlasEqtl
    gtex_loaded: bool = False
    regulatory_loaded: bool = False
    coloc: "pd.DataFrame | None" = None  # variant_uid,gene_id -> coloc H4/CLPP
    coloc_loaded: bool = False

    def variant_alleles(self) -> dict[str, tuple[str, str]]:
        """Map variant_uid -> (ref, alt) using the first tissue that has it."""
        alleles: dict[str, tuple[str, str]] = {}
        for pairs in self.gtex.values():
            if pairs is None or pairs.empty:
                continue
            for uid, ref, alt in zip(
                pairs["variant_uid"], pairs["ref"], pairs["alt"]
            ):
                alleles.setdefault(uid, (ref, alt))
        return alleles


def read_eqtl_pairs(
    path: Path,
    variant_uids: "set[str] | None" = None,
    p_max: float | None = None,
    report: ValidationReport | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """Stream a large eQTL pairs file, keeping candidate variants only.

    ``source='gtex'`` expects ``variant_id``/``pval_nominal`` columns;
    ``source='catalogue'`` expects ``variant``/``pvalue``/``gene_id``;
    ``'auto'`` picks based on the header.
    """
    report = report if report is not None else ValidationReport()
    wanted_uids = variant_uids
    pieces: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(path, sep="\t", chunksize=_CHUNK_ROWS, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"eQTL pairs file is empty or corrupt: {path}") from exc
    for chunk in reader:
        if source == "auto":
            source = "gtex" if "variant_id" in chunk.columns else "catalogue"
        if source == "gtex":
            ids = chunk["variant_id"]
            pcol = "pval_nominal"
        else:
            ids = chunk["variant"]
            pcol = "pvalue"
        parsed = parse_variant_ids(ids)
        keep = parsed["chrom"].notna() & parsed["pos"].notna()
        if wanted_uids is not None:
            uid_safe = np.where(
                keep,
                parsed["chrom"].astype(str) + ":"
                + parsed["pos"].astype("Int64").astype(str),
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
        sub["variant_uid"] = (
            sub["chrom"].astype(str) + ":" + sub["pos"].astype(str)
        )
        sub["gene_id"] = sub["gene_id"].map(strip_version)
        sub[pcol] = pd.to_numeric(sub[pcol], errors="coerce")
        if p_max is not None:
            sub = sub[sub[pcol] <= p_max]
        if not sub.empty:
            pieces.append(sub)
    if not pieces:
        out = pd.DataFrame(columns=["variant_uid", "gene_id"])
        report.add("info", "eqtl_scan", f"{path.name}: no rows matched the candidate variants.")
        return out
    out = pd.concat(pieces, ignore_index=True)
    pcol_name = "pval_nominal" if source == "gtex" else "pvalue"
    out = out.sort_values(pcol_name).drop_duplicates(
        subset=["variant_uid", "gene_id"], keep="first"
    )
    if "slope" in out.columns:
        out["slope"] = pd.to_numeric(out["slope"], errors="coerce")
    report.add(
        "info", "eqtl_scan",
        f"{path.name}: {len(out)} variant-gene pairs retained for "
        f"{out['variant_uid'].nunique()} candidate variants.",
    )
    return out.reset_index(drop=True)


def load_gtex_tissue(
    pairs_path: Path,
    egenes_path: Path,
    variant_uids: "set[str] | None" = None,
    report: ValidationReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GTEx significant variant-gene pairs plus eGene-level q-values."""
    report = report if report is not None else ValidationReport()
    pairs = read_eqtl_pairs(pairs_path, variant_uids, source="gtex", report=report)
    egenes = pd.read_csv(egenes_path, sep="\t", low_memory=False)
    if "gene_id" in egenes.columns:
        egenes = egenes.copy()
        egenes["gene_id"] = egenes["gene_id"].map(strip_version)
        if "qval" in egenes.columns:
            egenes["qval"] = pd.to_numeric(egenes["qval"], errors="coerce")
        # One row per gene (best qval); duplicates would multiply candidate
        # rows on the gene-level merge in build_feature_table.
        sort_col = "qval" if "qval" in egenes.columns else "gene_id"
        egenes = (
            egenes.dropna(subset=["gene_id"])
            .sort_values(sort_col, kind="mergesort")
            .drop_duplicates(subset=["gene_id"], keep="first")
            .reset_index(drop=True)
        )
    return pairs, egenes


def load_label_eqtl(
    pairs_path: Path,
    permuted_path: Path,
    variant_uids: "set[str] | None",
    p_threshold: float,
    report: ValidationReport | None = None,
    settings_cache_key: str = "",
) -> tuple[pd.DataFrame, set[str]]:
    """Load positive-label evidence from an independent eQTL study.

    A candidate pair is positive when the variant is associated with the
    gene at nominal ``p <= p_threshold`` in the independent study **and**
    the gene is an eGene there (permutation-based ``p_beta <= 0.05``).
    """
    report = report if report is not None else ValidationReport()
    # The cached pairs are filtered to the candidate-variant set, so the
    # cache key must encode that set; otherwise a settings change (window,
    # trait pattern, GWAS release) would silently reuse labels computed for
    # a different candidate universe.
    uid_key = ""
    if variant_uids is not None:
        digest = hashlib.sha256(
            "\n".join(sorted(variant_uids)).encode("utf-8")
        ).hexdigest()[:12]
        uid_key = f"_{digest}"
    settings_key = f"_{settings_cache_key}" if settings_cache_key else ""
    cache_path = (
        config.CACHE_DIR
        / f"label_pairs_{pairs_path.name}_{p_threshold:g}{uid_key}{settings_key}.parquet"
    )
    pairs: pd.DataFrame | None = None
    if cache_path.exists() and cache_path.stat().st_mtime >= pairs_path.stat().st_mtime:
        try:
            pairs = pd.read_parquet(cache_path)
            report.add(
                "info", "labels_cache",
                f"Loaded {len(pairs)} label pairs from cache {cache_path.name}.",
            )
        except Exception:
            pairs = None
    if pairs is None:
        pairs = read_eqtl_pairs(
            pairs_path, variant_uids, p_max=p_threshold, source="catalogue",
            report=report,
        )
        try:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pairs.to_parquet(cache_path, index=False)
        except Exception:
            pass
    permuted = pd.read_csv(permuted_path, sep="\t", low_memory=False)
    egenes: set[str] = set()
    if not permuted.empty:
        permuted = permuted.copy()
        gene_col = next(
            (
                c
                for c in (
                    "gene_id", "molecular_trait_id", "molecular_trait_object_id"
                )
                if c in permuted.columns
            ),
            None,
        )
        if gene_col is not None:
            permuted["gene_id"] = permuted[gene_col].map(strip_version)
            if "p_beta" in permuted.columns:
                p_beta = pd.to_numeric(permuted["p_beta"], errors="coerce")
                egenes = set(permuted.loc[p_beta <= 0.05, "gene_id"])
    if not pairs.empty and egenes:
        n_before = len(pairs)
        pairs = pairs[pairs["gene_id"].isin(egenes)].copy()
        report.add(
            "info", "label_egenes_filter",
            f"Kept {len(pairs)} of {n_before} pairs whose gene is an eGene "
            "in the label study.",
        )
    elif not pairs.empty and not egenes:
        report.add(
            "warning", "label_egenes_filter_missing",
            "Permutation file yielded no eGenes (missing gene or p_beta "
            "column); the eGene filter was NOT applied and label evidence "
            "may be weaker than designed.",
        )
    report.add(
        "info", "labels_loaded",
        f"{pairs_path.name}: {len(pairs)} positive-evidence pairs with "
        f"p <= {p_threshold:g}; {len(egenes)} eGenes in the label study.",
    )
    return pairs.reset_index(drop=True), egenes


def generate_candidates(
    variants: pd.DataFrame,
    genes: GeneAnnotation,
    window_bp: int = config.DEFAULT_WINDOW_BP,
) -> pd.DataFrame:
    """Generate candidate (variant, gene) pairs within +/- window of the TSS."""
    gene_df = genes.genes
    rows = []
    by_chrom_genes = {c: g for c, g in gene_df.groupby("chrom", sort=False)}
    for chrom, group in variants.groupby("chrom", sort=False):
        gene_group = by_chrom_genes.get(chrom)
        if gene_group is None or gene_group.empty:
            continue
        tss = gene_group["tss"].to_numpy(dtype=np.int64)
        order = np.argsort(tss)
        tss_sorted = tss[order]
        gene_rows = gene_group.iloc[order]
        pos = group["pos"].to_numpy(dtype=np.int64)
        left = np.searchsorted(tss_sorted, pos - window_bp, side="left")
        right = np.searchsorted(tss_sorted, pos + window_bp, side="right")
        for i, (_, var_row) in enumerate(group.iterrows()):
            for j in range(int(left[i]), int(right[i])):
                gene = gene_rows.iloc[j]
                rows.append(
                    {
                        "variant_uid": var_row["variant_uid"],
                        "chrom": chrom,
                        "pos": int(var_row["pos"]),
                        "rsid": var_row.get("rsid"),
                        "risk_allele": var_row.get("risk_allele"),
                        "pvalue": var_row.get("pvalue"),
                        "trait": var_row.get("trait"),
                        "pubmed": var_row.get("pubmed"),
                        "study": var_row.get("study"),
                        "locus_id": var_row.get("locus_id"),
                        "or_beta": var_row.get("or_beta"),
                        "gene_id": gene["gene_id"],
                        "gene_name": gene["gene_name"],
                        "gene_biotype": gene["biotype"],
                        "gene_strand": gene["strand"],
                        "gene_tss": int(gene["tss"]),
                        "gene_start": int(gene["start"]),
                        "gene_end": int(gene["end"]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["dist_tss"] = (out["pos"].to_numpy() - out["gene_tss"].to_numpy()) * np.where(
        out["gene_strand"].to_numpy() == "-", -1, 1
    )
    return out


def _consequence(row, exons_by_gene: dict) -> str:
    pos = int(row.pos)
    start, end = int(row.gene_start), int(row.gene_end)
    strand = row.gene_strand
    tss = int(row.gene_tss)
    if strand == "-":
        promoter = (tss - PROMOTER_DOWN_BP) <= pos <= (tss + PROMOTER_UP_BP)
    else:
        promoter = (tss - PROMOTER_UP_BP) <= pos <= (tss + PROMOTER_DOWN_BP)
    if promoter:
        return "promoter"
    exons = exons_by_gene.get(row.gene_id)
    if exons is not None:
        starts, ends = exons
        i = int(np.searchsorted(starts, pos, side="right") - 1)
        if i >= 0 and pos <= int(ends[i]):
            return "exon"
    if start <= pos <= end:
        return "intron" if exons is not None else "gene_body"
    if abs(pos - tss) <= PROXIMAL_BP:
        return "proximal"
    return "distal"


def build_feature_table(
    candidates: pd.DataFrame,
    context: FeatureContext,
    tissues: "list[str] | None" = None,
) -> pd.DataFrame:
    """Attach evidence features and provenance columns to candidate pairs."""
    if candidates.empty:
        return candidates.copy()
    tissues = tissues if tissues is not None else list(context.gtex.keys())
    df = candidates.copy().reset_index(drop=True)

    # --- distance / structural features -----------------------------------
    df["abs_log10_dist_tss"] = np.log10(1.0 + np.abs(df["dist_tss"].to_numpy(dtype=np.float64)))
    body_dist = np.where(
        (df["pos"] >= df["gene_start"]) & (df["pos"] <= df["gene_end"]),
        0,
        np.minimum(
            np.abs(df["pos"] - df["gene_start"]),
            np.abs(df["pos"] - df["gene_end"]),
        ),
    )
    df["log10_dist_body"] = np.log10(1.0 + body_dist.astype(np.float64))
    df["is_protein_coding"] = (df["gene_biotype"] == "protein_coding").astype(int)
    df["consequence"] = [
        _consequence(row, context.genes.exons_by_gene) for row in df.itertuples(index=False)
    ]

    # --- association strength ---------------------------------------------
    pvals = pd.to_numeric(df["pvalue"], errors="coerce")
    if pvals.notna().any():
        df["gwas_log10_p"] = -np.log10(pvals.clip(lower=1e-300))
        df["gwas_p_missing"] = df["gwas_log10_p"].isna().astype(int)
        df["gwas_log10_p"] = df["gwas_log10_p"].fillna(0.0)
    else:
        df["gwas_log10_p"] = 0.0
        df["gwas_p_missing"] = 1

    # --- fine-mapping PIP ---------------------------------------------------
    if context.pip is not None and len(context.pip):
        df["pip"] = df["variant_uid"].map(context.pip).astype(float)
        df["pip_missing"] = df["pip"].isna().astype(int)
        df["pip"] = df["pip"].fillna(0.0)
    else:
        df["pip"] = 0.0
        df["pip_missing"] = 1

    # --- regulatory overlap (variant level) ---------------------------------
    if context.regulatory_loaded and context.ccre is not None and len(context.ccre.index):
        uniq = df[["chrom", "pos"]].drop_duplicates().reset_index(drop=True)
        q_idx, t_idx = point_overlaps(
            uniq["chrom"].to_numpy(dtype=object),
            uniq["pos"].to_numpy(dtype=np.int64),
            context.ccre.index,
        )
        ccre_class = pd.Series("none", index=uniq.index, dtype=object)
        ccre_acc = pd.Series("", index=uniq.index, dtype=object)
        for qi, cls_, acc in zip(q_idx, context.ccre.classes[t_idx], context.ccre.accessions[t_idx]):
            if ccre_class.iloc[int(qi)] == "none":
                ccre_class.iloc[int(qi)] = cls_
                ccre_acc.iloc[int(qi)] = acc
        uniq["ccre_class"] = ccre_class.to_numpy()
        uniq["ccre_accession"] = ccre_acc.to_numpy()
        uniq["ccre_dist"] = point_distances(
            uniq["chrom"].to_numpy(dtype=object),
            uniq["pos"].to_numpy(dtype=np.int64),
            context.ccre.index,
        )
        # Contigs absent from the cCRE index yield inf; keep a finite
        # sentinel so downstream nan_to_num cannot read it as distance 0.
        uniq["ccre_dist"] = np.where(
            np.isfinite(uniq["ccre_dist"].to_numpy(dtype=np.float64)),
            uniq["ccre_dist"],
            CCRE_DIST_UNOBSERVED,
        )
        df = df.merge(uniq, on=["chrom", "pos"], how="left")
        df["ccre_class"] = df["ccre_class"].fillna("none")
        df["ccre_accession"] = df["ccre_accession"].fillna("")
    else:
        df["ccre_class"] = "none"
        df["ccre_accession"] = ""
        df["ccre_dist"] = CCRE_DIST_UNOBSERVED
    df["regulatory_missing"] = int(not context.regulatory_loaded)

    # --- GTEx expression-QTL evidence ---------------------------------------
    if context.gtex_loaded:
        df["gtex_missing"] = 0
        for tissue in tissues:
            pairs = context.gtex.get(tissue)
            suffix = _tissue_suffix(tissue)
            if pairs is None or pairs.empty:
                df[f"gtex_log10p_{suffix}"] = 0.0
                df[f"gtex_slope_{suffix}"] = 0.0
                df[f"gtex_pair_{suffix}"] = 0
                df[f"gtex_egenes_{suffix}"] = 0.0
                continue
            sub = pairs[["variant_uid", "gene_id", "pval_nominal", "slope"]].rename(
                columns={"pval_nominal": f"p_{suffix}", "slope": f"slope_{suffix}"}
            )
            df = df.merge(sub, on=["variant_uid", "gene_id"], how="left")
            pair_found = df[f"p_{suffix}"].notna()
            df[f"gtex_pair_{suffix}"] = pair_found.astype(int)
            df[f"gtex_log10p_{suffix}"] = (
                -np.log10(df[f"p_{suffix}"].clip(lower=1e-300)).fillna(0.0)
            )
            df[f"gtex_slope_{suffix}"] = df[f"slope_{suffix}"].fillna(0.0)
            df = df.drop(columns=[f"p_{suffix}", f"slope_{suffix}"])
            egenes = context.gtex_egenes.get(tissue)
            if (
                egenes is not None and not egenes.empty
                and "gene_id" in egenes.columns and "qval" in egenes.columns
            ):
                df = df.merge(
                    egenes[["gene_id", "qval"]].rename(columns={"qval": f"q_{suffix}"}),
                    on="gene_id", how="left",
                )
                df[f"gtex_egenes_{suffix}"] = (
                    -np.log10(df[f"q_{suffix}"].clip(lower=1e-300))
                ).fillna(0.0)
                df = df.drop(columns=[f"q_{suffix}"])
            else:
                df[f"gtex_egenes_{suffix}"] = 0.0
        support_cols = [c for c in df.columns if c.startswith("gtex_pair_")]
        df["gtex_n_tissues"] = df[support_cols].sum(axis=1).astype(int) if support_cols else 0
        slope_cols = [c for c in df.columns if c.startswith("gtex_slope_")]
        if len(slope_cols) >= 2 and support_cols:
            slopes = df[slope_cols].to_numpy(dtype=float)
            support = df[support_cols].to_numpy(dtype=float)
            pos = ((slopes > 0) & (support > 0)).any(axis=1)
            neg = ((slopes < 0) & (support > 0)).any(axis=1)
            df["eqtl_direction_conflict"] = (pos & neg).astype(int)
        else:
            df["eqtl_direction_conflict"] = 0

        # --- GWAS-eQTL slope concordance ---------------------------------------
        # Compare GWAS effect direction (or_beta) with eQTL slope direction.
        # Concordant: risk allele increases disease AND increases expression (or_beta > 0, slope > 0)
        # or risk allele decreases disease AND decreases expression (or_beta < 0, slope < 0).
        if "or_beta" in df.columns:
            gwas_dir = np.sign(pd.to_numeric(df["or_beta"], errors="coerce").fillna(0))
            concordant = np.zeros(len(df), dtype=bool)
            for sc in slope_cols:
                slope_dir = np.sign(df[sc].fillna(0))
                concordant |= (gwas_dir == slope_dir) & (df[sc].notna()) & (gwas_dir != 0)
            df["gwas_eqtl_concordant"] = concordant.astype(int)
        else:
            df["gwas_eqtl_concordant"] = 0
    else:
        df["gtex_missing"] = 1
        df["gtex_n_tissues"] = 0
        df["eqtl_direction_conflict"] = 0
        for tissue in tissues:
            suffix = _tissue_suffix(tissue)
            df[f"gtex_log10p_{suffix}"] = 0.0
            df[f"gtex_slope_{suffix}"] = 0.0
            df[f"gtex_pair_{suffix}"] = 0
            df[f"gtex_egenes_{suffix}"] = 0.0

    # --- Colocalization H4/CLPP evidence (from OT L2G features) ---------------
    if context.coloc_loaded and context.coloc is not None and not context.coloc.empty:
        df["coloc_missing"] = 0
        for col in ("coloc_eqtl_h4", "coloc_eqtl_clpp", "coloc_sqtl_h4", "coloc_pqtl_h4"):
            if col in context.coloc.columns:
                sub = context.coloc[["variant_uid", "gene_id", col]].rename(
                    columns={col: f"_merge_{col}"}
                )
                df = df.merge(sub, on=["variant_uid", "gene_id"], how="left")
                df[col] = df[f"_merge_{col}"].fillna(0.0)
                df = df.drop(columns=[f"_merge_{col}"])
            else:
                df[col] = 0.0
    else:
        df["coloc_missing"] = 1
        df["coloc_eqtl_h4"] = 0.0
        df["coloc_eqtl_clpp"] = 0.0
        df["coloc_sqtl_h4"] = 0.0
        df["coloc_pqtl_h4"] = 0.0

    # --- ABC Model enhancer-gene links ----------------------------------------
    if context.abc is not None and context.abc.predictions is not None and not context.abc.predictions.empty:
        df = merge_abc_with_candidates(df, context.abc, report=None)
    else:
        df["abc_score"] = 0.0
        df["abc_missing"] = 1

    # --- Gut Cell Atlas cell-type-specific eQTL --------------------------------
    if context.gut_atlas:
        df = merge_gut_atlas_with_candidates(df, context.gut_atlas, report=None)
    else:
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

    # --- allele verification against GTEx reference alleles ------------------
    alleles = context.variant_alleles()
    df["allele_match"] = [
        harmonize_allele_pair(row.risk_allele, *alleles.get(row.variant_uid, (None, None)))
        if row.variant_uid in alleles
        else "unverified"
        for row in df.itertuples(index=False)
    ]

    # --- GWAS-eQTL slope concordance ---------------------------------------
    # Compare GWAS effect direction (or_beta) with eQTL slope direction (GTEx + Gut Atlas).
    # Concordant: risk allele increases disease AND increases expression (or_beta > 0, slope > 0)
    # or risk allele decreases disease AND decreases expression (or_beta < 0, slope < 0).
    if "or_beta" in df.columns:
        gwas_dir = np.sign(pd.to_numeric(df["or_beta"], errors="coerce").fillna(0))
        slope_cols = [c for c in df.columns if c.startswith("gtex_slope_") or c.startswith("gut_slope_")]
        concordant = np.zeros(len(df), dtype=bool)
        for sc in slope_cols:
            slope_dir = np.sign(df[sc].fillna(0))
            concordant |= (gwas_dir == slope_dir) & (df[sc].notna()) & (gwas_dir != 0)
        df["gwas_eqtl_concordant"] = concordant.astype(int)
    else:
        df["gwas_eqtl_concordant"] = 0

    return df


def _tissue_suffix(tissue: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", tissue).strip("_").lower()


# Explicit feature list; identifier / provenance columns are never included.
STRUCTURAL_FEATURES = (
    "abs_log10_dist_tss",
    "log10_dist_body",
    "is_protein_coding",
    "gwas_log10_p",
    "gwas_p_missing",
    "pip",
    "pip_missing",
    "regulatory_missing",
    "gtex_missing",
    "gtex_n_tissues",
    "eqtl_direction_conflict",
)
COLOC_FEATURES = (
    "coloc_eqtl_h4",
    "coloc_eqtl_clpp",
    "coloc_sqtl_h4",
    "coloc_pqtl_h4",
    "coloc_missing",
)
CONSEQUENCE_FEATURES = tuple(f"conseq_{level}" for level in CONSEQUENCE_LEVELS)
CCRE_FEATURES = tuple(f"ccre_{cls_.replace('-', '_')}" for cls_ in CCRE_CLASSES) + ("ccre_dist",)

ID_COLUMNS = {
    "variant_uid", "chrom", "pos", "rsid", "gene_id", "gene_name", "locus_id",
    "trait", "pubmed", "study", "risk_allele", "pvalue", "gene_biotype",
    "gene_strand", "gene_tss", "gene_start", "gene_end", "ccre_accession",
    "ccre_class", "consequence", "allele_match", "dist_tss", "label",
    "label_pvalue", "label_study",
}


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and return the model-ready table."""
    out = df.copy()
    conseq = pd.get_dummies(out["consequence"], prefix="conseq", dtype=int)
    for level in CONSEQUENCE_FEATURES:
        if level not in conseq.columns:
            conseq[level] = 0
    out = pd.concat([out, conseq[list(CONSEQUENCE_FEATURES)]], axis=1)
    ccre_cols = [f"ccre_{c.replace('-', '_')}" for c in CCRE_CLASSES]
    # get_dummies keeps hyphens in column names ("ccre_CTCF-only"), while
    # the model feature list uses underscores; without this the real dummies
    # were dropped and replaced by all-zero placeholders.
    classes = out["ccre_class"].astype(str).str.replace("-", "_", regex=False)
    ccre = pd.get_dummies(classes, prefix="ccre", dtype=int)
    for col in ccre_cols:
        if col not in ccre.columns:
            ccre[col] = 0
    out = pd.concat([out, ccre[ccre_cols]], axis=1)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns present in *df* (leakage-safe order)."""
    wanted = (
        list(STRUCTURAL_FEATURES) + list(COLOC_FEATURES)
        + list(CONSEQUENCE_FEATURES) + list(CCRE_FEATURES)
    )
    return [c for c in wanted if c in df.columns]


def coloc_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in COLOC_FEATURES if c in df.columns]


def gtex_feature_columns(df: pd.DataFrame, tissues: "list[str]") -> list[str]:
    suffixes = [_tissue_suffix(t) for t in tissues]
    cols = []
    for suffix in suffixes:
        for prefix in ("gtex_log10p_", "gtex_slope_", "gtex_pair_", "gtex_egenes_"):
            col = f"{prefix}{suffix}"
            if col in df.columns:
                cols.append(col)
    for col in ("gtex_n_tissues", "eqtl_direction_conflict"):
        if col in df.columns:
            cols.append(col)
    return cols


def regulatory_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in CCRE_FEATURES if c in df.columns]


def finemapping_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ("pip", "pip_missing") if c in df.columns]


def assert_no_leakage(df: pd.DataFrame, columns: "list[str] | None" = None) -> None:
    """Raise if identifier columns leak into the feature list."""
    columns = columns if columns is not None else feature_columns(df)
    bad = [c for c in columns if c in ID_COLUMNS]
    if bad:
        raise ValueError(f"Identifier columns leaked into features: {bad}")


def attach_labels(
    df: pd.DataFrame,
    label_pairs: pd.DataFrame,
    label_study: str,
    p_threshold: float,
) -> pd.DataFrame:
    """Attach independent-study labels (1) and provenance to candidates.

    Candidate pairs absent from the label study are marked ``label=0`` but
    must be treated as *unlabelled*, not confirmed negatives.
    """
    out = df.copy()
    if label_pairs is None or label_pairs.empty:
        out["label"] = 0
        out["label_pvalue"] = np.nan
        out["label_study"] = label_study
        return out
    sub = label_pairs[["variant_uid", "gene_id", "pvalue"]].rename(
        columns={"pvalue": "label_pvalue"}
    )
    # Belt-and-braces: callers normally pre-filter, but never let a
    # higher-p row become a positive by accident.
    sub = sub[pd.to_numeric(sub["label_pvalue"], errors="coerce") <= p_threshold]
    sub = sub.drop_duplicates(subset=["variant_uid", "gene_id"])
    out = out.merge(sub, on=["variant_uid", "gene_id"], how="left")
    out["label"] = out["label_pvalue"].notna().astype(int)
    out["label_study"] = label_study
    return out

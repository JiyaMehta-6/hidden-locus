"""Disease-locus selection and variant harmonization (pipeline stages A-B).

Loads GWAS Catalog association tables, filters them to the research scope,
harmonizes coordinates to GRCh38, deduplicates index variants across studies
and clusters them into loci.  Association evidence is never treated as
causal evidence.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import (
    ValidationReport,
    normalize_chrom,
    parse_positions,
    parse_strongest_risk_allele,
)

# Column aliases looked up case-insensitively in GWAS Catalog releases.
_TRAIT_COLS = ("DISEASE/TRAIT", "MAPPED_TRAIT")
_CHROM_COLS = ("CHR_ID",)
_POS_COLS = ("CHR_POS",)
_P_COLS = ("P-VALUE", "PVALUE")
_RISK_COLS = ("STRONGEST SNP-RISK ALLELE", "RISK ALLELE")
_SNP_COLS = ("SNPS",)
_PUBMED_COLS = ("PUBMEDID",)
_OR_BETA_COLS = ("OR or BETA",)
_MAPPED_GENE_COLS = ("MAPPED_GENE",)
_STUDY_COLS = ("STUDY",)
_SAMPLE_COLS = ("INITIAL SAMPLE SIZE",)


def _find_col(columns: list[str], candidates: tuple[str, ...]) -> "str | None":
    lookup = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in lookup:
            return lookup[cand.upper()]
    return None


def gwas_zip_member(zip_path: Path) -> str:
    """Return the TSV member name inside a GWAS Catalog associations zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n for n in zf.namelist()
            if n.lower().endswith(".tsv") and not n.endswith("/")
        ]
    if not names:
        raise ValueError(f"No TSV member found in {zip_path.name}")
    # Prefer the ontology-annotated associations file.
    names.sort(key=lambda n: ("ontology" not in n.lower(), len(n)))
    return names[0]


def load_gwas_associations(zip_path: Path, report: ValidationReport) -> pd.DataFrame:
    """Load GWAS Catalog associations from the downloaded release zip."""
    zip_path = Path(zip_path)
    member = gwas_zip_member(zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as handle:
        header = handle.readline().decode("utf-8", errors="replace").rstrip("\n").split("\t")

    wanted = {}
    mapping = {
        "trait": _TRAIT_COLS,
        "chrom_raw": _CHROM_COLS,
        "pos_raw": _POS_COLS,
        "pvalue": _P_COLS,
        "risk_raw": _RISK_COLS,
        "snp_raw": _SNP_COLS,
        "pubmed": _PUBMED_COLS,
        "or_beta": _OR_BETA_COLS,
        "mapped_gene": _MAPPED_GENE_COLS,
        "study": _STUDY_COLS,
        "sample": _SAMPLE_COLS,
    }
    for name, candidates in mapping.items():
        col = _find_col(header, candidates)
        if col is not None:
            wanted[name] = col
    for required in ("trait", "chrom_raw", "pos_raw", "pvalue"):
        if required not in wanted:
            raise ValueError(
                f"GWAS Catalog file is missing the column for '{required}'. "
                f"Found columns: {', '.join(header[:20])}..."
            )

    usecols = sorted(set(wanted.values()))
    frame = pd.read_csv(
        _MemberReader(zip_path, member),
        sep="\t",
        usecols=usecols,
        dtype=str,
        low_memory=False,
    )
    rename = {col: name for name, col in wanted.items()}
    frame = frame.rename(columns=rename)
    report.add("info", "gwas_loaded", f"Loaded {len(frame)} association rows from {zip_path.name}.")
    return frame


class _MemberReader:
    """File-like wrapper exposing a zip member to pandas."""

    def __init__(self, zip_path: Path, member: str) -> None:
        self._zf = zipfile.ZipFile(zip_path)
        self._handle = self._zf.open(member)

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def readline(self) -> bytes:
        return self._handle.readline()

    def close(self) -> None:
        self._handle.close()
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def flush(self) -> None:
        return None


def select_index_variants(
    associations: pd.DataFrame,
    trait_pattern: str,
    p_threshold: float = 5e-8,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Select disease-associated index variants for the research scope.

    Keeps associations whose trait matches *trait_pattern* and whose p-value
    is at most *p_threshold*, harmonizes chromosome names, and parses the
    strongest risk allele.  Rows are association reports, not causal claims.
    """
    report = report if report is not None else ValidationReport()
    pattern = re.compile(trait_pattern)

    trait_col = "trait" if "trait" in associations.columns else None
    if trait_col is None:
        raise ValueError("Association table has no trait column")
    keep_traits = associations[trait_col].fillna("").str.contains(pattern, regex=True)
    filtered = associations[keep_traits].copy()
    if filtered.empty:
        report.add(
            "error", "no_trait_matches",
            f"No associations matched the trait pattern {trait_pattern!r}.",
        )
        return pd.DataFrame()
    report.add(
        "info", "trait_filtered",
        f"{len(filtered)} associations matched trait pattern {trait_pattern!r}.",
    )

    pvals = pd.to_numeric(filtered.get("pvalue"), errors="coerce")
    below = pvals <= p_threshold
    dropped = int((~below & pvals.notna()).sum())
    if dropped:
        report.add(
            "info", "pvalue_filtered",
            f"Excluded {dropped} associations with p > {p_threshold:g}.",
        )
    filtered = filtered[below].copy()
    filtered["pvalue"] = pvals[below]

    chrom = filtered["chrom_raw"].map(lambda v: normalize_chrom(str(v).split(",")[0]))
    pos = parse_positions(filtered["pos_raw"].map(lambda v: str(v).split(",")[0]))
    parsed = filtered["risk_raw"].map(parse_strongest_risk_allele) if "risk_raw" in filtered else None

    out = pd.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "rsid": [p[0] for p in parsed] if parsed is not None else None,
            "risk_allele": [p[1] for p in parsed] if parsed is not None else None,
            "risk_strand": [p[2] for p in parsed] if parsed is not None else None,
            "pvalue": filtered["pvalue"].to_numpy(),
            "trait": filtered[trait_col].to_numpy(),
            "pubmed": filtered["pubmed"].astype(str).to_numpy() if "pubmed" in filtered else None,
            "or_beta": filtered["or_beta"] if "or_beta" in filtered else None,
            "mapped_gene": filtered["mapped_gene"] if "mapped_gene" in filtered else None,
            "study": filtered["study"] if "study" in filtered else None,
            "sample": filtered["sample"] if "sample" in filtered else None,
        }
    )
    out = out[out["chrom"].notna() & out["pos"].notna()].copy()
    out["pos"] = out["pos"].astype(np.int64)
    missing_risk = int(out["risk_allele"].isna().sum())
    if missing_risk:
        report.add(
            "warning", "risk_allele_missing",
            f"{missing_risk} index variants have no parsable risk allele; "
            "downstream allele checks fall back to position-only matching.",
        )
    report.add("info", "index_variants", f"{len(out)} index variant rows retained.")
    return out.reset_index(drop=True)


def deduplicate_variants(variants: pd.DataFrame, report: ValidationReport | None = None) -> pd.DataFrame:
    """Collapse repeated reports of the same position across studies.

    Keeps the strongest p-value per position and aggregates study metadata.
    """
    report = report if report is not None else ValidationReport()
    if variants.empty:
        return variants
    # Reset so idxmin labels are unique row numbers; duplicate index labels
    # would make .loc retrieve the wrong (or extra) rows.
    df = variants.copy().reset_index(drop=True)
    df["variant_uid"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)

    def _join_unique(series: pd.Series) -> str:
        values = sorted({str(v) for v in series.dropna() if str(v) not in {"", "nan", "None"}})
        return "; ".join(values)

    grouped = df.groupby("variant_uid", sort=False)
    out = grouped.first().reset_index()
    out["pvalue"] = grouped["pvalue"].min().to_numpy()
    # Allele / rsid provenance must come from the strongest association at
    # this position, not an arbitrary first row (grouped.first() is per
    # column and could mix studies).
    pnum = pd.to_numeric(df["pvalue"], errors="coerce")
    idx = df.assign(_p=pnum).groupby("variant_uid", sort=False)["_p"].idxmin()
    idx = idx.dropna()
    if len(idx):
        strong = df.loc[idx.to_numpy()].drop_duplicates(subset=["variant_uid"], keep="first")
        strong = strong.set_index("variant_uid")
        for col in ("rsid", "risk_allele", "risk_strand"):
            if col in out.columns:
                mapped = out["variant_uid"].map(strong[col])
                out[col] = mapped.fillna(out[col])
    agg_cols = [c for c in ("pubmed", "trait", "mapped_gene", "study") if c in df.columns]
    for col in agg_cols:
        out[col] = grouped[col].apply(_join_unique).to_numpy()
    n_before = len(df)
    if len(out) != n_before:
        report.add(
            "info", "variants_deduplicated",
            f"Collapsed {n_before} association rows into {len(out)} unique "
            "index variants (strongest p-value kept).",
        )
    return out


def assign_loci(variants: pd.DataFrame, gap_bp: int = 500_000) -> pd.DataFrame:
    """Cluster index variants into loci along each chromosome.

    Variants separated by more than *gap_bp* start a new locus.  Locus IDs
    are stable within a single call: ``chr{chrom}_{n}``.
    """
    if variants.empty:
        out = variants.copy()
        out["locus_id"] = []
        return out
    df = variants.sort_values(["chrom", "pos"]).copy().reset_index(drop=True)
    locus_parts = []
    counter = 0
    for chrom, group in df.groupby("chrom", sort=False):
        positions = group["pos"].to_numpy()
        breaks = np.flatnonzero(np.diff(positions) > gap_bp) + 1
        edges = np.concatenate([[0], breaks, [len(positions)]])
        block_ids = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
        local_ids = [f"chr{chrom}_{counter + int(i) + 1}" for i in block_ids]
        counter += len(set(block_ids))
        locus_parts.append(pd.Series(local_ids, index=group.index))
    df["locus_id"] = pd.concat(locus_parts).sort_index()
    return df


def variants_from_user_frame(
    frame: pd.DataFrame,
    pvalue: float | None = None,
    trait: str = "user variants",
) -> pd.DataFrame:
    """Build an index-variant table from a user-imported variant frame."""
    pos = pd.to_numeric(frame["pos"], errors="coerce")
    keep = pos.notna() & frame["chrom"].notna() & (pos >= 1)
    if not keep.all():
        frame = frame.loc[keep]
        pos = pos.loc[keep]
    out = pd.DataFrame(
        {
            "chrom": frame["chrom"].to_numpy(),
            "pos": pos.round().to_numpy(dtype=np.int64),
            "rsid": frame.get("rsid"),
            "risk_allele": frame.get("alt"),
            "risk_strand": "+",
            "pvalue": pvalue,
            "trait": trait,
            "pubmed": None,
            "or_beta": None,
            "mapped_gene": None,
            "study": "user_input",
            "sample": None,
        }
    )
    out["variant_uid"] = out["chrom"].astype(str) + ":" + out["pos"].astype(str)
    return out

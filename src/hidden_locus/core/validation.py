"""Input validation and harmonization helpers.

Every imported file is treated as untrusted input: extensions, required
columns, coordinate ranges, chromosome names, alleles and genome assembly
are checked before anything enters the analysis pipeline.  Problems are
reported as :class:`Issue` objects with actionable messages; hard errors
stop processing.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

CHROM_ALIASES = {
    "CHR1": "1", "CHR2": "2", "CHR3": "3", "CHR4": "4", "CHR5": "5",
    "CHR6": "6", "CHR7": "7", "CHR8": "8", "CHR9": "9", "CHR10": "10",
    "CHR11": "11", "CHR12": "12", "CHR13": "13", "CHR14": "14",
    "CHR15": "15", "CHR16": "16", "CHR17": "17", "CHR18": "18",
    "CHR19": "19", "CHR20": "20", "CHR21": "21", "CHR22": "22",
    "CHRX": "X", "CHRY": "Y", "CHRM": "MT", "CHRMT": "MT",
    "M": "MT", "MT": "MT", "X": "X", "Y": "Y",
}
VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
_MAX_CONTIG_LEN = 260_000_000  # generous GRCh38 upper bound

# Maximum accepted size for a single imported text file (untrusted input guard).
MAX_IMPORT_BYTES = 8 * 1024 ** 3


class ValidationError(RuntimeError):
    """Raised when input data cannot be used safely."""


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.level.upper()} {self.code}{loc}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue(level, code, message, where))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def raise_if_errors(self) -> None:
        if self.errors:
            joined = "\n".join(str(i) for i in self.errors)
            raise ValidationError(joined)

    def summary(self) -> str:
        if not self.issues:
            return "No issues found."
        return "\n".join(str(i) for i in self.issues)


def normalize_chrom(value) -> "str | None":
    """Return a canonical GRCh38 chromosome name, or None if invalid."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = CHROM_ALIASES.get(text, text)
    if text in VALID_CHROMS:
        return text
    return None


def check_assembly(
    declared: "str | None", report: ValidationReport, where: str = ""
) -> str:
    """Validate a declared assembly; only GRCh38 is supported in this release."""
    if declared is None or str(declared).strip() == "":
        report.add(
            "warning",
            "assembly_unspecified",
            "No genome assembly declared; assuming GRCh38. Confirm the input "
            "uses GRCh38 coordinates before trusting results.",
            where,
        )
        return config.SUPPORTED_ASSEMBLY
    normalized = str(declared).strip().upper().replace(" ", "")
    aliases = {
        "GRCH38": "GRCh38", "HG38": "GRCh38", "Hg38": "GRCh38",
        "GRCH38.p14": "GRCh38", "V38": "GRCh38",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        report.add(
            "error",
            "assembly_unsupported",
            f"Genome assembly '{declared}' is not supported. This release "
            f"works on {config.SUPPORTED_ASSEMBLY} only; lift over your data "
            "instead of mixing assemblies.",
            where,
        )
        return str(declared)
    return resolved


def parse_positions(series: pd.Series) -> pd.Series:
    """Coerce a position column to integers; invalid values become NA."""
    cleaned = (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def validate_coordinates(df: pd.DataFrame, report: ValidationReport, where: str = "") -> pd.DataFrame:
    """Check chrom/pos columns; drop invalid rows and record issues."""
    bad_chrom = df["chrom"].isna()
    bad_pos = df["pos"].isna() | (df["pos"] <= 0) | (df["pos"] > _MAX_CONTIG_LEN)
    n_bad = int((bad_chrom | bad_pos).sum())
    if n_bad:
        report.add(
            "error" if n_bad == len(df) else "warning",
            "invalid_coordinates",
            f"Dropped {n_bad} row(s) with missing or out-of-range coordinates.",
            where,
        )
    ok = ~(bad_chrom | bad_pos)
    out = df.loc[ok].copy()
    out["pos"] = out["pos"].astype(np.int64)
    return out.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], keep="first")


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_vcf(path: Path, report: ValidationReport | None = None) -> pd.DataFrame:
    """Parse a VCF (optionally gzipped) into a validated variant frame."""
    report = report if report is not None else ValidationReport()
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ValidationError(
            f"Refusing to import {path.name}: larger than "
            f"{MAX_IMPORT_BYTES / 1024 ** 3:.1f} GiB"
        )
    suffix = "".join(path.suffixes[-2:]).lower()
    if not (suffix.endswith(".vcf") or suffix.endswith(".vcf.gz")):
        report.add("error", "bad_extension", f"{path.name} is not a .vcf or .vcf.gz file", str(path))

    rows = []
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                if header[:8] != ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]:
                    report.add(
                        "error", "bad_vcf_header",
                        "VCF header does not start with the standard 8 columns.",
                        f"{path.name}:{line_no}",
                    )
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                report.add("warning", "short_vcf_row",
                           f"Line has {len(parts)} columns; skipped.",
                           f"{path.name}:{line_no}")
                continue
            chrom = normalize_chrom(parts[0])
            try:
                pos = int(parts[1])
            except ValueError:
                pos = None
            alt_field = parts[4]
            alts = [a.strip() for a in alt_field.split(",") if a.strip() and a.strip() != "."]
            rows.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "rsid": None if parts[2] == "." else parts[2],
                    "ref": parts[3].upper(),
                    "alt": ",".join(a.upper() for a in alts) if alts else None,
                    "qual": None if parts[5] == "." else parts[5],
                    "filter": None if parts[6] == "." else parts[6],
                }
            )
    frame = pd.DataFrame(rows, columns=["chrom", "pos", "rsid", "ref", "alt", "qual", "filter"])
    if frame.empty:
        report.add("error", "empty_vcf", "VCF contains no variant records.", str(path))
        return frame
    frame = validate_coordinates(frame.assign(pos=parse_positions(frame["pos"])), report, str(path))
    frame = frame[frame["alt"].notna()].copy()
    multi = frame["alt"].str.contains(",")
    if multi.any():
        report.add(
            "warning", "multiallelic",
            f"{int(multi.sum())} multiallelic record(s) kept with their first "
            "alternate allele only.", str(path),
        )
        frame.loc[multi, "alt"] = frame.loc[multi, "alt"].str.split(",").str[0]
    report.add("info", "vcf_loaded", f"Loaded {len(frame)} variants.", str(path))
    return frame.reset_index(drop=True)


def read_bed(path: Path, report: ValidationReport | None = None) -> pd.DataFrame:
    """Parse a BED file (0-based half-open) into a validated interval frame."""
    report = report if report is not None else ValidationReport()
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    frame = pd.read_csv(
        path, sep="\t", comment="#", header=None, dtype=str, low_memory=False
    )
    if frame.shape[1] < 3:
        report.add("error", "bad_bed", "BED file needs at least 3 columns.", str(path))
        return pd.DataFrame(columns=["chrom", "start", "end", "name"])
    out = pd.DataFrame(
        {
            "chrom": frame[0].map(normalize_chrom),
            "start": pd.to_numeric(frame[1], errors="coerce"),
            "end": pd.to_numeric(frame[2], errors="coerce"),
            "name": frame[3] if frame.shape[1] > 3 else None,
        }
    )
    bad = out["start"].isna() | out["end"].isna() | (out["start"] < 0) | (out["end"] <= out["start"])
    if bad.any():
        report.add("warning", "bad_bed_rows", f"Dropped {int(bad.sum())} invalid BED rows.", str(path))
        out = out[~bad]
    return out.reset_index(drop=True)


def read_fasta_contigs(path: Path, report: ValidationReport | None = None) -> pd.DataFrame:
    """Validate a FASTA file and return contig names with lengths."""
    from Bio import SeqIO

    report = report if report is not None else ValidationReport()
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    records = []
    with _open_text(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append({"contig": record.id, "length": len(record.seq)})
    frame = pd.DataFrame(records)
    if frame.empty:
        report.add("error", "empty_fasta", "FASTA contains no sequences.", str(path))
        return frame
    report.add(
        "info", "fasta_loaded",
        f"Loaded {len(frame)} contigs. FASTA alone does not define a disease "
        "region; pair it with association or interval data.", str(path),
    )
    return frame


def harmonize_allele_pair(gwas_allele: "str | None", ref: "str | None", alt: "str | None") -> str:
    """Classify how a GWAS risk allele relates to a reference/alternate pair.

    Returns one of ``allele_verified``, ``strand_flip``, ``position_only`` or
    ``allele_mismatch``.
    """
    if not gwas_allele or not ref or not alt:
        return "position_only"
    risk = str(gwas_allele).strip().upper()
    ref_u, alt_u = str(ref).upper(), str(alt).upper()
    if risk in {ref_u, alt_u}:
        return "allele_verified"
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    if len(risk) == 1 and len(ref_u) == 1 and len(alt_u) == 1:
        if complement.get(risk) in {ref_u, alt_u}:
            return "strand_flip"
    if len(risk) != len(ref_u) and len(risk) != len(alt_u):
        return "position_only"  # indel representation differences
    return "allele_mismatch"


_RS_ALLELE_RE = re.compile(
    r"^(rs\d+)\s*[-:]?\s*([ACGT]+|\?)?\s*(?:\(([+-])\))?$", re.IGNORECASE
)


def parse_strongest_risk_allele(text: "str | None") -> tuple["str | None", "str | None", "str | None"]:
    """Split a GWAS Catalog 'STRONGEST SNP-RISK ALLELE' cell.

    Returns ``(rsid, risk_allele, strand)`` where strand may be ``+``, ``-``
    or None.  A reported ``-?`` (allele not stated) yields ``risk_allele=None``.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None, None, None
    match = _RS_ALLELE_RE.match(str(text).strip())
    if not match:
        return None, None, None
    rsid, allele, strand = match.groups()
    if allele and allele.upper() == "?":
        allele = None
    return rsid.lower(), allele.upper() if allele else None, strand


def check_required_columns(
    frame: pd.DataFrame, required: list[str], report: ValidationReport, where: str = ""
) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        report.add(
            "error", "missing_columns",
            f"Missing required column(s): {', '.join(missing)}", where,
        )


def ensure_unique(df: pd.DataFrame, subset: list[str], report: ValidationReport, where: str = "") -> pd.DataFrame:
    """Drop duplicated rows on *subset*, recording how many were removed."""
    dupes = int(df.duplicated(subset=subset).sum())
    if dupes:
        report.add(
            "warning", "duplicates_removed",
            f"Removed {dupes} duplicate row(s) on {subset}.", where,
        )
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

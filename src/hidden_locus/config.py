"""Application configuration, paths and persisted user settings.

All runtime data (downloads, caches, models, reports, logs) lives under a
single data directory that is separate from the source tree.  The directory
defaults to ``<project root>/data`` and can be redirected with the
``HIDDEN_LOCUS_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "Hidden Locus"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HIDDEN_LOCUS_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()

DATASETS_DIR = DATA_DIR / "datasets"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"
SETTINGS_FILE = DATA_DIR / "settings.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# The MVP targets a single genome assembly.  Coordinates from other
# assemblies must never be mixed in silently.
SUPPORTED_ASSEMBLY = "GRCh38"
ASSEMBLY_CHOICES = ("GRCh38",)

# Candidate variant-to-gene window around the transcription start site.
DEFAULT_WINDOW_BP = 1_000_000
# Variants closer than this distance are merged into one locus.
DEFAULT_LOCUS_GAP_BP = 500_000
# Nominal p-value threshold defining a positive label in the independent
# (label) eQTL study.  Documented, fixed for reproducibility.
DEFAULT_LABEL_P_THRESHOLD = 1e-5
# Disease/phenotype selection pattern applied to the GWAS Catalog trait
# column for the default research scope (Crohn's disease / IBD).
DEFAULT_TRAIT_PATTERN = r"(?i)crohn|inflammatory bowel|ulcerative colitis"
DEFAULT_DISEASE_QUERY = "Crohn's disease / inflammatory bowel disease"
# GTEx tissues used as tissue-context evidence for the IBD scope.
# Whole_Blood removed: overlaps with BLUEPRINT monocyte label study.
DEFAULT_TISSUES = (
    "Small_Intestine_Terminal_Ileum",
    "Colon_Transverse",
)
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
# PU bagging is fixed for reproducibility; recorded in provenance.json.
DEFAULT_PU_N_BAGS = 20

GENOME_BUILD_NOTE = (
    "Hidden Locus works on GRCh38 coordinates only in this release. "
    "Input files on another assembly must be lifted over before import; "
    "coordinates are never mixed across assemblies."
)


def ensure_data_dirs() -> None:
    """Create the data directory tree if it does not exist yet."""
    for path in (DATASETS_DIR, CACHE_DIR, MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """User-configurable analysis settings, persisted as JSON."""

    disease_query: str = DEFAULT_DISEASE_QUERY
    trait_pattern: str = DEFAULT_TRAIT_PATTERN
    assembly: str = SUPPORTED_ASSEMBLY
    tissues: list[str] = field(default_factory=lambda: list(DEFAULT_TISSUES))
    window_bp: int = DEFAULT_WINDOW_BP
    locus_gap_bp: int = DEFAULT_LOCUS_GAP_BP
    label_p_threshold: float = DEFAULT_LABEL_P_THRESHOLD
    gene_biotypes: list[str] = field(
        default_factory=lambda: ["protein_coding", "lncRNA"]
    )
    seed: int = DEFAULT_SEED
    n_splits: int = DEFAULT_N_SPLITS
    model_name: str = "xgboost"
    use_pu: bool = True
    candidate_variant_p: float = 5e-8
    use_coloc_features: bool = True
    coloc_pp4_threshold: float = 0.8

    def save(self) -> None:
        ensure_data_dirs()
        SETTINGS_FILE.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @property
    def cache_key(self) -> str:
        """Hash of all settings that affect feature computation."""
        import hashlib
        relevant = {
            "trait_pattern": self.trait_pattern,
            "window_bp": self.window_bp,
            "locus_gap_bp": self.locus_gap_bp,
            "label_p_threshold": self.label_p_threshold,
            "tissues": tuple(sorted(self.tissues)),
            "gene_biotypes": tuple(sorted(self.gene_biotypes)),
            "use_coloc_features": self.use_coloc_features,
            "coloc_pp4_threshold": self.coloc_pp4_threshold,
        }
        return hashlib.sha256(str(relevant).encode()).hexdigest()[:12]

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return cls()
            if not isinstance(data, dict):
                return cls()
            known = {f for f in cls.__dataclass_fields__}
            data = {k: v for k, v in data.items() if k in known}
            defaults = cls()
            # Coerce fields a hand-edited settings.json could get wrong so a
            # bad type surfaces as a default, not a TypeError mid-pipeline.
            for name, cast in (
                ("window_bp", int), ("locus_gap_bp", int), ("seed", int),
                ("n_splits", int), ("candidate_variant_p", float),
                ("label_p_threshold", float),
            ):
                if name in data:
                    try:
                        data[name] = cast(data[name])
                    except (TypeError, ValueError):
                        data[name] = getattr(defaults, name)
            if "tissues" in data:
                data["tissues"] = (
                    [str(t) for t in data["tissues"] if str(t).strip()]
                    if isinstance(data["tissues"], list) else list(defaults.tissues)
                )
            if "gene_biotypes" in data:
                data["gene_biotypes"] = (
                    [str(g) for g in data["gene_biotypes"]]
                    if isinstance(data["gene_biotypes"], list)
                    else list(defaults.gene_biotypes)
                )
            if "use_pu" in data and not isinstance(data["use_pu"], bool):
                data["use_pu"] = bool(data["use_pu"])
            return cls(**data)
        return cls()

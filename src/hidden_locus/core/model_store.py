"""Model persistence and scoring API for deploying trained models."""

from __future__ import annotations

import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODELS_DIR, Settings
from .features import build_feature_table, FeatureContext, finalize_features
from .pipeline import PipelineResult
from .ranking import TrainedModel, train_candidate_model
from .validation import ValidationReport


@dataclass
class ModelArtifact:
    """Serializable model artifact with metadata."""

    model: TrainedModel
    settings_hash: str
    feature_columns: list[str]
    n_positive: int
    n_candidates: int
    cv_metrics: dict | None = None
    created_at: str = ""
    version: str = "1.0"


def save_model(
    model: TrainedModel,
    path: Path | str,
    pipeline_result: PipelineResult | None = None,
) -> Path:
    """Save a trained model with metadata to disk.

    Uses joblib for efficient serialization of numpy arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    settings_hash = ""
    n_pos = 0
    n_cand = 0
    cv_metrics = None
    if pipeline_result is not None:
        settings_hash = pipeline_result.settings.cache_key
        n_pos = pipeline_result.n_positives
        n_cand = len(pipeline_result.candidates)
        if pipeline_result.cv is not None:
            cv_metrics = pipeline_result.cv.summary()

    artifact = ModelArtifact(
        model=model,
        settings_hash=settings_hash,
        feature_columns=model.features,
        n_positive=n_pos,
        n_candidates=n_cand,
        cv_metrics=cv_metrics,
        created_at=pd.Timestamp.now().isoformat(),
    )

    joblib.dump(artifact, path)
    return path


def load_model(path: Path | str) -> TrainedModel:
    """Load a trained model from disk."""
    path = Path(path)
    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise ValueError(f"Invalid model artifact at {path}")
    return artifact.model


def score_new_variants(
    model: TrainedModel,
    variants_df: pd.DataFrame,
    genes: Any,
    ccre: Any = None,
    abc: Any = None,
    gut_atlas: dict | None = None,
    report: ValidationReport | None = None,
) -> pd.DataFrame:
    """Score new user-provided variants against a trained model.

    Parameters
    ----------
    model : TrainedModel
        The trained model to use for scoring.
    variants_df : DataFrame
        User-provided variants with columns: chrom, pos, rsid (optional), risk_allele (optional).
    genes : GeneAnnotation
        Gene annotation for candidate generation.
    ccre : CcreAnnotation, optional
        Regulatory annotations.
    abc : AbcPredictions, optional
        ABC Model predictions.
    gut_atlas : dict, optional
        Gut Atlas eQTL data.
    report : ValidationReport, optional
        For validation issues.

    Returns
    -------
    DataFrame with ranked candidates and model scores.
    """
    report = report if report is not None else ValidationReport()

    # Convert user variants to candidate format
    from .preprocessing import variants_from_user_frame
    variants = variants_from_user_frame(variants_df)

    # Generate candidates
    from .features import generate_candidates
    candidates = generate_candidates(variants, genes)

    if candidates.empty:
        report.add("error", "no_candidates", "No variant-gene candidates generated from input variants.")
        return pd.DataFrame()

    # Build features (without labels)
    from .features import FeatureContext, build_feature_table, finalize_features
    uids = set(candidates["variant_uid"])
    ctx = FeatureContext(
        genes=genes, ccre=ccre, abc=abc, gut_atlas=gut_atlas,
        gtex_loaded=False, regulatory_loaded=ccre is not None,
    )

    # Note: GTEx features won't be available without GTEx data
    features = build_feature_table(candidates, ctx, tissues=[])
    features = finalize_features(features)

    # Score with model
    from .pipeline import rank_candidates
    from .pipeline import PipelineResult
    from .config import Settings

    dummy_result = PipelineResult(report=ValidationReport(), settings=Settings())
    dummy_result.features = features
    dummy_result.model = model
    ranked = rank_candidates(features, model)

    return ranked


class ModelRegistry:
    """Simple registry for managing multiple trained models."""

    def __init__(self, registry_dir: Path | str = MODELS_DIR):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        import json
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def _save_index(self):
        import json
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def register(self, name: str, model_path: Path, metadata: dict | None = None) -> str:
        """Register a model under a name."""
        if name in self._index:
            raise ValueError(f"Model '{name}' already registered")
        self._index[name] = {
            "path": str(model_path),
            "metadata": metadata or {},
            "registered_at": pd.Timestamp.now().isoformat(),
        }
        self._save_index()
        return name

    def get(self, name: str) -> TrainedModel:
        """Load a registered model by name."""
        if name not in self._index:
            raise KeyError(f"Model '{name}' not found in registry")
        path = Path(self._index[name]["path"])
        return load_model(path)

    def list_models(self) -> list[dict]:
        """List all registered models."""
        return [{"name": k, **v} for k, v in self._index.items()]

    def remove(self, name: str) -> None:
        """Remove a model from registry (does not delete file)."""
        if name in self._index:
            del self._index[name]
            self._save_index()
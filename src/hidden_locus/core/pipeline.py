"""End-to-end pipeline orchestration for the default IBD research scope.

Stages (each callable independently, with progress reporting):

1. ``prepare_variants``   GWAS Catalog -> deduplicated index variants + loci
2. ``prepare_annotations`` GENCODE + SCREEN (+ optional GTEx extraction)
3. ``build_candidates``   variant x gene candidate pairs
4. ``build_features``     evidence features per candidate pair
5. ``attach_labels``      independent-study (BLUEPRINT) positive labels
6. ``train_and_evaluate`` grouped CV, baselines, ablations
7. ``export_report``      tables, figures, provenance bundle

Every stage appends to a shared :class:`ValidationReport`.  Nothing is
downloaded here; datasets must already be present (see
:mod:`hidden_locus.core.datasets`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config
from . import datasets as ds
from .evaluation import CVResult, compare_baselines, feature_group_ablations, grouped_cross_validate
from .explain import global_importance
from .features import (
    FeatureContext,
    GeneAnnotation,
    CcreAnnotation,
    attach_labels,
    build_feature_table,
    finalize_features,
    generate_candidates,
    load_gtex_tissue,
    load_label_eqtl,
)
from .abc_model import AbcPredictions, load_abc_predictions, merge_abc_with_candidates
from .coloc import run_coloc_all_loci
from .gut_atlas import load_gut_atlas_all
from .preprocessing import (
    assign_loci,
    deduplicate_variants,
    load_gwas_associations,
    select_index_variants,
)
from .ranking import train_candidate_model
from .tuning import nested_cv, train_ensemble, tune_lightgbm, tune_xgboost
from .validation import ValidationReport

ProgressFn = Callable[[str, int, int], None]  # (stage, done, total)


def _noop_progress(_stage: str, _done: int, _total: int) -> None:
    return None


@dataclass
class PipelineResult:
    """Everything produced by a full pipeline run."""

    report: ValidationReport
    settings: config.Settings
    variants: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels_pvalue_threshold: float = config.DEFAULT_LABEL_P_THRESHOLD
    cv: "CVResult | None" = None
    baseline_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    ablation_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: object = None
    model: object = None
    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: dict = field(default_factory=dict)
    enrichment_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    external_benchmark_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    coloc_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    pathway_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    drug_overlap_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    drug_overlap_summary: dict = field(default_factory=dict)

    @property
    def n_positives(self) -> int:
        if self.features.empty or "label" not in self.features.columns:
            return 0
        return int(self.features["label"].sum())


def prepare_variants(
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> pd.DataFrame:
    """Stage 1: GWAS associations -> index variants clustered into loci."""
    t0 = time.time()
    progress("gwas", 0, 4)
    zip_paths = ds.require("gwas_catalog")
    progress("gwas", 1, 4)
    assoc = load_gwas_associations(zip_paths[0], report)
    progress("gwas", 2, 4)
    idx = select_index_variants(
        assoc, settings.trait_pattern, p_threshold=settings.candidate_variant_p,
        report=report,
    )
    dedup = deduplicate_variants(idx, report)
    loci = assign_loci(dedup, settings.locus_gap_bp)
    report.raise_if_errors()
    progress("gwas", 4, 4)
    report.add(
        "info", "stage_variants",
        f"{len(loci)} index variants in {loci['locus_id'].nunique()} loci "
        f"({time.time() - t0:.1f}s).",
    )
    return loci


def prepare_annotations(
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> tuple[GeneAnnotation, "CcreAnnotation | None", "AbcPredictions | None", dict]:
    """Stage 2: load GENCODE genes, (if present) SCREEN cCREs, ABC Model, and Gut Atlas eQTL."""
    progress("annotation", 0, 5)
    gtf_paths = ds.require("gene_annotation")
    progress("annotation", 1, 5)
    genes = GeneAnnotation.load_gtf(gtf_paths[0], biotypes=settings.gene_biotypes, report=report)
    ccre = None
    try:
        ccre_paths = ds.require("encode_screen")
        ccre = CcreAnnotation.load_bed(ccre_paths[0], report)
    except ds.DatasetError as exc:
        report.add("warning", "no_ccre", str(exc))
    progress("annotation", 2, 5)
    abc = None
    try:
        abc_paths = ds.require("abc_model")
        abc = load_abc_predictions("all", report)
    except ds.DatasetError as exc:
        report.add("warning", "no_abc", str(exc))
    progress("annotation", 3, 5)
    gut_atlas = None
    try:
        gut_atlas = load_gut_atlas_all("all", report=report)
    except ds.DatasetError as exc:
        report.add("warning", "no_gut_atlas", str(exc))
    progress("annotation", 4, 5)
    progress("annotation", 5, 5)
    return genes, ccre, abc, gut_atlas


def build_candidates_stage(
    variants: pd.DataFrame,
    genes: GeneAnnotation,
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> pd.DataFrame:
    progress("candidates", 0, 1)
    cands = generate_candidates(variants, genes, window_bp=settings.window_bp)
    if cands.empty:
        report.add("error", "no_candidates", "No variant-gene candidates were generated.")
    else:
        report.add(
            "info", "stage_candidates",
            f"{len(cands)} candidate pairs over {cands['variant_uid'].nunique()} variants.",
        )
    progress("candidates", 1, 1)
    return cands


def build_features_stage(
    candidates: pd.DataFrame,
    genes: GeneAnnotation,
    ccre: "CcreAnnotation | None",
    abc: "AbcPredictions | None",
    gut_atlas: dict,
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> tuple[pd.DataFrame, FeatureContext]:
    """Stage 4: load GTEx evidence for candidate variants, build features."""
    if candidates.empty:
        return pd.DataFrame(), FeatureContext(genes=genes, ccre=ccre, abc=abc, gut_atlas=gut_atlas)
    uids = set(candidates["variant_uid"])
    ctx = FeatureContext(
        genes=genes, ccre=ccre, abc=abc, gut_atlas=gut_atlas,
        gtex_loaded=False, regulatory_loaded=ccre is not None,
    )
    progress("features", 0, 3)
    try:
        gtex_paths = ds.require("gtex_eqtl")
        # ensure tissue files are extracted from the tar
        ds.extract_gtex_tissues(settings.tissues)
    except ds.DatasetError as exc:
        report.add("warning", "no_gtex", f"GTEx features unavailable: {exc}")
    else:
        ex_dir = gtex_paths[0].parent / "extracted"
        total = len(settings.tissues)
        for i, tissue in enumerate(settings.tissues):
            pairs_path = ex_dir / f"{tissue}.signif_variant_gene_pairs.txt.gz"
            egenes_path = ex_dir / f"{tissue}.egenes.txt.gz"
            if not (pairs_path.exists() and egenes_path.exists()):
                report.add(
                    "warning", "missing_tissue_files",
                    f"Missing GTEx files for tissue {tissue}; skipped.",
                )
                progress("features", i + 1, total)
                continue
            pairs, egenes = load_gtex_tissue(pairs_path, egenes_path, uids, report)
            ctx.gtex[tissue] = pairs
            ctx.gtex_egenes[tissue] = egenes
            progress("features", i + 1, total)
        ctx.gtex_loaded = bool(ctx.gtex)

    pip = _load_pips(candidates, report)
    if pip is not None:
        ctx.pip = pip

    if settings.use_coloc_features:
        coloc = _load_coloc(candidates, report)
        if coloc is not None:
            ctx.coloc = coloc
            ctx.coloc_loaded = True

    progress("features", 3, 3)

    ft = build_feature_table(candidates, ctx, tissues=list(settings.tissues))
    ft = finalize_features(ft)
    return ft, ctx


def _load_pips(candidates: pd.DataFrame, report: ValidationReport) -> "pd.Series | None":
    """Optional fine-mapping PIPs from a local cache file, else None.

    Looks for ``<data>/cache/fine_mapping_pips.csv`` with columns
    ``variant_uid`` (or chrom/pos) and ``pip``.  Absence is reported and the
    PIP feature is marked missing — never invented.
    """
    path = config.CACHE_DIR / "fine_mapping_pips.csv"
    if not path.exists():
        report.add(
            "info", "no_pips",
            "No fine-mapping PIP file found (data/cache/fine_mapping_pips.csv); "
            "PIP feature marked missing.",
        )
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        report.add("warning", "bad_pip_file", f"Could not read PIP file: {exc}")
        return None
    if "pip" not in frame.columns:
        report.add("warning", "bad_pip_file", "PIP file lacks a 'pip' column.")
        return None
    if "variant_uid" not in frame.columns:
        if {"chrom", "pos"} <= set(frame.columns):
            # Guard against CSV round-trip turning integers into floats.
            chrom_s = frame["chrom"].astype(str).str.replace(r"\.0$", "", regex=True)
            pos_s = pd.to_numeric(frame["pos"], errors="coerce").astype("Int64").astype(str)
            frame["variant_uid"] = chrom_s + ":" + pos_s
        else:
            report.add("warning", "bad_pip_file", "PIP file lacks variant_uid or chrom/pos.")
            return None
    s = frame.drop_duplicates("variant_uid").set_index("variant_uid")["pip"]
    covered = 0
    if candidates is not None and not candidates.empty and len(s):
        # Count unique variants, not (variant, gene) rows — the message
        # promises variant coverage.
        covered = int(candidates.loc[
            candidates["variant_uid"].isin(s.index), "variant_uid"
        ].nunique())
    report.add(
        "info", "pips_loaded",
        f"Loaded PIPs for {len(s)} variants; {covered} candidate variants covered.",
    )
    return s.astype(float)


def _load_coloc(candidates: pd.DataFrame, report: ValidationReport) -> "pd.DataFrame | None":
    """Load coloc H4/CLPP evidence from local cache (from OT L2G features).

    Looks for ``<data>/cache/opentargets_coloc.csv`` with columns
    ``chrom``, ``pos``, ``gene_id``, ``coloc_eqtl_h4``, ``coloc_eqtl_clpp``,
    ``coloc_sqtl_h4``, ``coloc_pqtl_h4``.  Absence is reported and the coloc
    feature is marked missing — never invented.
    """
    path = config.CACHE_DIR / "opentargets_coloc.csv"
    if not path.exists():
        report.add(
            "info", "no_coloc_cache",
            "No coloc cache (data/cache/opentargets_coloc.csv); coloc features "
            "marked missing. Build with: data/cache/_build_ot_cache.py",
        )
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        report.add("warning", "bad_coloc_cache", f"Could not read coloc cache: {exc}")
        return None
    needed = {"chrom", "pos", "gene_id"}
    if not needed <= set(frame.columns):
        report.add("warning", "bad_coloc_cache", f"Coloc cache needs columns {sorted(needed)}.")
        return None
    # Normalize chrom/pos into variant_uid like candidates.
    chrom_s = frame["chrom"].astype(str).str.replace(r"\.0$", "", regex=True)
    pos_s = pd.to_numeric(frame["pos"], errors="coerce").astype("Int64").astype(str)
    frame["variant_uid"] = chrom_s + ":" + pos_s
    frame = frame.drop_duplicates(subset=["variant_uid", "gene_id"], keep="first")
    covered = 0
    if candidates is not None and not candidates.empty and len(frame):
        covered = int(candidates.loc[
            candidates["variant_uid"].isin(set(frame["variant_uid"])), "variant_uid"
        ].nunique())
    report.add(
        "info", "coloc_loaded",
        f"Loaded coloc evidence for {len(frame)} pairs; {covered} candidate "
        "variants covered.",
    )
    return frame


def attach_labels_stage(
    features: pd.DataFrame,
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> pd.DataFrame:
    """Stage 5: positive labels from the independent BLUEPRINT study."""
    if features.empty:
        return features
    uids = set(features["variant_uid"])
    progress("labels", 0, 2)
    try:
        pair_paths = ds.require("label_eqtl")
        perm_paths = ds.require("label_eqtl_permuted")
    except ds.DatasetError as exc:
        report.add("warning", "no_labels", f"{exc}; all rows remain unlabelled.")
        out = features.copy()
        out["label"] = 0
        out["label_pvalue"] = float("nan")
        out["label_study"] = "none"
        progress("labels", 2, 2)
        return out
    pairs, egenes = load_label_eqtl(
        pair_paths[0], perm_paths[0], uids, settings.label_p_threshold, report,
        settings_cache_key=settings.cache_key,
    )
    progress("labels", 2, 2)
    return attach_labels(
        features, pairs, label_study="eQTL Catalogue BLUEPRINT QTD000021",
        p_threshold=settings.label_p_threshold,
    )


def train_and_evaluate_stage(
    features: pd.DataFrame,
    settings: config.Settings,
    report: ValidationReport,
    progress: ProgressFn = _noop_progress,
) -> tuple[CVResult, pd.DataFrame, pd.DataFrame, object, object]:
    """Stage 6: grouped CV for the candidate model, baselines, ablations."""
    if features.empty:
        raise ValueError("No feature table to train on")
    if "label" not in features.columns or int(features["label"].sum()) == 0:
        report.add(
            "warning", "no_positive_labels",
            "No positive labels available; training skipped. Download the "
            "label dataset (label_eqtl + label_eqtl_permuted) and rerun.",
        )
        return CVResult("none"), pd.DataFrame(), pd.DataFrame(), None, None
    progress("cv", 0, 4)
    cv = grouped_cross_validate(
        features, model_name=settings.model_name, use_pu=settings.use_pu,
        n_splits=settings.n_splits, seed=settings.seed,
    )
    progress("cv", 1, 4)
    baselines = compare_baselines(
        features, cv, l2g_scores=_load_l2g(report),
        model_name=settings.model_name, use_pu=settings.use_pu,
        seed=settings.seed,
    )
    progress("cv", 2, 4)
    try:
        ablations = feature_group_ablations(
            features, model_name=settings.model_name,
            n_splits=settings.n_splits, seed=settings.seed,
            use_pu=settings.use_pu,
            full_result=cv,
        )
    except (ValueError, RuntimeError) as exc:
        report.add("warning", "ablation_failed", f"Ablations skipped: {exc}")
        ablations = pd.DataFrame()
    progress("cv", 3, 4)

    # Optional hyperparameter tuning
    best_params = None
    if getattr(settings, "tune_hyperparams", False):
        progress("cv", 3.5, 4)
        try:
            if settings.model_name == "xgboost":
                best_params, best_auc = tune_xgboost(
                    features, n_trials=30, n_splits=settings.n_splits,
                    seed=settings.seed, use_pu=settings.use_pu,
                )
            elif settings.model_name == "lightgbm":
                best_params, best_auc = tune_lightgbm(
                    features, n_trials=30, n_splits=settings.n_splits,
                    seed=settings.seed, use_pu=settings.use_pu,
                )
            report.add("info", "hyperparam_tuned", f"Best AUC: {best_auc:.4f}, params: {best_params}")
        except Exception as exc:
            report.add("warning", "tuning_failed", f"Hyperparameter tuning failed: {exc}")
        progress("cv", 4, 4)
    else:
        progress("cv", 4, 4)

    # Final model on all labelled data for explanations / scoring.
    # Use tuned params if available
    model_kwargs = dict(
        model_name=settings.model_name, use_pu=settings.use_pu,
        seed=settings.seed,
    )
    if best_params:
        # Note: train_candidate_model doesn't accept custom params yet
        report.add("warning", "tuned_params_not_applied", "Tuned params not yet integrated into final model")
    model = train_candidate_model(features, **model_kwargs)
    importance = None
    try:
        importance = global_importance(model, features)
    except (ValueError, RuntimeError) as exc:
        report.add("warning", "importance_failed", f"Importance skipped: {exc}")
    return cv, baselines, ablations, model, importance


def _load_l2g(report: ValidationReport) -> "pd.DataFrame | None":
    """Optional Open Targets L2G comparator file from the local cache."""
    path = config.CACHE_DIR / "opentargets_l2g.csv"
    if not path.exists():
        report.add(
            "info", "no_l2g",
            "No Open Targets L2G file found (data/cache/opentargets_l2g.csv); "
            "external comparator omitted.",
        )
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        report.add("warning", "bad_l2g_file", f"Could not read L2G file: {exc}")
        return None
    needed = {"chrom", "pos", "gene_id", "l2g_score"}
    if not needed <= set(frame.columns):
        report.add("warning", "bad_l2g_file", f"L2G file needs columns {sorted(needed)}.")
        return None
    report.add("info", "l2g_loaded", f"Loaded {len(frame)} L2G comparator rows.")
    return frame


def rank_candidates(
    features: pd.DataFrame,
    model,
    progress: ProgressFn = _noop_progress,
) -> pd.DataFrame:
    """Stage 7: score every candidate row with the final model and rank."""
    if model is None or features.empty:
        return features.copy()
    progress("rank", 0, 1)
    out = features.copy()
    out["model_score"] = model.predict_proba(out)
    out = out.sort_values(
        ["model_score", "locus_id", "gene_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    out["rank"] = out.index + 1
    progress("rank", 1, 1)
    return out


def run_full_pipeline(
    settings: config.Settings | None = None,
    report: ValidationReport | None = None,
    progress: ProgressFn = _noop_progress,
) -> PipelineResult:
    """Run every pipeline stage in order and return a result bundle."""
    settings = settings or config.Settings.load()
    report = report if report is not None else ValidationReport()
    result = PipelineResult(report=report, settings=settings,
                            labels_pvalue_threshold=settings.label_p_threshold)

    t0 = time.time()
    result.variants = prepare_variants(settings, report, progress)
    result.timing["variants"] = time.time() - t0

    t0 = time.time()
    genes, ccre, abc, gut_atlas = prepare_annotations(settings, report, progress)
    result.timing["annotations"] = time.time() - t0

    t0 = time.time()
    result.candidates = build_candidates_stage(result.variants, genes, settings, report, progress)
    result.timing["candidates"] = time.time() - t0

    t0 = time.time()
    result.features, _ctx = build_features_stage(
        result.candidates, genes, ccre, abc, gut_atlas, settings, report, progress
    )
    result.timing["features"] = time.time() - t0

    t0 = time.time()
    result.features = attach_labels_stage(result.features, settings, report, progress)
    result.timing["labels"] = time.time() - t0

    t0 = time.time()
    (result.cv, result.baseline_table, result.ablation_table,
     result.model, result.importance) = train_and_evaluate_stage(
        result.features, settings, report, progress
    )
    result.timing["train_eval"] = time.time() - t0

    result.ranked = rank_candidates(result.features, result.model, progress)

    # Run enrichment validation vs curated truth sets.
    t0 = time.time()
    try:
        from .enrichment import enrichment_summary
        result.enrichment_table = enrichment_summary(
            result.ranked, settings, report
        )
    except Exception as exc:
        report.add("warning", "enrichment_failed", f"Enrichment validation skipped: {exc}")
    result.timing["enrichment"] = time.time() - t0

    # Run external benchmark vs L2G / PoPS.
    t0 = time.time()
    try:
        from .external_benchmark import load_l2g_scores, load_pops_scores, run_external_benchmark
        l2g = load_l2g_scores(report)
        pops = load_pops_scores(report)
        result.external_benchmark_table = run_external_benchmark(
            result.ranked, l2g, pops, curated_truth=None, report=report
        )
    except Exception as exc:
        report.add("warning", "bench_failed", f"External benchmark skipped: {exc}")
    result.timing["external_benchmark"] = time.time() - t0

    # Run pathway enrichment of top genes.
    t0 = time.time()
    try:
        from .pathway_enrichment import pathway_summary
        result.pathway_table = pathway_summary(result.ranked, report)
    except Exception as exc:
        report.add("warning", "pathway_failed", f"Pathway enrichment skipped: {exc}")
    result.timing["pathway_enrichment"] = time.time() - t0

    # Run drug-target overlap.
    t0 = time.time()
    try:
        from .drug_target_overlap import drug_target_overlap, drug_target_summary
        result.drug_overlap_table = drug_target_overlap(result.ranked, top_n=100)
        result.drug_overlap_summary = drug_target_summary(result.ranked, report)
    except Exception as exc:
        report.add("warning", "drug_overlap_failed", f"Drug-target overlap skipped: {exc}")
    result.timing["drug_target_overlap"] = time.time() - t0

    report.add(
        "info", "pipeline_done",
        f"Pipeline finished in {sum(result.timing.values()):.1f}s; "
        f"{result.n_positives} positive labels.",
    )
    return result


def export_report(
    result: PipelineResult,
    out_dir: "Path | None" = None,
    progress: ProgressFn = _noop_progress,
) -> dict:
    """Write ranked table, metrics, importance, figures and provenance."""
    from . import reporting as rp

    out_dir = Path(out_dir) if out_dir else config.REPORTS_DIR / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    progress("export", 0, 8)
    written["ranked_table"] = rp.write_ranked_table(result.ranked, out_dir / "ranked_candidates.csv")
    if not result.baseline_table.empty:
        written["metrics"] = rp.write_metrics_table(result.baseline_table, out_dir / "metrics.csv")
    if not result.ablation_table.empty:
        written["ablations"] = rp.write_metrics_table(result.ablation_table, out_dir / "ablations.csv")
    if result.importance is not None:
        written["importance"] = rp.write_importance_table(result.importance, out_dir / "feature_importance.csv")
    if result.model is not None and not result.ranked.empty:
        try:
            written["contributions"] = rp.write_row_contributions(
                result.model, result.ranked, out_dir / "top_row_contributions.csv"
            )
        except (ValueError, RuntimeError) as exc:
            result.report.add("warning", "contrib_export_failed", str(exc))
    progress("export", 4, 8)
    # Write enrichment table.
    if not result.enrichment_table.empty:
        result.enrichment_table.to_csv(out_dir / "enrichment.csv", index=False)
        written["enrichment"] = out_dir / "enrichment.csv"
    # Write external benchmark table.
    if not result.external_benchmark_table.empty:
        result.external_benchmark_table.to_csv(out_dir / "external_benchmark.csv", index=False)
        written["external_benchmark"] = out_dir / "external_benchmark.csv"
    # Write coloc summary if present.
    if not result.coloc_table.empty:
        result.coloc_table.to_csv(out_dir / "coloc_summary.csv", index=False)
        written["coloc_summary"] = out_dir / "coloc_summary.csv"
    # Write pathway enrichment table.
    if not result.pathway_table.empty:
        result.pathway_table.to_csv(out_dir / "pathway_enrichment.csv", index=False)
        written["pathway_enrichment"] = out_dir / "pathway_enrichment.csv"
    # Write drug-target overlap table.
    if not result.drug_overlap_table.empty:
        result.drug_overlap_table.to_csv(out_dir / "drug_target_overlap.csv", index=False)
        written["drug_target_overlap"] = out_dir / "drug_target_overlap.csv"
    progress("export", 5, 8)
    if result.cv is not None and result.cv.folds:
        try:
            written["roc"] = rp.plot_roc_curves([result.cv], out_dir / "roc_curves.png")
        except (ValueError, ImportError) as exc:
            result.report.add("warning", "roc_plot_failed", str(exc))
    if result.importance is not None:
        try:
            written["importance_plot"] = rp.plot_feature_importance(
                result.importance, out_dir / "feature_importance.png"
            )
        except (ValueError, ImportError) as exc:
            result.report.add("warning", "imp_plot_failed", str(exc))
    progress("export", 6, 8)
    written["validation"] = rp.write_validation_report(result.report, out_dir / "validation_report.txt")
    extra = {
        "timing_seconds": {k: round(v, 2) for k, v in result.timing.items()},
        "n_variants": int(len(result.variants)),
        "n_candidates": int(len(result.candidates)),
        "n_positive_labels": result.n_positives,
        "cv_summary": result.cv.summary() if result.cv is not None else {},
        # Fixed PU-bagging hyperparameter (not user-configurable by design).
        "pu_n_bags": config.DEFAULT_PU_N_BAGS,
        "settings_hash": getattr(result.settings, "cache_key", ""),
        # Coloc provenance.
        "use_coloc_features": getattr(result.settings, "use_coloc_features", False),
        "coloc_pp4_threshold": getattr(result.settings, "coloc_pp4_threshold", None),
        "n_enrichment_rows": int(len(result.enrichment_table)),
        "n_external_benchmark_rows": int(len(result.external_benchmark_table)),
        # Pathway + drug-target validation.
        "n_pathway_rows": int(len(result.pathway_table)),
        "drug_target_overlap": result.drug_overlap_summary,
    }
    written["provenance"] = rp.write_provenance(
        result.settings, result.report.summary(), extra, out_dir / "provenance.json"
    )
    progress("export", 8, 8)
    return written

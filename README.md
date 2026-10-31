# Hidden Locus

A local desktop research tool that ranks disease-associated **non-coding
variant-to-gene (V2G) relationships** using free public datasets, classical
machine learning and leakage-aware evaluation.

**Default research scope:** Crohn's disease / inflammatory bowel disease
(IBD) on **GRCh38**.

[![tests](https://github.com/JiyaMehta-6/hidden-locus/actions/workflows/tests.yml/badge.svg)](https://github.com/JiyaMehta-6/hidden-locus/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.14%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-GPL--3.0-2ea44f.svg)](LICENSE)

> **Research use only.** Hidden Locus produces ranked hypotheses for research
> prioritisation. It makes no clinical claims, provides no diagnostic or
> treatment guidance, and its scores must not be used for medical decisions.

---

## What it does

1. **Selects study loci** — pulls genome-wide significant index variants for
   the IBD trait scope from the NHGRI-EBI GWAS Catalog, deduplicates them
   across studies and clusters them into loci (500 kb gap by default).
2. **Generates candidate pairs** — every protein-coding / lncRNA gene whose
   transcription start site lies within ±1 Mb of an index variant becomes a
   candidate (variant, gene) pair.
3. **Extracts evidence features** per candidate:
   - distance to TSS (signed and gene-body), predicted consequence class
     (exon / intron / promoter / proximal / distal),
   - overlap with ENCODE SCREEN candidate *cis*-regulatory elements,
   - tissue-specific GTEx v8 expression-QTL evidence (whole blood, small
     intestine — terminal ileum, transverse colon),
   - optional fine-mapping PIPs, if the user supplies a PIP cache file.
4. **Builds positive labels from an independent study** — a candidate pair is
   labelled positive when the variant-gene pair is significant
   (p ≤ 1×10⁻⁵, default) in the BLUEPRINT monocyte eQTL dataset (eQTL
   Catalogue, QTD000021) **and** the gene is an eGene there (permutation
   p ≤ 0.05). GTEx data are used **only as features, never as labels**, so
   label evidence and feature evidence never come from the same study.
5. **Trains and evaluates a ranking model** — XGBoost (or logistic
   regression) with optional positive-unlabelled (PU) bagging, evaluated with
   **locus-grouped cross-validation** (no locus appears in both train and
   test folds). Explicit baselines are reported: nearest gene, logistic
   regression, and — when available locally — Open Targets locus-to-gene
   (L2G) scores as an *external comparator only* (never as labels).
6. **Explains and exports** — global feature importance, per-row
   contributions, ROC curves, metric tables, ranked candidate CSV, and a
   provenance bundle recording datasets, settings and validation issues.

Unlabelled pairs are **not** treated as confirmed negatives; the PU learner
exists precisely because of this.

---

## Screenshots

![Analysis tab: research scope, model settings and the run workflow](docs/images/gui_analysis.png)

| Results — ranked candidates | Data — opt-in dataset manager |
|---|---|
| ![Ranked candidates table with model scores](docs/images/gui_results_ranked.png) | ![Dataset status table with present and missing entries](docs/images/gui_data.png) |

![ROC curves per locus-grouped fold inside the app](docs/images/gui_results_roc.png)

*(Captured offline from a saved reference report with
`scripts/make_readme_assets.py`; the app ships its own dark theme, so the
captures do not depend on the operating-system palette.)*

---

## Install

Requirements: Python ≥ 3.14 with a virtual environment (this repository was
developed against an existing uv-managed `.venv`).

```powershell
# from the project root
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Or with plain `pip` in any Python ≥ 3.14 environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the desktop application:

```powershell
.venv\Scripts\python.exe main.py
# or
.venv\Scripts\python.exe -m hidden_locus.app
```

Run the tests:

```powershell
.venv\Scripts\python.exe -m pytest -q               # full suite (106 tests)
.venv\Scripts\python.exe -m pytest -q -m "not slow" # fast subset (~1 min)
```

---

## Data (opt-in downloads)

Nothing is downloaded until you ask for it — use the **Data** tab in the
application, or the CLI:

```powershell
.venv\Scripts\python.exe -m hidden_locus.core.datasets --list
.venv\Scripts\python.exe -m hidden_locus.core.datasets --fetch gwas_catalog
.venv\Scripts\python.exe -m hidden_locus.core.datasets --fetch-all
.venv\Scripts\python.exe -m hidden_locus.core.datasets --verify
```

| Key | Resource | Size (approx.) | Used for |
|---|---|---|---|
| `gwas_catalog` | GWAS Catalog associations (ontology annotated) | 73 MB | index variants, loci |
| `gene_annotation` | GENCODE v26 (GRCh38, GTEx reference) | 134 MB | genes, TSS, exons |
| `encode_screen` | ENCODE SCREEN cCRE registry (GRCh38) | 64 MB | regulatory overlap |
| `gtex_eqtl` | GTEx v8 significant cis-eQTL pairs (tar) | 1.5 GB | tissue eQTL features |
| `label_eqtl` | eQTL Catalogue BLUEPRINT monocytes (QTD000021 `.all`) | 2.8 GB | positive labels |
| `label_eqtl_permuted` | BLUEPRINT permutation results | 0.6 MB | eGene filter for labels |
| `opentargets_l2g` | Open Targets L2G scores (optional) | ~70 MB | external comparator |
| `opentargets_credible_set` | Open Targets GWAS credible sets (optional) | ~2.5 GB | fine-mapping PIPs, L2G joins |
| `abc_model` | ENCODE ABC Model enhancer–gene predictions (optional) | ~500 MB | distal regulatory links |
| `gut_cell_atlas_eqtl` | Gut Cell Atlas cell-type-specific eQTL (optional) | ~2 GB | cell-type eQTL features |
| `clinvar` | ClinVar GRCh38 VCF (optional) | ~200 MB | variant context |

Every downloaded file is recorded in `data/manifest.json` with source URL,
size, retrieval date, licence note and assembly. Datasets are stored under
`data/datasets/` (override the root with the `HIDDEN_LOCUS_DATA_DIR`
environment variable).

**Licence notes:** GWAS Catalog — EMBL-EBI terms of use, summary statistics
CC0; GTEx v8 — GTEx Portal data licence (dbGaP phs000424); ENCODE — free for
unrestricted use; eQTL Catalogue — CC BY 4.0; GENCODE — freely available per
gencodegenes.org; Open Targets Platform outputs — CC0.

### Optional cache files (never auto-downloaded)

- `data/cache/fine_mapping_pips.csv` — columns `variant_uid` (or
  `chrom`,`pos`) and `pip`. If absent, the PIP feature is marked missing.
- `data/cache/opentargets_l2g.csv` — columns `chrom`, `pos`, `gene_id`,
  `l2g_score`. If absent, the external comparator is omitted.

---

## Using the application

1. **Data tab** — select datasets, *Fetch selected* / *Fetch all required*,
   then *Verify local*.
2. **Analysis tab** — review the scope (tissues, window, locus gap, label
   threshold, model, CV folds, seed) and press **Run analysis**.
3. **Results tab** — ranked candidates, baseline metrics, ablation table,
   feature importance, ROC figure. *Export report* writes
   `data/reports/<timestamp>/` containing:
   - `ranked_candidates.csv`, `metrics.csv`, `ablations.csv`,
     `feature_importance.csv`, `top_row_contributions.csv`
   - `roc_curves.png`, `feature_importance.png`
   - `validation_report.txt`, `provenance.json`
4. **Validation / Log tab** — every validation issue (errors, warnings,
   provenance notes) and the live log.

---

## Results (reference run)

Figures and numbers below come from the saved reference report
`data/reports/20261016_174459`: IBD scope, GRCh38, **49,110 candidate
pairs**, 5-fold locus-grouped CV, seed 42.

![ROC curves by locus-grouped fold](docs/images/roc_curves.png)

| Model | ROC AUC | Avg. precision |
|---|---|---|
| Logistic regression + PU bagging | **0.927 ± 0.024** | 0.316 |
| Logistic regression | 0.923 ± 0.027 | 0.331 |
| XGBoost + PU bagging | 0.916 ± 0.019 | 0.277 |
| XGBoost | 0.911 ± 0.020 | 0.282 |
| Nearest-gene baseline | 0.599 | 0.035 |

Positive labels come only from the BLUEPRINT monocyte eQTL study (never
from GTEx, which is feature-only). Open Targets L2G scores are exported
separately in `external_benchmark.csv` as an external comparator — a
different task definition, never a training signal.

**Feature-group ablations** (XGBoost, 5-fold) — which evidence actually
carries the ranking:

| Feature group removed | ROC AUC | Δ vs full |
|---|---|---|
| *full model* | 0.916 | — |
| – GTEx tissue eQTL | 0.852 | −0.064 |
| – structural (distance to TSS) | 0.859 | −0.057 |
| – colocalisation | 0.896 | −0.020 |
| – cCRE overlap | 0.916 | −0.001 |
| – consequence class | 0.916 | ±0.000 |
| – fine-mapping PIPs | 0.919 | +0.002 |

![Global feature importance (gain)](docs/images/feature_importance.png)

**Generalisation across traits** — the same pipeline retrained per trait
(`data/reports/multitrait_benchmark.csv`):

![Multi-trait benchmark vs nearest-gene baseline](docs/images/multitrait_benchmark.png)

**Independent enrichment check** — top-ranked genes against curated IBD
genes: **16.3× enrichment at the top 100** (Fisher p = 2.2 × 10⁻⁸) and
**11.4× at the top 250** (p = 5.0 × 10⁻¹²).

Reproduce by fetching the datasets above and pressing *Run analysis*;
`scripts/make_readme_assets.py` regenerates these images offline from a
saved report.

---

## Evaluation design (anti-leakage rules)

- Splits are **grouped by locus** (`GroupKFold` on `locus_id`); a genomic
  region never straddles train and test.
- Identifier columns (`variant_uid`, `gene_id`, coordinates, p-values of the
  label study, …) are asserted to be absent from the feature list
  (`assert_no_leakage`).
- Label evidence comes from a study **independent** of the GTEx feature
  evidence.
- Open Targets L2G scores are a comparator, never training signal.
- Baselines reported: nearest-TSS gene, plain logistic regression, PU bagging
  ablation, plus feature-group ablations (structural / consequence / cCRE /
  GTEx / fine-mapping).
- PU bagging hyperparameters (20 bags, seed 42 default) are fixed and
  recorded in `provenance.json`.

### Interpreting scores

`model_score` is a model ranking output, **not** a calibrated probability of
causality. Positive labels are study-specific associations; everything
unlabelled is unlabelled. Treat results as prioritisation for follow-up
(e.g. literature review, experimental validation), not as proof.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q               # full suite (106 tests)
.venv\Scripts\python.exe -m pytest -q -m "not slow" # fast subset (~1 min)
```

The suite covers interval math, allele harmonisation, GWAS locus
construction, feature building, leakage guards, PU training, grouped
evaluation, baselines and explainability — all on synthetic data (no
network access). The four long-running benchmark tests carry the `slow`
marker; the rest finish in about a minute. Continuous integration runs
the full suite on every push (`.github/workflows/tests.yml`).

---

## Known limitations

- GRCh38 only; other assemblies must be lifted over before import.
- Labels depend on a single independent eQTL study (BLUEPRINT monocytes);
  immune-cell context is relevant for IBD but incomplete.
- GTEx significant-pair files only contain pairs passing GTEx's own
  thresholds; absent pairs are missing evidence, not evidence of absence.
- No causal inference: GWAS association ≠ causal variant, eQTL association ≠
  causal gene.
- Fine-mapping PIPs and Open Targets scores are optional local inputs; the
  pipeline runs without them (features/comparators marked missing).

---

## References

Methods and datasets used by this tool:

- Mountjoy E, *et al.* An open approach to systematically prioritize causal
  variants and genes at all published human GWAS trait-associated loci.
  *Nature Genetics* 53, 1527–1533 (2021).
  doi:[10.1038/s41588-021-00945-5](https://doi.org/10.1038/s41588-021-00945-5)
  (L2G — external comparator scores).
- Weeks EM, *et al.* Leveraging polygenic enrichments of gene features to
  predict genes underlying complex traits and diseases.
  *Nature Genetics* 55, 1267–1276 (2023).
  doi:[10.1038/s41588-023-01443-6](https://doi.org/10.1038/s41588-023-01443-6)
  (PoPS — related gene-prioritisation approach discussed in the design).
- Wen X, Pique-Regi R, Luca F. Integrating molecular QTL data into
  genome-wide genetic association analysis: probabilistic assessment of
  enrichment and colocalization. *PLoS Genetics* 13(3): e1006646 (2017).
  doi:[10.1371/journal.pgen.1006646](https://doi.org/10.1371/journal.pgen.1006646)
  (colocalisation-based evidence integration concept).
- The GTEx Consortium. The Genotype-Tissue Expression (GTEx) project.
  *Science* 369(6509):1318–1330 (2020).
  doi:[10.1126/science.aaz1776](https://doi.org/10.1126/science.aaz1776)

Public data resources (accessed via the Data tab):

- NHGRI-EBI GWAS Catalog — https://www.ebi.ac.uk/gwas/
- GTEx v8 — https://gtexportal.org/
- ENCODE SCREEN — https://screen.encodeproject.org/
- GENCODE — https://www.gencodegenes.org/
- eQTL Catalogue — https://www.ebi.ac.uk/eqtl/
- Open Targets Platform — https://platform.opentargets.org/

No other external tools are required, and no paid APIs or accounts are used.

---

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
The datasets above remain under their own terms (see *Licence notes* in
the Data section).

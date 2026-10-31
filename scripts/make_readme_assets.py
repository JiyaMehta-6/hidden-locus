"""Generate the README image assets, fully offline.

Produces in ``docs/images/``:

- ``gui_*.png``       screenshots of the desktop application, populated from
                      the latest saved report (same data a completed run
                      would display),
- ``roc_curves.png`` / ``feature_importance.png``
                      copies of the figures from the reference report,
- ``multitrait_benchmark.png``
                      bar chart of the multi-trait benchmark CSV.

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts\\make_readme_assets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
REPORTS = ROOT / "data" / "reports"

# Columns the Results tab shows in the "Ranked candidates" preview.
PREVIEW_COLS = {
    "rank", "model_score", "locus_id", "variant_uid", "rsid",
    "gene_name", "gene_id", "consequence", "dist_tss", "ccre_class",
    "gtex_n_tissues", "label", "allele_match", "trait", "pubmed",
}

TRAIT_LABELS = {
    "ibd": "IBD / Crohn's",
    "t2d": "Type 2 diabetes",
    "ra": "Rheumatoid arthritis",
    "asthma": "Asthma",
}


def latest_report() -> Path:
    """Newest report directory that contains the reference figures."""
    candidates = [
        p for p in REPORTS.iterdir()
        if p.is_dir() and (p / "roc_curves.png").exists() and (p / "ranked_candidates.csv").exists()
    ]
    if not candidates:
        raise SystemExit("no saved report with figures found under data/reports/")
    return max(candidates, key=lambda p: p.name)


def gui_screenshots(report: Path) -> None:
    """Capture the application tabs with a saved report loaded."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from hidden_locus import config
    from hidden_locus.app.main_window import MainWindow
    from hidden_locus.app.theme import apply_theme
    from hidden_locus.core.pipeline import PipelineResult
    from hidden_locus.core.validation import ValidationReport

    ranked = pd.read_csv(
        report / "ranked_candidates.csv",
        usecols=lambda c: c in PREVIEW_COLS,
    )
    importance = pd.read_csv(report / "feature_importance.csv")
    series = (
        importance.drop_duplicates("feature")
        .set_index("feature")["importance"]
        .sort_values(ascending=False)
    )
    result = PipelineResult(
        report=ValidationReport(),
        settings=config.Settings(),
        ranked=ranked,
        baseline_table=pd.read_csv(report / "metrics.csv"),
        ablation_table=pd.read_csv(report / "ablations.csv"),
        importance=series,
    )

    config.ensure_data_dirs()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    # Render offscreen: a real window is clamped to the desktop work area,
    # which would shrink the captures on small displays.
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1500, 900)
    window.show()
    window._last_result = result
    window.results_panel.show_result({
        "result": result,
        "written": {"roc": str(report / "roc_curves.png")},
    })

    # (file, main tab index, results sub-tab index)
    shots = [
        ("gui_analysis.png", 0, None),
        ("gui_data.png", 1, None),
        ("gui_results_ranked.png", 2, 0),
        ("gui_results_metrics.png", 2, 1),
        ("gui_results_roc.png", 2, 4),
    ]
    for name, tab, sub in shots:
        window.tabs.setCurrentIndex(tab)
        if sub is not None:
            window.results_panel.tabs.setCurrentIndex(sub)
        for _ in range(3):
            app.processEvents()
        if not window.grab().save(str(IMAGES / name)):
            raise RuntimeError(f"could not save {IMAGES / name}")
        print(f"wrote docs/images/{name}")
    window.close()
    app.processEvents()


def copy_reference_figures(report: Path) -> None:
    for name in ("roc_curves.png", "feature_importance.png"):
        src = report / name
        if src.exists():
            shutil.copy2(src, IMAGES / name)
            print(f"wrote docs/images/{name} (from {report.name})")


def multitrait_chart() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = REPORTS / "multitrait_benchmark.csv"
    df = pd.read_csv(path)
    labels = [TRAIT_LABELS.get(t, t) for t in df["trait"]]
    x = range(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=160)
    b1 = ax.bar([i - width / 2 for i in x], df["roc_auc"], width,
                label="Hidden Locus model", color="#2b6cb0")
    b2 = ax.bar([i + width / 2 for i in x], df["nearest_gene_auc"], width,
                label="Nearest-gene baseline", color="#a0aec0")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ROC AUC (locus-grouped CV)")
    ax.set_title("Multi-trait benchmark vs nearest-gene baseline")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "multitrait_benchmark.png")
    plt.close(fig)
    print("wrote docs/images/multitrait_benchmark.png")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    report = latest_report()
    print(f"reference report: {report.name}")
    copy_reference_figures(report)
    multitrait_chart()
    gui_screenshots(report)


if __name__ == "__main__":
    main()

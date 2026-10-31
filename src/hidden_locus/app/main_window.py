"""Main window for the Hidden Locus desktop application.

Tabs: Data manager, Analysis, Results, Validation log.  Long operations run
in :class:`~hidden_locus.app.workers` on the global thread pool; widgets only
receive signals.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, QThreadPool
from PySide6.QtGui import QAction, QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..core import datasets as ds
from ..core.evaluation import CVResult
from ..core.pipeline import PipelineResult
from . import theme
from .workers import DownloadWorker, ExportWorker, PipelineWorker, WorkerSignals


def _finite(value: object) -> "float | None":
    """Return *value* as a finite float, or ``None`` for missing input."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class PandasModel(QAbstractTableModel):
    """Read-only table model exposing a pandas DataFrame to QTableView.

    ``model_score`` cells carry an accent heat tint and positive ``label``
    cells are highlighted green so the ranked list reads at a glance.
    """

    _SCORE_LOW = QColor(47, 111, 237, 26)
    _SCORE_HIGH = QColor(47, 111, 237, 200)
    _LABEL_BG = QColor(22, 101, 52, 130)
    _LABEL_FG = QColor(74, 222, 128)

    def __init__(self, frame: "pd.DataFrame | None" = None) -> None:
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()
        self._score_min = 0.0
        self._score_max = 1.0

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True)
        self._score_min, self._score_max = 0.0, 1.0
        if "model_score" in self.frame.columns:
            scores = pd.to_numeric(self.frame["model_score"], errors="coerce").dropna()
            if not scores.empty:
                self._score_min = float(scores.min())
                self._score_max = float(scores.max())
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or index.column() >= len(self.frame.columns):
            return None
        name = str(self.frame.columns[index.column()])
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)
        if name == "model_score" and role == Qt.ItemDataRole.BackgroundRole:
            score = _finite(value)
            if score is None:
                return None
            span = self._score_max - self._score_min
            t = 1.0 if span <= 0 else (score - self._score_min) / span
            low, high = self._SCORE_LOW.alpha(), self._SCORE_HIGH.alpha()
            color = QColor(self._SCORE_LOW)
            color.setAlpha(int(low + t * (high - low)))
            return color
        if name == "label" and str(value) == "1":
            if role == Qt.ItemDataRole.BackgroundRole:
                return self._LABEL_BG
            if role == Qt.ItemDataRole.ForegroundRole:
                return self._LABEL_FG
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)


class DataPanel(QWidget):
    """Dataset status, fetch controls and download progress."""

    def __init__(self, parent: "MainWindow | None" = None) -> None:
        super().__init__()
        self.main = parent
        self.signals = WorkerSignals()
        self._download_worker: "DownloadWorker | None" = None

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch selected")
        self.fetch_all_btn = QPushButton("Fetch all required")
        self.verify_btn = QPushButton("Verify local")
        self.refresh_btn = QPushButton("Refresh")
        top.addWidget(self.fetch_btn)
        top.addWidget(self.fetch_all_btn)
        top.addWidget(self.verify_btn)
        top.addWidget(self.refresh_btn)
        top.addStretch(1)
        self.summary = QLabel("")
        self.summary.setObjectName("mutedNote")
        top.addWidget(self.summary)
        layout.addLayout(top)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Key", "Name", "State", "Bytes", "Size", "Required", "Verified"]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log, 1)

        self.fetch_btn.clicked.connect(self._fetch_selected)
        self.fetch_all_btn.clicked.connect(self._fetch_all)
        self.verify_btn.clicked.connect(self._verify)
        self.refresh_btn.clicked.connect(self.refresh)
        self.signals.stage.connect(self._on_stage)
        self.signals.log.connect(self._append)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)

        self.refresh()

    def refresh(self) -> None:
        statuses = ds.statuses()
        self.table.setRowCount(len(statuses))
        present = 0
        total_bytes = 0
        missing_required = 0
        for row, st in enumerate(statuses):
            if st["state"] == "present":
                present += 1
                total_bytes += int(st["bytes"] or 0)
            elif st["required"]:
                missing_required += 1
            values = [
                st["key"], st["name"], st["state"], f"{st['bytes']:,}",
                st["approx_size"], "yes" if st["required"] else "no",
                "yes" if st["verified"] else "no",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 2:
                    state = st["state"]
                    if state == "present":
                        item.setForeground(QColor(theme.GOOD))
                    elif st["required"]:
                        item.setForeground(QColor(theme.WARN))
                    else:
                        item.setForeground(QColor(theme.MUTED))
                elif col == 6 and st["verified"]:
                    item.setForeground(QColor(theme.GOOD))
                self.table.setItem(row, col, item)
        size = f"{total_bytes / 1e9:.1f} GB" if total_bytes >= 1e9 else f"{total_bytes / 1e6:.0f} MB"
        note = (
            f"{present} / {len(statuses)} datasets present · {size} on disk"
        )
        if missing_required:
            note += f" · {missing_required} required missing"
        self.summary.setText(note)

    def _selected_keys(self) -> list[str]:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        statuses = ds.statuses()
        return [statuses[r]["key"] for r in rows if 0 <= r < len(statuses)]

    def _fetch_selected(self) -> None:
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(self, "Hidden Locus", "Select one or more datasets first.")
            return
        self._start_download(keys)

    def _fetch_all(self) -> None:
        self._start_download([d.key for d in ds.DATASETS if d.required])

    def _start_download(self, keys: list[str]) -> None:
        if self._download_worker is not None:
            QMessageBox.information(self, "Hidden Locus", "A download is already running.")
            return
        worker = DownloadWorker(keys)
        worker.signals.stage.connect(self._on_stage)
        worker.signals.log.connect(self._append)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._download_worker = worker
        self.fetch_btn.setEnabled(False)
        self.fetch_all_btn.setEnabled(False)
        QThreadPool.globalInstance().start(worker)
        self._append(f"Started download: {', '.join(keys)}")

    def _verify(self) -> None:
        for spec in ds.DATASETS:
            result = ds.verify(spec.key)
            self._append(f"{spec.key}: {result['state']}")
        self.refresh()

    def _on_stage(self, key: str, done: int, total: int) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(min(done, total))

    def _append(self, message: str) -> None:
        self.log.append(message)

    def _on_finished(self, payload: object) -> None:
        self._download_worker = None
        self.fetch_btn.setEnabled(True)
        self.fetch_all_btn.setEnabled(True)
        self.refresh()
        self._append("Download finished.")

    def _on_error(self, message: str) -> None:
        self._download_worker = None
        self.fetch_btn.setEnabled(True)
        self.fetch_all_btn.setEnabled(True)
        self._append(f"ERROR:\n{message}")


class AnalysisPanel(QWidget):
    """Settings form plus the pipeline run control."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__()
        self.main = parent
        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        right = QVBoxLayout()
        right.setSpacing(10)

        form_box = QGroupBox("Research scope and settings")
        form = QFormLayout(form_box)

        self.disease = QLineEdit(config.Settings().disease_query)
        self.disease.setReadOnly(True)
        form.addRow("Disease scope", self.disease)

        self.tissues = QLineEdit(", ".join(config.Settings().tissues))
        form.addRow("GTEx tissues (comma)", self.tissues)

        self.window = QSpinBox()
        self.window.setRange(10_000, 5_000_000)
        self.window.setSingleStep(100_000)
        self.window.setValue(config.Settings().window_bp)
        self.window.setSuffix(" bp")
        form.addRow("Candidate window (±TSS)", self.window)

        self.gap = QSpinBox()
        self.gap.setRange(10_000, 5_000_000)
        self.gap.setSingleStep(50_000)
        self.gap.setValue(config.Settings().locus_gap_bp)
        self.gap.setSuffix(" bp")
        form.addRow("Locus gap", self.gap)

        self.label_p = QDoubleSpinBox()
        # 12 decimals so the full range down to the 1e-12 minimum is
        # representable (with 10, values below 1e-10 rounded to 0).
        self.label_p.setDecimals(12)
        self.label_p.setRange(1e-12, 1e-2)
        self.label_p.setValue(config.Settings().label_p_threshold)
        form.addRow("Label p threshold", self.label_p)

        self.model = QComboBox()
        self.model.addItems(["xgboost", "logreg"])
        form.addRow("Model", self.model)

        self.use_pu = QCheckBox("Positive-unlabelled bagging")
        self.use_pu.setChecked(config.Settings().use_pu)
        form.addRow("", self.use_pu)

        self.n_splits = QSpinBox()
        self.n_splits.setRange(2, 20)
        self.n_splits.setValue(config.Settings().n_splits)
        form.addRow("Grouped CV folds", self.n_splits)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(config.Settings().seed)
        form.addRow("Random seed", self.seed)

        left.addWidget(form_box)

        hint = QLabel(
            "Loci  →  candidate pairs  →  features (eQTL · cCRE · coloc · ABC · gut atlas)"
            "  →  locus-grouped CV  →  ranked report"
        )
        hint.setObjectName("stageHint")
        hint.setWordWrap(True)
        left.addWidget(hint)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.setMinimumHeight(36)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch(1)
        left.addLayout(btn_row)

        self.stage_label = QLabel("Idle.")
        self.stage_label.setObjectName("mutedNote")
        left.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left.addWidget(self.progress)

        note = QLabel(
            "Runs entirely on local data — downloads are opt-in on the Data tab."
        )
        note.setObjectName("mutedNote")
        left.addWidget(note)
        left.addStretch(1)

        right.addWidget(self._build_info_card())
        right.addStretch(1)

        body.addLayout(left, 3)
        body.addLayout(right, 2)
        layout.addLayout(body)
        form_box.setMaximumWidth(760)
        self.run_btn.clicked.connect(self._run)

    @staticmethod
    def _build_info_card() -> QFrame:
        """Static explainer describing what pressing *Run analysis* does."""
        card = QFrame()
        card.setObjectName("infoCard")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(16, 14, 16, 14)
        inner.setSpacing(10)

        title = QLabel("What a run does")
        title.setObjectName("infoTitle")
        inner.addWidget(title)
        steps = [
            ("1", "Select loci",
             "Genome-wide significant IBD index variants from the GWAS Catalog, "
             "clustered into independent loci."),
            ("2", "Build candidate pairs",
             "Every gene inside the candidate window of a lead variant becomes a "
             "variant–gene pair."),
            ("3", "Assemble evidence",
             "eQTL, cCRE overlap, colocalisation, ABC links, gut-atlas eQTL and "
             "consequence class."),
            ("4", "Validate honestly",
             "Locus-grouped CV keeps held-out loci fully unseen; nearest-gene and "
             "Open Targets are comparators, not training signal."),
            ("5", "Rank and report",
             "Scores every candidate and exports tables, ablations, ROC and a "
             "provenance bundle."),
        ]
        for number, heading, text in steps:
            row = QHBoxLayout()
            row.setSpacing(10)
            badge = QLabel(number)
            badge.setObjectName("stepBadge")
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
            column = QVBoxLayout()
            column.setSpacing(1)
            head = QLabel(heading)
            head.setObjectName("stepTitle")
            body = QLabel(text)
            body.setObjectName("stepBody")
            body.setWordWrap(True)
            column.addWidget(head)
            column.addWidget(body)
            row.addLayout(column, 1)
            inner.addLayout(row)
        return card

    def collect_settings(self) -> config.Settings:
        settings = config.Settings.load()
        settings.tissues = [t.strip() for t in self.tissues.text().split(",") if t.strip()]
        settings.window_bp = int(self.window.value())
        settings.locus_gap_bp = int(self.gap.value())
        settings.label_p_threshold = float(self.label_p.value())
        settings.model_name = self.model.currentText()
        settings.use_pu = self.use_pu.isChecked()
        settings.n_splits = int(self.n_splits.value())
        settings.seed = int(self.seed.value())
        settings.save()
        return settings

    def _run(self) -> None:
        self.main.start_analysis(self.collect_settings())

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)


class FigureView(QWidget):
    """Centre a generated figure in a framed card, scaled to fit."""

    _CARD_QSS = (
        "background: #ffffff; border: 1px solid #2b3442;"
        " border-radius: 8px; padding: 8px;"
    )

    def __init__(self) -> None:
        super().__init__()
        self._source: "QPixmap | None" = None
        self.label = QLabel("Run the analysis to generate figures.")
        self.label.setObjectName("plotEmpty")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 150)

    def set_image(self, pixmap: "QPixmap | None") -> None:
        """Show *pixmap* scaled to the current viewport (``None`` clears it)."""
        self._source = None if pixmap is None or pixmap.isNull() else pixmap
        if self._source is None:
            self.label.setPixmap(QPixmap())
            self.label.setStyleSheet("")
            self.label.setText("No ROC figure produced.")
            self.label.setFixedSize(self.label.sizeHint())
            return
        self.label.setText("")
        self.label.setStyleSheet(self._CARD_QSS)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None:
            return
        available = self.contentsRect().size() - QSize(36, 36)
        if available.width() < 80 or available.height() < 80:
            return
        scaled = self._source.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
        self.label.setFixedSize(scaled.width() + 16, scaled.height() + 16)


class ResultsPanel(QWidget):
    """Ranked candidates, metrics and plots."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__()
        self.main = parent
        self._export_worker: "ExportWorker | None" = None
        layout = QVBoxLayout(self)

        self.cards = QHBoxLayout()
        self.cards.setSpacing(10)
        self.auc_value, self.auc_sub = self._add_card(
            self.cards, "ROC AUC", theme.ACCENT_LIGHT
        )
        self.candidates_value, self.candidates_sub = self._add_card(
            self.cards, "Candidates ranked", theme.TEXT
        )
        self.labels_value, self.labels_sub = self._add_card(
            self.cards, "Positive labels", theme.GOOD
        )
        self.gene_value, self.gene_sub = self._add_card(
            self.cards, "Top-ranked gene", theme.WARN
        )
        layout.addLayout(self.cards)

        self.tabs = QTabWidget()

        self.ranked_view = QTableView()
        self.ranked_model = PandasModel()
        self.ranked_view.setModel(self.ranked_model)
        self.ranked_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.tabs.addTab(self.ranked_view, "Ranked candidates")

        self.metrics_table = QTableWidget()
        self.tabs.addTab(self.metrics_table, "Metrics")

        self.ablation_table = QTableWidget()
        self.tabs.addTab(self.ablation_table, "Ablations")

        self.importance_table = QTableWidget()
        self.tabs.addTab(self.importance_table, "Feature importance")

        self.plot_view = FigureView()
        self.tabs.addTab(self.plot_view, "ROC figure")

        layout.addWidget(self.tabs)
        self.export_btn = QPushButton("Export report to data/reports/ ...")
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self._export)

    def _export(self) -> None:
        result = self.main._last_result
        if result is None:
            return
        if self._export_worker is not None:
            return
        worker = ExportWorker(result)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self._export_worker = worker
        self.export_btn.setEnabled(False)
        self.main.validation_panel.append_log("Re-exporting report...")
        self.main.thread_pool.start(worker)

    def _on_export_finished(self, payload: object) -> None:
        self._export_worker = None
        self.export_btn.setEnabled(True)
        assert isinstance(payload, dict)
        written = payload.get("written") or {}
        paths = ", ".join(str(p) for p in written.values())
        self.main.validation_panel.append_log(f"Report written: {paths}")
        self.main.statusBar().showMessage("Report exported.")

    def _on_export_error(self, message: str) -> None:
        self._export_worker = None
        self.export_btn.setEnabled(True)
        self.main.validation_panel.append_log(f"EXPORT ERROR:\n{message}")
        self.main.statusBar().showMessage("Export failed — see Validation / Log tab.")

    def show_result(self, payload: dict) -> None:
        result: PipelineResult = payload["result"]
        ranked = result.ranked
        self._update_summary(result)
        preview_cols = [
            c for c in (
                "rank", "model_score", "locus_id", "variant_uid", "rsid",
                "gene_name", "gene_id", "consequence", "dist_tss",
                "ccre_class", "gtex_n_tissues", "label", "allele_match",
                "trait", "pubmed",
            ) if c in ranked.columns
        ]
        if preview_cols:
            self.ranked_model.set_frame(ranked[preview_cols])
        self._fill_table(self.metrics_table, result.baseline_table)
        self._fill_table(self.ablation_table, result.ablation_table)
        if result.importance is not None:
            self._fill_table(self.importance_table, result.importance.to_frame().head(40))
        self._show_plot(payload["written"])
        self.export_btn.setEnabled(True)

    @staticmethod
    def _add_card(row: QHBoxLayout, caption: str, color: str):
        """Append a KPI card to *row*; return its value and sub-line labels."""
        card = QFrame()
        card.setObjectName("statCard")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 10, 14, 10)
        inner.setSpacing(2)

        title = QLabel(caption.upper())
        title.setObjectName("statCaption")
        value = QLabel("—")
        value.setObjectName("statValue")
        value.setStyleSheet(f"color: {color};")
        sub = QLabel("")
        sub.setObjectName("statSub")
        for label in (title, value, sub):
            inner.addWidget(label)
        row.addWidget(card, 1)
        return value, sub

    def _update_summary(self, result: PipelineResult) -> None:
        """Fill the KPI cards from a completed pipeline result."""
        baseline = result.baseline_table
        auc = None
        delta = None
        if baseline is not None and not baseline.empty and "model" in baseline.columns:
            by_model = baseline.set_index("model")["mean_roc_auc"]
            wanted = getattr(result.settings, "model_name", "")
            if "use_pu" in baseline.columns:
                pu_rows = baseline[baseline["use_pu"].astype(str).isin(["True", "1"])]
                if not pu_rows.empty:
                    by_model = pu_rows.set_index("model")["mean_roc_auc"]
            if wanted in by_model.index:
                auc = _finite(by_model.loc[wanted])
            elif not by_model.empty:
                auc = _finite(by_model.iloc[0])
            if "nearest_gene" in by_model.index:
                delta = _finite(by_model.loc["nearest_gene"])
        if auc is not None:
            self.auc_value.setText(f"{auc:.3f}")
            if delta:
                lift = auc - delta
                self.auc_sub.setText(
                    f"+{lift:.2f} vs nearest gene ({delta:.2f})"
                    if lift >= 0
                    else f"{lift:.2f} vs nearest gene ({delta:.2f})"
                )
            else:
                self.auc_sub.setText("locus-grouped CV")
        else:
            self.auc_value.setText("—")
            self.auc_sub.setText("no metrics")

        ranked = result.ranked
        n_rows = 0 if ranked is None else len(ranked)
        loci = (
            ranked["locus_id"].nunique()
            if ranked is not None and "locus_id" in ranked.columns
            else 0
        )
        self.candidates_value.setText(f"{n_rows:,}")
        self.candidates_sub.setText(f"across {loci:,} loci" if loci else "")

        positives = None
        if ranked is not None and "label" in ranked.columns:
            labelled = pd.to_numeric(ranked["label"], errors="coerce").fillna(0)
            positives = int((labelled > 0).sum())
        if positives is None:
            positives = int(result.n_positives)
        self.labels_value.setText(f"{positives:,}")
        self.labels_sub.setText("BLUEPRINT eQTL labels")

        if ranked is not None and not ranked.empty and "gene_name" in ranked.columns:
            top = ranked.iloc[0]
            gene = str(top.get("gene_name", "—"))
            self.gene_value.setText(gene if gene and gene != "nan" else "—")
            details = []
            score = _finite(top.get("model_score"))
            if score is not None:
                details.append(f"score {score:.3f}")
            rsid = str(top.get("rsid", ""))
            if rsid and rsid != "nan":
                details.append(rsid)
            self.gene_sub.setText(" · ".join(details))
        else:
            self.gene_value.setText("—")
            self.gene_sub.setText("")

    @staticmethod
    def _fill_table(widget: QTableWidget, frame: "pd.DataFrame | None") -> None:
        if frame is None or frame.empty:
            widget.setRowCount(0)
            widget.setColumnCount(0)
            return
        widget.setColumnCount(len(frame.columns))
        widget.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        widget.setRowCount(len(frame))
        for r in range(len(frame)):
            for c, col in enumerate(frame.columns):
                value = frame.iat[r, c]
                if isinstance(value, float):
                    text = "" if pd.isna(value) else f"{value:.4g}"
                else:
                    text = str(value)
                widget.setItem(r, c, QTableWidgetItem(text))
        widget.resizeColumnsToContents()
        ResultsPanel._tint_metric_columns(widget, frame)

    @staticmethod
    def _tint_metric_columns(widget: QTableWidget, frame: pd.DataFrame) -> None:
        """Heat-tint score columns and highlight the best value in green."""
        for c, col in enumerate(frame.columns):
            name = str(col)
            if not name.endswith("roc_auc") and name not in {"roc_auc", "importance"}:
                continue
            numeric = pd.to_numeric(frame[col], errors="coerce")
            peak = numeric.max()
            if pd.isna(peak) or peak <= 0:
                continue
            for r, value in enumerate(numeric):
                if pd.isna(value):
                    continue
                item = widget.item(r, c)
                if item is None:
                    continue
                item.setForeground(
                    QColor(theme.GOOD) if value == peak else QColor(theme.TEXT)
                )
                tint = QColor(theme.ACCENT)
                tint.setAlpha(int(14 + 70 * (float(value) / float(peak))))
                item.setBackground(tint)

    def _show_plot(self, written: dict) -> None:
        roc = written.get("roc")
        if roc and Path(roc).exists():
            self.plot_view.set_image(QPixmap(str(roc)))
            return
        self.plot_view.set_image(None)


class ValidationPanel(QWidget):
    """Validation issues plus the live worker log."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.issues = QTableWidget(0, 4)
        self.issues.setHorizontalHeaderLabels(["Level", "Code", "Where", "Message"])
        self.issues.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.issues.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.issues, 3)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 2)

    def append_log(self, message: str) -> None:
        self.log.append(message)

    def show_report(self, report) -> None:
        self.issues.setRowCount(len(report.issues))
        colors = {
            "error": "#ff6b81",
            "warning": theme.WARN,
            "info": theme.MUTED,
        }
        for row, issue in enumerate(report.issues):
            for col, text in enumerate(
                [issue.level, issue.code, issue.where, issue.message]
            ):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(QColor(colors.get(issue.level, theme.TEXT)))
                self.issues.setItem(row, col, item)


class MainWindow(QMainWindow):
    """Top-level window wiring panels, thread pool and workers together."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{config.APP_NAME} {config.APP_VERSION}")
        self.resize(1240, 840)
        self.thread_pool = QThreadPool.globalInstance()
        self._pipeline_worker: "PipelineWorker | None" = None
        self._last_result: "PipelineResult | None" = None

        self.tabs = QTabWidget()
        self.data_panel = DataPanel(self)
        self.analysis_panel = AnalysisPanel(self)
        self.results_panel = ResultsPanel(self)
        self.validation_panel = ValidationPanel()

        self.tabs.addTab(self.analysis_panel, "Analysis")
        self.tabs.addTab(self.data_panel, "Data")
        self.tabs.addTab(self.results_panel, "Results")
        self.tabs.addTab(self.validation_panel, "Validation / Log")

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 8, 10, 8)
        body_layout.addWidget(self.tabs)
        outer.addWidget(body, 1)
        self.setCentralWidget(central)

        self._build_menu()
        self.statusBar().showMessage("Ready.")

    def _build_header(self) -> QFrame:
        """Application banner: product identity plus the active research scope."""
        header = QFrame()
        header.setObjectName("appHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(1)
        title = QLabel(config.APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "Variant-to-gene prioritisation · leakage-aware evaluation · local data only"
        )
        subtitle.setObjectName("appSubtitle")
        text.addWidget(title)
        text.addWidget(subtitle)
        row.addLayout(text)
        row.addStretch(1)

        settings = config.Settings.load()
        scope = QLabel(f"IBD · {config.SUPPORTED_ASSEMBLY} · v{config.APP_VERSION}")
        scope.setObjectName("appChip")
        chip = QLabel(settings.model_name + (" + PU" if settings.use_pu else ""))
        chip.setObjectName("appChip")
        row.addWidget(scope)
        row.addWidget(chip)
        return header

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._pipeline_worker is not None:
            answer = QMessageBox.question(
                self, config.APP_NAME,
                "An analysis is still running. Quit anyway? The running "
                "analysis will be aborted and no report will be saved.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            f"About {config.APP_NAME}",
            f"<b>{config.APP_NAME} {config.APP_VERSION}</b><br>"
            "Local research tool for ranking disease-associated non-coding "
            "variant-to-gene relationships.<br>"
            "Uses free public datasets (GWAS Catalog, GTEx, ENCODE SCREEN, "
            "GENCODE, eQTL Catalogue).<br><br>"
            "Research use only — not for clinical decision-making.",
        )

    # ---- pipeline wiring -------------------------------------------------
    def start_analysis(self, settings: config.Settings) -> None:
        if self._pipeline_worker is not None:
            QMessageBox.information(self, config.APP_NAME, "An analysis is already running.")
            return
        worker = PipelineWorker(settings)
        worker.signals.stage.connect(self._on_stage)
        worker.signals.log.connect(self._on_log)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._pipeline_worker = worker
        self.analysis_panel.set_running(True)
        self.validation_panel.append_log("Analysis started.")
        self.thread_pool.start(worker)

    def _on_stage(self, stage: str, done: int, total: int) -> None:
        self.analysis_panel.stage_label.setText(f"Stage: {stage} ({done}/{total})")
        if total > 0:
            self.analysis_panel.progress.setMaximum(max(total, 1))
            self.analysis_panel.progress.setValue(min(done, total))
        self.statusBar().showMessage(f"{stage}: {done}/{total}")

    def _on_log(self, message: str) -> None:
        self.validation_panel.append_log(message)

    def _on_finished(self, payload: object) -> None:
        self._pipeline_worker = None
        self.analysis_panel.set_running(False)
        assert isinstance(payload, dict)
        result: PipelineResult = payload["result"]
        self._last_result = result
        self.validation_panel.show_report(result.report)
        self.results_panel.show_result(payload)
        self.tabs.setCurrentWidget(self.results_panel)
        n_pos = result.n_positives
        message = (
            f"Finished: {len(result.ranked)} candidates, {n_pos} positive labels."
            if n_pos else
            "Finished with warnings — no positive labels (see Validation tab)."
        )
        self.statusBar().showMessage(message)
        self.validation_panel.append_log(message)
        if n_pos == 0 and not result.report.errors:
            QMessageBox.warning(
                self, config.APP_NAME,
                "The analysis finished but found no positive labels. "
                "Download the label datasets (label_eqtl, label_eqtl_permuted) "
                "in the Data tab, then rerun.",
            )

    def _on_error(self, message: str) -> None:
        self._pipeline_worker = None
        self.analysis_panel.set_running(False)
        self.validation_panel.append_log(f"ERROR:\n{message}")
        self.statusBar().showMessage("Analysis failed — see Validation / Log tab.")
        self.tabs.setCurrentWidget(self.validation_panel)
        QMessageBox.critical(
            self, config.APP_NAME,
            "The analysis failed. See the Validation / Log tab for details.",
        )

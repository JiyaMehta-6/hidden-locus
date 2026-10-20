"""Main window for the Hidden Locus desktop application.

Tabs: Data manager, Analysis, Results, Validation log.  Long operations run
in :class:`~hidden_locus.app.workers` on the global thread pool; widgets only
receive signals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThreadPool
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
from .workers import DownloadWorker, ExportWorker, PipelineWorker, WorkerSignals


class PandasModel(QAbstractTableModel):
    """Read-only table model exposing a pandas DataFrame to QTableView."""

    def __init__(self, frame: "pd.DataFrame | None" = None) -> None:
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.frame.iat[index.row(), index.column()]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

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
        for row, st in enumerate(statuses):
            values = [
                st["key"], st["name"], st["state"], f"{st['bytes']:,}",
                st["approx_size"], "yes" if st["required"] else "no",
                "yes" if st["verified"] else "no",
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

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

        layout.addWidget(form_box)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.setMinimumHeight(36)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.stage_label = QLabel("Idle.")
        layout.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.run_btn.clicked.connect(self._run)

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


class ResultsPanel(QWidget):
    """Ranked candidates, metrics and plots."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__()
        self.main = parent
        self._export_worker: "ExportWorker | None" = None
        layout = QVBoxLayout(self)
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

        self.plot_label = QLabel("Run the analysis to generate figures.")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(self.plot_label, "ROC figure")

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

    def _show_plot(self, written: dict) -> None:
        roc = written.get("roc")
        if roc and Path(roc).exists():
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(str(roc))
            if not pixmap.isNull():
                self.plot_label.setPixmap(
                    pixmap.scaled(
                        900, 700,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return
        self.plot_label.setText("No ROC figure produced.")


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
        for row, issue in enumerate(report.issues):
            for col, text in enumerate(
                [issue.level, issue.code, issue.where, issue.message]
            ):
                self.issues.setItem(row, col, QTableWidgetItem(text))


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
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self.statusBar().showMessage("Ready.")

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

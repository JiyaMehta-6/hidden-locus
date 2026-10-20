"""Background workers for long-running pipeline and download tasks.

Workers run on a ``QThreadPool``; they never touch Qt widgets directly —
they only emit signals, which the GUI thread consumes.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal

from .. import config
from ..core import datasets as ds
from ..core.pipeline import PipelineResult, export_report, run_full_pipeline
from ..core.validation import ValidationReport


class WorkerSignals(QObject):
    """Signals emitted by :class:`PipelineWorker` / :class:`DownloadWorker`."""

    stage = Signal(str, int, int)          # (stage, done, total)
    finished = Signal(object)              # payload depends on worker
    error = Signal(str)                    # formatted traceback / message
    log = Signal(str)                      # human-readable status line


class PipelineWorker(QRunnable):
    """Run the full Hidden Locus pipeline on a background thread."""

    def __init__(self, settings: config.Settings) -> None:
        super().__init__()
        self.settings = settings
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - exercised via Qt event loop
        try:
            report = ValidationReport()

            def progress(stage: str, done: int, total: int) -> None:
                self.signals.stage.emit(stage, done, total)
                self.signals.log.emit(f"{stage}: {done}/{total}")

            result = run_full_pipeline(
                settings=self.settings, report=report, progress=progress
            )
            written = export_report(result, progress=progress)
            result_dict = {"result": result, "written": written}
            self.signals.finished.emit(result_dict)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class ExportWorker(QRunnable):
    """Re-export an existing pipeline result (e.g. after changing output dir)."""

    def __init__(self, result: PipelineResult) -> None:
        super().__init__()
        self.result = result
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover
        try:
            written = export_report(self.result)
            self.signals.finished.emit({"result": self.result, "written": written})
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class DownloadWorker(QRunnable):
    """Download one dataset (or all required ones) in the background."""

    def __init__(self, keys: "list[str]", force: bool = False) -> None:
        super().__init__()
        self.keys = keys
        self.force = force
        self.signals = WorkerSignals()
        self.cancel = threading.Event()
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover
        try:
            done: list[str] = []
            for key in self.keys:
                if self.cancel.is_set():
                    self.signals.log.emit("Download cancelled.")
                    break
                spec = ds.get_spec(key)
                self.signals.log.emit(f"Fetching {spec.name}...")

                def progress(done_bytes: int, total: "int | None",
                             key: str = key) -> None:
                    if total:
                        self.signals.stage.emit(key, done_bytes, total)
                    self.signals.log.emit(
                        f"{key}: {done_bytes:,} / "
                        f"{total or 0:,} bytes"
                    )

                paths = ds.fetch(key, progress=progress, cancel=self.cancel, force=self.force)
                ds.verify(key)
                done.append(key)
                self.signals.log.emit(f"{key}: saved to {paths[0].name}")
            self.signals.finished.emit({"done": done})
        except Exception:
            self.signals.error.emit(traceback.format_exc())

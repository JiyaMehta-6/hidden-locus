"""Hidden Locus desktop application package.

Run with::

    python -m hidden_locus.app
"""

from __future__ import annotations

import sys


def run() -> int:
    """Launch the PySide6 application."""
    from PySide6.QtWidgets import QApplication

    from .. import config
    from .main_window import MainWindow

    config.ensure_data_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())

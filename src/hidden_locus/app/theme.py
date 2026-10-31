"""Application-wide look and feel: Fusion style, dark palette, accent QSS.

Applied in :func:`hidden_locus.app.run` and by the README screenshot
generator, so the desktop app renders identically on every machine instead
of inheriting the operating-system theme.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Shared accent colours (also used inline for KPI values).
ACCENT = "#2f6fed"
ACCENT_LIGHT = "#4dabf7"
ACCENT_SOFT = "#8fb6ff"
GOOD = "#4ade80"
WARN = "#ffb454"
TEXT = "#dce3ee"
MUTED = "#8b94a7"

STYLE_SHEET = """
QMainWindow { background: #14181f; }

QMenuBar {
    background: #1d232d; color: #dce3ee; padding: 2px;
    border-bottom: 1px solid #232a35;
}
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 4px; }
QMenuBar::item:selected { background: #2a3342; }
QMenu { background: #1d232d; color: #dce3ee; border: 1px solid #2a313d; padding: 4px; }
QMenu::item { padding: 5px 24px 5px 12px; border-radius: 4px; }
QMenu::item:selected { background: #2f6fed; color: #ffffff; }

QStatusBar { background: #10141a; color: #8b94a7; border-top: 1px solid #232a35; }
QStatusBar::item { border: none; }

QFrame#appHeader {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #1c2534, stop:1 #14181f);
    border-bottom: 1px solid #2f6fed;
}
QLabel#appTitle { color: #ffffff; font-size: 17px; font-weight: 700; }
QLabel#appSubtitle { color: #8b94a7; font-size: 12px; }
QLabel#appChip {
    color: #8fb6ff; background: #1a2436; border: 1px solid #33497a;
    border-radius: 9px; padding: 3px 10px; font-size: 11px;
}

QTabWidget::pane {
    border: 1px solid #232a35; border-radius: 8px; top: -1px;
    background: #151a22;
}
QTabBar::tab {
    background: transparent; color: #8b94a7; padding: 8px 16px;
    margin-right: 2px; border: 1px solid transparent;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background: #151a22; color: #ffffff;
    border: 1px solid #232a35; border-bottom-color: #151a22;
}
QTabBar::tab:hover:!selected { color: #dce3ee; }

QGroupBox {
    background: #181e28; border: 1px solid #232a35; border-radius: 8px;
    margin-top: 16px; padding: 10px 10px 6px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 4px; color: #8fb6ff; font-weight: 600;
}

QPushButton {
    background: #232a35; border: 1px solid #2f3948; border-radius: 6px;
    padding: 6px 14px; color: #dce3ee;
}
QPushButton:hover { background: #2a3342; border-color: #3c4a5e; }
QPushButton:pressed { background: #1d2530; }
QPushButton:disabled { color: #5b6474; background: #1a202a; border-color: #232a35; }
QPushButton#primary {
    background: #2f6fed; border: 1px solid #2f6fed; color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover { background: #3c7cf5; border-color: #3c7cf5; }
QPushButton#primary:disabled {
    background: #22304d; border-color: #22304d; color: #7d8ba8;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background: #10141a; border: 1px solid #2a313d; border-radius: 6px;
    padding: 4px 8px; color: #dce3ee;
    selection-background-color: #2f6fed; selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus { border-color: #2f6fed; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #171d27; border: 1px solid #2a313d;
    selection-background-color: #2f6fed; selection-color: #ffffff;
}
QCheckBox { spacing: 8px; }
QTextEdit { font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; }

QProgressBar {
    background: #10141a; border: 1px solid #232a35; border-radius: 7px;
    height: 14px; text-align: center; color: #dce3ee; font-size: 10px;
}
QProgressBar::chunk { background: #2f6fed; border-radius: 6px; }

QHeaderView::section {
    background: #181e28; color: #94a1b8; border: none;
    border-right: 1px solid #232a35; border-bottom: 1px solid #232a35;
    padding: 6px;
}
QTableView, QTableWidget {
    background: #10141a; alternate-background-color: #151b24;
    gridline-color: #1e2530; selection-background-color: #2f6fed;
    selection-color: #ffffff; border: 1px solid #232a35; border-radius: 8px;
}
QTableView::item:selected, QTableWidget::item:selected {
    background: #2f6fed; color: #ffffff;
}

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical {
    background: #2f3948; border-radius: 5px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #3c4a5e; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal {
    background: #2f3948; border-radius: 5px; min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: #3c4a5e; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

QFrame#statCard {
    background: #181e28; border: 1px solid #232a35; border-radius: 10px;
}
QLabel#statCaption { color: #8b94a7; font-size: 10px; font-weight: 600; }
QLabel#statValue { font-size: 20px; font-weight: 700; }
QLabel#statSub { color: #6f7a8f; font-size: 11px; }

QLabel#stageHint {
    color: #7f8ba1; font-size: 11px; padding: 2px 2px 6px 2px;
}
QLabel#mutedNote { color: #6f7a8f; font-size: 11px; }
QLabel#plotEmpty { color: #8b94a7; font-size: 12px; }

QFrame#infoCard {
    background: #181e28; border: 1px solid #232a35; border-radius: 10px;
}
QLabel#infoTitle { color: #ffffff; font-size: 13px; font-weight: 600; }
QLabel#stepBadge {
    background: #1a2436; color: #8fb6ff; border: 1px solid #33497a;
    border-radius: 11px; font-size: 10px; font-weight: 700;
}
QLabel#stepTitle { color: #dce3ee; font-size: 12px; font-weight: 600; }
QLabel#stepBody { color: #8b94a7; font-size: 11px; }

QToolTip {
    background: #1d232d; color: #dce3ee; border: 1px solid #2a313d;
    padding: 4px 6px;
}
"""


def _dark_palette() -> QPalette:
    """Fusion palette tuned to the dark theme (covers unstyled widgets)."""
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "#14181f",
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: "#10141a",
        QPalette.ColorRole.AlternateBase: "#151b24",
        QPalette.ColorRole.ToolTipBase: "#1d232d",
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: "#232a35",
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#ff6b81",
        QPalette.ColorRole.Link: ACCENT_LIGHT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#6f7a8f",
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5b6474"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#22304d")
    )
    return palette


def apply_theme(app: QApplication) -> None:
    """Apply the Hidden Locus theme to *app* (idempotent)."""
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(STYLE_SHEET)

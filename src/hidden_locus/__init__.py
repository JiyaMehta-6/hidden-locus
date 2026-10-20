"""Hidden Locus package root."""

__version__ = "0.1.0"


def main() -> None:
    """Console-script entry point: launch the desktop application."""
    from .app import run

    raise SystemExit(run())

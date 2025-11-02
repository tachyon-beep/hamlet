"""Pytest configuration for Hamlet project test suite."""

from __future__ import annotations

from pathlib import Path


def pytest_sessionstart(session):  # pragma: no cover - test infrastructure
    """Ensure legacy coverage artefacts do not break branch coverage runs.

    pytest-cov uses coverage.py's branch mode by default in this project.
    Older `.coverage` files produced without branch data cause coverage.py to
    raise `DataError: Can't combine branch coverage data with statement data`
    when reports are aggregated.  Proactively delete any stale coverage data
    before the run so each invocation starts clean.
    """

    coverage_root = Path(".")
    for data_file in coverage_root.glob(".coverage*"):
        # Avoid touching directories or freshly-created files mid-run.
        if data_file.is_file():
            try:
                data_file.unlink()
            except OSError:
                # If deletion fails we fall back to letting coverage.py handle it.
                pass

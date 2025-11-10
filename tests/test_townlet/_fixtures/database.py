'"""Database fixtures for demo components."""'

from __future__ import annotations

from pathlib import Path

import pytest

__all__ = ["demo_database"]


@pytest.fixture
def demo_database(tmp_path: Path):
    """Create a DemoDatabase with automatic cleanup."""

    from townlet.demo.database import DemoDatabase

    db_path = tmp_path / "test.db"
    db = DemoDatabase(db_path)
    yield db
    db.close()

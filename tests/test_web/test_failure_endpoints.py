import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("httpx")

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # Missing httpx dependency
    pytest.skip(str(exc), allow_module_level=True)

from hamlet.web.server import app as server_app
from hamlet.web.training_server import app as training_app


def _create_failure_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE failure_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            episode INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            reason TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO failure_events (timestamp, episode, agent_id, reason)
        VALUES
            ('2025-01-01T12:00:00', 10, 'agent_0', 'energy_depleted'),
            ('2025-01-02T12:00:00', 12, 'agent_1', 'bankrupt')
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.mark.parametrize("app", [server_app, training_app])
def test_failure_endpoints_return_data(tmp_path, app):
    db_path = _create_failure_db(Path(tmp_path))
    client = TestClient(app)

    failures_resp = client.get(f"/api/failures?db_path={db_path}")
    assert failures_resp.status_code == 200
    data = failures_resp.json()
    assert data["failures"], "Expected failures list to be non-empty"

    summary_resp = client.get(f"/api/failure_summary?db_path={db_path}")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["summary"], "Expected summary list to be non-empty"


@pytest.mark.parametrize("app", [server_app, training_app])
def test_failure_endpoints_handle_missing_db(tmp_path, app):
    db_path = Path(tmp_path) / "missing.db"
    client = TestClient(app)

    failures_resp = client.get(f"/api/failures?db_path={db_path}")
    assert failures_resp.status_code == 200
    assert failures_resp.json()["failures"] == []

    summary_resp = client.get(f"/api/failure_summary?db_path={db_path}")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["summary"] == []

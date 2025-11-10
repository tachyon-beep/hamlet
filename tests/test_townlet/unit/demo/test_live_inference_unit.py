"""Targeted unit tests for LiveInferenceServer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from townlet.demo import live_inference
from townlet.demo.live_inference import LiveInferenceServer, build_agent_telemetry_payload
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate


class _StubPopulation:
    def __init__(self):
        self.calls: list[int | None] = []

    def build_telemetry_snapshot(self, episode_index: int | None = None) -> dict[str, Any]:
        self.calls.append(episode_index)
        return {"episode_index": episode_index, "agents": [{"agent_id": "a0"}]}


@pytest.fixture
def noop_qvalue_log(monkeypatch):
    """Avoid touching the real qvalues log file."""

    monkeypatch.setattr(LiveInferenceServer, "_open_qvalue_log", lambda self: None)


@pytest.fixture
def live_server(tmp_path: Path, test_config_pack_path: Path, noop_qvalue_log):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    server = LiveInferenceServer(
        checkpoint_dir=checkpoint_dir,
        port=9999,
        step_delay=0.01,
        total_episodes=10,
        config_dir=test_config_pack_path,
        training_config_path=test_config_pack_path / "training.yaml",
    )
    yield server
    # ensure we don't leave open descriptors
    if getattr(server, "_qvalue_log_file", None):
        server._qvalue_log_file.close()


def test_build_agent_telemetry_payload_includes_schema():
    population = _StubPopulation()

    payload = build_agent_telemetry_payload(population, episode_index=7)

    assert payload["schema_version"] == live_inference.TELEMETRY_SCHEMA_VERSION
    assert payload["agents"] == [{"agent_id": "a0"}]
    assert population.calls == [7]


def test_build_agent_telemetry_payload_handles_missing_population():
    payload = build_agent_telemetry_payload(None, episode_index=None)

    assert payload == {"schema_version": live_inference.TELEMETRY_SCHEMA_VERSION, "episode_index": None, "agents": []}


def test_build_substrate_metadata_for_grid2d(live_server: LiveInferenceServer):
    live_server.env = type("Env", (), {"substrate": Grid2DSubstrate(width=4, height=5, boundary="clamp", distance_metric="manhattan")})()

    metadata = live_server._build_substrate_metadata()

    assert metadata["type"] == "grid2d"
    assert metadata["width"] == 4
    assert metadata["height"] == 5
    assert metadata["position_dim"] == 2


def test_build_grid_data_handles_aspatial(live_server: LiveInferenceServer):
    live_server.env = type("Env", (), {"substrate": AspatialSubstrate()})()

    grid_data = live_server._build_grid_data(agent_pos=[], last_action=0, affordances=[])

    assert grid_data == {"type": "aspatial"}


@pytest.mark.asyncio
async def test_broadcast_to_clients_removes_dead_connections(live_server: LiveInferenceServer):
    class _StubClient:
        def __init__(self, should_fail: bool = False):
            self.should_fail = should_fail
            self.messages: list[dict[str, Any]] = []

        async def send_json(self, message: dict[str, Any]):
            if self.should_fail:
                raise RuntimeError("boom")
            self.messages.append(message)

    healthy = _StubClient()
    failing = _StubClient(should_fail=True)
    live_server.clients = {healthy, failing}

    await live_server._broadcast_to_clients({"type": "update"})

    assert healthy.messages == [{"type": "update"}]
    assert failing not in live_server.clients


def test_build_agent_telemetry_payload_uses_episode_index(live_server: LiveInferenceServer):
    live_server.population = _StubPopulation()

    telemetry = live_server._build_agent_telemetry()

    assert telemetry["schema_version"] == live_inference.TELEMETRY_SCHEMA_VERSION
    assert telemetry["episode_index"] == 0

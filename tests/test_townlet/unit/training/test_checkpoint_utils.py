"""Tests for checkpoint metadata helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from townlet.training.checkpoint_utils import (
    assert_checkpoint_dimensions,
    attach_universe_metadata,
    config_hash_warning,
    persist_checkpoint_digest,
    safe_torch_load,
    verify_checkpoint_digest,
)
from townlet.training.state import PopulationCheckpoint
from townlet.universe.compiler import UniverseCompiler


@pytest.fixture(scope="module")
def compiled_universe():
    compiler = UniverseCompiler()
    return compiler.compile(Path("configs/L0_0_minimal"))


def test_attach_universe_metadata(compiled_universe) -> None:
    checkpoint: dict[str, object] = {}
    attach_universe_metadata(checkpoint, compiled_universe)

    assert checkpoint["config_hash"] == compiled_universe.metadata.config_hash
    assert checkpoint["observation_dim"] == compiled_universe.metadata.observation_dim
    assert checkpoint["action_dim"] == compiled_universe.metadata.action_count
    assert checkpoint["meter_count"] == compiled_universe.metadata.meter_count
    assert checkpoint["observation_field_uuids"] == [field.uuid for field in compiled_universe.observation_spec.fields]


def test_config_hash_warning_detects_mismatch(compiled_universe) -> None:
    attach = {}
    attach_universe_metadata(attach, compiled_universe)
    warn = config_hash_warning(attach, compiled_universe)
    assert warn is None

    attach["config_hash"] = "deadbeef"
    warn = config_hash_warning(attach, compiled_universe)
    assert warn is not None
    assert "Checkpoint" in warn


def test_assert_checkpoint_dimensions_raises_on_mismatch(compiled_universe) -> None:
    checkpoint: dict[str, object] = {}
    attach_universe_metadata(checkpoint, compiled_universe)

    assert_checkpoint_dimensions(checkpoint, compiled_universe)

    checkpoint["observation_dim"] = -1
    with pytest.raises(ValueError):
        assert_checkpoint_dimensions(checkpoint, compiled_universe)

    checkpoint = {}
    attach_universe_metadata(checkpoint, compiled_universe)
    checkpoint["observation_field_uuids"][0] = "deadbeefdeadbeef"
    with pytest.raises(ValueError):
        assert_checkpoint_dimensions(checkpoint, compiled_universe)


def test_safe_torch_load_rejects_custom_objects(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy.pt"
    legacy_checkpoint = PopulationCheckpoint(
        generation=0,
        num_agents=1,
        agent_ids=["agent_0"],
        curriculum_states={},
        exploration_states={},
        pareto_frontier=[],
        metrics_summary={},
    )
    torch.save(legacy_checkpoint, checkpoint_path)

    with pytest.raises(Exception) as excinfo:
        safe_torch_load(checkpoint_path)
    assert "weights only load failed" in str(excinfo.value).lower()


def test_safe_torch_load_roundtrip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "safe.pt"
    payload = {"weights": torch.ones(2), "metadata": {"episode": 5}}
    torch.save(payload, checkpoint_path)

    loaded = safe_torch_load(checkpoint_path)
    assert torch.equal(loaded["weights"], payload["weights"])
    assert loaded["metadata"]["episode"] == 5


def test_checkpoint_digest_roundtrip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_ep00010.pt"
    checkpoint_path.write_bytes(b"demo-checkpoint")

    digest = persist_checkpoint_digest(checkpoint_path)
    assert len(digest) == 64  # hex sha256
    assert verify_checkpoint_digest(checkpoint_path, required=True)


def test_checkpoint_digest_detects_tampering(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_ep00011.pt"
    checkpoint_path.write_bytes(b"demo-checkpoint")
    persist_checkpoint_digest(checkpoint_path)

    checkpoint_path.write_bytes(b"demo-checkpoint-corrupted")
    with pytest.raises(ValueError):
        verify_checkpoint_digest(checkpoint_path, required=True)

"""Tests for checkpoint metadata helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.training.checkpoint_utils import (
    assert_checkpoint_dimensions,
    attach_universe_metadata,
    config_hash_warning,
)
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

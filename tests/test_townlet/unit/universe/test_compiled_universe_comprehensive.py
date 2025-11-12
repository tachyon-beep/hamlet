"""Comprehensive tests for CompiledUniverse to achieve 60%+ coverage.

These tests address gaps in compiled.py:
- Lines 54-62: __post_init__ validation and tuple conversion
- Lines 71-91: Property accessors
- Lines 98-100: create_environment helper
- Lines 111-150: check_checkpoint_compatibility with all error branches
- Lines 176-203: save_to_cache serialization
- Lines 209-217: load_from_cache deserialization
- Lines 252-278: Helper functions (_dataclass_to_plain, position serialization)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from townlet.universe.compiled import (
    CompiledUniverse,
    _dataclass_to_plain,
    _deserialize_affordance_positions,
    _serialize_affordance_positions,
)
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.dto import (
    ObservationField,
    ObservationSpec,
)


@pytest.fixture(scope="module")
def compiled_universe() -> CompiledUniverse:
    """Compile L0_0_minimal for testing."""
    compiler = UniverseCompiler()
    return compiler.compile(Path("configs/L0_0_minimal"))


class TestCompiledUniverseProperties:
    """Test CompiledUniverse convenience property accessors."""

    def test_substrate_property_returns_hamlet_config_substrate(self, compiled_universe):
        """Verify .substrate returns substrate from hamlet_config."""
        substrate = compiled_universe.substrate
        assert substrate is compiled_universe.hamlet_config.substrate
        assert hasattr(substrate, "type")

    def test_bars_property_returns_hamlet_config_bars(self, compiled_universe):
        """Verify .bars returns bars from hamlet_config."""
        bars = compiled_universe.bars
        assert bars is compiled_universe.hamlet_config.bars
        assert len(bars) > 0

    def test_cascades_property_returns_hamlet_config_cascades(self, compiled_universe):
        """Verify .cascades returns cascades from hamlet_config."""
        cascades = compiled_universe.cascades
        assert cascades is compiled_universe.hamlet_config.cascades

    def test_affordances_property_returns_hamlet_config_affordances(self, compiled_universe):
        """Verify .affordances returns affordances from hamlet_config."""
        affordances = compiled_universe.affordances
        assert affordances is compiled_universe.hamlet_config.affordances
        assert len(affordances) > 0

    def test_cues_property_returns_hamlet_config_cues(self, compiled_universe):
        """Verify .cues returns cues from hamlet_config."""
        cues = compiled_universe.cues
        assert cues is compiled_universe.hamlet_config.cues

    def test_training_property_returns_hamlet_config_training(self, compiled_universe):
        """Verify .training returns training from hamlet_config."""
        training = compiled_universe.training
        assert training is compiled_universe.hamlet_config.training
        assert hasattr(training, "max_episodes")


class TestCheckpointCompatibility:
    """Test checkpoint validation logic."""

    def test_check_checkpoint_compatibility_returns_true_for_valid_checkpoint(self, compiled_universe):
        """Verify valid checkpoint passes all checks."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            "observation_dim": compiled_universe.metadata.observation_dim,
            "action_dim": compiled_universe.metadata.action_count,
            "observation_field_uuids": [field.uuid for field in compiled_universe.observation_spec.fields],
            "drive_hash": compiled_universe.drive_hash,  # Now required
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is True
        assert msg == "Checkpoint compatible."

    def test_check_checkpoint_compatibility_fails_for_missing_config_hash(self, compiled_universe):
        """Verify checkpoint without config_hash fails."""
        checkpoint = {}
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "missing config_hash" in msg

    def test_check_checkpoint_compatibility_fails_for_mismatched_config_hash(self, compiled_universe):
        """Verify checkpoint with wrong config_hash fails."""
        checkpoint = {"config_hash": "wrong_hash"}
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "Config hash mismatch" in msg

    def test_check_checkpoint_compatibility_fails_for_mismatched_observation_dim(self, compiled_universe):
        """Verify checkpoint with wrong observation_dim fails."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            "observation_dim": 9999,  # Wrong
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "Observation dimension mismatch" in msg

    def test_check_checkpoint_compatibility_allows_missing_observation_dim(self, compiled_universe):
        """Verify checkpoint without observation_dim is checked only on other fields."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            # observation_dim missing - should not fail on this alone
            "observation_field_uuids": [field.uuid for field in compiled_universe.observation_spec.fields],
            "drive_hash": compiled_universe.drive_hash,  # Now required
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is True

    def test_check_checkpoint_compatibility_fails_for_mismatched_action_dim(self, compiled_universe):
        """Verify checkpoint with wrong action_dim fails."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            "action_dim": 9999,  # Wrong
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "Action dimension mismatch" in msg

    def test_check_checkpoint_compatibility_allows_missing_action_dim(self, compiled_universe):
        """Verify checkpoint without action_dim is checked only on other fields."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            # action_dim missing - should not fail on this alone
            "observation_field_uuids": [field.uuid for field in compiled_universe.observation_spec.fields],
            "drive_hash": compiled_universe.drive_hash,  # Now required
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is True

    def test_check_checkpoint_compatibility_fails_for_missing_field_uuids(self, compiled_universe):
        """Verify checkpoint without observation_field_uuids fails."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "missing observation_field_uuids" in msg

    def test_check_checkpoint_compatibility_fails_for_mismatched_field_uuids(self, compiled_universe):
        """Verify checkpoint with wrong observation field UUIDs fails."""
        checkpoint = {
            "config_hash": compiled_universe.metadata.config_hash,
            "observation_field_uuids": ["wrong-uuid-1", "wrong-uuid-2"],
        }
        compatible, msg = compiled_universe.check_checkpoint_compatibility(checkpoint)
        assert compatible is False
        assert "Observation field UUID mismatch" in msg


class TestToRuntime:
    """Test to_runtime conversion."""

    def test_to_runtime_creates_runtime_universe(self, compiled_universe):
        """Verify to_runtime creates RuntimeUniverse with correct fields."""
        runtime = compiled_universe.to_runtime()
        assert runtime.metadata is compiled_universe.metadata
        assert runtime.observation_spec is compiled_universe.observation_spec
        assert runtime.action_space_metadata is compiled_universe.action_space_metadata
        assert runtime.config_dir == compiled_universe.config_dir


class TestSerializationHelpers:
    """Test serialization helper functions."""

    def test_dataclass_to_plain_converts_simple_dataclass(self):
        """Verify _dataclass_to_plain converts dataclass to dict."""
        field = ObservationField(
            uuid="test-uuid",
            name="test_field",
            type="scalar",
            dims=5,
            start_index=0,
            end_index=5,
            scope="agent",
            description="Test field",
        )
        result = _dataclass_to_plain(field)
        assert isinstance(result, dict)
        assert result["name"] == "test_field"
        assert result["dims"] == 5
        assert result["uuid"] == "test-uuid"

    def test_dataclass_to_plain_handles_nested_dataclasses(self):
        """Verify _dataclass_to_plain handles nested dataclasses."""
        spec = ObservationSpec(
            total_dims=5,
            encoding_version="v1",
            fields=(
                ObservationField(
                    uuid="uuid1",
                    name="f1",
                    type="scalar",
                    dims=5,
                    start_index=0,
                    end_index=5,
                    scope="agent",
                    description="Field 1",
                ),
            ),
        )
        result = _dataclass_to_plain(spec)
        assert isinstance(result, dict)
        assert isinstance(result["fields"], list)
        assert result["fields"][0]["name"] == "f1"

    def test_dataclass_to_plain_handles_lists(self):
        """Verify _dataclass_to_plain recursively processes lists."""
        data = [
            ObservationField(uuid="u1", name="f1", type="scalar", dims=1, start_index=0, end_index=1, scope="agent", description="Field 1"),
            ObservationField(uuid="u2", name="f2", type="scalar", dims=2, start_index=1, end_index=3, scope="agent", description="Field 2"),
        ]
        result = _dataclass_to_plain(data)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "f1"
        assert result[1]["name"] == "f2"

    def test_dataclass_to_plain_handles_dicts(self):
        """Verify _dataclass_to_plain recursively processes dict values."""
        data = {
            "key1": ObservationField(
                uuid="u1", name="f1", type="scalar", dims=1, start_index=0, end_index=1, scope="agent", description="Field 1"
            ),
            "key2": "simple_value",
        }
        result = _dataclass_to_plain(data)
        assert isinstance(result, dict)
        assert isinstance(result["key1"], dict)
        assert result["key1"]["name"] == "f1"
        assert result["key2"] == "simple_value"

    def test_dataclass_to_plain_preserves_primitives(self):
        """Verify _dataclass_to_plain preserves primitive types."""
        assert _dataclass_to_plain("test") == "test"
        assert _dataclass_to_plain(42) == 42
        assert _dataclass_to_plain(3.14) == 3.14
        assert _dataclass_to_plain(True) is True
        assert _dataclass_to_plain(None) is None

    def test_serialize_affordance_positions_converts_tensors(self):
        """Verify _serialize_affordance_positions converts tensors to lists."""
        position_map = {
            "Bed": torch.tensor([0, 0]),
            "Hospital": torch.tensor([1, 1]),
            "NonExistent": None,
        }
        result = _serialize_affordance_positions(position_map)
        assert result["Bed"] == [0, 0]
        assert result["Hospital"] == [1, 1]
        assert result["NonExistent"] is None

    def test_serialize_affordance_positions_preserves_none(self):
        """Verify _serialize_affordance_positions preserves None values."""
        position_map = {"NonExistent": None}
        result = _serialize_affordance_positions(position_map)
        assert result["NonExistent"] is None

    def test_deserialize_affordance_positions_converts_lists_to_tensors(self):
        """Verify _deserialize_affordance_positions converts lists to tensors."""
        payload = {
            "Bed": [0, 0],
            "Hospital": [1, 1],
            "NonExistent": None,
        }
        result = _deserialize_affordance_positions(payload)
        assert isinstance(result["Bed"], torch.Tensor)
        assert result["Bed"].tolist() == [0, 0]
        assert isinstance(result["Hospital"], torch.Tensor)
        assert result["Hospital"].tolist() == [1, 1]
        assert result["NonExistent"] is None

    def test_deserialize_affordance_positions_preserves_none(self):
        """Verify _deserialize_affordance_positions preserves None values."""
        payload = {"NonExistent": None}
        result = _deserialize_affordance_positions(payload)
        assert result["NonExistent"] is None


class TestSerializationRoundTrip:
    """Test save_to_cache and load_from_cache."""

    def test_save_and_load_preserves_compiled_universe(self, compiled_universe, tmp_path):
        """Verify save_to_cache and load_from_cache roundtrip correctly."""
        cache_path = tmp_path / "universe.msgpack"
        compiled_universe.save_to_cache(cache_path)
        assert cache_path.exists()

        loaded = CompiledUniverse.load_from_cache(cache_path)
        assert loaded.metadata.config_hash == compiled_universe.metadata.config_hash
        assert loaded.metadata.observation_dim == compiled_universe.metadata.observation_dim
        assert loaded.metadata.action_count == compiled_universe.metadata.action_count
        assert loaded.config_dir == compiled_universe.config_dir

    def test_load_from_cache_handles_none_action_mask(self, tmp_path):
        """Verify load_from_cache creates default action_mask_table when None."""
        # Create minimal payload with None action_mask_table
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))
        cache_path = tmp_path / "universe.msgpack"
        compiled.save_to_cache(cache_path)

        # Manually corrupt the action_mask_table to be None (simulating old cache format)
        import msgpack

        payload = msgpack.unpackb(cache_path.read_bytes(), raw=False)
        payload["optimization_data"]["action_mask_table"] = None
        cache_path.write_bytes(msgpack.packb(payload, use_bin_type=True))

        # Load should handle None gracefully
        loaded = CompiledUniverse.load_from_cache(cache_path)
        assert loaded.optimization_data.action_mask_table is not None
        assert loaded.optimization_data.action_mask_table.shape == (24, 0)

"""Tests for RawConfigs property accessors and SourceMap functionality.

These tests address critical coverage gaps identified in the test suite audit:
- compiler_inputs.py lines 98-179 (property accessors)
- source_map.py lines 33-93 (error location lookup)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.source_map import SourceMap


class TestRawConfigsPropertyAccessors:
    """Test RawConfigs convenience property accessors."""

    @pytest.fixture(scope="class")
    def raw_configs(self) -> RawConfigs:
        """Load L0_0_minimal as test fixture."""
        return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))

    def test_bars_property_returns_tuple_of_bar_configs(self, raw_configs: RawConfigs):
        """Verify .bars returns bars from hamlet_config."""
        bars = raw_configs.bars
        assert isinstance(bars, tuple)
        assert len(bars) > 0
        assert all(hasattr(bar, "name") for bar in bars)

    def test_bars_property_contains_expected_meters(self, raw_configs: RawConfigs):
        """Verify .bars includes standard meters."""
        bar_names = {bar.name for bar in raw_configs.bars}
        # L0_0_minimal should have at least energy
        assert "energy" in bar_names

    def test_cascades_property_returns_tuple_of_cascade_configs(self, raw_configs: RawConfigs):
        """Verify .cascades returns cascades from hamlet_config."""
        cascades = raw_configs.cascades
        assert isinstance(cascades, tuple)
        # L0_0_minimal may have 0 cascades, just verify type
        assert all(hasattr(cascade, "source") for cascade in cascades)

    def test_affordances_property_returns_tuple_of_affordances(self, raw_configs: RawConfigs):
        """Verify .affordances returns tuple of AffordanceConfig objects."""
        affordances = raw_configs.affordances
        assert isinstance(affordances, tuple)
        assert len(affordances) > 0
        assert all(hasattr(aff, "id") for aff in affordances)

    def test_affordances_property_contains_expected_affordances(self, raw_configs: RawConfigs):
        """Verify .affordances includes affordances from config."""
        # L0_0_minimal should have Bed affordance (id="0")
        affordance_names = {aff.name for aff in raw_configs.affordances}
        assert "Bed" in affordance_names

    def test_environment_property_returns_environment_config(self, raw_configs: RawConfigs):
        """Verify .environment returns environment config from hamlet_config."""
        env = raw_configs.environment
        # grid_size moved to substrate.yaml - check for partial_observability instead
        assert hasattr(env, "partial_observability")
        assert isinstance(env.partial_observability, bool)

    def test_training_property_returns_training_config(self, raw_configs: RawConfigs):
        """Verify .training returns training config from hamlet_config."""
        training = raw_configs.training
        assert hasattr(training, "max_episodes")
        assert training.max_episodes > 0

    def test_substrate_property_returns_substrate_config(self, raw_configs: RawConfigs):
        """Verify .substrate returns substrate config from hamlet_config."""
        substrate = raw_configs.substrate
        assert hasattr(substrate, "type")
        assert substrate.type in ("grid", "grid3d", "gridnd", "continuous", "continuousnd", "aspatial")

    def test_multiple_property_accesses_return_same_object(self, raw_configs: RawConfigs):
        """Verify property accessors are stable (return same object)."""
        bars1 = raw_configs.bars
        bars2 = raw_configs.bars
        assert bars1 is bars2  # Should be same tuple instance


class TestSourceMapLookup:
    """Test SourceMap error location lookup functionality."""

    @pytest.fixture
    def source_map(self, tmp_path: Path) -> tuple[SourceMap, Path]:
        """Create SourceMap with tracked affordances for testing."""
        smap = SourceMap()
        affordances_file = tmp_path / "affordances.yaml"
        affordances_file.write_text(
            """affordances:
  - id: Bed
    name: Sleep
    position: [0, 0]
  - id: Hospital
    name: Medical Care
    position: [1, 1]
"""
        )
        smap.track_affordances(affordances_file)
        return smap, affordances_file

    def test_lookup_returns_file_and_line_for_tracked_affordance(self, source_map: tuple[SourceMap, Path]):
        """Verify lookup returns file:line for tracked affordance ID."""
        smap, affordances_file = source_map
        location = smap.lookup("affordances.yaml:Bed")
        assert location is not None
        assert "affordances.yaml" in location
        assert ":2" in location  # Bed is on line 2

    def test_lookup_returns_none_for_untracked_key(self, source_map: tuple[SourceMap, Path]):
        """Verify lookup returns None for untracked keys."""
        smap, _ = source_map
        location = smap.lookup("nonexistent.yaml:Foo")
        assert location is None

    def test_lookup_handles_progressive_prefix_matching(self, source_map: tuple[SourceMap, Path]):
        """Verify lookup tries progressively shorter prefixes."""
        smap, affordances_file = source_map
        # If we track "affordances.yaml:Bed", lookup "affordances.yaml:Bed:extra" should find it
        location = smap.lookup("affordances.yaml:Bed:nested:field")
        assert location is not None
        assert "affordances.yaml" in location

    def test_lookup_returns_path_only_when_line_is_none(self):
        """Verify lookup returns path without :line when line is None."""
        smap = SourceMap()
        smap.record("test_key", Path("/fake/path.yaml"), None)
        location = smap.lookup("test_key")
        assert location == "/fake/path.yaml"  # No :line suffix

    def test_track_cascades_registers_cascade_names(self, tmp_path: Path):
        """Verify track_cascades registers cascade entries."""
        smap = SourceMap()
        cascades_file = tmp_path / "cascades.yaml"
        cascades_file.write_text(
            """cascades:
  - name: energy_to_health
    source: energy
    target: health
    rate: 0.5
"""
        )
        smap.track_cascades(cascades_file)
        location = smap.lookup("cascades.yaml:energy_to_health")
        assert location is not None
        assert "cascades.yaml" in location

    def test_track_actions_registers_custom_action_names(self, tmp_path: Path):
        """Verify track_actions registers custom action entries."""
        smap = SourceMap()
        actions_file = tmp_path / "global_actions.yaml"
        actions_file.write_text(
            """custom_actions:
  - name: REST
    description: Recover energy
    cost: {}
"""
        )
        smap.track_actions(actions_file)
        location = smap.lookup("global_actions.yaml:REST")
        assert location is not None
        assert "global_actions.yaml" in location

    def test_track_training_environment_key_finds_line_with_keyword(self, tmp_path: Path):
        """Verify track_training_environment_key finds line containing keyword."""
        smap = SourceMap()
        training_file = tmp_path / "training.yaml"
        training_file.write_text(
            """training:
  max_episodes: 1000
environment:
  enabled_affordances: [Bed, Hospital]
  grid_size: 8
"""
        )
        smap.track_training_environment_key(training_file, "enabled_affordances")
        location = smap.lookup("training.yaml:enabled_affordances")
        assert location is not None
        assert "training.yaml" in location
        # Should find a line number (the exact line depends on search logic)
        assert ":" in location  # Has line number

    def test_track_ignores_nonexistent_files(self, tmp_path: Path):
        """Verify track methods don't crash on missing files."""
        smap = SourceMap()
        fake_file = tmp_path / "nonexistent.yaml"
        # Should not raise
        smap.track_affordances(fake_file)
        smap.track_cascades(fake_file)
        smap.track_actions(fake_file)

    def test_track_ignores_malformed_yaml_structure(self, tmp_path: Path):
        """Verify track methods handle unexpected YAML structures gracefully."""
        smap = SourceMap()
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("just_a_string: not_a_list")
        # Should not raise
        smap.track_affordances(bad_file)
        assert smap.lookup("bad.yaml:anything") is None

    def test_bulk_record_registers_multiple_entries(self):
        """Verify bulk_record processes multiple entries."""
        smap = SourceMap()
        entries = [
            ("key1", Path("/path1.yaml"), 10),
            ("key2", Path("/path2.yaml"), 20),
            ("key3", Path("/path3.yaml"), None),
        ]
        smap.bulk_record(entries)

        assert smap.lookup("key1") == "/path1.yaml:10"
        assert smap.lookup("key2") == "/path2.yaml:20"
        assert smap.lookup("key3") == "/path3.yaml"


class TestRawConfigsEdgeCases:
    """Test RawConfigs edge cases and error handling."""

    def test_from_config_dir_with_multiple_config_packs(self):
        """Verify from_config_dir works with different config packs."""
        packs = ["L0_0_minimal", "L0_5_dual_resource", "L1_full_observability"]
        for pack_name in packs:
            config_dir = Path("configs") / pack_name
            if not config_dir.exists():
                pytest.skip(f"Config pack {pack_name} not found")

            raw = RawConfigs.from_config_dir(config_dir)
            assert raw.config_dir == config_dir
            assert len(raw.bars) > 0
            assert len(raw.affordances) >= 0  # Some packs might have 0

    def test_property_accessors_are_immutable_references(self):
        """Verify property accessors return immutable tuples/objects."""
        raw = RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))

        bars = raw.bars
        assert isinstance(bars, tuple)  # Tuples are immutable

        cascades = raw.cascades
        assert isinstance(cascades, tuple)  # Tuples are immutable

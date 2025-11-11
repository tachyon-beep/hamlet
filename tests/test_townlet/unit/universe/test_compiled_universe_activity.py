"""Test ObservationActivity wiring in CompiledUniverse and RuntimeUniverse."""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


class TestObservationActivityInCompiledUniverse:
    def test_compiled_universe_has_observation_activity(self, tmp_path):
        """CompiledUniverse should include ObservationActivity after compilation."""
        config_dir = Path("configs/L0_0_minimal")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)

        assert hasattr(compiled, "observation_activity")
        assert compiled.observation_activity is not None

        # Should have valid mask and slices
        assert len(compiled.observation_activity.active_mask) > 0
        assert compiled.observation_activity.total_dims == compiled.metadata.observation_dim

    def test_observation_activity_mask_matches_observation_dim(self):
        """active_mask length should equal total observation_dim."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)

        assert len(compiled.observation_activity.active_mask) == compiled.metadata.observation_dim

    def test_observation_activity_has_group_slices(self):
        """ObservationActivity should have slices for semantic groups."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        activity = compiled.observation_activity

        # L0_5 should have spatial and bars groups at minimum
        # Note: affordance may be grouped under 'custom' depending on semantic_type mapping
        assert activity.get_group_slice("spatial") is not None
        assert activity.get_group_slice("bars") is not None
        # Verify at least 3 groups exist
        assert len(activity.group_slices) >= 3

    def test_observation_activity_persists_through_cache(self, tmp_path):
        """ObservationActivity should survive MessagePack serialization."""
        config_dir = Path("configs/L0_0_minimal")

        # Compile and cache (use_cache=True to save to cache)
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(config_dir, use_cache=True)
        original_mask = compiled1.observation_activity.active_mask
        original_slices = compiled1.observation_activity.group_slices

        # Load from cache (use_cache=True to load from cache)
        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(config_dir, use_cache=True)

        assert compiled2.observation_activity.active_mask == original_mask
        assert compiled2.observation_activity.group_slices == original_slices


class TestObservationActivityInRuntimeUniverse:
    def test_runtime_universe_exposes_observation_activity(self):
        """RuntimeUniverse should expose observation_activity from compiled."""
        config_dir = Path("configs/L0_5_dual_resource")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        runtime = compiled.to_runtime()

        assert hasattr(runtime, "observation_activity")
        assert runtime.observation_activity is compiled.observation_activity

    def test_runtime_observation_activity_immutable(self):
        """RuntimeUniverse.observation_activity should reference frozen DTO."""
        config_dir = Path("configs/L0_0_minimal")

        compiler = UniverseCompiler()
        compiled = compiler.compile(config_dir, use_cache=False)
        runtime = compiled.to_runtime()

        # Should raise error (frozen dataclass)
        with pytest.raises(Exception):
            runtime.observation_activity.active_mask = (False, False)

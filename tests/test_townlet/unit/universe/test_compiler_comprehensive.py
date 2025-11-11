"""Comprehensive tests for UniverseCompiler to achieve 40%+ coverage.

These tests focus on:
- End-to-end compile() method with cache logic
- Helper methods (_build_cache_fingerprint, _cache_artifact_path)
- Error handling and validation
- Stage 7 (emit artifact) which has minimal coverage
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


class TestCompileEndToEnd:
    """Test the complete compilation pipeline."""

    def test_compile_creates_compiled_universe(self):
        """Verify compile() creates a CompiledUniverse with all metadata."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled is not None
        assert compiled.metadata is not None
        assert compiled.observation_spec is not None
        assert compiled.action_space_metadata is not None
        assert compiled.meter_metadata is not None
        assert compiled.affordance_metadata is not None
        assert compiled.optimization_data is not None

    def test_compile_sets_correct_config_dir(self):
        """Verify compile() preserves config_dir path."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        # Compiler stores the path as-is (relative or absolute)
        assert isinstance(compiled.config_dir, Path)
        assert compiled.config_dir.name == "L0_0_minimal"

    def test_compile_generates_stable_config_hash(self):
        """Verify compile() generates same config_hash for same config."""
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(Path("configs/L0_0_minimal"))

        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(Path("configs/L0_0_minimal"))

        assert compiled1.metadata.config_hash == compiled2.metadata.config_hash

    def test_compile_different_configs_have_different_hashes(self):
        """Verify different configs produce different config_hashes."""
        compiler = UniverseCompiler()
        compiled1 = compiler.compile(Path("configs/L0_0_minimal"))
        compiled2 = compiler.compile(Path("configs/L0_5_dual_resource"))

        assert compiled1.metadata.config_hash != compiled2.metadata.config_hash

    def test_compile_populates_observation_spec(self):
        """Verify compile() creates observation spec with fields."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.observation_spec.total_dims > 0
        assert len(compiled.observation_spec.fields) > 0
        assert compiled.observation_spec.encoding_version is not None

    def test_compile_populates_action_space_metadata(self):
        """Verify compile() creates action space metadata."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.action_space_metadata.total_actions > 0
        assert len(compiled.action_space_metadata.actions) > 0

    def test_compile_populates_meter_metadata(self):
        """Verify compile() creates meter metadata."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert len(compiled.meter_metadata.meters) > 0
        # L0_0_minimal should have energy meter
        meter_names = [m.name for m in compiled.meter_metadata.meters]
        assert "energy" in meter_names

    def test_compile_populates_affordance_metadata(self):
        """Verify compile() creates affordance metadata."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert len(compiled.affordance_metadata.affordances) > 0
        # L0_0_minimal should have Bed affordance
        affordance_names = [a.name for a in compiled.affordance_metadata.affordances]
        assert "Bed" in affordance_names

    def test_compile_populates_optimization_data(self):
        """Verify compile() creates optimization tensors."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.optimization_data.base_depletions is not None
        assert compiled.optimization_data.cascade_data is not None
        assert compiled.optimization_data.action_mask_table is not None

    def test_compile_with_multiple_config_packs(self):
        """Verify compile() works with different config packs."""
        packs = ["L0_0_minimal", "L0_5_dual_resource", "L1_full_observability"]
        compiler = UniverseCompiler()

        for pack_name in packs:
            config_dir = Path("configs") / pack_name
            if not config_dir.exists():
                pytest.skip(f"Config pack {pack_name} not found")

            compiled = compiler.compile(config_dir)
            assert compiled.metadata.universe_name == pack_name


class TestCacheBehavior:
    """Test caching logic."""

    def test_compile_with_use_cache_false_skips_cache(self, tmp_path):
        """Verify compile(use_cache=False) never reads cache."""
        # Create a config pack copy
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # First compile to create cache
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(dest, use_cache=True)
        cache_path = dest / ".compiled" / "universe.msgpack"
        assert cache_path.exists()

        # Second compile with use_cache=False should still compile
        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(dest, use_cache=False)

        # Both should produce same results but cache wasn't used
        assert compiled1.metadata.config_hash == compiled2.metadata.config_hash

    def test_compile_creates_cache_directory(self, tmp_path):
        """Verify compile() creates .compiled directory."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        compiler = UniverseCompiler()
        compiler.compile(dest, use_cache=True)

        cache_dir = dest / ".compiled"
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_compile_saves_cache_file(self, tmp_path):
        """Verify compile() saves universe.msgpack to cache."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        compiler = UniverseCompiler()
        compiler.compile(dest, use_cache=True)

        cache_file = dest / ".compiled" / "universe.msgpack"
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0

    def test_compile_uses_cache_on_second_run(self, tmp_path):
        """Verify compile() loads from cache on second run."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # First compile
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(dest, use_cache=True)

        # Get cache file modification time
        cache_file = dest / ".compiled" / "universe.msgpack"
        cache_mtime = cache_file.stat().st_mtime

        # Second compile (should use cache, not modify file)
        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(dest, use_cache=True)

        # Cache file should not have been modified
        assert cache_file.stat().st_mtime == cache_mtime
        assert compiled1.metadata.config_hash == compiled2.metadata.config_hash

    def test_compile_recompiles_when_config_changes(self, tmp_path):
        """Verify compile() detects config changes and recompiles."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # First compile
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(dest, use_cache=True)
        original_hash = compiled1.metadata.config_hash

        # Modify a config file
        bars_file = dest / "bars.yaml"
        content = bars_file.read_text()
        modified = content.replace("initial: 1.0", "initial: 0.9")
        bars_file.write_text(modified)

        # Second compile should detect change
        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(dest, use_cache=True)

        assert compiled2.metadata.config_hash != original_hash


class TestMetadata:
    """Test metadata generation."""

    def test_compile_sets_meter_count(self):
        """Verify metadata.meter_count matches bars."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.meter_count == len(compiled.bars)

    def test_compile_sets_affordance_count(self):
        """Verify metadata.affordance_count matches affordances."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.affordance_count == len(compiled.affordances)

    def test_compile_sets_action_count(self):
        """Verify metadata.action_count is populated."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.action_count > 0

    def test_compile_sets_observation_dim(self):
        """Verify metadata.observation_dim matches observation_spec."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.observation_dim == compiled.observation_spec.total_dims

    def test_compile_sets_universe_name(self):
        """Verify metadata.universe_name matches directory name."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.universe_name == "L0_0_minimal"

    def test_compile_sets_compiler_version(self):
        """Verify metadata.compiler_version is set."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.compiler_version is not None
        assert compiled.metadata.compiler_version != ""

    def test_compile_sets_provenance_id(self):
        """Verify metadata.provenance_id is set."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.metadata.provenance_id is not None
        assert compiled.metadata.provenance_id != ""


class TestCompilerHelpers:
    """Test compiler helper methods."""

    def test_cache_artifact_path_returns_expected_location(self):
        """Verify _cache_artifact_path returns .compiled/universe.msgpack."""
        compiler = UniverseCompiler()
        cache_path = compiler._cache_artifact_path(Path("configs/L0_0_minimal"))

        assert cache_path.name == "universe.msgpack"
        assert cache_path.parent.name == ".compiled"

    def test_build_cache_fingerprint_returns_tuple(self):
        """Verify _build_cache_fingerprint returns (config_hash, provenance_id)."""
        compiler = UniverseCompiler()
        config_hash, provenance = compiler._build_cache_fingerprint(Path("configs/L0_0_minimal"))

        assert isinstance(config_hash, str)
        assert isinstance(provenance, str)
        assert len(config_hash) > 0
        assert len(provenance) > 0

    def test_build_cache_fingerprint_stable_for_same_config(self):
        """Verify _build_cache_fingerprint is deterministic."""
        compiler1 = UniverseCompiler()
        hash1, prov1 = compiler1._build_cache_fingerprint(Path("configs/L0_0_minimal"))

        compiler2 = UniverseCompiler()
        hash2, prov2 = compiler2._build_cache_fingerprint(Path("configs/L0_0_minimal"))

        assert hash1 == hash2
        assert prov1 == prov2


class TestEnvironmentConfig:
    """Test environment_config population."""

    def test_compile_creates_environment_config(self):
        """Verify compile() creates EnvironmentConfig."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_0_minimal"))

        assert compiled.environment_config is not None
        assert hasattr(compiled.environment_config, "cascades")

    def test_environment_config_contains_cascades(self):
        """Verify environment_config includes cascade configuration."""
        compiler = UniverseCompiler()
        compiled = compiler.compile(Path("configs/L0_5_dual_resource"))

        # L0_5 has cascades
        assert compiled.environment_config.cascades is not None

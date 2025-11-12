"""Tests for DAC compiler validation."""

import shutil
from pathlib import Path

import pytest
import yaml

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


class TestDACReferenceValidation:
    """Test DAC reference validation in compiler Stage 3."""

    def test_dac_references_undefined_bar_in_modifier(self, tmp_path):
        """DAC modifier referencing undefined bar raises CompilationError."""
        # Copy L0_0_minimal as base
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Add drive_as_code.yaml with invalid bar reference
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {
                    "hunger_crisis": {
                        "bar": "hunger",  # UNDEFINED! (no hunger bar, only satiation exists)
                        "ranges": [{"name": "crisis", "min": 0.0, "max": 1.0, "multiplier": 0.0}],
                    }
                },
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (dest / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        # Should raise CompilationError
        compiler = UniverseCompiler()
        with pytest.raises(CompilationError, match="undefined bar|hunger"):
            compiler.compile(dest, use_cache=False)

    def test_dac_references_undefined_bar_in_extrinsic(self, tmp_path):
        """Extrinsic strategy referencing undefined bar raises CompilationError."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {
                    "type": "multiplicative",
                    "base": 1.0,
                    "bars": ["energy", "nonexistent_bar"],  # nonexistent_bar UNDEFINED!
                },
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (dest / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError, match="undefined bar|nonexistent_bar"):
            compiler.compile(dest, use_cache=False)

    def test_dac_optional_when_file_missing(self, tmp_path):
        """Compilation fails when drive_as_code.yaml is missing (now required)."""
        # Copy L0_0_minimal but remove drive_as_code.yaml
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Remove drive_as_code.yaml if it exists
        dac_file = dest / "drive_as_code.yaml"
        if dac_file.exists():
            dac_file.unlink()

        # Should fail compilation without DAC
        compiler = UniverseCompiler()
        with pytest.raises(CompilationError, match="drive_as_code.yaml is required"):
            compiler.compile(dest, use_cache=False)

    def test_dac_valid_config_compiles(self, tmp_path):
        """Valid DAC configuration compiles successfully."""
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Add VALID drive_as_code.yaml (references only "energy" which exists)
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {
                    "energy_crisis": {
                        "bar": "energy",  # VALID!
                        "ranges": [
                            {"name": "crisis", "min": 0.0, "max": 0.3, "multiplier": 0.0},
                            {"name": "normal", "min": 0.3, "max": 1.0, "multiplier": 1.0},
                        ],
                    }
                },
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (dest / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        # Should compile successfully
        compiler = UniverseCompiler()
        compiled = compiler.compile(dest, use_cache=False)

        # DAC should be loaded
        assert compiled.dac_config is not None
        assert compiled.dac_config.version == "1.0"
        assert "energy_crisis" in compiled.dac_config.modifiers


class TestDriveHashComputation:
    """Test drive_hash computation for DAC provenance."""

    def test_drive_hash_included_when_dac_present(self, tmp_path):
        """Compiled universe includes drive_hash when DAC config present."""
        # Copy L0_0_minimal as base
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Add valid drive_as_code.yaml
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {
                    "energy_crisis": {
                        "bar": "energy",
                        "ranges": [
                            {"name": "crisis", "min": 0.0, "max": 0.3, "multiplier": 0.0},
                            {"name": "normal", "min": 0.3, "max": 1.0, "multiplier": 1.0},
                        ],
                    }
                },
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (dest / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        # Compile
        compiler = UniverseCompiler()
        compiled = compiler.compile(dest, use_cache=False)

        # drive_hash should be present
        assert compiled.drive_hash is not None
        assert isinstance(compiled.drive_hash, str)
        assert len(compiled.drive_hash) == 64  # SHA256 hex digest

    def test_drive_hash_none_when_dac_missing(self, tmp_path):
        """Compilation fails when DAC not present (now required)."""
        # Copy L0_0_minimal and remove drive_as_code.yaml
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Remove drive_as_code.yaml if it exists
        dac_file = dest / "drive_as_code.yaml"
        if dac_file.exists():
            dac_file.unlink()

        # Compilation should fail without DAC
        compiler = UniverseCompiler()
        with pytest.raises(CompilationError, match="drive_as_code.yaml is required"):
            compiler.compile(dest, use_cache=False)

    def test_different_dac_configs_have_different_hashes(self, tmp_path):
        """Different DAC configurations produce different drive_hash values."""
        source = Path("configs/L0_0_minimal")

        # Compile with first DAC config
        dest1 = tmp_path / "config1"
        shutil.copytree(source, dest1)
        dac1 = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.1},
            }
        }
        (dest1 / "drive_as_code.yaml").write_text(yaml.dump(dac1))
        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(dest1, use_cache=False)

        # Compile with different DAC config (different base_weight)
        dest2 = tmp_path / "config2"
        shutil.copytree(source, dest2)
        dac2 = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                "intrinsic": {"strategy": "rnd", "base_weight": 0.2},  # DIFFERENT!
            }
        }
        (dest2 / "drive_as_code.yaml").write_text(yaml.dump(dac2))
        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(dest2, use_cache=False)

        # Hashes should be different
        assert compiled1.drive_hash != compiled2.drive_hash

    def test_identical_dac_configs_have_same_hash(self, tmp_path):
        """Identical DAC configurations produce identical drive_hash values."""
        source = Path("configs/L0_0_minimal")

        # Compile twice with identical DAC config
        for i in range(1, 3):
            dest = tmp_path / f"config{i}"
            shutil.copytree(source, dest)
            dac = {
                "drive_as_code": {
                    "version": "1.0",
                    "modifiers": {},
                    "extrinsic": {"type": "multiplicative", "base": 1.0, "bars": ["energy"]},
                    "intrinsic": {"strategy": "rnd", "base_weight": 0.15},
                }
            }
            (dest / "drive_as_code.yaml").write_text(yaml.dump(dac))

        compiler1 = UniverseCompiler()
        compiled1 = compiler1.compile(tmp_path / "config1", use_cache=False)

        compiler2 = UniverseCompiler()
        compiled2 = compiler2.compile(tmp_path / "config2", use_cache=False)

        # Hashes should be identical
        assert compiled1.drive_hash == compiled2.drive_hash

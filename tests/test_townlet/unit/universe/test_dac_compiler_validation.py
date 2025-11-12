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
        """Compilation succeeds when drive_as_code.yaml is missing (optional for now)."""
        # Copy L0_0_minimal (which doesn't have drive_as_code.yaml)
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Should compile successfully without DAC
        compiler = UniverseCompiler()
        compiled = compiler.compile(dest, use_cache=False)

        # DAC fields should be None
        assert compiled.dac_config is None
        assert compiled.drive_hash is None

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

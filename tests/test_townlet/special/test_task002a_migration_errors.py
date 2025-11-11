"""TASK-002A specific test: Migration error messages.

This test validates that users get helpful error messages when substrate.yaml is missing.
This is migration-specific behavior that can be removed once all configs have substrate.yaml.
"""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


def test_missing_substrate_yaml_raises_helpful_error(tmp_path):
    """TASK-002A: Should fail fast with migration instructions when substrate.yaml missing."""
    import shutil

    # Create config pack directory without substrate.yaml
    config_pack = tmp_path / "test_config"
    test_config = Path("configs/test")
    shutil.copytree(test_config, config_pack)
    (config_pack / "substrate.yaml").unlink()

    compiler = UniverseCompiler()

    # Attempt to compile without substrate.yaml
    with pytest.raises(CompilationError) as exc_info:
        compiler.compile(config_pack)

    # Verify error message contains migration instructions
    error_msg = str(exc_info.value)
    assert "substrate" in error_msg.lower()
    assert "substrate.yaml" in error_msg

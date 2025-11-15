"""Test that compiler respects explicit vision_range values without hidden defaults.

BUG-18: Compiler should not inject hidden default vision_range=3 for POMDP local_window.

The no-defaults principle requires:
- vision_range must be explicitly specified in config (Pydantic enforces this)
- Compiler must respect ALL valid values, including vision_range=0
- No silent fallbacks that override operator's explicit choices

This test verifies that vision_range=0 (valid per schema: ge=0) produces a 1×1 local window,
not a 5×5 window from hidden default of 3.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


class TestVisionRangeNoDefaults:
    """Test that compiler respects explicit vision_range without hidden defaults."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory with minimal POMDP config."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "test_vision_range_0"
        config_dir.mkdir()

        # Copy L2 config as base and modify vision_range to 0
        l2_source = Path("configs/L2_partial_observability")

        # Copy all required files
        for file in [
            "substrate.yaml",
            "bars.yaml",
            "cascades.yaml",
            "affordances.yaml",
            "cues.yaml",
            "training.yaml",
            "variables_reference.yaml",
            "drive_as_code.yaml",
            "brain.yaml",
            "action_config.yaml",
        ]:
            if (l2_source / file).exists():
                shutil.copy(l2_source / file, config_dir / file)

        # Modify training.yaml to set vision_range: 0
        training_yaml = config_dir / "training.yaml"
        content = training_yaml.read_text()
        # Replace vision_range: 2 with vision_range: 0
        modified = content.replace("vision_range: 2", "vision_range: 0")
        training_yaml.write_text(modified)

        yield config_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_compiler_respects_vision_range_zero(self, temp_config_dir):
        """Compiler should respect vision_range=0 (1×1 window), not default to 3 (5×5).

        BUG-18: The compiler had `vision_range = raw_configs.environment.vision_range or 3`
        which silently changed explicit 0 → 3, violating no-defaults principle.

        Expected behavior:
        - vision_range=0 → local_window size = (2*0+1)² = 1 (single cell)
        - NOT vision_range=3 → local_window size = (2*3+1)² = 49 (5×5 grid)
        """
        compiler = UniverseCompiler()
        compiled = compiler.compile(temp_config_dir)

        # Find local_window variable in compiled universe
        local_window_var = None
        for var in compiled.variables_reference:
            if var.id == "local_window":
                local_window_var = var
                break

        assert local_window_var is not None, "local_window variable not found in compiled universe"

        # For vision_range=0, window should be (2*0+1)×(2*0+1) = 1×1 = 1 cell
        expected_window_size = 1  # (2*0+1)² = 1
        actual_window_size = local_window_var.dims

        assert actual_window_size == expected_window_size, (
            f"Compiler should respect vision_range=0 (1×1 window), but got {actual_window_size} dims. "
            f"Expected {expected_window_size} (1×1), not 49 (7×7 from hidden default=3). "
            f"This violates the no-defaults principle."
        )

    def test_compiler_respects_vision_range_two(self):
        """Baseline test: vision_range=2 should produce 5×5 window (25 cells)."""
        compiler = UniverseCompiler()
        config_dir = Path("configs/L2_partial_observability")
        compiled = compiler.compile(config_dir)

        # Find local_window variable
        local_window_var = None
        for var in compiled.variables_reference:
            if var.id == "local_window":
                local_window_var = var
                break

        assert local_window_var is not None, "local_window variable not found"

        # For vision_range=2, window should be (2*2+1)×(2*2+1) = 5×5 = 25 cells
        expected_window_size = 25  # (2*2+1)² = 25
        actual_window_size = local_window_var.dims

        assert actual_window_size == expected_window_size, f"Expected 5×5 window (25 cells) for vision_range=2, got {actual_window_size}"

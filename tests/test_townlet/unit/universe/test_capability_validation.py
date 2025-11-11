"""Tests for capability validation in UniverseCompiler."""

import shutil
from pathlib import Path

import pytest
import yaml

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


class TestPrerequisiteValidation:
    """Test prerequisite capability validation."""

    def test_prerequisite_with_valid_affordance_ids_passes(self, tmp_path: Path):
        """Prerequisites referencing existing affordances should pass validation."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with valid prerequisite
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Foundation"
    name: "Foundation Course"
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.1

  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["Foundation"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Foundation Course", "Advanced Course"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        # Should not raise
        universe = compiler.compile(target)
        assert universe is not None

    def test_prerequisite_with_invalid_affordance_id_fails(self, tmp_path: Path):
        """Prerequisites referencing non-existent affordances should fail validation."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with INVALID prerequisite
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["NonExistentCourse"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Advanced Course"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(target)

        error_msg = str(exc_info.value)
        assert "NonExistentCourse" in error_msg
        assert "prerequisite" in error_msg.lower()
        assert "does not exist" in error_msg.lower() or "not found" in error_msg.lower()

    def test_prerequisite_with_multiple_invalid_ids_reports_all(self, tmp_path: Path):
        """Prerequisites with multiple invalid IDs should report all errors."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with multiple INVALID prerequisites
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Advanced"
    name: "Advanced Course"
    capabilities:
      - type: prerequisite
        required_affordances: ["Missing1", "Missing2", "Missing3"]
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Advanced Course"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(target)

        error_msg = str(exc_info.value)
        assert "Missing1" in error_msg
        assert "Missing2" in error_msg
        assert "Missing3" in error_msg


class TestProbabilisticValidation:
    """Test probabilistic capability validation."""

    def test_probabilistic_with_both_outcomes_passes(self, tmp_path: Path):
        """Probabilistic capability with both success and failure effects should pass."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with probabilistic + complete pipeline
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_start:
        - meter: energy
          amount: -0.05
      on_completion:
        - meter: energy
          amount: 0.2
      on_failure:
        - meter: energy
          amount: -0.1
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Slot Machine"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        # Should not raise
        universe = compiler.compile(target)
        assert universe is not None

    def test_probabilistic_without_on_failure_warns(self, tmp_path: Path):
        """Probabilistic capability without on_failure should emit warning."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with probabilistic but NO on_failure
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_completion:
        - meter: energy
          amount: 0.2
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Slot Machine"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(target)

        error_msg = str(exc_info.value)
        assert "probabilistic" in error_msg.lower()
        assert "on_failure" in error_msg or "failure" in error_msg.lower()
        assert "Casino" in error_msg

    def test_probabilistic_without_on_completion_warns(self, tmp_path: Path):
        """Probabilistic capability without on_completion should emit warning."""
        # Copy L0 config as base
        source = Path("configs/L0_0_minimal")
        target = tmp_path / "test_config"
        shutil.copytree(source, target)

        # Replace affordances.yaml with probabilistic but NO on_completion
        (target / "affordances.yaml").write_text(
            """
affordances:
  - id: "Casino"
    name: "Slot Machine"
    capabilities:
      - type: probabilistic
        success_probability: 0.3
    effect_pipeline:
      on_failure:
        - meter: energy
          amount: -0.1
"""
        )

        # Update training.yaml to reference new affordances
        training_path = target / "training.yaml"
        training = yaml.safe_load(training_path.read_text())
        training["environment"]["enabled_affordances"] = ["Slot Machine"]
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(target)

        error_msg = str(exc_info.value)
        assert "probabilistic" in error_msg.lower()
        assert "on_completion" in error_msg or "success" in error_msg.lower()
        assert "Casino" in error_msg

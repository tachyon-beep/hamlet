"""
Cascade Configuration Module

Provides type-safe loading and validation of cascade configuration from YAML files.
Uses Pydantic for schema validation and error reporting.

Configuration Files:
- bars.yaml: Meter definitions and base depletion rates
- cascades.yaml: Threshold-based cascade effects and modulations
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Bars Configuration (bars.yaml)
# ============================================================================


class BarConfig(BaseModel):
    """Configuration for a single meter (bar)."""

    name: str = Field(description="Meter name (e.g., 'energy', 'health')")
    index: int = Field(ge=0, le=7, description="Meter index in tensor [0-7]")
    tier: Literal["pivotal", "primary", "secondary", "resource"] = Field(description="Tier in cascade hierarchy")
    range: tuple[float, float] = Field(default=(0.0, 1.0), description="Min and max values")
    initial: float = Field(ge=0.0, le=1.0, description="Initial value at spawn")
    base_depletion: float = Field(ge=0.0, description="Passive decay per step")
    description: str = Field(description="Human-readable description")

    # Optional fields for documentation
    key_insight: str | None = None
    special: str | None = None
    cascade_pattern: str | None = None

    @field_validator("range")
    @classmethod
    def validate_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate that range is [0.0, 1.0]."""
        if v != (0.0, 1.0):
            raise ValueError(f"Range must be (0.0, 1.0), got {v}")
        return v


class TerminalCondition(BaseModel):
    """Configuration for a terminal (death) condition."""

    meter: str = Field(description="Meter name to check")
    operator: Literal["<=", ">=", "<", ">", "=="] = Field(description="Comparison operator")
    value: float = Field(description="Threshold value")
    description: str = Field(description="Human-readable description")


class BarsConfig(BaseModel):
    """Complete bars.yaml configuration."""

    version: str = Field(description="Config version")
    description: str = Field(description="Config description")
    bars: list[BarConfig] = Field(description="List of meter configurations")
    terminal_conditions: list[TerminalCondition] = Field(description="Death conditions")
    notes: list[str] | None = None

    @field_validator("bars")
    @classmethod
    def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
        """Validate bar list."""
        if len(v) != 8:
            raise ValueError(f"Expected 8 bars, got {len(v)}")

        # Check all indices are unique and cover 0-7
        indices = {bar.index for bar in v}
        if indices != {0, 1, 2, 3, 4, 5, 6, 7}:
            raise ValueError(f"Bar indices must be 0-7, got {sorted(indices)}")

        # Check all names are unique
        names = [bar.name for bar in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate bar names found: {names}")

        return v


# ============================================================================
# Cascades Configuration (cascades.yaml)
# ============================================================================


class ModulationConfig(BaseModel):
    """Configuration for a depletion rate modulation."""

    name: str = Field(description="Modulation name")
    description: str = Field(description="Human-readable description")
    source: str = Field(description="Source meter name")
    target: str = Field(description="Target meter name")
    type: Literal["depletion_multiplier"] = Field(description="Modulation type (currently only depletion_multiplier)")

    # Gradient multiplier parameters
    base_multiplier: float = Field(gt=0.0, description="Base multiplier (at source=100%)")
    range: float = Field(gt=0.0, description="Multiplier range (base + range = max)")
    baseline_depletion: float = Field(ge=0.0, description="Baseline depletion rate to modulate")

    # Optional documentation
    note: str | None = None


class CascadeConfig(BaseModel):
    """Configuration for a threshold-based cascade effect."""

    name: str = Field(description="Cascade name")
    description: str = Field(description="Human-readable description")
    category: str = Field(description="Category (e.g., 'primary_to_pivotal', 'secondary_to_primary')")

    source: str = Field(description="Source meter name")
    source_index: int = Field(ge=0, le=7, description="Source meter index")
    target: str = Field(description="Target meter name")
    target_index: int = Field(ge=0, le=7, description="Target meter index")

    threshold: float = Field(gt=0.0, le=1.0, description="Threshold below which cascade applies")
    strength: float = Field(gt=0.0, description="Penalty strength (gradient factor)")

    # Optional documentation fields
    teaching_note: str | None = None
    why_it_matters: str | None = None


class CascadesConfig(BaseModel):
    """Complete cascades.yaml configuration."""

    version: str = Field(description="Config version")
    description: str = Field(description="Config description")
    math_type: Literal["gradient_penalty", "multiplier"] = Field(description="Cascade math approach")

    modulations: list[ModulationConfig] = Field(description="Depletion rate modulations")
    cascades: list[CascadeConfig] = Field(description="Threshold-based cascades")
    execution_order: list[str] = Field(description="Cascade execution order")

    # Optional documentation
    notes: list[str] | None = None
    teaching_insights: dict | None = None

    @field_validator("cascades")
    @classmethod
    def validate_cascades(cls, v: list[CascadeConfig]) -> list[CascadeConfig]:
        """Validate cascade list."""
        # Check all names are unique
        names = [cascade.name for cascade in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate cascade names found: {names}")

        return v


# ============================================================================
# Combined Configuration
# ============================================================================


class EnvironmentConfig(BaseModel):
    """Combined environment configuration from bars.yaml and cascades.yaml."""

    bars: BarsConfig
    cascades: CascadesConfig

    def get_bar_by_name(self, name: str) -> BarConfig:
        """Get bar config by name."""
        for bar in self.bars.bars:
            if bar.name == name:
                return bar
        raise ValueError(f"Bar '{name}' not found")

    def get_bar_by_index(self, index: int) -> BarConfig:
        """Get bar config by index."""
        for bar in self.bars.bars:
            if bar.index == index:
                return bar
        raise ValueError(f"Bar with index {index} not found")

    def get_cascade_by_name(self, name: str) -> CascadeConfig:
        """Get cascade config by name."""
        for cascade in self.cascades.cascades:
            if cascade.name == name:
                return cascade
        raise ValueError(f"Cascade '{name}' not found")


# ============================================================================
# Config Loading Functions
# ============================================================================


def load_yaml_file(filepath: Path) -> Any:
    """
    Load and parse a YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML as dict

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath) as f:
        return yaml.safe_load(f)


def load_bars_config(filepath: Path) -> BarsConfig:
    """
    Load and validate bars.yaml configuration.

    Args:
        filepath: Path to bars.yaml

    Returns:
        Validated BarsConfig

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If config is invalid
    """
    data = load_yaml_file(filepath)
    return BarsConfig(**data)


def load_cascades_config(filepath: Path) -> CascadesConfig:
    """
    Load and validate cascades.yaml configuration.

    Args:
        filepath: Path to cascades.yaml

    Returns:
        Validated CascadesConfig

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If config is invalid
    """
    data = load_yaml_file(filepath)
    return CascadesConfig(**data)


def load_environment_config(config_dir: Path) -> EnvironmentConfig:
    """
    Load complete environment configuration from directory.

    Args:
        config_dir: Directory containing bars.yaml and cascades.yaml

    Returns:
        Combined EnvironmentConfig

    Raises:
        FileNotFoundError: If files don't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If configs are invalid
    """
    bars_path = config_dir / "bars.yaml"
    cascades_path = config_dir / "cascades.yaml"

    bars = load_bars_config(bars_path)
    cascades = load_cascades_config(cascades_path)

    return EnvironmentConfig(bars=bars, cascades=cascades)


def load_default_config() -> EnvironmentConfig:
    """
    Load default configuration from configs/ directory.

    Returns:
        Default EnvironmentConfig

    Raises:
        FileNotFoundError: If config files don't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If configs are invalid
    """
    # Assume we're in src/townlet/environment/ and configs/ is at project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    config_dir = project_root / "configs" / "test"

    return load_environment_config(config_dir)

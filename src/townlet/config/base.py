"""Base configuration utilities for UNIVERSE_AS_CODE validation.

Provides common utilities for loading and validating configuration files.
Follows no-defaults principle: all behavioral parameters must be explicit.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError


def load_yaml_section(config_dir: Path, filename: str, section: str) -> dict[str, Any]:
    """Load a section from a YAML file.

    Args:
        config_dir: Config pack directory (e.g., configs/L0_0_minimal)
        filename: YAML filename (e.g., "training.yaml")
        section: Top-level section name (e.g., "training", "environment")

    Returns:
        Dict of configuration data from the specified section

    Raises:
        FileNotFoundError: If file doesn't exist with helpful path information
        KeyError: If section doesn't exist with list of available sections

    Example:
        >>> data = load_yaml_section(Path("configs/L0_0_minimal"), "training.yaml", "training")
        >>> print(data["epsilon_decay"])
        0.99
    """
    config_file = config_dir / filename
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Expected: {config_dir}/{filename}\n"
            f"Check that config pack directory exists and contains required files."
        )

    with open(config_file) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {config_file}")

    if section not in data:
        available_sections = list(data.keys()) if isinstance(data, dict) else []
        raise KeyError(
            f"Section '{section}' not found in {filename}\n"
            f"Available sections: {available_sections}\n"
            f"File path: {config_file}"
        )

    return data[section]


def format_validation_error(error: ValidationError, context: str) -> str:
    """Format Pydantic ValidationError with helpful context.

    Transforms cryptic Pydantic errors into actionable error messages for operators.
    Follows UNIVERSE_AS_CODE principle: errors should guide operators to fix configs.

    Args:
        error: Pydantic validation error from DTO instantiation
        context: Context string (e.g., "training.yaml", "environment section")

    Returns:
        Formatted error message with fix suggestions and template references

    Example:
        >>> try:
        ...     TrainingConfig()  # Missing required fields
        ... except ValidationError as e:
        ...     print(format_validation_error(e, "training.yaml"))
        ❌ training.yaml VALIDATION FAILED

        [Pydantic error details]

        All parameters must be explicitly specified.
        See configs/templates/training.yaml for reference.
    """
    lines = [
        f"❌ {context.upper()} VALIDATION FAILED",
        "",
        str(error),
        "",
        "All parameters must be explicitly specified (no-defaults principle).",
        "Each parameter affects universe behavior and must be consciously chosen.",
        "",
        "Fix: Add missing/invalid fields to your config YAML.",
        "Reference: See configs/templates/ for annotated examples.",
    ]
    return "\n".join(lines)

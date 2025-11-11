"""Temporary-directory helper fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

__all__ = ["temp_test_dir", "temp_yaml_file", "recording_output_dir"]


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for tests."""

    return tmp_path


@pytest.fixture
def temp_yaml_file(temp_test_dir: Path) -> Path:
    """Provide a temporary YAML file path."""

    return temp_test_dir / "test.yaml"


@pytest.fixture
def recording_output_dir(temp_test_dir: Path) -> Path:
    """Provide a recording output directory under the temporary root."""

    target = temp_test_dir / "recordings"
    target.mkdir(exist_ok=True)
    return target

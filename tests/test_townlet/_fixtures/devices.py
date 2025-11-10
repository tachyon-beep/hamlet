"""Device selection fixtures."""

from __future__ import annotations

import pytest
import torch

__all__ = ["device", "cpu_device"]


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for deterministic behavior."""

    return torch.device("cpu")

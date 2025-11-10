"""Top-level pytest configuration for the Townlet test suite."""

from __future__ import annotations

import pytest
import torch

pytest_plugins = [
    "tests.test_townlet._fixtures.config",
    "tests.test_townlet._fixtures.devices",
    "tests.test_townlet._fixtures.temp",
    "tests.test_townlet._fixtures.environment",
    "tests.test_townlet._fixtures.networks",
    "tests.test_townlet._fixtures.training",
    "tests.test_townlet._fixtures.variable_meters",
    "tests.test_townlet._fixtures.database",
    "tests.test_townlet._fixtures.utils",
]


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used across the suite."""

    config.addinivalue_line("markers", "slow: mark test as slow (run with --runslow)")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (skipped if no CUDA)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU tests automatically when CUDA is unavailable."""

    if torch.cuda.is_available():
        return

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

"""Focused tests for UnifiedServer helper methods."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from townlet.demo.unified_server import UnifiedServer


@pytest.fixture
def minimal_config_dir(tmp_path: Path) -> Path:
    """Create a minimal config directory with a training.yaml file."""

    config_dir = tmp_path / "config_pack"
    config_dir.mkdir()
    training_yaml = config_dir / "training.yaml"
    training_yaml.write_text("run_metadata:\n  output_subdir: 'Phase 1: Demo'\nenvironment:\n  grid_size: 8\n")
    return config_dir


@pytest.fixture
def unified_server(minimal_config_dir: Path) -> UnifiedServer:
    """Unified server instance pointing at the minimal config pack."""

    return UnifiedServer(
        config_dir=str(minimal_config_dir),
        total_episodes=10,
        training_config_path=str(minimal_config_dir / "training.yaml"),
        checkpoint_dir=str(minimal_config_dir / "checkpoints"),
    )


def test_determine_run_directory_uses_sanitised_output_subdir(unified_server: UnifiedServer):
    run_dir = unified_server._determine_run_directory("2025-11-03_120000")

    assert run_dir == Path("runs") / "Phase_1_Demo" / "2025-11-03_120000"


def test_persist_config_snapshot_is_idempotent(unified_server: UnifiedServer, tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    unified_server._persist_config_snapshot(run_root)
    snapshot_dir = run_root / "config_snapshot"
    assert (snapshot_dir / "training.yaml").exists()

    # Second call should no-op (directory already exists)
    snapshot_mtime = (snapshot_dir / "training.yaml").stat().st_mtime
    unified_server._persist_config_snapshot(run_root)
    assert (snapshot_dir / "training.yaml").stat().st_mtime == snapshot_mtime


def test_setup_file_logging_attaches_handler(unified_server: UnifiedServer, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    run_dir = tmp_path / "runs" / "L1"
    run_dir.mkdir(parents=True)

    unified_server._setup_file_logging(run_dir)
    log_path = run_dir / "training.log"

    with caplog.at_level(logging.INFO):
        logging.getLogger().info("hello-from-test")

    unified_server.stop()  # ensures handler is removed

    assert log_path.exists()
    assert "hello-from-test" in log_path.read_text()

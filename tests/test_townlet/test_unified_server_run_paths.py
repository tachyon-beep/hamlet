"""
Tests for UnifiedServer run directory derivation (config-driven runs layout).
"""

from pathlib import Path

import textwrap

from townlet.demo.unified_server import UnifiedServer


def _make_config(tmp_path, filename, body: str) -> Path:
    path = tmp_path / filename
    path.write_text(textwrap.dedent(body))
    return path


def test_run_directory_uses_configured_output_subdir(tmp_path):
    config_path = _make_config(
        tmp_path,
        "level_0_minimal.yaml",
        """
        run_metadata:
          output_subdir: L0_minimal
        """,
    )

    server = UnifiedServer(
        config_path=str(config_path),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "L0_minimal" / "2025-01-01_010203"


def test_run_directory_falls_back_to_legacy_detection(tmp_path):
    config_path = _make_config(
        tmp_path,
        "level_2_custom.yaml",
        """
        environment:
          grid_size: 8
        """,
    )

    server = UnifiedServer(
        config_path=str(config_path),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "L2_partial_observability" / "2025-01-01_010203"


def test_run_directory_sanitises_invalid_characters(tmp_path):
    config_path = _make_config(
        tmp_path,
        "custom.yaml",
        """
        run_metadata:
          output_subdir: "My Fancy Level!"
        """,
    )

    server = UnifiedServer(
        config_path=str(config_path),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "My_Fancy_Level" / "2025-01-01_010203"

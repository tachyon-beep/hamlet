"""
Tests for UnifiedServer run directory derivation (config-driven runs layout).
"""

import textwrap

from pathlib import Path

from townlet.demo.unified_server import UnifiedServer


def _make_config_pack(tmp_path, folder_name: str, body: str) -> tuple[Path, Path]:
    pack_dir = tmp_path / folder_name
    pack_dir.mkdir()
    training_path = pack_dir / "training.yaml"
    training_path.write_text(textwrap.dedent(body))
    return pack_dir, training_path


def test_run_directory_uses_configured_output_subdir(tmp_path):
    config_dir, training_path = _make_config_pack(
        tmp_path,
        "level_0_minimal",
        """
        run_metadata:
          output_subdir: L0_minimal
        """,
    )

    server = UnifiedServer(
        config_dir=str(config_dir),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
        training_config_path=str(training_path),
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "L0_minimal" / "2025-01-01_010203"


def test_run_directory_falls_back_to_legacy_detection(tmp_path):
    config_dir, training_path = _make_config_pack(
        tmp_path,
        "level_2_custom",
        """
        environment:
          grid_size: 8
        """,
    )

    server = UnifiedServer(
        config_dir=str(config_dir),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
        training_config_path=str(training_path),
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "L2_partial_observability" / "2025-01-01_010203"


def test_run_directory_sanitises_invalid_characters(tmp_path):
    config_dir, training_path = _make_config_pack(
        tmp_path,
        "custom",
        """
        run_metadata:
          output_subdir: "My Fancy Level!"
        """,
    )

    server = UnifiedServer(
        config_dir=str(config_dir),
        total_episodes=10,
        checkpoint_dir=None,
        inference_port=8766,
        training_config_path=str(training_path),
    )

    run_dir = server._determine_run_directory(timestamp="2025-01-01_010203")
    assert run_dir == Path("runs") / "My_Fancy_Level" / "2025-01-01_010203"

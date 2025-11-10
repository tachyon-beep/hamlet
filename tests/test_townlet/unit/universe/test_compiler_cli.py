"""Tests for the `python -m townlet.compiler` CLI."""

from __future__ import annotations

import json

from townlet.compiler import __main__ as compiler_cli


def test_cli_compile_creates_cache(config_pack_factory, capsys) -> None:
    config_dir = config_pack_factory()
    cache_path = config_dir / ".compiled" / "universe.msgpack"

    exit_code = compiler_cli.main(["compile", str(config_dir)])

    assert exit_code == 0
    assert cache_path.exists()

    out = capsys.readouterr().out
    assert "Compilation succeeded" in out
    assert "Universe" in out


def test_cli_inspect_displays_metadata(config_pack_factory, capsys) -> None:
    config_dir = config_pack_factory()
    cache_path = config_dir / ".compiled" / "universe.msgpack"
    compiler_cli.main(["compile", str(config_dir)])
    capsys.readouterr()  # Clear compile output

    exit_code = compiler_cli.main(["inspect", str(cache_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Artifact path" in out
    assert "Config Hash" in out


def test_cli_inspect_json_output(config_pack_factory, capsys) -> None:
    config_dir = config_pack_factory()
    cache_path = config_dir / ".compiled" / "universe.msgpack"
    compiler_cli.main(["compile", str(config_dir)])
    capsys.readouterr()

    exit_code = compiler_cli.main(["inspect", str(cache_path), "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["universe_name"] == config_dir.name


def test_cli_validate_skips_cache(config_pack_factory, capsys) -> None:
    config_dir = config_pack_factory()
    cache_dir = config_dir / ".compiled"

    exit_code = compiler_cli.main(["validate", str(config_dir)])

    assert exit_code == 0
    assert not cache_dir.exists(), "Validate should not write cache artifacts"
    out = capsys.readouterr().out
    assert "Validation succeeded" in out

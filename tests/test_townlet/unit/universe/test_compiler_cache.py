"""Tests for compiler cache helper utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


def _copy_config_pack(tmp_path: Path, pack_name: str = "L0_0_minimal") -> Path:
    dest = tmp_path / pack_name
    shutil.copytree(Path("configs") / pack_name, dest)
    return dest


def test_cache_directory_resolves_inside_config_dir(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    config_dir = tmp_path / "pack"
    config_dir.mkdir()

    cache_dir = compiler._cache_directory_for(config_dir)

    assert cache_dir == config_dir / ".compiled"


def test_prepare_cache_directory_creates_directory(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    cache_dir = tmp_path / ".compiled"

    compiler._prepare_cache_directory(cache_dir)

    assert cache_dir.exists()
    assert cache_dir.is_dir()


def test_prepare_cache_directory_errors_when_path_is_file(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    cache_dir = tmp_path / ".compiled"
    cache_dir.write_text("not a directory")

    with pytest.raises(RuntimeError):
        compiler._prepare_cache_directory(cache_dir)


def test_cache_artifact_path_points_inside_cache_dir(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    config_dir = tmp_path / "pack"
    config_dir.mkdir()

    artifact_path = compiler._cache_artifact_path(config_dir)

    assert artifact_path == config_dir / ".compiled" / "universe.msgpack"


def test_compile_uses_cache_when_hash_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = _copy_config_pack(tmp_path)

    builder = UniverseCompiler()
    builder.compile(config_dir, use_cache=True)

    flag = {"stage1_called": False}

    def _fail_stage1(self, _config_dir: Path):
        flag["stage1_called"] = True
        raise AssertionError("Stage 1 should not run when loading from cache")

    monkeypatch.setattr(UniverseCompiler, "_stage_1_parse_individual_files", _fail_stage1)

    cached_compiler = UniverseCompiler()
    compiled = cached_compiler.compile(config_dir, use_cache=True)

    assert not flag["stage1_called"]
    assert compiled.metadata.universe_name == config_dir.name


def test_compile_rebuilds_cache_when_hash_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = _copy_config_pack(tmp_path)
    builder = UniverseCompiler()
    builder.compile(config_dir, use_cache=True)

    training_path = config_dir / "training.yaml"
    training_text = training_path.read_text()
    training_path.write_text(training_text.replace("max_episodes: 500", "max_episodes: 501"))

    original_stage1 = UniverseCompiler._stage_1_parse_individual_files
    counter = {"calls": 0}

    def _wrapped_stage1(self, cfg_dir: Path):
        counter["calls"] += 1
        return original_stage1(self, cfg_dir)

    monkeypatch.setattr(UniverseCompiler, "_stage_1_parse_individual_files", _wrapped_stage1)

    refreshed_compiler = UniverseCompiler()
    refreshed_compiler.compile(config_dir, use_cache=True)

    assert counter["calls"] == 1


def test_compile_recovers_from_corrupted_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = _copy_config_pack(tmp_path)
    compiler = UniverseCompiler()
    compiler.compile(config_dir, use_cache=True)

    cache_path = compiler._cache_artifact_path(config_dir)
    cache_path.write_bytes(b"corrupted")

    original_stage1 = UniverseCompiler._stage_1_parse_individual_files
    counter = {"calls": 0}

    def _wrapped_stage1(self, cfg_dir: Path):
        counter["calls"] += 1
        return original_stage1(self, cfg_dir)

    monkeypatch.setattr(UniverseCompiler, "_stage_1_parse_individual_files", _wrapped_stage1)

    UniverseCompiler().compile(config_dir, use_cache=True)

    assert counter["calls"] == 1

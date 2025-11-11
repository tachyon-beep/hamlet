"""Tests for the UniverseCompiler Stage 1 loader."""

import shutil
from pathlib import Path

import pytest
import yaml

from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError
from townlet.universe.symbol_table import UniverseSymbolTable


@pytest.mark.parametrize("pack_name", ["L0_0_minimal", "L1_full_observability"])
def test_stage1_parses_config_pack(pack_name: str):
    compiler = UniverseCompiler()
    config_dir = Path("configs") / pack_name

    raw_configs = compiler._stage_1_parse_individual_files(config_dir)

    assert raw_configs.training.max_episodes > 0
    # variables_reference can be empty (auto-generated), just check it's a list
    assert raw_configs.variables_reference is not None
    assert isinstance(raw_configs.variables_reference, list)
    assert len(raw_configs.global_actions.actions) > 0


def test_compile_returns_compiled_universe():
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    assert isinstance(compiled, CompiledUniverse)
    assert compiled.metadata.universe_name == "L0_0_minimal"


def test_stage2_builds_symbol_table():
    compiler = UniverseCompiler()
    raw_configs = RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))

    table = compiler._stage_2_build_symbol_tables(raw_configs)

    assert "energy" in table.meters
    assert table.actions


def test_stage1_reports_missing_files(tmp_path: Path) -> None:
    compiler = UniverseCompiler()

    with pytest.raises(CompilationError) as exc_info:
        compiler._stage_1_parse_individual_files(tmp_path)

    error = exc_info.value
    assert error.stage == "Stage 1: Parse"
    combined = "\n".join(error.errors)
    assert "variables_reference.yaml" in combined
    assert "hamlet_config" in combined


def test_stage1_reports_invalid_yaml(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    source_pack = Path("configs/L0_0_minimal")
    test_pack = tmp_path / "broken_pack"
    shutil.copytree(source_pack, test_pack)
    (test_pack / "bars.yaml").write_text("bars: [broken::")

    with pytest.raises(CompilationError) as exc_info:
        compiler._stage_1_parse_individual_files(test_pack)

    combined = "\n".join(exc_info.value.errors)
    assert "bars" in combined or "YAML" in combined


def test_compile_executes_stage2_before_stage3(monkeypatch) -> None:
    compiler = UniverseCompiler()
    called: dict[str, bool] = {"stage2": False, "stage3": False}

    def fake_stage2(self: UniverseCompiler, raw_configs: RawConfigs) -> UniverseSymbolTable:  # type: ignore[override]
        called["stage2"] = True
        return UniverseSymbolTable()

    def fake_stage3(self: UniverseCompiler, raw_configs: RawConfigs, table: UniverseSymbolTable, errors):  # type: ignore[override]
        called["stage3"] = True
        errors.add_error("boom")

    monkeypatch.setattr(UniverseCompiler, "_stage_2_build_symbol_tables", fake_stage2, raising=False)
    monkeypatch.setattr(UniverseCompiler, "_stage_3_resolve_references", fake_stage3, raising=False)

    with pytest.raises(CompilationError):
        compiler.compile(Path("configs/L0_0_minimal"), use_cache=False)

    assert called["stage2"] is True
    assert called["stage3"] is True


def test_stage1_rejects_unknown_enabled_action(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    source_pack = Path("configs/L0_0_minimal")
    test_pack = tmp_path / "invalid_actions"
    shutil.copytree(source_pack, test_pack)

    training_path = test_pack / "training.yaml"
    data = yaml.safe_load(training_path.read_text())
    data.setdefault("training", {})["enabled_actions"] = ["UP", "DOES_NOT_EXIST"]
    training_path.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(CompilationError) as exc_info:
        compiler._stage_1_parse_individual_files(test_pack)

    combined = "\n".join(exc_info.value.errors)
    assert "UAC-ACT-001" in combined

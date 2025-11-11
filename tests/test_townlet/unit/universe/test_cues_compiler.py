"""Unit tests for the CuesCompiler helper."""

from pathlib import Path

import pytest

from townlet.config.cues import VisualCueConfig
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.cues_compiler import CuesCompiler
from townlet.universe.errors import CompilationError, CompilationErrorCollector, CompilationMessage


def _formatter(code: str, message: str, location: str | None) -> CompilationMessage:
    return CompilationMessage(code=code, message=message, location=location)


@pytest.fixture(scope="module")
def base_raw_configs() -> RawConfigs:
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


@pytest.fixture(scope="module")
def symbol_table(base_raw_configs: RawConfigs):
    compiler = UniverseCompiler()
    return compiler._stage_2_build_symbol_tables(base_raw_configs)


def test_cues_compiler_detects_invalid_meter(base_raw_configs: RawConfigs, symbol_table) -> None:
    cues_compiler = CuesCompiler()
    cues = base_raw_configs.hamlet_config.cues

    mutated_simple = list(cues.simple_cues)
    original = mutated_simple[0]
    mutated_simple[0] = original.model_copy(
        update={
            "condition": original.condition.model_copy(update={"meter": "ghost_meter"}),
        }
    )
    mutated_cues = cues.model_copy(update={"simple_cues": mutated_simple})

    collector = CompilationErrorCollector(stage="Cues")
    cues_compiler.validate(mutated_cues, symbol_table, collector, _formatter)

    with pytest.raises(CompilationError):
        collector.check_and_raise("Cues")
    assert any(issue.code == "UAC-VAL-005" for issue in collector.issues)


def test_cues_compiler_detects_overlapping_visual_ranges(base_raw_configs: RawConfigs, symbol_table) -> None:
    cues_compiler = CuesCompiler()
    cues = base_raw_configs.hamlet_config.cues

    overlapping_visuals = {
        "energy": [
            VisualCueConfig(range=(0.0, 0.6), label="low"),
            VisualCueConfig(range=(0.5, 1.0), label="high"),
        ]
    }
    mutated_cues = cues.model_copy(update={"visual_cues": overlapping_visuals})

    collector = CompilationErrorCollector(stage="Cues")
    cues_compiler.validate(mutated_cues, symbol_table, collector, _formatter)

    with pytest.raises(CompilationError):
        collector.check_and_raise("Cues")
    assert any(issue.code == "UAC-VAL-009" for issue in collector.issues)

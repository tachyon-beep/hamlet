"""Tests for UniverseSymbolTable and Stage 2 registration."""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError
from townlet.universe.symbol_table import UniverseSymbolTable


@pytest.fixture(scope="module")
def raw_configs() -> RawConfigs:
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


def test_stage2_registers_entities(raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()
    table = compiler._stage_2_build_symbol_tables(raw_configs)

    assert table.get_meter_count() == len(raw_configs.bars)
    assert table.get_meter("energy").name == "energy"
    assert table.get_variable("energy")
    assert table.affordances
    assert table.get_affordance_count() == len(raw_configs.affordances)
    assert table.cascades
    assert table.cues


def test_meter_names_follow_index_order(raw_configs: RawConfigs) -> None:
    table = UniverseSymbolTable()
    for bar in raw_configs.bars:
        table.register_meter(bar)

    names = table.get_meter_names()
    expected = [bar.name for bar in sorted(raw_configs.bars, key=lambda b: b.index)]
    assert names == expected


def test_register_cues_rejects_duplicates(raw_configs: RawConfigs) -> None:
    table = UniverseSymbolTable()
    for cue in raw_configs.cues:
        table.register_cue(cue)

    with pytest.raises(CompilationError):
        table.register_cue(raw_configs.cues[0])


def test_register_action_by_id(raw_configs: RawConfigs) -> None:
    table = UniverseSymbolTable()
    action = raw_configs.global_actions.actions[0]
    table.register_action(action)

    assert table.get_action(action.id) is action
    with pytest.raises(CompilationError):
        table.register_action(action)

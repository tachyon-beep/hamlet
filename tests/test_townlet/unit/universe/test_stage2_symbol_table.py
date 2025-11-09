"""Comprehensive tests for UniverseSymbolTable and Stage 2 behavior."""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.errors import CompilationError
from townlet.universe.symbol_table import UniverseSymbolTable


def make_raw_configs() -> RawConfigs:
    return RawConfigs.from_config_dir(Path("configs/L0_0_minimal"))


def test_stage2_registers_all_entities():
    compiler = UniverseCompiler()
    raw_configs = make_raw_configs()

    table = compiler._stage_2_build_symbol_tables(raw_configs)

    assert table.get_meter_count() == len(raw_configs.bars)
    assert table.get_meter("energy").name == "energy"
    assert table.get_action(0)
    assert table.get_variable("energy")
    assert table.cascades
    assert table.affordances
    assert table.cues


def test_meter_name_helper_returns_sorted():
    table = UniverseSymbolTable()
    bars = RawConfigs.from_config_dir(Path("configs/L0_0_minimal")).bars
    for bar in bars:
        table.register_meter(bar)

    meter_names = table.get_meter_names()
    assert meter_names == sorted(meter_names)


def test_register_cues_rejects_duplicates():
    table = UniverseSymbolTable()
    raw_cues = RawConfigs.from_config_dir(Path("configs/L0_0_minimal")).cues
    table.register_cues(raw_cues)
    with pytest.raises(CompilationError):
        table.register_cues(raw_cues)


def test_register_action_by_id():
    table = UniverseSymbolTable()
    action = RawConfigs.from_config_dir(Path("configs/L0_0_minimal")).global_actions.actions[0]
    table.register_action(action)
    assert table.get_action(action.id) is action
    with pytest.raises(CompilationError):
        table.register_action(action)

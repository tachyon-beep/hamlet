"""Smoke tests for Stage 6 optimization placeholders."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs


def test_stage6_returns_placeholder_optimization_data() -> None:
    config_dir = Path("configs/L0_0_minimal")
    raw_configs = RawConfigs.from_config_dir(config_dir)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    metadata, _ = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)

    optimization = compiler._stage_6_optimize(raw_configs, metadata)

    assert optimization.base_depletions.shape[0] == metadata.meter_count
    assert optimization.action_mask_table is not None
    assert optimization.action_mask_table.shape[0] == 24
    assert optimization.action_mask_table.shape[1] == max(metadata.affordance_count, 1)
    assert set(optimization.affordance_position_map.keys()) == {aff.id for aff in raw_configs.affordances}
    assert torch.all(optimization.action_mask_table)


def test_stage6_infers_depletions_from_bars() -> None:
    config_dir = Path("configs/L0_0_minimal")
    raw_configs = RawConfigs.from_config_dir(config_dir)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    metadata, _ = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)

    optimization = compiler._stage_6_optimize(raw_configs, metadata)

    energy_bar = next(bar for bar in raw_configs.bars if bar.name == "energy")
    assert optimization.base_depletions[energy_bar.index].item() == pytest.approx(float(energy_bar.base_depletion))

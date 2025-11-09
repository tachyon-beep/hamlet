"""Unit tests for Stage 6 optimization data."""

from __future__ import annotations

from pathlib import Path

import torch

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.dto import UniverseMetadata


def _build_metadata(config_dir: Path) -> tuple[UniverseCompiler, RawConfigs, UniverseMetadata]:
    compiler = UniverseCompiler()
    raw_configs = RawConfigs.from_config_dir(config_dir)
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    metadata, _ = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)
    return compiler, raw_configs, metadata


def test_stage6_builds_cascade_and_modulation_tables() -> None:
    config_dir = Path("configs/L0_0_minimal")
    compiler, raw_configs, metadata = _build_metadata(config_dir)

    optimization = compiler._stage_6_optimize(raw_configs, metadata)

    energy_idx = metadata.meter_name_to_index["energy"]
    assert torch.isclose(optimization.base_depletions[energy_idx], torch.tensor(0.005))

    primary_cat = optimization.cascade_data["primary_to_pivotal"]
    sources = {(entry["source_idx"], entry["target_idx"]) for entry in primary_cat}
    assert (metadata.meter_name_to_index["satiation"], metadata.meter_name_to_index["health"]) in sources

    assert optimization.modulation_data, "Expected modulation entry from cascades.yaml"
    modulation = optimization.modulation_data[0]
    assert modulation["base_multiplier"] == 0.5
    assert modulation["range"] == 2.5


def test_stage6_builds_action_mask_with_operating_hours() -> None:
    config_dir = Path("configs/L0_0_minimal")
    compiler, raw_configs, metadata = _build_metadata(config_dir)

    optimization = compiler._stage_6_optimize(raw_configs, metadata)

    assert optimization.action_mask_table.shape == (24, metadata.affordance_count)

    bed_idx = next(idx for idx, aff in enumerate(raw_configs.affordances) if aff.name == "Bed")
    bar_idx = next(idx for idx, aff in enumerate(raw_configs.affordances) if aff.name == "Bar")

    assert torch.all(optimization.action_mask_table[:, bed_idx])
    # Bar open 18:00 - 04:00 (wrap). Verify 20:00 open, 10:00 closed.
    assert optimization.action_mask_table[20, bar_idx]
    assert not optimization.action_mask_table[10, bar_idx]


def test_stage6_invariants_on_cascades_and_modulations() -> None:
    config_dir = Path("configs/L0_5_dual_resource")
    compiler, raw_configs, metadata = _build_metadata(config_dir)

    optimization = compiler._stage_6_optimize(raw_configs, metadata)

    for category, entries in optimization.cascade_data.items():
        target_idxs = [entry["target_idx"] for entry in entries]
        assert target_idxs == sorted(target_idxs), f"{category} cascade targets must stay sorted"

        for entry in entries:
            assert 0 <= entry["source_idx"] < metadata.meter_count
            assert 0 <= entry["target_idx"] < metadata.meter_count

    modulation_targets = [entry["target_idx"] for entry in optimization.modulation_data]
    if modulation_targets:
        assert modulation_targets == sorted(modulation_targets), "Modulation targets must stay sorted"

    for entry in optimization.modulation_data:
        assert 0 <= entry["source_idx"] < metadata.meter_count
        assert 0 <= entry["target_idx"] < metadata.meter_count

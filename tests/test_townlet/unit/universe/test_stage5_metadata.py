"""Tests for Stage 5 metadata computation."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs


@pytest.fixture(scope="module")
def base_config_dir() -> Path:
    return Path("configs/L0_0_minimal")


@pytest.fixture(scope="module")
def base_raw_configs(base_config_dir: Path) -> RawConfigs:
    return RawConfigs.from_config_dir(base_config_dir)


def test_stage5_computes_metadata_and_observation_spec(base_config_dir: Path, base_raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(base_raw_configs)

    metadata, observation_spec, _ = compiler._stage_5_compute_metadata(base_config_dir, base_raw_configs, symbol_table)

    assert metadata.universe_name == base_config_dir.name
    assert metadata.meter_count == len(base_raw_configs.bars)
    assert metadata.affordance_count == len(base_raw_configs.affordances)
    assert metadata.action_count == len(base_raw_configs.global_actions.actions)
    assert metadata.observation_dim == observation_spec.total_dims
    assert metadata.config_hash
    assert metadata.provenance_id

    grid_field = observation_spec.get_field_by_name("obs_grid_encoding")
    assert grid_field.dims == 9
    assert grid_field.scope == "agent", "Grid encoding should stay agent scoped under full observability"
    affordance_field = observation_spec.get_field_by_name("obs_affordance_at_position")
    assert affordance_field.dims == 15
    assert affordance_field.scope == "agent"


SNAPSHOT_CASES: dict[str, dict[str, object]] = {
    "L0_0_minimal": {
        "expected_dim": 41,
        "fields": [
            ("obs_grid_encoding", "agent", 9),
            ("obs_position", "agent", 2),
            ("obs_velocity_x", "agent", 1),
            ("obs_velocity_y", "agent", 1),
            ("obs_velocity_magnitude", "agent", 1),
            ("obs_energy", "agent", 1),
            ("obs_hygiene", "agent", 1),
            ("obs_satiation", "agent", 1),
            ("obs_money", "agent", 1),
            ("obs_mood", "agent", 1),
            ("obs_social", "agent", 1),
            ("obs_health", "agent", 1),
            ("obs_fitness", "agent", 1),
            ("obs_affordance_at_position", "agent", 15),
            ("obs_time_sin", "global", 1),
            ("obs_time_cos", "global", 1),
            ("obs_interaction_progress", "agent", 1),
            ("obs_lifetime_progress", "agent", 1),
        ],
    },
    "L0_5_dual_resource": {
        "expected_dim": 57,  # Updated from 81: config changed from 7×7 to 5×5
        "fields": [
            ("obs_grid_encoding", "agent", 25),  # Updated from 49: 5×5 = 25
            ("obs_position", "agent", 2),
            ("obs_velocity_x", "agent", 1),
            ("obs_velocity_y", "agent", 1),
            ("obs_velocity_magnitude", "agent", 1),
            ("obs_energy", "agent", 1),
            ("obs_hygiene", "agent", 1),
            ("obs_satiation", "agent", 1),
            ("obs_money", "agent", 1),
            ("obs_mood", "agent", 1),
            ("obs_social", "agent", 1),
            ("obs_health", "agent", 1),
            ("obs_fitness", "agent", 1),
            ("obs_affordance_at_position", "agent", 15),
            ("obs_time_sin", "global", 1),
            ("obs_time_cos", "global", 1),
            ("obs_interaction_progress", "agent", 1),
            ("obs_lifetime_progress", "agent", 1),
        ],
    },
    "L1_full_observability": {
        "expected_dim": 96,
        "fields": [
            ("obs_grid_encoding", "agent", 64),
            ("obs_position", "agent", 2),
            ("obs_velocity_x", "agent", 1),
            ("obs_velocity_y", "agent", 1),
            ("obs_velocity_magnitude", "agent", 1),
            ("obs_energy", "agent", 1),
            ("obs_hygiene", "agent", 1),
            ("obs_satiation", "agent", 1),
            ("obs_money", "agent", 1),
            ("obs_mood", "agent", 1),
            ("obs_social", "agent", 1),
            ("obs_health", "agent", 1),
            ("obs_fitness", "agent", 1),
            ("obs_affordance_at_position", "agent", 15),
            ("obs_time_sin", "global", 1),
            ("obs_time_cos", "global", 1),
            ("obs_interaction_progress", "agent", 1),
            ("obs_lifetime_progress", "agent", 1),
        ],
    },
    "L2_partial_observability": {
        "expected_dim": 42,
        "fields": [
            ("obs_local_window", "agent", 25),  # 5×5 POMDP window (vision_range=2)
            ("obs_position", "agent", 2),
            ("obs_velocity_x", "agent", 1),  # NEW: velocity observations
            ("obs_velocity_y", "agent", 1),
            ("obs_velocity_magnitude", "agent", 1),
            ("obs_energy", "agent", 1),
            ("obs_hygiene", "agent", 1),
            ("obs_satiation", "agent", 1),
            ("obs_money", "agent", 1),
            ("obs_mood", "agent", 1),
            ("obs_social", "agent", 1),
            ("obs_health", "agent", 1),
            ("obs_fitness", "agent", 1),
            # NOTE: affordance_at_position excluded in POMDP mode (redundant with local_window)
            ("obs_time_sin", "global", 1),
            ("obs_time_cos", "global", 1),
            ("obs_interaction_progress", "agent", 1),
            ("obs_lifetime_progress", "agent", 1),
        ],
    },
    "L3_temporal_mechanics": {
        "expected_dim": 42,
        "fields": [
            ("obs_local_window", "agent", 25),  # 5×5 POMDP window (vision_range=2)
            ("obs_position", "agent", 2),
            ("obs_velocity_x", "agent", 1),  # NEW: velocity observations
            ("obs_velocity_y", "agent", 1),
            ("obs_velocity_magnitude", "agent", 1),
            ("obs_energy", "agent", 1),
            ("obs_hygiene", "agent", 1),
            ("obs_satiation", "agent", 1),
            ("obs_money", "agent", 1),
            ("obs_mood", "agent", 1),
            ("obs_social", "agent", 1),
            ("obs_health", "agent", 1),
            ("obs_fitness", "agent", 1),
            # NOTE: affordance_at_position excluded in POMDP mode (redundant with local_window)
            ("obs_time_sin", "global", 1),
            ("obs_time_cos", "global", 1),
            ("obs_interaction_progress", "agent", 1),
            ("obs_lifetime_progress", "agent", 1),
        ],
    },
}


@pytest.mark.parametrize("pack_name", sorted(SNAPSHOT_CASES))
def test_observation_spec_field_snapshots(pack_name: str) -> None:
    """Regress ObservationSpec ordering, scopes, and dims for every reference pack."""

    case = SNAPSHOT_CASES[pack_name]
    config_dir = Path("configs") / pack_name
    raw_configs = RawConfigs.from_config_dir(config_dir)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)

    _, observation_spec, _ = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)

    expected_fields = case["fields"]
    actual_triplets = [(field.name, field.scope, field.dims) for field in observation_spec.fields]

    assert observation_spec.total_dims == case["expected_dim"]
    assert actual_triplets == expected_fields, f"{pack_name}: ObservationSpec drift detected"


def test_stage5_config_hash_changes_when_config_changes(tmp_path: Path, base_config_dir: Path) -> None:
    temp_pack = tmp_path / "pack"
    shutil.copytree(base_config_dir, temp_pack)

    original = RawConfigs.from_config_dir(temp_pack)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(original)
    metadata_before, _, _ = compiler._stage_5_compute_metadata(temp_pack, original, symbol_table)

    training_path = temp_pack / "training.yaml"
    text = training_path.read_text()
    text = text.replace("max_episodes: 500", "max_episodes: 501")
    training_path.write_text(text)

    mutated = RawConfigs.from_config_dir(temp_pack)
    symbol_table_mut = compiler._stage_2_build_symbol_tables(mutated)
    metadata_after, _, _ = compiler._stage_5_compute_metadata(temp_pack, mutated, symbol_table_mut)

    assert metadata_before.config_hash != metadata_after.config_hash


def test_stage5_builds_rich_metadata(base_raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()

    action_meta, meter_meta, affordance_meta = compiler._stage_5_build_rich_metadata(base_raw_configs)

    assert action_meta.total_actions == len(base_raw_configs.global_actions.actions)
    assert meter_meta.get_meter_by_name("energy").index == 0
    bed_info = affordance_meta.get_affordance_by_name("Bed")
    assert bed_info.enabled


def test_stage5_metadata_reflects_enabled_actions(tmp_path: Path, base_config_dir: Path) -> None:
    pack = tmp_path / "masked"
    shutil.copytree(base_config_dir, pack)

    training_path = pack / "training.yaml"
    data = yaml.safe_load(training_path.read_text())
    data.setdefault("training", {})["enabled_actions"] = ["INTERACT"]
    training_path.write_text(yaml.safe_dump(data, sort_keys=False))

    raw_configs = RawConfigs.from_config_dir(pack)
    compiler = UniverseCompiler()
    action_meta, _, _ = compiler._stage_5_build_rich_metadata(raw_configs)

    enabled_actions = [action.name for action in action_meta.get_enabled_actions()]
    assert enabled_actions == ["INTERACT"]


def test_compute_max_income_includes_completion_bonus_and_pipeline() -> None:
    compiler = UniverseCompiler()

    legacy_affordance = SimpleNamespace(
        effects=[{"meter": "money", "amount": 2.0}],
        effects_per_tick=[{"meter": "money", "amount": 0.5}],
        completion_bonus=[{"meter": "money", "amount": 3.0}],
        effect_pipeline=None,
    )

    pipeline_affordance = SimpleNamespace(
        effects=[],
        effects_per_tick=[],
        completion_bonus=[],
        effect_pipeline=EffectPipeline(
            on_start=[AffordanceEffect(meter="money", amount=1.0)],
            per_tick=[AffordanceEffect(meter="money", amount=0.25)],
            on_completion=[AffordanceEffect(meter="money", amount=4.0)],
            on_early_exit=[AffordanceEffect(meter="money", amount=-2.0)],
            on_failure=[AffordanceEffect(meter="health", amount=-1.0)],
        ),
    )

    total = compiler._compute_max_income((legacy_affordance, pipeline_affordance))

    assert total == pytest.approx(2.0 + 0.5 + 3.0 + 1.0 + 0.25 + 4.0)


def test_extract_money_cost_includes_per_tick_costs() -> None:
    compiler = UniverseCompiler()

    affordance = SimpleNamespace(
        costs=[{"meter": "money", "amount": 5.0}],
        costs_per_tick=[{"meter": "money", "amount": 1.5}],
    )

    assert compiler._extract_money_cost(affordance) == pytest.approx(6.5)


def test_provenance_changes_when_git_sha_changes() -> None:
    compiler = UniverseCompiler()
    kwargs = {
        "config_hash": "abc",
        "compiler_version": "0.1.0",
        "python_version": "3.11.9",
        "torch_version": "2.2.0",
        "pydantic_version": "2.6.0",
    }

    base = compiler._compute_provenance_id(git_sha="aaaabbbb", **kwargs)
    changed = compiler._compute_provenance_id(git_sha="ccccdddd", **kwargs)

    assert base != changed


def test_stage5_metadata_records_expected_provenance(base_config_dir: Path, base_raw_configs: RawConfigs) -> None:
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(base_raw_configs)

    metadata, _, _ = compiler._stage_5_compute_metadata(base_config_dir, base_raw_configs, symbol_table)

    regenerated = compiler._compute_provenance_id(
        config_hash=metadata.config_hash,
        compiler_version=metadata.compiler_version,
        git_sha=metadata.compiler_git_sha,
        python_version=metadata.python_version,
        torch_version=metadata.torch_version,
        pydantic_version=metadata.pydantic_version,
    )

    assert metadata.provenance_id == regenerated

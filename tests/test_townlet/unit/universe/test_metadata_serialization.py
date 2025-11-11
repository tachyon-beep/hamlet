"""Round-trip serialization tests for compiler metadata DTOs."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest
import torch

from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.dto import (
    ActionMetadata,
    ActionSpaceMetadata,
    AffordanceInfo,
    AffordanceMetadata,
    MeterInfo,
    MeterMetadata,
    ObservationField,
    ObservationSpec,
    UniverseMetadata,
)


def _to_plain(obj):
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    return obj


def _load_stage5_artifacts(config_name: str):
    config_dir = Path("configs") / config_name
    raw_configs = RawConfigs.from_config_dir(config_dir)
    compiler = UniverseCompiler()
    symbol_table = compiler._stage_2_build_symbol_tables(raw_configs)
    metadata, observation_spec, _ = compiler._stage_5_compute_metadata(config_dir, raw_configs, symbol_table)
    action_meta, meter_meta, affordance_meta = compiler._stage_5_build_rich_metadata(raw_configs)
    return metadata, observation_spec, action_meta, meter_meta, affordance_meta


def test_universe_metadata_round_trip() -> None:
    metadata, observation_spec, action_meta, meter_meta, affordance_meta = _load_stage5_artifacts("L0_0_minimal")

    def _round_trip(dataclass_obj, factory):
        payload = json.loads(json.dumps(_to_plain(dataclass_obj)))
        return factory(payload)

    def _meta_factory(payload: dict) -> UniverseMetadata:
        payload["meter_names"] = tuple(payload["meter_names"])
        payload["meter_name_to_index"] = payload["meter_name_to_index"]
        payload["affordance_ids"] = tuple(payload["affordance_ids"])
        payload["affordance_id_to_index"] = payload["affordance_id_to_index"]
        return UniverseMetadata(**payload)

    def _obs_spec_factory(payload: dict) -> ObservationSpec:
        obs_fields = tuple(ObservationField(**field) for field in payload["fields"])
        return ObservationSpec(total_dims=payload["total_dims"], fields=obs_fields, encoding_version=payload["encoding_version"])

    def _action_meta_factory(payload: dict) -> ActionSpaceMetadata:
        actions = tuple(ActionMetadata(**action) for action in payload["actions"])
        return ActionSpaceMetadata(total_actions=payload["total_actions"], actions=actions)

    def _meter_meta_factory(payload: dict) -> MeterMetadata:
        meters = tuple(MeterInfo(**meter) for meter in payload["meters"])
        return MeterMetadata(meters=meters)

    def _affordance_meta_factory(payload: dict) -> AffordanceMetadata:
        affordances = tuple(AffordanceInfo(**aff) for aff in payload["affordances"])
        return AffordanceMetadata(affordances=affordances)

    reconstructed_metadata = _round_trip(metadata, _meta_factory)
    reconstructed_obs = _round_trip(observation_spec, _obs_spec_factory)
    reconstructed_action_meta = _round_trip(action_meta, _action_meta_factory)
    reconstructed_meter_meta = _round_trip(meter_meta, _meter_meta_factory)
    reconstructed_affordance_meta = _round_trip(affordance_meta, _affordance_meta_factory)

    assert reconstructed_metadata == metadata
    assert reconstructed_obs == observation_spec
    assert reconstructed_action_meta == action_meta
    assert reconstructed_meter_meta == meter_meta
    assert reconstructed_affordance_meta == affordance_meta


def test_compiled_universe_msgpack_round_trip(tmp_path: Path) -> None:
    compiler = UniverseCompiler()
    compiled = compiler.compile(Path("configs/L0_0_minimal"))

    artifact_path = tmp_path / "compiled.msgpack"
    compiled.save_to_cache(artifact_path)
    reconstructed = CompiledUniverse.load_from_cache(artifact_path)

    assert reconstructed.metadata == compiled.metadata
    assert reconstructed.observation_spec == compiled.observation_spec
    assert reconstructed.action_space_metadata == compiled.action_space_metadata
    assert reconstructed.meter_metadata == compiled.meter_metadata
    assert reconstructed.affordance_metadata == compiled.affordance_metadata
    assert torch.allclose(
        reconstructed.optimization_data.base_depletions,
        compiled.optimization_data.base_depletions,
    )
    assert torch.equal(
        reconstructed.optimization_data.action_mask_table,
        compiled.optimization_data.action_mask_table,
    )
    with pytest.raises(FrozenInstanceError):
        reconstructed.metadata = None  # type: ignore[attr-defined]

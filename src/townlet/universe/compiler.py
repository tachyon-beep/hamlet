"""UniverseCompiler implementation (Stage 1 scaffolding)."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from townlet.config.affordance import AffordanceConfig
from townlet.config.cascade import CascadeConfig
from townlet.config.effect_pipeline import EffectPipeline
from townlet.environment.cascade_config import load_cascades_config as load_full_cascades_config
from townlet.environment.substrate_action_validator import SubstrateActionValidator
from townlet.substrate.config import SubstrateConfig
from townlet.universe.adapters.vfs_adapter import vfs_to_observation_spec
from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler_inputs import RawConfigs
from townlet.universe.dto import (
    ActionMetadata,
    ActionSpaceMetadata,
    AffordanceInfo,
    AffordanceMetadata,
    MeterInfo,
    MeterMetadata,
    ObservationSpec,
    UniverseMetadata,
)
from townlet.universe.optimization import OptimizationData
from townlet.vfs.observation_builder import VFSObservationSpecBuilder
from townlet.vfs.registry import VariableRegistry

from .errors import CompilationError, CompilationErrorCollector
from .symbol_table import UniverseSymbolTable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
COMPILER_VERSION = "0.1.0"


class UniverseCompiler:
    """Entry point for compiling config packs into CompiledUniverse artifacts."""

    def __init__(self) -> None:
        self._symbol_table = UniverseSymbolTable()
        self._metadata: UniverseMetadata | None = None
        self._observation_spec: ObservationSpec | None = None
        self._action_metadata: ActionSpaceMetadata | None = None
        self._meter_metadata: MeterMetadata | None = None
        self._affordance_metadata: AffordanceMetadata | None = None
        self._optimization_data: OptimizationData | None = None

    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile a config pack into a CompiledUniverse (with optional caching)."""

        config_dir = Path(config_dir)
        cache_path = self._cache_artifact_path(config_dir)
        precomputed_hash: str | None = None
        precomputed_provenance: str | None = None

        if use_cache and cache_path.exists():
            precomputed_hash, precomputed_provenance = self._build_cache_fingerprint(config_dir)
            try:
                cached_universe = CompiledUniverse.load_from_cache(cache_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load cached universe from %s: %s", cache_path, exc)
            else:
                if (
                    cached_universe.metadata.config_hash == precomputed_hash
                    and cached_universe.metadata.provenance_id == precomputed_provenance
                ):
                    logger.info("Loaded compiled universe from cache: %s", cache_path)
                    return cached_universe
                logger.info(
                    "Cache stale for %s (cached=%s/%s, current=%s/%s). Recompiling.",
                    cache_path,
                    cached_universe.metadata.config_hash[:8],
                    precomputed_hash[:8] if precomputed_hash else "unknown",
                    (cached_universe.metadata.provenance_id or "")[:8],
                    (precomputed_provenance or "unknown")[:8],
                )

        raw_configs = self._stage_1_parse_individual_files(config_dir)

        symbol_table = self._stage_2_build_symbol_tables(raw_configs)
        self._symbol_table = symbol_table

        errors = CompilationErrorCollector(stage="Stage 3: Resolve References")
        self._stage_3_resolve_references(raw_configs, symbol_table, errors)
        errors.check_and_raise("Stage 3: Resolve References")

        stage4_errors = CompilationErrorCollector(stage="Stage 4: Cross-Validation")
        self._stage_4_cross_validate(raw_configs, symbol_table, stage4_errors)
        # Emit warnings even if Stage 4 ultimately raises so operators see every diagnostic.
        for warning in stage4_errors.warnings:
            logger.warning(warning)
        stage4_errors.check_and_raise("Stage 4: Cross-Validation")

        metadata, observation_spec = self._stage_5_compute_metadata(
            config_dir,
            raw_configs,
            symbol_table,
            precomputed_config_hash=precomputed_hash,
        )
        (
            action_space_metadata,
            meter_metadata,
            affordance_metadata,
        ) = self._stage_5_build_rich_metadata(raw_configs)

        optimization_data = self._stage_6_optimize(raw_configs, metadata)

        compiled = self._stage_7_emit_compiled_universe(
            raw_configs=raw_configs,
            metadata=metadata,
            observation_spec=observation_spec,
            action_space_metadata=action_space_metadata,
            meter_metadata=meter_metadata,
            affordance_metadata=affordance_metadata,
            optimization_data=optimization_data,
        )

        if use_cache:
            cache_dir = self._cache_directory_for(config_dir)
            try:
                self._prepare_cache_directory(cache_dir)
                compiled.save_to_cache(cache_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to write compiled universe cache at %s: %s", cache_path, exc)

        self._metadata = compiled.metadata
        self._observation_spec = compiled.observation_spec
        self._action_metadata = compiled.action_space_metadata
        self._meter_metadata = compiled.meter_metadata
        self._affordance_metadata = compiled.affordance_metadata
        self._optimization_data = compiled.optimization_data

        return compiled

    def _stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
        """Stage 1 – load all YAML files into DTOs using shared loaders."""

        return RawConfigs.from_config_dir(config_dir)

    def _stage_2_build_symbol_tables(self, raw_configs: RawConfigs) -> UniverseSymbolTable:
        """Stage 2 – register meters, variables, actions, cascades, and affordances."""

        table = UniverseSymbolTable()

        for bar in raw_configs.bars:
            table.register_meter(bar)

        for variable in raw_configs.variables_reference:
            table.register_variable(variable)

        for action in raw_configs.global_actions.actions:
            table.register_action(action)

        for cascade in raw_configs.cascades:
            table.register_cascade(cascade)

        for affordance in raw_configs.affordances:
            table.register_affordance(affordance)

        for cue in raw_configs.cues:
            table.register_cue(cue)

        return table

    def _stage_3_resolve_references(
        self,
        raw_configs: RawConfigs,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
    ) -> None:
        """Stage 3 – ensure every cross-file reference points to a known symbol."""

        source_map = getattr(raw_configs, "source_map", None)
        hints_added: set[str] = set()

        def _add_hint(key: str, text: str) -> None:
            if key not in hints_added:
                errors.add_hint(text)
                hints_added.add(key)

        def _format_error(code: str, message: str, location: str | None = None) -> str:
            location_str = None
            if location and source_map is not None:
                location_str = source_map.lookup(location)
            if not location_str:
                location_str = location
            if location_str:
                return f"[{code}] {location_str} - {message}"
            return f"[{code}] {message}"

        def _record_meter_reference(
            meter_name: str | None,
            location: str,
            *,
            code: str,
            hint_key: str | None = None,
            hint_text: str | None = None,
        ) -> None:
            if not meter_name:
                return
            try:
                symbol_table.resolve_meter_reference(meter_name, location=location)
            except ReferenceError as exc:
                if hint_key and hint_text:
                    _add_hint(hint_key, hint_text)
                errors.add(_format_error(code, str(exc), location))

        def _handle_missing_meter(location: str) -> None:
            _add_hint(
                "missing_meter",
                "Each cost/effect entry must include a 'meter' field (case-sensitive).",
            )
            errors.add(
                _format_error(
                    "UAC-RES-003",
                    "Entry missing required 'meter' field.",
                    location,
                )
            )

        def _get_attr(obj: object | None, key: str) -> object | None:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _get_meter(obj: object | None) -> str | None:
            value = _get_attr(obj, "meter")
            if isinstance(value, str) and value:
                return value
            return None

        # Cascades: validate source/target meters exist.
        cascade_hint = (
            "invalid_meter",
            "Meter references must match names defined in bars.yaml (case-sensitive).",
        )
        for cascade in raw_configs.cascades:
            _record_meter_reference(
                cascade.source,
                f"cascades.yaml:{cascade.name}:source",
                code="UAC-RES-001",
                hint_key=cascade_hint[0],
                hint_text=cascade_hint[1],
            )
            _record_meter_reference(
                cascade.target,
                f"cascades.yaml:{cascade.name}:target",
                code="UAC-RES-001",
                hint_key=cascade_hint[0],
                hint_text=cascade_hint[1],
            )

        # Affordances: validate every meter reference across costs/effects/etc.
        meter_fields = (
            ("costs", "costs"),
            ("costs_per_tick", "costs_per_tick"),
            ("effects", "effects"),
            ("effects_per_tick", "effects_per_tick"),
            ("completion_bonus", "completion_bonus"),
        )

        for affordance in raw_configs.affordances:
            for attr_name, label in meter_fields:
                entries = getattr(affordance, attr_name, None)
                if not entries:
                    continue
                base_location = f"affordances.yaml:{affordance.id}:{label}"
                for idx, entry in enumerate(entries):
                    meter = _get_meter(entry)
                    if meter:
                        _record_meter_reference(
                            meter,
                            base_location,
                            code="UAC-RES-002",
                            hint_key="invalid_meter",
                            hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                        )
                    else:
                        _handle_missing_meter(f"{base_location}[{idx}]")

            # Capabilities (meter-gated)
            capabilities = getattr(affordance, "capabilities", None)
            if capabilities:
                for idx, capability in enumerate(capabilities):
                    if _get_attr(capability, "type") == "meter_gated":
                        meter = _get_meter(capability)
                        location = f"affordances.yaml:{affordance.id}:capabilities[{idx}]"
                        if meter:
                            _record_meter_reference(
                                meter,
                                location,
                                code="UAC-RES-002",
                                hint_key="invalid_meter",
                                hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                            )
                        else:
                            _handle_missing_meter(location)

            # Effect pipeline stages (if defined)
            effect_pipeline = getattr(affordance, "effect_pipeline", None)
            if effect_pipeline:
                for stage_name in ("on_start", "per_tick", "on_completion", "on_early_exit", "on_failure"):
                    stage_effects = _get_attr(effect_pipeline, stage_name)
                    if stage_effects:
                        base_location = f"affordances.yaml:{affordance.id}:effect_pipeline.{stage_name}"
                        for idx, entry in enumerate(stage_effects):
                            meter = _get_meter(entry)
                            if meter:
                                _record_meter_reference(
                                    meter,
                                    base_location,
                                    code="UAC-RES-002",
                                    hint_key="invalid_meter",
                                    hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                                )
                            else:
                                _handle_missing_meter(f"{base_location}[{idx}]")

            # Availability constraints
            availability = getattr(affordance, "availability", None)
            if availability:
                for idx, constraint in enumerate(availability):
                    meter = _get_meter(constraint)
                    location = f"affordances.yaml:{affordance.id}:availability[{idx}]"
                    if meter:
                        _record_meter_reference(
                            meter,
                            location,
                            code="UAC-RES-002",
                            hint_key="invalid_meter",
                            hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                        )
                    else:
                        _handle_missing_meter(location)

        # Environment enabled_affordances (names)
        enabled_affordances = raw_configs.environment.enabled_affordances
        if enabled_affordances:
            for affordance_name in enabled_affordances:
                if affordance_name not in symbol_table.affordances_by_name:
                    errors.add(
                        _format_error(
                            "UAC-RES-004",
                            f"References non-existent affordance '{affordance_name}'. "
                            f"Valid affordances: {symbol_table.affordance_names}",
                            "training.yaml:environment.enabled_affordances",
                        )
                    )
                    _add_hint(
                        "invalid_affordance_name",
                        "Ensure environment.enabled_affordances lists valid affordance names from affordances.yaml.",
                    )

        # Global action costs/effects (dict[str, float])
        for action in raw_configs.global_actions.actions:
            for meter in action.costs.keys():
                _record_meter_reference(
                    meter,
                    f"global_actions.yaml:{action.name}:costs",
                    code="UAC-RES-005",
                    hint_key="invalid_meter",
                    hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                )
            for meter in action.effects.keys():
                _record_meter_reference(
                    meter,
                    f"global_actions.yaml:{action.name}:effects",
                    code="UAC-RES-005",
                    hint_key="invalid_meter",
                    hint_text="Meter references must match names defined in bars.yaml (case-sensitive).",
                )

    def _stage_4_cross_validate(
        self,
        raw_configs: RawConfigs,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
    ) -> None:
        """Stage 4 – enforce cross-config semantic constraints (subset of spec for TASK-004A)."""

        source_map = getattr(raw_configs, "source_map", None)
        hints_added: set[str] = set()

        def _add_hint(key: str, text: str) -> None:
            if key not in hints_added:
                errors.add_hint(text)
                hints_added.add(key)

        def _format_error(code: str, message: str, location: str | None = None) -> str:
            location_str = None
            if location and source_map is not None:
                location_str = source_map.lookup(location)
            if not location_str:
                location_str = location
            prefix = f"[{code}] "
            if location_str:
                return f"{prefix}{location_str} - {message}"
            return f"{prefix}{message}"

        self._validate_spatial_feasibility(raw_configs, errors, _format_error)
        self._validate_economic_balance(raw_configs, errors, _format_error)
        self._validate_cascade_cycles(raw_configs, errors, _format_error)
        self._validate_operating_hours(raw_configs, errors, _format_error)
        cues_config = raw_configs.hamlet_config.cues
        self._validate_availability_and_modes(raw_configs, symbol_table, errors, _format_error)
        self._validate_basic_cues(cues_config, symbol_table, errors, _format_error)
        self._validate_visual_cues(cues_config, symbol_table, errors, _format_error)
        self._validate_capabilities_and_effect_pipelines(raw_configs, errors, _format_error)
        self._validate_affordance_positions(raw_configs, errors, _format_error)
        self._validate_substrate_action_compatibility(raw_configs, errors, _format_error, _add_hint)

    def _validate_spatial_feasibility(self, raw_configs: RawConfigs, errors: CompilationErrorCollector, formatter) -> None:
        grid_size = getattr(raw_configs.environment, "grid_size", None)
        if grid_size is None or grid_size <= 0:
            return

        grid_cells = grid_size * grid_size
        enabled_affordances = raw_configs.environment.enabled_affordances
        if enabled_affordances is None:
            required = len(raw_configs.affordances)
        else:
            required = len(enabled_affordances)

        required_cells = required + 1  # +1 for the agent
        if required_cells > grid_cells:
            message = (
                f"Spatial impossibility: Grid has {grid_cells} cells ({grid_size}×{grid_size}) but need {required_cells} "
                f"({required} affordances + 1 agent)."
            )
            errors.add(formatter("UAC-VAL-001", message, "training.yaml:environment.grid_size"))

    def _validate_economic_balance(self, raw_configs: RawConfigs, errors: CompilationErrorCollector, formatter) -> None:
        total_income = self._compute_max_income(raw_configs.affordances)
        total_costs = self._compute_total_costs(raw_configs.affordances)

        if total_income < total_costs:
            errors.add_warning(
                formatter(
                    "UAC-VAL-002",
                    f"Economic imbalance: Total income ({total_income:.2f}) < total costs ({total_costs:.2f}).",
                    "affordances.yaml",
                )
            )

    def _validate_cascade_cycles(self, raw_configs: RawConfigs, errors: CompilationErrorCollector, formatter) -> None:
        graph = self._build_cascade_graph(raw_configs.cascades)
        cycles = self._detect_cycles(graph)
        if not cycles:
            return
        for cycle in cycles:
            cycle_str = " → ".join(cycle + [cycle[0]])
            errors.add(formatter("UAC-VAL-003", f"Cascade circularity detected: {cycle_str}.", "cascades.yaml"))

    def _validate_operating_hours(self, raw_configs: RawConfigs, errors: CompilationErrorCollector, formatter) -> None:
        for affordance in raw_configs.affordances:
            operating_hours = getattr(affordance, "operating_hours", None)
            if not operating_hours:
                continue
            if len(operating_hours) != 2:
                errors.add(
                    formatter(
                        "UAC-VAL-004",
                        "operating_hours must contain exactly two entries [open_hour, close_hour]",
                        f"affordances.yaml:{affordance.id}:operating_hours",
                    )
                )
                continue
            open_hour, close_hour = operating_hours
            if open_hour < 0 or open_hour > 23:
                errors.add(
                    formatter(
                        "UAC-VAL-004",
                        f"open_hour must be 0-23, got {open_hour}",
                        f"affordances.yaml:{affordance.id}:operating_hours",
                    )
                )
            if close_hour < 1 or close_hour > 28:
                errors.add(
                    formatter(
                        "UAC-VAL-004",
                        f"close_hour must be 1-28, got {close_hour}",
                        f"affordances.yaml:{affordance.id}:operating_hours",
                    )
                )

    def _validate_availability_and_modes(
        self,
        raw_configs: RawConfigs,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        for affordance in raw_configs.affordances:
            for idx, constraint in enumerate(getattr(affordance, "availability", []) or []):
                location = f"affordances.yaml:{affordance.id}:availability[{idx}]"
                meter = self._get_attr_value(constraint, "meter")
                if meter not in symbol_table.meters:
                    errors.add(
                        formatter(
                            "UAC-VAL-007",
                            f"Availability constraint references unknown meter '{meter}'",
                            location,
                        )
                    )
                for bound_name in ("min", "max"):
                    bound_value = self._get_attr_value(constraint, bound_name)
                    if bound_value is None:
                        continue
                    if bound_value < 0.0 or bound_value > 1.0:
                        errors.add(
                            formatter(
                                "UAC-VAL-007",
                                f"Availability {bound_name} must be within [0.0, 1.0], got {bound_value}",
                                location,
                            )
                        )
                min_value = self._get_attr_value(constraint, "min")
                max_value = self._get_attr_value(constraint, "max")
                if min_value is not None and max_value is not None and min_value >= max_value:
                    errors.add(
                        formatter(
                            "UAC-VAL-007",
                            f"Availability min ({min_value}) must be < max ({max_value}).",
                            location,
                        )
                    )

            modes = getattr(affordance, "modes", {}) or {}
            for mode_name, mode in modes.items():
                hours = self._get_attr_value(mode, "hours")
                if not hours:
                    continue
                start, end = hours
                if not (0 <= start <= 23 and 0 <= end <= 23):
                    errors.add(
                        formatter(
                            "UAC-VAL-007",
                            f"Mode '{mode_name}' hours must be within 0-23, got {hours}",
                            f"affordances.yaml:{affordance.id}:modes:{mode_name}",
                        )
                    )

    def _validate_basic_cues(
        self,
        cues_config,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        for cue in cues_config.simple_cues:
            meter = cue.condition.meter
            if meter not in symbol_table.meters:
                errors.add(
                    formatter(
                        "UAC-VAL-005",
                        f"Cue '{cue.cue_id}' references unknown meter '{meter}'",
                        f"cues.yaml:{cue.cue_id}",
                    )
                )
            threshold = cue.condition.threshold
            if threshold < 0.0 or threshold > 1.0:
                errors.add(
                    formatter(
                        "UAC-VAL-005",
                        f"Cue threshold must be within [0.0, 1.0], got {threshold}",
                        f"cues.yaml:{cue.cue_id}",
                    )
                )

        for cue in cues_config.compound_cues:
            for condition in cue.conditions:
                if condition.meter not in symbol_table.meters:
                    errors.add(
                        formatter(
                            "UAC-VAL-005",
                            f"Cue '{cue.cue_id}' references unknown meter '{condition.meter}'",
                            f"cues.yaml:{cue.cue_id}",
                        )
                    )
                if condition.threshold < 0.0 or condition.threshold > 1.0:
                    errors.add(
                        formatter(
                            "UAC-VAL-005",
                            f"Cue threshold must be within [0.0, 1.0], got {condition.threshold}",
                            f"cues.yaml:{cue.cue_id}",
                        )
                    )

    def _validate_visual_cues(
        self,
        cues_config,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        if not cues_config.visual_cues:
            return

        for meter_name, cues in cues_config.visual_cues.items():
            if meter_name not in symbol_table.meters:
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue block references unknown meter '{meter_name}'",
                        f"cues.yaml:{meter_name}",
                    )
                )
                continue

            ranges = [tuple(cue.range) for cue in cues]
            if not self._ranges_cover_domain(ranges, 0.0, 1.0):
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue ranges for '{meter_name}' do not cover [0.0, 1.0] without gaps.",
                        f"cues.yaml:{meter_name}",
                    )
                )
            if self._ranges_overlap(ranges):
                errors.add(
                    formatter(
                        "UAC-VAL-009",
                        f"Visual cue ranges for '{meter_name}' overlap.",
                        f"cues.yaml:{meter_name}",
                    )
                )

    def _validate_capabilities_and_effect_pipelines(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        for affordance in raw_configs.affordances:
            capabilities = getattr(affordance, "capabilities", []) or []
            types = [self._get_attr_value(cap, "type") for cap in capabilities]
            multi_tick_caps = [cap for cap, cap_type in zip(capabilities, types) if cap_type == "multi_tick"]
            has_resumable_flag = any(bool(self._get_attr_value(cap, "resumable")) for cap in capabilities)

            if affordance.interaction_type and affordance.interaction_type.lower() == "instant" and multi_tick_caps:
                errors.add(
                    formatter(
                        "UAC-VAL-008",
                        "Instant affordances cannot declare multi_tick capabilities.",
                        f"affordances.yaml:{affordance.id}",
                    )
                )

            pipeline = affordance.effect_pipeline
            if pipeline is not None and not isinstance(pipeline, EffectPipeline):
                pipeline = EffectPipeline.model_validate(pipeline)
            if multi_tick_caps:
                if pipeline is None or (not pipeline.per_tick and not pipeline.on_completion):
                    errors.add(
                        formatter(
                            "UAC-VAL-008",
                            "multi_tick capability requires per_tick or on_completion effects.",
                            f"affordances.yaml:{affordance.id}",
                        )
                    )
                else:
                    cap = multi_tick_caps[0]
                    early_exit_allowed = bool(self._get_attr_value(cap, "early_exit_allowed"))
                    if pipeline.on_early_exit and not early_exit_allowed:
                        errors.add_warning(
                            formatter(
                                "UAC-VAL-008",
                                "on_early_exit effects defined but early_exit_allowed is False.",
                                f"affordances.yaml:{affordance.id}",
                            )
                        )
            elif pipeline and pipeline.per_tick:
                errors.add_warning(
                    formatter(
                        "UAC-VAL-008",
                        "Per-tick effects defined without multi_tick capability.",
                        f"affordances.yaml:{affordance.id}",
                    )
                )

            if "cooldown" in types and affordance.interaction_type and affordance.interaction_type.lower() == "instant":
                # Instant affordances with cooldowns are permitted, but highlight to operators.
                errors.add_warning(
                    formatter(
                        "UAC-VAL-008",
                        "Instant affordance declares a cooldown capability; ensure this is intentional.",
                        f"affordances.yaml:{affordance.id}",
                    )
                )

            if has_resumable_flag and not multi_tick_caps:
                errors.add(
                    formatter(
                        "UAC-VAL-008",
                        "'resumable' flag requires a multi_tick capability.",
                        f"affordances.yaml:{affordance.id}:capabilities",
                    )
                )

    def _validate_affordance_positions(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        for affordance in raw_configs.affordances:
            position = getattr(affordance, "position", None)
            if position is None:
                continue
            in_bounds, message = self._position_in_bounds(position, raw_configs.substrate)
            if not in_bounds:
                errors.add(
                    formatter(
                        "UAC-VAL-010",
                        message,
                        f"affordances.yaml:{affordance.id}:position",
                    )
                )

    @staticmethod
    def _ranges_cover_domain(ranges: list[tuple[float, float]], domain_min: float, domain_max: float) -> bool:
        if not ranges:
            return False
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        eps = 1e-6
        if abs(sorted_ranges[0][0] - domain_min) > eps:
            return False
        current_end = sorted_ranges[0][1]
        for start, end in sorted_ranges[1:]:
            if abs(start - current_end) > eps:
                return False
            current_end = end
        if abs(current_end - domain_max) > eps:
            return False
        return True

    @staticmethod
    def _ranges_overlap(ranges: list[tuple[float, float]]) -> bool:
        if not ranges:
            return False
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        eps = 1e-6
        current_end = sorted_ranges[0][1]
        for start, end in sorted_ranges[1:]:
            if start < current_end - eps:
                return True
            current_end = max(current_end, end)
        return False

    def _position_in_bounds(self, position: object, substrate: SubstrateConfig) -> tuple[bool, str]:
        if substrate.type == "grid" and substrate.grid is not None:
            grid = substrate.grid
            width, height = grid.width, grid.height
            depth = grid.depth or 1
            if isinstance(position, list):
                if len(position) == 2:
                    x, y = position
                    if 0 <= x < width and 0 <= y < height:
                        return True, ""
                    return False, f"Position {position} outside grid bounds 0-{width-1}, 0-{height-1}."
                if len(position) == 3:
                    if grid.depth is None:
                        return False, "Position includes depth but substrate is 2D."
                    x, y, z = position
                    if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                        return True, ""
                    return False, f"Position {position} outside 3D grid bounds."
                return False, f"Grid positions must be length 2 or 3. Got {len(position)} elements."
            if isinstance(position, int):
                total_nodes = width * height * depth
                if 0 <= position < total_nodes:
                    return True, ""
                return False, f"Graph node id {position} outside 0-{total_nodes - 1}."
            if isinstance(position, dict):
                # Hex/axial grids do not currently expose explicit bounds; assume valid.
                return True, ""
            return False, f"Unsupported position format '{type(position).__name__}'."
        return True, ""

    @staticmethod
    def _get_attr_value(obj: object, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _validate_substrate_action_compatibility(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
        add_hint,
    ) -> None:
        validator = SubstrateActionValidator(raw_configs.substrate, raw_configs.global_actions)
        result = validator.validate()
        for message in result.errors:
            errors.add(formatter("UAC-VAL-006", message, "configs/global_actions.yaml"))
        for warning in result.warnings:
            errors.add_warning(formatter("UAC-VAL-006", warning, "configs/global_actions.yaml"))

    def _compute_total_costs(self, affordances: tuple[AffordanceConfig, ...]) -> float:
        total = 0.0
        for affordance in affordances:
            total += self._sum_amounts(getattr(affordance, "costs", []))
            total += self._sum_amounts(getattr(affordance, "costs_per_tick", []))
        return total

    def _compute_max_income(self, affordances: tuple[AffordanceConfig, ...]) -> float:
        total = 0.0
        for affordance in affordances:
            pipeline = getattr(affordance, "effect_pipeline", None)
            if pipeline is not None:
                total += self._sum_money_entries(pipeline.on_start, positive_only=True)
                total += self._sum_money_entries(pipeline.per_tick, positive_only=True)
                total += self._sum_money_entries(pipeline.on_completion, positive_only=True)
                total += self._sum_money_entries(pipeline.on_early_exit, positive_only=True)
                total += self._sum_money_entries(pipeline.on_failure, positive_only=True)
            else:
                total += self._sum_money_entries(getattr(affordance, "effects", []), positive_only=True)
                total += self._sum_money_entries(getattr(affordance, "effects_per_tick", []), positive_only=True)
                total += self._sum_money_entries(getattr(affordance, "completion_bonus", []), positive_only=True)
        return total

    def _sum_money_entries(self, entries: object | None, *, positive_only: bool) -> float:
        total = 0.0
        if not entries:
            return total
        for entry in entries:
            if self._get_meter(entry) != "money":
                continue
            amount = self._get_amount(entry)
            if amount is None:
                continue
            if positive_only and amount <= 0:
                continue
            total += amount
        return total

    def _sum_amounts(self, entries: object | None) -> float:
        total = 0.0
        if not entries:
            return total
        for entry in entries:
            amount = self._get_amount(entry)
            if amount:
                total += amount
        return total

    def _get_meter(self, entry: object | None) -> str | None:
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry.get("meter")
        return getattr(entry, "meter", None)

    def _get_amount(self, entry: object | None) -> float | None:
        if entry is None:
            return None
        value = entry.get("amount") if isinstance(entry, dict) else getattr(entry, "amount", None)
        if isinstance(value, int | float):
            return float(value)
        return None

    def _build_cascade_graph(self, cascades: tuple[CascadeConfig, ...]) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for cascade in cascades:
            graph.setdefault(cascade.source, []).append(cascade.target)
        return graph

    def _detect_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            stack.add(node)
            path.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in stack:
                    try:
                        start_index = path.index(neighbor)
                        cycles.append(path[start_index:])
                    except ValueError:
                        cycles.append([neighbor])
            stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _stage_5_compute_metadata(
        self,
        config_dir: Path,
        raw_configs: RawConfigs,
        symbol_table: UniverseSymbolTable,
        *,
        precomputed_config_hash: str | None = None,
    ) -> tuple[UniverseMetadata, ObservationSpec]:
        """Stage 5 – compute derived metadata and observation specification."""

        import torch
        from pydantic import __version__ as pydantic_version  # lazy import to avoid startup penalty

        exposures = self._load_observation_exposures(config_dir, raw_configs)
        variable_registry = VariableRegistry(
            variables=list(raw_configs.variables_reference),
            num_agents=raw_configs.population.num_agents,
            device=torch.device("cpu"),
        )

        obs_builder = VFSObservationSpecBuilder()
        variables = list(variable_registry.variables.values())
        vfs_fields = obs_builder.build_observation_spec(variables, exposures)
        var_scope_lookup = {var.id: var.scope for var in variables}
        observation_spec = vfs_to_observation_spec(vfs_fields, var_scope_lookup)

        sorted_bars = sorted(raw_configs.bars, key=lambda bar: bar.index)
        meter_names = tuple(bar.name for bar in sorted_bars)
        meter_name_to_index = {bar.name: bar.index for bar in sorted_bars}

        affordances = tuple(raw_configs.affordances)
        affordance_ids = tuple(aff.id for aff in affordances)
        affordance_id_to_index = {aff.id: idx for idx, aff in enumerate(affordances)}

        action_count = len(raw_configs.global_actions.actions)

        max_income = self._compute_max_income(raw_configs.affordances)
        total_costs = self._compute_total_costs(raw_configs.affordances)
        economic_balance = max_income / total_costs if total_costs > 0 else float("inf")

        grid_size, grid_cells = self._derive_grid_dimensions(raw_configs.substrate)

        config_hash = precomputed_config_hash or self._compute_config_hash(config_dir)
        compiler_git_sha = self._get_git_sha()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        provenance_id = self._compute_provenance_id(
            config_hash=config_hash,
            compiler_version=COMPILER_VERSION,
            git_sha=compiler_git_sha,
            python_version=python_version,
            torch_version=torch.__version__,
            pydantic_version=pydantic_version,
        )

        metadata = UniverseMetadata(
            universe_name=config_dir.name,
            schema_version=SCHEMA_VERSION,
            substrate_type=self._label_substrate_type(raw_configs.substrate),
            position_dim=self._infer_position_dim(raw_configs.substrate),
            meter_count=len(sorted_bars),
            meter_names=meter_names,
            meter_name_to_index=meter_name_to_index,
            affordance_count=len(affordance_ids),
            affordance_ids=affordance_ids,
            affordance_id_to_index=affordance_id_to_index,
            action_count=action_count,
            observation_dim=observation_spec.total_dims,
            grid_size=grid_size,
            grid_cells=grid_cells,
            max_sustainable_income=max_income,
            total_affordance_costs=total_costs,
            economic_balance=economic_balance,
            ticks_per_day=24,
            config_version=self._resolve_config_version(raw_configs),
            compiler_version=COMPILER_VERSION,
            compiled_at=datetime.now(UTC).isoformat(),
            config_hash=config_hash,
            provenance_id=provenance_id,
            compiler_git_sha=compiler_git_sha,
            python_version=python_version,
            torch_version=torch.__version__,
            pydantic_version=pydantic_version,
        )

        return metadata, observation_spec

    def _stage_5_build_rich_metadata(
        self,
        raw_configs: RawConfigs,
    ) -> tuple[ActionSpaceMetadata, MeterMetadata, AffordanceMetadata]:
        """Stage 5 – build training-facing metadata structures."""

        actions_meta: list[ActionMetadata] = []
        for action in raw_configs.global_actions.actions:
            actions_meta.append(
                ActionMetadata(
                    id=action.id,
                    name=action.name,
                    type=action.type,
                    enabled=getattr(action, "enabled", True),
                    source=getattr(action, "source", "custom"),
                    costs=dict(action.costs),
                    description=action.description or "",
                )
            )

        action_space_metadata = ActionSpaceMetadata(
            total_actions=len(actions_meta),
            actions=tuple(actions_meta),
        )

        meter_infos = [
            MeterInfo(
                name=bar.name,
                index=bar.index,
                critical=getattr(bar, "critical", False),
                initial_value=bar.initial,
                observable=True,
                description=bar.description or "",
            )
            for bar in sorted(raw_configs.bars, key=lambda bar: bar.index)
        ]
        meter_metadata = MeterMetadata(meters=tuple(meter_infos))

        enabled_affordances = raw_configs.environment.enabled_affordances
        enabled_set = set(enabled_affordances) if enabled_affordances else None

        affordance_infos: list[AffordanceInfo] = []
        for aff in raw_configs.affordances:
            if enabled_set is None:
                is_enabled = True
            else:
                is_enabled = aff.name in enabled_set or aff.id in enabled_set
            affordance_infos.append(
                AffordanceInfo(
                    id=aff.id,
                    name=aff.name,
                    enabled=is_enabled,
                    effects=self._summarize_affordance_effects(aff),
                    cost=self._extract_money_cost(aff),
                    category=getattr(aff, "category", None),
                    description=aff.description or "",
                )
            )

        affordance_metadata = AffordanceMetadata(affordances=tuple(affordance_infos))

        return action_space_metadata, meter_metadata, affordance_metadata

    def _stage_6_optimize(
        self,
        raw_configs: RawConfigs,
        metadata: UniverseMetadata,
        *,
        device: torch.device | None = None,
    ) -> OptimizationData:
        """Stage 6 – pre-compute optimization tensors and lookup tables."""

        torch_device = device or torch.device("cpu")
        meter_lookup = metadata.meter_name_to_index

        base_depletions = torch.zeros(metadata.meter_count, dtype=torch.float32, device=torch_device)
        for bar in raw_configs.bars:
            index = meter_lookup.get(bar.name, bar.index)
            base_depletions[index] = float(getattr(bar, "base_depletion", 0.0))

        cascade_data: dict[str, list[dict[str, float]]] = defaultdict(list)
        for cascade in raw_configs.cascades:
            source_idx = meter_lookup.get(cascade.source)
            target_idx = meter_lookup.get(cascade.target)
            if source_idx is None or target_idx is None:
                continue
            cascade_data[cascade.category].append(
                {
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "threshold": cascade.threshold,
                    "strength": cascade.strength,
                }
            )

        for category in cascade_data:
            cascade_data[category].sort(key=lambda entry: entry["target_idx"])

        modulation_data: list[dict[str, float]] = []
        cascades_yaml = raw_configs.config_dir / "cascades.yaml"
        try:
            cascades_config = load_full_cascades_config(cascades_yaml)
        except Exception:
            cascades_config = None

        if cascades_config:
            for modulation in cascades_config.modulations:
                source_idx = meter_lookup.get(modulation.source)
                target_idx = meter_lookup.get(modulation.target)
                if source_idx is None or target_idx is None:
                    continue
                modulation_data.append(
                    {
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "base_multiplier": modulation.base_multiplier,
                        "range": modulation.range,
                        "baseline_depletion": modulation.baseline_depletion,
                    }
                )
            modulation_data.sort(key=lambda entry: entry["target_idx"])

        affordance_count = metadata.affordance_count
        action_mask_table = torch.zeros((24, affordance_count), dtype=torch.bool, device=torch_device)

        if affordance_count > 0:
            for hour in range(24):
                for affordance_idx, affordance in enumerate(raw_configs.affordances):
                    hours = getattr(affordance, "operating_hours", None)
                    if not hours:
                        action_mask_table[hour, affordance_idx] = True
                        continue
                    open_hour, close_hour = hours
                    action_mask_table[hour, affordance_idx] = self._is_open(hour, open_hour, close_hour)

        affordance_position_map = {aff.id: None for aff in raw_configs.affordances}

        return OptimizationData(
            base_depletions=base_depletions,
            cascade_data=dict(cascade_data),
            modulation_data=modulation_data,
            action_mask_table=action_mask_table,
            affordance_position_map=affordance_position_map,
        )

    def _stage_7_emit_compiled_universe(
        self,
        *,
        raw_configs: RawConfigs,
        metadata: UniverseMetadata,
        observation_spec: ObservationSpec,
        action_space_metadata: ActionSpaceMetadata,
        meter_metadata: MeterMetadata,
        affordance_metadata: AffordanceMetadata,
        optimization_data: OptimizationData,
    ) -> CompiledUniverse:
        """Stage 7 – produce immutable CompiledUniverse artifact."""

        universe = CompiledUniverse(
            hamlet_config=raw_configs.hamlet_config,
            variables_reference=raw_configs.variables_reference,
            global_actions=raw_configs.global_actions,
            config_dir=raw_configs.config_dir,
            metadata=metadata,
            observation_spec=observation_spec,
            action_space_metadata=action_space_metadata,
            meter_metadata=meter_metadata,
            affordance_metadata=affordance_metadata,
            optimization_data=optimization_data,
        )

        if not dataclasses.is_dataclass(universe):
            raise CompilationError(
                stage="Stage 7: Emit",
                errors=["CompiledUniverse must be a dataclass"],
                hints=["Ensure @dataclass decorator remains applied to CompiledUniverse"],
            )

        try:
            universe.metadata = metadata  # type: ignore[attr-defined]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise CompilationError(
                stage="Stage 7: Emit",
                errors=["CompiledUniverse must be frozen (immutable)"],
                hints=["Annotate CompiledUniverse with @dataclass(frozen=True)"],
            )

        return universe

    def _derive_grid_dimensions(self, substrate: SubstrateConfig) -> tuple[int | None, int | None]:
        if substrate.type == "grid" and substrate.grid is not None:
            width = substrate.grid.width
            height = substrate.grid.height
            return width, width * height
        return None, None

    def _label_substrate_type(self, substrate: SubstrateConfig) -> str:
        if substrate.type != "grid":
            return substrate.type
        if substrate.grid is None:
            return "grid"
        return f"grid_{substrate.grid.topology}"

    def _infer_position_dim(self, substrate: SubstrateConfig) -> int:
        if substrate.type == "aspatial":
            return 0
        if substrate.type == "grid":
            if substrate.grid and substrate.grid.topology == "cubic":
                return 3
            return 2
        if substrate.type == "gridnd" and substrate.gridnd is not None:
            return len(substrate.gridnd.dimension_sizes)
        if substrate.type == "continuous" and substrate.continuous is not None:
            return substrate.continuous.dimensions
        if substrate.type == "continuousnd" and substrate.continuous is not None:
            return len(substrate.continuous.bounds)
        return 0

    def _resolve_config_version(self, raw_configs: RawConfigs) -> str:
        return getattr(raw_configs.hamlet_config, "version", "1.0")

    def _load_observation_exposures(self, config_dir: Path, raw_configs: RawConfigs) -> list[dict[str, Any]]:
        yaml_path = config_dir / "variables_reference.yaml"
        exposures: list[dict[str, Any]] = []
        try:
            with yaml_path.open() as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            data = {}

        raw_exposures = data.get("exposed_observations")
        if raw_exposures:
            exposures = [deepcopy(obs) for obs in raw_exposures]
        else:
            for var in raw_configs.variables_reference:
                readable = getattr(var, "readable_by", []) or []
                if "agent" in readable:
                    exposures.append(
                        {
                            "id": f"obs_{var.id}",
                            "source_variable": var.id,
                            "exposed_to": ["agent"],
                        }
                    )

        if raw_configs.environment.partial_observability:
            exposures = [obs for obs in exposures if obs.get("source_variable") != "grid_encoding"]
        else:
            exposures = [obs for obs in exposures if obs.get("source_variable") != "local_window"]

        return exposures

    def _normalize_yaml(self, file_path: Path) -> str:
        with file_path.open() as handle:
            data = yaml.safe_load(handle) or {}
        return yaml.dump(data, sort_keys=True)

    def _build_cache_fingerprint(self, config_dir: Path) -> tuple[str, str]:
        config_hash = self._compute_config_hash(config_dir)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        from pydantic import __version__ as pydantic_version  # lazy import to avoid startup penalty

        provenance = self._compute_provenance_id(
            config_hash=config_hash,
            compiler_version=COMPILER_VERSION,
            git_sha=self._get_git_sha(),
            python_version=python_version,
            torch_version=torch.__version__,
            pydantic_version=pydantic_version,
        )
        return config_hash, provenance

    def _cache_directory_for(self, config_dir: Path) -> Path:
        """Return the cache directory path for a config pack."""

        return config_dir / ".compiled"

    def _cache_artifact_path(self, config_dir: Path) -> Path:
        """Return the expected cache artifact path for a config pack."""

        return self._cache_directory_for(config_dir) / "universe.msgpack"

    def _prepare_cache_directory(self, cache_dir: Path) -> None:
        """Ensure the cache directory exists and is writable."""

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unable to create cache directory at {cache_dir}: {exc}") from exc

        if not cache_dir.is_dir():
            raise RuntimeError(f"Cache path {cache_dir} exists but is not a directory")

        if not os.access(cache_dir, os.W_OK):
            raise RuntimeError(f"Cache directory {cache_dir} is not writable")

    def _compute_config_hash(self, config_dir: Path) -> str:
        yaml_files = sorted(config_dir.glob("*.yaml"))
        yaml_files.append(Path("configs") / "global_actions.yaml")

        digest = hashlib.sha256()
        for file_path in yaml_files:
            if not file_path.exists():
                continue
            normalized = self._normalize_yaml(file_path)
            digest.update(file_path.name.encode("utf-8"))
            digest.update(normalized.encode("utf-8"))
        return digest.hexdigest()

    def _compute_provenance_id(
        self,
        *,
        config_hash: str,
        compiler_version: str,
        git_sha: str,
        python_version: str,
        torch_version: str,
        pydantic_version: str,
    ) -> str:
        payload = "|".join(
            [
                config_hash,
                compiler_version,
                git_sha,
                python_version,
                torch_version,
                pydantic_version,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_git_sha(self) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _summarize_affordance_effects(self, affordance: AffordanceConfig) -> dict[str, float]:
        totals: defaultdict[str, float] = defaultdict(float)

        def _add_entries(entries: object | None) -> None:
            if not entries:
                return
            for entry in entries:
                meter = self._get_meter(entry)
                amount = self._get_amount(entry)
                if meter and amount is not None:
                    totals[meter] += amount

        pipeline = affordance.effect_pipeline
        if pipeline is not None:
            _add_entries(pipeline.on_start)
            _add_entries(pipeline.per_tick)
            _add_entries(pipeline.on_completion)
            _add_entries(pipeline.on_early_exit)
            _add_entries(pipeline.on_failure)
        else:
            _add_entries(getattr(affordance, "effects", []))
            _add_entries(getattr(affordance, "effects_per_tick", []))
            _add_entries(getattr(affordance, "completion_bonus", []))

        return dict(totals)

    def _extract_money_cost(self, affordance: AffordanceConfig) -> float:
        total = 0.0
        total += self._sum_money_entries(getattr(affordance, "costs", []), positive_only=True)
        total += self._sum_money_entries(getattr(affordance, "costs_per_tick", []), positive_only=True)
        return total

    @staticmethod
    def _is_open(hour: int, open_hour: int, close_hour: int) -> bool:
        """Return True if an affordance is open for the given hour."""

        hour %= 24
        open_mod = open_hour % 24
        close_mod = close_hour % 24

        # 24/7 if interval covers full day
        if (close_hour - open_hour) % 24 == 0:
            return True

        if open_mod < close_mod:
            return open_mod <= hour < close_mod
        return hour >= open_mod or hour < close_mod

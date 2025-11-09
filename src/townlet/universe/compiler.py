"""UniverseCompiler implementation (Stage 1 scaffolding)."""

from __future__ import annotations

import logging
from pathlib import Path

from townlet.config.affordance import AffordanceConfig
from townlet.config.cascade import CascadeConfig
from townlet.config.effect_pipeline import EffectPipeline
from townlet.environment.substrate_action_validator import SubstrateActionValidator
from townlet.substrate.config import SubstrateConfig
from townlet.universe.compiler_inputs import RawConfigs

from .errors import CompilationErrorCollector
from .symbol_table import UniverseSymbolTable

logger = logging.getLogger(__name__)


class UniverseCompiler:
    """Entry point for compiling config packs into CompiledUniverse artifacts."""

    def __init__(self) -> None:
        self._symbol_table = UniverseSymbolTable()

    def compile(self, config_dir: Path, use_cache: bool = True):  # pragma: no cover - placeholder
        """Compile a config pack into a CompiledUniverse.

        Currently only Stage 1 is implemented; later stages will populate metadata
        and emit CompiledUniverse artifacts.
        """

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

        raise NotImplementedError("UniverseCompiler.compile is not yet fully implemented")

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
            for entry in getattr(affordance, "effects", []) or []:
                if self._get_meter(entry) == "money":
                    amount = self._get_amount(entry)
                    if amount and amount > 0:
                        total += amount
            for entry in getattr(affordance, "effects_per_tick", []) or []:
                if self._get_meter(entry) == "money":
                    amount = self._get_amount(entry)
                    if amount and amount > 0:
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

"""UniverseCompiler implementation (Stage 1 scaffolding)."""

from __future__ import annotations

from pathlib import Path

from townlet.universe.compiler_inputs import RawConfigs

from .errors import CompilationErrorCollector
from .symbol_table import UniverseSymbolTable


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

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

        def _record_meter_reference(meter_name: str | None, location: str) -> None:
            if not meter_name:
                return
            try:
                symbol_table.resolve_meter_reference(meter_name, location=location)
            except ReferenceError as exc:
                errors.add(str(exc))

        def _iter_meter_entries(entries: object | None) -> list[str]:
            meters: list[str] = []
            if not entries:
                return meters
            for entry in entries:
                meter = _get_meter(entry)
                if meter:
                    meters.append(meter)
            return meters

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
        for cascade in raw_configs.cascades:
            _record_meter_reference(cascade.source, f"cascades.yaml:{cascade.name}:source")
            _record_meter_reference(cascade.target, f"cascades.yaml:{cascade.name}:target")

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
                for meter in _iter_meter_entries(entries):
                    _record_meter_reference(meter, f"affordances.yaml:{affordance.id}:{label}")

            # Capabilities (meter-gated)
            capabilities = getattr(affordance, "capabilities", None)
            if capabilities:
                for idx, capability in enumerate(capabilities):
                    if _get_attr(capability, "type") == "meter_gated":
                        meter = _get_meter(capability)
                        _record_meter_reference(
                            meter,
                            f"affordances.yaml:{affordance.id}:capabilities[{idx}]",
                        )

            # Effect pipeline stages (if defined)
            effect_pipeline = getattr(affordance, "effect_pipeline", None)
            if effect_pipeline:
                for stage_name in ("on_start", "per_tick", "on_completion", "on_early_exit", "on_failure"):
                    stage_effects = _get_attr(effect_pipeline, stage_name)
                    if stage_effects:
                        for meter in _iter_meter_entries(stage_effects):
                            _record_meter_reference(
                                meter,
                                f"affordances.yaml:{affordance.id}:effect_pipeline.{stage_name}",
                            )

            # Availability constraints
            availability = getattr(affordance, "availability", None)
            if availability:
                for idx, constraint in enumerate(availability):
                    meter = _get_meter(constraint)
                    _record_meter_reference(
                        meter,
                        f"affordances.yaml:{affordance.id}:availability[{idx}]",
                    )

        # Environment enabled_affordances (names)
        enabled_affordances = raw_configs.environment.enabled_affordances
        if enabled_affordances:
            for affordance_name in enabled_affordances:
                if affordance_name not in symbol_table.affordances_by_name:
                    errors.add(
                        "training.yaml:environment.enabled_affordances: "
                        f"References non-existent affordance '{affordance_name}'. "
                        f"Valid affordances: {symbol_table.affordance_names}"
                    )

        # Global action costs/effects (dict[str, float])
        for action in raw_configs.global_actions.actions:
            for meter in action.costs.keys():
                _record_meter_reference(meter, f"global_actions.yaml:{action.name}:costs")
            for meter in action.effects.keys():
                _record_meter_reference(meter, f"global_actions.yaml:{action.name}:effects")

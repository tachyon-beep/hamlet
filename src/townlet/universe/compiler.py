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

        _ = self._stage_1_parse_individual_files(config_dir)
        # TODO(compiler): implement caching and subsequent stages.
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

        return table

    def _stage_3_resolve_references(
        self,
        raw_configs: RawConfigs,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
    ) -> None:
        del raw_configs, symbol_table, errors
        raise NotImplementedError

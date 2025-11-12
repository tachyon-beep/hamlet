"""UniverseCompiler implementation (Stage 1 scaffolding)."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from numbers import Number
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from townlet.config.affordance import AffordanceConfig
from townlet.config.bar import BarConfig
from townlet.config.cascade import CascadeConfig
from townlet.config.drive_as_code import DriveAsCodeConfig, load_drive_as_code_config
from townlet.config.effect_pipeline import EffectPipeline
from townlet.environment.cascade_config import EnvironmentConfig
from townlet.environment.cascade_config import (
    load_cascades_config as load_full_cascades_config,
)
from townlet.environment.substrate_action_validator import SubstrateActionValidator
from townlet.substrate.config import SubstrateConfig
from townlet.universe.adapters.vfs_adapter import VFSAdapter, vfs_to_observation_spec
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
from townlet.vfs.schema import ObservationField as VFSObservationField
from townlet.vfs.schema import VariableDef

from .cues_compiler import CuesCompiler
from .errors import CompilationError, CompilationErrorCollector, CompilationMessage
from .symbol_table import UniverseSymbolTable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
COMPILER_VERSION = "0.1.0"

MAX_METERS = 100
MAX_AFFORDANCES = 100
MAX_CASCADES = 500
MAX_ACTIONS = 300  # Increased for discretized continuous actions (32×7 = 195+)
MAX_VARIABLES = 200
MAX_GRID_CELLS = 10000  # 100×100 maximum (DoS protection)
MAX_CACHE_FILE_SIZE = 10 * 1024 * 1024  # 10MB (cache bomb protection)


class UniverseCompiler:
    """Entry point for compiling config packs into CompiledUniverse artifacts."""

    def __init__(self) -> None:
        self._symbol_table = UniverseSymbolTable()
        self._cues_compiler = CuesCompiler()
        self._metadata: UniverseMetadata | None = None
        self._observation_spec: ObservationSpec | None = None
        self._action_metadata: ActionSpaceMetadata | None = None
        self._meter_metadata: MeterMetadata | None = None
        self._affordance_metadata: AffordanceMetadata | None = None
        self._optimization_data: OptimizationData | None = None

    def compile(self, config_dir: Path, use_cache: bool = True) -> CompiledUniverse:
        """Compile a config pack into a CompiledUniverse (with optional caching)."""

        config_dir = Path(config_dir).resolve()  # Resolve to absolute path
        self._validate_config_dir(config_dir)

        # Phase 0: Validate YAML syntax before doing any work
        self._phase_0_validate_yaml_syntax(config_dir)

        cache_path = self._cache_artifact_path(config_dir)
        precomputed_hash: str | None = None
        precomputed_provenance: str | None = None

        if use_cache and cache_path.exists():
            # Validate cache file size before loading (cache bomb protection)
            cache_size = cache_path.stat().st_size
            if cache_size > MAX_CACHE_FILE_SIZE:
                logger.warning(
                    "Cache file %s exceeds size limit (%d bytes > %d bytes). Recompiling.",
                    cache_path,
                    cache_size,
                    MAX_CACHE_FILE_SIZE,
                )
            else:
                # Optimization: Check mtime first (fast) before computing hash (slow)
                current_mtime = self._compute_config_mtime(config_dir)
                try:
                    cached_universe = CompiledUniverse.load_from_cache(cache_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load cached universe from %s: %s", cache_path, exc)
                else:
                    # Check both content hash AND modification time for cache validity
                    # mtime check catches ANY file change (comments, whitespace, field removal)
                    cached_mtime = cached_universe.metadata.config_mtime
                    if current_mtime > cached_mtime:
                        logger.info(
                            "Cache stale for %s (config files modified: cached_mtime=%.2f, current_mtime=%.2f). Recompiling.",
                            cache_path,
                            cached_mtime,
                            current_mtime,
                        )
                    else:
                        # mtime matches - now compute hash to check content equality
                        precomputed_hash, precomputed_provenance = self._build_cache_fingerprint(config_dir)
                        if (
                            cached_universe.metadata.config_hash == precomputed_hash
                            and cached_universe.metadata.provenance_id == precomputed_provenance
                        ):
                            logger.info("Loaded compiled universe from cache: %s", cache_path)
                            return cached_universe
                        else:
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

        # Load DAC configuration (REQUIRED)
        try:
            dac_config = load_drive_as_code_config(config_dir)
            logger.info("Loaded drive_as_code.yaml")
        except FileNotFoundError as e:
            raise CompilationError(
                stage="Load DAC Configuration",
                errors=[
                    f"drive_as_code.yaml is required but not found in {config_dir}",
                    "All config packs must include a drive_as_code.yaml file",
                ],
                hints=["See docs/config-schemas/drive_as_code.md for creating DAC configs"],
            ) from e

        errors = CompilationErrorCollector(stage="Stage 3: Resolve References")
        self._stage_3_resolve_references(raw_configs, symbol_table, errors)

        # Stage 3: Validate DAC references
        self._validate_dac_references(dac_config, symbol_table, errors)

        errors.check_and_raise("Stage 3: Resolve References")

        stage4_errors = CompilationErrorCollector(stage="Stage 4: Cross-Validation")
        self._stage_4_cross_validate(raw_configs, symbol_table, stage4_errors)
        # Emit warnings even if Stage 4 ultimately raises so operators see every diagnostic.
        for warning in stage4_errors.warnings:
            logger.warning(warning)
        stage4_errors.check_and_raise("Stage 4: Cross-Validation")

        metadata, observation_spec, vfs_fields = self._stage_5_compute_metadata(
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
            symbol_table=symbol_table,
            metadata=metadata,
            observation_spec=observation_spec,
            vfs_observation_fields=vfs_fields,
            action_space_metadata=action_space_metadata,
            meter_metadata=meter_metadata,
            affordance_metadata=affordance_metadata,
            optimization_data=optimization_data,
            environment_config=raw_configs.environment_config,
            dac_config=dac_config,
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

    def _validate_config_dir(self, config_dir: Path) -> None:
        """Validate config_dir for security and sanity.

        Ensures:
        - Path is a directory
        - Path doesn't contain suspicious traversal patterns
        - Path exists

        Raises:
            CompilationError: If validation fails
        """
        if not config_dir.exists():
            raise CompilationError(
                stage="Config Directory Validation",
                errors=[f"Config directory does not exist: {config_dir}"],
            )

        if not config_dir.is_dir():
            raise CompilationError(
                stage="Config Directory Validation",
                errors=[f"Config path is not a directory: {config_dir}"],
            )

        # Warn about suspicious patterns (though resolve() already normalized them)
        path_str = str(config_dir)
        if ".." in path_str:
            logger.warning(
                "Config directory path contains '..' after resolution: %s. This may indicate a path traversal attempt.",
                config_dir,
            )

    def _phase_0_validate_yaml_syntax(self, config_dir: Path) -> None:
        """Phase 0 – validate all YAML files can be parsed before compilation begins."""
        errors = CompilationErrorCollector(stage="Phase 0: YAML Syntax Validation")

        # Required config files (consolidated structure)
        # Note: training.yaml contains training/environment/population/curriculum/exploration sections
        required_files = [
            "training.yaml",
            "bars.yaml",
            "cascades.yaml",
            "affordances.yaml",
            "substrate.yaml",
            "cues.yaml",
            "variables_reference.yaml",  # Required file, but variables list can be empty
        ]

        # Optional files
        optional_files = ["action_labels.yaml"]

        # Also check global actions (outside config_dir)
        global_actions_path = Path("configs") / "global_actions.yaml"
        all_files_to_check = [(config_dir / f, f, True) for f in required_files]
        all_files_to_check.extend([(config_dir / f, f, False) for f in optional_files])
        all_files_to_check.append((global_actions_path, "global_actions.yaml", True))

        for file_path, file_name, is_required in all_files_to_check:
            if not file_path.exists():
                if is_required:
                    errors.add(
                        f"{file_name}: File not found",
                        code="MISSING_FILE",
                        location=str(file_path),
                    )
                continue

            try:
                with file_path.open() as handle:
                    yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                error_msg = str(exc)
                if hasattr(exc, "problem_mark"):
                    mark = exc.problem_mark
                    problem = getattr(exc, "problem", None) or "syntax error"
                    error_msg = f"line {mark.line + 1}, column {mark.column + 1}: {problem}"
                    if hasattr(exc, "context") and exc.context:
                        error_msg = f"{exc.context}\n  {error_msg}"

                errors.add(
                    error_msg,
                    code="YAML_SYNTAX_ERROR",
                    location=file_name,
                )

        if errors.errors:
            errors.add_hint("Check YAML indentation (use spaces, not tabs)")
            errors.add_hint("Ensure lists use proper '- item' syntax")
            errors.add_hint("Validate YAML syntax at yamllint.com or with 'yamllint <file>'")
            errors.check_and_raise()

    def _stage_1_parse_individual_files(self, config_dir: Path) -> RawConfigs:
        """Stage 1 – load all YAML files into DTOs using shared loaders."""

        return RawConfigs.from_config_dir(config_dir)

    def _auto_generate_standard_variables(self, raw_configs: RawConfigs) -> list[VariableDef]:
        """Auto-generate standard system variables from substrate, bars, and affordances.

        Standard variables represent directly observable state that is always available
        in every universe. These are the fundamental building blocks for observations.

        Standard variables include:
        - Spatial: grid_encoding/local_window (what's on the grid), position (where agent is)
        - Meters: One variable per meter from bars.yaml (agent internal state)
        - Affordances: affordance_at_position one-hot encoding (what's at current position)
        - Temporal: time_sin, time_cos, interaction_progress, lifetime_progress (time state)

        Custom variables (defined in variables_reference.yaml) should extend these with:
        1. Environmental phenomena: Weather, lighting, noise (world state)
        2. Derived features: Ratios, deficits, progress metrics (computed from state)

        All variables (standard + custom) are automatically exposed as observations.

        Returns:
            List of auto-generated VariableDef objects
        """
        variables: list[VariableDef] = []

        # 1. SPATIAL VARIABLES (substrate-dependent)
        substrate = raw_configs.substrate
        if substrate.type == "grid" and substrate.grid is not None:
            width = substrate.grid.width
            height = substrate.grid.height

            # Check for 3D grid (depth field)
            depth = getattr(substrate.grid, "depth", None)
            is_3d = depth is not None

            # Calculate grid cells based on dimensionality
            if is_3d:
                assert depth is not None  # Type narrowing for mypy
                grid_cells = width * height * depth
                grid_desc = f"{width}×{height}×{depth} grid encoding (0=empty, 1=agent, 2=affordance, 3=both)"
                position_dims = 3
                position_default = [0.0, 0.0, 0.0]
                position_desc = "Normalized agent position (x, y, z) in [0, 1] range"
            else:
                grid_cells = width * height
                grid_desc = f"{width}×{height} grid encoding (0=empty, 1=agent, 2=affordance, 3=both)"
                position_dims = 2
                position_default = [0.0, 0.0]
                position_desc = "Normalized agent position (x, y) in [0, 1] range"

            # Grid encoding for full observability
            variables.append(
                VariableDef(
                    id="grid_encoding",
                    scope="agent",
                    type="vecNf",
                    dims=grid_cells,
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=[0.0] * grid_cells,
                    description=grid_desc,
                )
            )

            # Local window for POMDP (if partial observability enabled)
            if raw_configs.environment.partial_observability:
                vision_range = raw_configs.environment.vision_range or 3
                if is_3d:
                    # 3D POMDP: (2r+1)³ window
                    window_size = (2 * vision_range + 1) ** 3
                    window_dim = 2 * vision_range + 1
                    window_desc = f"{window_dim}×{window_dim}×{window_dim} local observation window (POMDP 3D)"
                else:
                    # 2D POMDP: (2r+1)² window
                    window_size = (2 * vision_range + 1) ** 2
                    window_dim = 2 * vision_range + 1
                    window_desc = f"{window_dim}×{window_dim} local observation window (POMDP)"

                variables.append(
                    VariableDef(
                        id="local_window",
                        scope="agent",
                        type="vecNf",
                        dims=window_size,
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=[0.0] * window_size,
                        description=window_desc,
                    )
                )

            # Position (normalized coordinates)
            variables.append(
                VariableDef(
                    id="position",
                    scope="agent",
                    type="vecNf",
                    dims=position_dims,
                    lifetime="episode",
                    readable_by=["agent", "engine", "acs"],
                    writable_by=["actions", "engine"],
                    default=position_default,
                    description=position_desc,
                )
            )

            # Velocity tracking (movement delta from previous step)
            # Enables agents to remember their movement direction and speed
            variables.extend(
                [
                    VariableDef(
                        id="velocity_x",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="X-component of velocity (movement delta since last step)",
                    ),
                    VariableDef(
                        id="velocity_y",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="Y-component of velocity (movement delta since last step)",
                    ),
                ]
            )

            # 3D velocity component
            if is_3d:
                variables.append(
                    VariableDef(
                        id="velocity_z",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="Z-component of velocity (movement delta since last step)",
                    )
                )

            # Velocity magnitude (speed)
            variables.append(
                VariableDef(
                    id="velocity_magnitude",
                    scope="agent",
                    type="scalar",
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=0.0,
                    description="Speed (magnitude of velocity vector)",
                )
            )

        elif substrate.type in ("continuous", "continuousnd"):
            # Continuous substrates: position + velocity (no grid encoding)
            position_dims = self._infer_position_dim(substrate)

            variables.append(
                VariableDef(
                    id="position",
                    scope="agent",
                    type="vecNf",
                    dims=position_dims,
                    lifetime="episode",
                    readable_by=["agent", "engine", "acs"],
                    writable_by=["actions", "engine"],
                    default=[0.0] * position_dims,
                    description=f"Agent position in {position_dims}D continuous space",
                )
            )

            # Velocity tracking for continuous substrates
            if position_dims >= 1:
                variables.append(
                    VariableDef(
                        id="velocity_x",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="X-component of velocity (continuous movement delta)",
                    )
                )

            if position_dims >= 2:
                variables.append(
                    VariableDef(
                        id="velocity_y",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="Y-component of velocity (continuous movement delta)",
                    )
                )

            if position_dims >= 3:
                variables.append(
                    VariableDef(
                        id="velocity_z",
                        scope="agent",
                        type="scalar",
                        lifetime="tick",
                        readable_by=["agent", "engine"],
                        writable_by=["engine"],
                        default=0.0,
                        description="Z-component of velocity (continuous movement delta)",
                    )
                )

            # Velocity magnitude
            variables.append(
                VariableDef(
                    id="velocity_magnitude",
                    scope="agent",
                    type="scalar",
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=0.0,
                    description="Speed (magnitude of velocity vector)",
                )
            )

        elif substrate.type == "aspatial":
            # Aspatial has no spatial variables
            pass

        # 2. METER VARIABLES (from bars.yaml)
        for bar in sorted(raw_configs.bars, key=lambda b: b.index):
            variables.append(
                VariableDef(
                    id=bar.name,
                    scope="agent",
                    type="scalar",
                    lifetime="episode",
                    readable_by=["agent", "engine", "acs"],
                    writable_by=["actions", "engine"],
                    default=1.0,  # Meters typically start at 100%
                    description=f"{bar.name.capitalize()} level [0.0-1.0]",
                )
            )

        # 3. AFFORDANCE VARIABLES
        affordance_count = len(raw_configs.affordances) + 1  # +1 for "none"
        variables.append(
            VariableDef(
                id="affordance_at_position",
                scope="agent",
                type="vecNf",
                dims=affordance_count,
                lifetime="tick",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=[0.0] * (affordance_count - 1) + [1.0],  # Last element (none) = 1.0
                description=f"One-hot encoding of affordance at agent position ({affordance_count} categories)",
            )
        )

        # 4. TEMPORAL VARIABLES (standard)
        variables.extend(
            [
                VariableDef(
                    id="time_sin",
                    scope="global",
                    type="scalar",
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=0.0,
                    description="Sine of current time (24-hour cycle)",
                ),
                VariableDef(
                    id="time_cos",
                    scope="global",
                    type="scalar",
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=1.0,
                    description="Cosine of current time (24-hour cycle)",
                ),
                VariableDef(
                    id="interaction_progress",
                    scope="agent",
                    type="scalar",
                    lifetime="tick",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=0.0,
                    description="Progress through current multi-tick interaction [0.0-1.0]",
                ),
                VariableDef(
                    id="lifetime_progress",
                    scope="agent",
                    type="scalar",
                    lifetime="episode",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    default=0.0,
                    description="Progress through episode lifetime [0.0-1.0]",
                ),
            ]
        )

        return variables

    def _stage_2_build_symbol_tables(self, raw_configs: RawConfigs) -> UniverseSymbolTable:
        """Stage 2 – register meters, variables, actions, cascades, and affordances."""

        table = UniverseSymbolTable()

        for bar in raw_configs.bars:
            table.register_meter(bar)

        # Auto-generate standard system variables
        auto_generated_vars = self._auto_generate_standard_variables(raw_configs)

        # Get user-defined variable IDs to avoid duplicates
        user_defined_var_ids = set()
        if raw_configs.variables_reference is not None:
            user_defined_var_ids = {var.id for var in raw_configs.variables_reference}

        # Register auto-generated variables only if not overridden by user
        for variable in auto_generated_vars:
            if variable.id not in user_defined_var_ids:
                table.register_variable(variable)

        # Register custom user-defined variables (these take precedence over auto-generated)
        if raw_configs.variables_reference is not None:
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

        def _format_error(code: str, message: str, location: str | None = None) -> CompilationMessage:
            location_str = None
            if location and source_map is not None:
                location_str = source_map.lookup(location)
            if not location_str:
                location_str = location
            return CompilationMessage(code=code, message=message, location=location_str)

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
                if not entries or not isinstance(entries, Sequence):
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
            if capabilities and isinstance(capabilities, Sequence):
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
                    if not stage_effects or not isinstance(stage_effects, Sequence):
                        continue
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
            if availability and isinstance(availability, Sequence):
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
                            f"References non-existent affordance '{affordance_name}'. Valid affordances: {symbol_table.affordance_names}",
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

    def _validate_dac_references(
        self,
        dac_config: DriveAsCodeConfig,
        symbol_table: UniverseSymbolTable,
        errors: CompilationErrorCollector,
    ) -> None:
        """Validate DAC references to bars, variables, and affordances.

        Stage 3 validation: Ensures DAC configurations reference valid entities.

        Checks:
        - Modifiers reference valid bars or VFS variables
        - Extrinsic strategies reference valid bars/variables
        - Shaping bonuses reference valid affordances
        """
        # Validate modifier sources
        for mod_name, mod_config in dac_config.modifiers.items():
            if mod_config.bar:
                if mod_config.bar not in symbol_table.meters:
                    errors.add(
                        CompilationMessage(
                            code="DAC-REF-001",
                            message=f"Modifier '{mod_name}' references undefined bar: {mod_config.bar}",
                            location=f"drive_as_code.yaml:modifiers.{mod_name}",
                        )
                    )
            elif mod_config.variable:
                if mod_config.variable not in symbol_table.vfs_variables:
                    errors.add(
                        CompilationMessage(
                            code="DAC-REF-002",
                            message=f"Modifier '{mod_name}' references undefined VFS variable: {mod_config.variable}",
                            location=f"drive_as_code.yaml:modifiers.{mod_name}",
                        )
                    )

        # Validate extrinsic strategy bar references
        if dac_config.extrinsic.bars:
            for bar in dac_config.extrinsic.bars:
                if bar not in symbol_table.meters:
                    errors.add(
                        CompilationMessage(
                            code="DAC-REF-003",
                            message=f"Extrinsic strategy references undefined bar: {bar}",
                            location="drive_as_code.yaml:extrinsic.bars",
                        )
                    )

        # Validate extrinsic bar_bonuses (if present)
        for idx, bonus in enumerate(dac_config.extrinsic.bar_bonuses):
            if bonus.bar not in symbol_table.meters:
                errors.add(
                    CompilationMessage(
                        code="DAC-REF-004",
                        message=f"Extrinsic bar bonus references undefined bar: {bonus.bar}",
                        location=f"drive_as_code.yaml:extrinsic.bar_bonuses[{idx}]",
                    )
                )

        # Validate extrinsic variable_bonuses (if present)
        for idx, bonus in enumerate(dac_config.extrinsic.variable_bonuses):
            if bonus.variable not in symbol_table.vfs_variables:
                errors.add(
                    CompilationMessage(
                        code="DAC-REF-005",
                        message=f"Extrinsic variable bonus references undefined VFS variable: {bonus.variable}",
                        location=f"drive_as_code.yaml:extrinsic.variable_bonuses[{idx}]",
                    )
                )

        # Validate shaping bonus affordance references (if present)
        for idx, shaping in enumerate(dac_config.shaping):
            if shaping.type == "approach_reward":
                if shaping.target_affordance not in symbol_table.affordances:
                    errors.add(
                        CompilationMessage(
                            code="DAC-REF-006",
                            message=f"Shaping bonus references undefined affordance: {shaping.target_affordance}",
                            location=f"drive_as_code.yaml:shaping[{idx}]",
                        )
                    )

    def _compute_dac_hash(self, dac_config: DriveAsCodeConfig) -> str:
        """Compute SHA256 content hash of DAC configuration for provenance.

        Args:
            dac_config: DAC configuration to hash

        Returns:
            SHA256 hex digest (64 character string)

        Purpose:
            - Checkpoint validation (detect DAC changes)
            - Provenance tracking (which drive functions were used)
            - Reproducibility (verify exact reward configuration)

        Example:
            >>> dac = DriveAsCodeConfig(...)
            >>> hash_val = self._compute_dac_hash(dac)
            >>> len(hash_val)
            64
        """
        import hashlib
        import json

        # Convert to dict for stable JSON serialization
        dac_dict = dac_config.model_dump(mode="json")

        # Compute SHA256 hash with sorted keys for determinism
        json_str = json.dumps(dac_dict, sort_keys=True)
        hash_digest = hashlib.sha256(json_str.encode()).hexdigest()

        return hash_digest

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

        def _format_error(code: str, message: str, location: str | None = None) -> CompilationMessage:
            location_str = None
            if location and source_map is not None:
                location_str = source_map.lookup(location)
            if not location_str:
                location_str = location
            return CompilationMessage(code=code, message=message, location=location_str)

        self._validate_spatial_feasibility(raw_configs, errors, _format_error)
        self._enforce_security_limits(raw_configs, errors)
        allow_unfeasible = bool(getattr(raw_configs.training, "allow_unfeasible_universe", False))
        self._validate_economic_balance(raw_configs, errors, _format_error, allow_unfeasible)
        self._validate_cascade_cycles(raw_configs, errors, _format_error)
        self._validate_operating_hours(raw_configs, errors, _format_error)
        self._validate_availability_and_modes(raw_configs, symbol_table, errors, _format_error)
        self._cues_compiler.validate(raw_configs.hamlet_config.cues, symbol_table, errors, _format_error)
        self._validate_capabilities_and_effect_pipelines(raw_configs, errors, _format_error)
        self._validate_affordance_positions(raw_configs, errors, _format_error)
        self._validate_substrate_action_compatibility(raw_configs, errors, _format_error, _add_hint)
        self._validate_capacity_and_sustainability(raw_configs, errors, _format_error, allow_unfeasible)

    def _validate_spatial_feasibility(self, raw_configs: RawConfigs, errors: CompilationErrorCollector, formatter) -> None:
        """Validate that substrate has enough space for all affordances + agent.

        Reads spatial dimensions from substrate.yaml (single source of truth).
        Only applies to discrete grid substrates (grid, gridnd).
        """
        substrate = raw_configs.substrate

        # Calculate grid cells based on substrate type
        grid_cells: int | None = None
        dimensions_str = ""

        if substrate.type == "grid" and substrate.grid is not None:
            # Grid2D (square) or Grid3D (cubic)
            if substrate.grid.topology == "square":
                grid_cells = substrate.grid.width * substrate.grid.height
                dimensions_str = f"{substrate.grid.width}×{substrate.grid.height}"
            elif substrate.grid.topology == "cubic":
                if substrate.grid.depth is not None:
                    grid_cells = substrate.grid.width * substrate.grid.height * substrate.grid.depth
                    dimensions_str = f"{substrate.grid.width}×{substrate.grid.height}×{substrate.grid.depth}"
        elif substrate.type == "gridnd" and substrate.gridnd is not None:
            # GridND (N-dimensional discrete grid)
            grid_cells = 1
            for dim_size in substrate.gridnd.dimension_sizes:
                grid_cells *= dim_size
            dimensions_str = "×".join(str(d) for d in substrate.gridnd.dimension_sizes)
        else:
            # Continuous, aspatial - no spatial feasibility check
            return

        if grid_cells is None or grid_cells <= 0:
            return

        # Enforce upper bound for DoS protection
        if grid_cells > MAX_GRID_CELLS:
            errors.add(
                formatter(
                    "UAC-VAL-001",
                    f"Grid size exceeds safety limit: {grid_cells} cells (max {MAX_GRID_CELLS})",
                    "substrate.yaml:grid",
                )
            )
            return

        enabled_affordances = raw_configs.environment.enabled_affordances
        if enabled_affordances is None:
            required = len(raw_configs.affordances)
        else:
            required = len(enabled_affordances)

        required_cells = required + 1  # +1 for the agent
        if required_cells > grid_cells:
            message = (
                f"Spatial impossibility: Grid has {grid_cells} cells ({dimensions_str}) but need {required_cells} "
                f"({required} affordances + 1 agent)."
            )
            errors.add(formatter("UAC-VAL-001", message, "substrate.yaml:grid"))

    def _enforce_security_limits(self, raw_configs: RawConfigs, errors: CompilationErrorCollector) -> None:
        checks = (
            (len(raw_configs.bars), MAX_METERS, "bars.yaml", "meters"),
            (len(raw_configs.affordances), MAX_AFFORDANCES, "affordances.yaml", "affordances"),
            (len(raw_configs.cascades), MAX_CASCADES, "cascades.yaml", "cascades"),
            (len(raw_configs.global_actions.actions), MAX_ACTIONS, "configs/global_actions.yaml", "actions"),
            (len(raw_configs.variables_reference), MAX_VARIABLES, "variables_reference.yaml", "variables"),
        )

        for count, limit, location, label in checks:
            if count > limit:
                errors.add(
                    f"Too many {label}: found {count} (max {limit}). This may indicate config injection or duplication.",
                    code="UAC-VAL-006",
                    location=location,
                )

    def _validate_economic_balance(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
        allow_unfeasible: bool,
    ) -> None:
        enabled_lookup = self._build_enabled_affordance_lookup(getattr(raw_configs.environment, "enabled_affordances", None))

        total_income = self._compute_max_income(raw_configs.affordances)
        total_costs = self._compute_total_costs(raw_configs.affordances)

        if total_income <= 0.0 and total_costs > 0.0:
            self._record_feasibility_issue(
                errors,
                formatter,
                allow_unfeasible,
                "UAC-VAL-002",
                "No income-generating affordances available while costs accrue. Universe is unwinnable.",
                "affordances.yaml",
            )
        elif total_income < total_costs:
            errors.add_warning(
                formatter(
                    "UAC-VAL-002",
                    f"Economic imbalance: Total income ({total_income:.2f}) < total costs ({total_costs:.2f}).",
                    "affordances.yaml",
                )
            )

        income_hours = self._count_income_hours(raw_configs, enabled_lookup)
        if total_income > 0.0 and income_hours == 0:
            self._record_feasibility_issue(
                errors,
                formatter,
                allow_unfeasible,
                "UAC-VAL-002",
                (
                    "Income-generating affordances exist but none are available during the day. "
                    "Adjust operating_hours or enable additional jobs."
                ),
                "affordances.yaml",
            )
        elif 0 < income_hours < 12:
            errors.add_warning(
                formatter(
                    "UAC-VAL-002",
                    f"Income stress: jobs only available {income_hours:.0f}h/day. Costs accrue 24h/day.",
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

    def _validate_capabilities_and_effect_pipelines(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        # Build affordance ID set for prerequisite validation
        affordance_ids = {aff.id for aff in raw_configs.affordances}
        # Build meter name set for skill_scaling validation
        meter_names = {bar.name for bar in raw_configs.bars}

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

            # Validate capability-specific references (combined loop for efficiency)
            for idx, capability in enumerate(capabilities):
                cap_type = self._get_attr_value(capability, "type")

                # UAC-VAL-010: Validate prerequisite affordance references
                if cap_type == "prerequisite":
                    required = self._get_attr_value(capability, "required_affordances") or []
                    for req_id in required:
                        if req_id not in affordance_ids:
                            errors.add(
                                formatter(
                                    "UAC-VAL-010",
                                    f"Prerequisite affordance '{req_id}' does not exist in affordances.yaml",
                                    f"affordances.yaml:{affordance.id}:capabilities[{idx}]",
                                )
                            )

                # UAC-VAL-012: Validate skill_scaling meter references
                elif cap_type == "skill_scaling":
                    skill_meter = self._get_attr_value(capability, "skill")
                    if skill_meter and skill_meter not in meter_names:
                        errors.add(
                            formatter(
                                "UAC-VAL-012",
                                f"Skill scaling capability references non-existent meter '{skill_meter}'. "
                                f"Valid meters: {sorted(meter_names)}",
                                f"affordances.yaml:{affordance.id}:capabilities[{idx}]",
                            )
                        )

            # UAC-VAL-011: Validate probabilistic effect pipeline completeness
            has_probabilistic = any(self._get_attr_value(cap, "type") == "probabilistic" for cap in capabilities)

            if has_probabilistic:
                if pipeline is None:
                    errors.add(
                        formatter(
                            "UAC-VAL-011",
                            f"Probabilistic affordance '{affordance.id}' must define effect_pipeline with on_completion and on_failure",
                            f"affordances.yaml:{affordance.id}",
                        )
                    )
                else:
                    missing_stages = []
                    if not pipeline.on_completion:
                        missing_stages.append("on_completion (success path)")
                    if not pipeline.on_failure:
                        missing_stages.append("on_failure (failure path)")

                    if missing_stages:
                        errors.add(
                            formatter(
                                "UAC-VAL-011",
                                f"Probabilistic affordance '{affordance.id}' should define both success and failure effects. "
                                f"Missing: {', '.join(missing_stages)}",
                                f"affordances.yaml:{affordance.id}:effect_pipeline",
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

    def _validate_capacity_and_sustainability(
        self,
        raw_configs: RawConfigs,
        errors: CompilationErrorCollector,
        formatter,
        allow_unfeasible: bool,
    ) -> None:
        enabled_lookup = self._build_enabled_affordance_lookup(getattr(raw_configs.environment, "enabled_affordances", None))
        self._validate_meter_sustainability(raw_configs, enabled_lookup, errors, formatter, allow_unfeasible)
        self._validate_capacity_constraints(raw_configs, enabled_lookup, errors, formatter)

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
                    return False, f"Position {position} outside grid bounds 0-{width - 1}, 0-{height - 1}."
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
        for entry in self._iter_entries(entries):
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
        for entry in self._iter_entries(entries):
            amount = self._get_amount(entry)
            if amount is not None:
                total += amount
        return total

    def _sum_positive_meter_entries(self, entries: object | None, meter_name: str) -> float:
        total = 0.0
        for entry in self._iter_entries(entries):
            if self._get_meter(entry) != meter_name:
                continue
            amount = self._get_amount(entry)
            if amount is None or amount <= 0:
                continue
            total += amount
        return total

    def _build_enabled_affordance_lookup(self, enabled_affordances: list[str] | None) -> set[str] | None:
        if not enabled_affordances:
            return None
        return {str(name) for name in enabled_affordances}

    def _is_affordance_enabled(self, affordance: AffordanceConfig, enabled_lookup: set[str] | None) -> bool:
        if enabled_lookup is None:
            return True
        return affordance.name in enabled_lookup or affordance.id in enabled_lookup

    def _count_income_hours(self, raw_configs: RawConfigs, enabled_lookup: set[str] | None) -> float:
        income_affordances = [
            aff
            for aff in raw_configs.affordances
            if self._is_affordance_enabled(aff, enabled_lookup) and self._affordance_positive_amount_for_meter(aff, "money") > 0
        ]
        if not income_affordances:
            return 0.0

        # If temporal mechanics are disabled, operating_hours are ignored (all affordances available 24/7)
        if not raw_configs.environment.enable_temporal_mechanics:
            return 24.0

        if any(getattr(aff, "operating_hours", None) is None for aff in income_affordances):
            return 24.0

        hours_with_income = 0
        for hour in range(24):
            if any(self._affordance_open_for_hour(aff, hour) for aff in income_affordances):
                hours_with_income += 1
        return float(hours_with_income)

    def _affordance_open_for_hour(self, affordance: AffordanceConfig, hour: int) -> bool:
        operating_hours = getattr(affordance, "operating_hours", None)
        if not operating_hours:
            return True
        open_hour, close_hour = operating_hours
        return self._is_open(hour, open_hour, close_hour)

    def _affordance_positive_amount_for_meter(self, affordance: AffordanceConfig, meter_name: str) -> float:
        pipeline = getattr(affordance, "effect_pipeline", None)
        total = 0.0
        if pipeline is not None and not isinstance(pipeline, EffectPipeline):
            pipeline = EffectPipeline.model_validate(pipeline)

        if pipeline is not None:
            total += self._sum_positive_meter_entries(pipeline.on_start, meter_name)
            total += self._sum_positive_meter_entries(pipeline.per_tick, meter_name)
            total += self._sum_positive_meter_entries(pipeline.on_completion, meter_name)
            total += self._sum_positive_meter_entries(pipeline.on_early_exit, meter_name)
            total += self._sum_positive_meter_entries(pipeline.on_failure, meter_name)
        else:
            total += self._sum_positive_meter_entries(getattr(affordance, "effects", []), meter_name)
            total += self._sum_positive_meter_entries(getattr(affordance, "effects_per_tick", []), meter_name)
            total += self._sum_positive_meter_entries(getattr(affordance, "completion_bonus", []), meter_name)

        return total

    def _compute_max_restoration_for_meter(
        self,
        meter_name: str,
        affordances: tuple[AffordanceConfig, ...],
        enabled_lookup: set[str] | None,
    ) -> float:
        max_restoration = 0.0
        for affordance in affordances:
            if not self._is_affordance_enabled(affordance, enabled_lookup):
                continue
            restoration = self._affordance_positive_amount_for_meter(affordance, meter_name)
            if restoration > max_restoration:
                max_restoration = restoration
        return max_restoration

    def _validate_meter_sustainability(
        self,
        raw_configs: RawConfigs,
        enabled_lookup: set[str] | None,
        errors: CompilationErrorCollector,
        formatter,
        allow_unfeasible: bool,
    ) -> None:
        critical_meter_names = self._collect_critical_meter_names(raw_configs)
        if not critical_meter_names:
            return

        for bar in raw_configs.bars:
            if bar.name not in critical_meter_names:
                continue
            depletion = float(getattr(bar, "base_depletion", 0.0))
            if depletion <= 0.0:
                continue
            restoration = self._compute_max_restoration_for_meter(bar.name, raw_configs.affordances, enabled_lookup)
            if restoration <= 0.0:
                self._record_feasibility_issue(
                    errors,
                    formatter,
                    allow_unfeasible,
                    "UAC-VAL-005",
                    f"Meter {bar.name} unsustainable: passive depletion {depletion:.4f}/tick but no restoring affordances are enabled.",
                    f"bars.yaml:{bar.name}",
                )
            elif restoration < depletion:
                self._record_feasibility_issue(
                    errors,
                    formatter,
                    allow_unfeasible,
                    "UAC-VAL-005",
                    f"Meter {bar.name} unsustainable: depletion ({depletion:.4f}/tick) > max restoration ({restoration:.4f}/tick).",
                    f"bars.yaml:{bar.name}",
                )

    def _collect_critical_meter_names(self, raw_configs: RawConfigs) -> set[str]:
        names: set[str] = set()
        for bar in raw_configs.bars:
            if self._is_meter_critical(bar):
                names.add(bar.name)
        return names

    def _is_meter_critical(self, bar: BarConfig) -> bool:
        if getattr(bar, "critical", False):
            return True
        tier = getattr(bar, "tier", None)
        return isinstance(tier, str) and tier.lower() == "pivotal"

    def _validate_capacity_constraints(
        self,
        raw_configs: RawConfigs,
        enabled_lookup: set[str] | None,
        errors: CompilationErrorCollector,
        formatter,
    ) -> None:
        num_agents = getattr(raw_configs.population, "num_agents", 1)
        if num_agents <= 1:
            return

        critical_affordances = self._find_critical_path_affordances(raw_configs, enabled_lookup)
        for affordance in critical_affordances:
            capacity = getattr(affordance, "capacity", None)
            if capacity is None:
                continue
            if capacity < num_agents:
                errors.add_warning(
                    formatter(
                        "UAC-VAL-005",
                        f"Affordance {affordance.name} capacity {capacity} < num_agents ({num_agents}). Contentions may cause starvation.",
                        f"affordances.yaml:{affordance.id}",
                    )
                )

    def _find_critical_path_affordances(
        self,
        raw_configs: RawConfigs,
        enabled_lookup: set[str] | None,
    ) -> list[AffordanceConfig]:
        critical_meters = self._collect_critical_meter_names(raw_configs)
        if not critical_meters:
            return []

        critical_affordances: list[AffordanceConfig] = []
        for affordance in raw_configs.affordances:
            if not self._is_affordance_enabled(affordance, enabled_lookup):
                continue
            if any(self._affordance_positive_amount_for_meter(affordance, meter) > 0.0 for meter in critical_meters):
                critical_affordances.append(affordance)
        return critical_affordances

    def _record_feasibility_issue(
        self,
        errors: CompilationErrorCollector,
        formatter,
        allow_unfeasible: bool,
        code: str,
        message: str,
        location: str,
    ) -> None:
        issue = formatter(code, message, location)
        if allow_unfeasible:
            errors.add_warning(f"{issue.format()} (allow_unfeasible_universe=true)")
        else:
            errors.add(issue)

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

    @staticmethod
    def _iter_entries(entries: object | None) -> Iterable[object]:
        if entries is None:
            return ()
        if isinstance(entries, Iterable) and not isinstance(entries, str | bytes):
            return entries
        return ()

    def _build_cascade_graph(self, cascades: tuple[CascadeConfig, ...]) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for cascade in cascades:
            graph.setdefault(cascade.source, []).append(cascade.target)
        return graph

    def _detect_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
        """Detect cycles in cascade dependency graph using depth-first search.

        Algorithm: DFS with path tracking to identify back edges (cycles).
        Time Complexity: O(V + E) where V=number of meters, E=number of cascades
        Space Complexity: O(V) for visited set and recursion stack

        Args:
            graph: Adjacency list mapping source meter -> list of target meters

        Returns:
            List of cycles, where each cycle is a list of meter names forming a loop
        """
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
    ) -> tuple[UniverseMetadata, ObservationSpec, tuple[VFSObservationField, ...]]:
        """Stage 5 – compute derived metadata and observation specification."""

        import torch
        from pydantic import __version__ as pydantic_version  # lazy import to avoid startup penalty

        exposures = self._load_observation_exposures(raw_configs, symbol_table)

        # Get all variables from symbol table (auto-generated + custom)
        all_variables = list(symbol_table.variables.values())
        variable_registry = VariableRegistry(
            variables=all_variables,
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

        # Compute config mtime for cache invalidation
        config_mtime = self._compute_config_mtime(config_dir)

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
            config_mtime=config_mtime,
            provenance_id=provenance_id,
            compiler_git_sha=compiler_git_sha,
            python_version=python_version,
            torch_version=torch.__version__,
            pydantic_version=pydantic_version,
        )

        return metadata, observation_spec, tuple(vfs_fields)

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
                    enabled=action.enabled,
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
                    position=self._normalize_affordance_position_metadata(getattr(aff, "position", None)),
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
            category_key = cascade.category or "uncategorized"
            cascade_data[category_key].append(
                {
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "threshold": cascade.threshold,
                    "strength": cascade.strength,
                }
            )

        for category, entries in cascade_data.items():
            entries.sort(key=lambda entry: entry["target_idx"])

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

        affordance_position_map = {
            aff.id: self._tensorize_affordance_position(getattr(aff, "position", None), torch_device) for aff in raw_configs.affordances
        }

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
        symbol_table: UniverseSymbolTable,
        metadata: UniverseMetadata,
        observation_spec: ObservationSpec,
        vfs_observation_fields: tuple[VFSObservationField, ...],
        action_space_metadata: ActionSpaceMetadata,
        meter_metadata: MeterMetadata,
        affordance_metadata: AffordanceMetadata,
        optimization_data: OptimizationData,
        environment_config: EnvironmentConfig,
        dac_config: DriveAsCodeConfig | None,
    ) -> CompiledUniverse:
        """Stage 7 – produce immutable CompiledUniverse artifact."""

        # Get all variables from symbol table (auto-generated + custom)
        all_variables = list(symbol_table.variables.values())

        # Build observation activity metadata for masking
        # Convert observation_spec fields back to VFS format for activity building
        from typing import Literal

        from townlet.vfs.schema import ObservationField as VFSObservField

        vfs_fields_for_activity = []
        for field in observation_spec.fields:
            # Map semantic_type from observation spec to VFS field
            semantic_map: dict[str | None, Literal["bars", "spatial", "affordance", "temporal", "custom"]] = {
                "position": "spatial",
                "meter": "bars",
                "affordance": "affordance",
                "temporal": "temporal",
                None: "custom",
            }
            semantic_type = semantic_map.get(field.semantic_type, "custom")

            # Create VFS field with curriculum_active=True (all fields in observation_spec are active)
            vfs_fields_for_activity.append(
                VFSObservField(
                    id=field.name,
                    source_variable=field.description or field.name,
                    exposed_to=["agent"],
                    shape=[field.dims] if field.dims > 0 else [],
                    normalization=None,
                    semantic_type=semantic_type,
                    curriculum_active=True,  # All fields in observation_spec are active
                )
            )

        field_uuids = {field.name: field.uuid for field in observation_spec.fields if field.uuid is not None}
        observation_activity = VFSAdapter.build_observation_activity(
            observation_spec=vfs_fields_for_activity,
            field_uuids=field_uuids,
        )

        # Compute drive_hash (Task 2.3)
        drive_hash = self._compute_dac_hash(dac_config)

        universe = CompiledUniverse(
            hamlet_config=raw_configs.hamlet_config,
            variables_reference=all_variables,
            global_actions=raw_configs.global_actions,
            config_dir=raw_configs.config_dir,
            metadata=metadata,
            observation_spec=observation_spec,
            observation_activity=observation_activity,
            vfs_observation_fields=vfs_observation_fields,
            action_space_metadata=action_space_metadata,
            meter_metadata=meter_metadata,
            affordance_metadata=affordance_metadata,
            optimization_data=optimization_data,
            action_labels_config=raw_configs.action_labels,
            environment_config=environment_config,
            dac_config=dac_config,
            drive_hash=drive_hash,
        )

        if not dataclasses.is_dataclass(universe):
            raise CompilationError(
                stage="Stage 7: Emit",
                errors=["CompiledUniverse must be a dataclass"],
                hints=["Ensure @dataclass decorator remains applied to CompiledUniverse"],
            )

        try:
            setattr(cast(Any, universe), "metadata", metadata)
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
        """Calculate grid dimensions for metadata.

        Returns:
            (grid_size, grid_cells): For square grids, grid_size=width.
                                     For non-square or non-grid substrates, returns (None, None).
        """
        if substrate.type == "grid" and substrate.grid is not None:
            width = substrate.grid.width
            height = substrate.grid.height

            # Handle 3D grids
            depth = getattr(substrate.grid, "depth", None)
            if depth is not None:
                # For 3D, grid_size is ambiguous (use width as representative)
                grid_cells = width * height * depth
                return width, grid_cells

            # 2D grid
            grid_cells = width * height

            # Only return grid_size if square (for backward compatibility)
            if width == height:
                return width, grid_cells
            else:
                # Non-square grids: grid_size concept doesn't apply
                return None, grid_cells

        # Continuous, aspatial, gridnd: no grid_size concept
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

    def _auto_generate_standard_exposures(self, symbol_table: UniverseSymbolTable) -> list[dict[str, Any]]:
        """Auto-generate standard observation exposures for all system variables.

        Creates exposures for:
        - Spatial variables (grid_encoding/local_window, position)
        - All meters
        - Affordance encoding
        - Temporal variables

        Returns:
            List of exposure dictionaries matching the expected schema
        """
        exposures: list[dict[str, Any]] = []

        # Get all variables from symbol table
        for var_id, var in symbol_table.variables.items():
            # Determine observation shape
            if var.type == "scalar":
                shape: list[int] = []
            elif var.type == "vecNf" and var.dims:
                shape = [var.dims]
            else:
                continue  # Skip unsupported types

            # Create exposure with obs_ prefix
            exposures.append(
                {
                    "id": f"obs_{var_id}",
                    "source_variable": var_id,
                    "exposed_to": ["agent"],
                    "shape": shape,
                }
            )

        return exposures

    def _load_observation_exposures(self, raw_configs: RawConfigs, symbol_table: UniverseSymbolTable) -> list[dict[str, Any]]:
        """Auto-generate observation exposures for ALL variables in symbol table.

        Design principle: All variables (standard + custom) are automatically observable.
        The exposed_observations field in variables_reference.yaml is deprecated/ignored.
        """
        # Auto-generate exposures for ALL variables (standard system + custom computed)
        exposures = self._auto_generate_standard_exposures(symbol_table)

        # Filter based on POMDP mode (only keep grid_encoding OR local_window, not both)
        if raw_configs.environment.partial_observability:
            # POMDP mode: use local window instead of full grid
            # Keep affordance_at_position for transfer learning (padded with zeros in environment)
            exposures = [obs for obs in exposures if obs.get("source_variable") != "grid_encoding"]
        else:
            # Full observability: use grid encoding instead of local window
            exposures = [obs for obs in exposures if obs.get("source_variable") != "local_window"]

        return exposures

    def _normalize_yaml(self, file_path: Path) -> str:
        try:
            with file_path.open() as handle:
                data = yaml.safe_load(handle) or {}
            return yaml.dump(data, sort_keys=True)
        except yaml.YAMLError as exc:
            # Transform raw YAML errors into friendly syntax errors
            error_msg = str(exc)
            if hasattr(exc, "problem_mark"):
                mark = exc.problem_mark
                error_msg = f"line {mark.line + 1}, column {mark.column + 1}: {getattr(exc, 'problem', None) or 'syntax error'}"
                if hasattr(exc, "context"):
                    error_msg = f"{exc.context}\n  {error_msg}"

            raise CompilationError(
                stage="Config Validation",
                errors=[
                    CompilationMessage(
                        code="YAML_SYNTAX_ERROR",
                        message=error_msg,
                        location=str(file_path),
                    )
                ],
                hints=[
                    "Check YAML indentation (use spaces, not tabs)",
                    "Ensure lists use proper '- item' syntax",
                    "Validate YAML syntax at yamllint.com or with 'yamllint <file>'",
                ],
            ) from exc

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

    def _compute_config_mtime(self, config_dir: Path) -> float:
        """Compute maximum modification time of all config files.

        Returns the latest mtime across all YAML files in the config directory
        and global_actions.yaml. This ensures cache is invalidated when ANY
        config file changes (including comment/whitespace-only changes).
        """
        yaml_files = sorted(config_dir.glob("*.yaml"))
        yaml_files.append(Path("configs") / "global_actions.yaml")

        max_mtime = 0.0
        for file_path in yaml_files:
            if not file_path.exists():
                continue
            mtime = file_path.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        return max_mtime

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
        """Compute full provenance ID including all dependencies.

        Note: This is for debugging/reproducibility only. Cache invalidation uses
        config_mtime + config_hash, NOT provenance_id, so dependency version changes
        don't trigger unnecessary recompilation. This is intentional - the compiler
        logic is version-stable, and dependency updates don't affect compiled output.
        """
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
            for entry in self._iter_entries(entries):
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

    def _normalize_affordance_position_metadata(self, position: Any) -> Any:
        if position is None:
            return None
        if isinstance(position, list):
            return tuple(position)
        if isinstance(position, dict):
            return dict(position)
        return position

    def _tensorize_affordance_position(self, position: Any, device: torch.device) -> torch.Tensor | None:
        if position is None:
            return None
        if isinstance(position, torch.Tensor):
            return position.to(device=device, dtype=torch.float32)

        if isinstance(position, dict):
            if set(position.keys()) == {"q", "r"}:
                coords = [position["q"], position["r"]]
            else:
                return None
            return torch.tensor(coords, dtype=torch.float32, device=device)

        if isinstance(position, list | tuple):
            return torch.tensor(list(position), dtype=torch.float32, device=device)

        if isinstance(position, Number):
            return torch.tensor([position], dtype=torch.float32, device=device)

        return None

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

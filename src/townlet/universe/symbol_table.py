"""Symbol table utilities for the Universe Compiler."""

from __future__ import annotations

from dataclasses import dataclass, field

from townlet.config.affordance import AffordanceConfig
from townlet.config.bar import BarConfig
from townlet.config.cascade import CascadeConfig
from townlet.config.cues import CompoundCueConfig, SimpleCueConfig
from townlet.environment.action_config import ActionConfig
from townlet.vfs.schema import VariableDef

from .errors import CompilationError


@dataclass
class UniverseSymbolTable:
    """Stores registered entities for cross-stage validation."""

    meters: dict[str, BarConfig] = field(default_factory=dict)
    cascades: dict[str, CascadeConfig] = field(default_factory=dict)
    affordances: dict[str, AffordanceConfig] = field(default_factory=dict)
    variables: dict[str, VariableDef] = field(default_factory=dict)
    actions: dict[int, ActionConfig] = field(default_factory=dict)
    cues: dict[str, SimpleCueConfig | CompoundCueConfig] = field(default_factory=dict)

    def register_meter(self, config: BarConfig) -> None:
        if config.name in self.meters:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate meter '{config.name}' detected."])
        self.meters[config.name] = config

    def register_variable(self, config: VariableDef) -> None:
        if config.id in self.variables:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate variable '{config.id}' detected."])
        self.variables[config.id] = config

    def register_action(self, config: ActionConfig) -> None:
        if config.id in self.actions:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate action id '{config.id}' detected."])
        self.actions[config.id] = config

    def register_cascade(self, config: CascadeConfig) -> None:
        if config.name in self.cascades:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate cascade '{config.name}' detected."])
        self.cascades[config.name] = config

    def register_affordance(self, config: AffordanceConfig) -> None:
        if config.id in self.affordances:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate affordance '{config.id}' detected."])
        self.affordances[config.id] = config

    def register_cue(self, cue: SimpleCueConfig | CompoundCueConfig) -> None:
        if cue.cue_id in self.cues:
            raise CompilationError("Stage 2: Symbol Table", [f"Duplicate cue '{cue.cue_id}' detected."])
        self.cues[cue.cue_id] = cue

    def get_meter(self, name: str) -> BarConfig:
        return self.meters[name]

    def get_meter_names(self) -> list[str]:
        return [meter.name for meter in sorted(self.meters.values(), key=lambda m: m.index)]

    def get_meter_count(self) -> int:
        return len(self.meters)

    def get_action(self, action_id: int) -> ActionConfig:
        return self.actions[action_id]

    def get_variable(self, variable_id: str) -> VariableDef:
        return self.variables[variable_id]

    def get_affordance_count(self) -> int:
        return len(self.affordances)

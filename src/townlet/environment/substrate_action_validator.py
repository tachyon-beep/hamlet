"""Validation helpers for substrate ↔ action compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field

from townlet.environment.action_config import ActionSpaceConfig
from townlet.substrate.config import SubstrateConfig


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SubstrateActionValidator:
    """Validates that the action space is compatible with the chosen substrate."""

    def __init__(self, substrate: SubstrateConfig, actions: ActionSpaceConfig) -> None:
        self._substrate = substrate
        self._actions = actions.actions

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self._substrate.type == "aspatial":
            self._validate_aspatial(result)
        elif self._substrate.type == "grid" and self._substrate.grid is not None:
            topology = self._substrate.grid.topology
            if topology == "square":
                self._validate_grid_square(result)
            elif topology == "cubic":
                self._validate_grid_cubic(result)
            elif topology == "hex":
                self._validate_hex_grid(result)
        # Additional substrate types (continuous, gridnd, etc.) currently do not impose
        # discrete action requirements. Future TASK-004 expansions can extend this validator.

        self._ensure_interact_action(result)
        result.valid = not result.errors
        return result

    def _movement_deltas(self) -> set[tuple[int, ...]]:
        deltas: set[tuple[int, ...]] = set()
        for action in self._actions:
            if action.type == "movement" and action.delta is not None:
                deltas.add(tuple(action.delta))
        return deltas

    def _validate_aspatial(self, result: ValidationResult) -> None:
        for action in self._actions:
            if action.type == "movement":
                result.errors.append("Aspatial substrate cannot define movement actions. Found movement action '" f"{action.name}'.")

    def _validate_grid_square(self, result: ValidationResult) -> None:
        required = {(0, -1), (0, 1), (-1, 0), (1, 0)}
        missing = required - self._movement_deltas()
        if missing:
            result.errors.append("Square grid requires 4-way movement (up/down/left/right). Missing deltas: " f"{sorted(missing)}")

    def _validate_grid_cubic(self, result: ValidationResult) -> None:
        required = {
            (0, -1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (0, 0, -1),
            (0, 0, 1),
        }
        missing = required - self._movement_deltas()
        if missing:
            result.errors.append("Cubic grid requires 6-way movement (horizontal + vertical). Missing deltas: " f"{sorted(missing)}")

    def _validate_hex_grid(self, result: ValidationResult) -> None:
        required = {
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (-1, 1),
        }
        missing = required - self._movement_deltas()
        if missing:
            result.errors.append("Hex grid requires 6 axial movements. Missing deltas: " f"{sorted(missing)}")

    def _ensure_interact_action(self, result: ValidationResult) -> None:
        if any(action.name.upper() == "INTERACT" for action in self._actions):
            return
        result.warnings.append(
            "Global action space does not define an INTERACT action. Some substrates expect "
            "agents to interact with affordances explicitly."
        )

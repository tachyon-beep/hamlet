"""Variable & Feature System (VFS) module.

The VFS provides a declarative way to define variables, observations, and
action effects for the Townlet environment.

Phase 1: Schema definitions, registry, observation specs
Phase 2: Derivation graphs, complex types, expression evaluation
"""

from townlet.vfs.registry import VariableRegistry
from townlet.vfs.schema import (
    NormalizationSpec,
    ObservationField,
    VariableDef,
    WriteSpec,
)

__all__ = [
    "NormalizationSpec",
    "ObservationField",
    "VariableDef",
    "VariableRegistry",
    "WriteSpec",
]

"""VFS schema definitions using Pydantic.

Variable & Feature System (VFS) schemas for defining variables, observations,
and action effects. These schemas enable declarative configuration of the
environment's state space.

Phase 1: Basic types and validation
Phase 2: Derivation graphs, complex types, expression parsing
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "NormalizationSpec",
    "WriteSpec",
    "ObservationField",
    "VariableDef",
]


class NormalizationSpec(BaseModel):
    """Observation normalization specification.

    Supports two normalization kinds:
    - minmax: Linear scaling to [min, max] range
    - zscore: Z-score normalization (value - mean) / std

    Examples:
        # Scalar minmax
        NormalizationSpec(kind="minmax", min=0.0, max=1.0)

        # Vector minmax
        NormalizationSpec(kind="minmax", min=[0.0, 0.0], max=[7.0, 7.0])

        # Scalar z-score
        NormalizationSpec(kind="zscore", mean=0.5, std=0.2)
    """

    kind: Literal["minmax", "zscore"] = Field(
        description="Normalization method: minmax or zscore"
    )

    # MinMax parameters (can be scalar or list)
    min: float | list[float] | None = Field(
        default=None,
        description="Minimum value(s) for minmax normalization",
    )
    max: float | list[float] | None = Field(
        default=None,
        description="Maximum value(s) for minmax normalization",
    )

    # Z-score parameters (can be scalar or list)
    mean: float | list[float] | None = Field(
        default=None,
        description="Mean value(s) for zscore normalization",
    )
    std: float | list[float] | None = Field(
        default=None,
        description="Standard deviation(s) for zscore normalization",
    )

    @model_validator(mode="after")
    def validate_normalization_params(self) -> "NormalizationSpec":
        """Validate that required parameters are present for each kind."""
        if self.kind == "minmax":
            if self.min is None:
                raise ValueError("minmax normalization requires 'min' parameter")
            if self.max is None:
                raise ValueError("minmax normalization requires 'max' parameter")
        elif self.kind == "zscore":
            if self.mean is None:
                raise ValueError("zscore normalization requires 'mean' parameter")
            if self.std is None:
                raise ValueError("zscore normalization requires 'std' parameter")
        return self


class WriteSpec(BaseModel):
    """Action write specification (variable update).

    Defines how an action modifies a variable's value.

    Phase 1: Expression is stored as string (no parsing)
    Phase 2: Expression will be parsed into AST for validation and execution

    Examples:
        # Simple constant
        WriteSpec(variable_id="energy", expression="-0.005")

        # Complex expression (Phase 2)
        WriteSpec(variable_id="money", expression="money + 10.0")
    """

    variable_id: str = Field(
        min_length=1,
        description="ID of the variable to write to",
    )

    expression: str = Field(
        min_length=1,
        description="Expression to evaluate (Phase 1: string, Phase 2: parsed AST)",
    )


class ObservationField(BaseModel):
    """Observation field specification.

    Maps a variable to an observation field that will be exposed to agents
    or other systems (like the BAC compiler).

    Examples:
        # Scalar observation
        ObservationField(
            id="obs_energy",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[],
        )

        # Vector observation with normalization
        ObservationField(
            id="obs_position",
            source_variable="position",
            exposed_to=["agent"],
            shape=[2],
            normalization=NormalizationSpec(kind="minmax", min=[0,0], max=[7,7]),
        )
    """

    id: str = Field(
        min_length=1,
        description="Unique identifier for this observation field",
    )

    source_variable: str = Field(
        min_length=1,
        description="ID of the variable this observation reads from",
    )

    exposed_to: list[str] = Field(
        min_length=1,
        description="Who can observe this field (e.g., ['agent', 'acs', 'bac'])",
    )

    shape: list[int] = Field(
        description="Shape of observation ([] for scalar, [N] for vector)",
    )

    normalization: NormalizationSpec | None = Field(
        default=None,
        description="Optional normalization to apply before exposing",
    )


class VariableDef(BaseModel):
    """Variable definition for VFS.

    Defines a single variable in the VFS state space with its type, scope,
    lifetime, and access control.

    Scope semantics:
    - global: Single value shared by all agents (e.g., time_of_day)
    - agent: Per-agent value, observable by all agents (e.g., energy, position)
    - agent_private: Per-agent value, observable only by owner (e.g., home_position)

    Type system (Phase 1):
    - scalar: Single float value
    - vec2i, vec3i: Fixed 2D/3D integer vectors
    - vecNi: N-dimensional integer vector (requires dims field)
    - vecNf: N-dimensional float vector (requires dims field)
    - bool: Boolean value

    Examples:
        # Scalar variable
        VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=1.0,
        )

        # Vector variable
        VariableDef(
            id="position",
            scope="agent",
            type="vecNf",
            dims=2,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0.0, 0.0],
        )
    """

    id: str = Field(
        min_length=1,
        description="Unique identifier for this variable",
    )

    scope: Literal["global", "agent", "agent_private"] = Field(
        description="Scope: global (shared), agent (per-agent public), agent_private (per-agent private)",
    )

    type: Literal["scalar", "vec2i", "vec3i", "vecNi", "vecNf", "bool"] = Field(
        description="Variable type (scalar, vector, or bool)",
    )

    dims: int | None = Field(
        default=None,
        ge=1,
        description="Number of dimensions for vecNi/vecNf types (required for those types)",
    )

    lifetime: Literal["tick", "episode"] = Field(
        description="Lifetime: tick (recomputed each step) or episode (persistent)",
    )

    readable_by: list[str] = Field(
        min_length=1,
        description="Who can read this variable (e.g., ['agent', 'engine', 'acs'])",
    )

    writable_by: list[str] = Field(
        min_length=1,
        description="Who can write this variable (e.g., ['engine', 'actions'])",
    )

    default: Any = Field(
        description="Default value (type depends on 'type' field)",
    )

    description: str | None = Field(
        default=None,
        description="Human-readable description of this variable",
    )

    @model_validator(mode="after")
    def validate_vector_types(self) -> "VariableDef":
        """Validate that vecNi/vecNf have dims field, scalar/bool do not."""
        if self.type in ("vecNi", "vecNf"):
            if self.dims is None:
                raise ValueError(
                    f"Variable '{self.id}' with type '{self.type}' requires 'dims' field"
                )
        elif self.type in ("scalar", "bool"):
            if self.dims is not None:
                raise ValueError(
                    f"Variable '{self.id}' with type '{self.type}' should not have 'dims' field"
                )
        # vec2i, vec3i have implicit dims (2, 3) - no dims field needed
        return self

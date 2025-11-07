"""VFS observation spec builder for generating observation specs from variables.

NOTE: This is NOT the same as environment.observation_builder.ObservationBuilder!
- environment.ObservationBuilder: Runtime observation construction (tensors)
- vfs.VFSObservationSpecBuilder: Compile-time spec generation (schemas for BAC)

The VFSObservationSpecBuilder generates observation specifications (schemas)
from variable definitions and exposure configurations. These specs are used by
the BAC (Behavioral Action Compiler) for dynamic network input head generation.
"""

from typing import Any

from townlet.vfs.schema import NormalizationSpec, ObservationField, VariableDef


class VFSObservationSpecBuilder:
    """Constructs observation specifications from variable definitions.

    Generates ObservationField specs that BAC compiler can use to
    build network input heads dynamically.

    This class generates SPECIFICATIONS (schemas), not runtime observations.
    For runtime observation construction, see environment.observation_builder.ObservationBuilder.
    """

    def build_observation_spec(
        self,
        variables: list[VariableDef],
        exposures: dict[str, dict[str, Any]],
    ) -> list[ObservationField]:
        """Build observation specification from variables and exposure config.

        Args:
            variables: List of variable definitions
            exposures: Dict mapping variable_id -> exposure config
                      e.g., {"energy": {"normalization": {...}}}

        Returns:
            List of ObservationField specs

        Raises:
            ValueError: If exposed variable not found in definitions

        Examples:
            >>> variables = [VariableDef(id="energy", scope="agent", type="scalar", ...)]
            >>> exposures = {"energy": {"normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}}}
            >>> spec = builder.build_observation_spec(variables, exposures)
            >>> len(spec)
            1
            >>> spec[0].source_variable
            'energy'
        """
        # Build variable lookup map
        var_map = {v.id: v for v in variables}
        obs_fields = []

        for var_id, exposure_config in exposures.items():
            if var_id not in var_map:
                raise ValueError(f"Variable {var_id} not found in definitions")

            var_def = var_map[var_id]

            # Infer shape from variable type
            shape = self._infer_shape(var_def)

            # Build normalization spec if provided
            norm_spec = None
            if exposure_config.get("normalization"):
                norm_config = exposure_config["normalization"]
                norm_spec = NormalizationSpec(**norm_config)

            # Create observation field
            field = ObservationField(
                id=f"obs_{var_id}",
                source_variable=var_id,
                exposed_to=["agent"],  # Default to agent for Phase 1
                shape=shape,
                normalization=norm_spec,
            )

            obs_fields.append(field)

        return obs_fields

    def _infer_shape(self, var_def: VariableDef) -> list[int]:
        """Infer observation shape from variable type.

        Args:
            var_def: Variable definition

        Returns:
            Shape as list (empty list for scalar)

        Raises:
            ValueError: If variable type is unknown

        Examples:
            >>> # Scalar variable
            >>> var = VariableDef(id="energy", type="scalar", ...)
            >>> builder._infer_shape(var)
            []

            >>> # Vec2i variable
            >>> var = VariableDef(id="position", type="vec2i", ...)
            >>> builder._infer_shape(var)
            [2]

            >>> # VecNf variable with dims=64
            >>> var = VariableDef(id="grid_encoding", type="vecNf", dims=64, ...)
            >>> builder._infer_shape(var)
            [64]
        """
        if var_def.type == "scalar":
            return []
        elif var_def.type == "bool":
            return []
        elif var_def.type == "vec2i":
            return [2]
        elif var_def.type == "vec3i":
            return [3]
        elif var_def.type in ["vecNi", "vecNf"]:
            if var_def.dims is None:
                raise ValueError(f"Variable {var_def.id} with type {var_def.type} must have dims field")
            return [var_def.dims]
        else:
            raise ValueError(f"Unknown variable type: {var_def.type}")

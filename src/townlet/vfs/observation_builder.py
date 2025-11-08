"""VFS observation spec builder for generating observation specs from variables.

NOTE: This is NOT the same as environment.observation_builder.ObservationBuilder!
- environment.ObservationBuilder: Runtime observation construction (tensors)
- vfs.VFSObservationSpecBuilder: Compile-time spec generation (schemas for BAC)

The VFSObservationSpecBuilder generates observation specifications (schemas)
from variable definitions and exposure configurations. These specs are used by
the BAC (Behavioral Action Compiler) for dynamic network input head generation.
"""

import math
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
        exposures: list[dict[str, Any]],
    ) -> list[ObservationField]:
        """Build observation specification from variables and exposure config.

        Args:
            variables: List of variable definitions
            exposures: Exposure entries describing which variables to expose.

        Returns:
            List of ObservationField specs

        Raises:
            ValueError: If exposed variable not found in definitions

        Examples:
            >>> variables = [VariableDef(id="energy", scope="agent", type="scalar", ...)]
            >>> exposures = [{"source_variable": "energy", "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}}]
            >>> spec = builder.build_observation_spec(variables, exposures)
            >>> len(spec)
            1
            >>> spec[0].source_variable
            'energy'
        """
        normalized_exposures = self._copy_exposures(exposures)

        # Build variable lookup map
        var_map = {v.id: v for v in variables}
        obs_fields = []

        for exposure_config in normalized_exposures:
            raw_var_id = exposure_config.get("source_variable")
            if not isinstance(raw_var_id, str) or not raw_var_id:
                raise ValueError("Exposure entry missing 'source_variable'")
            var_id = raw_var_id
            if var_id not in var_map:
                raise ValueError(f"Variable {var_id} not found in definitions")

            var_def = var_map[var_id]

            # Use provided metadata when present, otherwise infer defaults
            raw_field_id = exposure_config.get("id")
            if raw_field_id is None:
                field_id = f"obs_{var_id}"
            else:
                if not isinstance(raw_field_id, str):
                    raise ValueError(f"Exposure field id for '{var_id}' must be a string")
                field_id = raw_field_id
            exposed_to = exposure_config.get("exposed_to") or ["agent"]

            shape_config = exposure_config.get("shape")
            shape = shape_config if shape_config is not None else self._infer_shape(var_def)

            # Build normalization spec if provided
            norm_spec = self._build_normalization_spec(exposure_config.get("normalization"))

            # Create observation field
            field = ObservationField(
                id=field_id,
                source_variable=var_id,
                exposed_to=exposed_to,
                shape=shape,
                normalization=norm_spec,
            )

            self._validate_normalization_shape(field_id, shape, norm_spec)
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

    def _copy_exposures(self, exposures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create shallow copies of exposure definitions and validate input type."""
        if not isinstance(exposures, list):
            raise TypeError("exposures must be provided as a list of dictionaries")

        normalized: list[dict[str, Any]] = []
        for idx, exposure in enumerate(exposures):
            if not isinstance(exposure, dict):
                raise TypeError(f"Exposure entry at index {idx} must be a dict, got {type(exposure).__name__}")
            normalized.append(dict(exposure))
        return normalized

    def _build_normalization_spec(self, norm_config: dict[str, Any] | None) -> NormalizationSpec | None:
        """Build a NormalizationSpec from config dict, if provided."""
        if not norm_config:
            return None
        return NormalizationSpec(**norm_config)

    def _validate_normalization_shape(
        self,
        field_id: str,
        shape: list[int],
        norm_spec: NormalizationSpec | None,
    ) -> None:
        """Ensure normalization parameters align with observation shape."""
        if norm_spec is None:
            return

        dims = self._shape_volume(shape)

        def _validate_param(label: str, values: float | list[float] | None) -> None:
            if values is None:
                return
            if isinstance(values, list):
                if dims == 1 and len(values) == 1:
                    return
                if len(values) != dims:
                    raise ValueError(
                        f"Normalization '{label}' for observation '{field_id}' must provide {dims} values "
                        f"to match shape {shape}, got {len(values)}"
                    )
            else:
                if dims > 1:
                    raise ValueError(
                        f"Normalization '{label}' for observation '{field_id}' must be a list of length {dims} "
                        f"to match shape {shape}, not a scalar"
                    )

        _validate_param("min", norm_spec.min)
        _validate_param("max", norm_spec.max)
        _validate_param("mean", norm_spec.mean)
        _validate_param("std", norm_spec.std)

    @staticmethod
    def _shape_volume(shape: list[int]) -> int:
        """Return flattened size for an observation shape."""
        if not shape:
            return 1
        return math.prod(shape)

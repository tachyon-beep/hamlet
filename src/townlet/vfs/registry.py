"""Variable registry for VFS runtime storage.

The VariableRegistry manages runtime storage of VFS variables with access control
and scope semantics. It handles three scope patterns:

- global: Single value shared by all agents (shape [] or [dims])
- agent: Per-agent values, observable by all (shape [num_agents] or [num_agents, dims])
- agent_private: Per-agent values, observable only by owner (shape [num_agents] or [num_agents, dims])
"""

import torch

from townlet.vfs.schema import VariableDef


class VariableRegistry:
    """Runtime storage for VFS variables with access control.

    Examples:
        # Create registry with variables
        registry = VariableRegistry(
            variables=[
                VariableDef(id="energy", scope="agent", type="scalar", ...),
                VariableDef(id="position", scope="agent", type="vecNf", dims=2, ...),
            ],
            num_agents=4,
            device=torch.device("cpu"),
        )

        # Read variable (with access control)
        energy = registry.get("energy", reader="agent")

        # Write variable (with access control)
        registry.set("energy", new_values, writer="engine")
    """

    def __init__(
        self,
        variables: list[VariableDef],
        num_agents: int,
        device: torch.device,
    ):
        """Initialize variable registry.

        Args:
            variables: List of variable definitions
            num_agents: Number of agents in the environment
            device: PyTorch device (cpu or cuda)
        """
        self.num_agents = num_agents
        self.device = device

        # Store variable definitions by ID, guarding against duplicate IDs
        self._definitions: dict[str, VariableDef] = {}
        for var in variables:
            if var.id in self._definitions:
                raise ValueError(f"Duplicate variable id '{var.id}' in registry initialization")
            self._definitions[var.id] = var

        # Initialize storage tensors
        self._storage: dict[str, torch.Tensor] = {}
        self._expected_shapes: dict[str, torch.Size] = {}
        self._expected_dtypes: dict[str, torch.dtype] = {}
        self._initialize_storage()

    @property
    def variables(self) -> dict[str, VariableDef]:
        """Get variable definitions dictionary.

        Returns:
            Dictionary mapping variable IDs to their definitions.

        Examples:
            # Check if variable exists
            if "energy" in registry.variables:
                var_def = registry.variables["energy"]
                print(f"Energy type: {var_def.type}")

            # Iterate over all variables
            for var_id, var_def in registry.variables.items():
                print(f"{var_id}: {var_def.scope}")
        """
        return self._definitions

    def _initialize_storage(self) -> None:
        """Initialize storage tensors with default values for all variables."""
        for var_id, var_def in self._definitions.items():
            # Determine tensor shape based on scope and type
            self._compute_shape(var_def)

            # Initialize tensor with default value
            if var_def.type == "scalar":
                # Scalar: default is a single float
                if var_def.scope == "global":
                    # Global scalar: shape []
                    tensor = torch.tensor(var_def.default, device=self.device, dtype=torch.float32)
                else:
                    # Agent/agent_private scalar: shape [num_agents]
                    tensor = torch.full(
                        (self.num_agents,),
                        var_def.default,
                        device=self.device,
                        dtype=torch.float32,
                    )
            elif var_def.type in ("vecNi", "vecNf", "vec2i", "vec3i"):
                base_default = self._build_vector_default(var_def)

                if var_def.scope == "global":
                    tensor = base_default.clone()
                else:
                    tensor = base_default.unsqueeze(0).expand(self.num_agents, -1).clone()
            elif var_def.type == "bool":
                # Bool: default is a boolean
                if var_def.scope == "global":
                    # Global bool: shape []
                    tensor = torch.tensor(var_def.default, device=self.device, dtype=torch.bool)
                else:
                    # Agent/agent_private bool: shape [num_agents]
                    tensor = torch.full(
                        (self.num_agents,),
                        var_def.default,
                        device=self.device,
                        dtype=torch.bool,
                    )
            else:
                raise ValueError(f"Unsupported variable type: {var_def.type}")

            self._storage[var_id] = tensor
            self._expected_shapes[var_id] = tensor.shape
            self._expected_dtypes[var_id] = tensor.dtype

    def _compute_shape(self, var_def: VariableDef) -> tuple[int, ...]:
        """Compute tensor shape for a variable definition.

        Args:
            var_def: Variable definition

        Returns:
            Tuple representing tensor shape

        Examples:
            global scalar: ()
            global vector (dims=2): (2,)
            agent scalar: (num_agents,)
            agent vector (dims=2): (num_agents, 2)
        """
        if var_def.type == "scalar" or var_def.type == "bool":
            if var_def.scope == "global":
                return ()  # Shape []
            else:
                return (self.num_agents,)  # Shape [num_agents]

        elif var_def.type == "vec2i":
            dims = 2
        elif var_def.type == "vec3i":
            dims = 3
        elif var_def.type in ("vecNi", "vecNf"):
            if var_def.dims is None:
                raise ValueError(f"vecNi/vecNf variable '{var_def.id}' must have dims field defined")
            dims = var_def.dims
        else:
            raise ValueError(f"Unsupported variable type: {var_def.type}")

        # Vector variable
        if var_def.scope == "global":
            return (dims,)  # Shape [dims]
        else:
            return (self.num_agents, dims)  # Shape [num_agents, dims]

    def get(self, variable_id: str, reader: str) -> torch.Tensor:
        """Get variable value with access control.

        Args:
            variable_id: ID of the variable to read
            reader: Who is reading (e.g., "agent", "engine", "acs")

        Returns:
            Tensor containing the variable value

        Raises:
            KeyError: If variable_id doesn't exist
            PermissionError: If reader not allowed to read this variable

        Examples:
            # Read energy (agent scope)
            energy = registry.get("energy", reader="agent")
            # Returns: tensor([1.0, 1.0, 1.0, 1.0])  # shape [num_agents]

            # Read global time_sin
            time_sin = registry.get("time_sin", reader="agent")
            # Returns: tensor(0.0)  # shape []
        """
        if variable_id not in self._definitions:
            raise KeyError(f"Variable '{variable_id}' not found in registry")

        var_def = self._definitions[variable_id]

        # Check read permission
        if reader not in var_def.readable_by:
            raise PermissionError(f"'{reader}' is not allowed to read variable '{variable_id}'. " f"Readable by: {var_def.readable_by}")

        value = self._storage[variable_id]

        if var_def.scope == "agent_private" and reader == "agent":
            raise PermissionError(
                f"'{reader}' is not allowed to read agent_private variable '{variable_id}'. "
                "Only privileged readers (engine, acs, etc.) may access raw values."
            )

        return value.clone()

    def set(self, variable_id: str, value: torch.Tensor, writer: str) -> None:
        """Set variable value with access control.

        Args:
            variable_id: ID of the variable to write
            value: New tensor value
            writer: Who is writing (e.g., "engine", "actions")

        Raises:
            KeyError: If variable_id doesn't exist
            PermissionError: If writer not allowed to write this variable

        Examples:
            # Update energy for all agents
            new_energy = torch.tensor([0.9, 0.8, 0.7, 0.6])
            registry.set("energy", new_energy, writer="engine")

            # Update global time_sin
            registry.set("time_sin", torch.tensor(0.707), writer="engine")
        """
        if variable_id not in self._definitions:
            raise KeyError(f"Variable '{variable_id}' not found in registry")

        var_def = self._definitions[variable_id]

        # Check write permission
        if writer not in var_def.writable_by:
            raise PermissionError(f"'{writer}' is not allowed to write variable '{variable_id}'. " f"Writable by: {var_def.writable_by}")

        expected_shape = self._expected_shapes[variable_id]
        expected_dtype = self._expected_dtypes[variable_id]

        if value.shape != expected_shape:
            raise ValueError(f"Value for '{variable_id}' has shape {tuple(value.shape)}, expected {tuple(expected_shape)}")

        if value.dtype != expected_dtype:
            raise ValueError(f"Value for '{variable_id}' has dtype {value.dtype}, expected {expected_dtype}")

        # Update storage (defensive copy to avoid aliasing)
        self._storage[variable_id] = value.to(self.device).clone()

    def _get_vector_dims(self, var_def: VariableDef) -> int:
        """Return expected dimensionality for vector variables."""
        if var_def.type == "vec2i":
            return 2
        if var_def.type == "vec3i":
            return 3
        if var_def.type in ("vecNi", "vecNf"):
            if var_def.dims is None:
                raise ValueError(f"Variable '{var_def.id}' missing 'dims' for type '{var_def.type}'")
            return var_def.dims
        default_list = var_def.default
        if isinstance(default_list, list):
            return len(default_list)
        raise ValueError(f"Variable '{var_def.id}' must provide default list for type '{var_def.type}'")

    def _build_vector_default(self, var_def: VariableDef) -> torch.Tensor:
        """Build default tensor for vector variables, padding if necessary."""
        dims = self._get_vector_dims(var_def)
        dtype = torch.long if var_def.type in ("vecNi", "vec2i", "vec3i") else torch.float32
        tensor = torch.zeros(dims, device=self.device, dtype=dtype)

        default_values = var_def.default
        if isinstance(default_values, list) and default_values:
            copy_len = min(len(default_values), dims)
            default_tensor = torch.tensor(default_values[:copy_len], device=self.device, dtype=dtype)
            tensor[:copy_len] = default_tensor
        return tensor

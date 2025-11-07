"""Variable registry for VFS runtime storage.

The VariableRegistry manages runtime storage of VFS variables with access control
and scope semantics. It handles three scope patterns:

- global: Single value shared by all agents (shape [] or [dims])
- agent: Per-agent values, observable by all (shape [num_agents] or [num_agents, dims])
- agent_private: Per-agent values, observable only by owner (shape [num_agents] or [num_agents, dims])
"""

import torch
from typing import Any

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

        # Store variable definitions by ID
        self._definitions: dict[str, VariableDef] = {var.id: var for var in variables}

        # Initialize storage tensors
        self._storage: dict[str, torch.Tensor] = {}
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
            shape = self._compute_shape(var_def)

            # Initialize tensor with default value
            if var_def.type == "scalar":
                # Scalar: default is a single float
                if var_def.scope == "global":
                    # Global scalar: shape []
                    tensor = torch.tensor(var_def.default, device=self.device)
                else:
                    # Agent/agent_private scalar: shape [num_agents]
                    tensor = torch.full(
                        (self.num_agents,),
                        var_def.default,
                        device=self.device,
                        dtype=torch.float32,
                    )
            elif var_def.type in ("vecNi", "vecNf", "vec2i", "vec3i"):
                # Vector: default is a list
                default_list = var_def.default

                if var_def.scope == "global":
                    # Global vector: shape [dims]
                    tensor = torch.tensor(default_list, device=self.device)
                    if var_def.type in ("vecNi", "vec2i", "vec3i"):
                        tensor = tensor.long()
                    else:
                        tensor = tensor.float()
                else:
                    # Agent/agent_private vector: shape [num_agents, dims]
                    # Create [num_agents, dims] tensor filled with default
                    dims = len(default_list)
                    tensor = torch.zeros(
                        (self.num_agents, dims),
                        device=self.device,
                        dtype=torch.long if var_def.type in ("vecNi", "vec2i", "vec3i") else torch.float32,
                    )
                    # Fill with default values
                    for i, val in enumerate(default_list):
                        tensor[:, i] = val
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
            raise PermissionError(
                f"'{reader}' is not allowed to read variable '{variable_id}'. "
                f"Readable by: {var_def.readable_by}"
            )

        return self._storage[variable_id]

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
            raise PermissionError(
                f"'{writer}' is not allowed to write variable '{variable_id}'. "
                f"Writable by: {var_def.writable_by}"
            )

        # Update storage
        self._storage[variable_id] = value.to(self.device)

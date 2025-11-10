"""Environment and substrate fixtures."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.test_townlet._fixtures.config import _apply_config_overrides
from tests.test_townlet.helpers.config_builder import mutate_training_yaml
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.universe.compiled import CompiledUniverse

# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def basic_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    test_config_pack_path: Path,
    device: torch.device,
) -> VectorizedHamletEnv:
    """Create a basic environment with default test config.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance
    """
    universe = compile_universe(test_config_pack_path)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def pomdp_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Create a POMDP environment with partial observability.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Partial observability (5×5 vision)
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with POMDP
    """
    universe = compile_universe(Path("configs/L2_partial_observability"))
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def temporal_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Create an environment with temporal mechanics enabled.

    Configuration:
        - 1 agent
        - 8×8 grid
        - Full observability
        - Temporal mechanics (24-hour cycle)
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with temporal mechanics
    """
    universe = compile_universe(Path("configs/L3_temporal_mechanics"))
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def multi_agent_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    test_config_pack_path: Path,
    device: torch.device,
) -> VectorizedHamletEnv:
    """Create an environment with multiple agents.

    Configuration:
        - 4 agents (for batching tests)
        - 8×8 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Returns:
        VectorizedHamletEnv instance with 4 agents
    """
    universe = compile_universe(test_config_pack_path)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=4,
        device=device,
    )


@pytest.fixture
def env_factory(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    test_config_pack_path: Path,
    device: torch.device,
):
    """Return a helper for building environments from compiled universes.

    Most tests only need to vary the config pack or agent count.  This
    factory centralizes the compile → from_universe pipeline so individual
    tests do not reach for the legacy constructor.
    """

    def _build_env(
        *,
        config_dir: Path | str | None = None,
        universe: CompiledUniverse | None = None,
        num_agents: int = 1,
        device_override: torch.device | str | None = None,
    ) -> VectorizedHamletEnv:
        target_universe = universe
        if target_universe is None:
            pack_path = Path(config_dir) if config_dir is not None else test_config_pack_path
            target_universe = compile_universe(pack_path)

        target_device = device_override if device_override is not None else device

        return VectorizedHamletEnv.from_universe(
            target_universe,
            num_agents=num_agents,
            device=target_device,
        )

    return _build_env


@pytest.fixture
def cpu_env_factory(env_factory, cpu_device):
    """Convenience factory that always targets the CPU device."""

    def _build_env(**kwargs):
        return env_factory(device_override=cpu_device, **kwargs)

    return _build_env


@pytest.fixture
def custom_env_builder(
    tmp_path: Path,
    test_config_pack_path: Path,
    env_factory,
    cpu_device,
):
    """Factory to build environments with optional training overrides."""

    def _build_env(
        *,
        num_agents: int = 1,
        overrides: dict[str, Any] | None = None,
        source_pack: Path | str | None = None,
    ):
        source_path = Path(source_pack) if source_pack is not None else test_config_pack_path
        target_dir = tmp_path / f"config_pack_{uuid.uuid4().hex}"
        shutil.copytree(source_path, target_dir)

        if overrides:
            mutate_training_yaml(target_dir, lambda data: _apply_config_overrides(data, overrides))

        return env_factory(
            config_dir=target_dir,
            num_agents=num_agents,
            device_override=cpu_device,
        )

    return _build_env


# =============================================================================
# TASK-002A: SUBSTRATE-PARAMETERIZED FIXTURES
# =============================================================================


@pytest.fixture
def grid2d_3x3_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Small 3×3 Grid2D environment for fast tests.

    Configuration:
        - 1 agent
        - 3×3 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Use for: Fast unit tests that don't need full 8×8 grid

    Returns:
        VectorizedHamletEnv with 3×3 Grid2D substrate
    """
    universe = compile_universe(Path("configs/L0_0_minimal"))
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def grid2d_8x8_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    test_config_pack_path: Path,
    device: torch.device,
) -> VectorizedHamletEnv:
    """Standard 8×8 Grid2D environment (same as basic_env, explicit name).

    Configuration:
        - 1 agent
        - 8×8 grid
        - Full observability
        - No temporal mechanics
        - Device: CUDA if available, else CPU

    Use for: Tests requiring standard grid size (legacy compatibility)

    Returns:
        VectorizedHamletEnv with 8×8 Grid2D substrate
    """
    universe = compile_universe(test_config_pack_path)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def aspatial_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Aspatial environment (no grid, meters only).

    Configuration:
        - 1 agent
        - No spatial substrate (aspatial)
        - 4 meters: energy, health, money, mood
        - 4 affordances: Bed, Hospital, HomeMeal, Job
        - Device: CUDA if available, else CPU

    Use for: Testing aspatial substrate behavior (no positions, no movement)

    Returns:
        VectorizedHamletEnv with Aspatial substrate
    """
    # Use aspatial config pack created in Task 8.1
    repo_root = Path(__file__).parent.parent.parent.parent
    aspatial_config_path = repo_root / "configs" / "aspatial_test"

    universe = compile_universe(aspatial_config_path)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def continuous1d_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Continuous 1D environment for movement/action-mask tests."""

    universe = compile_universe(Path("configs/L1_continuous_1D"))
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


@pytest.fixture
def continuous3d_env(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    device: torch.device,
) -> VectorizedHamletEnv:
    """Continuous 3D environment for movement/action-mask tests."""

    universe = compile_universe(Path("configs/L1_continuous_3D"))
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


# Parameterization helper for multi-substrate tests
SUBSTRATE_FIXTURES = ["grid2d_3x3_env", "grid2d_8x8_env", "aspatial_env"]

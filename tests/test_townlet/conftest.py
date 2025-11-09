"""Shared test fixtures for Townlet test suite.

This module provides common fixtures used across unit, integration, and e2e tests.
Fixtures are organized by component area for easy discovery.

Usage:
    from pathlib import Path

    def test_something(mock_config_path, basic_env):
        # Use fixtures directly in test functions
        assert basic_env.grid_size == 8
"""

import copy
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.replay_buffer import ReplayBuffer
from townlet.universe.compiled import CompiledUniverse
from townlet.universe.compiler import UniverseCompiler

# Default observation dimensionality for the standard 8×8 full-observability setup:
#   64 grid cells + 2 normalized position features + 8 meters + 15 affordance slots + 4 temporal
FULL_OBS_DIM_8X8 = 93

# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def mock_config_path() -> Path:
    """Return path to frozen mock configuration for exact-value assertions.

    This configuration is LOCKED and should not be modified. Tests that
    assert exact values MUST use this config to ensure stability.

    Returns:
        Path to fixtures/mock_config.yaml
    """
    return Path(__file__).parent / "fixtures" / "mock_config.yaml"


@pytest.fixture(scope="session")
def test_config_pack_path() -> Path:
    """Return path to test configuration pack (configs/test).

    This is a lightweight config pack for integration testing.
    Unlike mock_config, this can evolve with the codebase.

    Returns:
        Path to configs/test directory
    """
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "configs" / "test"


@pytest.fixture
def temp_config_pack(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create a temporary writable copy of the test config pack.

    Use this fixture when you need to modify config files during tests.

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary config pack directory
    """
    target_pack = tmp_path / "config_pack"
    shutil.copytree(test_config_pack_path, target_pack)
    return target_pack


@pytest.fixture
def mock_config(mock_config_path: Path) -> dict[str, Any]:
    """Load mock configuration as a dictionary.

    Returns:
        Dict containing frozen mock configuration
    """
    with open(mock_config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# DEVICE FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Returns:
        torch.device: Preferred device for tests
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for tests that need deterministic behavior.

    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def compile_universe() -> Callable[[Path | str], CompiledUniverse]:
    """Return a compiler helper that caches compiled reference packs."""

    compiler = UniverseCompiler()
    cache: dict[Path, CompiledUniverse] = {}
    repo_configs = (Path(__file__).parent.parent.parent / "configs").resolve()

    def _compile(config_dir: Path | str) -> CompiledUniverse:
        target_path = Path(config_dir).resolve()
        cache_key: Path | None = None
        try:
            target_path.relative_to(repo_configs)
            cache_key = target_path
        except ValueError:
            cache_key = None

        if cache_key is not None and cache_key in cache:
            return cache[cache_key]

        compiled = compiler.compile(target_path)
        if cache_key is not None:
            cache[cache_key] = compiled
        return compiled

    return _compile


# =============================================================================
# TEMPFILE FIXTURES
# =============================================================================


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for test files.

    This fixture eliminates the need for repetitive tempfile.TemporaryDirectory()
    context managers throughout the test suite.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to temporary directory (cleaned up after test)

    Usage:
        def test_something(temp_test_dir):
            config_path = temp_test_dir / "test.yaml"
            # Test logic
    """
    return tmp_path


@pytest.fixture
def temp_yaml_file(temp_test_dir: Path) -> Path:
    """Provide temporary YAML file path.

    Common pattern for config tests that need to write YAML files.

    Returns:
        Path to temporary YAML file

    Usage:
        def test_yaml_loading(temp_yaml_file):
            with open(temp_yaml_file, 'w') as f:
                yaml.dump(config_data, f)
            # Test loading
    """
    return temp_test_dir / "test.yaml"


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
    repo_root = Path(__file__).parent.parent.parent
    aspatial_config_path = repo_root / "configs" / "aspatial_test"

    universe = compile_universe(aspatial_config_path)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=device,
    )


# Parameterization helper for multi-substrate tests
SUBSTRATE_FIXTURES = ["grid2d_3x3_env", "grid2d_8x8_env", "aspatial_env"]


# =============================================================================
# NETWORK FIXTURES
# =============================================================================


@pytest.fixture
def simple_qnetwork(basic_env: VectorizedHamletEnv, device: torch.device) -> SimpleQNetwork:
    """Create a SimpleQNetwork for full observability.

    Args:
        basic_env: Environment to get observation dimension
        device: Device to place network on

    Returns:
        SimpleQNetwork instance
    """
    obs_dim = basic_env.observation_dim  # VFS: Use observation_dim directly (observation_builder removed)
    return SimpleQNetwork(obs_dim=obs_dim, action_dim=basic_env.action_dim, hidden_dim=128).to(device)


@pytest.fixture
def recurrent_qnetwork(pomdp_env: VectorizedHamletEnv, device: torch.device) -> RecurrentSpatialQNetwork:
    """Create a RecurrentSpatialQNetwork for POMDP.

    Args:
        pomdp_env: POMDP environment to get observation dimension
        device: Device to place network on

    Returns:
        RecurrentSpatialQNetwork instance
    """
    return RecurrentSpatialQNetwork(
        action_dim=pomdp_env.action_dim,
        window_size=5,  # 5×5 local vision
        num_meters=pomdp_env.meter_count,  # Dynamic meter count (TASK-001)
        num_affordance_types=14,
        enable_temporal_features=False,
        hidden_dim=256,
    ).to(device)


# =============================================================================
# TRAINING COMPONENT FIXTURES
# =============================================================================


@pytest.fixture
def replay_buffer(device: torch.device) -> ReplayBuffer:
    """Create a basic replay buffer.

    Configuration:
        - Capacity: 1000
        - Observation dimension: FULL_OBS_DIM_8X8 (standard 8×8 full observability)
        - Device: CUDA if available, else CPU

    Returns:
        ReplayBuffer instance
    """
    return ReplayBuffer(capacity=1000, obs_dim=FULL_OBS_DIM_8X8, device=device)


@pytest.fixture
def adversarial_curriculum() -> AdversarialCurriculum:
    """Create an adversarial curriculum with test parameters.

    Configuration:
        - Max steps: 200 (short for tests)
        - Survival advance: 70%
        - Survival retreat: 30%
        - Entropy gate: 0.5
        - Min steps at stage: 50

    Returns:
        AdversarialCurriculum instance
    """
    return AdversarialCurriculum(
        max_steps_per_episode=200,
        survival_advance_threshold=0.7,
        survival_retreat_threshold=0.3,
        entropy_gate=0.5,
        min_steps_at_stage=50,
    )


@pytest.fixture
def static_curriculum() -> StaticCurriculum:
    """Create a static curriculum with fixed difficulty.

    Configuration:
        - Max steps: 200

    Returns:
        StaticCurriculum instance
    """
    return StaticCurriculum(max_steps_per_episode=200)


@pytest.fixture
def epsilon_greedy_exploration(device: torch.device) -> EpsilonGreedyExploration:
    """Create an epsilon-greedy exploration strategy.

    Configuration:
        - Epsilon: 1.0
        - Epsilon min: 0.1
        - Epsilon decay: 0.99

    Returns:
        EpsilonGreedyExploration instance
    """
    return EpsilonGreedyExploration(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
    )


@pytest.fixture
def adaptive_intrinsic_exploration(basic_env: VectorizedHamletEnv, device: torch.device) -> AdaptiveIntrinsicExploration:
    """Create an adaptive intrinsic exploration strategy with RND.

    Configuration:
        - Embed dimension: 128
        - Initial intrinsic weight: 1.0
        - Variance threshold: 100.0
        - Survival window: 50

    Returns:
        AdaptiveIntrinsicExploration instance
    """
    obs_dim = basic_env.observation_dim  # VFS: Use observation_dim directly (observation_builder removed)
    return AdaptiveIntrinsicExploration(
        obs_dim=obs_dim,
        embed_dim=128,
        initial_intrinsic_weight=1.0,
        variance_threshold=100.0,
        survival_window=50,
        device=device,
    )


@pytest.fixture
def vectorized_population(
    basic_env: VectorizedHamletEnv,
    adversarial_curriculum: AdversarialCurriculum,
    epsilon_greedy_exploration: EpsilonGreedyExploration,
    device: torch.device,
) -> VectorizedPopulation:
    """Create a vectorized population for training.

    Configuration:
        - Network type: Simple (MLP)
        - Learning rate: 0.00025
        - Gamma: 0.99
        - Replay buffer capacity: 1000
        - Batch size: 32

    Returns:
        VectorizedPopulation instance
    """
    return VectorizedPopulation(
        env=basic_env,
        curriculum=adversarial_curriculum,
        exploration=epsilon_greedy_exploration,
        network_type="simple",
        learning_rate=0.00025,
        gamma=0.99,
        replay_buffer_capacity=1000,
        batch_size=32,
        device=device,
    )


@pytest.fixture
def non_training_recurrent_population(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
) -> VectorizedPopulation:
    """Create recurrent population with training DISABLED for unit tests.

    This fixture is designed for tests that focus on:
    - Hidden state management (persistence, reset, shape)
    - Episode flushing behavior
    - LSTM forward pass mechanics
    - Observation processing

    NOT for tests that verify:
    - Training convergence
    - Q-network weight updates
    - Replay buffer sampling during training

    Configuration:
        - Network type: Recurrent (LSTM)
        - POMDP: 5×5 vision window
        - Training: DISABLED (train_frequency=10000)
        - Sequence length: 8 (production default)
        - Batch size: 8 (production default)
        - Device: CPU (for deterministic behavior)

    Returns:
        VectorizedPopulation instance with training disabled
    """
    pomdp_universe = compile_universe(Path("configs/L2_partial_observability"))
    env = VectorizedHamletEnv.from_universe(
        pomdp_universe,
        num_agents=1,
        device=cpu_device,
    )

    curriculum = StaticCurriculum()
    exploration = EpsilonGreedyExploration(
        epsilon=0.1,
        epsilon_min=0.1,
        epsilon_decay=1.0,
        device=cpu_device,
    )

    return VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=["agent_0"],
        device=cpu_device,
        # action_dim defaults to env.action_dim
        network_type="recurrent",
        vision_window_size=5,
        train_frequency=10000,  # DISABLED: Prevents unintended training in tests
        sequence_length=8,  # Production default
        batch_size=8,  # Production default
    )


# =============================================================================
# TASK-001: VARIABLE METER CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def task001_config_4meter(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create temporary 4-meter config pack for TASK-001 testing.

    Meters: energy, health, money, mood
    Use ONLY for: TASK-001 variable meter tests
    Do NOT use for: L0 curriculum (use separate curriculum fixtures)

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary 4-meter config pack directory
    """
    config_4m = tmp_path / "config_4m"
    shutil.copytree(test_config_pack_path, config_4m)

    # Create 4-meter bars.yaml
    bars_config = {
        "version": "2.0",
        "description": "4-meter test universe",
        "bars": [
            {
                "name": "energy",
                "index": 0,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.005,
                "description": "Energy level",
            },
            {
                "name": "health",
                "index": 1,
                "tier": "pivotal",
                "range": [0.0, 1.0],
                "initial": 1.0,
                "base_depletion": 0.0,
                "description": "Health status",
            },
            {
                "name": "money",
                "index": 2,
                "tier": "resource",
                "range": [0.0, 1.0],
                "initial": 0.5,
                "base_depletion": 0.0,
                "description": "Financial resources",
            },
            {
                "name": "mood",
                "index": 3,
                "tier": "secondary",
                "range": [0.0, 1.0],
                "initial": 0.7,
                "base_depletion": 0.001,
                "description": "Mood state",
            },
        ],
        "terminal_conditions": [
            {"meter": "energy", "operator": "<=", "value": 0.0, "description": "Death by energy depletion"},
            {"meter": "health", "operator": "<=", "value": 0.0, "description": "Death by health failure"},
        ],
    }

    with open(config_4m / "bars.yaml", "w") as f:
        yaml.safe_dump(bars_config, f)

    # Simplify cascades.yaml
    cascades_config = {
        "version": "2.0",
        "description": "Simplified cascades for 4-meter testing",
        "math_type": "gradient_penalty",
        "modulations": [],
        "cascades": [
            {
                "name": "low_mood_hits_energy",
                "category": "secondary_to_pivotal",
                "description": "Low mood drains energy",
                "source": "mood",
                "source_index": 3,
                "target": "energy",
                "target_index": 0,
                "threshold": 0.2,
                "strength": 0.01,
            }
        ],
        "execution_order": ["secondary_to_pivotal"],
    }

    with open(config_4m / "cascades.yaml", "w") as f:
        yaml.safe_dump(cascades_config, f)

    # Create affordances.yaml with FULL 14-affordance vocabulary but only using 4 meters
    # This maintains observation vocabulary consistency while validating meter references
    # Note: Only enabled_affordances (Bed, Hospital, HomeMeal=FastFood, Job) will be deployed
    affordances_config = {
        "version": "2.0",
        "description": "4-meter test affordances (full vocabulary, 4-meter compatible)",
        "status": "TEST",
        "affordances": [
            {
                "id": "0",
                "name": "Bed",
                "category": "energy",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.05}],
                "effects": [{"meter": "energy", "amount": 0.50}, {"meter": "health", "amount": 0.02}],
                "operating_hours": [0, 24],
            },
            {
                "id": "1",
                "name": "LuxuryBed",
                "category": "energy",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.11}],
                "effects": [{"meter": "energy", "amount": 0.75}, {"meter": "health", "amount": 0.05}],
                "operating_hours": [0, 24],
            },
            {
                "id": "2",
                "name": "Shower",
                "category": "hygiene",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.03}],
                "effects": [{"meter": "mood", "amount": 0.20}],  # Use mood instead of hygiene
                "operating_hours": [0, 24],
            },
            {
                "id": "3",
                "name": "HomeMeal",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.04}],
                "effects": [{"meter": "energy", "amount": 0.20}, {"meter": "mood", "amount": 0.10}],
                "operating_hours": [0, 24],
            },
            {
                "id": "4",
                "name": "FastFood",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.06}],
                "effects": [{"meter": "energy", "amount": 0.30}],
                "operating_hours": [0, 24],
            },
            {
                "id": "5",
                "name": "Restaurant",
                "category": "food",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.12}],
                "effects": [{"meter": "energy", "amount": 0.40}, {"meter": "mood", "amount": 0.15}],
                "operating_hours": [11, 22],
            },
            {
                "id": "6",
                "name": "Gym",
                "category": "fitness",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.08}, {"meter": "energy", "amount": 0.10}],
                "effects": [{"meter": "health", "amount": 0.15}, {"meter": "mood", "amount": 0.05}],
                "operating_hours": [6, 22],
            },
            {
                "id": "7",
                "name": "Hospital",
                "category": "health",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.15}],
                "effects": [{"meter": "health", "amount": 0.60}],
                "operating_hours": [0, 24],
            },
            {
                "id": "8",
                "name": "Job",
                "category": "income",
                "interaction_type": "instant",
                "costs": [{"meter": "energy", "amount": 0.15}],
                "effects": [{"meter": "money", "amount": 0.225}, {"meter": "mood", "amount": -0.05}],
                "operating_hours": [8, 18],
            },
            {
                "id": "9",
                "name": "Park",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [],
                "effects": [{"meter": "mood", "amount": 0.20}],
                "operating_hours": [6, 20],
            },
            {
                "id": "10",
                "name": "Library",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [],
                "effects": [{"meter": "mood", "amount": 0.15}],
                "operating_hours": [8, 20],
            },
            {
                "id": "11",
                "name": "Bar",
                "category": "social",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.15}],
                "effects": [{"meter": "mood", "amount": 0.25}],
                "operating_hours": [18, 28],
            },
            {
                "id": "12",
                "name": "Recreation",
                "category": "leisure",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.10}],
                "effects": [{"meter": "mood", "amount": 0.30}],
                "operating_hours": [12, 24],
            },
            {
                "id": "13",
                "name": "SocialEvent",
                "category": "social",
                "interaction_type": "instant",
                "costs": [{"meter": "money", "amount": 0.08}],
                "effects": [{"meter": "mood", "amount": 0.20}],
                "operating_hours": [18, 23],
            },
        ],
    }

    with open(config_4m / "affordances.yaml", "w") as f:
        yaml.safe_dump(affordances_config, f)

    # Generate matching variables_reference.yaml for 4 meters
    # Must match bars_config meter count to avoid VFS/bars mismatch
    vfs_config = {
        "version": "1.0",
        "variables": [
            # Grid encoding (64 dims for 8×8 grid)
            {
                "id": "grid_encoding",
                "scope": "agent",
                "type": "vecNf",
                "dims": 64,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 64,
                "description": "8×8 grid encoding",
            },
            # Local window for POMDP (5×5 = 25 cells)
            {
                "id": "local_window",
                "scope": "agent",
                "type": "vecNf",
                "dims": 25,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 25,
                "description": "5×5 local window for POMDP",
            },
            # Position (2 dims)
            {
                "id": "position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 2,
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": [0.0, 0.0],
                "description": "Normalized agent position (x, y)",
            },
            # 4 meters: energy, health, money, mood
            {
                "id": "energy",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "health",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "money",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.0,
            },
            {
                "id": "mood",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            # Affordance at position (15 dims)
            {
                "id": "affordance_at_position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 15,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 14 + [1.0],
            },
            # Temporal features (4 scalars)
            {
                "id": "time_sin",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "time_cos",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 1.0,
            },
            {
                "id": "interaction_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "lifetime_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
        ],
        "exposed_observations": [
            {"id": "obs_grid_encoding", "source_variable": "grid_encoding", "exposed_to": ["agent"], "shape": [64], "normalization": None},
            {"id": "obs_local_window", "source_variable": "local_window", "exposed_to": ["agent"], "shape": [25], "normalization": None},
            {
                "id": "obs_position",
                "source_variable": "position",
                "exposed_to": ["agent"],
                "shape": [2],
                "normalization": {"kind": "minmax", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            {
                "id": "obs_energy",
                "source_variable": "energy",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_health",
                "source_variable": "health",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {"id": "obs_money", "source_variable": "money", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_mood",
                "source_variable": "mood",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_affordance_at_position",
                "source_variable": "affordance_at_position",
                "exposed_to": ["agent"],
                "shape": [15],
                "normalization": None,
            },
            {"id": "obs_time_sin", "source_variable": "time_sin", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {"id": "obs_time_cos", "source_variable": "time_cos", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_interaction_progress",
                "source_variable": "interaction_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_lifetime_progress",
                "source_variable": "lifetime_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
        ],
    }

    with open(config_4m / "variables_reference.yaml", "w") as f:
        yaml.safe_dump(vfs_config, f, sort_keys=False)

    return config_4m


@pytest.fixture
def task001_config_12meter(tmp_path: Path, test_config_pack_path: Path) -> Path:
    """Create temporary 12-meter config pack for TASK-001 testing.

    Meters: 8 standard + reputation, skill, spirituality, community_trust
    Use ONLY for: TASK-001 variable meter scaling tests
    Do NOT use for: L2 curriculum (use separate curriculum fixtures)

    Args:
        tmp_path: pytest's temporary directory
        test_config_pack_path: Path to source config pack

    Returns:
        Path to temporary 12-meter config pack directory
    """
    config_12m = tmp_path / "config_12m"
    shutil.copytree(test_config_pack_path, config_12m)

    # Load existing 8-meter bars
    with open(test_config_pack_path / "bars.yaml") as f:
        bars_8m = yaml.safe_load(f)

    # Add 4 new meters
    extra_meters = [
        {
            "name": "reputation",
            "index": 8,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.5,
            "base_depletion": 0.002,
            "description": "Social reputation",
        },
        {
            "name": "skill",
            "index": 9,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.3,
            "base_depletion": 0.001,
            "description": "Professional skills",
        },
        {
            "name": "spirituality",
            "index": 10,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.6,
            "base_depletion": 0.002,
            "description": "Spiritual wellbeing",
        },
        {
            "name": "community_trust",
            "index": 11,
            "tier": "secondary",
            "range": [0.0, 1.0],
            "initial": 0.7,
            "base_depletion": 0.001,
            "description": "Community trust level",
        },
    ]

    bars_12m = copy.deepcopy(bars_8m)  # Deep copy to avoid modifying original
    bars_12m["bars"].extend(extra_meters)

    with open(config_12m / "bars.yaml", "w") as f:
        yaml.safe_dump(bars_12m, f)

    # Generate matching variables_reference.yaml for 12 meters
    # Must match bars_config meter count to avoid VFS/bars mismatch
    vfs_config = {
        "version": "1.0",
        "variables": [
            # Grid encoding (64 dims for 8×8 grid)
            {
                "id": "grid_encoding",
                "scope": "agent",
                "type": "vecNf",
                "dims": 64,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 64,
                "description": "8×8 grid encoding",
            },
            # Local window for POMDP (5×5 = 25 cells)
            {
                "id": "local_window",
                "scope": "agent",
                "type": "vecNf",
                "dims": 25,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 25,
                "description": "5×5 local window for POMDP",
            },
            # Position (2 dims)
            {
                "id": "position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 2,
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": [0.0, 0.0],
                "description": "Normalized agent position (x, y)",
            },
            # 8 standard meters
            {
                "id": "energy",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "health",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "satiation",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "money",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.0,
            },
            {
                "id": "mood",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "social",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "fitness",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            {
                "id": "hygiene",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 1.0,
            },
            # 4 additional meters
            {
                "id": "reputation",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.5,
            },
            {
                "id": "skill",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.3,
            },
            {
                "id": "spirituality",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.6,
            },
            {
                "id": "community_trust",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine", "acs"],
                "writable_by": ["actions", "engine"],
                "default": 0.7,
            },
            # Affordance at position (15 dims)
            {
                "id": "affordance_at_position",
                "scope": "agent",
                "type": "vecNf",
                "dims": 15,
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": [0.0] * 14 + [1.0],
            },
            # Temporal features (4 scalars)
            {
                "id": "time_sin",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "time_cos",
                "scope": "global",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 1.0,
            },
            {
                "id": "interaction_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "tick",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
            {
                "id": "lifetime_progress",
                "scope": "agent",
                "type": "scalar",
                "lifetime": "episode",
                "readable_by": ["agent", "engine"],
                "writable_by": ["engine"],
                "default": 0.0,
            },
        ],
        "exposed_observations": [
            {"id": "obs_grid_encoding", "source_variable": "grid_encoding", "exposed_to": ["agent"], "shape": [64], "normalization": None},
            {"id": "obs_local_window", "source_variable": "local_window", "exposed_to": ["agent"], "shape": [25], "normalization": None},
            {
                "id": "obs_position",
                "source_variable": "position",
                "exposed_to": ["agent"],
                "shape": [2],
                "normalization": {"kind": "minmax", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            # 8 standard meters
            {
                "id": "obs_energy",
                "source_variable": "energy",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_health",
                "source_variable": "health",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_satiation",
                "source_variable": "satiation",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {"id": "obs_money", "source_variable": "money", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_mood",
                "source_variable": "mood",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_social",
                "source_variable": "social",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_fitness",
                "source_variable": "fitness",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_hygiene",
                "source_variable": "hygiene",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            # 4 additional meters
            {
                "id": "obs_reputation",
                "source_variable": "reputation",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_skill",
                "source_variable": "skill",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_spirituality",
                "source_variable": "spirituality",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_community_trust",
                "source_variable": "community_trust",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            # Affordance and temporal
            {
                "id": "obs_affordance_at_position",
                "source_variable": "affordance_at_position",
                "exposed_to": ["agent"],
                "shape": [15],
                "normalization": None,
            },
            {"id": "obs_time_sin", "source_variable": "time_sin", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {"id": "obs_time_cos", "source_variable": "time_cos", "exposed_to": ["agent"], "shape": [], "normalization": None},
            {
                "id": "obs_interaction_progress",
                "source_variable": "interaction_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
            {
                "id": "obs_lifetime_progress",
                "source_variable": "lifetime_progress",
                "exposed_to": ["agent"],
                "shape": [],
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0},
            },
        ],
    }

    with open(config_12m / "variables_reference.yaml", "w") as f:
        yaml.safe_dump(vfs_config, f, sort_keys=False)

    return config_12m


@pytest.fixture
def task001_env_4meter(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_4meter: Path,
) -> VectorizedHamletEnv:
    """4-meter environment for TASK-001 testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_4meter: Path to 4-meter config pack

    Returns:
        VectorizedHamletEnv instance with 4 meters
    """
    universe = compile_universe(task001_config_4meter)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_4meter_pomdp(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_4meter: Path,
) -> VectorizedHamletEnv:
    """4-meter POMDP environment for TASK-001 recurrent network testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_4meter: Path to 4-meter config pack

    Returns:
        VectorizedHamletEnv instance with 4 meters and partial observability
    """
    pomdp_config = task001_config_4meter.parent / "config_4m_pomdp"
    shutil.copytree(task001_config_4meter, pomdp_config)

    training_yaml = pomdp_config / "training.yaml"
    with open(training_yaml) as f:
        training_config = yaml.safe_load(f)

    training_env = training_config.get("environment", {})
    training_env["partial_observability"] = True
    training_env["vision_range"] = 2
    training_config["environment"] = training_env

    with open(training_yaml, "w") as f:
        yaml.safe_dump(training_config, f, sort_keys=False)

    universe = compile_universe(pomdp_config)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_12meter(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_12meter: Path,
) -> VectorizedHamletEnv:
    """12-meter environment for TASK-001 testing.

    Args:
        cpu_device: CPU device for deterministic behavior
        task001_config_12meter: Path to 12-meter config pack

    Returns:
        VectorizedHamletEnv instance with 12 meters
    """
    universe = compile_universe(task001_config_12meter)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


@pytest.fixture
def task001_env_12meter_pomdp(
    compile_universe: Callable[[Path | str], CompiledUniverse],
    cpu_device: torch.device,
    task001_config_12meter: Path,
) -> VectorizedHamletEnv:
    """12-meter POMDP environment for TASK-001 testing."""

    pomdp_config = task001_config_12meter.parent / "config_12m_pomdp"
    shutil.copytree(task001_config_12meter, pomdp_config)

    training_yaml = pomdp_config / "training.yaml"
    with open(training_yaml) as f:
        training_config = yaml.safe_load(f)

    training_env = training_config.get("environment", {})
    training_env["partial_observability"] = True
    training_env["vision_range"] = 2
    training_config["environment"] = training_env

    with open(training_yaml, "w") as f:
        yaml.safe_dump(training_config, f, sort_keys=False)

    universe = compile_universe(pomdp_config)
    return VectorizedHamletEnv.from_universe(
        universe,
        num_agents=1,
        device=cpu_device,
    )


# =============================================================================
# DATABASE FIXTURES
# =============================================================================


@pytest.fixture
def demo_database(tmp_path: Path):
    """Create a DemoDatabase with automatic cleanup.

    This fixture ensures database connections are properly closed after each test,
    preventing ResourceWarnings from unclosed sqlite3 connections.

    Args:
        tmp_path: pytest's temporary directory

    Yields:
        DemoDatabase instance

    Example:
        def test_something(demo_database):
            demo_database.insert_episode(...)
            # Database automatically closed after test
    """
    from townlet.demo.database import DemoDatabase

    db_path = tmp_path / "test.db"
    db = DemoDatabase(db_path)
    yield db
    db.close()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def sample_observations(basic_env: VectorizedHamletEnv, device: torch.device) -> torch.Tensor:
    """Generate sample observations from basic environment.

    Args:
        basic_env: Environment to generate observations from
        device: Device to place observations on

    Returns:
        Tensor of shape (num_agents, obs_dim)
    """
    obs = basic_env.reset()
    return obs.to(device)


@pytest.fixture
def sample_actions(device: torch.device) -> torch.Tensor:
    """Generate sample actions tensor.

    Returns:
        Tensor of shape (4,) with actions [0, 1, 2, 3]
    """
    return torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (run with --runslow)")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU (skipped if no CUDA)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA unavailable."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

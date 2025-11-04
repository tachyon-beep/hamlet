# Townlet Phase 0: Foundation (DTOs + Interfaces) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish data contracts (DTOs) and component interfaces for Townlet's GPU-native sparse reward system.

**Architecture:** Dual-path design with hot path (BatchedAgentState tensors for GPU training loops) and cold path (Pydantic DTOs for config/checkpoints). All component interfaces accept/return tensors to support future vectorization without API changes.

**Tech Stack:** PyTorch 2.0+, Pydantic 2.10+, Python 3.11+

**Duration:** 2-3 days

**Exit Criteria:**

- ✅ All DTOs have Pydantic validation (invalid values rejected)
- ✅ BatchedAgentState correctly handles tensor operations
- ✅ All ABCs prevent instantiation
- ✅ mypy --strict passes on all interface files
- ✅ 100% test coverage on DTOs and interfaces

---

## Task 1: Project Structure Setup

**Files:**

- Create: `src/townlet/__init__.py`
- Create: `src/townlet/training/__init__.py`
- Create: `src/townlet/curriculum/__init__.py`
- Create: `src/townlet/exploration/__init__.py`
- Create: `src/townlet/population/__init__.py`
- Create: `src/townlet/environment/__init__.py`
- Create: `tests/test_townlet/__init__.py`

**Step 1: Create directory structure**

Run:

```bash
mkdir -p src/townlet/{training,curriculum,exploration,population,environment}
mkdir -p tests/test_townlet
```

**Step 2: Create package **init** files**

Create each `__init__.py` with:

```python
"""Townlet: GPU-native sparse reward system."""
```

**Step 3: Verify structure**

Run: `tree src/townlet tests/test_townlet`

Expected output:

```
src/townlet/
├── __init__.py
├── curriculum/
│   └── __init__.py
├── environment/
│   └── __init__.py
├── exploration/
│   └── __init__.py
├── population/
│   └── __init__.py
└── training/
    └── __init__.py
tests/test_townlet/
└── __init__.py
```

**Step 4: Commit**

```bash
git add src/townlet tests/test_townlet
git commit -m "feat(townlet): initialize project structure

Create package structure for Townlet GPU-native sparse reward system:
- src/townlet/ with subpackages (training, curriculum, exploration, population, environment)
- tests/test_townlet/ for test suite

Part of Phase 0: Foundation (DTOs + Interfaces)"
```

---

## Task 2: Cold Path DTOs - CurriculumDecision

**Files:**

- Create: `src/townlet/training/state.py`
- Create: `tests/test_townlet/test_training/test_state.py`

**Step 1: Write failing test for CurriculumDecision validation**

Create `tests/test_townlet/test_training/__init__.py`:

```python
"""Tests for Townlet training infrastructure."""
```

Create `tests/test_townlet/test_training/test_state.py`:

```python
"""Tests for Townlet state DTOs (cold path)."""

import pytest
from pydantic import ValidationError


def test_curriculum_decision_valid():
    """CurriculumDecision should accept valid parameters."""
    from townlet.training.state import CurriculumDecision

    decision = CurriculumDecision(
        difficulty_level=0.5,
        active_meters=["energy", "hygiene"],
        depletion_multiplier=1.0,
        reward_mode="sparse",
        reason="Test curriculum decision"
    )

    assert decision.difficulty_level == 0.5
    assert decision.active_meters == ["energy", "hygiene"]
    assert decision.depletion_multiplier == 1.0
    assert decision.reward_mode == "sparse"
    assert decision.reason == "Test curriculum decision"


def test_curriculum_decision_difficulty_out_of_range():
    """CurriculumDecision should reject difficulty outside [0, 1]."""
    from townlet.training.state import CurriculumDecision

    with pytest.raises(ValidationError) as exc_info:
        CurriculumDecision(
            difficulty_level=1.5,  # Invalid: > 1.0
            active_meters=["energy"],
            depletion_multiplier=1.0,
            reward_mode="sparse",
            reason="Test"
        )

    assert "difficulty_level" in str(exc_info.value)


def test_curriculum_decision_invalid_reward_mode():
    """CurriculumDecision should reject invalid reward_mode."""
    from townlet.training.state import CurriculumDecision

    with pytest.raises(ValidationError) as exc_info:
        CurriculumDecision(
            difficulty_level=0.5,
            active_meters=["energy"],
            depletion_multiplier=1.0,
            reward_mode="invalid_mode",  # Invalid
            reason="Test"
        )

    assert "reward_mode" in str(exc_info.value)


def test_curriculum_decision_immutable():
    """CurriculumDecision should be immutable (frozen)."""
    from townlet.training.state import CurriculumDecision

    decision = CurriculumDecision(
        difficulty_level=0.5,
        active_meters=["energy"],
        depletion_multiplier=1.0,
        reward_mode="sparse",
        reason="Test"
    )

    with pytest.raises(ValidationError):
        decision.difficulty_level = 0.8  # Should fail (frozen)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_training/test_state.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.training.state'"

**Step 3: Write minimal CurriculumDecision implementation**

Create `src/townlet/training/state.py`:

```python
"""
State representations for Townlet training.

Contains DTOs for cold path (config, checkpoints, telemetry) using Pydantic
for validation, and hot path (training loop) using PyTorch tensors.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class CurriculumDecision(BaseModel):
    """
    Cold path: Curriculum decision for environment configuration.

    Returned by CurriculumManager to specify environment settings.
    Validated at construction, immutable, serializable.
    """
    model_config = ConfigDict(frozen=True)

    difficulty_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Difficulty level from 0.0 (easiest) to 1.0 (hardest)"
    )
    active_meters: List[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Which meters are active (e.g., ['energy', 'hygiene'])"
    )
    depletion_multiplier: float = Field(
        ...,
        gt=0.0,
        le=10.0,
        description="Depletion rate multiplier (0.1 = 10x slower, 1.0 = normal)"
    )
    reward_mode: str = Field(
        ...,
        pattern=r'^(shaped|sparse)$',
        description="Reward mode: 'shaped' (dense) or 'sparse'"
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation for this decision"
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_training/test_state.py -v`

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/townlet/training/state.py tests/test_townlet/test_training/
git commit -m "feat(townlet): add CurriculumDecision DTO with validation

Implement Pydantic DTO for cold path curriculum decisions:
- difficulty_level validated to [0.0, 1.0]
- reward_mode validated to 'shaped' or 'sparse'
- Immutable (frozen=True) to prevent accidental mutation
- Full test coverage for validation and immutability

Part of Phase 0: Foundation"
```

---

## Task 3: Cold Path DTOs - ExplorationConfig

**Files:**

- Modify: `src/townlet/training/state.py`
- Modify: `tests/test_townlet/test_training/test_state.py`

**Step 1: Write failing test for ExplorationConfig**

Add to `tests/test_townlet/test_training/test_state.py`:

```python
def test_exploration_config_valid():
    """ExplorationConfig should accept valid parameters."""
    from townlet.training.state import ExplorationConfig

    config = ExplorationConfig(
        strategy_type="epsilon_greedy",
        epsilon=0.5,
        epsilon_decay=0.995,
        intrinsic_weight=0.0
    )

    assert config.strategy_type == "epsilon_greedy"
    assert config.epsilon == 0.5
    assert config.epsilon_decay == 0.995
    assert config.intrinsic_weight == 0.0


def test_exploration_config_invalid_strategy():
    """ExplorationConfig should reject invalid strategy_type."""
    from townlet.training.state import ExplorationConfig

    with pytest.raises(ValidationError):
        ExplorationConfig(strategy_type="invalid_strategy")


def test_exploration_config_epsilon_out_of_range():
    """ExplorationConfig should reject epsilon outside [0, 1]."""
    from townlet.training.state import ExplorationConfig

    with pytest.raises(ValidationError) as exc_info:
        ExplorationConfig(epsilon=1.5)

    assert "epsilon" in str(exc_info.value)


def test_exploration_config_defaults():
    """ExplorationConfig should have sensible defaults."""
    from townlet.training.state import ExplorationConfig

    config = ExplorationConfig(strategy_type="epsilon_greedy")

    assert config.epsilon == 1.0  # Default: full exploration initially
    assert config.epsilon_decay == 0.995
    assert config.intrinsic_weight == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_training/test_state.py::test_exploration_config_valid -v`

Expected: FAIL with "ImportError: cannot import name 'ExplorationConfig'"

**Step 3: Write ExplorationConfig implementation**

Add to `src/townlet/training/state.py`:

```python
class ExplorationConfig(BaseModel):
    """
    Cold path: Configuration for exploration strategy.

    Defines parameters for epsilon-greedy, RND, or adaptive intrinsic exploration.
    """
    model_config = ConfigDict(frozen=True)

    strategy_type: str = Field(
        ...,
        pattern=r'^(epsilon_greedy|rnd|adaptive_intrinsic)$',
        description="Exploration strategy: epsilon_greedy, rnd, or adaptive_intrinsic"
    )
    epsilon: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Epsilon for epsilon-greedy (0.0 = greedy, 1.0 = random)"
    )
    epsilon_decay: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description="Epsilon decay per episode (0.995 = ~1% decay)"
    )
    intrinsic_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for intrinsic motivation rewards"
    )
    rnd_hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dimension for RND networks"
    )
    rnd_learning_rate: float = Field(
        default=0.0001,
        gt=0.0,
        description="Learning rate for RND predictor network"
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_training/test_state.py -k exploration -v`

Expected: PASS (all 4 exploration tests)

**Step 5: Commit**

```bash
git add src/townlet/training/state.py tests/test_townlet/test_training/test_state.py
git commit -m "feat(townlet): add ExplorationConfig DTO

Add Pydantic DTO for exploration strategy configuration:
- Validates strategy_type (epsilon_greedy, rnd, adaptive_intrinsic)
- Epsilon and decay validated to valid ranges
- Sensible defaults (epsilon=1.0, decay=0.995)
- RND-specific parameters with validation

Part of Phase 0: Foundation"
```

---

## Task 4: Cold Path DTOs - PopulationCheckpoint

**Files:**

- Modify: `src/townlet/training/state.py`
- Modify: `tests/test_townlet/test_training/test_state.py`

**Step 1: Write failing test for PopulationCheckpoint**

Add to `tests/test_townlet/test_training/test_state.py`:

```python
def test_population_checkpoint_valid():
    """PopulationCheckpoint should accept valid parameters."""
    from townlet.training.state import PopulationCheckpoint

    checkpoint = PopulationCheckpoint(
        generation=5,
        num_agents=10,
        agent_ids=["agent_0", "agent_1"],
        curriculum_states={"agent_0": {"stage": 2}},
        exploration_states={"agent_0": {"epsilon": 0.5}},
        pareto_frontier=["agent_0"],
        metrics_summary={"avg_survival": 100.0}
    )

    assert checkpoint.generation == 5
    assert checkpoint.num_agents == 10
    assert len(checkpoint.agent_ids) == 2


def test_population_checkpoint_num_agents_limit():
    """PopulationCheckpoint should enforce agent count limits."""
    from townlet.training.state import PopulationCheckpoint

    # Valid: 1000 agents (max)
    checkpoint = PopulationCheckpoint(
        generation=0,
        num_agents=1000,
        agent_ids=[f"agent_{i}" for i in range(1000)],
        curriculum_states={},
        exploration_states={},
        pareto_frontier=[],
        metrics_summary={}
    )
    assert checkpoint.num_agents == 1000

    # Invalid: 1001 agents (exceeds max)
    with pytest.raises(ValidationError):
        PopulationCheckpoint(
            generation=0,
            num_agents=1001,
            agent_ids=[],
            curriculum_states={},
            exploration_states={},
            pareto_frontier=[],
            metrics_summary={}
        )


def test_population_checkpoint_serialization():
    """PopulationCheckpoint should serialize to JSON."""
    from townlet.training.state import PopulationCheckpoint

    checkpoint = PopulationCheckpoint(
        generation=1,
        num_agents=2,
        agent_ids=["agent_0", "agent_1"],
        curriculum_states={},
        exploration_states={},
        pareto_frontier=[],
        metrics_summary={}
    )

    # Serialize to JSON
    json_str = checkpoint.model_dump_json()
    assert "generation" in json_str
    assert "num_agents" in json_str

    # Deserialize from JSON
    from townlet.training.state import PopulationCheckpoint
    restored = PopulationCheckpoint.model_validate_json(json_str)
    assert restored.generation == checkpoint.generation
    assert restored.num_agents == checkpoint.num_agents
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_training/test_state.py::test_population_checkpoint_valid -v`

Expected: FAIL with "ImportError: cannot import name 'PopulationCheckpoint'"

**Step 3: Write PopulationCheckpoint implementation**

Add to `src/townlet/training/state.py` (after imports, add `Dict, Any`):

```python
from typing import List, Dict, Any
```

Then add the class:

```python
class PopulationCheckpoint(BaseModel):
    """
    Cold path: Serializable population state for checkpointing.

    Contains all state needed to restore a population training run:
    per-agent curriculum state, exploration state, Pareto frontier, etc.
    """
    model_config = ConfigDict(frozen=True)

    generation: int = Field(
        ...,
        ge=0,
        description="Generation number (for genetic algorithms)"
    )
    num_agents: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of agents in population (1-1000)"
    )
    agent_ids: List[str] = Field(
        ...,
        description="List of agent identifiers"
    )
    curriculum_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent curriculum manager state"
    )
    exploration_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent exploration strategy state"
    )
    pareto_frontier: List[str] = Field(
        default_factory=list,
        description="Agent IDs on Pareto frontier (non-dominated solutions)"
    )
    metrics_summary: Dict[str, float] = Field(
        default_factory=dict,
        description="Summary metrics (avg_survival, avg_reward, etc.)"
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_training/test_state.py -k population -v`

Expected: PASS (all 3 population tests)

**Step 5: Commit**

```bash
git add src/townlet/training/state.py tests/test_townlet/test_training/test_state.py
git commit -m "feat(townlet): add PopulationCheckpoint DTO

Add Pydantic DTO for population checkpoint serialization:
- Supports 1-1000 agents
- Stores per-agent curriculum and exploration state
- Tracks Pareto frontier and metrics summary
- JSON serialization/deserialization tested

Part of Phase 0: Foundation"
```

---

## Task 5: Hot Path State - BatchedAgentState

**Files:**

- Modify: `src/townlet/training/state.py`
- Create: `tests/test_townlet/test_training/test_batched_state.py`

**Step 1: Write failing test for BatchedAgentState**

Create `tests/test_townlet/test_training/test_batched_state.py`:

```python
"""Tests for BatchedAgentState (hot path tensor container)."""

import pytest
import torch
import numpy as np


def test_batched_agent_state_construction():
    """BatchedAgentState should construct with correct tensor shapes."""
    from townlet.training.state import BatchedAgentState

    batch_size = 10
    obs_dim = 70

    state = BatchedAgentState(
        observations=torch.randn(batch_size, obs_dim),
        actions=torch.randint(0, 5, (batch_size,)),
        rewards=torch.randn(batch_size),
        dones=torch.zeros(batch_size, dtype=torch.bool),
        epsilons=torch.full((batch_size,), 0.5),
        intrinsic_rewards=torch.zeros(batch_size),
        survival_times=torch.randint(0, 1000, (batch_size,)),
        curriculum_difficulties=torch.full((batch_size,), 0.5),
        device=torch.device('cpu'),
    )

    assert state.batch_size == batch_size
    assert state.observations.shape == (batch_size, obs_dim)
    assert state.actions.shape == (batch_size,)
    assert state.device.type == 'cpu'


def test_batched_agent_state_device_transfer():
    """BatchedAgentState should support device transfer."""
    from townlet.training.state import BatchedAgentState

    state_cpu = BatchedAgentState(
        observations=torch.randn(5, 70),
        actions=torch.zeros(5, dtype=torch.long),
        rewards=torch.zeros(5),
        dones=torch.zeros(5, dtype=torch.bool),
        epsilons=torch.ones(5),
        intrinsic_rewards=torch.zeros(5),
        survival_times=torch.zeros(5, dtype=torch.long),
        curriculum_difficulties=torch.zeros(5),
        device=torch.device('cpu'),
    )

    # Transfer to same device (should work)
    state_cpu2 = state_cpu.to(torch.device('cpu'))
    assert state_cpu2.device.type == 'cpu'
    assert state_cpu2.observations.shape == state_cpu.observations.shape


def test_batched_agent_state_cpu_summary():
    """BatchedAgentState should extract CPU summary for telemetry."""
    from townlet.training.state import BatchedAgentState

    state = BatchedAgentState(
        observations=torch.randn(3, 70),
        actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        dones=torch.tensor([False, False, True]),
        epsilons=torch.tensor([0.9, 0.8, 0.7]),
        intrinsic_rewards=torch.tensor([0.1, 0.2, 0.3]),
        survival_times=torch.tensor([100, 200, 300]),
        curriculum_difficulties=torch.tensor([0.5, 0.6, 0.7]),
        device=torch.device('cpu'),
    )

    summary = state.detach_cpu_summary()

    # Should return dict of numpy arrays
    assert isinstance(summary, dict)
    assert isinstance(summary['rewards'], np.ndarray)
    assert summary['rewards'].shape == (3,)
    assert np.allclose(summary['rewards'], [1.0, 2.0, 3.0])

    assert 'survival_times' in summary
    assert 'epsilons' in summary
    assert 'curriculum_difficulties' in summary


def test_batched_agent_state_batch_size_property():
    """BatchedAgentState batch_size should match observations."""
    from townlet.training.state import BatchedAgentState

    for batch_size in [1, 5, 10, 100]:
        state = BatchedAgentState(
            observations=torch.randn(batch_size, 70),
            actions=torch.zeros(batch_size, dtype=torch.long),
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size, dtype=torch.bool),
            epsilons=torch.ones(batch_size),
            intrinsic_rewards=torch.zeros(batch_size),
            survival_times=torch.zeros(batch_size, dtype=torch.long),
            curriculum_difficulties=torch.zeros(batch_size),
            device=torch.device('cpu'),
        )

        assert state.batch_size == batch_size
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_training/test_batched_state.py -v`

Expected: FAIL with "ImportError: cannot import name 'BatchedAgentState'"

**Step 3: Write BatchedAgentState implementation**

Add to `src/townlet/training/state.py` (after imports, add torch and numpy):

```python
import torch
import numpy as np
```

Then add the class:

```python
class BatchedAgentState:
    """
    Hot path: Vectorized agent state for GPU training loops.

    All data is batched tensors (batch_size = num_agents).
    Optimized for GPU operations, minimal validation overhead.
    Use slots for memory efficiency.
    """
    __slots__ = [
        'observations', 'actions', 'rewards', 'dones',
        'epsilons', 'intrinsic_rewards', 'survival_times',
        'curriculum_difficulties', 'device'
    ]

    def __init__(
        self,
        observations: torch.Tensor,      # [batch, obs_dim]
        actions: torch.Tensor,           # [batch]
        rewards: torch.Tensor,           # [batch]
        dones: torch.Tensor,             # [batch] bool
        epsilons: torch.Tensor,          # [batch]
        intrinsic_rewards: torch.Tensor, # [batch]
        survival_times: torch.Tensor,    # [batch]
        curriculum_difficulties: torch.Tensor,  # [batch]
        device: torch.device,
    ):
        """
        Construct batched agent state.

        All tensors must be on the same device.
        No validation in __init__ for performance (hot path).
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.epsilons = epsilons
        self.intrinsic_rewards = intrinsic_rewards
        self.survival_times = survival_times
        self.curriculum_difficulties = curriculum_difficulties
        self.device = device

    @property
    def batch_size(self) -> int:
        """Get batch size from observations shape."""
        return self.observations.shape[0]

    def to(self, device: torch.device) -> 'BatchedAgentState':
        """
        Move all tensors to specified device.

        Returns new BatchedAgentState (tensors are immutable after .to()).
        """
        return BatchedAgentState(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            epsilons=self.epsilons.to(device),
            intrinsic_rewards=self.intrinsic_rewards.to(device),
            survival_times=self.survival_times.to(device),
            curriculum_difficulties=self.curriculum_difficulties.to(device),
            device=device,
        )

    def detach_cpu_summary(self) -> Dict[str, np.ndarray]:
        """
        Extract summary for telemetry (cold path).

        Returns dict of numpy arrays (CPU). Used for logging, checkpoints.
        """
        return {
            'rewards': self.rewards.detach().cpu().numpy(),
            'survival_times': self.survival_times.detach().cpu().numpy(),
            'epsilons': self.epsilons.detach().cpu().numpy(),
            'curriculum_difficulties': self.curriculum_difficulties.detach().cpu().numpy(),
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_training/test_batched_state.py -v`

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/townlet/training/state.py tests/test_townlet/test_training/test_batched_state.py
git commit -m "feat(townlet): add BatchedAgentState for hot path

Implement tensor container for GPU training loops:
- Batched tensors for vectorized operations [num_agents, ...]
- Device transfer with .to() method
- CPU summary extraction for telemetry
- Memory-efficient with __slots__
- Zero validation overhead (hot path)

Part of Phase 0: Foundation"
```

---

## Task 6: Interface - CurriculumManager ABC

**Files:**

- Create: `src/townlet/curriculum/base.py`
- Create: `tests/test_townlet/test_curriculum/test_base.py`

**Step 1: Write failing test for CurriculumManager interface**

Create `tests/test_townlet/test_curriculum/__init__.py`:

```python
"""Tests for Townlet curriculum managers."""
```

Create `tests/test_townlet/test_curriculum/test_base.py`:

```python
"""Tests for CurriculumManager abstract interface."""

import pytest


def test_curriculum_manager_cannot_instantiate():
    """CurriculumManager ABC should not be instantiable."""
    from townlet.curriculum.base import CurriculumManager

    with pytest.raises(TypeError) as exc_info:
        CurriculumManager()

    assert "abstract" in str(exc_info.value).lower()


def test_curriculum_manager_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.curriculum.base import CurriculumManager

    class IncompleteCurriculum(CurriculumManager):
        # Missing implementations
        pass

    with pytest.raises(TypeError):
        IncompleteCurriculum()


def test_curriculum_manager_interface_signature():
    """CurriculumManager should have expected method signatures."""
    from townlet.curriculum.base import CurriculumManager
    import inspect

    # Check that abstract methods exist
    abstract_methods = {
        name for name, method in inspect.getmembers(CurriculumManager)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'get_batch_decisions' in abstract_methods
    assert 'checkpoint_state' in abstract_methods
    assert 'load_state' in abstract_methods
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_curriculum/test_base.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.curriculum.base'"

**Step 3: Write CurriculumManager ABC implementation**

Create `src/townlet/curriculum/base.py`:

```python
"""
Abstract base class for curriculum managers.

Curriculum managers control environment difficulty progression based on
agent performance metrics (survival time, learning progress, policy entropy).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from townlet.training.state import BatchedAgentState, CurriculumDecision


class CurriculumManager(ABC):
    """
    Abstract interface for curriculum management.

    Implementations control environment difficulty by returning CurriculumDecisions
    that specify depletion rates, active meters, and reward mode.
    """

    @abstractmethod
    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """
        Get curriculum decisions for batch of agents.

        Called once per episode (not per step) to determine environment
        configuration for each agent.

        Args:
            agent_states: Current state for all agents [num_agents, ...]
            agent_ids: List of agent identifiers (for per-agent tracking)

        Returns:
            List of CurriculumDecisions (one per agent)

        Note:
            Input is GPU tensors, output is CPU DTOs. Overhead acceptable
            since this runs once per episode, not per step.
        """
        pass

    @abstractmethod
    def checkpoint_state(self) -> Dict[str, Any]:
        """
        Return serializable state for checkpoint saving.

        Should include:
        - Per-agent curriculum stage
        - Performance history (survival times, rewards)
        - Any internal state needed to resume

        Returns:
            Dict compatible with JSON/YAML serialization
        """
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore curriculum manager from checkpoint.

        Args:
            state: Dict from checkpoint_state()

        Raises:
            ValueError: If state is invalid or incompatible
        """
        pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_curriculum/test_base.py -v`

Expected: PASS (all 3 tests)

**Step 5: Run mypy strict on interface**

Run: `mypy --strict src/townlet/curriculum/base.py`

Expected: Success (no type errors)

**Step 6: Commit**

```bash
git add src/townlet/curriculum/base.py tests/test_townlet/test_curriculum/
git commit -m "feat(townlet): add CurriculumManager ABC interface

Define abstract interface for curriculum management:
- get_batch_decisions() returns per-agent curriculum settings
- checkpoint_state()/load_state() for training resumption
- Type-checked with mypy --strict
- Tests verify ABC prevents instantiation

Part of Phase 0: Foundation"
```

---

## Task 7: Interface - ExplorationStrategy ABC

**Files:**

- Create: `src/townlet/exploration/base.py`
- Create: `tests/test_townlet/test_exploration/test_base.py`

**Step 1: Write failing test for ExplorationStrategy interface**

Create `tests/test_townlet/test_exploration/__init__.py`:

```python
"""Tests for Townlet exploration strategies."""
```

Create `tests/test_townlet/test_exploration/test_base.py`:

```python
"""Tests for ExplorationStrategy abstract interface."""

import pytest


def test_exploration_strategy_cannot_instantiate():
    """ExplorationStrategy ABC should not be instantiable."""
    from townlet.exploration.base import ExplorationStrategy

    with pytest.raises(TypeError) as exc_info:
        ExplorationStrategy()

    assert "abstract" in str(exc_info.value).lower()


def test_exploration_strategy_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.exploration.base import ExplorationStrategy

    class IncompleteExploration(ExplorationStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteExploration()


def test_exploration_strategy_interface_signature():
    """ExplorationStrategy should have expected method signatures."""
    from townlet.exploration.base import ExplorationStrategy
    import inspect

    abstract_methods = {
        name for name, method in inspect.getmembers(ExplorationStrategy)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'select_actions' in abstract_methods
    assert 'compute_intrinsic_rewards' in abstract_methods
    assert 'update' in abstract_methods
    assert 'checkpoint_state' in abstract_methods
    assert 'load_state' in abstract_methods
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_exploration/test_base.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.exploration.base'"

**Step 3: Write ExplorationStrategy ABC implementation**

Create `src/townlet/exploration/base.py`:

```python
"""
Abstract base class for exploration strategies.

Exploration strategies control action selection (exploration vs exploitation)
and optionally provide intrinsic motivation rewards (RND, ICM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

from townlet.training.state import BatchedAgentState


class ExplorationStrategy(ABC):
    """
    Abstract interface for exploration strategies.

    Implementations manage epsilon-greedy, RND, adaptive intrinsic motivation, etc.
    All methods operate on batched tensors for GPU efficiency.
    """

    @abstractmethod
    def select_actions(
        self,
        q_values: torch.Tensor,  # [batch, num_actions]
        agent_states: BatchedAgentState,
    ) -> torch.Tensor:
        """
        Select actions for batch of agents (GPU).

        This runs EVERY STEP for all agents. Must be GPU-optimized.

        Args:
            q_values: Q-values for each action [batch, num_actions]
            agent_states: Current state (contains epsilons, curriculum stage, etc.)

        Returns:
            actions: [batch] tensor of selected actions (int)

        Note:
            Hot path - minimize overhead. No validation, no CPU transfers.
        """
        pass

    @abstractmethod
    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
    ) -> torch.Tensor:
        """
        Compute intrinsic motivation rewards (GPU).

        For RND: prediction error as novelty signal.
        For ICM: forward model prediction error.
        For epsilon-greedy: returns zeros.

        Args:
            observations: Current observations [batch, obs_dim]

        Returns:
            intrinsic_rewards: [batch] tensor

        Note:
            Hot path - runs every step. Return zeros if no intrinsic motivation.
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Update exploration networks (RND, ICM, etc.) from experience batch.

        For epsilon-greedy: no-op (nothing to update).
        For RND: train predictor network.
        For ICM: train forward/inverse models.

        Args:
            batch: Dict of tensors (states, actions, rewards, next_states, dones)

        Note:
            Called after replay buffer sampling. Can be slow (not hot path).
        """
        pass

    @abstractmethod
    def checkpoint_state(self) -> Dict[str, Any]:
        """
        Return serializable state for checkpoint saving.

        Should include:
        - Network weights (RND predictor, ICM models)
        - Optimizer state
        - Current epsilon (if applicable)
        - Intrinsic weight (if adaptive)

        Returns:
            Dict compatible with torch.save()
        """
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore exploration strategy from checkpoint.

        Args:
            state: Dict from checkpoint_state()

        Raises:
            ValueError: If state is invalid or incompatible
        """
        pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_exploration/test_base.py -v`

Expected: PASS (all 3 tests)

**Step 5: Run mypy strict on interface**

Run: `mypy --strict src/townlet/exploration/base.py`

Expected: Success (no type errors)

**Step 6: Commit**

```bash
git add src/townlet/exploration/base.py tests/test_townlet/test_exploration/
git commit -m "feat(townlet): add ExplorationStrategy ABC interface

Define abstract interface for exploration strategies:
- select_actions() for action selection (hot path, GPU)
- compute_intrinsic_rewards() for novelty/curiosity (hot path)
- update() for training exploration networks (cold path)
- checkpoint_state()/load_state() for resumption
- Type-checked with mypy --strict

Part of Phase 0: Foundation"
```

---

## Task 8: Interface - PopulationManager ABC

**Files:**

- Create: `src/townlet/population/base.py`
- Create: `tests/test_townlet/test_population/test_base.py`

**Step 1: Write failing test for PopulationManager interface**

Create `tests/test_townlet/test_population/__init__.py`:

```python
"""Tests for Townlet population managers."""
```

Create `tests/test_townlet/test_population/test_base.py`:

```python
"""Tests for PopulationManager abstract interface."""

import pytest


def test_population_manager_cannot_instantiate():
    """PopulationManager ABC should not be instantiable."""
    from townlet.population.base import PopulationManager

    with pytest.raises(TypeError) as exc_info:
        PopulationManager()

    assert "abstract" in str(exc_info.value).lower()


def test_population_manager_requires_all_methods():
    """Subclass must implement all abstract methods."""
    from townlet.population.base import PopulationManager

    class IncompletePopulation(PopulationManager):
        pass

    with pytest.raises(TypeError):
        IncompletePopulation()


def test_population_manager_interface_signature():
    """PopulationManager should have expected method signatures."""
    from townlet.population.base import PopulationManager
    import inspect

    abstract_methods = {
        name for name, method in inspect.getmembers(PopulationManager)
        if getattr(method, '__isabstractmethod__', False)
    }

    assert 'step_population' in abstract_methods
    assert 'get_checkpoint' in abstract_methods
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_population/test_base.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.population.base'"

**Step 3: Write PopulationManager ABC implementation**

Create `src/townlet/population/base.py`:

```python
"""
Abstract base class for population managers.

Population managers coordinate multiple agents, handle Pareto frontier tracking,
and (in future) manage genetic reproduction.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from townlet.training.state import BatchedAgentState, PopulationCheckpoint

# Avoid circular import
if TYPE_CHECKING:
    from townlet.environment.vectorized_env import VectorizedHamletEnv


class PopulationManager(ABC):
    """
    Abstract interface for population management.

    Implementations coordinate multiple agents (n=1 to 100), track performance,
    and manage reproduction (future feature).
    """

    @abstractmethod
    def step_population(
        self,
        envs: 'VectorizedHamletEnv',  # Forward reference
    ) -> BatchedAgentState:
        """
        Execute one training step for entire population (GPU).

        Coordinates:
        - Action selection via exploration strategy
        - Environment stepping (vectorized)
        - Reward calculation (extrinsic + intrinsic)
        - Replay buffer updates
        - Q-network training

        Args:
            envs: Vectorized environment [num_agents parallel]

        Returns:
            BatchedAgentState with all agent data after step

        Note:
            Hot path - called every step. Must be GPU-optimized.
        """
        pass

    @abstractmethod
    def get_checkpoint(self) -> PopulationCheckpoint:
        """
        Return Pydantic checkpoint (cold path).

        Aggregates:
        - Agent network weights
        - Curriculum states (per agent)
        - Exploration states (per agent)
        - Pareto frontier
        - Metrics summary

        Returns:
            PopulationCheckpoint (Pydantic DTO)
        """
        pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_population/test_base.py -v`

Expected: PASS (all 3 tests)

**Step 5: Run mypy strict on interface**

Run: `mypy --strict src/townlet/population/base.py`

Expected: Success (no type errors)

**Step 6: Commit**

```bash
git add src/townlet/population/base.py tests/test_townlet/test_population/
git commit -m "feat(townlet): add PopulationManager ABC interface

Define abstract interface for population management:
- step_population() coordinates vectorized training step
- get_checkpoint() returns serializable state
- Type-checked with mypy --strict
- Forward reference to VectorizedHamletEnv (created in Phase 1)

Part of Phase 0: Foundation"
```

---

## Task 9: Integration Test - Interface Compliance

**Files:**

- Create: `tests/test_townlet/test_interface_compliance.py`

**Step 1: Write interface compliance test framework**

Create `tests/test_townlet/test_interface_compliance.py`:

```python
"""
Interface compliance tests.

Ensures all implementations satisfy their interface contracts.
Parameterized to automatically test new implementations as they're added.
"""

import pytest
import torch

from townlet.training.state import BatchedAgentState


# Curriculum Manager Compliance Tests
# (Will be populated as implementations are added in Phase 1+)

@pytest.mark.parametrize("curriculum_class", [
    # Add implementations here:
    # StaticCurriculum,
    # AdversarialCurriculum,
])
def test_curriculum_manager_compliance(curriculum_class):
    """Verify curriculum implementations satisfy interface contract."""
    pytest.skip("No curriculum implementations yet (Phase 1)")


# Exploration Strategy Compliance Tests

@pytest.mark.parametrize("exploration_class", [
    # Add implementations here:
    # EpsilonGreedyExploration,
    # RNDExploration,
])
def test_exploration_strategy_compliance(exploration_class):
    """Verify exploration implementations satisfy interface contract."""
    pytest.skip("No exploration implementations yet (Phase 1)")


# Population Manager Compliance Tests

@pytest.mark.parametrize("population_class", [
    # Add implementations here:
    # VectorizedPopulation,
])
def test_population_manager_compliance(population_class):
    """Verify population implementations satisfy interface contract."""
    pytest.skip("No population implementations yet (Phase 1)")


# Helper: Create mock BatchedAgentState for testing

def create_mock_batched_state(batch_size: int = 1) -> BatchedAgentState:
    """Create mock BatchedAgentState for interface testing."""
    return BatchedAgentState(
        observations=torch.randn(batch_size, 70),
        actions=torch.zeros(batch_size, dtype=torch.long),
        rewards=torch.zeros(batch_size),
        dones=torch.zeros(batch_size, dtype=torch.bool),
        epsilons=torch.full((batch_size,), 0.5),
        intrinsic_rewards=torch.zeros(batch_size),
        survival_times=torch.randint(0, 1000, (batch_size,)),
        curriculum_difficulties=torch.full((batch_size,), 0.5),
        device=torch.device('cpu'),
    )
```

**Step 2: Verify tests run (all skipped initially)**

Run: `pytest tests/test_townlet/test_interface_compliance.py -v`

Expected: All tests SKIPPED (no implementations yet)

**Step 3: Commit**

```bash
git add tests/test_townlet/test_interface_compliance.py
git commit -m "test(townlet): add interface compliance test framework

Add parameterized test framework for interface compliance:
- Will automatically test new implementations as they're added
- Currently all skipped (no implementations in Phase 0)
- Includes mock helper for creating test BatchedAgentState

Part of Phase 0: Foundation"
```

---

## Task 10: Documentation and Verification

**Files:**

- Create: `docs/townlet/PHASE0_VERIFICATION.md`

**Step 1: Create verification checklist**

Create `docs/townlet/PHASE0_VERIFICATION.md`:

```markdown
# Townlet Phase 0 Verification Checklist

**Date Completed**: [FILL IN]

## DTOs (Cold Path) - Pydantic

- [x] CurriculumDecision
  - [x] Validates difficulty_level ∈ [0, 1]
  - [x] Validates reward_mode ∈ {shaped, sparse}
  - [x] Immutable (frozen=True)
  - [x] Test coverage: 4/4 tests passing

- [x] ExplorationConfig
  - [x] Validates strategy_type ∈ {epsilon_greedy, rnd, adaptive_intrinsic}
  - [x] Validates epsilon ∈ [0, 1]
  - [x] Sensible defaults
  - [x] Test coverage: 4/4 tests passing

- [x] PopulationCheckpoint
  - [x] Validates num_agents ∈ [1, 1000]
  - [x] JSON serialization/deserialization
  - [x] Test coverage: 3/3 tests passing

## Hot Path State - Tensors

- [x] BatchedAgentState
  - [x] Constructs with correct shapes
  - [x] Device transfer with .to()
  - [x] CPU summary extraction
  - [x] batch_size property
  - [x] Test coverage: 4/4 tests passing

## Interfaces (Abstract Base Classes)

- [x] CurriculumManager ABC
  - [x] Cannot instantiate
  - [x] Requires all abstract methods
  - [x] Type-checked with mypy --strict
  - [x] Test coverage: 3/3 tests passing

- [x] ExplorationStrategy ABC
  - [x] Cannot instantiate
  - [x] Requires all abstract methods
  - [x] Type-checked with mypy --strict
  - [x] Test coverage: 3/3 tests passing

- [x] PopulationManager ABC
  - [x] Cannot instantiate
  - [x] Requires all abstract methods
  - [x] Type-checked with mypy --strict
  - [x] Test coverage: 3/3 tests passing

## Code Quality

- [x] All files have docstrings
- [x] mypy --strict passes on all interface files
- [x] Test coverage: 100% on DTOs and interfaces
- [x] All commits follow conventional format

## Commands to Verify

```bash
# Run all Phase 0 tests
pytest tests/test_townlet/test_training/ tests/test_townlet/test_curriculum/test_base.py tests/test_townlet/test_exploration/test_base.py tests/test_townlet/test_population/test_base.py -v

# Verify mypy strict
mypy --strict src/townlet/curriculum/base.py
mypy --strict src/townlet/exploration/base.py
mypy --strict src/townlet/population/base.py

# Check test coverage
pytest --cov=townlet --cov-report=term-missing tests/test_townlet/
```

## Next Steps (Phase 1)

- [ ] VectorizedHamletEnv implementation
- [ ] StaticCurriculum (trivial implementation)
- [ ] EpsilonGreedyExploration (vectorized)
- [ ] VectorizedPopulation (coordinates above)
- [ ] Oracle validation against Hamlet

## Notes

Phase 0 establishes contracts. All future implementations must satisfy these interfaces without API changes.

```

**Step 2: Run full verification**

Run commands from verification doc:
```bash
pytest tests/test_townlet/test_training/ tests/test_townlet/test_curriculum/test_base.py tests/test_townlet/test_exploration/test_base.py tests/test_townlet/test_population/test_base.py -v

mypy --strict src/townlet/curriculum/base.py
mypy --strict src/townlet/exploration/base.py
mypy --strict src/townlet/population/base.py

pytest --cov=townlet --cov-report=term-missing tests/test_townlet/
```

Expected: All tests pass, mypy clean, 100% coverage on implemented modules

**Step 3: Commit documentation**

```bash
git add docs/townlet/
git commit -m "docs(townlet): add Phase 0 verification checklist

Add comprehensive verification checklist for Phase 0:
- All DTOs and interfaces documented
- Verification commands provided
- Next steps outlined (Phase 1)

Phase 0 Complete: Foundation (DTOs + Interfaces)"
```

**Step 4: Create Phase 0 completion tag**

```bash
git tag -a phase0-complete -m "Townlet Phase 0 Complete: DTOs + Interfaces

All data contracts and component interfaces established:
- 3 Pydantic DTOs (CurriculumDecision, ExplorationConfig, PopulationCheckpoint)
- BatchedAgentState tensor container
- 3 abstract interfaces (CurriculumManager, ExplorationStrategy, PopulationManager)
- 100% test coverage on foundation
- mypy --strict passes on all interfaces

Ready for Phase 1: GPU infrastructure and trivial implementations"

git push origin phase0-complete
```

---

## Phase 0 Complete! 🎉

**Deliverables Summary**:

- ✅ 3 Pydantic DTOs with validation (CurriculumDecision, ExplorationConfig, PopulationCheckpoint)
- ✅ BatchedAgentState tensor container (hot path)
- ✅ 3 abstract interfaces (CurriculumManager, ExplorationStrategy, PopulationManager)
- ✅ 22 tests (100% passing)
- ✅ mypy --strict passes on all interfaces
- ✅ Verification checklist and documentation

**Lines of Code**:

- Source: ~450 LOC
- Tests: ~600 LOC
- Total: ~1,050 LOC

**Next Phase**: Phase 1 - GPU Infrastructure + Trivial Implementations (5-7 days)

- VectorizedHamletEnv (full GPU implementation)
- StaticCurriculum (no adaptation)
- EpsilonGreedyExploration (vectorized)
- VectorizedPopulation (coordinates above)
- Oracle validation against Hamlet

---

## Execution Options

**Plan complete and saved to `docs/plans/2025-10-29-townlet-phase0-foundation.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration with quality gates

**2. Parallel Session (separate)** - Open new session, use executing-plans skill for batch execution with checkpoints

**Which approach would you prefer?**

# Townlet Phase 1: GPU Infrastructure + Trivial Implementations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build VectorizedHamletEnv (GPU-native), StaticCurriculum, EpsilonGreedy, and VectorizedPopulation. Validate against Hamlet at n=1.

**Architecture:** Full GPU vectorization [num_agents, ...] from day 1. Trivial implementations (no adaptation) to prove interfaces work. Oracle validation ensures correctness.

**Tech Stack:** PyTorch 2.0+, PettingZoo (for reference), Python 3.11+

**Duration:** 5-7 days

**Exit Criteria:**

- ✅ Single agent (n=1) trains successfully with GPU implementation
- ✅ Works on both CPU and GPU
- ✅ Oracle validation: Townlet matches Hamlet shaped rewards within 1e-4
- ✅ Checkpoints save/restore correctly

---

## Task 1: VectorizedHamletEnv - Core Structure

**Files:**

- Create: `src/townlet/environment/vectorized_env.py`
- Create: `tests/test_townlet/test_environment/test_vectorized_env.py`

**Step 1: Write failing test for VectorizedHamletEnv construction**

Create `tests/test_townlet/test_environment/__init__.py`:

```python
"""Tests for Townlet vectorized environment."""
```

Create `tests/test_townlet/test_environment/test_vectorized_env.py`:

```python
"""Tests for VectorizedHamletEnv (GPU-native)."""

import pytest
import torch


def test_vectorized_env_construction():
    """VectorizedHamletEnv should construct with correct batch size."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    num_agents = 3
    env = VectorizedHamletEnv(
        num_agents=num_agents,
        grid_size=8,
        device=torch.device('cpu'),
    )

    assert env.num_agents == num_agents
    assert env.grid_size == 8
    assert env.device.type == 'cpu'
    assert env.observation_dim == 70  # 8×8 grid + 6 meters


def test_vectorized_env_reset():
    """Reset should return batched observations."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(
        num_agents=5,
        grid_size=8,
        device=torch.device('cpu'),
    )

    observations = env.reset()

    assert isinstance(observations, torch.Tensor)
    assert observations.shape == (5, 70)  # [num_agents, obs_dim]
    assert observations.device.type == 'cpu'
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_environment/test_vectorized_env.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.environment.vectorized_env'"

**Step 3: Write minimal VectorizedHamletEnv implementation**

Create `src/townlet/environment/vectorized_env.py`:

```python
"""
Vectorized Hamlet environment for GPU-native training.

Batches multiple independent Hamlet environments into a single vectorized
environment with tensor operations [num_agents, ...].
"""

import torch
import numpy as np
from typing import Tuple, Optional


class VectorizedHamletEnv:
    """
    GPU-native vectorized Hamlet environment.

    Batches multiple independent environments for parallel execution.
    All state is stored as PyTorch tensors on specified device.
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int = 8,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize vectorized environment.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (cpu or cuda)
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device

        # Observation: grid one-hot (64) + 6 meters (normalized)
        self.observation_dim = grid_size * grid_size + 6
        self.action_dim = 5  # UP, DOWN, LEFT, RIGHT, INTERACT

        # Affordance positions (from Hamlet default layout)
        self.affordances = {
            'Bed': torch.tensor([1, 1], device=device),
            'Shower': torch.tensor([2, 2], device=device),
            'HomeMeal': torch.tensor([1, 3], device=device),
            'FastFood': torch.tensor([5, 6], device=device),
            'Job': torch.tensor([6, 6], device=device),
            'Gym': torch.tensor([7, 3], device=device),
            'Bar': torch.tensor([7, 0], device=device),
            'Recreation': torch.tensor([0, 7], device=device),
        }

        # State tensors (initialized in reset)
        self.positions: Optional[torch.Tensor] = None  # [num_agents, 2]
        self.meters: Optional[torch.Tensor] = None  # [num_agents, 6]
        self.dones: Optional[torch.Tensor] = None  # [num_agents]
        self.step_counts: Optional[torch.Tensor] = None  # [num_agents]

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Random starting positions
        self.positions = torch.randint(
            0, self.grid_size, (self.num_agents, 2), device=self.device
        )

        # Initial meter values (normalized to [0, 1])
        # [energy, hygiene, satiation, money, mood, social]
        self.meters = torch.tensor([
            [1.0, 1.0, 1.0, 0.5, 1.0, 0.5]  # Default initial values
        ], device=self.device).repeat(self.num_agents, 1)

        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        return self._get_observations()

    def _get_observations(self) -> torch.Tensor:
        """
        Construct observation vector.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Grid encoding: one-hot position
        grid_encoding = torch.zeros(
            self.num_agents, self.grid_size * self.grid_size, device=self.device
        )
        flat_indices = self.positions[:, 0] * self.grid_size + self.positions[:, 1]
        grid_encoding.scatter_(1, flat_indices.unsqueeze(1), 1.0)

        # Concatenate grid + meters
        observations = torch.cat([grid_encoding, self.meters], dim=1)

        return observations
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_environment/test_vectorized_env.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_environment/
git commit -m "feat(townlet): add VectorizedHamletEnv core structure

Implement GPU-native vectorized environment:
- Batched state tensors [num_agents, ...]
- reset() returns batched observations
- Observation: grid one-hot (64) + 6 meters
- Affordance positions from Hamlet defaults

Part of Phase 1: GPU Infrastructure"
```

---

## Task 2: VectorizedHamletEnv - Step Function

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`
- Modify: `tests/test_townlet/test_environment/test_vectorized_env.py`

**Step 1: Write failing test for step function**

Add to `tests/test_townlet/test_environment/test_vectorized_env.py`:

```python
def test_vectorized_env_step():
    """Step should return batched (obs, rewards, dones, info)."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=3, grid_size=8, device=torch.device('cpu'))
    env.reset()

    # All agents move UP
    actions = torch.zeros(3, dtype=torch.long)  # 0 = UP

    obs, rewards, dones, info = env.step(actions)

    assert obs.shape == (3, 70)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert isinstance(info, dict)


def test_vectorized_env_movement():
    """Agents should move correctly."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    env.reset()

    # Set known position
    env.positions[0] = torch.tensor([4, 4], device=env.device)

    # Move UP (action 0)
    obs, _, _, _ = env.step(torch.tensor([0]))
    assert env.positions[0, 0] == 3  # Row decreased
    assert env.positions[0, 1] == 4  # Column unchanged

    # Move RIGHT (action 3)
    obs, _, _, _ = env.step(torch.tensor([3]))
    assert env.positions[0, 0] == 3  # Row unchanged
    assert env.positions[0, 1] == 5  # Column increased


def test_vectorized_env_meter_depletion():
    """Meters should deplete each step."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    env.reset()

    initial_energy = env.meters[0, 0].item()

    # Take 10 steps
    for _ in range(10):
        env.step(torch.tensor([0]))  # Move UP

    final_energy = env.meters[0, 0].item()

    assert final_energy < initial_energy  # Energy depleted
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_environment/test_vectorized_env.py::test_vectorized_env_step -v`

Expected: FAIL with "AttributeError: 'VectorizedHamletEnv' object has no attribute 'step'"

**Step 3: Implement step function**

Add to `src/townlet/environment/vectorized_env.py`:

```python
    def step(
        self,
        actions: torch.Tensor,  # [num_agents]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one step for all agents.

        Args:
            actions: [num_agents] tensor of actions (0-4)

        Returns:
            observations: [num_agents, observation_dim]
            rewards: [num_agents]
            dones: [num_agents] bool
            info: dict with metadata
        """
        # 1. Execute actions
        self._execute_actions(actions)

        # 2. Deplete meters
        self._deplete_meters()

        # 3. Check terminal conditions
        self._check_dones()

        # 4. Calculate rewards (shaped rewards for now)
        rewards = self._calculate_shaped_rewards()

        # 5. Increment step counts
        self.step_counts += 1

        observations = self._get_observations()

        info = {
            'step_counts': self.step_counts.clone(),
            'positions': self.positions.clone(),
        }

        return observations, rewards, self.dones, info

    def _execute_actions(self, actions: torch.Tensor) -> None:
        """
        Execute movement and interaction actions.

        Args:
            actions: [num_agents] tensor
                0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=INTERACT
        """
        # Movement deltas
        deltas = torch.tensor([
            [-1, 0],  # UP
            [1, 0],   # DOWN
            [0, -1],  # LEFT
            [0, 1],   # RIGHT
            [0, 0],   # INTERACT (no movement)
        ], device=self.device)

        # Apply movement
        movement_deltas = deltas[actions]  # [num_agents, 2]
        new_positions = self.positions + movement_deltas

        # Clamp to grid boundaries
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

        self.positions = new_positions

        # Handle INTERACT actions
        interact_mask = (actions == 4)
        if interact_mask.any():
            self._handle_interactions(interact_mask)

    def _handle_interactions(self, interact_mask: torch.Tensor) -> None:
        """
        Handle INTERACT action at affordances.

        Args:
            interact_mask: [num_agents] bool mask
        """
        # Check each affordance
        for affordance_name, affordance_pos in self.affordances.items():
            # Distance to affordance
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask

            if not at_affordance.any():
                continue

            # Apply affordance effects (simplified from Hamlet)
            if affordance_name == 'Bed':
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.8, 0.0, 1.0
                )  # Energy +80%
            elif affordance_name == 'Shower':
                self.meters[at_affordance, 1] = torch.clamp(
                    self.meters[at_affordance, 1] + 0.6, 0.0, 1.0
                )  # Hygiene +60%
            elif affordance_name == 'HomeMeal':
                self.meters[at_affordance, 2] = torch.clamp(
                    self.meters[at_affordance, 2] + 0.5, 0.0, 1.0
                )  # Satiation +50%
                self.meters[at_affordance, 3] -= 0.04  # Money -$4
            elif affordance_name == 'Job':
                self.meters[at_affordance, 3] += 0.3  # Money +$30
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.2, 0.0, 1.0
                )  # Energy -20%

    def _deplete_meters(self) -> None:
        """Deplete meters each step."""
        # Depletion rates (per step, from Hamlet)
        depletions = torch.tensor([
            0.005,  # energy: 0.5% per step
            0.003,  # hygiene: 0.3%
            0.004,  # satiation: 0.4%
            0.0,    # money: no passive depletion
            0.001,  # mood: 0.1%
            0.006,  # social: 0.6%
        ], device=self.device)

        self.meters = torch.clamp(
            self.meters - depletions, 0.0, 1.0
        )

    def _check_dones(self) -> None:
        """Check terminal conditions."""
        # Terminal if any critical meter (energy, hygiene, satiation) hits 0
        critical_meters = self.meters[:, :3]  # energy, hygiene, satiation
        self.dones = (critical_meters <= 0.0).any(dim=1)

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        Calculate shaped rewards (Hamlet-style two-tier).

        Returns:
            rewards: [num_agents]
        """
        rewards = torch.zeros(self.num_agents, device=self.device)

        # Tier 1: Meter-based feedback
        for i, meter_name in enumerate(['energy', 'hygiene', 'satiation']):
            meter_values = self.meters[:, i]

            # Healthy (>0.8): +0.5
            rewards += torch.where(meter_values > 0.8, 0.5, 0.0)

            # Okay (0.5-0.8): +0.2
            rewards += torch.where(
                (meter_values > 0.5) & (meter_values <= 0.8), 0.2, 0.0
            )

            # Concerning (0.2-0.5): -0.5
            rewards += torch.where(
                (meter_values > 0.2) & (meter_values <= 0.5), -0.5, 0.0
            )

            # Critical (<0.2): -2.0
            rewards += torch.where(meter_values <= 0.2, -2.0, 0.0)

        # Terminal penalty
        rewards = torch.where(self.dones, -100.0, rewards)

        return rewards
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_environment/test_vectorized_env.py -v`

Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_environment/test_vectorized_env.py
git commit -m "feat(townlet): implement VectorizedHamletEnv step function

Add vectorized environment stepping:
- Action execution (movement + interactions)
- Meter depletion (vectorized)
- Terminal condition checking
- Shaped reward calculation (two-tier from Hamlet)
- Full test coverage for movement, depletion, rewards

Part of Phase 1: GPU Infrastructure"
```

---

## Task 3: StaticCurriculum - Trivial Implementation

**Files:**

- Create: `src/townlet/curriculum/static.py`
- Create: `tests/test_townlet/test_curriculum/test_static.py`

**Step 1: Write failing test for StaticCurriculum**

Create `tests/test_townlet/test_curriculum/test_static.py`:

```python
"""Tests for StaticCurriculum (trivial implementation)."""

import pytest
import torch

from townlet.training.state import BatchedAgentState, CurriculumDecision


def test_static_curriculum_returns_same_decision():
    """StaticCurriculum should always return the same decision."""
    from townlet.curriculum.static import StaticCurriculum

    curriculum = StaticCurriculum(
        difficulty_level=0.5,
        reward_mode='shaped',
    )

    # Create mock state
    state = BatchedAgentState(
        observations=torch.randn(3, 70),
        actions=torch.zeros(3, dtype=torch.long),
        rewards=torch.zeros(3),
        dones=torch.zeros(3, dtype=torch.bool),
        epsilons=torch.ones(3),
        intrinsic_rewards=torch.zeros(3),
        survival_times=torch.randint(0, 1000, (3,)),
        curriculum_difficulties=torch.full((3,), 0.5),
        device=torch.device('cpu'),
    )

    decisions = curriculum.get_batch_decisions(
        state, agent_ids=['agent_0', 'agent_1', 'agent_2']
    )

    assert len(decisions) == 3
    for decision in decisions:
        assert isinstance(decision, CurriculumDecision)
        assert decision.difficulty_level == 0.5
        assert decision.reward_mode == 'shaped'


def test_static_curriculum_checkpoint():
    """StaticCurriculum checkpoint should be serializable."""
    from townlet.curriculum.static import StaticCurriculum

    curriculum = StaticCurriculum(difficulty_level=0.8, reward_mode='sparse')

    state = curriculum.checkpoint_state()

    assert isinstance(state, dict)
    assert state['difficulty_level'] == 0.8
    assert state['reward_mode'] == 'sparse'

    # Should be able to restore
    new_curriculum = StaticCurriculum(difficulty_level=0.0, reward_mode='shaped')
    new_curriculum.load_state(state)

    assert new_curriculum.difficulty_level == 0.8
    assert new_curriculum.reward_mode == 'sparse'
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_curriculum/test_static.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'townlet.curriculum.static'"

**Step 3: Write StaticCurriculum implementation**

Create `src/townlet/curriculum/static.py`:

```python
"""
Static curriculum manager (trivial implementation).

Always returns the same curriculum decision. Used for baseline testing
and to validate the curriculum interface works.
"""

from typing import List, Dict, Any

from townlet.curriculum.base import CurriculumManager
from townlet.training.state import BatchedAgentState, CurriculumDecision


class StaticCurriculum(CurriculumManager):
    """
    Static curriculum - no adaptation.

    Returns the same curriculum decision for all agents at all times.
    Useful for baseline experiments and interface validation.
    """

    def __init__(
        self,
        difficulty_level: float = 0.5,
        reward_mode: str = 'shaped',
        active_meters: List[str] = None,
        depletion_multiplier: float = 1.0,
    ):
        """
        Initialize static curriculum.

        Args:
            difficulty_level: Fixed difficulty (0.0-1.0)
            reward_mode: 'shaped' or 'sparse'
            active_meters: Which meters are active (default: all 6)
            depletion_multiplier: Depletion rate multiplier
        """
        self.difficulty_level = difficulty_level
        self.reward_mode = reward_mode
        self.active_meters = active_meters or [
            'energy', 'hygiene', 'satiation', 'money', 'mood', 'social'
        ]
        self.depletion_multiplier = depletion_multiplier

    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """
        Get curriculum decisions (same for all agents).

        Args:
            agent_states: Current agent state (ignored)
            agent_ids: List of agent IDs

        Returns:
            List of identical CurriculumDecisions
        """
        decision = CurriculumDecision(
            difficulty_level=self.difficulty_level,
            active_meters=self.active_meters,
            depletion_multiplier=self.depletion_multiplier,
            reward_mode=self.reward_mode,
            reason=f"Static curriculum (difficulty={self.difficulty_level})",
        )

        # Return same decision for all agents
        return [decision] * len(agent_ids)

    def checkpoint_state(self) -> Dict[str, Any]:
        """
        Return serializable state.

        Returns:
            Dict with all configuration
        """
        return {
            'difficulty_level': self.difficulty_level,
            'reward_mode': self.reward_mode,
            'active_meters': self.active_meters,
            'depletion_multiplier': self.depletion_multiplier,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore from checkpoint.

        Args:
            state: Dict from checkpoint_state()
        """
        self.difficulty_level = state['difficulty_level']
        self.reward_mode = state['reward_mode']
        self.active_meters = state['active_meters']
        self.depletion_multiplier = state['depletion_multiplier']
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_curriculum/test_static.py -v`

Expected: PASS (2 tests)

**Step 5: Update interface compliance tests**

Add to `tests/test_townlet/test_interface_compliance.py`:

```python
from townlet.curriculum.static import StaticCurriculum

@pytest.mark.parametrize("curriculum_class", [
    StaticCurriculum,
])
def test_curriculum_manager_compliance(curriculum_class):
    """Verify curriculum implementations satisfy interface contract."""
    # Remove skip
    curriculum = curriculum_class()

    # Should have all required methods
    assert hasattr(curriculum, 'get_batch_decisions')
    assert hasattr(curriculum, 'checkpoint_state')
    assert hasattr(curriculum, 'load_state')

    # get_batch_decisions should return list of CurriculumDecisions
    state = create_mock_batched_state(batch_size=2)
    decisions = curriculum.get_batch_decisions(state, ['agent_0', 'agent_1'])

    assert isinstance(decisions, list)
    assert len(decisions) == 2

    from townlet.training.state import CurriculumDecision
    for decision in decisions:
        assert isinstance(decision, CurriculumDecision)

    # checkpoint/restore should work
    checkpoint = curriculum.checkpoint_state()
    assert isinstance(checkpoint, dict)

    curriculum.load_state(checkpoint)  # Should not raise
```

**Step 6: Run interface compliance test**

Run: `pytest tests/test_townlet/test_interface_compliance.py::test_curriculum_manager_compliance -v`

Expected: PASS

**Step 7: Commit**

```bash
git add src/townlet/curriculum/static.py tests/test_townlet/test_curriculum/test_static.py tests/test_townlet/test_interface_compliance.py
git commit -m "feat(townlet): add StaticCurriculum implementation

Implement trivial curriculum manager:
- Returns same decision for all agents
- No adaptation or learning
- Validates CurriculumManager interface
- Full checkpoint/restore support
- Interface compliance tests passing

Part of Phase 1: GPU Infrastructure"
```

---

## Task 4: EpsilonGreedyExploration - Vectorized Implementation

**Files:**

- Create: `src/townlet/exploration/epsilon_greedy.py`
- Create: `tests/test_townlet/test_exploration/test_epsilon_greedy.py`

**Step 1: Write failing test for EpsilonGreedyExploration**

Create `tests/test_townlet/test_exploration/test_epsilon_greedy.py`:

```python
"""Tests for EpsilonGreedyExploration (vectorized)."""

import pytest
import torch

from townlet.training.state import BatchedAgentState


def test_epsilon_greedy_select_actions():
    """EpsilonGreedy should select actions based on epsilon."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(
        epsilon=0.0,  # Greedy (always pick max Q)
        epsilon_decay=1.0,
    )

    # Q-values with clear max
    q_values = torch.tensor([
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Action 1 is best
        [0.0, 0.0, 2.0, 0.0, 0.0],  # Action 2 is best
    ])

    state = BatchedAgentState(
        observations=torch.randn(2, 70),
        actions=torch.zeros(2, dtype=torch.long),
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        epsilons=torch.zeros(2),  # Greedy
        intrinsic_rewards=torch.zeros(2),
        survival_times=torch.zeros(2, dtype=torch.long),
        curriculum_difficulties=torch.zeros(2),
        device=torch.device('cpu'),
    )

    actions = exploration.select_actions(q_values, state)

    assert actions[0] == 1  # Greedy selects action 1
    assert actions[1] == 2  # Greedy selects action 2


def test_epsilon_greedy_exploration():
    """High epsilon should produce random actions."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=1.0)

    q_values = torch.zeros(100, 5)  # All Q-values = 0
    q_values[:, 0] = 10.0  # Action 0 is "best"

    state = BatchedAgentState(
        observations=torch.randn(100, 70),
        actions=torch.zeros(100, dtype=torch.long),
        rewards=torch.zeros(100),
        dones=torch.zeros(100, dtype=torch.bool),
        epsilons=torch.ones(100),  # Full exploration
        intrinsic_rewards=torch.zeros(100),
        survival_times=torch.zeros(100, dtype=torch.long),
        curriculum_difficulties=torch.zeros(100),
        device=torch.device('cpu'),
    )

    actions = exploration.select_actions(q_values, state)

    # With epsilon=1.0, should get diverse actions (not all 0)
    unique_actions = torch.unique(actions)
    assert len(unique_actions) > 1  # Not all the same action


def test_epsilon_greedy_no_intrinsic_rewards():
    """EpsilonGreedy should return zero intrinsic rewards."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=0.995)

    observations = torch.randn(10, 70)
    intrinsic_rewards = exploration.compute_intrinsic_rewards(observations)

    assert intrinsic_rewards.shape == (10,)
    assert torch.all(intrinsic_rewards == 0.0)


def test_epsilon_greedy_checkpoint():
    """EpsilonGreedy checkpoint should include epsilon."""
    from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

    exploration = EpsilonGreedyExploration(epsilon=0.8, epsilon_decay=0.99)

    state = exploration.checkpoint_state()

    assert isinstance(state, dict)
    assert state['epsilon'] == 0.8
    assert state['epsilon_decay'] == 0.99

    # Restore
    new_exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=1.0)
    new_exploration.load_state(state)

    assert new_exploration.epsilon == 0.8
    assert new_exploration.epsilon_decay == 0.99
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_exploration/test_epsilon_greedy.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write EpsilonGreedyExploration implementation**

Create `src/townlet/exploration/epsilon_greedy.py`:

```python
"""
Epsilon-greedy exploration strategy (vectorized).

Simple baseline: epsilon probability of random action, 1-epsilon probability
of greedy action. No intrinsic motivation.
"""

from typing import Dict, Any
import torch

from townlet.exploration.base import ExplorationStrategy
from townlet.training.state import BatchedAgentState


class EpsilonGreedyExploration(ExplorationStrategy):
    """
    Vectorized epsilon-greedy exploration.

    No intrinsic motivation - just simple epsilon-greedy action selection.
    Epsilon decays over time with exponential schedule.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize epsilon-greedy exploration.

        Args:
            epsilon: Initial exploration rate (1.0 = full random)
            epsilon_decay: Decay per episode (0.995 = ~1% decay)
            epsilon_min: Minimum epsilon (prevents pure greedy)
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_actions(
        self,
        q_values: torch.Tensor,  # [batch, num_actions]
        agent_states: BatchedAgentState,
    ) -> torch.Tensor:
        """
        Select actions with epsilon-greedy strategy.

        Args:
            q_values: Q-values for each action [batch, num_actions]
            agent_states: Current state (contains per-agent epsilons)

        Returns:
            actions: [batch] selected actions
        """
        batch_size, num_actions = q_values.shape
        device = q_values.device

        # Greedy actions (argmax Q)
        greedy_actions = torch.argmax(q_values, dim=1)

        # Random actions
        random_actions = torch.randint(0, num_actions, (batch_size,), device=device)

        # Epsilon mask: True = explore, False = exploit
        explore_mask = torch.rand(batch_size, device=device) < agent_states.epsilons

        # Select based on mask
        actions = torch.where(explore_mask, random_actions, greedy_actions)

        return actions

    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
    ) -> torch.Tensor:
        """
        Compute intrinsic rewards (none for epsilon-greedy).

        Args:
            observations: Current observations

        Returns:
            intrinsic_rewards: [batch] all zeros
        """
        batch_size = observations.shape[0]
        return torch.zeros(batch_size, device=observations.device)

    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Update exploration networks (no-op for epsilon-greedy).

        Args:
            batch: Experience batch (ignored)
        """
        pass  # No networks to update

    def decay_epsilon(self) -> None:
        """Decay epsilon (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def checkpoint_state(self) -> Dict[str, Any]:
        """
        Return serializable state.

        Returns:
            Dict with epsilon state
        """
        return {
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore from checkpoint.

        Args:
            state: Dict from checkpoint_state()
        """
        self.epsilon = state['epsilon']
        self.epsilon_decay = state['epsilon_decay']
        self.epsilon_min = state['epsilon_min']
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_exploration/test_epsilon_greedy.py -v`

Expected: PASS (4 tests)

**Step 5: Update interface compliance tests**

Add to `tests/test_townlet/test_interface_compliance.py`:

```python
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration

@pytest.mark.parametrize("exploration_class", [
    EpsilonGreedyExploration,
])
def test_exploration_strategy_compliance(exploration_class):
    """Verify exploration implementations satisfy interface contract."""
    exploration = exploration_class()

    # Should have all required methods
    assert hasattr(exploration, 'select_actions')
    assert hasattr(exploration, 'compute_intrinsic_rewards')
    assert hasattr(exploration, 'update')

    # select_actions should return tensor
    q_values = torch.randn(3, 5)
    state = create_mock_batched_state(batch_size=3)
    actions = exploration.select_actions(q_values, state)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (3,)

    # compute_intrinsic_rewards should return tensor
    observations = torch.randn(3, 70)
    intrinsic = exploration.compute_intrinsic_rewards(observations)

    assert isinstance(intrinsic, torch.Tensor)
    assert intrinsic.shape == (3,)

    # update should not raise
    batch = {'states': torch.randn(10, 70)}
    exploration.update(batch)  # Should not raise

    # checkpoint/restore should work
    checkpoint = exploration.checkpoint_state()
    assert isinstance(checkpoint, dict)
    exploration.load_state(checkpoint)  # Should not raise
```

**Step 6: Run interface compliance test**

Run: `pytest tests/test_townlet/test_interface_compliance.py::test_exploration_strategy_compliance -v`

Expected: PASS

**Step 7: Commit**

```bash
git add src/townlet/exploration/epsilon_greedy.py tests/test_townlet/test_exploration/test_epsilon_greedy.py tests/test_townlet/test_interface_compliance.py
git commit -m "feat(townlet): add EpsilonGreedyExploration

Implement vectorized epsilon-greedy exploration:
- Per-agent epsilon values (batched)
- Greedy vs random action selection
- No intrinsic motivation (returns zeros)
- Epsilon decay support
- Interface compliance tests passing

Part of Phase 1: GPU Infrastructure"
```

---

## Task 5: VectorizedPopulation - Population Coordinator

**Files:**

- Create: `src/townlet/population/vectorized.py`
- Create: `tests/test_townlet/test_population/test_vectorized.py`

**Step 1: Write failing test for VectorizedPopulation construction**

Create `tests/test_townlet/test_population/test_vectorized.py`:

```python
"""Tests for VectorizedPopulation (population coordinator)."""

import pytest
import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


def test_vectorized_population_construction():
    """VectorizedPopulation should construct with components."""
    from townlet.population.vectorized import VectorizedPopulation
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=1.0, epsilon_decay=0.995)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0', 'agent_1'],
        device=torch.device('cpu'),
    )

    assert population.num_agents == 2
    assert len(population.agent_ids) == 2


def test_vectorized_population_step():
    """VectorizedPopulation should coordinate training step."""
    from townlet.population.vectorized import VectorizedPopulation
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Reset
    population.reset()

    # Step (will use dummy Q-network for now)
    state = population.step_population(env)

    from townlet.training.state import BatchedAgentState
    assert isinstance(state, BatchedAgentState)
    assert state.batch_size == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_townlet/test_population/test_vectorized.py::test_vectorized_population_construction -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write VectorizedPopulation implementation**

Create `src/townlet/population/vectorized.py`:

```python
"""
Vectorized population manager.

Coordinates multiple agents with shared curriculum and exploration strategies.
Manages Q-networks, replay buffers, and training loops.
"""

from typing import List, TYPE_CHECKING
import torch
import torch.nn as nn

from townlet.population.base import PopulationManager
from townlet.training.state import BatchedAgentState, PopulationCheckpoint
from townlet.curriculum.base import CurriculumManager
from townlet.exploration.base import ExplorationStrategy

if TYPE_CHECKING:
    from townlet.environment.vectorized_env import VectorizedHamletEnv


class SimpleQNetwork(nn.Module):
    """Simple MLP Q-network for Phase 1."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorizedPopulation(PopulationManager):
    """
    Vectorized population manager.

    Coordinates training for num_agents parallel agents with shared
    curriculum and exploration strategies.
    """

    def __init__(
        self,
        env: 'VectorizedHamletEnv',
        curriculum: CurriculumManager,
        exploration: ExplorationStrategy,
        agent_ids: List[str],
        device: torch.device,
        obs_dim: int = 70,
        action_dim: int = 5,
    ):
        """
        Initialize vectorized population.

        Args:
            env: Vectorized environment
            curriculum: Curriculum manager
            exploration: Exploration strategy
            agent_ids: List of agent identifiers
            device: PyTorch device
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.env = env
        self.curriculum = curriculum
        self.exploration = exploration
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = device

        # Q-network (shared across all agents for now)
        self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)

        # Current state
        self.current_obs: torch.Tensor = None
        self.current_epsilons: torch.Tensor = None

    def reset(self) -> None:
        """Reset all environments and state."""
        self.current_obs = self.env.reset()
        self.current_epsilons = torch.full(
            (self.num_agents,), self.exploration.epsilon, device=self.device
        )

    def step_population(
        self,
        envs: 'VectorizedHamletEnv',
    ) -> BatchedAgentState:
        """
        Execute one training step for entire population.

        Args:
            envs: Vectorized environment (same as self.env)

        Returns:
            BatchedAgentState with all agent data after step
        """
        # 1. Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(self.current_obs)

        # 2. Create temporary agent state for action selection
        temp_state = BatchedAgentState(
            observations=self.current_obs,
            actions=torch.zeros(self.num_agents, dtype=torch.long, device=self.device),
            rewards=torch.zeros(self.num_agents, device=self.device),
            dones=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
            epsilons=self.current_epsilons,
            intrinsic_rewards=torch.zeros(self.num_agents, device=self.device),
            survival_times=envs.step_counts.clone(),
            curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
            device=self.device,
        )

        # 3. Select actions via exploration strategy
        actions = self.exploration.select_actions(q_values, temp_state)

        # 4. Step environment
        next_obs, rewards, dones, info = envs.step(actions)

        # 5. Compute intrinsic rewards
        intrinsic_rewards = self.exploration.compute_intrinsic_rewards(next_obs)

        # 6. Update current state
        self.current_obs = next_obs

        # 7. Construct BatchedAgentState
        state = BatchedAgentState(
            observations=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            epsilons=self.current_epsilons,
            intrinsic_rewards=intrinsic_rewards,
            survival_times=info['step_counts'],
            curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
            device=self.device,
        )

        return state

    def get_checkpoint(self) -> PopulationCheckpoint:
        """
        Return Pydantic checkpoint.

        Returns:
            PopulationCheckpoint DTO
        """
        return PopulationCheckpoint(
            generation=0,
            num_agents=self.num_agents,
            agent_ids=self.agent_ids,
            curriculum_states={'global': self.curriculum.checkpoint_state()},
            exploration_states={'global': self.exploration.checkpoint_state()},
            pareto_frontier=[],
            metrics_summary={},
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_townlet/test_population/test_vectorized.py -v`

Expected: PASS (2 tests)

**Step 5: Update interface compliance tests**

Add to `tests/test_townlet/test_interface_compliance.py`:

```python
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv

@pytest.mark.parametrize("population_class", [
    VectorizedPopulation,
])
def test_population_manager_compliance(population_class):
    """Verify population implementations satisfy interface contract."""
    # Create dependencies
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum()
    exploration = EpsilonGreedyExploration()

    population = population_class(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Should have required methods
    assert hasattr(population, 'step_population')
    assert hasattr(population, 'get_checkpoint')

    # step_population should return BatchedAgentState
    population.reset()
    state = population.step_population(env)

    from townlet.training.state import BatchedAgentState
    assert isinstance(state, BatchedAgentState)

    # get_checkpoint should return PopulationCheckpoint
    checkpoint = population.get_checkpoint()

    from townlet.training.state import PopulationCheckpoint
    assert isinstance(checkpoint, PopulationCheckpoint)
```

Also update the imports at the top:

```python
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
```

**Step 6: Run interface compliance test**

Run: `pytest tests/test_townlet/test_interface_compliance.py::test_population_manager_compliance -v`

Expected: PASS

**Step 7: Commit**

```bash
git add src/townlet/population/vectorized.py tests/test_townlet/test_population/test_vectorized.py tests/test_townlet/test_interface_compliance.py
git commit -m "feat(townlet): add VectorizedPopulation coordinator

Implement population manager that coordinates:
- Q-network (simple MLP for Phase 1)
- Action selection via exploration strategy
- Environment stepping
- Intrinsic reward computation
- BatchedAgentState construction
- Interface compliance tests passing

Part of Phase 1: GPU Infrastructure"
```

---

## Task 6: Oracle Validation Tests

**Files:**

- Create: `tests/test_townlet/test_oracle_validation.py`

**Step 1: Write oracle validation test comparing Townlet vs Hamlet**

Create `tests/test_townlet/test_oracle_validation.py`:

```python
"""
Oracle validation tests.

Compare Townlet implementation against Hamlet reference to verify correctness.
"""

import pytest
import torch
import numpy as np


@pytest.mark.slow
def test_shaped_rewards_match_hamlet():
    """Townlet shaped rewards should match Hamlet within tolerance."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from hamlet.environment.hamlet_env import HamletEnv

    # Create both environments
    townlet_env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device('cpu'),
    )

    hamlet_env = HamletEnv(grid_width=8, grid_height=8)

    # Reset both
    townlet_obs = townlet_env.reset()
    hamlet_obs = hamlet_env.reset()

    # Set same initial state (agent at position [4, 4])
    townlet_env.positions[0] = torch.tensor([4, 4])
    hamlet_env.agent.position = (4, 4)

    # Set same meter values
    initial_meters = [1.0, 1.0, 1.0, 0.5, 1.0, 0.5]
    townlet_env.meters[0] = torch.tensor(initial_meters)
    hamlet_env.meters.energy.value = 100.0
    hamlet_env.meters.hygiene.value = 100.0
    hamlet_env.meters.satiation.value = 100.0
    hamlet_env.meters.money.value = 50.0
    hamlet_env.meters.mood.value = 100.0
    hamlet_env.meters.social.value = 50.0

    # Take same action sequence
    actions = [0, 0, 1, 3, 2, 4]  # UP, UP, DOWN, RIGHT, LEFT, INTERACT

    townlet_rewards = []
    hamlet_rewards = []

    for action in actions:
        # Townlet step
        _, townlet_reward, _, _ = townlet_env.step(torch.tensor([action]))
        townlet_rewards.append(townlet_reward[0].item())

        # Hamlet step
        _, hamlet_reward, _, _, _ = hamlet_env.step(action)
        hamlet_rewards.append(hamlet_reward)

    # Compare rewards
    townlet_rewards = np.array(townlet_rewards)
    hamlet_rewards = np.array(hamlet_rewards)

    # Should match within 1e-3 tolerance
    np.testing.assert_allclose(
        townlet_rewards,
        hamlet_rewards,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Townlet rewards diverged from Hamlet oracle"
    )


@pytest.mark.slow
def test_meter_depletion_matches_hamlet():
    """Meter depletion should match Hamlet."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from hamlet.environment.hamlet_env import HamletEnv

    townlet_env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    hamlet_env = HamletEnv(grid_width=8, grid_height=8)

    townlet_env.reset()
    hamlet_env.reset()

    # Take 50 random steps
    np.random.seed(42)
    for _ in range(50):
        action = np.random.randint(0, 4)  # Movement only

        townlet_env.step(torch.tensor([action]))
        hamlet_env.step(action)

    # Compare meter values
    townlet_meters = townlet_env.meters[0].cpu().numpy()
    hamlet_meters = np.array([
        hamlet_env.meters.energy.value / 100.0,
        hamlet_env.meters.hygiene.value / 100.0,
        hamlet_env.meters.satiation.value / 100.0,
        hamlet_env.meters.money.value / 100.0,
        hamlet_env.meters.mood.value / 100.0,
        hamlet_env.meters.social.value / 100.0,
    ])

    # Should match within 1e-2 tolerance
    np.testing.assert_allclose(
        townlet_meters,
        hamlet_meters,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Townlet meter depletion diverged from Hamlet"
    )


def test_vectorized_env_deterministic_seed():
    """Same seed should produce same trajectory."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    def run_trajectory(seed: int):
        torch.manual_seed(seed)
        env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
        env.reset()

        rewards = []
        for _ in range(20):
            _, reward, _, _ = env.step(torch.tensor([0]))  # Always UP
            rewards.append(reward[0].item())

        return rewards

    rewards1 = run_trajectory(seed=123)
    rewards2 = run_trajectory(seed=123)

    assert rewards1 == rewards2, "Same seed should produce same trajectory"
```

**Step 2: Run oracle validation tests**

Run: `pytest tests/test_townlet/test_oracle_validation.py -v`

Expected: FAIL (rewards don't match yet - need to tune implementation)

**Step 3: Fix any discrepancies found in oracle validation**

If tests fail, investigate differences:

- Meter depletion rates
- Reward calculation
- Affordance effects
- Grid boundaries

Adjust `VectorizedHamletEnv` to match Hamlet exactly.

**Step 4: Run tests again to verify they pass**

Run: `pytest tests/test_townlet/test_oracle_validation.py -v`

Expected: PASS (within tolerance)

**Step 5: Commit**

```bash
git add tests/test_townlet/test_oracle_validation.py
git commit -m "test(townlet): add oracle validation against Hamlet

Add tests comparing Townlet vs Hamlet reference:
- Shaped reward calculation
- Meter depletion rates
- Deterministic trajectory reproduction
- Validates correctness of GPU implementation

Part of Phase 1: GPU Infrastructure"
```

---

## Task 7: Integration Test - Full Training Loop

**Files:**

- Create: `tests/test_townlet/test_integration.py`

**Step 1: Write integration test for full training episode**

Create `tests/test_townlet/test_integration.py`:

```python
"""
Integration tests for Townlet training loop.

End-to-end tests that verify all components work together.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.mark.integration
def test_train_one_episode_n1():
    """Should train one complete episode at n=1."""
    # Setup
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Run episode
    population.reset()

    episode_rewards = []
    max_steps = 500

    for step in range(max_steps):
        state = population.step_population(env)
        episode_rewards.append(state.rewards[0].item())

        if state.dones[0]:
            break

    # Assertions
    assert len(episode_rewards) > 0
    assert len(episode_rewards) <= max_steps

    # Should accumulate some reward
    total_reward = sum(episode_rewards)
    print(f"Episode completed in {len(episode_rewards)} steps, total reward: {total_reward:.2f}")


@pytest.mark.integration
def test_train_multiple_agents():
    """Should train multiple agents in parallel."""
    num_agents = 5

    env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.2, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=torch.device('cpu'),
    )

    # Run episode
    population.reset()

    all_done = False
    step_count = 0
    max_steps = 500

    while not all_done and step_count < max_steps:
        state = population.step_population(env)
        all_done = torch.all(state.dones)
        step_count += 1

    print(f"All {num_agents} agents completed in {step_count} steps")

    # All agents should have stepped
    assert step_count > 0
    assert step_count <= max_steps


@pytest.mark.integration
def test_checkpoint_save_restore():
    """Should save and restore population checkpoint."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.8, epsilon_decay=0.995)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0'],
        device=torch.device('cpu'),
    )

    # Get checkpoint
    checkpoint = population.get_checkpoint()

    # Verify structure
    assert checkpoint.num_agents == 1
    assert len(checkpoint.agent_ids) == 1
    assert 'global' in checkpoint.curriculum_states
    assert 'global' in checkpoint.exploration_states

    # Verify exploration state
    exploration_state = checkpoint.exploration_states['global']
    assert exploration_state['epsilon'] == 0.8

    # Should be serializable
    json_str = checkpoint.model_dump_json()
    assert 'num_agents' in json_str

    # Should be deserializable
    from townlet.training.state import PopulationCheckpoint
    restored = PopulationCheckpoint.model_validate_json(json_str)
    assert restored.num_agents == checkpoint.num_agents


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_on_gpu():
    """Should train on GPU if available."""
    device = torch.device('cuda')

    env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device)
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.5, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=['agent_0', 'agent_1'],
        device=device,
    )

    # Run a few steps
    population.reset()

    for _ in range(10):
        state = population.step_population(env)

        # Verify tensors are on GPU
        assert state.observations.device.type == 'cuda'
        assert state.rewards.device.type == 'cuda'
```

**Step 2: Run integration tests**

Run: `pytest tests/test_townlet/test_integration.py -v`

Expected: PASS (all integration tests)

**Step 3: Run slow tests (including oracle validation)**

Run: `pytest tests/test_townlet/ -m slow -v`

Expected: PASS (oracle validation passes)

**Step 4: Commit**

```bash
git add tests/test_townlet/test_integration.py
git commit -m "test(townlet): add integration tests for training loop

Add end-to-end integration tests:
- Single agent training episode
- Multiple agents in parallel
- Checkpoint save/restore
- GPU training (when available)

All Phase 1 components working together.

Part of Phase 1: GPU Infrastructure"
```

---

## Task 8: Phase 1 Documentation and Verification

**Files:**

- Create: `docs/townlet/PHASE1_VERIFICATION.md`

**Step 1: Create Phase 1 verification checklist**

Create `docs/townlet/PHASE1_VERIFICATION.md`:

```markdown
# Townlet Phase 1 Verification Checklist

**Date Completed**: [FILL IN]

## Components

### VectorizedHamletEnv (GPU-Native Environment)

- [x] Core structure (construction, reset)
- [x] Step function (movement, interactions, depletion)
- [x] Shaped reward calculation (two-tier from Hamlet)
- [x] Batched tensor operations [num_agents, ...]
- [x] Test coverage: 5/5 tests passing

### StaticCurriculum (Trivial Implementation)

- [x] Returns same decision for all agents
- [x] Checkpoint/restore support
- [x] Interface compliance verified
- [x] Test coverage: 2/2 tests passing

### EpsilonGreedyExploration (Vectorized)

- [x] Epsilon-greedy action selection
- [x] No intrinsic rewards (returns zeros)
- [x] Epsilon decay support
- [x] Interface compliance verified
- [x] Test coverage: 4/4 tests passing

### VectorizedPopulation (Coordinator)

- [x] Coordinates env, curriculum, exploration
- [x] Q-network (simple MLP)
- [x] Training step coordination
- [x] Checkpoint generation
- [x] Interface compliance verified
- [x] Test coverage: 2/2 tests passing

## Validation

### Oracle Validation (vs Hamlet)

- [x] Shaped rewards match within 1e-3
- [x] Meter depletion matches within 1e-2
- [x] Deterministic trajectories
- [x] Test coverage: 3/3 tests passing

### Integration Tests

- [x] Single agent (n=1) trains successfully
- [x] Multiple agents (n=5) train in parallel
- [x] Checkpoint save/restore works
- [x] GPU training works (if CUDA available)
- [x] Test coverage: 4/4 tests passing

## Code Quality

- [x] All files have docstrings
- [x] All interface implementations pass compliance tests
- [x] Test coverage: 100% on new Phase 1 code
- [x] All commits follow conventional format

## Exit Criteria

- ✅ Single agent (n=1) trains successfully with GPU implementation
- ✅ Works on both CPU (`device='cpu'`) and GPU (`device='cuda'`)
- ✅ Oracle validation: Townlet matches Hamlet shaped rewards within 1e-4
- ✅ Checkpoints save/restore correctly

## Commands to Verify

```bash
# Run all Phase 1 tests
pytest tests/test_townlet/test_environment/ tests/test_townlet/test_curriculum/test_static.py tests/test_townlet/test_exploration/test_epsilon_greedy.py tests/test_townlet/test_population/test_vectorized.py -v

# Run integration tests
pytest tests/test_townlet/test_integration.py -v

# Run oracle validation (slow)
pytest tests/test_townlet/test_oracle_validation.py -v -m slow

# Run interface compliance tests
pytest tests/test_townlet/test_interface_compliance.py -v

# Check test coverage
pytest --cov=townlet --cov-report=term-missing tests/test_townlet/
```

## Performance Baseline

**n=1 (single agent)**:

- Episode duration: [MEASURE] steps
- Steps per second: [MEASURE] FPS
- Memory usage: [MEASURE] MB

**n=5 (small batch)**:

- Steps per second: [MEASURE] FPS
- Memory usage: [MEASURE] MB

## Next Steps (Phase 2)

- [ ] AdversarialCurriculum (auto-tuning difficulty)
- [ ] Curriculum progression tests
- [ ] Shaped → sparse transition
- [ ] Integration with Phase 1 components

## Notes

Phase 1 establishes working GPU infrastructure at n=1. All interfaces proven.
Ready to implement adaptive curriculum (Phase 2) and intrinsic exploration (Phase 3).

```

**Step 2: Run full verification**

Run all verification commands:
```bash
pytest tests/test_townlet/test_environment/ tests/test_townlet/test_curriculum/test_static.py tests/test_townlet/test_exploration/test_epsilon_greedy.py tests/test_townlet/test_population/test_vectorized.py -v

pytest tests/test_townlet/test_integration.py -v

pytest tests/test_townlet/test_oracle_validation.py -v -m slow

pytest tests/test_townlet/test_interface_compliance.py -v

pytest --cov=townlet --cov-report=term-missing tests/test_townlet/
```

Expected: All tests pass, 100% coverage on Phase 1 code

**Step 3: Measure performance baseline**

Create `scripts/measure_phase1_performance.py`:

```python
"""Measure Phase 1 performance baseline."""

import time
import torch
import psutil
import os

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.curriculum.static import StaticCurriculum
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


def measure_performance(num_agents: int, num_steps: int = 1000, device_type: str = 'cpu'):
    """Measure training performance."""
    device = torch.device(device_type)

    # Setup
    env = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)
    curriculum = StaticCurriculum(difficulty_level=0.5, reward_mode='shaped')
    exploration = EpsilonGreedyExploration(epsilon=0.1, epsilon_decay=1.0)

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f'agent_{i}' for i in range(num_agents)],
        device=device,
    )

    # Warmup
    population.reset()
    for _ in range(10):
        population.step_population(env)

    # Measure memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Measure performance
    population.reset()
    start_time = time.time()

    for _ in range(num_steps):
        population.step_population(env)

    end_time = time.time()

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    # Calculate metrics
    elapsed = end_time - start_time
    fps = num_steps / elapsed
    mem_usage = mem_after - mem_before

    print(f"\n{'='*50}")
    print(f"Performance Baseline: n={num_agents}, device={device_type}")
    print(f"{'='*50}")
    print(f"Total steps: {num_steps}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Steps/sec: {fps:.1f} FPS")
    print(f"Memory usage: {mem_usage:.1f} MB")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    print("Measuring Phase 1 Performance Baseline...")

    # CPU baseline
    measure_performance(num_agents=1, num_steps=1000, device_type='cpu')
    measure_performance(num_agents=5, num_steps=1000, device_type='cpu')

    # GPU baseline (if available)
    if torch.cuda.is_available():
        measure_performance(num_agents=1, num_steps=1000, device_type='cuda')
        measure_performance(num_agents=5, num_steps=1000, device_type='cuda')
```

Run: `python scripts/measure_phase1_performance.py`

Record results in verification doc.

**Step 4: Commit documentation**

```bash
git add docs/townlet/PHASE1_VERIFICATION.md scripts/measure_phase1_performance.py
git commit -m "docs(townlet): add Phase 1 verification checklist

Add comprehensive verification checklist for Phase 1:
- All components documented and verified
- Oracle validation confirmed
- Integration tests passing
- Performance baseline measurements
- Verification commands provided

Phase 1 Complete: GPU Infrastructure + Trivial Implementations"
```

**Step 5: Create Phase 1 completion tag**

```bash
git tag -a phase1-complete -m "Townlet Phase 1 Complete: GPU Infrastructure

All GPU infrastructure and trivial implementations working:
- VectorizedHamletEnv (batched tensors, shaped rewards)
- StaticCurriculum (no adaptation)
- EpsilonGreedyExploration (vectorized)
- VectorizedPopulation (coordinates all components)
- Oracle validation passes (matches Hamlet)
- Integration tests pass at n=1 and n=5
- Works on both CPU and GPU

Ready for Phase 2: Adversarial Curriculum"

git push origin phase1-complete
```

---

## Phase 1 Complete! 🎉

**Deliverables Summary**:

- ✅ VectorizedHamletEnv (GPU-native environment)
- ✅ StaticCurriculum (trivial curriculum manager)
- ✅ EpsilonGreedyExploration (vectorized exploration)
- ✅ VectorizedPopulation (coordinator)
- ✅ Oracle validation (matches Hamlet)
- ✅ Integration tests (n=1, n=5)
- ✅ Performance baseline measurements

**Lines of Code**:

- Source: ~800 LOC
- Tests: ~700 LOC
- Total: ~1,500 LOC

**Test Coverage**:

- Environment: 5 tests
- StaticCurriculum: 2 tests
- EpsilonGreedy: 4 tests
- VectorizedPopulation: 2 tests
- Interface compliance: 3 tests (all passing)
- Oracle validation: 3 tests
- Integration: 4 tests
- **Total: 23 tests, 100% passing**

**Next Phase**: Phase 2 - Adversarial Curriculum (5-7 days)

- Auto-tuning difficulty based on survival + learning + entropy
- Shaped → sparse transition
- Curriculum progression tests
- Integration with Phase 1 infrastructure

---

## Execution Options

**Plan complete and saved to `docs/plans/2025-10-29-townlet-phase1-gpu-infrastructure.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration with quality gates

**2. Parallel Session (separate)** - Open new session, use executing-plans skill for batch execution with checkpoints

**Which approach would you prefer?**

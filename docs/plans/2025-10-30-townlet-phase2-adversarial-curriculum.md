# Phase 2: Adversarial Curriculum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement auto-tuning curriculum that adapts difficulty based on agent performance, enabling progression from easy shaped rewards to full sparse reward challenge.

**Architecture:** AdversarialCurriculum tracks per-agent performance metrics (survival rate, learning progress, entropy) and makes multi-signal decisions to advance/retreat through 5 stages. Stage 1-4 use shaped rewards with progressive meter activation and depletion rates. Stage 5 transitions to sparse rewards (graduation). Integrates with VectorizedPopulation via CurriculumManager interface.

**Tech Stack:** PyTorch 2.0+ (performance tracking), Pydantic 2.10+ (validated decisions), Python 3.11+ (type hints), pytest (TDD)

**Estimated Duration:** 5-7 days

**Exit Criteria:**
- [ ] AdversarialCurriculum implements CurriculumManager interface
- [ ] 5-stage progression with exact specifications
- [ ] Multi-signal decision logic (survival + learning + entropy)
- [ ] 6 unit tests pass (mastery, struggle, entropy gate, no-advancement, checkpoint, sparse transition)
- [ ] End-to-end integration test: train population through full curriculum
- [ ] YAML config for fast testing (quick progression thresholds)
- [ ] Documentation complete

---

## Task 1: AdversarialCurriculum Structure + Performance Tracking

**Files:**
- Create: `src/townlet/curriculum/adversarial.py`
- Test: `tests/test_townlet/test_curriculum/test_adversarial.py`
- Reference: `src/townlet/curriculum/base.py` (interface)

**Step 1: Write the failing test for basic construction**

Create `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
"""Tests for AdversarialCurriculum."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.state import BatchedAgentState


def test_adversarial_curriculum_construction():
    """AdversarialCurriculum should initialize with stage 1 defaults."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=torch.device('cpu'),
    )

    assert curriculum.current_stage == 1
    assert curriculum.max_steps_per_episode == 500
    assert curriculum.device.type == 'cpu'

    # Stage 1 specs
    assert curriculum._get_active_meters(1) == ['energy', 'hygiene']
    assert curriculum._get_depletion_multiplier(1) == 0.2
    assert curriculum._get_reward_mode(1) == 'shaped'
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_adversarial_curriculum_construction -xvs`

Expected: FAIL with "No module named 'townlet.curriculum.adversarial'"

**Step 3: Write minimal implementation**

Create `src/townlet/curriculum/adversarial.py`:

```python
"""Adversarial curriculum that auto-tunes difficulty based on performance.

Tracks per-agent metrics (survival, learning, entropy) and adapts stage
through 5 progressive difficulty levels culminating in sparse rewards.
"""

from typing import List, Dict, Tuple
import torch
from pydantic import BaseModel, Field

from townlet.curriculum.base import CurriculumManager, CurriculumDecision
from townlet.population.state import BatchedAgentState


class StageConfig(BaseModel):
    """Configuration for a single curriculum stage."""

    stage: int
    active_meters: List[str]
    depletion_multiplier: float
    reward_mode: str  # 'shaped' or 'sparse'
    description: str


# Stage progression: 5 stages from easy shaped to full sparse
STAGE_CONFIGS = [
    StageConfig(
        stage=1,
        active_meters=['energy', 'hygiene'],
        depletion_multiplier=0.2,
        reward_mode='shaped',
        description="Stage 1: Basic needs (energy, hygiene) at 20% depletion",
    ),
    StageConfig(
        stage=2,
        active_meters=['energy', 'hygiene', 'satiation'],
        depletion_multiplier=0.5,
        reward_mode='shaped',
        description="Stage 2: Add hunger at 50% depletion",
    ),
    StageConfig(
        stage=3,
        active_meters=['energy', 'hygiene', 'satiation', 'money'],
        depletion_multiplier=0.8,
        reward_mode='shaped',
        description="Stage 3: Add money management at 80% depletion",
    ),
    StageConfig(
        stage=4,
        active_meters=['energy', 'hygiene', 'satiation', 'money', 'mood', 'social'],
        depletion_multiplier=1.0,
        reward_mode='shaped',
        description="Stage 4: All meters at full depletion",
    ),
    StageConfig(
        stage=5,
        active_meters=['energy', 'hygiene', 'satiation', 'money', 'mood', 'social'],
        depletion_multiplier=1.0,
        reward_mode='sparse',
        description="Stage 5: SPARSE REWARDS - Graduation!",
    ),
]


class PerformanceTracker:
    """Tracks per-agent performance metrics for curriculum decisions."""

    def __init__(self, num_agents: int, device: torch.device):
        self.num_agents = num_agents
        self.device = device

        # Performance metrics (per agent)
        self.episode_rewards = torch.zeros(num_agents, device=device)
        self.episode_steps = torch.zeros(num_agents, device=device)
        self.prev_avg_reward = torch.zeros(num_agents, device=device)

        # Stage tracking
        self.agent_stages = torch.ones(num_agents, dtype=torch.long, device=device)
        self.steps_at_stage = torch.zeros(num_agents, device=device)

    def update_step(self, rewards: torch.Tensor, dones: torch.Tensor):
        """Update metrics after environment step."""
        self.episode_rewards += rewards
        self.episode_steps += 1.0

        # Reset completed episodes
        self.episode_rewards = torch.where(dones, 0.0, self.episode_rewards)
        self.episode_steps = torch.where(dones, 0.0, self.episode_steps)

    def get_survival_rate(self, max_steps: int) -> torch.Tensor:
        """Calculate survival rate (0-1) for each agent."""
        return self.episode_steps / max_steps

    def get_learning_progress(self) -> torch.Tensor:
        """Calculate learning progress (reward improvement) for each agent."""
        current_avg = self.episode_rewards / torch.clamp(self.episode_steps, min=1.0)
        progress = current_avg - self.prev_avg_reward
        return progress

    def update_baseline(self):
        """Update reward baseline for learning progress calculation."""
        current_avg = self.episode_rewards / torch.clamp(self.episode_steps, min=1.0)
        self.prev_avg_reward = current_avg


class AdversarialCurriculum(CurriculumManager):
    """Auto-tuning curriculum based on survival, learning, and entropy.

    Advances agents through 5 stages when they demonstrate:
    - High survival rate (>70%)
    - Positive learning progress
    - Low entropy (<0.5) - policy convergence

    Retreats agents when they struggle:
    - Low survival rate (<30%)
    - Negative learning progress
    """

    def __init__(
        self,
        max_steps_per_episode: int = 500,
        survival_advance_threshold: float = 0.7,
        survival_retreat_threshold: float = 0.3,
        entropy_gate: float = 0.5,
        min_steps_at_stage: int = 1000,
        device: torch.device = torch.device('cpu'),
    ):
        self.max_steps_per_episode = max_steps_per_episode
        self.survival_advance_threshold = survival_advance_threshold
        self.survival_retreat_threshold = survival_retreat_threshold
        self.entropy_gate = entropy_gate
        self.min_steps_at_stage = min_steps_at_stage
        self.device = device

        self.current_stage = 1  # Start at stage 1
        self.tracker: PerformanceTracker = None  # Set when population initialized

    def initialize_population(self, num_agents: int):
        """Initialize performance tracking for population."""
        self.tracker = PerformanceTracker(num_agents, self.device)

    def _get_active_meters(self, stage: int) -> List[str]:
        """Get active meters for stage."""
        return STAGE_CONFIGS[stage - 1].active_meters

    def _get_depletion_multiplier(self, stage: int) -> float:
        """Get depletion multiplier for stage."""
        return STAGE_CONFIGS[stage - 1].depletion_multiplier

    def _get_reward_mode(self, stage: int) -> str:
        """Get reward mode for stage."""
        return STAGE_CONFIGS[stage - 1].reward_mode

    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """Generate curriculum decisions for batch of agents."""
        # For now, return same decision for all agents (will add per-agent logic in Task 2)
        stage = self.current_stage
        config = STAGE_CONFIGS[stage - 1]

        decision = CurriculumDecision(
            difficulty_level=stage,
            active_meters=config.active_meters,
            depletion_multiplier=config.depletion_multiplier,
            reward_mode=config.reward_mode,
            reason=config.description,
        )

        return [decision] * len(agent_ids)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_adversarial_curriculum_construction -xvs`

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py
git commit -m "feat(curriculum): add AdversarialCurriculum structure with 5-stage configs"
```

---

## Task 2: Stage Advancement Logic (Multi-Signal Decision)

**Files:**
- Modify: `src/townlet/curriculum/adversarial.py`
- Modify: `tests/test_townlet/test_curriculum/test_adversarial.py`

**Step 1: Write the failing test for advancement**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_advancement_when_mastery_achieved():
    """Agents should advance when survival high + learning positive + entropy low."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        entropy_gate=0.5,
        min_steps_at_stage=100,  # Low for testing
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=3)

    # Simulate mastery conditions
    curriculum.tracker.episode_steps = torch.tensor([400.0, 450.0, 420.0])  # High survival
    curriculum.tracker.episode_rewards = torch.tensor([80.0, 90.0, 85.0])  # Good rewards
    curriculum.tracker.prev_avg_reward = torch.tensor([0.1, 0.1, 0.1])  # Learning progress
    curriculum.tracker.steps_at_stage = torch.tensor([150.0, 150.0, 150.0])  # Enough steps

    # Mock agent states with low entropy (converged policy)
    agent_states = BatchedAgentState(
        observations=torch.zeros(3, 70),
        rewards=torch.zeros(3),
        dones=torch.zeros(3, dtype=torch.bool),
        epsilons=torch.tensor([0.1, 0.1, 0.1]),  # Low exploration
        curriculum_stages=torch.ones(3, dtype=torch.long),
        step_counts=torch.tensor([400, 450, 420]),
    )

    # Mock entropy calculation (will be implemented in Task 4)
    def mock_calculate_entropy(states):
        return torch.tensor([0.3, 0.3, 0.3])  # Low entropy

    curriculum._calculate_action_entropy = mock_calculate_entropy

    # Get decisions and check for advancement
    decisions = curriculum.get_batch_decisions(agent_states, ['agent_0', 'agent_1', 'agent_2'])

    # Should advance to stage 2
    assert curriculum.tracker.agent_stages[0].item() == 2
    assert decisions[0].difficulty_level == 2
    assert decisions[0].active_meters == ['energy', 'hygiene', 'satiation']
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_advancement_when_mastery_achieved -xvs`

Expected: FAIL with assertion error (stages still at 1)

**Step 3: Implement advancement logic**

Modify `src/townlet/curriculum/adversarial.py`, add method before `get_batch_decisions`:

```python
    def _should_advance(self, agent_idx: int, entropy: float) -> bool:
        """Check if agent should advance to next stage."""
        if self.tracker.agent_stages[agent_idx] >= 5:
            return False  # Already at max stage

        # Check minimum steps at stage
        if self.tracker.steps_at_stage[agent_idx] < self.min_steps_at_stage:
            return False

        # Calculate metrics
        survival_rate = self.tracker.get_survival_rate(self.max_steps_per_episode)[agent_idx]
        learning_progress = self.tracker.get_learning_progress()[agent_idx]

        # Multi-signal decision
        high_survival = survival_rate > self.survival_advance_threshold
        positive_learning = learning_progress > 0
        converged = entropy < self.entropy_gate

        return high_survival and positive_learning and converged
```

Modify `get_batch_decisions` method to implement per-agent decisions:

```python
    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """Generate curriculum decisions for batch of agents."""
        if self.tracker is None:
            raise RuntimeError("Must call initialize_population before get_batch_decisions")

        # Calculate entropy for each agent
        entropies = self._calculate_action_entropy(agent_states)

        decisions = []
        for i, agent_id in enumerate(agent_ids):
            # Check for advancement
            if self._should_advance(i, entropies[i].item()):
                self.tracker.agent_stages[i] += 1
                self.tracker.steps_at_stage[i] = 0
                self.tracker.update_baseline()

            # Get current stage
            stage = self.tracker.agent_stages[i].item()
            config = STAGE_CONFIGS[stage - 1]

            decision = CurriculumDecision(
                difficulty_level=stage,
                active_meters=config.active_meters,
                depletion_multiplier=config.depletion_multiplier,
                reward_mode=config.reward_mode,
                reason=f"{config.description} (agent {agent_id})",
            )
            decisions.append(decision)

        # Update step counter
        self.tracker.steps_at_stage += 1

        return decisions

    def _calculate_action_entropy(self, agent_states: BatchedAgentState) -> torch.Tensor:
        """Calculate action distribution entropy (stub for Task 4)."""
        # Placeholder - will implement in Task 4
        return torch.zeros(agent_states.observations.shape[0], device=self.device)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_advancement_when_mastery_achieved -xvs`

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py
git commit -m "feat(curriculum): add stage advancement logic with multi-signal decision"
```

---

## Task 3: Stage Retreat Logic (Struggle Detection)

**Files:**
- Modify: `src/townlet/curriculum/adversarial.py`
- Modify: `tests/test_townlet/test_curriculum/test_adversarial.py`

**Step 1: Write the failing test for retreat**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_retreat_when_struggling():
    """Agents should retreat when survival low or learning negative."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_retreat_threshold=0.3,
        min_steps_at_stage=100,
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=2)

    # Start at stage 3 (will retreat to 2)
    curriculum.tracker.agent_stages = torch.tensor([3, 3], dtype=torch.long)

    # Simulate struggle: low survival + negative learning
    curriculum.tracker.episode_steps = torch.tensor([100.0, 120.0])  # Low survival
    curriculum.tracker.episode_rewards = torch.tensor([-50.0, -40.0])  # Bad rewards
    curriculum.tracker.prev_avg_reward = torch.tensor([0.5, 0.5])  # Was doing better
    curriculum.tracker.steps_at_stage = torch.tensor([150.0, 150.0])  # Enough steps

    agent_states = BatchedAgentState(
        observations=torch.zeros(2, 70),
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        epsilons=torch.tensor([0.2, 0.2]),
        curriculum_stages=torch.tensor([3, 3], dtype=torch.long),
        step_counts=torch.tensor([100, 120]),
    )

    # Mock entropy (doesn't matter for retreat)
    curriculum._calculate_action_entropy = lambda s: torch.tensor([0.5, 0.5])

    decisions = curriculum.get_batch_decisions(agent_states, ['agent_0', 'agent_1'])

    # Should retreat to stage 2
    assert curriculum.tracker.agent_stages[0].item() == 2
    assert decisions[0].difficulty_level == 2
    assert decisions[0].active_meters == ['energy', 'hygiene', 'satiation']
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_retreat_when_struggling -xvs`

Expected: FAIL with assertion error (stages still at 3)

**Step 3: Implement retreat logic**

Modify `src/townlet/curriculum/adversarial.py`, add method after `_should_advance`:

```python
    def _should_retreat(self, agent_idx: int) -> bool:
        """Check if agent should retreat to previous stage."""
        if self.tracker.agent_stages[agent_idx] <= 1:
            return False  # Already at minimum stage

        # Check minimum steps at stage
        if self.tracker.steps_at_stage[agent_idx] < self.min_steps_at_stage:
            return False

        # Calculate metrics
        survival_rate = self.tracker.get_survival_rate(self.max_steps_per_episode)[agent_idx]
        learning_progress = self.tracker.get_learning_progress()[agent_idx]

        # Retreat conditions
        low_survival = survival_rate < self.survival_retreat_threshold
        negative_learning = learning_progress < 0

        return low_survival or negative_learning
```

Modify `get_batch_decisions` to check retreat before advance:

```python
    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """Generate curriculum decisions for batch of agents."""
        if self.tracker is None:
            raise RuntimeError("Must call initialize_population before get_batch_decisions")

        # Calculate entropy for each agent
        entropies = self._calculate_action_entropy(agent_states)

        decisions = []
        for i, agent_id in enumerate(agent_ids):
            # Check for retreat first (takes priority)
            if self._should_retreat(i):
                self.tracker.agent_stages[i] -= 1
                self.tracker.steps_at_stage[i] = 0
                self.tracker.update_baseline()
            # Then check for advancement
            elif self._should_advance(i, entropies[i].item()):
                self.tracker.agent_stages[i] += 1
                self.tracker.steps_at_stage[i] = 0
                self.tracker.update_baseline()

            # Get current stage
            stage = self.tracker.agent_stages[i].item()
            config = STAGE_CONFIGS[stage - 1]

            decision = CurriculumDecision(
                difficulty_level=stage,
                active_meters=config.active_meters,
                depletion_multiplier=config.depletion_multiplier,
                reward_mode=config.reward_mode,
                reason=f"{config.description} (agent {agent_id})",
            )
            decisions.append(decision)

        # Update step counter
        self.tracker.steps_at_stage += 1

        return decisions
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_retreat_when_struggling -xvs`

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py
git commit -m "feat(curriculum): add stage retreat logic for struggling agents"
```

---

## Task 4: Entropy Calculation (Action Distribution)

**Files:**
- Modify: `src/townlet/curriculum/adversarial.py`
- Modify: `tests/test_townlet/test_curriculum/test_adversarial.py`

**Step 1: Write the failing test for entropy gate**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_entropy_gate_prevents_premature_advancement():
    """High entropy (random policy) should prevent advancement even with good metrics."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        entropy_gate=0.5,
        min_steps_at_stage=100,
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=2)

    # Good survival and learning, but need Q-values to calculate real entropy
    curriculum.tracker.episode_steps = torch.tensor([400.0, 450.0])
    curriculum.tracker.episode_rewards = torch.tensor([80.0, 90.0])
    curriculum.tracker.prev_avg_reward = torch.tensor([0.1, 0.1])
    curriculum.tracker.steps_at_stage = torch.tensor([150.0, 150.0])

    # High entropy: uniform Q-values (random policy)
    q_values = torch.ones(2, 5) * 1.0  # All actions equal value

    agent_states = BatchedAgentState(
        observations=torch.zeros(2, 70),
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        epsilons=torch.tensor([0.1, 0.1]),
        curriculum_stages=torch.ones(2, dtype=torch.long),
        step_counts=torch.tensor([400, 450]),
    )

    # Need to pass Q-values for entropy calculation
    decisions = curriculum.get_batch_decisions_with_qvalues(
        agent_states,
        ['agent_0', 'agent_1'],
        q_values,
    )

    # Should NOT advance (entropy too high)
    assert curriculum.tracker.agent_stages[0].item() == 1
    assert decisions[0].difficulty_level == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_entropy_gate_prevents_premature_advancement -xvs`

Expected: FAIL with "no attribute 'get_batch_decisions_with_qvalues'"

**Step 3: Implement entropy calculation**

Modify `src/townlet/curriculum/adversarial.py`, replace stub `_calculate_action_entropy`:

```python
    def _calculate_action_entropy(self, q_values: torch.Tensor) -> torch.Tensor:
        """Calculate action distribution entropy from Q-values.

        Higher entropy = more uniform distribution (exploring)
        Lower entropy = peaked distribution (converged policy)

        Args:
            q_values: [batch_size, num_actions] Q-values

        Returns:
            [batch_size] entropy values (0 = deterministic, ~1.6 = uniform for 5 actions)
        """
        # Convert Q-values to probabilities (softmax with temperature=1)
        probs = torch.softmax(q_values, dim=-1)

        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add epsilon for numerical stability
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Normalize to [0, 1] range (max entropy for 5 actions = log(5) ≈ 1.609)
        normalized_entropy = entropy / torch.log(torch.tensor(q_values.shape[-1], dtype=torch.float32))

        return normalized_entropy
```

Add new method that accepts Q-values:

```python
    def get_batch_decisions_with_qvalues(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
        q_values: torch.Tensor,
    ) -> List[CurriculumDecision]:
        """Generate curriculum decisions with Q-values for entropy calculation.

        This is the main entry point when called from VectorizedPopulation.
        """
        if self.tracker is None:
            raise RuntimeError("Must call initialize_population before get_batch_decisions")

        # Calculate entropy from Q-values
        entropies = self._calculate_action_entropy(q_values)

        decisions = []
        for i, agent_id in enumerate(agent_ids):
            # Check for retreat first (takes priority)
            if self._should_retreat(i):
                self.tracker.agent_stages[i] -= 1
                self.tracker.steps_at_stage[i] = 0
                self.tracker.update_baseline()
            # Then check for advancement
            elif self._should_advance(i, entropies[i].item()):
                self.tracker.agent_stages[i] += 1
                self.tracker.steps_at_stage[i] = 0
                self.tracker.update_baseline()

            # Get current stage
            stage = self.tracker.agent_stages[i].item()
            config = STAGE_CONFIGS[stage - 1]

            decision = CurriculumDecision(
                difficulty_level=stage,
                active_meters=config.active_meters,
                depletion_multiplier=config.depletion_multiplier,
                reward_mode=config.reward_mode,
                reason=f"{config.description} (agent {agent_id})",
            )
            decisions.append(decision)

        # Update step counter
        self.tracker.steps_at_stage += 1

        return decisions

    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """Generate curriculum decisions without Q-values (for testing).

        Uses zero entropy (assumes converged policy).
        """
        # Create dummy Q-values with peaked distribution (low entropy)
        num_agents = len(agent_ids)
        q_values = torch.zeros(num_agents, 5, device=self.device)
        q_values[:, 0] = 10.0  # Make action 0 dominant

        return self.get_batch_decisions_with_qvalues(agent_states, agent_ids, q_values)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_entropy_gate_prevents_premature_advancement -xvs`

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py
git commit -m "feat(curriculum): add entropy calculation for advancement gating"
```

---

## Task 5: Integration with VectorizedPopulation

**Files:**
- Modify: `src/townlet/population/vectorized.py`
- Modify: `tests/test_townlet/test_integration.py`

**Step 1: Write the failing test for integration**

Add to `tests/test_townlet/test_integration.py`:

```python
def test_integration_with_adversarial_curriculum():
    """VectorizedPopulation should work with AdversarialCurriculum."""
    from townlet.curriculum.adversarial import AdversarialCurriculum

    device = torch.device('cpu')
    num_agents = 3

    # Create curriculum
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        min_steps_at_stage=10,  # Low for fast testing
        device=device,
    )

    # Create population with curriculum
    population = VectorizedPopulation(
        num_agents=num_agents,
        state_dim=70,
        action_dim=5,
        grid_size=8,
        curriculum=curriculum,
        exploration=EpsilonGreedyExploration(initial_epsilon=0.5),
        device=device,
    )

    # Create environment
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)
    population.reset(envs)

    # Run 50 steps
    for _ in range(50):
        agent_state = population.step_population(envs)

        # Verify decisions are being made
        assert len(population.current_curriculum_decisions) == num_agents
        assert all(d.difficulty_level >= 1 for d in population.current_curriculum_decisions)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_integration.py::test_integration_with_adversarial_curriculum -xvs`

Expected: FAIL (VectorizedPopulation doesn't pass Q-values to curriculum)

**Step 3: Modify VectorizedPopulation to pass Q-values**

Modify `src/townlet/population/vectorized.py`, update `step_population` method:

```python
    def step_population(self, envs: 'VectorizedHamletEnv') -> BatchedAgentState:
        """Execute one step for entire population."""
        # 1. Get Q-values from network
        q_values = self.q_network(self.current_obs)

        # 2. Get curriculum decisions (pass Q-values if curriculum supports it)
        if hasattr(self.curriculum, 'get_batch_decisions_with_qvalues'):
            # Create temporary state for curriculum decision
            temp_state = BatchedAgentState(
                observations=self.current_obs,
                rewards=torch.zeros(self.num_agents, device=self.device),
                dones=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
                epsilons=self.epsilons,
                curriculum_stages=torch.tensor(
                    [d.difficulty_level for d in self.current_curriculum_decisions],
                    device=self.device,
                ),
                step_counts=self.step_counts,
            )

            self.current_curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(
                temp_state,
                self.agent_ids,
                q_values,
            )
        else:
            # Fallback for curricula without Q-value support (like StaticCurriculum)
            temp_state = BatchedAgentState(
                observations=self.current_obs,
                rewards=torch.zeros(self.num_agents, device=self.device),
                dones=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
                epsilons=self.epsilons,
                curriculum_stages=torch.tensor(
                    [d.difficulty_level for d in self.current_curriculum_decisions],
                    device=self.device,
                ),
                step_counts=self.step_counts,
            )

            self.current_curriculum_decisions = self.curriculum.get_batch_decisions(
                temp_state,
                self.agent_ids,
            )

        # 3. Select actions using exploration strategy
        actions = self.exploration.select_actions(q_values, temp_state)

        # 4. Step environment
        next_obs, rewards, dones, info = envs.step(actions)

        # 5. Compute intrinsic rewards (curriculum bonus)
        intrinsic_rewards = self._compute_intrinsic_rewards(temp_state)
        total_rewards = rewards + intrinsic_rewards

        # 6. Update internal state
        self.current_obs = next_obs
        self.step_counts += 1

        # Reset completed episodes
        if dones.any():
            reset_indices = torch.where(dones)[0]
            for idx in reset_indices:
                self.step_counts[idx] = 0

        # 7. Return batched state
        return BatchedAgentState(
            observations=next_obs,
            rewards=total_rewards,
            dones=dones,
            epsilons=self.epsilons,
            curriculum_stages=torch.tensor(
                [d.difficulty_level for d in self.current_curriculum_decisions],
                device=self.device,
            ),
            step_counts=self.step_counts,
        )
```

Add method to update curriculum tracker after environment step:

```python
    def update_curriculum_tracker(self, rewards: torch.Tensor, dones: torch.Tensor):
        """Update curriculum performance tracking after step.

        Call this after step_population if using AdversarialCurriculum.
        """
        if hasattr(self.curriculum, 'tracker') and self.curriculum.tracker is not None:
            self.curriculum.tracker.update_step(rewards, dones)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_integration.py::test_integration_with_adversarial_curriculum -xvs`

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/population/vectorized.py tests/test_townlet/test_integration.py
git commit -m "feat(population): integrate AdversarialCurriculum with Q-value passing"
```

---

## Task 6: Unit Tests (Checkpoint & Sparse Transition)

**Files:**
- Modify: `tests/test_townlet/test_curriculum/test_adversarial.py`

**Step 1: Write test for no advancement when conditions not met**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_no_advancement_without_all_conditions():
    """Should not advance unless ALL conditions met."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        survival_advance_threshold=0.7,
        entropy_gate=0.5,
        min_steps_at_stage=100,
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=1)

    curriculum.tracker.agent_stages = torch.tensor([1], dtype=torch.long)
    curriculum.tracker.steps_at_stage = torch.tensor([150.0])  # Enough steps

    # Test case 1: Good survival + learning, but high entropy
    curriculum.tracker.episode_steps = torch.tensor([400.0])
    curriculum.tracker.episode_rewards = torch.tensor([80.0])
    curriculum.tracker.prev_avg_reward = torch.tensor([0.1])

    q_values_random = torch.ones(1, 5)  # High entropy
    agent_states = BatchedAgentState(
        observations=torch.zeros(1, 70),
        rewards=torch.zeros(1),
        dones=torch.zeros(1, dtype=torch.bool),
        epsilons=torch.tensor([0.1]),
        curriculum_stages=torch.ones(1, dtype=torch.long),
        step_counts=torch.tensor([400]),
    )

    decisions = curriculum.get_batch_decisions_with_qvalues(
        agent_states, ['agent_0'], q_values_random
    )
    assert curriculum.tracker.agent_stages[0].item() == 1  # No advancement

    # Test case 2: Good survival + low entropy, but negative learning
    curriculum.tracker.prev_avg_reward = torch.tensor([1.0])  # Was better before
    q_values_peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])  # Low entropy

    decisions = curriculum.get_batch_decisions_with_qvalues(
        agent_states, ['agent_0'], q_values_peaked
    )
    assert curriculum.tracker.agent_stages[0].item() == 1  # No advancement

    # Test case 3: All conditions met NOW
    curriculum.tracker.prev_avg_reward = torch.tensor([0.1])  # Reset baseline
    decisions = curriculum.get_batch_decisions_with_qvalues(
        agent_states, ['agent_0'], q_values_peaked
    )
    assert curriculum.tracker.agent_stages[0].item() == 2  # NOW advances
```

**Step 2: Run test**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_no_advancement_without_all_conditions -xvs`

Expected: PASS (logic already implemented)

**Step 3: Write test for checkpoint/restore**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_checkpoint_restore_preserves_state():
    """Curriculum state should be checkpointable."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=2)

    # Set some state
    curriculum.tracker.agent_stages = torch.tensor([2, 3], dtype=torch.long)
    curriculum.tracker.episode_rewards = torch.tensor([50.0, 60.0])
    curriculum.tracker.prev_avg_reward = torch.tensor([0.5, 0.6])
    curriculum.tracker.steps_at_stage = torch.tensor([200.0, 300.0])

    # Checkpoint
    checkpoint = curriculum.state_dict()

    # Modify state
    curriculum.tracker.agent_stages = torch.tensor([1, 1], dtype=torch.long)
    curriculum.tracker.episode_rewards = torch.zeros(2)

    # Restore
    curriculum.load_state_dict(checkpoint)

    # Verify restoration
    assert torch.equal(curriculum.tracker.agent_stages, torch.tensor([2, 3], dtype=torch.long))
    assert torch.equal(curriculum.tracker.episode_rewards, torch.tensor([50.0, 60.0]))
    assert torch.equal(curriculum.tracker.prev_avg_reward, torch.tensor([0.5, 0.6]))
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_checkpoint_restore_preserves_state -xvs`

Expected: FAIL (methods not implemented)

**Step 5: Implement checkpoint methods**

Add to `src/townlet/curriculum/adversarial.py`:

```python
    def state_dict(self) -> Dict:
        """Get curriculum state for checkpointing."""
        if self.tracker is None:
            return {}

        return {
            'agent_stages': self.tracker.agent_stages.cpu(),
            'episode_rewards': self.tracker.episode_rewards.cpu(),
            'episode_steps': self.tracker.episode_steps.cpu(),
            'prev_avg_reward': self.tracker.prev_avg_reward.cpu(),
            'steps_at_stage': self.tracker.steps_at_stage.cpu(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Restore curriculum state from checkpoint."""
        if self.tracker is None:
            raise RuntimeError("Must initialize_population before loading state")

        self.tracker.agent_stages = state_dict['agent_stages'].to(self.device)
        self.tracker.episode_rewards = state_dict['episode_rewards'].to(self.device)
        self.tracker.episode_steps = state_dict['episode_steps'].to(self.device)
        self.tracker.prev_avg_reward = state_dict['prev_avg_reward'].to(self.device)
        self.tracker.steps_at_stage = state_dict['steps_at_stage'].to(self.device)
```

Add import at top:

```python
from typing import List, Dict, Tuple
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_checkpoint_restore_preserves_state -xvs`

Expected: PASS

**Step 7: Write test for sparse reward transition**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_sparse_reward_transition_at_stage_5():
    """Stage 5 should switch to sparse rewards."""
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=500,
        device=torch.device('cpu'),
    )
    curriculum.initialize_population(num_agents=1)

    # Jump to stage 5
    curriculum.tracker.agent_stages = torch.tensor([5], dtype=torch.long)

    agent_states = BatchedAgentState(
        observations=torch.zeros(1, 70),
        rewards=torch.zeros(1),
        dones=torch.zeros(1, dtype=torch.bool),
        epsilons=torch.tensor([0.1]),
        curriculum_stages=torch.tensor([5], dtype=torch.long),
        step_counts=torch.tensor([100]),
    )

    decisions = curriculum.get_batch_decisions(agent_states, ['agent_0'])

    assert decisions[0].difficulty_level == 5
    assert decisions[0].reward_mode == 'sparse'
    assert decisions[0].active_meters == ['energy', 'hygiene', 'satiation', 'money', 'mood', 'social']
    assert decisions[0].depletion_multiplier == 1.0
    assert 'SPARSE' in decisions[0].reason
```

**Step 8: Run test**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_sparse_reward_transition_at_stage_5 -xvs`

Expected: PASS (already implemented in stage configs)

**Step 9: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py
git commit -m "test(curriculum): add checkpoint and sparse transition tests"
```

---

## Task 7: End-to-End Integration Test (Full Curriculum Progression)

**Files:**
- Create: `tests/test_townlet/test_curriculum_progression.py`

**Step 1: Write end-to-end test**

Create `tests/test_townlet/test_curriculum_progression.py`:

```python
"""End-to-end test for curriculum progression through all stages."""

import pytest
import torch
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.vectorized import VectorizedPopulation
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration


def test_curriculum_progression_through_stages():
    """Train population through full 5-stage curriculum progression.

    This is a smoke test, not a full training run. We verify:
    1. Population can train with adversarial curriculum
    2. Stages advance when conditions are met
    3. System reaches stage 5 (sparse rewards)
    4. No crashes or errors
    """
    device = torch.device('cpu')
    num_agents = 5
    max_steps = 200  # Short episodes for fast testing

    # Create curriculum with aggressive advancement for testing
    curriculum = AdversarialCurriculum(
        max_steps_per_episode=max_steps,
        survival_advance_threshold=0.5,  # Lower threshold for testing
        survival_retreat_threshold=0.2,
        entropy_gate=0.6,  # Higher gate for testing
        min_steps_at_stage=50,  # Low for fast progression
        device=device,
    )

    # Create population
    population = VectorizedPopulation(
        num_agents=num_agents,
        state_dim=70,
        action_dim=5,
        grid_size=8,
        curriculum=curriculum,
        exploration=EpsilonGreedyExploration(
            initial_epsilon=1.0,
            min_epsilon=0.1,
            decay_rate=0.995,
        ),
        device=device,
    )

    # Create environment
    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)
    population.reset(envs)

    # Track progression
    max_stage_reached = 1
    stage_history = []

    # Run training for limited steps (not full training)
    num_episodes = 50
    for episode in range(num_episodes):
        envs.reset()
        population.reset(envs)

        for step in range(max_steps):
            # Step population
            agent_state = population.step_population(envs)

            # Update curriculum tracker
            population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

            # Track max stage
            current_stages = curriculum.tracker.agent_stages
            max_stage_reached = max(max_stage_reached, current_stages.max().item())

            if step % 50 == 0:
                stage_history.append(current_stages.clone())

    # Verification
    print(f"\nMax stage reached: {max_stage_reached}")
    print(f"Final stages: {curriculum.tracker.agent_stages}")

    # Should have advanced beyond stage 1 (even if not to stage 5)
    assert max_stage_reached > 1, "Curriculum should advance beyond stage 1"

    # Should have some variation in stages (not all agents identical)
    final_stages = curriculum.tracker.agent_stages
    assert len(torch.unique(final_stages)) >= 1, "Should have stage diversity"

    # Should not crash with sparse rewards if reached stage 5
    if max_stage_reached >= 5:
        # Verify sparse reward mode is active
        decisions = curriculum.get_batch_decisions(
            agent_state,
            [f'agent_{i}' for i in range(num_agents)],
        )
        sparse_decisions = [d for d in decisions if d.reward_mode == 'sparse']
        assert len(sparse_decisions) > 0, "Stage 5 should use sparse rewards"


@pytest.mark.slow
def test_long_curriculum_progression():
    """Longer test to verify reaching stage 5 (marked as slow).

    Run with: pytest -m slow
    """
    device = torch.device('cpu')
    num_agents = 3
    max_steps = 300

    curriculum = AdversarialCurriculum(
        max_steps_per_episode=max_steps,
        survival_advance_threshold=0.6,
        survival_retreat_threshold=0.25,
        entropy_gate=0.55,
        min_steps_at_stage=100,
        device=device,
    )

    population = VectorizedPopulation(
        num_agents=num_agents,
        state_dim=70,
        action_dim=5,
        grid_size=8,
        curriculum=curriculum,
        exploration=EpsilonGreedyExploration(initial_epsilon=1.0, decay_rate=0.99),
        device=device,
    )

    envs = VectorizedHamletEnv(num_agents=num_agents, grid_size=8, device=device)
    population.reset(envs)

    max_stage_reached = 1

    # Longer training
    num_episodes = 200
    for episode in range(num_episodes):
        envs.reset()
        population.reset(envs)

        for step in range(max_steps):
            agent_state = population.step_population(envs)
            population.update_curriculum_tracker(agent_state.rewards, agent_state.dones)

            max_stage_reached = max(
                max_stage_reached,
                curriculum.tracker.agent_stages.max().item()
            )

        if episode % 50 == 0:
            print(f"Episode {episode}: Max stage = {max_stage_reached}, "
                  f"Stages = {curriculum.tracker.agent_stages.tolist()}")

    print(f"\nFinal max stage: {max_stage_reached}")

    # With 200 episodes, should reach at least stage 3
    assert max_stage_reached >= 3, f"Should reach stage 3+ in 200 episodes, got {max_stage_reached}"
```

**Step 2: Run short test**

Run: `uv run pytest tests/test_townlet/test_curriculum_progression.py::test_curriculum_progression_through_stages -xvs`

Expected: PASS (may take 30-60 seconds)

**Step 3: Commit**

```bash
git add tests/test_townlet/test_curriculum_progression.py
git commit -m "test(curriculum): add end-to-end progression test"
```

---

## Task 8: YAML Config + Documentation

**Files:**
- Create: `configs/curriculum_quick_test.yaml`
- Create: `docs/townlet/ADVERSARIAL_CURRICULUM.md`
- Modify: `src/townlet/curriculum/adversarial.py` (add from_config)

**Step 1: Write test for YAML config loading**

Add to `tests/test_townlet/test_curriculum/test_adversarial.py`:

```python
def test_curriculum_from_yaml_config():
    """Should load curriculum config from YAML."""
    import yaml
    import tempfile
    from pathlib import Path

    config = {
        'curriculum': {
            'type': 'adversarial',
            'max_steps_per_episode': 400,
            'survival_advance_threshold': 0.75,
            'survival_retreat_threshold': 0.25,
            'entropy_gate': 0.45,
            'min_steps_at_stage': 500,
        }
    }

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Load curriculum from config
        curriculum = AdversarialCurriculum.from_yaml(config_path)

        assert curriculum.max_steps_per_episode == 400
        assert curriculum.survival_advance_threshold == 0.75
        assert curriculum.survival_retreat_threshold == 0.25
        assert curriculum.entropy_gate == 0.45
        assert curriculum.min_steps_at_stage == 500
    finally:
        Path(config_path).unlink()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_curriculum_from_yaml_config -xvs`

Expected: FAIL (method not implemented)

**Step 3: Implement from_yaml method**

Add to `src/townlet/curriculum/adversarial.py`, add import at top:

```python
import yaml
from pathlib import Path
```

Add classmethod:

```python
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AdversarialCurriculum':
        """Load curriculum from YAML config file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configured AdversarialCurriculum instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        curriculum_config = config.get('curriculum', {})

        # Extract device config
        device_str = curriculum_config.get('device', 'cpu')
        device = torch.device(device_str)

        return cls(
            max_steps_per_episode=curriculum_config.get('max_steps_per_episode', 500),
            survival_advance_threshold=curriculum_config.get('survival_advance_threshold', 0.7),
            survival_retreat_threshold=curriculum_config.get('survival_retreat_threshold', 0.3),
            entropy_gate=curriculum_config.get('entropy_gate', 0.5),
            min_steps_at_stage=curriculum_config.get('min_steps_at_stage', 1000),
            device=device,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py::test_curriculum_from_yaml_config -xvs`

Expected: PASS

**Step 5: Create quick test config**

Create `configs/curriculum_quick_test.yaml`:

```yaml
# Quick progression config for testing adversarial curriculum
# Lower thresholds and fewer steps for fast advancement

experiment:
  name: curriculum_quick_test
  description: Fast curriculum progression for testing

curriculum:
  type: adversarial
  max_steps_per_episode: 300
  survival_advance_threshold: 0.5  # Lower than default (0.7)
  survival_retreat_threshold: 0.2   # Lower than default (0.3)
  entropy_gate: 0.6                 # Higher than default (0.5) - easier to pass
  min_steps_at_stage: 50            # Much lower than default (1000)
  device: cpu

population:
  num_agents: 5
  state_dim: 70
  action_dim: 5
  grid_size: 8

exploration:
  type: epsilon_greedy
  initial_epsilon: 1.0
  min_epsilon: 0.1
  decay_rate: 0.995

training:
  num_episodes: 100
  device: cpu
```

**Step 6: Create documentation**

Create `docs/townlet/ADVERSARIAL_CURRICULUM.md`:

```markdown
# Adversarial Curriculum

**Status:** Phase 2 Complete ✅

## Overview

AdversarialCurriculum is an auto-tuning difficulty system that progressively challenges agents through 5 stages, from easy shaped rewards to full sparse reward challenge. Unlike StaticCurriculum (Phase 1 baseline), AdversarialCurriculum adapts difficulty based on per-agent performance metrics.

## Architecture

**Multi-Signal Decision Logic:**
```python
def should_advance(agent):
    survival_rate = agent.steps / max_steps
    learning_progress = current_reward - baseline_reward
    entropy = calculate_entropy(q_values)

    return (
        survival_rate > 0.7 AND
        learning_progress > 0 AND
        entropy < 0.5
    )
```

**Components:**
- `AdversarialCurriculum`: Main curriculum manager
- `PerformanceTracker`: Per-agent metrics (survival, learning, entropy)
- `StageConfig`: Stage specifications (meters, depletion, rewards)

## 5-Stage Progression

| Stage | Active Meters | Depletion | Reward Mode | Description |
|-------|--------------|-----------|-------------|-------------|
| 1 | energy, hygiene | 0.2x | shaped | Basic survival needs |
| 2 | +satiation | 0.5x | shaped | Add hunger management |
| 3 | +money | 0.8x | shaped | Add economic planning |
| 4 | +mood, social | 1.0x | shaped | Full complexity |
| 5 | all 6 meters | 1.0x | **sparse** | Graduation! |

**Key insight:** Stage 5 is the only sparse reward stage. Stages 1-4 provide dense gradient signals to learn basic skills before the final challenge.

## Decision Metrics

### 1. Survival Rate
```python
survival_rate = episode_steps / max_steps_per_episode
```

**Thresholds:**
- Advance: > 0.7 (surviving 70%+ of episode)
- Retreat: < 0.3 (dying early)

### 2. Learning Progress
```python
learning_progress = current_avg_reward - prev_avg_reward
```

**Thresholds:**
- Advance: > 0 (improving)
- Retreat: < 0 (regressing)

### 3. Action Entropy
```python
entropy = -sum(p * log(p)) / log(num_actions)
```

**Threshold:**
- Advance gate: < 0.5 (converged policy)
- High entropy: Still exploring randomly

**Why entropy matters:** Prevents premature advancement when agent is still exploring. Only advance when policy has converged (low entropy).

## Usage

### Basic Usage

```python
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.population.vectorized import VectorizedPopulation

# Create curriculum
curriculum = AdversarialCurriculum(
    max_steps_per_episode=500,
    survival_advance_threshold=0.7,
    survival_retreat_threshold=0.3,
    entropy_gate=0.5,
    min_steps_at_stage=1000,
    device=torch.device('cuda'),
)

# Create population with curriculum
population = VectorizedPopulation(
    num_agents=32,
    curriculum=curriculum,
    # ...
)

# Training loop
for episode in range(num_episodes):
    envs.reset()
    population.reset(envs)

    for step in range(max_steps):
        agent_state = population.step_population(envs)

        # IMPORTANT: Update curriculum tracker
        population.update_curriculum_tracker(
            agent_state.rewards,
            agent_state.dones,
        )
```

### YAML Configuration

```yaml
curriculum:
  type: adversarial
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
  device: cuda
```

Load from config:
```python
curriculum = AdversarialCurriculum.from_yaml('configs/my_config.yaml')
```

### Quick Testing Config

For fast iteration during development, use `configs/curriculum_quick_test.yaml`:
- Lower thresholds (advance at 50% survival instead of 70%)
- Fewer steps required (50 instead of 1000)
- Results in rapid progression through stages for testing

## Checkpointing

Save/restore curriculum state:

```python
# Checkpoint
checkpoint = {
    'curriculum': curriculum.state_dict(),
    'population': population.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Restore
checkpoint = torch.load('checkpoint.pt')
curriculum.load_state_dict(checkpoint['curriculum'])
population.load_state_dict(checkpoint['population'])
```

**Curriculum state includes:**
- Agent stages
- Episode rewards/steps
- Reward baselines
- Steps at current stage

## Integration with VectorizedPopulation

VectorizedPopulation automatically:
1. Passes Q-values to curriculum for entropy calculation
2. Calls `update_curriculum_tracker()` after each step
3. Provides curriculum decisions to environment

**No manual intervention needed** - population handles all integration.

## Testing

Run full test suite:
```bash
uv run pytest tests/test_townlet/test_curriculum/test_adversarial.py -v
```

Run end-to-end progression test:
```bash
uv run pytest tests/test_townlet/test_curriculum_progression.py -v
```

Run long progression test (slow):
```bash
uv run pytest tests/test_townlet/test_curriculum_progression.py -m slow -v
```

## Expected Behavior

**Typical progression timeline** (with default thresholds):
- **Episodes 0-50:** Stage 1 (learning basic movement + bed/shower)
- **Episodes 50-150:** Stage 2 (adding fridge management)
- **Episodes 150-300:** Stage 3 (learning job + money)
- **Episodes 300-500:** Stage 4 (mood + social complexity)
- **Episodes 500+:** Stage 5 (sparse reward challenge)

**Individual variation:** Agents progress at different rates. Some may advance faster, others may retreat temporarily when struggling.

## Design Rationale

**Why 5 stages?**
- Gradual complexity increase prevents overwhelming agents
- Each stage introduces 1-2 new concepts
- Shaped rewards (stages 1-4) build foundational skills
- Sparse rewards (stage 5) test true understanding

**Why per-agent progression?**
- Population diversity: Some agents explore different strategies
- Robust learning: Faster learners don't wait for slower ones
- Better curriculum signal: More data points for tuning

**Why entropy gating?**
- Prevents premature advancement during random exploration
- Ensures policy convergence before increasing difficulty
- Reduces regression after advancement

## Common Pitfalls

❌ **Forgetting to call `update_curriculum_tracker()`**
- Metrics won't update → no advancement
- Always call after `step_population()`

❌ **Setting min_steps_at_stage too low**
- Agents advance before learning → immediate retreat
- Use 1000+ for real training, 50-100 for testing only

❌ **Using wrong reward thresholds**
- Too high: Never advance (stuck at stage 1)
- Too low: Advance prematurely → fail → retreat loop

❌ **Not saving curriculum state in checkpoints**
- Training resumes from stage 1 → wasted time
- Always include curriculum.state_dict() in checkpoints

## Future Enhancements (Phase 3+)

- **Adaptive thresholds:** Auto-tune advancement criteria
- **Population-level decisions:** Consider population distribution
- **Curriculum rollback:** Revert entire population when regression detected
- **Multi-objective rewards:** Balance survival + exploration + efficiency
```

**Step 7: Run all tests to verify**

Run: `uv run pytest tests/test_townlet/test_curriculum/ -v`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/townlet/curriculum/adversarial.py tests/test_townlet/test_curriculum/test_adversarial.py configs/curriculum_quick_test.yaml docs/townlet/ADVERSARIAL_CURRICULUM.md
git commit -m "feat(curriculum): add YAML config support and documentation"
```

---

## Verification

After completing all tasks, run full verification:

```bash
# All curriculum tests
uv run pytest tests/test_townlet/test_curriculum/ -v

# Integration tests
uv run pytest tests/test_townlet/test_integration.py -v

# End-to-end progression
uv run pytest tests/test_townlet/test_curriculum_progression.py -v

# Full test suite
uv run pytest tests/test_townlet/ -v
```

**Exit criteria checklist:**
- [ ] All 6 unit tests pass (mastery, struggle, entropy, no-advance, checkpoint, sparse)
- [ ] Integration test with VectorizedPopulation passes
- [ ] End-to-end progression test passes
- [ ] YAML config loading works
- [ ] Documentation complete
- [ ] Code committed to feature branch

**Expected test count:** 6 unit tests + 1 integration + 2 progression = 9 new tests

---

## Notes for Engineer

**Key implementation details:**

1. **Entropy calculation:** Uses softmax(Q-values) to get action distribution, then calculates Shannon entropy normalized to [0, 1]

2. **Decision priority:** Retreat takes priority over advancement (check retreat first)

3. **Performance tracking:** PerformanceTracker updates every step, but decisions only made at episode boundaries (when conditions met)

4. **Q-value integration:** VectorizedPopulation passes Q-values via `get_batch_decisions_with_qvalues()`. Fallback `get_batch_decisions()` uses peaked distribution (low entropy) for testing.

5. **Stage configs:** Immutable STAGE_CONFIGS list defines all 5 stages. Don't modify these unless design changes.

6. **Testing strategy:** Unit tests mock metrics, integration tests use real environment, progression test runs limited training.

**Common mistakes to avoid:**

- Forgetting to call `initialize_population()` before using tracker
- Not passing Q-values from population (entropy will be wrong)
- Setting min_steps_at_stage too low in production (use 1000+)
- Not updating tracker after environment step

**TDD workflow:**
1. Write failing test
2. Run test (verify it fails correctly)
3. Implement minimal code
4. Run test (verify it passes)
5. Commit immediately

**Questions?** Check `docs/townlet/ADVERSARIAL_CURRICULUM.md` for usage examples.

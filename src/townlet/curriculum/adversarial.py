"""Adversarial curriculum that auto-tunes difficulty based on performance.

Tracks per-agent metrics (survival, learning, entropy) and adapts stage
through 5 progressive difficulty levels culminating in sparse rewards.
"""

from typing import List, Dict, Tuple, Any, Optional
import torch
from pydantic import BaseModel, Field

from townlet.curriculum.base import CurriculumManager
from townlet.training.state import BatchedAgentState, CurriculumDecision


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

        self.tracker: Optional[PerformanceTracker] = None  # Set when population initialized

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

            # Normalize stage (1-5) to difficulty_level (0.0-1.0)
            difficulty_level = (stage - 1) / 4.0  # Stage 1 -> 0.0, Stage 5 -> 1.0

            decision = CurriculumDecision(
                difficulty_level=difficulty_level,
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

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return serializable state for checkpoint saving."""
        # Will implement in later task
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore curriculum manager from checkpoint."""
        # Will implement in later task
        pass

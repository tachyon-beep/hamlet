"""Adversarial curriculum that auto-tunes difficulty based on performance.

Tracks per-agent metrics (survival, learning, entropy) and adapts stage
through 5 progressive difficulty levels culminating in sparse rewards.
"""

from typing import Any

import torch
import yaml
from pydantic import BaseModel

from townlet.curriculum.base import CurriculumManager
from townlet.training.state import BatchedAgentState, CurriculumDecision


class StageConfig(BaseModel):
    """Configuration for a single curriculum stage."""

    stage: int
    active_meters: list[str]
    depletion_multiplier: float
    reward_mode: str  # 'shaped' or 'sparse'
    description: str


# Stage progression: 5 stages from easy shaped to full sparse
STAGE_CONFIGS = [
    StageConfig(
        stage=1,
        active_meters=["energy", "hygiene"],
        depletion_multiplier=0.2,
        reward_mode="shaped",
        description="Stage 1: Basic needs (energy, hygiene) at 20% depletion",
    ),
    StageConfig(
        stage=2,
        active_meters=["energy", "hygiene", "satiation"],
        depletion_multiplier=0.5,
        reward_mode="shaped",
        description="Stage 2: Add hunger at 50% depletion",
    ),
    StageConfig(
        stage=3,
        active_meters=["energy", "hygiene", "satiation", "money"],
        depletion_multiplier=0.8,
        reward_mode="shaped",
        description="Stage 3: Add money management at 80% depletion",
    ),
    StageConfig(
        stage=4,
        active_meters=["energy", "hygiene", "satiation", "money", "mood", "social"],
        depletion_multiplier=1.0,
        reward_mode="shaped",
        description="Stage 4: All meters at full depletion",
    ),
    StageConfig(
        stage=5,
        active_meters=["energy", "hygiene", "satiation", "money", "mood", "social"],
        depletion_multiplier=1.0,
        reward_mode="sparse",
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
        self.steps_at_stage += 1.0

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
        device: torch.device = torch.device("cpu"),
    ):
        self.max_steps_per_episode = max_steps_per_episode
        self.survival_advance_threshold = survival_advance_threshold
        self.survival_retreat_threshold = survival_retreat_threshold
        self.entropy_gate = entropy_gate
        self.min_steps_at_stage = min_steps_at_stage
        self.device = device

        self.tracker: PerformanceTracker | None = None  # Set when population initialized

    @classmethod
    def from_yaml(cls, config_path: str) -> "AdversarialCurriculum":
        """Load curriculum from YAML config file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configured AdversarialCurriculum instance
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        curriculum_config = config.get("curriculum", {})

        # Extract device config
        device_str = curriculum_config.get("device", "cpu")
        device = torch.device(device_str)

        return cls(
            max_steps_per_episode=curriculum_config.get("max_steps_per_episode", 500),
            survival_advance_threshold=curriculum_config.get("survival_advance_threshold", 0.7),
            survival_retreat_threshold=curriculum_config.get("survival_retreat_threshold", 0.3),
            entropy_gate=curriculum_config.get("entropy_gate", 0.5),
            min_steps_at_stage=curriculum_config.get("min_steps_at_stage", 1000),
            device=device,
        )

    def initialize_population(self, num_agents: int):
        """Initialize performance tracking for population."""
        self.tracker = PerformanceTracker(num_agents, self.device)

    def _get_active_meters(self, stage: int) -> list[str]:
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
        # Bounds checking
        if not (0 <= agent_idx < self.tracker.num_agents):
            raise ValueError(f"Invalid agent_idx: {agent_idx}, must be in range [0, {self.tracker.num_agents})")

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

    def _should_retreat(self, agent_idx: int) -> bool:
        """Check if agent should retreat to previous stage."""
        # Bounds checking
        if not (0 <= agent_idx < self.tracker.num_agents):
            raise ValueError(f"Invalid agent_idx: {agent_idx}, must be in range [0, {self.tracker.num_agents})")

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

    def get_batch_decisions_with_qvalues(
        self,
        agent_states: BatchedAgentState,
        agent_ids: list[str],
        q_values: torch.Tensor,
    ) -> list[CurriculumDecision]:
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
                # Update baseline for retreating agent only
                current_avg_i = self.tracker.episode_rewards[i] / torch.clamp(self.tracker.episode_steps[i], min=1.0)
                self.tracker.prev_avg_reward[i] = current_avg_i
            # Then check for advancement
            elif self._should_advance(i, entropies[i].item()):
                self.tracker.agent_stages[i] += 1
                self.tracker.steps_at_stage[i] = 0
                # Update baseline for advancing agent only (not all agents)
                current_avg_i = self.tracker.episode_rewards[i] / torch.clamp(self.tracker.episode_steps[i], min=1.0)
                self.tracker.prev_avg_reward[i] = current_avg_i

            # Get current stage
            stage = self.tracker.agent_stages[i].item()
            config = STAGE_CONFIGS[stage - 1]

            # Normalize stage (1-5) to difficulty_level (0.0-1.0)
            # This maps Stage 1 -> 0.0 (easiest), Stage 5 -> 1.0 (hardest sparse rewards)
            difficulty_level = (stage - 1) / 4.0

            decision = CurriculumDecision(
                difficulty_level=difficulty_level,
                active_meters=config.active_meters,
                depletion_multiplier=config.depletion_multiplier,
                reward_mode=config.reward_mode,
                reason=f"{config.description} (agent {agent_id})",
            )
            decisions.append(decision)

        # NOTE: Per-agent step counting should be handled by PerformanceTracker.update_step()
        # which is called after each environment step. Removed global increment that was
        # incorrectly updating all agents simultaneously.

        return decisions

    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: list[str],
    ) -> list[CurriculumDecision]:
        """Generate curriculum decisions without Q-values (for testing).

        Uses zero entropy (assumes converged policy).
        """
        # Create dummy Q-values with peaked distribution (low entropy)
        num_agents = len(agent_ids)
        q_values = torch.zeros(num_agents, 5, device=self.device)
        q_values[:, 0] = 10.0  # Make action 0 dominant

        return self.get_batch_decisions_with_qvalues(agent_states, agent_ids, q_values)

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
        num_actions = torch.tensor(q_values.shape[-1], dtype=torch.float32, device=q_values.device)
        normalized_entropy = entropy / torch.log(num_actions)

        return normalized_entropy

    def checkpoint_state(self) -> dict[str, Any]:
        """Return serializable state for checkpoint saving."""
        if self.tracker is None:
            return {}

        return {
            "agent_stages": self.tracker.agent_stages.cpu(),
            "episode_rewards": self.tracker.episode_rewards.cpu(),
            "episode_steps": self.tracker.episode_steps.cpu(),
            "prev_avg_reward": self.tracker.prev_avg_reward.cpu(),
            "steps_at_stage": self.tracker.steps_at_stage.cpu(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore curriculum manager from checkpoint."""
        if self.tracker is None:
            raise RuntimeError("Must initialize_population before loading state")

        self.tracker.agent_stages = state["agent_stages"].to(self.device)
        self.tracker.episode_rewards = state["episode_rewards"].to(self.device)
        self.tracker.episode_steps = state["episode_steps"].to(self.device)
        self.tracker.prev_avg_reward = state["prev_avg_reward"].to(self.device)
        self.tracker.steps_at_stage = state["steps_at_stage"].to(self.device)

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Alias for load_state() for API consistency."""
        self.load_state(state)

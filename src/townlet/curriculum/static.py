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

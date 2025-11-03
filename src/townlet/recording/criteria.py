"""
Recording criteria evaluator.

Determines which episodes should be recorded based on multiple independent criteria.
Uses OR logic: any criterion can trigger recording.
"""

from typing import Optional, Tuple
from collections import deque, defaultdict
from townlet.recording.data_structures import EpisodeMetadata


class RecordingCriteria:
    """Evaluates whether episodes should be recorded based on configured criteria.

    Supports multiple criteria types:
    - periodic: Record every N episodes
    - stage_transitions: Record before/after curriculum stage transitions
    - performance: Record top/bottom percentile episodes
    - stage_boundaries: Record first/last N episodes at each stage

    Uses OR logic: any enabled criterion can trigger recording.
    """

    def __init__(self, config: dict, curriculum=None, database=None):
        """Initialize recording criteria evaluator.

        Args:
            config: Recording configuration with "criteria" section
            curriculum: Optional curriculum for stage transition prediction
            database: Optional database for performance history queries
        """
        self.config = config
        self.curriculum = curriculum
        self.database = database

        # Extract criteria configs
        criteria_config = config.get("criteria", {})
        self.periodic_config = criteria_config.get("periodic", {})
        self.stage_transitions_config = criteria_config.get("stage_transitions", {})
        self.performance_config = criteria_config.get("performance", {})
        self.stage_boundaries_config = criteria_config.get("stage_boundaries", {})

        # State tracking for stage transitions
        self.last_stage: Optional[int] = None
        self.transition_episodes: set[int] = set()

        # State tracking for performance criterion
        window_size = self.performance_config.get("window", 100)
        self.episode_history: deque[EpisodeMetadata] = deque(maxlen=window_size)

        # State tracking for stage boundaries
        self.stage_episode_counts: dict[int, int] = defaultdict(int)

    def should_record(self, metadata: EpisodeMetadata) -> Tuple[bool, str]:
        """Determine if episode should be recorded.

        Checks all enabled criteria in order. Returns True on first match.

        Args:
            metadata: Episode metadata to evaluate

        Returns:
            (should_record, reason) tuple:
                - should_record: True if any criterion matched
                - reason: String describing which criterion triggered (empty if none)
        """
        episode_id = metadata.episode_id
        stage = metadata.curriculum_stage

        # Detect stage transitions
        if self.last_stage is not None and stage != self.last_stage:
            self._mark_transition(episode_id)
        self.last_stage = stage

        # Update history
        self.episode_history.append(metadata)
        self.stage_episode_counts[stage] += 1

        # Check periodic criterion
        result = self._check_periodic(metadata)
        if result:
            return result

        # Check stage transitions criterion
        result = self._check_stage_transitions(metadata)
        if result:
            return result

        # Check performance criterion
        result = self._check_performance(metadata)
        if result:
            return result

        # Check stage boundaries criterion
        result = self._check_stage_boundaries(metadata)
        if result:
            return result

        # No criteria matched
        return False, ""

    def _check_periodic(self, metadata: EpisodeMetadata) -> Optional[Tuple[bool, str]]:
        """Check if periodic criterion is met.

        Records every N episodes where N is the configured interval.
        """
        if not self.periodic_config.get("enabled", False):
            return None

        interval = self.periodic_config.get("interval", 100)
        if metadata.episode_id % interval == 0:
            return True, f"periodic_{interval}"

        return None

    def _check_stage_transitions(self, metadata: EpisodeMetadata) -> Optional[Tuple[bool, str]]:
        """Check if stage transition criterion is met.

        Records episodes before and after stage transitions.
        """
        if not self.stage_transitions_config.get("enabled", False):
            return None

        if self.curriculum is None:
            return None

        episode_id = metadata.episode_id
        stage = metadata.curriculum_stage

        # Get transition configuration
        record_before = self.stage_transitions_config.get("record_before", 5)
        record_after = self.stage_transitions_config.get("record_after", 10)

        # Check if we're in a transition window (before or after)
        for trans_ep in self.transition_episodes:
            # Before transition: trans_ep - record_before <= episode_id < trans_ep
            if trans_ep - record_before <= episode_id < trans_ep:
                return True, f"before_transition_{trans_ep}"

            # After transition: trans_ep <= episode_id < trans_ep + record_after
            if trans_ep <= episode_id < trans_ep + record_after:
                return True, f"after_transition_{trans_ep}"

        # Check if transition is likely soon (for pre-recording)
        stage_info = self.curriculum.get_stage_info(agent_idx=0)
        if stage_info.get("likely_transition_soon", False):
            return True, f"stage_{stage}_pre_transition"

        return None

    def _check_performance(self, metadata: EpisodeMetadata) -> Optional[Tuple[bool, str]]:
        """Check if performance criterion is met.

        Records episodes with rewards in top or bottom percentile of recent history.
        """
        if not self.performance_config.get("enabled", False):
            return None

        # Need minimum history
        window_size = self.performance_config.get("window", 100)
        if len(self.episode_history) < max(10, window_size // 10):
            return None

        # Get percentile thresholds
        top_pct = self.performance_config.get("top_percent", 1.0)
        bottom_pct = self.performance_config.get("bottom_percent", 1.0)

        # Calculate thresholds from history
        rewards = [ep.total_reward for ep in self.episode_history]
        current_reward = metadata.total_reward

        # Check top percentile
        if top_pct > 0:
            # Top 10% means 90th percentile
            top_threshold_percentile = 100 - top_pct
            sorted_rewards = sorted(rewards)
            threshold_idx = int(len(sorted_rewards) * top_threshold_percentile / 100.0)
            top_threshold = sorted_rewards[min(threshold_idx, len(sorted_rewards) - 1)]

            if current_reward >= top_threshold:
                return True, f"top_{top_pct}pct"

        # Check bottom percentile
        if bottom_pct > 0:
            sorted_rewards = sorted(rewards)
            threshold_idx = int(len(sorted_rewards) * bottom_pct / 100.0)
            bottom_threshold = sorted_rewards[min(threshold_idx, len(sorted_rewards) - 1)]

            if current_reward <= bottom_threshold:
                return True, f"bottom_{bottom_pct}pct"

        return None

    def _check_stage_boundaries(self, metadata: EpisodeMetadata) -> Optional[Tuple[bool, str]]:
        """Check if stage boundaries criterion is met.

        Records first N and last N episodes at each curriculum stage.
        """
        if not self.stage_boundaries_config.get("enabled", False):
            return None

        if self.curriculum is None:
            return None

        stage = metadata.curriculum_stage
        stage_count = self.stage_episode_counts[stage]

        # Check first N episodes at this stage
        first_n = self.stage_boundaries_config.get("first_n", 10)
        if stage_count <= first_n:
            return True, f"stage_{stage}_first_{stage_count}"

        # Check last N episodes (using curriculum transition prediction)
        last_n = self.stage_boundaries_config.get("last_n", 10)
        stage_info = self.curriculum.get_stage_info(agent_idx=0)
        if stage_info.get("likely_transition_soon", False):
            return True, f"stage_{stage}_pre_transition"

        return None

    def _mark_transition(self, episode_id: int):
        """Mark an episode as a stage transition point."""
        self.transition_episodes.add(episode_id)

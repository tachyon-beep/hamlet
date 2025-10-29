"""
Training configuration for Hamlet.

Centralizes all hyperparameters and training settings.
Supports YAML-based configuration loading for experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    """
    Configuration for training runs.

    Centralizes all hyperparameters for easy experimentation.
    """

    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 64
    learning_starts: int = 1000  # Steps before learning begins

    # Agent parameters
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    # DQN specific
    target_update_frequency: int = 100  # Episodes between target network updates
    replay_buffer_size: int = 10000

    # Checkpointing
    save_frequency: int = 100  # Episodes between checkpoints
    checkpoint_dir: str = "checkpoints/"

    # Logging
    log_frequency: int = 10  # Episodes between logging
    metrics_file: str = "training_metrics.json"

    # Environment
    grid_size: int = 8
    render_mode: str = None  # None, "human", "rgb_array"


@dataclass
class EnvironmentConfig:
    """
    Configuration for environment setup.

    Defines grid size, affordance placement, meter parameters, etc.
    """

    # Grid
    grid_width: int = 8
    grid_height: int = 8

    # Reward configuration
    reward_mode: str = "shaped"  # 'shaped' (dense) or 'sparse'
    sparse_survival_reward: float = 0.1  # Per-step survival bonus in sparse mode
    sparse_healthy_meter_threshold: float = 0.75  # Threshold for optional healthy bonus
    sparse_healthy_meter_bonus: float = 0.05  # Bonus when key meters stay healthy
    sparse_terminal_reward_success: float = 100.0  # Not yet used (no success condition)
    sparse_terminal_reward_failure: float = -100.0  # Failure penalty in sparse mode

    # Initial meter values
    initial_energy: float = 100.0
    initial_hygiene: float = 100.0
    initial_satiation: float = 100.0
    initial_money: float = 50.0
    initial_mood: float = 100.0
    initial_social: float = 50.0

    # Depletion rates
    energy_depletion: float = 0.5
    hygiene_depletion: float = 0.3
    satiation_depletion: float = 0.4
    money_depletion: float = 0.0
    mood_depletion: float = 0.1
    social_depletion: float = 0.6

    # Meter bounds / interactions
    money_min: float = -100.0
    mood_social_penalty: float = 5.0  # Max additional mood drop per step when social is 0
    mood_social_threshold: float = 0.3  # Below this social level, apply mood penalty

    # Affordance positions (will be set during environment creation)
    affordance_positions: dict = None

    def __post_init__(self):
        """Set default affordance positions if not provided."""
        if not self.reward_mode:
            self.reward_mode = "shaped"
        self.reward_mode = self.reward_mode.lower()
        if self.reward_mode not in {"shaped", "sparse"}:
            raise ValueError(
                f"Invalid reward_mode '{self.reward_mode}'. Expected 'shaped' or 'sparse'."
            )

        if self.affordance_positions is None:
            # Default positions - can be overridden
            # Spatial clustering: Home zone, Work zone, distant Bar and Recreation
            self.affordance_positions = {
                # HOME ZONE (top-left cluster - close but not adjacent)
                "Bed": (1, 1),        # Sleep
                "Shower": (2, 2),     # Clean (diagonal from bed)
                "HomeMeal": (1, 3),   # Cheap healthy food at home

                # WORK ZONE (bottom-right cluster)
                "FastFood": (5, 6),   # Expensive convenience food near work
                "Job": (6, 6),        # Work (pays less if tired/dirty)
                "Gym": (7, 3),        # Mood recovery near work/bar corridor

                # DISTANT REWARDS (requires commitment!)
                "Bar": (7, 0),        # TOP-RIGHT: Social + food + mood boost
                "Recreation": (0, 7), # BOTTOM-LEFT: Relaxation
            }


@dataclass
class AgentConfig:
    """
    Configuration for a single agent.

    Supports multiple agents with different configurations and architectures.
    """

    agent_id: str = "agent_0"
    algorithm: str = "dqn"  # dqn, ppo, a2c, etc.
    state_dim: int = 72
    action_dim: int = 5
    learning_rate: float = 0.00025  # Reduced from 1e-3 for stability (Atari DQN standard)
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    device: str = "auto"  # auto, cpu, cuda
    network_type: str = "qnetwork"  # 'qnetwork', 'dueling', 'spatial', 'spatial_dueling'
    grid_size: int = 8  # For spatial networks


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment tracking (MLflow).
    """

    name: str = "hamlet_experiment"
    description: str = ""
    run_name: Optional[str] = None  # Auto-generated if None
    tracking_uri: str = "mlruns"  # Local directory or remote server


@dataclass
class MetricsConfig:
    """
    Configuration for metrics tracking.
    """

    tensorboard: bool = True
    tensorboard_dir: str = "runs"
    database: bool = True
    database_path: str = "metrics.db"
    replay_storage: bool = True
    replay_dir: str = "replays"
    replay_sample_rate: float = 0.1  # Fraction of episodes to save
    live_broadcast: bool = False  # Stream to visualization


@dataclass
class FullConfig:
    """
    Complete configuration for training run.

    Combines all config components for easy YAML loading.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agents: List[AgentConfig] = field(default_factory=lambda: [AgentConfig()])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> "FullConfig":
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML config file

        Returns:
            FullConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse each section
        experiment = ExperimentConfig(**config_dict.get('experiment', {}))
        environment = EnvironmentConfig(**config_dict.get('environment', {}))

        # Parse agents list
        agents_list = config_dict.get('agents', [{}])
        agents = [AgentConfig(**agent_config) for agent_config in agents_list]

        training = TrainingConfig(**config_dict.get('training', {}))
        metrics = MetricsConfig(**config_dict.get('metrics', {}))

        return cls(
            experiment=experiment,
            environment=environment,
            agents=agents,
            training=training,
            metrics=metrics
        )

    def to_yaml(self, filepath: str):
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to save YAML config
        """
        config_dict = {
            'experiment': asdict(self.experiment),
            'environment': asdict(self.environment),
            'agents': [asdict(agent) for agent in self.agents],
            'training': asdict(self.training),
            'metrics': asdict(self.metrics),
        }

        # Convert tuples to lists for YAML compatibility
        if config_dict['environment']['affordance_positions']:
            config_dict['environment']['affordance_positions'] = {
                k: list(v) for k, v in config_dict['environment']['affordance_positions'].items()
            }

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment': asdict(self.experiment),
            'environment': asdict(self.environment),
            'agents': [asdict(agent) for agent in self.agents],
            'training': asdict(self.training),
            'metrics': asdict(self.metrics),
        }

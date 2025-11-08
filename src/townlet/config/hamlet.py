"""HamletConfig - Master configuration DTO for complete training setup.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design: Composes all section DTOs (training, environment, population, curriculum,
exploration) and performs cross-config validation to ensure internal consistency.

This is the single entry point for loading complete training configurations.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from townlet.config.affordance import AffordanceConfig, load_affordances_config
from townlet.config.bar import BarConfig, load_bars_config
from townlet.config.cascade import CascadeConfig, load_cascades_config
from townlet.config.cues import CuesConfig, load_cues_config
from townlet.config.curriculum import CurriculumConfig, load_curriculum_config
from townlet.config.environment import TrainingEnvironmentConfig, load_environment_config
from townlet.config.exploration import ExplorationConfig, load_exploration_config
from townlet.config.population import PopulationConfig, load_population_config
from townlet.config.training import TrainingConfig, load_training_config
from townlet.substrate.config import SubstrateConfig, load_substrate_config

logger = logging.getLogger(__name__)


class HamletConfig(BaseModel):
    """Complete HAMLET training configuration (master DTO).

    Composes all section configs and validates cross-config consistency.
    This is the single source of truth for a training run.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.

    Philosophy: If it affects the universe, it's in the config. No exceptions.

    Example:
        >>> config = HamletConfig.load(Path("configs/L0_0_minimal"))
        >>> print(f"Training on {config.training.device} for {config.training.max_episodes} episodes")
        Training on cuda for 500 episodes
    """

    # Section configs (ALL REQUIRED)
    training: TrainingConfig = Field(description="Training hyperparameters (device, episodes, epsilon, batch size)")
    environment: TrainingEnvironmentConfig = Field(description="Environment parameters (grid, observability, affordances, energy costs)")
    population: PopulationConfig = Field(description="Population parameters (agents, learning rate, replay buffer, network type)")
    curriculum: CurriculumConfig = Field(description="Curriculum parameters (episode length, advancement thresholds)")
    exploration: ExplorationConfig = Field(description="Exploration parameters (RND, intrinsic motivation, annealing)")
    bars: tuple[BarConfig, ...] = Field(description="Meter definitions loaded from bars.yaml")
    cascades: tuple[CascadeConfig, ...] = Field(description="Cascade relationships loaded from cascades.yaml")
    affordances: tuple[AffordanceConfig, ...] = Field(description="Affordance definitions loaded from affordances.yaml")
    substrate: SubstrateConfig = Field(description="Spatial substrate configuration (grid, continuous, aspatial)")
    cues: CuesConfig | None = Field(default=None, description="Optional public cues configuration (cues.yaml)")

    @model_validator(mode="after")
    def validate_batch_size_vs_buffer(self) -> "HamletConfig":
        """Ensure batch_size <= replay_buffer_capacity.

        Can't sample more transitions than buffer holds.
        """
        if self.training.batch_size > self.population.replay_buffer_capacity:
            raise ValueError(
                f"training.batch_size ({self.training.batch_size}) cannot exceed "
                f"population.replay_buffer_capacity ({self.population.replay_buffer_capacity}). "
                f"Cannot sample {self.training.batch_size} transitions from buffer "
                f"that only holds {self.population.replay_buffer_capacity} transitions.\n\n"
                f"Fix: Either reduce batch_size or increase replay_buffer_capacity."
            )
        return self

    @model_validator(mode="after")
    def validate_recurrent_network_consistency(self) -> "HamletConfig":
        """Warn if network_type doesn't match observability mode.

        NOTE: This is a HINT, not enforcement. Operator may intentionally
        use recurrent networks with full observability for experimentation.

        Follows permissive semantics: allow unusual combinations but warn.
        """
        if self.population.network_type == "recurrent" and not self.environment.partial_observability:
            logger.warning(
                "Using recurrent network (LSTM) with full observability. "
                "Recurrent networks are typically for POMDP (partial_observability=true). "
                "This configuration is allowed but unusual. "
                "Typical combinations:\n"
                "  - Simple network + Full obs (L0, L0.5, L1)\n"
                "  - Recurrent network + POMDP (L2)\n"
                "This may be intentional for your experiment."
            )

        if self.population.network_type == "simple" and self.environment.partial_observability:
            logger.warning(
                "Using simple network (MLP) with partial observability (POMDP). "
                "POMDP typically requires recurrent networks (LSTM) to build memory. "
                "Simple networks are memoryless and may struggle with POMDP. "
                "This configuration is allowed but may perform poorly. "
                "Consider: network_type='recurrent' for POMDP environments."
            )

        return self

    @model_validator(mode="after")
    def validate_grid_capacity(self) -> "HamletConfig":
        """Warn if grid may be too small for agents + affordances.

        NOTE: This is a HINT for operator convenience, not strict enforcement.
        Operators may intentionally create crowded grids for challenge.
        """
        grid_cells = self.environment.grid_size**2
        num_agents = self.population.num_agents

        # Estimate affordance count
        if self.environment.enabled_affordances is None:
            num_affordances = len(self.affordances)
        else:
            num_affordances = len(self.environment.enabled_affordances)

        total_entities = num_agents + num_affordances

        if total_entities > grid_cells:
            logger.warning(
                f"Grid capacity warning: {self.environment.grid_size}×{self.environment.grid_size} grid "
                f"({grid_cells} cells) may not fit {num_agents} agents + "
                f"{num_affordances} affordances ({total_entities} entities total). "
                f"Entities may overlap or fail to place. "
                f"This may be intentional for high-density experiments."
            )
        elif total_entities > grid_cells * 0.8:
            logger.warning(
                f"Grid capacity warning: {self.environment.grid_size}×{self.environment.grid_size} grid "
                f"is {total_entities}/{grid_cells} ({total_entities/grid_cells*100:.0f}%) full. "
                f"High density may reduce agent mobility. "
                f"This may be intentional for your experiment."
            )

        return self

    @classmethod
    def load(cls, config_dir: Path, training_config_path: Path | None = None) -> "HamletConfig":
        """Load complete HAMLET configuration from directory.

        This is the preferred method for loading configs. Reads all sections
        from training.yaml and composes into validated HamletConfig. The training
        section can optionally be supplied from an explicit path to support CLI
        overrides (e.g., `--config /tmp/custom_training.yaml`).

        Args:
            config_dir: Directory containing training.yaml (e.g., configs/L0_0_minimal)
            training_config_path: Optional explicit training.yaml path overriding config_dir/training.yaml

        Returns:
            Validated HamletConfig with all sections loaded

        Raises:
            FileNotFoundError: If training.yaml not found
            ValueError: If any section validation fails

        Example:
            >>> config = HamletConfig.load(Path("configs/L0_0_minimal"))
            >>> print(f"Grid: {config.environment.grid_size}×{config.environment.grid_size}")
            Grid: 3×3
        """
        training = load_training_config(config_dir, training_config_path=training_config_path)
        environment = load_environment_config(config_dir)
        population = load_population_config(config_dir)
        curriculum = load_curriculum_config(config_dir)
        exploration = load_exploration_config(config_dir)
        bars = tuple(load_bars_config(config_dir))
        cascades = tuple(load_cascades_config(config_dir))
        affordances = tuple(load_affordances_config(config_dir))
        substrate = load_substrate_config(config_dir / "substrate.yaml")

        cues = None
        cues_path = config_dir / "cues.yaml"
        if cues_path.exists():
            cues = load_cues_config(cues_path)

        return cls(
            training=training,
            environment=environment,
            population=population,
            curriculum=curriculum,
            exploration=exploration,
            bars=bars,
            cascades=cascades,
            affordances=affordances,
            substrate=substrate,
            cues=cues,
        )

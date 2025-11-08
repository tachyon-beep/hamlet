"""Configuration DTOs for UNIVERSE_AS_CODE validation.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.

Design Decision (TASK-003):
- Renamed DTOs to avoid conflicts with existing cascade_config/affordance_config
- TrainingEnvironmentConfig (not EnvironmentConfig) - for grid_size, vision_range
- Dropped Bar/Cascade/Affordance DTOs - use existing cascade_config versions
- Added ExplorationConfig - found in all config packs but missing from original plan

Naming Strategy:
- Training-specific configs use "Training" prefix to distinguish from game mechanics
- Example: TrainingEnvironmentConfig vs cascade_config.EnvironmentConfig
- Prevents import conflicts and clarifies purpose
"""

from pathlib import Path

# Version tracking for schema evolution
CONFIG_SCHEMA_VERSION = "1.0.0"

# Import DTOs for convenient access
from townlet.config.training import TrainingConfig, load_training_config
from townlet.config.environment import TrainingEnvironmentConfig, load_environment_config

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "TrainingConfig",
    "load_training_config",
    "TrainingEnvironmentConfig",
    "load_environment_config",
]

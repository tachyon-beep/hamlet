"""Tests for DAC Engine."""

import torch

from townlet.config.drive_as_code import (
    DriveAsCodeConfig,
    ExtrinsicStrategyConfig,
    IntrinsicStrategyConfig,
)
from townlet.environment.dac_engine import DACEngine
from townlet.vfs.registry import VariableRegistry
from townlet.vfs.schema import VariableDef


class TestDACEngineInit:
    """Test DACEngine initialization."""

    def test_dac_engine_initializes(self):
        """DACEngine initializes with minimal config."""
        device = torch.device("cpu")
        num_agents = 4

        # Create minimal VFS registry
        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    lifetime="episode",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        # Create minimal DAC config
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy"],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="none",
                base_weight=0.0,
            ),
        )

        # Initialize engine
        engine = DACEngine(
            dac_config=dac_config,
            vfs_registry=vfs_registry,
            device=device,
            num_agents=num_agents,
        )

        assert engine is not None
        assert engine.device == device
        assert engine.num_agents == num_agents

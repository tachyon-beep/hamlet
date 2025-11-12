"""Tests for DAC Engine."""

import torch

from townlet.config.drive_as_code import (
    BarBonusConfig,
    DriveAsCodeConfig,
    ExtrinsicStrategyConfig,
    IntrinsicStrategyConfig,
    ModifierConfig,
    RangeConfig,
    VariableBonusConfig,
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


class TestModifierCompilation:
    """Test modifier compilation and evaluation."""

    def test_compile_bar_modifier(self):
        """Modifier with bar source compiles correctly."""
        device = torch.device("cpu")
        num_agents = 4

        # Minimal VFS
        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        # DAC with energy crisis modifier
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "energy_crisis": ModifierConfig(
                    bar="energy",
                    ranges=[
                        RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                        RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=1.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Verify modifier was compiled
        assert "energy_crisis" in engine.modifiers
        assert callable(engine.modifiers["energy_crisis"])

    def test_evaluate_bar_modifier_crisis_range(self):
        """Bar modifier returns correct multiplier for crisis range."""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "energy_crisis": ModifierConfig(
                    bar="energy",
                    ranges=[
                        RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                        RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=1.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Create meter values with 2 agents in crisis (< 0.3), 2 normal (>= 0.3)
        meters = torch.tensor([[0.1], [0.2], [0.5], [0.8]], device=device)  # [4, 1] energy only

        # Evaluate modifier
        multipliers = engine.modifiers["energy_crisis"](meters)

        # Should return 0.0 for crisis agents, 1.0 for normal agents
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
        assert torch.allclose(multipliers, expected)

    def test_evaluate_bar_modifier_normal_range(self):
        """Bar modifier returns correct multiplier for normal range."""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "energy_crisis": ModifierConfig(
                    bar="energy",
                    ranges=[
                        RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                        RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=1.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # All agents in normal range
        meters = torch.tensor([[0.4], [0.6], [0.8], [1.0]], device=device)

        multipliers = engine.modifiers["energy_crisis"](meters)

        # All should return 1.0
        expected = torch.ones(num_agents, device=device)
        assert torch.allclose(multipliers, expected)

    def test_compile_vfs_variable_modifier(self):
        """Modifier with VFS variable source compiles correctly."""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="social_score",
                    scope="agent",
                    type="scalar",
                    default=0.5,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "social_boost": ModifierConfig(
                    variable="social_score",  # VFS variable, not bar
                    ranges=[
                        RangeConfig(name="low", min=0.0, max=0.5, multiplier=0.5),
                        RangeConfig(name="high", min=0.5, max=1.0, multiplier=1.5),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=1.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        assert "social_boost" in engine.modifiers
        assert callable(engine.modifiers["social_boost"])


class TestExtrinsicStrategies:
    """Test extrinsic reward strategy compilation."""

    def test_multiplicative_strategy(self):
        """Multiplicative strategy: reward = base * bar1 * bar2 * ..."""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=2.0,
                bars=["energy"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Create meter values: [0.5, 0.8, 1.0, 0.0]
        meters = torch.tensor([[0.5], [0.8], [1.0], [0.0]], device=device)
        dones = torch.tensor([False, False, False, True], device=device)

        # Calculate extrinsic rewards
        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base * energy (2.0 * [0.5, 0.8, 1.0, 0.0])
        # Dead agents get 0.0
        expected = torch.tensor([1.0, 1.6, 2.0, 0.0], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_multiplicative_multiple_bars(self):
        """Multiplicative with multiple bars: reward = base * energy * health"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.8], [1.0, 1.0], [0.2, 0.5], [0.0, 1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 1.0 * energy * health
        expected = torch.tensor([0.4, 1.0, 0.1, 0.0], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_constant_base_with_shaped_bonus(self):
        """Constant base + shaped bonus: reward = base + sum(bonuses)"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="constant_base_with_shaped_bonus",
                base_reward=1.0,
                bars=["energy"],  # Define meter ordering
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Energy values: [0.0, 0.5, 1.0, 0.25]
        meters = torch.tensor([[0.0], [0.5], [1.0], [0.25]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base_reward + scale * (energy - center)
        # [1.0 + 0.5 * (0.0 - 0.5), 1.0 + 0.5 * (0.5 - 0.5), 1.0 + 0.5 * (1.0 - 0.5), 1.0 + 0.5 * (0.25 - 0.5)]
        # = [0.75, 1.0, 1.25, 0.875]
        expected = torch.tensor([0.75, 1.0, 1.25, 0.875], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_constant_base_multiple_bonuses(self):
        """Constant base with multiple bar bonuses and variable bonus"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                ),
                VariableDef(
                    id="custom_score",
                    scope="agent",
                    type="scalar",
                    default=0.5,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                ),
            ],
            num_agents=num_agents,
            device=device,
        )

        # Set custom_score values
        vfs_registry.set("custom_score", torch.tensor([0.1, 0.2, 0.3, 0.4], device=device), writer="engine")

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="constant_base_with_shaped_bonus",
                base_reward=2.0,
                bars=["energy", "health"],  # Define meter ordering
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                    BarBonusConfig(bar="health", center=0.5, scale=0.3),
                ],
                variable_bonuses=[
                    VariableBonusConfig(variable="custom_score", weight=2.0),
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.5], [0.8, 0.6], [0.2, 0.4], [1.0, 1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base_reward + energy_bonus + health_bonus + custom_score * 2.0
        # For agent 0: 2.0 + 0.5*(0.5-0.5) + 0.3*(0.5-0.5) + 2.0*0.1 = 2.0 + 0 + 0 + 0.2 = 2.2
        # For agent 1: 2.0 + 0.5*(0.8-0.5) + 0.3*(0.6-0.5) + 2.0*0.2 = 2.0 + 0.15 + 0.03 + 0.4 = 2.58
        # For agent 2: 2.0 + 0.5*(0.2-0.5) + 0.3*(0.4-0.5) + 2.0*0.3 = 2.0 - 0.15 - 0.03 + 0.6 = 2.42
        # For agent 3: 2.0 + 0.5*(1.0-0.5) + 0.3*(1.0-0.5) + 2.0*0.4 = 2.0 + 0.25 + 0.15 + 0.8 = 3.2
        expected = torch.tensor([2.2, 2.58, 2.42, 3.2], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

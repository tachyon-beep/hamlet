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

    def test_additive_unweighted_strategy(self):
        """Additive unweighted: reward = base + sum(bars)"""
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
                type="additive_unweighted",
                base=0.5,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.9], [0.0, 0.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + energy + health
        # [0.5 + 0.2 + 0.3, 0.5 + 0.5 + 0.5, 0.5 + 0.8 + 0.9, 0.5 + 0.0 + 0.0]
        expected = torch.tensor([1.0, 1.5, 2.2, 0.5], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_weighted_sum_strategy(self):
        """Weighted sum: reward = sum(weight_i * source_i)"""
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

        # For this test, assume weighted_sum uses bar_bonuses with weights
        # (simplified version without sources field for now)
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="weighted_sum",
                base=0.0,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.0, scale=2.0),  # weight=2.0
                    BarBonusConfig(bar="health", center=0.0, scale=1.5),  # weight=1.5
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.4], [1.0, 1.0], [0.2, 0.8], [0.0, 0.5]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 2.0 * energy + 1.5 * health
        # [2.0*0.5 + 1.5*0.4, 2.0*1.0 + 1.5*1.0, 2.0*0.2 + 1.5*0.8, 2.0*0.0 + 1.5*0.5]
        expected = torch.tensor([1.6, 3.5, 1.6, 0.75], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_polynomial_strategy(self):
        """Polynomial: reward = sum(weight_i * bar_i^exponent_i)"""
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

        # Using bar_bonuses: scale=weight, center=exponent
        # Note: center must be in [0.0, 1.0] due to BarBonusConfig constraints
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="polynomial",
                base=0.0,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=2.0),  # 2.0 * energy^0.5
                    BarBonusConfig(bar="health", center=1.0, scale=1.5),  # 1.5 * health^1.0
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.25, 0.4], [1.0, 1.0], [0.64, 0.9], [0.0, 0.5]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 2.0 * energy^0.5 + 1.5 * health^1.0
        # [2.0*0.5 + 1.5*0.4, 2.0*1.0 + 1.5*1.0, 2.0*0.8 + 1.5*0.9, 2.0*0.0 + 1.5*0.5]
        expected = torch.tensor(
            [
                2.0 * 0.25**0.5 + 1.5 * 0.4**1.0,
                2.0 * 1.0**0.5 + 1.5 * 1.0**1.0,
                2.0 * 0.64**0.5 + 1.5 * 0.9**1.0,
                2.0 * 0.0**0.5 + 1.5 * 0.5**1.0,
            ],
            device=device,
        )
        assert torch.allclose(extrinsic, expected, atol=1e-4)

    def test_all_strategies_handle_dead_agents(self):
        """All extrinsic strategies return 0.0 for dead agents."""
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

        # Test multiplicative strategy
        dac_multiplicative = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=2.0,
                bars=["energy"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )
        engine = DACEngine(dac_multiplicative, vfs_registry, device, num_agents)

        # Agents 0,1 alive with high energy, agents 2,3 dead with high energy
        meters = torch.tensor([[1.0], [0.8], [1.0], [0.9]], device=device)
        dones = torch.tensor([False, False, True, True], device=device)
        rewards = engine.extrinsic_fn(meters, dones)

        # Alive agents get rewards, dead agents get 0.0
        assert rewards[0] > 0.0  # Alive agent with energy=1.0
        assert rewards[1] > 0.0  # Alive agent with energy=0.8
        assert rewards[2] == 0.0  # Dead agent (should be 0.0 despite energy=1.0)
        assert rewards[3] == 0.0  # Dead agent (should be 0.0 despite energy=0.9)

        # Test constant_base_with_shaped_bonus strategy
        dac_constant = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="constant_base_with_shaped_bonus",
                base_reward=1.0,
                bar_bonuses=[BarBonusConfig(bar="energy", center=0.5, scale=0.5)],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )
        engine = DACEngine(dac_constant, vfs_registry, device, num_agents)
        rewards = engine.extrinsic_fn(meters, dones)

        assert rewards[0] > 0.0
        assert rewards[1] > 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

        # Test additive_unweighted strategy
        dac_additive = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="additive_unweighted",
                base=0.5,
                bars=["energy"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )
        engine = DACEngine(dac_additive, vfs_registry, device, num_agents)
        rewards = engine.extrinsic_fn(meters, dones)

        assert rewards[0] > 0.0
        assert rewards[1] > 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

        # Test weighted_sum strategy
        dac_weighted = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="weighted_sum",
                base=0.0,
                bar_bonuses=[BarBonusConfig(bar="energy", center=0.0, scale=2.0)],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )
        engine = DACEngine(dac_weighted, vfs_registry, device, num_agents)
        rewards = engine.extrinsic_fn(meters, dones)

        assert rewards[0] > 0.0
        assert rewards[1] > 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

        # Test polynomial strategy
        dac_polynomial = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="polynomial",
                base=0.0,
                bar_bonuses=[BarBonusConfig(bar="energy", center=0.5, scale=2.0)],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )
        engine = DACEngine(dac_polynomial, vfs_registry, device, num_agents)
        rewards = engine.extrinsic_fn(meters, dones)

        assert rewards[0] > 0.0
        assert rewards[1] > 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

    def test_threshold_based_strategy(self):
        """Threshold-based: bonus when bar crosses threshold"""
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

        # Threshold at 0.5: give bonus=1.0 if energy >= 0.5, else bonus=0.0
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="threshold_based",
                base=0.5,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=1.0),  # threshold=0.5, bonus=1.0
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Energy values: [0.2, 0.5, 0.8, 1.0]
        meters = torch.tensor([[0.2], [0.5], [0.8], [1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + (1.0 if energy >= 0.5 else 0.0)
        # [0.5 + 0.0, 0.5 + 1.0, 0.5 + 1.0, 0.5 + 1.0]
        expected = torch.tensor([0.5, 1.5, 1.5, 1.5], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_aggregation_min_strategy(self):
        """Aggregation (min): reward = base + min(bars)"""
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

        # For this simplified version, assume aggregation always uses min()
        # A more complete implementation would have an 'operation' field
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="aggregation",
                base=0.5,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.3, 0.8], [0.5, 0.4], [0.9, 0.7], [0.2, 0.1]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + min(energy, health)
        # [0.5 + min(0.3, 0.8), 0.5 + min(0.5, 0.4), 0.5 + min(0.9, 0.7), 0.5 + min(0.2, 0.1)]
        # = [0.5 + 0.3, 0.5 + 0.4, 0.5 + 0.7, 0.5 + 0.1]
        expected = torch.tensor([0.8, 0.9, 1.2, 0.6], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_vfs_variable_strategy(self):
        """VFS variable: reward = vfs[variable] * weight (escape hatch)"""
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
                    id="custom_reward",
                    scope="agent",
                    type="scalar",
                    default=0.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                ),
            ],
            num_agents=num_agents,
            device=device,
        )

        # Set custom reward values in VFS
        vfs_registry.set("custom_reward", torch.tensor([0.5, 1.0, 1.5, 2.0], device=device), writer="engine")

        # Use variable_bonuses for weight and variable
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="vfs_variable",
                base=0.0,
                variable_bonuses=[
                    VariableBonusConfig(variable="custom_reward", weight=2.0),
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Meters not used for vfs_variable strategy
        meters = torch.tensor([[1.0], [1.0], [1.0], [1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: weight * custom_reward = 2.0 * [0.5, 1.0, 1.5, 2.0]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_hybrid_strategy(self):
        """Hybrid: combine multiple approaches (simplified as weighted bars)"""
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

        # Simplified hybrid: combines multiple weighted bars
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="hybrid",
                base=1.0,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.0, scale=1.5),  # Linear term
                    BarBonusConfig(bar="health", center=0.5, scale=0.5),  # Shaped term
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.5], [0.8, 0.6], [0.2, 0.4], [1.0, 1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + 1.5*energy + 0.5*(health-0.5)
        # [1.0 + 1.5*0.5 + 0.5*(0.5-0.5), 1.0 + 1.5*0.8 + 0.5*(0.6-0.5), ...]
        expected = torch.tensor([1.75, 2.25, 1.25, 2.75], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-5)


class TestShapingBonuses:
    """Test shaping bonus compilation and evaluation."""

    def test_approach_reward_bonus(self):
        """approach_reward bonus rewards moving toward target affordance."""
        from townlet.config.drive_as_code import ApproachRewardConfig

        device = torch.device("cpu")
        num_agents = 3

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
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                ApproachRewardConfig(
                    type="approach_reward",
                    weight=0.5,
                    target_affordance="Bed",
                    max_distance=10.0,
                )
            ],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Agent 0: distance=2, Agent 1: distance=5, Agent 2: distance≈19 (far/clamped)
        agent_positions = torch.tensor([[1.0, 1.0], [8.0, 1.0], [20.0, 20.0]], device=device)
        affordance_positions = {"Bed": torch.tensor([3.0, 1.0], device=device)}

        # Calculate shaping bonuses
        bonuses = torch.zeros(num_agents, device=device)
        for shaping_fn in engine.shaping_fns:
            bonuses += shaping_fn(agent_positions=agent_positions, affordance_positions=affordance_positions)

        # Agent 0: distance=2 → bonus = 0.5 * (1 - 2/10) = 0.4
        # Agent 1: distance=5 → bonus = 0.5 * (1 - 5/10) = 0.25
        # Agent 2: distance≈24 → bonus = 0.0 (clamped)
        assert torch.isclose(bonuses[0], torch.tensor(0.4, device=device), atol=1e-5)
        assert torch.isclose(bonuses[1], torch.tensor(0.25, device=device), atol=1e-5)
        assert bonuses[2] == 0.0

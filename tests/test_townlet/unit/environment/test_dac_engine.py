"""Tests for DAC Engine."""

import pytest
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


@pytest.fixture
def bar_index_map_single():
    """Standard bar index map for single-bar tests (energy only)."""
    return {"energy": 0}


@pytest.fixture
def bar_index_map_dual():
    """Standard bar index map for dual-bar tests (energy and health)."""
    return {"energy": 0, "health": 1}


class TestDACEngineInit:
    """Test DACEngine initialization."""

    def test_dac_engine_initializes(self, bar_index_map_single):
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
            bar_index_map=bar_index_map_single,
        )

        assert engine is not None
        assert engine.device == device
        assert engine.num_agents == num_agents

    def test_get_bar_index_missing_bar_raises(self):
        """_get_bar_index raises KeyError for undefined bar."""
        device = torch.device("cpu")
        num_agents = 4

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

        # Config references "health" bar, but bar_index_map only has "energy"
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],  # health not in bar_index_map
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="none",
                base_weight=0.0,
            ),
        )

        bar_index_map = {"energy": 0}  # Only energy defined
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map)

        # Create dummy meters
        meters = torch.tensor([[1.0], [0.8], [0.5], [0.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        # Trying to compute extrinsic with undefined bar should raise
        with pytest.raises(KeyError, match="Bar 'health' not found in universe metadata"):
            engine.extrinsic_fn(meters, dones)


class TestModifierCompilation:
    """Test modifier compilation and evaluation."""

    def test_compile_bar_modifier(self, bar_index_map_single):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Verify modifier was compiled
        assert "energy_crisis" in engine.modifiers
        assert callable(engine.modifiers["energy_crisis"])

    def test_evaluate_bar_modifier_crisis_range(self, bar_index_map_single):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Create meter values with 2 agents in crisis (< 0.3), 2 normal (>= 0.3)
        meters = torch.tensor([[0.1], [0.2], [0.5], [0.8]], device=device)  # [4, 1] energy only

        # Evaluate modifier
        multipliers = engine.modifiers["energy_crisis"](meters)

        # Should return 0.0 for crisis agents, 1.0 for normal agents
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
        assert torch.allclose(multipliers, expected)

    def test_evaluate_bar_modifier_normal_range(self, bar_index_map_single):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # All agents in normal range
        meters = torch.tensor([[0.4], [0.6], [0.8], [1.0]], device=device)

        multipliers = engine.modifiers["energy_crisis"](meters)

        # All should return 1.0
        expected = torch.ones(num_agents, device=device)
        assert torch.allclose(multipliers, expected)

    def test_compile_vfs_variable_modifier(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        assert "social_boost" in engine.modifiers
        assert callable(engine.modifiers["social_boost"])


class TestExtrinsicStrategies:
    """Test extrinsic reward strategy compilation."""

    def test_multiplicative_strategy(self, bar_index_map_single):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Create meter values: [0.5, 0.8, 1.0, 0.0]
        meters = torch.tensor([[0.5], [0.8], [1.0], [0.0]], device=device)
        dones = torch.tensor([False, False, False, True], device=device)

        # Calculate extrinsic rewards
        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base * energy (2.0 * [0.5, 0.8, 1.0, 0.0])
        # Dead agents get 0.0
        expected = torch.tensor([1.0, 1.6, 2.0, 0.0], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_multiplicative_multiple_bars(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.8], [1.0, 1.0], [0.2, 0.5], [0.0, 1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 1.0 * energy * health
        expected = torch.tensor([0.4, 1.0, 0.1, 0.0], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_constant_base_with_shaped_bonus(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Energy values: [0.0, 0.5, 1.0, 0.25]
        meters = torch.tensor([[0.0], [0.5], [1.0], [0.25]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base_reward + scale * (energy - center)
        # [1.0 + 0.5 * (0.0 - 0.5), 1.0 + 0.5 * (0.5 - 0.5), 1.0 + 0.5 * (1.0 - 0.5), 1.0 + 0.5 * (0.25 - 0.5)]
        # = [0.75, 1.0, 1.25, 0.875]
        expected = torch.tensor([0.75, 1.0, 1.25, 0.875], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_constant_base_multiple_bonuses(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

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

    def test_additive_unweighted_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # meters: [energy, health]
        meters = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.9], [0.0, 0.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + energy + health
        # [0.5 + 0.2 + 0.3, 0.5 + 0.5 + 0.5, 0.5 + 0.8 + 0.9, 0.5 + 0.0 + 0.0]
        expected = torch.tensor([1.0, 1.5, 2.2, 0.5], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_weighted_sum_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # meters: [energy, health]
        meters = torch.tensor([[0.5, 0.4], [1.0, 1.0], [0.2, 0.8], [0.0, 0.5]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 2.0 * energy + 1.5 * health
        # [2.0*0.5 + 1.5*0.4, 2.0*1.0 + 1.5*1.0, 2.0*0.2 + 1.5*0.8, 2.0*0.0 + 1.5*0.5]
        expected = torch.tensor([1.6, 3.5, 1.6, 0.75], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)

    def test_polynomial_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

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

    def test_all_strategies_handle_dead_agents(self, bar_index_map_single):
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
        engine = DACEngine(dac_multiplicative, vfs_registry, device, num_agents, bar_index_map_single)

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
        engine = DACEngine(dac_constant, vfs_registry, device, num_agents, bar_index_map_single)
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
        engine = DACEngine(dac_additive, vfs_registry, device, num_agents, bar_index_map_single)
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
        engine = DACEngine(dac_weighted, vfs_registry, device, num_agents, bar_index_map_single)
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
        engine = DACEngine(dac_polynomial, vfs_registry, device, num_agents, bar_index_map_single)
        rewards = engine.extrinsic_fn(meters, dones)

        assert rewards[0] > 0.0
        assert rewards[1] > 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

    def test_threshold_based_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Energy values: [0.2, 0.5, 0.8, 1.0]
        meters = torch.tensor([[0.2], [0.5], [0.8], [1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + (1.0 if energy >= 0.5 else 0.0)
        # [0.5 + 0.0, 0.5 + 1.0, 0.5 + 1.0, 0.5 + 1.0]
        expected = torch.tensor([0.5, 1.5, 1.5, 1.5], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_aggregation_min_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # meters: [energy, health]
        meters = torch.tensor([[0.3, 0.8], [0.5, 0.4], [0.9, 0.7], [0.2, 0.1]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base + min(energy, health)
        # [0.5 + min(0.3, 0.8), 0.5 + min(0.5, 0.4), 0.5 + min(0.9, 0.7), 0.5 + min(0.2, 0.1)]
        # = [0.5 + 0.3, 0.5 + 0.4, 0.5 + 0.7, 0.5 + 0.1]
        expected = torch.tensor([0.8, 0.9, 1.2, 0.6], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_vfs_variable_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Meters not used for vfs_variable strategy
        meters = torch.tensor([[1.0], [1.0], [1.0], [1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: weight * custom_reward = 2.0 * [0.5, 1.0, 1.5, 2.0]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_hybrid_strategy(self, bar_index_map_dual):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

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

    def test_approach_reward_bonus(self, bar_index_map_single):
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

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

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

    def test_completion_bonus(self, bar_index_map_single):
        """completion_bonus gives fixed bonus when agent completes interaction."""
        from townlet.config.drive_as_code import CompletionBonusConfig

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
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                CompletionBonusConfig(
                    type="completion_bonus",
                    weight=1.0,
                    affordance="Bed",
                )
            ],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Agent 0: completed Bed, Agent 1: completed Fridge, Agent 2: no completion, Agent 3: completed Bed
        last_action_affordance = ["Bed", "Fridge", None, "Bed"]

        # Calculate shaping bonuses
        bonuses = torch.zeros(num_agents, device=device)
        for shaping_fn in engine.shaping_fns:
            bonuses += shaping_fn(last_action_affordance=last_action_affordance)

        # Agent 0: completed Bed → bonus = 1.0
        # Agent 1: completed Fridge (not target) → bonus = 0.0
        # Agent 2: no completion → bonus = 0.0
        # Agent 3: completed Bed → bonus = 1.0
        assert bonuses[0] == 1.0
        assert bonuses[1] == 0.0
        assert bonuses[2] == 0.0
        assert bonuses[3] == 1.0

    def test_efficiency_bonus(self, bar_index_map_single):
        """efficiency_bonus rewards maintaining bar above threshold."""
        from townlet.config.drive_as_code import EfficiencyBonusConfig

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
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                EfficiencyBonusConfig(
                    type="efficiency_bonus",
                    weight=0.5,
                    bar="energy",
                    threshold=0.7,
                )
            ],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Agent 0: energy=0.9 (above), Agent 1: energy=0.7 (at threshold), Agent 2: energy=0.5 (below), Agent 3: energy=0.0 (below)
        meters = torch.tensor([[0.9], [0.7], [0.5], [0.0]], device=device)

        # Calculate shaping bonuses
        bonuses = torch.zeros(num_agents, device=device)
        for shaping_fn in engine.shaping_fns:
            bonuses += shaping_fn(meters=meters)

        # Agent 0: energy >= 0.7 → bonus = 0.5
        # Agent 1: energy >= 0.7 → bonus = 0.5
        # Agent 2: energy < 0.7 → bonus = 0.0
        # Agent 3: energy < 0.7 → bonus = 0.0
        assert bonuses[0] == 0.5
        assert bonuses[1] == 0.5
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_state_achievement_bonus(self, bar_index_map_dual):
        """state_achievement rewards when ALL conditions are met."""
        from townlet.config.drive_as_code import BarCondition, StateAchievementConfig

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
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy", "health"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                StateAchievementConfig(
                    type="state_achievement",
                    weight=2.0,
                    conditions=[
                        BarCondition(bar="energy", min_value=0.8),
                        BarCondition(bar="health", min_value=0.7),
                    ],
                )
            ],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Agent 0: both above thresholds, Agent 1: energy low, Agent 2: health low, Agent 3: both low
        meters = torch.tensor(
            [
                [0.9, 0.8],  # Agent 0: energy=0.9, health=0.8
                [0.5, 0.9],  # Agent 1: energy=0.5, health=0.9
                [0.9, 0.5],  # Agent 2: energy=0.9, health=0.5
                [0.5, 0.5],  # Agent 3: energy=0.5, health=0.5
            ],
            device=device,
        )

        # Calculate shaping bonuses
        bonuses = torch.zeros(num_agents, device=device)
        for shaping_fn in engine.shaping_fns:
            bonuses += shaping_fn(meters=meters)

        # Agent 0: both conditions met → bonus = 2.0
        # Agent 1: energy < 0.8 → bonus = 0.0
        # Agent 2: health < 0.7 → bonus = 0.0
        # Agent 3: both below → bonus = 0.0
        assert bonuses[0] == 2.0
        assert bonuses[1] == 0.0
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_streak_bonus(self, bar_index_map_dual):
        """streak_bonus rewards consecutive uses of affordance."""
        from townlet.config.drive_as_code import StreakBonusConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                StreakBonusConfig(
                    type="streak_bonus",
                    weight=3.0,
                    affordance="Bed",
                    min_streak=3,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Agent 0: streak=5 (exceeds min), Agent 1: streak=3 (meets min),
        # Agent 2: streak=2 (below min), Agent 3: streak=0 (no streak)
        affordance_streak = {
            "Bed": torch.tensor([5, 3, 2, 0], device=device),
            "Fridge": torch.tensor([1, 1, 1, 1], device=device),
        }

        bonuses = engine.shaping_fns[0](affordance_streak=affordance_streak)

        # Agent 0,1: streak >= 3 → bonus = 3.0
        # Agent 2,3: streak < 3 → bonus = 0.0
        assert bonuses[0] == 3.0
        assert bonuses[1] == 3.0
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_diversity_bonus(self, bar_index_map_single):
        """diversity_bonus rewards using many different affordances."""
        from townlet.config.drive_as_code import DiversityBonusConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                DiversityBonusConfig(
                    type="diversity_bonus",
                    weight=2.5,
                    min_unique_affordances=3,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Agent 0: 5 unique (exceeds), Agent 1: 3 unique (meets),
        # Agent 2: 2 unique (below), Agent 3: 0 unique (none)
        unique_affordances_used = torch.tensor([5, 3, 2, 0], device=device)

        bonuses = engine.shaping_fns[0](unique_affordances_used=unique_affordances_used)

        # Agent 0,1: unique >= 3 → bonus = 2.5
        # Agent 2,3: unique < 3 → bonus = 0.0
        assert bonuses[0] == 2.5
        assert bonuses[1] == 2.5
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_timing_bonus(self, bar_index_map_single):
        """timing_bonus rewards using affordance during specific time windows."""
        from townlet.config.drive_as_code import TimeRange, TimingBonusConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                TimingBonusConfig(
                    type="timing_bonus",
                    weight=1.0,
                    time_ranges=[
                        TimeRange(start_hour=22, end_hour=6, affordance="Bed", multiplier=2.0),
                        TimeRange(start_hour=12, end_hour=13, affordance="Fridge", multiplier=1.5),
                    ],
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Agent 0: hour=23, last_action=Bed (in range 22-6, matches) → bonus = 1.0 * 2.0
        # Agent 1: hour=12, last_action=Fridge (in range 12-13, matches) → bonus = 1.0 * 1.5
        # Agent 2: hour=10, last_action=Bed (not in range 22-6) → bonus = 0.0
        # Agent 3: hour=12, last_action=Bed (in range 12-13, wrong affordance) → bonus = 0.0
        current_hour = torch.tensor([23, 12, 10, 12], device=device)
        last_action_affordance = ["Bed", "Fridge", "Bed", "Bed"]

        bonuses = engine.shaping_fns[0](current_hour=current_hour, last_action_affordance=last_action_affordance)

        # Agent 0: timing match → 1.0 * 2.0 = 2.0
        # Agent 1: timing match → 1.0 * 1.5 = 1.5
        # Agent 2: no match → 0.0
        # Agent 3: no match → 0.0
        assert bonuses[0] == 2.0
        assert bonuses[1] == 1.5
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_economic_efficiency(self):
        """economic_efficiency rewards maintaining money above threshold."""
        from townlet.config.drive_as_code import EconomicEfficiencyConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy", "money"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                EconomicEfficiencyConfig(
                    type="economic_efficiency",
                    weight=3.0,
                    money_bar="money",
                    min_balance=0.5,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        # This test uses energy and money bars
        bar_index_map = {"energy": 0, "money": 1}
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map)

        # Agent 0: money=0.8 (above), Agent 1: money=0.5 (at threshold),
        # Agent 2: money=0.3 (below), Agent 3: money=0.0 (below)
        meters = torch.tensor(
            [
                [1.0, 0.8],  # [energy, money]
                [1.0, 0.5],
                [1.0, 0.3],
                [1.0, 0.0],
            ],
            device=device,
        )

        bonuses = engine.shaping_fns[0](meters=meters)

        # Agent 0,1: money >= 0.5 → bonus = 3.0
        # Agent 2,3: money < 0.5 → bonus = 0.0
        assert bonuses[0] == 3.0
        assert bonuses[1] == 3.0
        assert bonuses[2] == 0.0
        assert bonuses[3] == 0.0

    def test_balance_bonus(self, bar_index_map_dual):
        """balance_bonus rewards keeping multiple bars balanced."""
        from townlet.config.drive_as_code import BalanceBonusConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy", "health"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                BalanceBonusConfig(
                    type="balance_bonus",
                    weight=5.0,
                    bars=["energy", "health"],
                    max_imbalance=0.2,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Agent 0: perfect balance (0.5, 0.5) → imbalance=0.0
        # Agent 1: slight imbalance (0.6, 0.5) → imbalance=0.1 (within threshold)
        # Agent 2: at threshold (0.8, 0.6) → imbalance=0.2 (at threshold, gets bonus)
        # Agent 3: too much imbalance (1.0, 0.5) → imbalance=0.5 (exceeds threshold)
        meters = torch.tensor(
            [
                [0.5, 0.5],  # Agent 0
                [0.6, 0.5],  # Agent 1
                [0.8, 0.6],  # Agent 2
                [1.0, 0.5],  # Agent 3
            ],
            device=device,
        )

        bonuses = engine.shaping_fns[0](meters=meters)

        # Agents 0,1,2: imbalance <= 0.2 → bonus = 5.0
        # Agent 3: imbalance > 0.2 → bonus = 0.0
        assert bonuses[0] == 5.0
        assert bonuses[1] == 5.0
        assert bonuses[2] == 5.0
        assert bonuses[3] == 0.0

    def test_crisis_avoidance(self, bar_index_map_dual):
        """crisis_avoidance rewards staying above crisis threshold."""
        from townlet.config.drive_as_code import CrisisAvoidanceConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                CrisisAvoidanceConfig(
                    type="crisis_avoidance",
                    weight=3.0,
                    bar="energy",
                    crisis_threshold=0.3,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 5
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Agent 0: energy=0.8 (well above crisis) → bonus
        # Agent 1: energy=0.4 (above crisis) → bonus
        # Agent 2: energy=0.31 (just above crisis) → bonus
        # Agent 3: energy=0.3 (at crisis, NOT above) → no bonus
        # Agent 4: energy=0.1 (in crisis) → no bonus
        meters = torch.tensor([[0.8], [0.4], [0.31], [0.3], [0.1]], device=device)

        bonuses = engine.shaping_fns[0](meters=meters)

        # Agents 0,1,2: energy > 0.3 → bonus = 3.0
        # Agents 3,4: energy <= 0.3 → bonus = 0.0
        assert bonuses[0] == 3.0
        assert bonuses[1] == 3.0
        assert bonuses[2] == 3.0
        assert bonuses[3] == 0.0
        assert bonuses[4] == 0.0

    def test_vfs_variable_bonus(self, bar_index_map_single):
        """vfs_variable bonus uses VFS variable value as bonus."""
        from townlet.config.drive_as_code import VfsVariableConfig

        dac_config = DriveAsCodeConfig(
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1),
            shaping=[
                VfsVariableConfig(
                    type="vfs_variable",
                    weight=2.0,
                    variable="custom_bonus",
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
                ),
                VariableDef(
                    id="custom_bonus",
                    scope="agent",
                    type="scalar",
                    default=0.0,
                    lifetime="episode",
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                ),
            ],
            num_agents=num_agents,
            device=device,
        )

        # Register VFS variable with different values per agent
        vfs_registry.set("custom_bonus", torch.tensor([1.0, 0.5, 0.0, -0.5], device=device), writer="engine")

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Call bonus (no kwargs needed - reads from VFS)
        bonuses = engine.shaping_fns[0]()

        # Bonus: weight * variable_value
        # Agent 0: 2.0 * 1.0 = 2.0
        # Agent 1: 2.0 * 0.5 = 1.0
        # Agent 2: 2.0 * 0.0 = 0.0
        # Agent 3: 2.0 * -0.5 = -1.0 (negative bonus!)
        assert bonuses[0] == 2.0
        assert bonuses[1] == 1.0
        assert bonuses[2] == 0.0
        assert bonuses[3] == -1.0


class TestShapingBonusEdgeCases:
    """Test edge cases for shaping bonuses (missing kwargs, invalid data)."""

    def test_approach_reward_missing_affordance(self, bar_index_map_single):
        """approach_reward returns zeros when target affordance not in positions."""
        from townlet.config.drive_as_code import ApproachRewardConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                ApproachRewardConfig(
                    type="approach_reward",
                    weight=0.5,
                    target_affordance="Bed",
                    max_distance=10.0,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Agent positions provided, but affordance_positions does NOT include "Bed"
        agent_positions = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], device=device)
        affordance_positions = {"Fridge": torch.tensor([10.0, 10.0], device=device)}  # No "Bed"

        # Call the compiled shaping function directly
        bonuses = engine.shaping_fns[0](
            agent_positions=agent_positions,
            affordance_positions=affordance_positions,
        )

        # Should return zeros for all agents (affordance not found)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_completion_bonus_missing_kwarg(self, bar_index_map_single):
        """completion_bonus returns zeros when last_action_affordance missing."""
        from townlet.config.drive_as_code import CompletionBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                CompletionBonusConfig(
                    type="completion_bonus",
                    weight=5.0,
                    affordance="Bed",
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Call without providing last_action_affordance kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_efficiency_bonus_missing_kwarg(self, bar_index_map_single):
        """efficiency_bonus returns zeros when meters missing."""
        from townlet.config.drive_as_code import EfficiencyBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                EfficiencyBonusConfig(
                    type="efficiency_bonus",
                    weight=2.0,
                    bar="energy",
                    threshold=0.7,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Call without providing meters kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_state_achievement_missing_kwarg(self, bar_index_map_single):
        """state_achievement returns zeros when meters missing."""
        from townlet.config.drive_as_code import BarCondition, StateAchievementConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                StateAchievementConfig(
                    type="state_achievement",
                    weight=10.0,
                    conditions=[
                        BarCondition(bar="energy", min_value=0.8),
                    ],
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Call without providing meters kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_streak_bonus_missing_kwarg(self, bar_index_map_dual):
        """streak_bonus returns zeros when affordance_streak missing."""
        from townlet.config.drive_as_code import StreakBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                StreakBonusConfig(
                    type="streak_bonus",
                    weight=5.0,
                    affordance="Bed",
                    min_streak=2,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing affordance_streak kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_streak_bonus_affordance_not_found(self, bar_index_map_single):
        """streak_bonus returns zeros when target affordance not in streak dict."""
        from townlet.config.drive_as_code import StreakBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                StreakBonusConfig(
                    type="streak_bonus",
                    weight=5.0,
                    affordance="Bed",
                    min_streak=2,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Provide affordance_streak dict but without "Bed"
        affordance_streak = {"Fridge": torch.tensor([5, 5, 5], device=device)}

        bonuses = engine.shaping_fns[0](affordance_streak=affordance_streak)

        # Should return zeros (affordance not in dict)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_diversity_bonus_missing_kwarg(self, bar_index_map_dual):
        """diversity_bonus returns zeros when unique_affordances_used missing."""
        from townlet.config.drive_as_code import DiversityBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                DiversityBonusConfig(
                    type="diversity_bonus",
                    weight=5.0,
                    min_unique_affordances=2,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing unique_affordances_used kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_timing_bonus_missing_kwargs(self, bar_index_map_dual):
        """timing_bonus returns zeros when current_hour or last_action_affordance missing."""
        from townlet.config.drive_as_code import TimeRange, TimingBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                TimingBonusConfig(
                    type="timing_bonus",
                    weight=5.0,
                    time_ranges=[
                        TimeRange(start_hour=22, end_hour=6, affordance="Bed", multiplier=2.0),
                    ],
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing any kwargs
        bonuses = engine.shaping_fns[0]()
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

        # Call with only current_hour (missing last_action_affordance)
        current_hour = torch.tensor([23, 12, 10], device=device)
        bonuses = engine.shaping_fns[0](current_hour=current_hour)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

        # Call with only last_action_affordance (missing current_hour)
        last_action_affordance = ["Bed", "Bed", "Bed"]
        bonuses = engine.shaping_fns[0](last_action_affordance=last_action_affordance)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_economic_efficiency_missing_kwarg(self, bar_index_map_dual):
        """economic_efficiency returns zeros when meters missing."""
        from townlet.config.drive_as_code import EconomicEfficiencyConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy", "money"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                EconomicEfficiencyConfig(
                    type="economic_efficiency",
                    weight=5.0,
                    money_bar="money",
                    min_balance=0.7,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing meters kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_balance_bonus_missing_kwarg(self, bar_index_map_dual):
        """balance_bonus returns zeros when meters missing."""
        from townlet.config.drive_as_code import BalanceBonusConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy", "health"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                BalanceBonusConfig(
                    type="balance_bonus",
                    weight=5.0,
                    bars=["energy", "health"],
                    max_imbalance=0.2,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing meters kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_crisis_avoidance_missing_kwarg(self, bar_index_map_dual):
        """crisis_avoidance returns zeros when meters missing."""
        from townlet.config.drive_as_code import CrisisAvoidanceConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                CrisisAvoidanceConfig(
                    type="crisis_avoidance",
                    weight=3.0,
                    bar="energy",
                    crisis_threshold=0.3,
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 3
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
        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_dual)

        # Call without providing meters kwarg
        bonuses = engine.shaping_fns[0]()

        # Should return zeros (kwarg missing)
        assert torch.allclose(bonuses, torch.zeros(num_agents, device=device))

    def test_vfs_variable_missing_variable(self, bar_index_map_single):
        """vfs_variable raises error when variable not in registry."""
        import pytest

        from townlet.config.drive_as_code import VfsVariableConfig

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                VfsVariableConfig(
                    type="vfs_variable",
                    weight=2.0,
                    variable="nonexistent_var",
                )
            ],
        )

        device = torch.device("cpu")
        num_agents = 4
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
        # Don't register the variable

        engine = DACEngine(dac_config, vfs_registry, device, num_agents, bar_index_map_single)

        # Should raise KeyError when trying to read missing variable
        with pytest.raises(KeyError, match="nonexistent_var"):
            engine.shaping_fns[0]()

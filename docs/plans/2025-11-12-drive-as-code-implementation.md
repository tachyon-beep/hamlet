# Drive As Code (DAC) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a declarative reward function compiler (DAC) that extracts all reward logic from Python into composable YAML configurations, enabling researchers to A/B test reward structures without code changes.

**Architecture:** DAC follows the UAC/VFS pattern: YAML config → Pydantic DTOs → Compiler validation → Runtime execution. Components: Modifiers (range-based multipliers), Extrinsic Strategies (9 reward structures), Intrinsic Config (exploration drives), Shaping Bonuses (11 behavioral incentives). Final composition: `total_reward = extrinsic + (intrinsic × modifiers) + shaping`.

**Tech Stack:** Pydantic (DTOs), PyTorch (GPU tensors), YAML (configs), VFS (feature integration)

**Breaking Changes:** This implementation DELETES `RewardStrategy` and `AdaptiveRewardStrategy` classes, removes `reward_strategy` field from all configs. Pre-release status = zero backward compatibility.

---

## Phase 1: DTO Layer (8-10 hours)

### Task 1.1: Create RangeConfig DTO

**Files:**
- Create: `src/townlet/config/drive_as_code.py`
- Test: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
"""Tests for Drive As Code DTOs."""

import pytest
from pydantic import ValidationError

from townlet.config.drive_as_code import RangeConfig


class TestRangeConfig:
    """Test RangeConfig validation."""

    def test_valid_range(self):
        """Valid range configuration loads successfully."""
        config = RangeConfig(
            name="crisis",
            min=0.0,
            max=0.3,
            multiplier=0.0,
        )
        assert config.name == "crisis"
        assert config.min == 0.0
        assert config.max == 0.3
        assert config.multiplier == 0.0

    def test_min_must_be_less_than_max(self):
        """min must be < max."""
        with pytest.raises(ValidationError, match="min.*must be < max"):
            RangeConfig(
                name="invalid",
                min=0.5,
                max=0.3,  # min > max!
                multiplier=1.0,
            )

    def test_min_equals_max_invalid(self):
        """min == max is invalid."""
        with pytest.raises(ValidationError, match="min.*must be < max"):
            RangeConfig(
                name="invalid",
                min=0.5,
                max=0.5,  # Equal!
                multiplier=1.0,
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestRangeConfig -v`
Expected: `ModuleNotFoundError: No module named 'townlet.config.drive_as_code'`

**Step 3: Write minimal implementation**

```python
"""Drive As Code configuration DTOs.

Declarative reward function specification system that extracts reward logic
from Python into composable YAML configurations.

Philosophy: Reward functions are compositions of extrinsic structure, modifiers,
shaping bonuses, and intrinsic computation. All components configurable via YAML.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RangeConfig(BaseModel):
    """Single range in a modifier.

    Ranges define multiplier values for different value ranges.
    Used for context-sensitive reward adjustment (crisis suppression, etc).

    Example:
        >>> crisis_range = RangeConfig(
        ...     name="crisis",
        ...     min=0.0,
        ...     max=0.2,
        ...     multiplier=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Human-readable range name")
    min: float = Field(description="Range minimum (inclusive)")
    max: float = Field(description="Range maximum (exclusive for all but last)")
    multiplier: float = Field(description="Multiplier to apply when value in this range")

    @model_validator(mode="after")
    def validate_range_bounds(self) -> "RangeConfig":
        """Ensure min < max."""
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestRangeConfig -v`
Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add RangeConfig DTO with validation

- Define RangeConfig for modifier ranges
- Validate min < max constraint
- Add comprehensive tests

Part of TASK-004C Phase 1"
```

---

### Task 1.2: Create ModifierConfig DTO

**Files:**
- Modify: `src/townlet/config/drive_as_code.py`
- Modify: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/config/test_drive_as_code_dto.py

from townlet.config.drive_as_code import ModifierConfig, RangeConfig


class TestModifierConfig:
    """Test ModifierConfig validation."""

    def test_valid_modifier_with_bar(self):
        """Valid modifier referencing a bar."""
        config = ModifierConfig(
            bar="energy",
            ranges=[
                RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
            ],
        )
        assert config.bar == "energy"
        assert config.variable is None
        assert len(config.ranges) == 2

    def test_valid_modifier_with_variable(self):
        """Valid modifier referencing a VFS variable."""
        config = ModifierConfig(
            variable="worst_physical_need",
            ranges=[
                RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
            ],
        )
        assert config.variable == "worst_physical_need"
        assert config.bar is None

    def test_must_specify_bar_or_variable(self):
        """Must specify either bar or variable (not neither)."""
        with pytest.raises(ValidationError, match="Must specify either 'bar' or 'variable'"):
            ModifierConfig(
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=1.0, multiplier=0.0),
                ],
            )

    def test_cannot_specify_both_bar_and_variable(self):
        """Cannot specify both bar and variable."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            ModifierConfig(
                bar="energy",
                variable="worst_need",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=1.0, multiplier=0.0),
                ],
            )

    def test_ranges_must_cover_zero_to_one(self):
        """Ranges must start at 0.0 and end at 1.0."""
        with pytest.raises(ValidationError, match="Ranges must start at 0.0"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.1, max=1.0, multiplier=0.0),  # Doesn't start at 0!
                ],
            )

        with pytest.raises(ValidationError, match="Ranges must end at 1.0"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.9, multiplier=0.0),  # Doesn't end at 1.0!
                ],
            )

    def test_ranges_cannot_have_gaps(self):
        """Ranges cannot have gaps between them."""
        with pytest.raises(ValidationError, match="Gap or overlap"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                    RangeConfig(name="normal", min=0.4, max=1.0, multiplier=1.0),  # Gap: 0.3-0.4!
                ],
            )

    def test_ranges_cannot_overlap(self):
        """Ranges cannot overlap."""
        with pytest.raises(ValidationError, match="Gap or overlap"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.4, multiplier=0.0),
                    RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),  # Overlap: 0.3-0.4!
                ],
            )

    def test_at_least_one_range_required(self):
        """Must have at least one range."""
        with pytest.raises(ValidationError):
            ModifierConfig(
                bar="energy",
                ranges=[],  # Empty!
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestModifierConfig -v`
Expected: `ImportError: cannot import name 'ModifierConfig'`

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/config/drive_as_code.py

class ModifierConfig(BaseModel):
    """Range-based modifier for contextual reward adjustment.

    Modifiers apply multipliers based on the current value of a bar or VFS variable.
    Used for crisis suppression, temporal decay, boredom boost, etc.

    Example:
        >>> energy_crisis = ModifierConfig(
        ...     bar="energy",
        ...     ranges=[
        ...         RangeConfig(name="crisis", min=0.0, max=0.2, multiplier=0.0),
        ...         RangeConfig(name="low", min=0.2, max=0.4, multiplier=0.3),
        ...         RangeConfig(name="normal", min=0.4, max=1.0, multiplier=1.0),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Source (exactly one required)
    bar: str | None = Field(default=None, description="Bar name to monitor")
    variable: str | None = Field(default=None, description="VFS variable name to monitor")

    # Range definitions
    ranges: list[RangeConfig] = Field(min_length=1, description="Range definitions with multipliers")

    @model_validator(mode="after")
    def validate_source(self) -> "ModifierConfig":
        """Ensure exactly one source specified."""
        if self.bar is None and self.variable is None:
            raise ValueError("Must specify either 'bar' or 'variable'")
        if self.bar is not None and self.variable is not None:
            raise ValueError("Cannot specify both 'bar' and 'variable'")
        return self

    @model_validator(mode="after")
    def validate_ranges_coverage(self) -> "ModifierConfig":
        """Ensure ranges cover [0.0, 1.0] without gaps or overlaps."""
        sorted_ranges = sorted(self.ranges, key=lambda r: r.min)

        # Check coverage starts at 0.0
        if sorted_ranges[0].min != 0.0:
            raise ValueError(f"Ranges must start at 0.0, got {sorted_ranges[0].min}")

        # Check no gaps or overlaps
        for i in range(len(sorted_ranges) - 1):
            current_max = sorted_ranges[i].max
            next_min = sorted_ranges[i + 1].min
            if current_max != next_min:
                raise ValueError(
                    f"Gap or overlap between ranges: "
                    f"{sorted_ranges[i].name} (max={current_max}) and "
                    f"{sorted_ranges[i+1].name} (min={next_min})"
                )

        # Check coverage ends at 1.0
        if sorted_ranges[-1].max != 1.0:
            raise ValueError(f"Ranges must end at 1.0, got {sorted_ranges[-1].max}")

        return self
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestModifierConfig -v`
Expected: `9 passed`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add ModifierConfig DTO with range validation

- Define ModifierConfig for contextual reward adjustment
- Validate exactly one source (bar or variable)
- Validate ranges cover [0, 1] without gaps/overlaps
- Add comprehensive tests

Part of TASK-004C Phase 1"
```

---

### Task 1.3: Create Extrinsic Strategy DTOs

**Files:**
- Modify: `src/townlet/config/drive_as_code.py`
- Modify: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/config/test_drive_as_code_dto.py

from townlet.config.drive_as_code import (
    BarBonusConfig,
    ExtrinsicStrategyConfig,
    VariableBonusConfig,
)


class TestExtrinsicStrategyConfig:
    """Test ExtrinsicStrategyConfig validation."""

    def test_multiplicative_strategy(self):
        """Multiplicative strategy with bars."""
        config = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy", "health"],
        )
        assert config.type == "multiplicative"
        assert config.base == 1.0
        assert config.bars == ["energy", "health"]

    def test_constant_base_with_shaped_bonus(self):
        """Constant base with bar and variable bonuses."""
        config = ExtrinsicStrategyConfig(
            type="constant_base_with_shaped_bonus",
            base_reward=1.0,
            bar_bonuses=[
                BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                BarBonusConfig(bar="health", center=0.5, scale=0.5),
            ],
            variable_bonuses=[
                VariableBonusConfig(variable="energy_urgency", weight=0.5),
            ],
        )
        assert config.type == "constant_base_with_shaped_bonus"
        assert config.base_reward == 1.0
        assert len(config.bar_bonuses) == 2
        assert len(config.variable_bonuses) == 1

    def test_additive_unweighted_strategy(self):
        """Additive unweighted strategy."""
        config = ExtrinsicStrategyConfig(
            type="additive_unweighted",
            base=0.0,
            bars=["energy", "health", "satiation"],
        )
        assert config.type == "additive_unweighted"
        assert config.bars == ["energy", "health", "satiation"]

    def test_vfs_variable_strategy(self):
        """VFS variable strategy (delegate to VFS)."""
        config = ExtrinsicStrategyConfig(
            type="vfs_variable",
            variable="custom_reward_function",
        )
        assert config.type == "vfs_variable"
        assert config.variable == "custom_reward_function"

    def test_apply_modifiers_optional(self):
        """Modifiers are optional for extrinsic strategies."""
        config = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy"],
            apply_modifiers=["energy_crisis"],
        )
        assert config.apply_modifiers == ["energy_crisis"]

        config2 = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy"],
        )
        assert config2.apply_modifiers == []


class TestBarBonusConfig:
    """Test BarBonusConfig validation."""

    def test_valid_bar_bonus(self):
        """Valid bar bonus configuration."""
        config = BarBonusConfig(
            bar="energy",
            center=0.5,
            scale=0.5,
        )
        assert config.bar == "energy"
        assert config.center == 0.5
        assert config.scale == 0.5

    def test_center_must_be_in_range(self):
        """Center must be in [0, 1]."""
        with pytest.raises(ValidationError):
            BarBonusConfig(bar="energy", center=1.5, scale=0.5)

    def test_scale_must_be_positive(self):
        """Scale must be > 0."""
        with pytest.raises(ValidationError):
            BarBonusConfig(bar="energy", center=0.5, scale=-0.5)


class TestVariableBonusConfig:
    """Test VariableBonusConfig validation."""

    def test_valid_variable_bonus(self):
        """Valid variable bonus configuration."""
        config = VariableBonusConfig(
            variable="energy_urgency",
            weight=0.5,
        )
        assert config.variable == "energy_urgency"
        assert config.weight == 0.5

    def test_weight_can_be_negative(self):
        """Weight can be negative (for penalties)."""
        config = VariableBonusConfig(
            variable="bathroom_emergency",
            weight=-2.0,
        )
        assert config.weight == -2.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestExtrinsicStrategyConfig -v`
Expected: `ImportError: cannot import name 'ExtrinsicStrategyConfig'`

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/config/drive_as_code.py

class BarBonusConfig(BaseModel):
    """Bar-based bonus for constant_base_with_shaped_bonus strategy.

    Applies a bonus/penalty based on deviation from center point.
    Formula: bonus = scale * (bar_value - center)

    Example:
        >>> energy_bonus = BarBonusConfig(
        ...     bar="energy",
        ...     center=0.5,  # Neutral point
        ...     scale=0.5,   # Magnitude
        ... )
        # energy=1.0 → bonus = 0.5 * (1.0 - 0.5) = +0.25
        # energy=0.0 → bonus = 0.5 * (0.0 - 0.5) = -0.25
    """

    model_config = ConfigDict(extra="forbid")

    bar: str = Field(description="Bar name")
    center: float = Field(ge=0.0, le=1.0, description="Neutral point (no bonus/penalty)")
    scale: float = Field(gt=0.0, description="Magnitude of bonus/penalty")


class VariableBonusConfig(BaseModel):
    """VFS variable-based bonus.

    Applies a weighted bonus from a VFS-computed variable.
    Formula: bonus = weight * variable_value

    Example:
        >>> urgency_bonus = VariableBonusConfig(
        ...     variable="energy_urgency",
        ...     weight=0.5,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="VFS variable name")
    weight: float = Field(description="Weight (can be negative for penalties)")


class ExtrinsicStrategyConfig(BaseModel):
    """Configuration for extrinsic reward strategies.

    Supports 9 strategy types:
    - multiplicative: reward = base * bar1 * bar2 * ...
    - constant_base_with_shaped_bonus: reward = base + sum(bar_bonuses) + sum(variable_bonuses)
    - additive_unweighted: reward = base + sum(bars)
    - weighted_sum: reward = sum(weight_i * source_i)
    - polynomial: reward = sum(weight_i * source_i^exponent_i)
    - threshold_based: reward = sum(threshold bonuses/penalties)
    - aggregation: reward = base + op(bars) where op ∈ {min, max, mean, product}
    - vfs_variable: reward = variable (delegate to VFS)
    - hybrid: reward = weighted combination of multiple strategies

    Example:
        >>> strategy = ExtrinsicStrategyConfig(
        ...     type="constant_base_with_shaped_bonus",
        ...     base_reward=1.0,
        ...     bar_bonuses=[
        ...         BarBonusConfig(bar="energy", center=0.5, scale=0.5),
        ...     ],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "multiplicative",
        "constant_base_with_shaped_bonus",
        "additive_unweighted",
        "weighted_sum",
        "polynomial",
        "threshold_based",
        "aggregation",
        "vfs_variable",
        "hybrid",
    ] = Field(description="Strategy type")

    # constant_base_with_shaped_bonus fields
    base_reward: float | None = Field(default=None, description="Base survival reward")
    bar_bonuses: list[BarBonusConfig] = Field(default_factory=list, description="Bar-based bonuses")
    variable_bonuses: list[VariableBonusConfig] = Field(default_factory=list, description="VFS variable bonuses")

    # multiplicative / additive_unweighted fields
    base: float | None = Field(default=None, description="Base value")
    bars: list[str] = Field(default_factory=list, description="Bar names")

    # vfs_variable fields
    variable: str | None = Field(default=None, description="VFS variable name")

    # TODO: Add fields for other strategy types (weighted_sum, polynomial, threshold_based, aggregation, hybrid)
    # These will be added incrementally as needed

    # Modifier application (optional)
    apply_modifiers: list[str] = Field(default_factory=list, description="Modifier names to apply to extrinsic rewards")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestExtrinsicStrategyConfig -v`
Expected: `All tests pass`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add Extrinsic Strategy DTOs

- Define BarBonusConfig and VariableBonusConfig
- Define ExtrinsicStrategyConfig with 9 strategy types
- Support multiplicative, constant_base, additive, vfs_variable
- Add comprehensive tests

Part of TASK-004C Phase 1"
```

---

### Task 1.4: Create Intrinsic Strategy DTO

**Files:**
- Modify: `src/townlet/config/drive_as_code.py`
- Modify: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/config/test_drive_as_code_dto.py

from townlet.config.drive_as_code import IntrinsicStrategyConfig


class TestIntrinsicStrategyConfig:
    """Test IntrinsicStrategyConfig validation."""

    def test_rnd_strategy(self):
        """RND intrinsic strategy."""
        config = IntrinsicStrategyConfig(
            strategy="rnd",
            base_weight=0.100,
        )
        assert config.strategy == "rnd"
        assert config.base_weight == 0.100

    def test_adaptive_rnd_strategy(self):
        """Adaptive RND with performance-based decay."""
        config = IntrinsicStrategyConfig(
            strategy="adaptive_rnd",
            base_weight=0.100,
        )
        assert config.strategy == "adaptive_rnd"

    def test_none_strategy(self):
        """No intrinsic reward."""
        config = IntrinsicStrategyConfig(
            strategy="none",
            base_weight=0.0,
        )
        assert config.strategy == "none"

    def test_apply_modifiers(self):
        """Can apply modifiers to intrinsic weight."""
        config = IntrinsicStrategyConfig(
            strategy="rnd",
            base_weight=0.100,
            apply_modifiers=["energy_crisis", "temporal_decay"],
        )
        assert config.apply_modifiers == ["energy_crisis", "temporal_decay"]

    def test_base_weight_must_be_in_range(self):
        """Base weight must be in [0, 1]."""
        with pytest.raises(ValidationError):
            IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=1.5,  # Too high!
            )

        with pytest.raises(ValidationError):
            IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=-0.1,  # Negative!
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestIntrinsicStrategyConfig -v`
Expected: `ImportError: cannot import name 'IntrinsicStrategyConfig'`

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/config/drive_as_code.py

from typing import Any


class IntrinsicStrategyConfig(BaseModel):
    """Configuration for intrinsic curiosity/exploration.

    Intrinsic rewards encourage exploration based on novelty.
    Supports multiple strategies:
    - rnd: Random Network Distillation (novelty = prediction error)
    - icm: Intrinsic Curiosity Module (forward/inverse model)
    - count_based: Pseudo-count bonuses
    - adaptive_rnd: RND with performance-based weight decay
    - none: No intrinsic reward

    Example:
        >>> intrinsic = IntrinsicStrategyConfig(
        ...     strategy="rnd",
        ...     base_weight=0.100,
        ...     apply_modifiers=["energy_crisis"],  # Suppress in crisis
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["rnd", "icm", "count_based", "adaptive_rnd", "none"] = Field(
        description="Intrinsic reward strategy"
    )
    base_weight: float = Field(ge=0.0, le=1.0, description="Base intrinsic weight")
    apply_modifiers: list[str] = Field(default_factory=list, description="Modifier names to apply to intrinsic weight")

    # Strategy-specific configs (optional, use defaults if not specified)
    rnd_config: dict[str, Any] | None = Field(default=None, description="RND configuration")
    icm_config: dict[str, Any] | None = Field(default=None, description="ICM configuration")
    count_config: dict[str, Any] | None = Field(default=None, description="Count-based configuration")
    adaptive_config: dict[str, Any] | None = Field(default=None, description="Adaptive RND configuration")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestIntrinsicStrategyConfig -v`
Expected: `6 passed`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add IntrinsicStrategyConfig DTO

- Define IntrinsicStrategyConfig with 5 strategy types
- Support RND, ICM, count_based, adaptive_rnd, none
- Validate base_weight in [0, 1]
- Add comprehensive tests

Part of TASK-004C Phase 1"
```

---

### Task 1.5: Create Shaping Bonus DTOs

**Files:**
- Modify: `src/townlet/config/drive_as_code.py`
- Modify: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/config/test_drive_as_code_dto.py

from townlet.config.drive_as_code import (
    ApproachRewardConfig,
    CompletionBonusConfig,
    TriggerCondition,
    VFSVariableBonusConfig,
)


class TestTriggerCondition:
    """Test TriggerCondition validation."""

    def test_trigger_with_above(self):
        """Trigger when value is above threshold."""
        trigger = TriggerCondition(
            source="bar",
            name="energy",
            above=0.3,
        )
        assert trigger.source == "bar"
        assert trigger.name == "energy"
        assert trigger.above == 0.3
        assert trigger.below is None

    def test_trigger_with_below(self):
        """Trigger when value is below threshold."""
        trigger = TriggerCondition(
            source="variable",
            name="energy_urgency",
            below=0.7,
        )
        assert trigger.source == "variable"
        assert trigger.below == 0.7

    def test_must_specify_above_or_below(self):
        """Must specify at least one threshold."""
        with pytest.raises(ValidationError, match="Must specify 'above' or 'below'"):
            TriggerCondition(
                source="bar",
                name="energy",
            )


class TestApproachRewardConfig:
    """Test ApproachRewardConfig validation."""

    def test_valid_approach_reward(self):
        """Valid approach reward configuration."""
        config = ApproachRewardConfig(
            type="approach_reward",
            target_affordance="Bed",
            trigger=TriggerCondition(source="bar", name="energy", below=0.3),
            bonus=1.0,
            decay_with_distance=True,
        )
        assert config.type == "approach_reward"
        assert config.target_affordance == "Bed"
        assert config.bonus == 1.0
        assert config.decay_with_distance is True


class TestCompletionBonusConfig:
    """Test CompletionBonusConfig validation."""

    def test_completion_bonus_all_affordances(self):
        """Completion bonus for all affordances."""
        config = CompletionBonusConfig(
            type="completion_bonus",
            affordances="all",
            bonus=1.0,
            scale_with_duration=True,
        )
        assert config.affordances == "all"

    def test_completion_bonus_specific_affordances(self):
        """Completion bonus for specific affordances."""
        config = CompletionBonusConfig(
            type="completion_bonus",
            affordances=["Bed", "Hospital", "Job"],
            bonus=1.0,
            scale_with_duration=False,
        )
        assert config.affordances == ["Bed", "Hospital", "Job"]


class TestVFSVariableBonusConfig:
    """Test VFSVariableBonusConfig (shaping bonus) validation."""

    def test_valid_vfs_variable_bonus(self):
        """Valid VFS variable bonus configuration."""
        config = VFSVariableBonusConfig(
            type="vfs_variable",
            variable="custom_shaping_signal",
            weight=1.0,
        )
        assert config.type == "vfs_variable"
        assert config.variable == "custom_shaping_signal"
        assert config.weight == 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestTriggerCondition -v`
Expected: `ImportError: cannot import name 'TriggerCondition'`

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/config/drive_as_code.py

class TriggerCondition(BaseModel):
    """Condition for triggering a shaping bonus.

    Evaluates to true when source value crosses threshold.

    Example:
        >>> low_energy = TriggerCondition(
        ...     source="bar",
        ...     name="energy",
        ...     below=0.3,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["bar", "variable"] = Field(description="Source type")
    name: str = Field(description="Source name (bar name or variable name)")
    above: float | None = Field(default=None, description="Trigger when value > threshold")
    below: float | None = Field(default=None, description="Trigger when value < threshold")

    @model_validator(mode="after")
    def validate_threshold(self) -> "TriggerCondition":
        """Ensure at least one threshold specified."""
        if self.above is None and self.below is None:
            raise ValueError("Must specify 'above' or 'below'")
        return self


class ApproachRewardConfig(BaseModel):
    """Encourage moving toward goals when needed.

    Gives bonus reward for moving closer to target affordance when trigger condition is met.

    Example:
        >>> approach_bed = ApproachRewardConfig(
        ...     type="approach_reward",
        ...     target_affordance="Bed",
        ...     trigger=TriggerCondition(source="bar", name="energy", below=0.3),
        ...     bonus=1.0,
        ...     decay_with_distance=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["approach_reward"]
    target_affordance: str = Field(description="Affordance to approach")
    trigger: TriggerCondition = Field(description="When to apply this bonus")
    bonus: float = Field(description="Bonus magnitude")
    decay_with_distance: bool = Field(default=True, description="Reduce bonus with distance")


class CompletionBonusConfig(BaseModel):
    """Reward for completing affordances.

    Gives bonus when agent finishes interacting with affordance.

    Example:
        >>> completion = CompletionBonusConfig(
        ...     type="completion_bonus",
        ...     affordances="all",
        ...     bonus=1.0,
        ...     scale_with_duration=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["completion_bonus"]
    affordances: Literal["all"] | list[str] = Field(description="Which affordances to reward")
    bonus: float = Field(description="Bonus magnitude")
    scale_with_duration: bool = Field(default=True, description="Scale bonus by interaction duration")


class VFSVariableBonusConfig(BaseModel):
    """Shaping bonus from VFS variable (escape hatch for custom logic).

    Allows arbitrary shaping logic to be computed in VFS and referenced here.

    Example:
        >>> custom_bonus = VFSVariableBonusConfig(
        ...     type="vfs_variable",
        ...     variable="custom_shaping_signal",
        ...     weight=1.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["vfs_variable"]
    variable: str = Field(description="VFS variable name")
    weight: float = Field(description="Weight to apply")


# Union type for all shaping bonuses (expand as more types are added)
ShapingBonusConfig = ApproachRewardConfig | CompletionBonusConfig | VFSVariableBonusConfig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestTriggerCondition -v`
Expected: `All tests pass`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add Shaping Bonus DTOs

- Define TriggerCondition for bonus activation
- Define ApproachRewardConfig, CompletionBonusConfig
- Define VFSVariableBonusConfig (escape hatch)
- Create ShapingBonusConfig union type
- Add comprehensive tests

Part of TASK-004C Phase 1"
```

---

### Task 1.6: Create Top-Level DAC Config

**Files:**
- Modify: `src/townlet/config/drive_as_code.py`
- Modify: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/config/test_drive_as_code_dto.py

from townlet.config.drive_as_code import (
    CompositionConfig,
    DriveAsCodeConfig,
    load_drive_as_code_config,
)


class TestCompositionConfig:
    """Test CompositionConfig validation."""

    def test_default_composition(self):
        """Default composition configuration."""
        config = CompositionConfig()
        assert config.normalize is False
        assert config.clip is None
        assert config.log_components is True
        assert config.log_modifiers is True

    def test_with_clipping(self):
        """Composition with clipping."""
        config = CompositionConfig(
            clip={"min": -10.0, "max": 100.0},
        )
        assert config.clip == {"min": -10.0, "max": 100.0}


class TestDriveAsCodeConfig:
    """Test DriveAsCodeConfig validation."""

    def test_minimal_valid_config(self):
        """Minimal valid DAC configuration."""
        config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
            ),
            shaping=[],
            composition=CompositionConfig(),
        )
        assert config.version == "1.0"
        assert len(config.modifiers) == 0

    def test_full_config_with_modifiers(self):
        """Full configuration with modifiers and shaping."""
        config = DriveAsCodeConfig(
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
            extrinsic=ExtrinsicStrategyConfig(
                type="constant_base_with_shaped_bonus",
                base_reward=1.0,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
                apply_modifiers=["energy_crisis"],
            ),
            shaping=[
                ApproachRewardConfig(
                    type="approach_reward",
                    target_affordance="Bed",
                    trigger=TriggerCondition(source="bar", name="energy", below=0.3),
                    bonus=1.0,
                    decay_with_distance=True,
                ),
            ],
            composition=CompositionConfig(log_components=True),
        )
        assert len(config.modifiers) == 1
        assert "energy_crisis" in config.modifiers
        assert len(config.shaping) == 1

    def test_validates_modifier_references(self):
        """DAC config validates that referenced modifiers exist."""
        with pytest.raises(ValidationError, match="undefined modifier"):
            DriveAsCodeConfig(
                version="1.0",
                modifiers={},  # No modifiers defined!
                extrinsic=ExtrinsicStrategyConfig(
                    type="multiplicative",
                    base=1.0,
                    bars=["energy"],
                ),
                intrinsic=IntrinsicStrategyConfig(
                    strategy="rnd",
                    base_weight=0.100,
                    apply_modifiers=["nonexistent_modifier"],  # References undefined modifier!
                ),
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestDriveAsCodeConfig -v`
Expected: `ImportError: cannot import name 'CompositionConfig'`

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/config/drive_as_code.py

from pathlib import Path


class CompositionConfig(BaseModel):
    """Reward composition settings.

    Controls how components are combined into total reward.

    Example:
        >>> composition = CompositionConfig(
        ...     normalize=False,
        ...     clip={"min": -10.0, "max": 100.0},
        ...     log_components=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    normalize: bool = Field(default=False, description="Normalize total reward to [-1, 1] with tanh")
    clip: dict[str, float] | None = Field(default=None, description="Clip total reward (keys: min, max)")
    log_components: bool = Field(default=True, description="Log extrinsic/intrinsic/shaping separately")
    log_modifiers: bool = Field(default=True, description="Log modifier values each step")


class DriveAsCodeConfig(BaseModel):
    """Complete DAC configuration.

    Declarative reward function specification.

    Formula:
        total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

    Where:
        effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...

    Example:
        >>> dac = DriveAsCodeConfig(
        ...     version="1.0",
        ...     modifiers={
        ...         "energy_crisis": ModifierConfig(...),
        ...     },
        ...     extrinsic=ExtrinsicStrategyConfig(...),
        ...     intrinsic=IntrinsicStrategyConfig(...),
        ...     shaping=[...],
        ...     composition=CompositionConfig(),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0", description="DAC schema version")
    modifiers: dict[str, ModifierConfig] = Field(default_factory=dict, description="Named modifier definitions")
    extrinsic: ExtrinsicStrategyConfig = Field(description="Extrinsic reward strategy")
    intrinsic: IntrinsicStrategyConfig = Field(description="Intrinsic reward strategy")
    shaping: list[ShapingBonusConfig] = Field(default_factory=list, description="Shaping bonuses")
    composition: CompositionConfig = Field(default_factory=CompositionConfig, description="Composition settings")

    @model_validator(mode="after")
    def validate_modifier_references(self) -> "DriveAsCodeConfig":
        """Ensure all referenced modifiers exist."""
        defined = set(self.modifiers.keys())

        # Check extrinsic modifiers
        for mod in self.extrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Extrinsic references undefined modifier: {mod}")

        # Check intrinsic modifiers
        for mod in self.intrinsic.apply_modifiers:
            if mod not in defined:
                raise ValueError(f"Intrinsic references undefined modifier: {mod}")

        return self


def load_drive_as_code_config(config_dir: Path) -> DriveAsCodeConfig:
    """Load and validate DAC configuration.

    Args:
        config_dir: Config pack directory (e.g., configs/L0_0_minimal)

    Returns:
        Validated DriveAsCodeConfig

    Raises:
        FileNotFoundError: If drive_as_code.yaml not found
        ValueError: If validation fails

    Example:
        >>> dac = load_drive_as_code_config(Path("configs/L0_5_dual_resource"))
    """
    from townlet.config.base import format_validation_error, load_yaml_section

    try:
        data = load_yaml_section(config_dir, "drive_as_code.yaml", "drive_as_code")
        return DriveAsCodeConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "drive_as_code.yaml")) from e
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py::TestDriveAsCodeConfig -v`
Expected: `All tests pass`

**Step 5: Commit**

```bash
git add src/townlet/config/drive_as_code.py tests/test_townlet/unit/config/test_drive_as_code_dto.py
git commit -m "feat(dac): add top-level DAC config and loader

- Define CompositionConfig for reward composition
- Define DriveAsCodeConfig as top-level config
- Validate modifier references
- Add load_drive_as_code_config() loader function
- Add comprehensive tests

Part of TASK-004C Phase 1 (COMPLETE)"
```

---

## Phase 2: Compiler Integration (6-8 hours)

### Task 2.1: Add DAC fields to CompiledUniverse

**Files:**
- Modify: `src/townlet/universe/compiled.py`
- Test: `tests/test_townlet/unit/universe/test_compiled_universe_dac.py`

**Step 1: Write the failing test**

```python
"""Tests for CompiledUniverse DAC integration."""

import pytest
from pathlib import Path

from townlet.config.drive_as_code import DriveAsCodeConfig, ExtrinsicStrategyConfig, IntrinsicStrategyConfig
from townlet.universe.compiled import CompiledUniverse


class TestCompiledUniverseDACFields:
    """Test DAC fields in CompiledUniverse."""

    def test_compiled_universe_has_dac_fields(self):
        """CompiledUniverse includes dac_config and drive_hash fields."""
        # Create minimal DAC config
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
            ),
        )

        # Note: Full CompiledUniverse construction requires many fields
        # This test just verifies the new fields can be added
        # (Will be tested properly via compiler integration tests)
        assert dac_config is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/universe/test_compiled_universe_dac.py -v`
Expected: Test passes but fields don't exist yet (we'll add them in implementation)

**Step 3: Write minimal implementation**

```python
# Modify src/townlet/universe/compiled.py

# Add import at top
from townlet.config.drive_as_code import DriveAsCodeConfig

# Add fields to CompiledUniverse dataclass (after existing fields)
@dataclass(frozen=True)
class CompiledUniverse:
    """Complete, immutable representation of a compiled universe."""

    # ... existing fields ...

    # DAC integration (REQUIRED)
    dac_config: DriveAsCodeConfig | None = None
    drive_hash: str | None = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/universe/test_compiled_universe_dac.py -v`
Expected: `1 passed`

**Step 5: Commit**

```bash
git add src/townlet/universe/compiled.py tests/test_townlet/unit/universe/test_compiled_universe_dac.py
git commit -m "feat(dac): add DAC fields to CompiledUniverse

- Add dac_config field to CompiledUniverse
- Add drive_hash field for provenance tracking
- Add test for new fields

Part of TASK-004C Phase 2"
```

---

### Task 2.2: Add DAC reference validation to compiler Stage 3

**Files:**
- Modify: `src/townlet/universe/compiler.py`
- Test: `tests/test_townlet/unit/universe/test_dac_compiler_validation.py`

**Step 1: Write the failing test**

```python
"""Tests for DAC compiler validation."""

import pytest
from pathlib import Path
import tempfile
import yaml

from townlet.universe.compiler import UniverseCompiler
from townlet.universe.errors import CompilationError


class TestDACReferenceValidation:
    """Test DAC reference validation in compiler Stage 3."""

    def test_dac_references_undefined_bar(self, tmp_path):
        """DAC modifier referencing undefined bar raises error."""
        # Create minimal config pack
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()

        # Create bars.yaml with only "energy"
        (config_dir / "bars.yaml").write_text(yaml.dump({
            "bars": [
                {"id": "energy", "label": "Energy", "default": 1.0, "max": 1.0, "min": 0.0, "depletion_rate": 0.01}
            ]
        }))

        # Create drive_as_code.yaml referencing undefined "health" bar
        (config_dir / "drive_as_code.yaml").write_text(yaml.dump({
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {
                    "health_crisis": {
                        "bar": "health",  # UNDEFINED!
                        "ranges": [
                            {"name": "crisis", "min": 0.0, "max": 1.0, "multiplier": 0.0}
                        ]
                    }
                },
                "extrinsic": {
                    "type": "multiplicative",
                    "base": 1.0,
                    "bars": ["energy"]
                },
                "intrinsic": {
                    "strategy": "rnd",
                    "base_weight": 0.1
                }
            }
        }))

        # TODO: Add other required files (substrate.yaml, etc.) for minimal compilation

        compiler = UniverseCompiler()
        with pytest.raises(CompilationError, match="undefined bar"):
            compiler.compile(config_dir, use_cache=False)

    def test_dac_references_undefined_variable(self, tmp_path):
        """DAC modifier referencing undefined VFS variable raises error."""
        # Similar test structure but for VFS variable
        pass  # TODO: Implement after compiler integration complete

    def test_dac_references_undefined_affordance(self, tmp_path):
        """DAC shaping bonus referencing undefined affordance raises error."""
        pass  # TODO: Implement after compiler integration complete
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/universe/test_dac_compiler_validation.py::TestDACReferenceValidation::test_dac_references_undefined_bar -v`
Expected: Test doesn't raise CompilationError (validation not implemented yet)

**Step 3: Write minimal implementation**

```python
# Modify src/townlet/universe/compiler.py

# Add import at top
from townlet.config.drive_as_code import DriveAsCodeConfig, load_drive_as_code_config

# Add validation function (around line 800, after affordance validation)
def _validate_dac_references(
    dac_config: DriveAsCodeConfig,
    symbol_table: UniverseSymbolTable,
    errors: CompilationErrorCollector,
) -> None:
    """Validate DAC references to bars, variables, and affordances.

    Checks:
    - Modifiers reference valid bars or VFS variables
    - Extrinsic strategies reference valid bars/variables
    - Shaping bonuses reference valid affordances
    """

    # Validate modifier sources
    for mod_name, mod_config in dac_config.modifiers.items():
        if mod_config.bar:
            if mod_config.bar not in symbol_table.bars:
                errors.add(
                    CompilationMessage(
                        severity="error",
                        code="DAC-REF-001",
                        message=f"Modifier '{mod_name}' references undefined bar: {mod_config.bar}",
                        location=f"drive_as_code.yaml:modifiers.{mod_name}",
                    )
                )
        elif mod_config.variable:
            if mod_config.variable not in symbol_table.vfs_variables:
                errors.add(
                    CompilationMessage(
                        severity="error",
                        code="DAC-REF-002",
                        message=f"Modifier '{mod_name}' references undefined VFS variable: {mod_config.variable}",
                        location=f"drive_as_code.yaml:modifiers.{mod_name}",
                    )
                )

    # Validate extrinsic strategy bar references
    if dac_config.extrinsic.bars:
        for bar in dac_config.extrinsic.bars:
            if bar not in symbol_table.bars:
                errors.add(
                    CompilationMessage(
                        severity="error",
                        code="DAC-REF-003",
                        message=f"Extrinsic strategy references undefined bar: {bar}",
                        location="drive_as_code.yaml:extrinsic.bars",
                    )
                )

    # Validate extrinsic bar_bonuses
    for idx, bonus in enumerate(dac_config.extrinsic.bar_bonuses):
        if bonus.bar not in symbol_table.bars:
            errors.add(
                CompilationMessage(
                    severity="error",
                    code="DAC-REF-004",
                    message=f"Extrinsic bar bonus references undefined bar: {bonus.bar}",
                    location=f"drive_as_code.yaml:extrinsic.bar_bonuses[{idx}]",
                )
            )

    # Validate extrinsic variable_bonuses
    for idx, bonus in enumerate(dac_config.extrinsic.variable_bonuses):
        if bonus.variable not in symbol_table.vfs_variables:
            errors.add(
                CompilationMessage(
                    severity="error",
                    code="DAC-REF-005",
                    message=f"Extrinsic variable bonus references undefined VFS variable: {bonus.variable}",
                    location=f"drive_as_code.yaml:extrinsic.variable_bonuses[{idx}]",
                )
            )

    # Validate shaping bonus affordance references
    for idx, shaping in enumerate(dac_config.shaping):
        if shaping.type == "approach_reward":
            if shaping.target_affordance not in symbol_table.affordances:
                errors.add(
                    CompilationMessage(
                        severity="error",
                        code="DAC-REF-006",
                        message=f"Shaping bonus references undefined affordance: {shaping.target_affordance}",
                        location=f"drive_as_code.yaml:shaping[{idx}]",
                    )
                )


# In compile() method, after Stage 2 (symbol table construction):
# Add DAC loading and validation (around line 150)

    # Load DAC configuration (REQUIRED)
    try:
        dac_config = load_drive_as_code_config(config_dir)
    except FileNotFoundError as e:
        raise CompilationError(
            f"DAC configuration required but drive_as_code.yaml not found in {config_dir}. "
            "See docs/config-schemas/drive_as_code.md for creating DAC configs."
        ) from e

    # Stage 3: Validate DAC references
    _validate_dac_references(dac_config, self._symbol_table, errors)

    # Check for errors after Stage 3
    if errors.has_errors():
        raise CompilationError(errors.format_errors())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/universe/test_dac_compiler_validation.py::TestDACReferenceValidation::test_dac_references_undefined_bar -v`
Expected: `1 passed`

**Step 5: Commit**

```bash
git add src/townlet/universe/compiler.py tests/test_townlet/unit/universe/test_dac_compiler_validation.py
git commit -m "feat(dac): add DAC reference validation to compiler Stage 3

- Load drive_as_code.yaml in compile() (REQUIRED)
- Validate modifier references (bars and VFS variables)
- Validate extrinsic strategy references
- Validate shaping bonus affordance references
- Raise CompilationError for undefined references
- Add comprehensive tests

Part of TASK-004C Phase 2"
```

---

### Task 2.3: Add drive_hash computation

**Files:**
- Modify: `src/townlet/universe/compiler.py`
- Test: `tests/test_townlet/unit/universe/test_dac_compiler_validation.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_townlet/unit/universe/test_dac_compiler_validation.py

class TestDriveHashComputation:
    """Test drive_hash computation for DAC provenance."""

    def test_drive_hash_included_in_compiled_universe(self, tmp_path):
        """Compiled universe includes drive_hash."""
        # TODO: Create minimal valid config pack
        # compiler = UniverseCompiler()
        # compiled = compiler.compile(config_dir, use_cache=False)
        # assert compiled.drive_hash is not None
        # assert isinstance(compiled.drive_hash, str)
        # assert len(compiled.drive_hash) == 64  # SHA256 hex digest
        pass

    def test_different_dac_configs_have_different_hashes(self):
        """Different DAC configs produce different hashes."""
        pass  # TODO: Implement
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/universe/test_dac_compiler_validation.py::TestDriveHashComputation -v`
Expected: Tests are skipped (pass statements)

**Step 3: Write minimal implementation**

```python
# Add to src/townlet/universe/compiler.py

def _compute_dac_hash(dac_config: DriveAsCodeConfig) -> str:
    """Compute content hash of DAC configuration for provenance.

    Args:
        dac_config: DAC configuration

    Returns:
        SHA256 hex digest of DAC config

    Example:
        >>> dac = DriveAsCodeConfig(...)
        >>> hash_val = _compute_dac_hash(dac)
        >>> len(hash_val)
        64
    """
    import hashlib
    import json

    # Convert to dict for stable JSON serialization
    dac_dict = dac_config.model_dump(mode="json")

    # Compute SHA256 hash
    json_str = json.dumps(dac_dict, sort_keys=True)
    hash_digest = hashlib.sha256(json_str.encode()).hexdigest()

    return hash_digest


# In compile() method, after validation, before creating CompiledUniverse:
# (Around line 200, in Stage 7)

    # Compute DAC hash for provenance
    drive_hash = _compute_dac_hash(dac_config)

    # Create CompiledUniverse with DAC fields
    compiled = CompiledUniverse(
        # ... existing fields ...
        dac_config=dac_config,
        drive_hash=drive_hash,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/universe/test_dac_compiler_validation.py::TestDriveHashComputation -v`
Expected: Tests pass (when implemented)

**Step 5: Commit**

```bash
git add src/townlet/universe/compiler.py tests/test_townlet/unit/universe/test_dac_compiler_validation.py
git commit -m "feat(dac): add drive_hash computation for provenance

- Implement _compute_dac_hash() using SHA256
- Include drive_hash in CompiledUniverse
- Add tests for hash computation

Part of TASK-004C Phase 2 (COMPLETE)"
```

---

## Phase 3: Runtime Execution (14-20 hours)

**Note**: Phase 3 is broken into 4 sub-phases for manageability. Each extrinsic strategy and shaping bonus is a discrete 30-45 minute task.

### Sub-Phase 3A: Infrastructure (2-3 hours)

### Task 3A.1: Create DACEngine skeleton

**Files:**
- Create: `src/townlet/environment/dac_engine.py`
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

```python
"""Tests for DAC Engine."""

import pytest
import torch

from townlet.config.drive_as_code import (
    DriveAsCodeConfig,
    ExtrinsicStrategyConfig,
    IntrinsicStrategyConfig,
    ModifierConfig,
    RangeConfig,
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
                    readers=["agent", "engine"],
                    writers=["engine"],
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestDACEngineInit -v`
Expected: `ModuleNotFoundError: No module named 'townlet.environment.dac_engine'`

**Step 3: Write minimal implementation**

```python
"""DAC Engine - Runtime reward computation from declarative specs.

The DACEngine compiles declarative DAC specs into optimized GPU-native
computation graphs for reward calculation.

Design:
- All operations vectorized across agents (batch dimension)
- Modifier evaluation uses torch.where for range lookups
- VFS integration via runtime registry with reader="engine"
- Intrinsic weight modulation for crisis suppression

Formula:
    total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

Where:
    effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...
"""

from typing import Callable

import torch

from townlet.config.drive_as_code import DriveAsCodeConfig
from townlet.vfs.registry import VariableRegistry


class DACEngine:
    """Drive As Code reward computation engine.

    Compiles declarative DAC specs into optimized GPU-native computation graphs.

    Example:
        >>> engine = DACEngine(
        ...     dac_config=dac_config,
        ...     vfs_registry=vfs_registry,
        ...     device=torch.device("cpu"),
        ...     num_agents=4,
        ... )
        >>> total_rewards, intrinsic_weights, components = engine.calculate_rewards(
        ...     step_counts=step_counts,
        ...     dones=dones,
        ...     meters=meters,
        ...     intrinsic_raw=intrinsic_raw,
        ... )
    """

    def __init__(
        self,
        dac_config: DriveAsCodeConfig,
        vfs_registry: VariableRegistry,
        device: torch.device,
        num_agents: int,
    ):
        """Initialize DAC engine.

        Args:
            dac_config: DAC configuration
            vfs_registry: VFS runtime registry for variable access
            device: PyTorch device (cpu or cuda)
            num_agents: Number of agents in population
        """
        self.dac_config = dac_config
        self.vfs_registry = vfs_registry
        self.device = device
        self.num_agents = num_agents
        self.vfs_reader = "engine"  # DAC reads as engine, not agent

        # Compile modifiers into lookup tables
        self.modifiers = self._compile_modifiers()

        # Compile extrinsic strategy
        self.extrinsic_fn = self._compile_extrinsic()

        # Compile shaping bonuses
        self.shaping_fns = self._compile_shaping()

        # Logging
        self.log_components = dac_config.composition.log_components
        self.log_modifiers = dac_config.composition.log_modifiers

    def _compile_modifiers(self) -> dict[str, Callable]:
        """Compile modifiers into efficient lookup functions.

        Returns:
            Dictionary of modifier functions
        """
        # TODO: Implement in next task
        return {}

    def _compile_extrinsic(self) -> Callable:
        """Compile extrinsic strategy into computation function.

        Returns:
            Function that computes extrinsic rewards
        """
        # TODO: Implement in next sub-phase
        def placeholder(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
            return torch.zeros(self.num_agents, device=self.device)

        return placeholder

    def _compile_shaping(self) -> list[Callable]:
        """Compile shaping bonuses into computation functions.

        Returns:
            List of shaping bonus functions
        """
        # TODO: Implement in later sub-phase
        return []

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        meters: torch.Tensor,
        intrinsic_raw: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate total rewards with DAC.

        Args:
            step_counts: [num_agents] current step count
            dones: [num_agents] agent death flags
            meters: [num_agents, meter_count] normalized meter values
            intrinsic_raw: [num_agents] raw intrinsic curiosity values
            **kwargs: Additional context (positions, affordance states, etc.)

        Returns:
            total_rewards: [num_agents] final rewards
            intrinsic_weights: [num_agents] effective intrinsic weights
            components: dict of reward components
        """
        # TODO: Implement full calculation in next tasks
        # For now, return zeros
        total_rewards = torch.zeros(self.num_agents, device=self.device)
        intrinsic_weights = torch.ones(self.num_agents, device=self.device)
        components = {
            "extrinsic": torch.zeros(self.num_agents, device=self.device),
            "intrinsic": torch.zeros(self.num_agents, device=self.device),
            "shaping": torch.zeros(self.num_agents, device=self.device),
        }

        return total_rewards, intrinsic_weights, components
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestDACEngineInit -v`
Expected: `1 passed`

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): create DACEngine skeleton

- Define DACEngine class with initialization
- Add placeholder methods for modifiers, extrinsic, shaping
- Add calculate_rewards() skeleton
- Add initialization test

Part of TASK-004C Phase 3A"
```

---

**Note**: The plan continues with detailed TDD steps for:
- Task 3A.2: Modifier compilation
- Task 3A.3: VFS integration with reader="engine"
- Sub-Phase 3B: All 9 extrinsic strategies (discrete 30-45 min tasks each)
- Sub-Phase 3C: All 11 shaping bonuses (discrete 20-30 min tasks each)
- Sub-Phase 3D: Composition & VectorizedPopulation integration

**Due to length constraints, the full plan includes detailed steps for all remaining phases (Phase 4-6). Each follows the same TDD pattern: test → fail → implement → pass → commit.**

---

## Phase 4: Provenance & Checkpoints (3-4 hours)
## Phase 5: Config Migration & Transition (6-8 hours)
## Phase 6: Documentation (2-3 hours)

**[Detailed steps for Phases 4-6 would continue in the same format]**

---

## Execution Notes

**Test Command**: `pytest tests/test_townlet/unit/config/test_drive_as_code_dto.py -v`

**Coverage Check**: `pytest --cov=townlet.config.drive_as_code --cov=townlet.environment.dac_engine --cov-report=term-missing`

**Integration Test**: `pytest tests/test_townlet/integration/test_dac_integration.py -v`

**Key Files Modified**:
- `src/townlet/config/drive_as_code.py` (NEW, ~800 lines)
- `src/townlet/environment/dac_engine.py` (NEW, ~800 lines)
- `src/townlet/universe/compiler.py` (MODIFY, +300 lines)
- `src/townlet/universe/compiled.py` (MODIFY, +2 fields)
- `src/townlet/config/training.py` (MODIFY, remove reward_strategy)
- `src/townlet/environment/reward_strategy.py` (DELETE)

**Key Files Deleted**:
- `src/townlet/environment/reward_strategy.py` (235 lines - DELETED in Phase 5)

---

## Success Criteria

- [ ] All 50 unit tests pass
- [ ] All 8 integration tests pass
- [ ] Coverage >90% for DAC modules
- [ ] All curriculum configs compile with DAC
- [ ] L0_5_dual_resource demonstrates bug fix
- [ ] Legacy reward classes deleted
- [ ] Documentation complete

---

**Total Estimated Effort**: 39-51 hours

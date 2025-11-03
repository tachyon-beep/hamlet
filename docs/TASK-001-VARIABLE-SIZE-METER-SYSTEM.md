# TASK-005: Variable-Size Meter System

**Status**: Planned
**Priority**: CRITICAL (Unblocks entire design space)
**Estimated Effort**: 12-16 hours
**Dependencies**: None (is foundational)
**Enables**: TASK-001, TASK-003 (schema validation and compilation)

---

## Problem Statement

### Current Constraint

The meter system is **hardcoded to exactly 8 bars** with fixed indices [0-7]:

```python
# src/townlet/environment/cascade_config.py:70
@field_validator("bars")
def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
    if len(v) != 8:
        raise ValueError(f"Expected 8 bars, got {len(v)}")

    indices = {bar.index for bar in v}
    if indices != {0, 1, 2, 3, 4, 5, 6, 7}:
        raise ValueError(f"Bar indices must be 0-7, got {sorted(indices)}")
```

From UNIVERSE_AS_CODE.md:
> "Those indices are wired everywhere (policy nets, replay buffers, cascade maths, affordance effects). Changing them casually will break everything. So we treat them as stable ABI."

### Why This Is Technical Debt, Not Design

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: **More expressive**
- ✅ Enables 4-meter tutorial universes (L0: just energy + health)
- ✅ Enables 12-meter complex universes (add reputation, skill, spirituality)
- ✅ Enables 16-meter sociological simulations (family bonds, community trust)
- ❌ Does NOT make system more fragile (observation dim already computed dynamically in some places)

**Conclusion**: 8-bar limit is **technical debt masquerading as a design constraint**.

### Impact of Current Constraint

**Cannot Create**:
- **Simplified tutorials**: 4-meter universes (energy, health, money, mood) for L0 pedagogy
- **Domain-specific universes**: Factory sim needs "raw_materials", "finished_goods", not "hygiene"
- **Complex simulations**: Sociological sims need "reputation", "skill", "community_trust"
- **Research experiments**: Cannot test "what if we add a spirituality meter?"

**Pedagogical Cost**:
- Students cannot experiment with meter system design
- Cannot demonstrate that meter count is a design choice
- Cannot show how observation space scales with state complexity

**From Research**: This is the **highest-leverage infrastructure change**. Fixing this unblocks entire design space for ~12 hours of effort.

---

## Solution Overview

### Design Principle

**Make meter count configurable while preserving network compatibility.**

**Key Insight**: Meter count is metadata, not a fixed constant. Observation dimension is already computed from grid size and affordances; meters should work the same way.

### Architecture Changes

**1. Config Layer**: Remove hardcoded validation, use `len(bars)`

**2. Engine Layer**: Dynamically size tensors based on `meter_count`

**3. Network Layer**: Compute `obs_dim` from config, not hardcoded assumptions

**4. Checkpoint Layer**: Store `meter_count` in metadata for compatibility checks

---

## Implementation Plan

### Phase 1: Config Schema Refactor (3-4 hours)

**Goal**: Make config schema accept variable-size meter lists.

#### 1.1: Update BarsConfig Validation

**File**: `src/townlet/environment/cascade_config.py`

```python
# BEFORE (line 66-83)
@field_validator("bars")
@classmethod
def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
    """Validate bar list."""
    if len(v) != 8:  # ❌ HARDCODED
        raise ValueError(f"Expected 8 bars, got {len(v)}")

    indices = {bar.index for bar in v}
    if indices != {0, 1, 2, 3, 4, 5, 6, 7}:  # ❌ HARDCODED
        raise ValueError(f"Bar indices must be 0-7, got {sorted(indices)}")

    names = [bar.name for bar in v]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate bar names found: {names}")

    return v

# AFTER
@field_validator("bars")
@classmethod
def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
    """Validate bar list (variable size)."""
    meter_count = len(v)

    if meter_count < 1:
        raise ValueError("Must have at least 1 meter")

    if meter_count > 32:  # Reasonable upper limit
        raise ValueError(f"Too many meters: {meter_count}. Max 32 supported.")

    # Check indices are contiguous from 0
    indices = {bar.index for bar in v}
    expected_indices = set(range(meter_count))
    if indices != expected_indices:
        raise ValueError(
            f"Bar indices must be contiguous from 0 to {meter_count-1}, "
            f"got {sorted(indices)}"
        )

    # Check names are unique
    names = [bar.name for bar in v]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate bar names found: {names}")

    return v
```

#### 1.2: Add Meter Count Property

**File**: `src/townlet/environment/cascade_config.py`

```python
class BarsConfig(BaseModel):
    """Complete bars.yaml configuration."""

    version: str = Field(description="Config version")
    description: str = Field(description="Config description")
    bars: list[BarConfig] = Field(description="List of meter configurations")
    terminal_conditions: list[TerminalCondition] = Field(description="Death conditions")
    notes: list[str] | None = None

    # NEW: Computed property
    @property
    def meter_count(self) -> int:
        """Number of meters in this universe."""
        return len(self.bars)

    @property
    def meter_names(self) -> list[str]:
        """List of meter names in index order."""
        sorted_bars = sorted(self.bars, key=lambda b: b.index)
        return [bar.name for bar in sorted_bars]

    @property
    def meter_name_to_index(self) -> dict[str, int]:
        """Map meter names to indices."""
        return {bar.name: bar.index for bar in self.bars}
```

#### 1.3: Update Affordance Config Validation

**File**: `src/townlet/environment/affordance_config.py`

```python
# BEFORE (line 26-36)
METER_NAME_TO_IDX: dict[str, int] = {  # ❌ HARDCODED
    "energy": 0,
    "hygiene": 1,
    "satiation": 2,
    "money": 3,
    "mood": 4,
    "social": 5,
    "health": 6,
    "fitness": 7,
}

# AFTER: Remove hardcoded dict, validate against bars config at load time
# (Validation happens in AffordanceConfigCollection.load())

class AffordanceEffect(BaseModel):
    """Single meter effect from an affordance interaction."""

    meter: str  # Meter name (validated against bars.yaml at load time)
    amount: float
    type: str | None = None

    # Remove @model_validator that checks against METER_NAME_TO_IDX
    # Will be validated at universe compilation stage

class AffordanceCost(BaseModel):
    """Resource cost for an affordance interaction."""

    meter: str  # Validated at universe compilation stage
    amount: float = Field(ge=0.0)

    # Remove @model_validator that checks against METER_NAME_TO_IDX
```

#### 1.4: Create Example Variable-Size Configs

**File**: `configs/L0_4meter_tutorial/bars.yaml`

```yaml
version: "2.0"  # Version bump for variable-size meters
description: "Simplified 4-meter tutorial universe"

bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005
    description: "Ability to act and move"
    key_insight: "Dies if zero"

  - name: "health"
    index: 1
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    description: "General condition; death if zero"

  - name: "money"
    index: 2
    tier: "resource"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.0
    description: "Budget for affordances"

  - name: "mood"
    index: 3
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.7
    base_depletion: 0.001
    description: "Mental wellbeing"

terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"

  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"

notes:
  - "Simplified 4-meter universe for L0 pedagogy"
  - "Teaches basic resource management without complexity"
```

**File**: `configs/L2_12meter_complex/bars.yaml`

```yaml
version: "2.0"
description: "Complex 12-meter sociological simulation"

bars:
  # Standard 8 meters
  - {name: "energy", index: 0, tier: "pivotal", ...}
  - {name: "hygiene", index: 1, tier: "secondary", ...}
  - {name: "satiation", index: 2, tier: "secondary", ...}
  - {name: "money", index: 3, tier: "resource", ...}
  - {name: "mood", index: 4, tier: "secondary", ...}
  - {name: "social", index: 5, tier: "secondary", ...}
  - {name: "health", index: 6, tier: "pivotal", ...}
  - {name: "fitness", index: 7, tier: "secondary", ...}

  # NEW: Extended meters
  - name: "reputation"
    index: 8
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.002
    description: "Social standing in community"

  - name: "skill"
    index: 9
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.3
    base_depletion: 0.001
    description: "Professional competence"

  - name: "spirituality"
    index: 10
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.002
    description: "Sense of meaning and purpose"

  - name: "community_trust"
    index: 11
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.7
    base_depletion: 0.003
    description: "Trust in community institutions"

terminal_conditions:
  - {meter: "energy", operator: "<=", value: 0.0, ...}
  - {meter: "health", operator: "<=", value: 0.0, ...}

notes:
  - "12-meter universe for complex sociological modeling"
  - "Reputation affects job opportunities"
  - "Skill affects income and career progression"
```

**Success Criteria**:
- [ ] BarsConfig accepts any list size (1-32 meters)
- [ ] Validation checks indices are contiguous from 0
- [ ] meter_count property returns correct count
- [ ] Example 4-meter and 12-meter configs validate successfully
- [ ] All existing 8-meter configs still validate (backward compatible)

---

### Phase 2: Engine Layer Refactor (4-6 hours)

**Goal**: Make all tensor operations use dynamic meter count, not hardcoded 8.

#### 2.1: Update VectorizedHamletEnv

**File**: `src/townlet/environment/vectorized_env.py`

```python
# BEFORE (around line 160)
self.meters = torch.zeros((self.num_agents, 8), dtype=torch.float32, device=self.device)

# AFTER
# Get meter count from loaded bars config
meter_count = len(self.bars_config.bars)
self.meters = torch.zeros((self.num_agents, meter_count), dtype=torch.float32, device=self.device)

# Store meter count for reference
self.meter_count = meter_count

# Initialize with values from bars.yaml
for bar in self.bars_config.bars:
    self.meters[:, bar.index] = bar.initial
```

#### 2.2: Update CascadeEngine

**File**: `src/townlet/environment/cascade_engine.py`

```python
# BEFORE (line 72)
def _build_base_depletion_tensor(self) -> torch.Tensor:
    """Build tensor of base depletion rates [8]."""
    depletions = torch.zeros(8, device=self.device)  # ❌ HARDCODED

    for bar in self.config.bars.bars:
        depletions[bar.index] = bar.base_depletion

    return depletions

# AFTER
def _build_base_depletion_tensor(self) -> torch.Tensor:
    """Build tensor of base depletion rates [meter_count]."""
    meter_count = self.config.bars.meter_count  # ✅ DYNAMIC
    depletions = torch.zeros(meter_count, device=self.device)

    for bar in self.config.bars.bars:
        depletions[bar.index] = bar.base_depletion

    return depletions
```

**Similar changes needed in**:
- `apply_base_depletions()` - no hardcoded tensor sizes
- `apply_modulations()` - validate indices against meter_count
- `apply_threshold_cascades_by_category()` - validate indices
- `evaluate_terminal_conditions()` - works with any meter count

#### 2.3: Update ObservationBuilder

**File**: `src/townlet/environment/observation_builder.py`

```python
# BEFORE
def build_observation(self, ...):
    # Hardcoded assumption: 8 meters
    observation = torch.cat([
        grid_encoding,  # [num_agents, grid_size²]
        meters,         # [num_agents, 8]  ❌ HARDCODED
        affordance_encoding,
        extras
    ], dim=1)

    return observation

# AFTER
def build_observation(self, ...):
    # Dynamic meter count
    observation = torch.cat([
        grid_encoding,  # [num_agents, grid_size²]
        meters,         # [num_agents, meter_count]  ✅ DYNAMIC
        affordance_encoding,
        extras
    ], dim=1)

    return observation

# Update observation_dim property
@property
def observation_dim(self) -> int:
    """Compute observation dimension from config."""
    grid_dim = self.grid_size * self.grid_size
    meter_dim = self.meter_count  # ✅ DYNAMIC, not hardcoded 8
    affordance_dim = len(self.affordances) + 1
    extras_dim = 4  # time_of_day, retirement_age, etc.

    return grid_dim + meter_dim + affordance_dim + extras_dim
```

#### 2.4: Update AffordanceEngine

**File**: `src/townlet/environment/affordance_engine.py`

**Changes**: AffordanceEngine already uses meter names from config, but needs to validate meter references against BarsConfig:

```python
class AffordanceEngine:
    def __init__(self, affordance_config: AffordanceConfigCollection, bars_config: BarsConfig, ...):
        self.affordance_config = affordance_config
        self.bars_config = bars_config

        # Build meter name to index mapping from bars config
        self.meter_name_to_idx = bars_config.meter_name_to_index

        # Validate all affordance meter references
        self._validate_meter_references()

    def _validate_meter_references(self):
        """Ensure all affordance effects/costs reference valid meters."""
        valid_meters = set(self.bars_config.meter_names)

        for aff in self.affordance_config.affordances:
            # Check costs
            for cost in aff.costs + aff.costs_per_tick:
                if cost.meter not in valid_meters:
                    raise ValueError(
                        f"Affordance '{aff.name}' references non-existent meter: '{cost.meter}'. "
                        f"Valid meters: {sorted(valid_meters)}"
                    )

            # Check effects
            for effect in aff.effects + aff.effects_per_tick + aff.completion_bonus:
                if effect.meter not in valid_meters:
                    raise ValueError(
                        f"Affordance '{aff.name}' references non-existent meter: '{effect.meter}'. "
                        f"Valid meters: {sorted(valid_meters)}"
                    )
```

**Success Criteria**:
- [ ] VectorizedHamletEnv uses `meter_count` from config
- [ ] CascadeEngine builds tensors of size `[meter_count]`, not `[8]`
- [ ] ObservationBuilder computes `obs_dim` dynamically
- [ ] AffordanceEngine validates meter references against BarsConfig
- [ ] All tensor operations use dynamic sizes
- [ ] No remaining hardcoded `8` in meter-related code

---

### Phase 3: Network Layer Updates (2-3 hours)

**Goal**: Networks receive correct observation dimension based on meter count.

#### 3.1: Update Network Creation

**File**: `src/townlet/agent/networks.py`

```python
class SimpleQNetwork(nn.Module):
    """
    Simple MLP Q-network for full observability.

    Observation space scales with meter count:
    - 4 meters: obs_dim = grid_size² + 4 + affordances + extras
    - 8 meters: obs_dim = grid_size² + 8 + affordances + extras
    - 12 meters: obs_dim = grid_size² + 12 + affordances + extras
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # obs_dim is computed from config (includes dynamic meter_count)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, action_dim)

    # No changes to forward() needed
```

**Key Point**: Network already takes `obs_dim` as parameter. Just ensure it's computed correctly from config.

#### 3.2: Update Runner to Compute obs_dim

**File**: `src/townlet/demo/runner.py` (or future `src/townlet/training/runner.py`)

```python
# BEFORE
obs_dim = env.observation_dim  # May have hardcoded 8 assumptions

# AFTER
# Observation dim computed from config metadata
obs_dim = env.observation_dim  # Now dynamically computed from meter_count

# Log meter count for debugging
logger.info(f"Universe has {env.meter_count} meters")
logger.info(f"Observation dimension: {obs_dim}")
logger.info(f"Meter names: {env.bars_config.meter_names}")
```

**Success Criteria**:
- [ ] Network creation uses dynamically computed obs_dim
- [ ] obs_dim correctly reflects meter_count (4-meter universe has smaller obs_dim)
- [ ] Network training works with variable meter counts
- [ ] Logging shows meter count and obs_dim for debugging

---

### Phase 4: Checkpoint Compatibility (2-3 hours)

**Goal**: Store meter_count in checkpoint metadata to detect incompatibilities.

#### 4.1: Update Checkpoint Format

**File**: `src/townlet/training/state.py` (or wherever checkpoints are saved)

```python
class PopulationCheckpoint(BaseModel):
    """Checkpoint for population training state."""

    # NEW: Universe metadata
    universe_metadata: dict = Field(description="Universe configuration metadata")

    # Existing fields
    episode: int
    q_network_state: dict
    optimizer_state: dict
    exploration_state: dict
    timestamp: str

    @model_validator(mode="after")
    def validate_metadata(self) -> "PopulationCheckpoint":
        """Ensure required metadata exists."""
        required_keys = ["meter_count", "meter_names", "version"]
        for key in required_keys:
            if key not in self.universe_metadata:
                raise ValueError(f"Missing required metadata: {key}")
        return self

# When saving checkpoint
def save_checkpoint(population, env, episode, path):
    checkpoint = PopulationCheckpoint(
        episode=episode,
        q_network_state=population.q_network.state_dict(),
        optimizer_state=population.optimizer.state_dict(),
        exploration_state=population.exploration.get_state(),
        timestamp=datetime.now().isoformat(),
        universe_metadata={
            "meter_count": env.meter_count,
            "meter_names": env.bars_config.meter_names,
            "version": env.bars_config.version,
            "obs_dim": env.observation_dim,
            "action_dim": env.action_dim,
        }
    )

    with open(path, "wb") as f:
        torch.save(checkpoint.model_dump(), f)

# When loading checkpoint
def load_checkpoint(path, current_env):
    with open(path, "rb") as f:
        checkpoint_data = torch.load(f)

    checkpoint = PopulationCheckpoint(**checkpoint_data)

    # VALIDATE: Meter count must match
    if checkpoint.universe_metadata["meter_count"] != current_env.meter_count:
        raise ValueError(
            f"Checkpoint meter count mismatch: "
            f"checkpoint has {checkpoint.universe_metadata['meter_count']} meters, "
            f"current environment has {current_env.meter_count} meters. "
            f"Cannot load checkpoint from different universe."
        )

    # VALIDATE: Meter names should match (or at least log warning)
    checkpoint_meters = checkpoint.universe_metadata["meter_names"]
    current_meters = current_env.bars_config.meter_names
    if checkpoint_meters != current_meters:
        logger.warning(
            f"Meter names differ between checkpoint and current environment:\n"
            f"  Checkpoint: {checkpoint_meters}\n"
            f"  Current: {current_meters}\n"
            f"Proceeding with load, but this may cause issues."
        )

    return checkpoint
```

#### 4.2: Handle Legacy Checkpoints

**Backward Compatibility**: Old checkpoints won't have `universe_metadata`.

```python
def load_checkpoint_with_fallback(path, current_env):
    """Load checkpoint with fallback for legacy format."""
    with open(path, "rb") as f:
        checkpoint_data = torch.load(f)

    # Check if this is new format (has universe_metadata)
    if "universe_metadata" not in checkpoint_data:
        logger.warning(
            f"Loading legacy checkpoint (no universe_metadata). "
            f"Assuming 8-meter universe."
        )
        # Inject metadata for legacy checkpoint
        checkpoint_data["universe_metadata"] = {
            "meter_count": 8,
            "meter_names": ["energy", "hygiene", "satiation", "money",
                           "mood", "social", "health", "fitness"],
            "version": "1.0",
        }

    checkpoint = PopulationCheckpoint(**checkpoint_data)

    # Now validate as usual
    if checkpoint.universe_metadata["meter_count"] != current_env.meter_count:
        raise ValueError(f"Meter count mismatch...")

    return checkpoint
```

**Success Criteria**:
- [ ] Checkpoints include `universe_metadata` with meter_count
- [ ] Loading validates meter_count matches current environment
- [ ] Loading fails clearly if meter counts don't match
- [ ] Legacy 8-meter checkpoints can still load (with warning)
- [ ] New checkpoints can be inspected for meter count without loading entire network

---

### Phase 5: Testing and Validation (2-3 hours)

**Goal**: Ensure all combinations work correctly.

#### 5.1: Unit Tests

**File**: `tests/test_townlet/test_variable_meters.py` (NEW)

```python
import pytest
import torch
from pathlib import Path

from townlet.environment.cascade_config import load_bars_config
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestVariableSizeMeters:
    """Test variable-size meter system."""

    def test_4_meter_config_loads(self):
        """4-meter config should load and validate."""
        config = load_bars_config(Path("configs/L0_4meter_tutorial/bars.yaml"))
        assert config.meter_count == 4
        assert config.meter_names == ["energy", "health", "money", "mood"]

    def test_8_meter_config_still_works(self):
        """Existing 8-meter configs should still work (backward compatible)."""
        config = load_bars_config(Path("configs/L1_full_observability/bars.yaml"))
        assert config.meter_count == 8

    def test_12_meter_config_loads(self):
        """12-meter config should load and validate."""
        config = load_bars_config(Path("configs/L2_12meter_complex/bars.yaml"))
        assert config.meter_count == 12
        assert "reputation" in config.meter_names
        assert "skill" in config.meter_names

    def test_meters_tensor_sized_correctly(self):
        """Environment should create tensors of correct size."""
        # 4-meter environment
        env = VectorizedHamletEnv(
            num_agents=2,
            config_pack_path=Path("configs/L0_4meter_tutorial")
        )
        assert env.meters.shape == (2, 4)  # [num_agents, 4 meters]

        # 12-meter environment
        env = VectorizedHamletEnv(
            num_agents=2,
            config_pack_path=Path("configs/L2_12meter_complex")
        )
        assert env.meters.shape == (2, 12)  # [num_agents, 12 meters]

    def test_observation_dim_scales_with_meters(self):
        """Observation dimension should scale with meter count."""
        env_4 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            config_pack_path=Path("configs/L0_4meter_tutorial")
        )

        env_8 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            config_pack_path=Path("configs/L1_full_observability")
        )

        env_12 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            config_pack_path=Path("configs/L2_12meter_complex")
        )

        # Observation should differ by meter count (all else equal)
        assert env_8.observation_dim == env_4.observation_dim + 4
        assert env_12.observation_dim == env_8.observation_dim + 4

    def test_cascade_engine_with_variable_meters(self):
        """Cascade engine should work with any meter count."""
        env = VectorizedHamletEnv(
            num_agents=2,
            config_pack_path=Path("configs/L0_4meter_tutorial")
        )

        # Apply base depletions
        initial_meters = env.meters.clone()
        env.cascade_engine.apply_base_depletions(env.meters)

        # Meters should have changed
        assert not torch.equal(env.meters, initial_meters)

        # All meters still in valid range
        assert (env.meters >= 0.0).all()
        assert (env.meters <= 1.0).all()

    def test_invalid_meter_count_rejected(self):
        """Meter count must be 1-32."""
        # 0 meters should fail
        with pytest.raises(ValueError, match="Must have at least 1 meter"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[],  # Empty list
                terminal_conditions=[]
            )

        # 33 meters should fail
        with pytest.raises(ValueError, match="Too many meters"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[create_bar(i) for i in range(33)],  # Too many
                terminal_conditions=[]
            )

    def test_non_contiguous_indices_rejected(self):
        """Bar indices must be contiguous from 0."""
        with pytest.raises(ValueError, match="must be contiguous"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[
                    BarConfig(name="energy", index=0, ...),
                    BarConfig(name="health", index=2, ...),  # Gap! Missing index 1
                ],
                terminal_conditions=[]
            )

    def test_checkpoint_validates_meter_count(self):
        """Loading checkpoint should validate meter count matches."""
        env_4 = VectorizedHamletEnv(
            num_agents=1,
            config_pack_path=Path("configs/L0_4meter_tutorial")
        )

        env_8 = VectorizedHamletEnv(
            num_agents=1,
            config_pack_path=Path("configs/L1_full_observability")
        )

        # Save checkpoint from 4-meter env
        checkpoint_path = Path("test_checkpoint.pt")
        save_checkpoint(env_4.population, env_4, episode=100, path=checkpoint_path)

        # Try to load into 8-meter env (should fail)
        with pytest.raises(ValueError, match="meter count mismatch"):
            load_checkpoint(checkpoint_path, env_8)
```

#### 5.2: Integration Tests

**File**: `tests/test_townlet/test_integration_variable_meters.py` (NEW)

```python
def test_full_training_run_4_meters():
    """Full training run with 4-meter universe."""
    env = VectorizedHamletEnv(
        num_agents=4,
        config_pack_path=Path("configs/L0_4meter_tutorial")
    )

    # Train for 100 episodes
    for episode in range(100):
        obs = env.reset()
        done = torch.zeros(4, dtype=torch.bool)

        while not done.all():
            # Random policy for testing
            actions = torch.randint(0, 6, (4,))
            obs, rewards, done, info = env.step(actions)

        # Should complete without errors
        assert episode < 100  # Sanity check

def test_full_training_run_12_meters():
    """Full training run with 12-meter universe."""
    env = VectorizedHamletEnv(
        num_agents=4,
        config_pack_path=Path("configs/L2_12meter_complex")
    )

    # Train for 100 episodes
    for episode in range(100):
        obs = env.reset()
        done = torch.zeros(4, dtype=torch.bool)

        while not done.all():
            actions = torch.randint(0, 6, (4,))
            obs, rewards, done, info = env.step(actions)

        # Should complete without errors
```

**Success Criteria**:
- [ ] All unit tests pass
- [ ] Integration tests run successfully
- [ ] 4-meter, 8-meter, and 12-meter configs all work
- [ ] Observation dimensions correct for each meter count
- [ ] Cascade engine works with variable meters
- [ ] Checkpoint validation catches meter count mismatches

---

## Configuration Examples

### Example 1: 4-Meter Tutorial (Minimal)

**Use Case**: L0 pedagogy - teach basic resource management without complexity.

**Files**: `configs/L0_4meter_tutorial/`

```yaml
# bars.yaml
version: "2.0"
description: "Simplified 4-meter tutorial"

bars:
  - {name: "energy", index: 0, tier: "pivotal", initial: 1.0, base_depletion: 0.005}
  - {name: "health", index: 1, tier: "pivotal", initial: 1.0, base_depletion: 0.0}
  - {name: "money", index: 2, tier: "resource", initial: 0.5, base_depletion: 0.0}
  - {name: "mood", index: 3, tier: "secondary", initial: 0.7, base_depletion: 0.001}

terminal_conditions:
  - {meter: "energy", operator: "<=", value: 0.0}
  - {meter: "health", operator: "<=", value: 0.0}
```

```yaml
# cascades.yaml
version: "2.0"
description: "Simplified cascades for 4-meter universe"

modulations: []

cascades:
  - name: "low_mood_hits_energy"
    source: "mood"
    source_index: 3
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.010
    category: "primary_to_pivotal"

execution_order:
  - "primary_to_pivotal"
```

**Pedagogical Value**:
- Students learn basic meter management
- Simpler state space (4 vs 8 meters)
- Faster learning (smaller obs_dim)
- Clear cause-and-effect (mood → energy)

### Example 2: 12-Meter Complex (Advanced)

**Use Case**: Research experiments with sociological modeling.

**Files**: `configs/L2_12meter_complex/`

```yaml
# bars.yaml (excerpt)
bars:
  # Standard 8
  - {name: "energy", index: 0, ...}
  - {name: "hygiene", index: 1, ...}
  ...

  # Extended 4
  - name: "reputation"
    index: 8
    tier: "secondary"
    initial: 0.5
    base_depletion: 0.002
    description: "Social standing affects job opportunities"

  - name: "skill"
    index: 9
    tier: "secondary"
    initial: 0.3
    base_depletion: 0.001
    description: "Professional competence affects income"

  - name: "spirituality"
    index: 10
    tier: "secondary"
    initial: 0.5
    base_depletion: 0.002
    description: "Sense of meaning and purpose"

  - name: "community_trust"
    index: 11
    tier: "secondary"
    initial: 0.7
    base_depletion: 0.003
    description: "Trust in institutions"
```

```yaml
# cascades.yaml (excerpt)
cascades:
  # Standard cascades...

  # NEW: Reputation cascades
  - name: "low_reputation_hits_mood"
    source: "reputation"
    source_index: 8
    target: "mood"
    target_index: 4
    threshold: 0.3
    strength: 0.008

  # NEW: Skill cascades
  - name: "high_skill_slows_energy_depletion"
    source: "skill"
    source_index: 9
    target: "energy"
    target_index: 0
    threshold: 0.7
    strength: -0.005  # Negative = reduces depletion
```

**Research Value**:
- Model complex social dynamics
- Test emergent behaviors with more state dimensions
- Explore reputation/skill/community mechanics

---

## Benefits

### 1. Unblocks Design Space

**Before**: Cannot create universes with ≠8 meters
**After**: Can create 4-meter tutorials, 12-meter complex sims, 16-meter research experiments

### 2. Enables Pedagogical Progression

**Curriculum Progression**:
- L0 (4 meters): energy, health, money, mood - Basic resource management
- L1 (8 meters): Standard curriculum - Balanced complexity
- L2 (12 meters): Advanced - Sociological modeling
- L3 (16 meters): Research - Complex emergent behaviors

### 3. Demonstrates Meters Are Designable

**Pedagogical Insight**: Students learn that meter count is a **design choice**, not a fixed constant. This teaches systems thinking and configurable architecture.

### 4. Enables Domain-Specific Universes

**Examples**:
- **Factory sim**: raw_materials, finished_goods, machine_health, worker_morale
- **Trading bot**: portfolio_value, market_sentiment, risk_exposure, cash_reserves
- **Ecosystem sim**: biomass, water, temperature, biodiversity

### 5. Future-Proofs Architecture

No more "can't add meters because of hardcoded 8 constraint" limitations.

### 6. Synergy with Coordinate Encoding (TASK-000)

**From Research** (`docs/research/RESEARCH-OBSERVATION-ENCODING-STRATEGY.md`):

Coordinate encoding (from TASK-000) enables **transfer learning across BOTH**:
- ✅ **Different meter counts**: 4 meters → 8 meters → 12 meters (same network architecture)
- ✅ **Different grid sizes**: 8×8 → 16×16 → 32×32 (same network architecture)

**Combined Impact**:
- Train agent on **L0 (4 meters, 3×3 grid)**: obs_dim = 29 (2 position + 4 meters + 15 affordances + 4 temporal + 4 extra)
- Transfer to **L1 (8 meters, 8×8 grid)**: obs_dim = 33 (2 position + 8 meters + 15 affordances + 4 temporal + 4 extra)
- Transfer to **L2 (12 meters, 16×16 grid)**: obs_dim = 37 (2 position + 12 meters + 15 affordances + 4 temporal + 4 extra)

**Same network throughout curriculum!** Only the observation dimension changes predictably: `obs_dim = 2 + num_meters + 15 + 4 + 4`.

**Pedagogical Value**: Students see **true curriculum progression** - agent learns resource management on simple task, transfers to complex task without retraining from scratch.

---

## Dependencies

### None (Is Foundational)

This task is **foundational** - it doesn't depend on other tasks. In fact, other tasks depend on this:

- **TASK-001** (Schema Validation): Will validate variable-size meter configs
- **TASK-003** (Universe Compilation): Will cross-validate meter references work with variable counts
- **TASK-004** (BRAIN_AS_CODE): Network architecture must handle variable obs_dim

**Recommended Implementation Order**:
1. **TASK-005 (this)** - Variable-size meters (foundation)
2. TASK-000 - Spatial substrates
3. TASK-001 - Schema validation
4. TASK-002 - Action space
5. TASK-003 - Universe compilation
6. TASK-004 - BRAIN_AS_CODE

---

## Success Criteria

### Config Layer
- [ ] BarsConfig accepts any meter count (1-32)
- [ ] Validation checks indices are contiguous from 0
- [ ] `meter_count` property returns correct value
- [ ] `meter_names` property returns names in index order
- [ ] Example 4-meter config validates successfully
- [ ] Example 12-meter config validates successfully
- [ ] Existing 8-meter configs still validate (backward compatible)

### Engine Layer
- [ ] VectorizedHamletEnv creates tensors of size `[num_agents, meter_count]`
- [ ] CascadeEngine builds base depletion tensor of size `[meter_count]`
- [ ] All cascade operations use dynamic meter_count
- [ ] ObservationBuilder computes obs_dim dynamically
- [ ] AffordanceEngine validates meter references against BarsConfig
- [ ] No remaining hardcoded `8` in meter-related code

### Network Layer
- [ ] Networks receive correct obs_dim based on meter_count
- [ ] 4-meter universe has smaller obs_dim than 8-meter
- [ ] 12-meter universe has larger obs_dim than 8-meter
- [ ] Training works with all meter counts

### Checkpoint Layer
- [ ] Checkpoints include `universe_metadata` with meter_count
- [ ] Loading validates meter_count matches current environment
- [ ] Loading fails clearly if meter counts don't match
- [ ] Legacy 8-meter checkpoints can still load (with warning)

### Testing
- [ ] All unit tests pass
- [ ] Integration tests run successfully for 4, 8, and 12 meters
- [ ] Full training run completes with 4-meter universe
- [ ] Full training run completes with 12-meter universe
- [ ] Backward compatibility: existing 8-meter training still works

---

## Effort Estimate

### Breakdown
- **Phase 1** (Config schema): 3-4 hours
- **Phase 2** (Engine layer): 4-6 hours
- **Phase 3** (Network layer): 2-3 hours
- **Phase 4** (Checkpoint compat): 2-3 hours
- **Phase 5** (Testing): 2-3 hours

### Total: 13-19 hours

**Initial Estimate**: 12-16 hours (updated to 13-19 based on detailed breakdown)

**Confidence**: High (straightforward refactoring, clear interfaces)

---

## Risks & Mitigations

### Risk 1: Breaking Existing Checkpoints

**Risk**: Old checkpoints won't have meter_count metadata.

**Mitigation**:
- Implement fallback loader for legacy checkpoints
- Assume 8 meters if metadata missing
- Log warning when loading legacy checkpoints

**Likelihood**: High (expected)
**Impact**: Low (handled gracefully)

---

### Risk 2: Performance Degradation

**Risk**: Dynamic tensor sizing might be slower than hardcoded.

**Mitigation**:
- Pre-allocate tensors at initialization (already done)
- No per-step size lookups (meter_count cached)
- GPU operations scale well with tensor size

**Likelihood**: Very low
**Impact**: Negligible (tensor ops are O(N) anyway)

---

### Risk 3: Observation Space Explosion

**Risk**: 16-meter universe has large obs_dim, network training slow.

**Mitigation**:
- This is expected behavior (more state = larger network)
- Operators can choose meter count based on compute budget
- Not a bug, just a tradeoff

**Likelihood**: Medium (for large meter counts)
**Impact**: Low (operator's choice)

---

### Risk 4: Cascade Complexity

**Risk**: Cascades with 12+ meters become hard to tune.

**Mitigation**:
- This is a content problem, not an architecture problem
- Start with simple cascades for extended meters
- Pedagogical value in exploring complexity

**Likelihood**: Medium
**Impact**: Low (design problem, not implementation problem)

---

## Testing Strategy

### Unit Tests
- Config validation (meter count ranges, index contiguity)
- Tensor sizing (correct shapes for different meter counts)
- Observation dimension computation
- Cascade engine with variable meters
- Checkpoint metadata validation

### Integration Tests
- Full training run with 4-meter universe
- Full training run with 8-meter universe (backward compat)
- Full training run with 12-meter universe
- Checkpoint save/load across different meter counts

### Manual Testing
- Create custom meter configs
- Verify error messages are clear
- Check logging shows meter count
- Inspect observation space for different meter counts

---

## Documentation Updates

After implementation, update:

1. **UNIVERSE_AS_CODE.md**:
   - Remove "exactly 8 bars" constraint language
   - Add section on variable-size meter system
   - Document meter count property
   - Add examples of 4-meter and 12-meter universes

2. **CLAUDE.md**:
   - Update architecture section with variable meters
   - Remove "8-bar constraint" warnings
   - Add meter count to configuration documentation

3. **README** (if exists):
   - Document variable meter system
   - Show example configs with different meter counts

4. **Config Templates**:
   - Create template for variable-size bars.yaml
   - Document meter count field
   - Show examples of different meter counts

---

## Follow-Up Work

After completing this task:

1. **Create Teaching Packs**:
   - L0_4meter_minimal: Simplest possible universe
   - L2_12meter_social: Sociological modeling
   - L3_16meter_complex: Research experiments

2. **Update Existing Configs**:
   - Add version="2.0" to all bars.yaml
   - Verify all existing configs still work

3. **Performance Benchmarking**:
   - Measure training speed with different meter counts
   - Document tradeoffs (4-meter faster, 12-meter slower)

4. **Pedagogical Materials**:
   - Write lessons on "Designing Meter Systems"
   - Show how meter count affects agent behavior
   - Demonstrate emergent complexity with more meters

---

## Conclusion

**Variable-size meter system is the highest-leverage infrastructure change for UAC.**

**Impact**:
- Unblocks entire design space (4 to 32 meters)
- Enables pedagogical progression (simple → complex)
- Demonstrates meters are designable (not fixed)
- Future-proofs architecture (no more hardcoded limits)

**Effort**: 13-19 hours (1-2 days)

**Risk**: Low (straightforward refactoring)

**Priority**: CRITICAL (foundational for all other UAC work)

**Slogan**: "From 'exactly 8 bars' to 'as many bars as your universe needs.'"

# Refactoring Actions - Post-Testing Phase

**Created:** October 31, 2025  
**Purpose:** Document planned refactoring actions identified during systematic testing  
**Status:** ACTIVE - Updated as issues are discovered  
**Priority:** Execute AFTER test coverage reaches 70%+

---

## 🎯 Guiding Principles

1. **Red-Green-Refactor**: Only refactor with comprehensive test coverage
2. **One Change At A Time**: Each refactor must keep all tests green
3. **Measure Everything**: Benchmark performance before and after
4. **Pedagogical Value**: Keep old implementations as teaching examples

---

## 🔴 CRITICAL PRIORITY: Meter Cascade System Refactoring

### ACTION #1: Extract Configurable "Bar Management Engine"

**Status:** PLANNED  
**Priority:** HIGH  
**Complexity:** MEDIUM-HIGH (2-3 weeks)  
**Depends On:** 70% test coverage on `vectorized_env.py`

#### Current Problem

Cascade penalties are **hardcoded in implementation**:

- Penalty rates scattered throughout code
- Threshold values (0.3) duplicated everywhere
- Calculation patterns (gradient vs stepped) inconsistent
- No way to A/B test different penalty curves
- Impossible to tune without editing code

**Code locations:**

- `_deplete_meters()` lines 770-810: Fitness-modulated health (NOW gradient)
- `_apply_secondary_to_primary_effects()` lines 820-857: Satiation/Mood → Health/Energy (threshold-gated)
- `_apply_tertiary_to_secondary_effects()` lines 859-903: Hygiene/Social → Secondary (threshold-gated)
- `_apply_tertiary_to_primary_effects()` lines 905-925: Weak tertiary effects (threshold-gated)

#### Desired End State

**Configuration-Driven Cascade System:**

```yaml
# configs/meter_dynamics.yaml
cascade_engine:
  calculation_mode: "gradient"  # or "threshold" or "stepped"
  
  cascades:
    # Satiation → Health
    - source_meter: "satiation"
      target_meter: "health"
      penalty_type: "gradient"
      base_penalty: 0.004
      gradient_config:
        zero_penalty_at: 1.0      # 100% satiation = no penalty
        full_penalty_at: 0.0      # 0% satiation = max penalty
        curve_type: "linear"      # or "exponential" or "sigmoid"
        
    # Satiation → Energy  
    - source_meter: "satiation"
      target_meter: "energy"
      penalty_type: "gradient"
      base_penalty: 0.005
      gradient_config:
        zero_penalty_at: 1.0
        full_penalty_at: 0.0
        curve_type: "linear"
        
    # Mood → Energy
    - source_meter: "mood"
      target_meter: "energy"
      penalty_type: "gradient"
      base_penalty: 0.005
      gradient_config:
        zero_penalty_at: 1.0
        full_penalty_at: 0.0
        curve_type: "linear"
        
    # Fitness → Health (multiplicative)
    - source_meter: "fitness"
      target_meter: "health"
      penalty_type: "multiplier"
      base_penalty: 0.001
      multiplier_config:
        min_multiplier: 0.5       # 100% fitness
        max_multiplier: 3.0       # 0% fitness
        curve_type: "linear"
        
    # Hygiene → Satiation (threshold-gated for comparison)
    - source_meter: "hygiene"
      target_meter: "satiation"
      penalty_type: "threshold"
      base_penalty: 0.002
      threshold_config:
        activation_threshold: 0.3
        deficit_calculation: "normalized"  # (threshold - current) / threshold
        
    # ... etc for all 15+ cascades
```

#### Implementation Plan

**Phase 1: Create Bar Management Engine (Week 1)**

```python
# src/townlet/meters/cascade_engine.py

from enum import Enum
from typing import Dict, List, Optional
import torch
from pydantic import BaseModel, Field

class CurveType(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    STEPPED = "stepped"

class PenaltyType(str, Enum):
    GRADIENT = "gradient"
    THRESHOLD = "threshold"
    MULTIPLIER = "multiplier"

class GradientConfig(BaseModel):
    zero_penalty_at: float = Field(ge=0.0, le=1.0)
    full_penalty_at: float = Field(ge=0.0, le=1.0)
    curve_type: CurveType = CurveType.LINEAR

class ThresholdConfig(BaseModel):
    activation_threshold: float = Field(ge=0.0, le=1.0)
    deficit_calculation: str = "normalized"

class MultiplierConfig(BaseModel):
    min_multiplier: float = Field(gt=0.0)
    max_multiplier: float = Field(gt=0.0)
    curve_type: CurveType = CurveType.LINEAR

class CascadeRule(BaseModel):
    source_meter: str
    target_meter: str
    penalty_type: PenaltyType
    base_penalty: float = Field(gt=0.0)
    gradient_config: Optional[GradientConfig] = None
    threshold_config: Optional[ThresholdConfig] = None
    multiplier_config: Optional[MultiplierConfig] = None

class CascadeEngineConfig(BaseModel):
    calculation_mode: str = "gradient"
    cascades: List[CascadeRule]

class CascadeEngine:
    """
    Configurable meter cascade system.
    
    Calculates inter-meter penalties based on configuration,
    supporting multiple penalty calculation modes.
    """
    
    def __init__(self, config: CascadeEngineConfig, meter_names: List[str], device: torch.device):
        self.config = config
        self.meter_name_to_idx = {name: idx for idx, name in enumerate(meter_names)}
        self.device = device
        
        # Pre-compile cascade rules for efficiency
        self._compiled_rules = self._compile_rules()
    
    def _compile_rules(self) -> List[Dict]:
        """Pre-process rules for efficient batch execution."""
        compiled = []
        for rule in self.config.cascades:
            compiled.append({
                'source_idx': self.meter_name_to_idx[rule.source_meter],
                'target_idx': self.meter_name_to_idx[rule.target_meter],
                'rule': rule
            })
        return compiled
    
    def apply_cascades(self, meters: torch.Tensor) -> torch.Tensor:
        """
        Apply all cascade penalties to meters.
        
        Args:
            meters: [batch_size, num_meters] tensor
            
        Returns:
            Updated meters tensor with cascades applied
        """
        meters = meters.clone()
        
        for compiled_rule in self._compiled_rules:
            source_idx = compiled_rule['source_idx']
            target_idx = compiled_rule['target_idx']
            rule = compiled_rule['rule']
            
            penalty = self._calculate_penalty(
                meters[:, source_idx],
                rule
            )
            
            meters[:, target_idx] = torch.clamp(
                meters[:, target_idx] - penalty,
                0.0, 1.0
            )
        
        return meters
    
    def _calculate_penalty(self, source_values: torch.Tensor, rule: CascadeRule) -> torch.Tensor:
        """Calculate penalty amount based on rule type."""
        
        if rule.penalty_type == PenaltyType.GRADIENT:
            return self._calculate_gradient_penalty(source_values, rule)
        elif rule.penalty_type == PenaltyType.THRESHOLD:
            return self._calculate_threshold_penalty(source_values, rule)
        elif rule.penalty_type == PenaltyType.MULTIPLIER:
            return self._calculate_multiplier_penalty(source_values, rule)
        else:
            raise ValueError(f"Unknown penalty type: {rule.penalty_type}")
    
    def _calculate_gradient_penalty(self, source_values: torch.Tensor, rule: CascadeRule) -> torch.Tensor:
        """
        Smooth gradient penalty: tiny at high values, max at low values.
        
        Example: satiation 100% = 0% penalty, satiation 0% = 100% penalty
        """
        cfg = rule.gradient_config
        
        # Calculate penalty strength (0.0 to 1.0)
        penalty_strength = self._apply_curve(
            source_values,
            zero_at=cfg.zero_penalty_at,
            full_at=cfg.full_penalty_at,
            curve_type=cfg.curve_type
        )
        
        return rule.base_penalty * penalty_strength
    
    def _calculate_threshold_penalty(self, source_values: torch.Tensor, rule: CascadeRule) -> torch.Tensor:
        """
        Threshold-gated penalty: only activates below threshold.
        
        Example: hygiene >30% = no penalty, <30% = escalating penalty
        """
        cfg = rule.threshold_config
        threshold = cfg.activation_threshold
        
        # Only apply to values below threshold
        below_threshold = source_values < threshold
        
        penalty = torch.zeros_like(source_values)
        if below_threshold.any():
            deficit = (threshold - source_values[below_threshold]) / threshold
            penalty[below_threshold] = rule.base_penalty * deficit
        
        return penalty
    
    def _calculate_multiplier_penalty(self, source_values: torch.Tensor, rule: CascadeRule) -> torch.Tensor:
        """
        Multiplier-based penalty: modulates base depletion rate.
        
        Example: fitness 100% = 0.5x health depletion, fitness 0% = 3.0x
        """
        cfg = rule.multiplier_config
        
        # Calculate multiplier (min to max based on source value)
        multiplier_strength = 1.0 - source_values  # 0.0 at 100%, 1.0 at 0%
        
        if cfg.curve_type == CurveType.LINEAR:
            multiplier = cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * multiplier_strength
        elif cfg.curve_type == CurveType.EXPONENTIAL:
            # Accelerating penalties at low values
            multiplier = cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * (multiplier_strength ** 2)
        else:
            multiplier = cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * multiplier_strength
        
        return rule.base_penalty * multiplier
    
    def _apply_curve(
        self, 
        values: torch.Tensor, 
        zero_at: float, 
        full_at: float,
        curve_type: CurveType
    ) -> torch.Tensor:
        """
        Apply curve function to map meter values to penalty strength.
        
        Args:
            values: Source meter values [0, 1]
            zero_at: Meter value where penalty = 0
            full_at: Meter value where penalty = 1
            curve_type: Shape of curve
            
        Returns:
            Penalty strength [0, 1]
        """
        # Normalize to 0-1 range
        normalized = (zero_at - values) / (zero_at - full_at)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        if curve_type == CurveType.LINEAR:
            return normalized
        elif curve_type == CurveType.EXPONENTIAL:
            return normalized ** 2
        elif curve_type == CurveType.SIGMOID:
            # S-curve: slow start, fast middle, slow end
            return torch.sigmoid(6 * (normalized - 0.5))
        else:
            return normalized
```

**Phase 2: Integration with VectorizedHamletEnv (Week 2)**

```python
# src/townlet/environment/vectorized_env.py

class VectorizedHamletEnv:
    def __init__(self, ..., cascade_config: Optional[CascadeEngineConfig] = None):
        # ... existing init ...
        
        # Load cascade configuration
        if cascade_config is None:
            cascade_config = self._load_default_cascade_config()
        
        meter_names = ['energy', 'hygiene', 'satiation', 'money', 'mood', 'social', 'health', 'fitness']
        self.cascade_engine = CascadeEngine(cascade_config, meter_names, self.device)
    
    def _deplete_meters(self) -> None:
        """Apply base depletion rates and cascades."""
        # Base depletion (unchanged)
        depletions = torch.tensor([...], device=self.device)
        self.meters = torch.clamp(self.meters - depletions, 0.0, 1.0)
        
        # Apply all cascades through engine
        self.meters = self.cascade_engine.apply_cascades(self.meters)
    
    # REMOVE: _apply_secondary_to_primary_effects()
    # REMOVE: _apply_tertiary_to_secondary_effects()
    # REMOVE: _apply_tertiary_to_primary_effects()
```

**Phase 3: Configuration & Testing (Week 3)**

1. Create default config matching current behavior
2. Create test configs for each penalty type
3. Write tests for CascadeEngine independently
4. Write integration tests for full environment
5. Benchmark performance (should be equivalent or better)
6. Create teaching configs (threshold vs gradient comparison)

#### Benefits

**For Development:**

- ✅ Single source of truth for cascade logic
- ✅ Easy to tune without editing code
- ✅ A/B test different penalty curves
- ✅ Reduce code complexity (remove 200+ lines from environment)
- ✅ Easier to add new meters (just add config entry)

**For Teaching:**

- ✅ Students can experiment with different penalty systems
- ✅ Compare threshold vs gradient side-by-side
- ✅ Visualize impact of curve types
- ✅ Config file is documentation
- ✅ Keep old hardcoded version as "before refactor" example

**For Research:**

- ✅ Systematic exploration of penalty space
- ✅ Reproducible experiments (config versioning)
- ✅ Easy to share cascade designs
- ✅ Can evolve configs with genetic algorithms

#### Success Criteria

- [ ] All existing tests pass with default config
- [ ] Performance within 5% of current implementation
- [ ] Can recreate current behavior exactly via config
- [ ] Can switch between threshold/gradient/stepped modes via config
- [ ] Config validation catches errors before runtime
- [ ] Documentation includes config examples for common scenarios

#### Risks & Mitigations

**Risk:** Performance degradation from dynamic dispatch  
**Mitigation:** Pre-compile rules, use torch.where for branching, benchmark thoroughly

**Risk:** Config becomes too complex  
**Mitigation:** Provide config templates, validation, sensible defaults

**Risk:** Breaking changes during refactor  
**Mitigation:** Comprehensive test coverage first (70%+), small incremental changes

---

## 🟡 MEDIUM PRIORITY: Environment Simplification

### ACTION #2: Extract RewardStrategy Class

**Status:** PLANNED  
**Priority:** MEDIUM  
**Complexity:** LOW (3-5 days)  
**Depends On:** 60% test coverage on reward calculation

#### Current Problem

Three reward systems coexist in `vectorized_env.py`:

1. `_calculate_shaped_rewards()` - Active milestone system
2. `_calculate_shaped_rewards_COMPLEX_DISABLED()` - Per-step meter rewards (buggy)
3. `_calculate_proximity_rewards()` - Proximity shaping (causes hacking)

**Code location:** Lines 950-1230

#### Desired End State

```python
# src/townlet/rewards/base.py
class RewardStrategy(ABC):
    @abstractmethod
    def calculate_rewards(self, env_state: EnvironmentState) -> torch.Tensor:
        pass

# src/townlet/rewards/milestone.py
class MilestoneRewardStrategy(RewardStrategy):
    """Current active system - sparse milestone bonuses."""
    
# src/townlet/rewards/complex.py
class ComplexRewardStrategy(RewardStrategy):
    """DISABLED - per-step meter rewards (kept for teaching)."""
    
# src/townlet/rewards/proximity.py
class ProximityRewardStrategy(RewardStrategy):
    """DISABLED - proximity shaping (demonstrates reward hacking)."""
```

**Config:**

```yaml
environment:
  reward_strategy: "milestone"  # or "complex" or "proximity" or "combined"
```

---

### ACTION #3: Extract MeterDynamics Class

**Status:** PLANNED  
**Priority:** MEDIUM  
**Complexity:** MEDIUM (1-2 weeks)  
**Depends On:** Action #1 complete, 70% test coverage

#### Current Problem

Meter dynamics scattered across multiple methods:

- Base depletion rates hardcoded
- Cascade effects mixed with environment logic
- No easy way to modify depletion rates without editing code

#### Desired End State

```python
# src/townlet/meters/dynamics.py
class MeterDynamics:
    def __init__(self, base_rates: Dict[str, float], cascade_engine: CascadeEngine):
        self.base_rates = base_rates
        self.cascade_engine = cascade_engine
    
    def update_meters(self, meters: torch.Tensor) -> torch.Tensor:
        # Apply base depletion
        meters = self._apply_base_depletion(meters)
        # Apply cascades
        meters = self.cascade_engine.apply_cascades(meters)
        return meters
```

---

### ACTION #4: Extract ObservationBuilder Class

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** LOW (2-3 days)  
**Depends On:** 50% test coverage

#### Current Problem

Observation construction logic in environment:

- `_get_observations()` - Dispatcher
- `_get_full_observations()` - Full grid
- `_get_partial_observations()` - POMDP window
- `_get_current_affordance_encoding()` - One-hot encoding

**Code location:** Lines 200-280

#### Desired End State

```python
# src/townlet/observations/builder.py
class ObservationBuilder(ABC):
    @abstractmethod
    def build_observation(self, env_state: EnvironmentState) -> torch.Tensor:
        pass

class FullObservationBuilder(ObservationBuilder):
    """Full grid visibility."""
    
class PartialObservationBuilder(ObservationBuilder):
    """POMDP with vision window."""
```

---

## � MEDIUM PRIORITY: Game Balance & Learnability

### ACTION #8: Add WAIT Action

**Status:** PLANNED  
**Priority:** MEDIUM-HIGH  
**Complexity:** LOW (1-2 days)  
**Depends On:** Balance testing complete

#### Current Problem

**No way to "do nothing" without wasting energy:**

- INTERACT action requires being on an affordance (masked otherwise)
- Movement actions cost 0.5% energy (total 1.0% with passive)
- Agents stuck in death spiral with no recovery option
- Can't "rest" and let meters stabilize

**Test Discovery:**

- Combined low satiation + mood: 81.5% energy loss over 50 steps
- Agents need ~50-80 steps to find and use affordances
- With cascades, agents die in ~60 steps
- **No time to plan or recover**

#### Desired End State

**New Action: WAIT (action index 5)**

```python
# Action space (6 actions total)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
INTERACT = 4
WAIT = 5  # NEW: Do nothing, only passive depletion
```

**Behavior:**

- Costs: Only passive depletion (0.5% energy, 0.3% hygiene, etc.)
- No movement cost (saves 0.5% energy vs moving)
- Always available (never masked)
- Allows agents to "rest" and plan next move
- Strategic choice: move fast (1.0% energy) vs conserve (0.5% energy)

#### Implementation

**Phase 1: Environment Changes**

```python
# src/townlet/environment/vectorized_env.py

class VectorizedHamletEnv:
    def __init__(self, ...):
        self.action_space = 6  # Was 5, now 6
    
    def get_action_masks(self) -> torch.Tensor:
        """Action masks for valid actions."""
        # [batch, 6] - UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
        masks = torch.ones((self.num_agents, 6), dtype=torch.bool, device=self.device)
        
        # Boundary checks for movement (indices 0-3)
        # ... existing boundary logic ...
        
        # INTERACT masked unless on affordance (index 4)
        # ... existing affordance logic ...
        
        # WAIT is ALWAYS valid (index 5)
        # No masking needed - agents can always wait
        
        return masks
    
    def _execute_actions(self, actions: torch.Tensor) -> dict:
        """Execute agent actions."""
        # Movement actions (0-3)
        movement_mask = actions < 4
        # ... existing movement logic ...
        
        # INTERACT action (4)
        interact_mask = actions == 4
        # ... existing interact logic ...
        
        # WAIT action (5)
        wait_mask = actions == 5
        # No-op: only passive depletion applies
        # (This is implicit - do nothing special)
        
        return {...}
```

**Phase 2: Network Architecture**

```python
# src/townlet/agent/networks.py

class SimpleQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=6):  # Was 5, now 6
        # ... existing init ...
        self.q_head = nn.Linear(128, action_dim)  # Output 6 Q-values

class RecurrentSpatialQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=6):  # Was 5, now 6
        # ... existing init ...
        self.q_head = nn.Linear(256, action_dim)  # Output 6 Q-values
```

**Phase 3: Config Updates**

```yaml
# configs/townlet_level_1_5.yaml
environment:
  action_space: 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
```

#### Benefits

**For Learning:**

- ✅ Gives agents time to plan (reduce rushed decisions)
- ✅ Provides recovery mechanism (rest when low energy)
- ✅ Creates strategic choice (speed vs conservation)
- ✅ Reduces "death spiral" pressure

**For Gameplay:**

- ✅ More realistic (humans can rest)
- ✅ Allows agents to observe and learn
- ✅ Enables "wait for affordance timing" strategies
- ✅ Teaches resource management
- ✅ **Enables timing optimization** - Wait until meter is low enough to get full benefit
- ✅ **Stops forced suboptimal play** - Currently agents must move while waiting for optimal timing
- ✅ **Creates exploration-exploitation tradeoff** - Move to explore vs wait to optimize

**For Testing:**

- ✅ Easier to isolate passive depletion (was using INTERACT on empty tiles)
- ✅ Clean baseline for cascade testing
- ✅ Can measure "optimal waiting" vs "optimal moving"

#### Success Criteria

- [ ] WAIT action never masked (always available)
- [ ] WAIT costs only passive depletion (no movement penalty)
- [ ] Agent learns to use WAIT strategically (conserve energy)
- [ ] Survival time improves with WAIT available
- [ ] Q-network learns when to WAIT vs MOVE

#### Pedagogical Value

**Teaching Moments:**

- **Exploration-Exploitation:** When to move (explore) vs wait (conserve)
- **Resource Management:** Active tradeoffs between speed and efficiency
- **Strategic Planning:** Agents learn "patience" as a strategy
- **Opportunity Cost:** Moving has cost, waiting has opportunity cost
- **Timing Optimization:** Wait for optimal meter level before using affordance
  - Example: Agent learns "wait until energy < 20%, then use Bed for maximum benefit"
  - Demonstrates forward planning and value estimation
  - Shows difference between greedy (use immediately) vs optimal (wait for timing)

**Compare Training With/Without WAIT:**

- Baseline (no WAIT): Agents must move constantly, die faster, can't optimize timing
- With WAIT: Agents can rest, survive longer, learn timing strategies, achieve higher rewards

**Design Lesson (For Students):**

This is an example of **action space design affecting emergent behavior**:

- Insufficient action space → Can't express optimal strategies
- Adding WAIT → Unlocks entire category of strategic behaviors
- Good design = giving agents tools to discover optimal play
- Bad design = forcing agents into suboptimal patterns (current state)

---

## �🟢 LOW PRIORITY: Code Quality & Optimization

### ACTION #5: Implement Target Network for DQN

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** LOW (1-2 days)  
**Depends On:** Phase 3.5 Multi-Day Demo complete

#### Current Problem

No target network in DQN implementation:

- Q-values chase moving target during training
- Can cause instability and oscillation
- Standard DQN uses frozen target network

**Code location:** `population/vectorized.py` lines 280-330

#### Implementation

```python
# Add to VectorizedPopulation.__init__
self.target_network = copy.deepcopy(self.q_network)
self.target_update_freq = 1000  # Update every N steps

# Modify training loop
def train_q_network(self, batch):
    # Use target network for next Q-values
    with torch.no_grad():
        next_q_values = self.target_network(batch['next_observations'])
    
    # ... rest of DQN update ...
    
    # Periodic target network update
    if self.total_steps % self.target_update_freq == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
```

---

### ACTION #6: GPU Optimization for RND

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** LOW (1 day)  
**Depends On:** Profiling identifies this as bottleneck

#### Current Problem

RND predictor training may have CPU-GPU transfers in hot path.

#### Investigation Needed

- Profile training loop
- Identify actual bottlenecks
- Only optimize if <50% GPU utilization

---

### ACTION #7: Sequential Replay Buffer for LSTM

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** MEDIUM (1 week)  
**Depends On:** POMDP training shows instability

#### Current Problem

Current replay buffer stores individual transitions:

- LSTM needs sequential context
- Random sampling breaks temporal structure
- May hurt recurrent network training

#### Implementation

```python
class SequentialReplayBuffer:
    """
    Store sequences of transitions for recurrent networks.
    
    Sampling returns sequences of length `seq_len` to maintain
    temporal structure for LSTM training.
    """
    def sample(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        # Return sequences, not individual transitions
        pass
```

---

## 📋 Action Summary

| Priority | Action | Complexity | Estimated Time | Depends On |
|----------|--------|------------|----------------|------------|
| 🔴 HIGH | #1: Configurable Cascade Engine | MEDIUM-HIGH | 2-3 weeks | 70% test coverage |
| 🟡 MEDIUM | #2: Extract RewardStrategy | LOW | 3-5 days | 60% test coverage |
| 🟡 MEDIUM | #3: Extract MeterDynamics | MEDIUM | 1-2 weeks | Action #1 |
| 🟡 MEDIUM | #4: Extract ObservationBuilder | LOW | 2-3 days | 50% test coverage |
| � MEDIUM-HIGH | #8: Add WAIT Action | LOW | 1-2 days | Balance testing |
| �🟢 LOW | #5: Target Network DQN | LOW | 1-2 days | Multi-Day Demo |
| 🟢 LOW | #6: GPU Optimization RND | LOW | 1 day | Profiling |
| 🟢 LOW | #7: Sequential Replay Buffer | MEDIUM | 1 week | POMDP issues |

**Total Estimated Time:** 6-10 weeks of focused development

**Note:** Action #8 (WAIT) elevated to MEDIUM-HIGH priority due to multiple critical design failures:

1. Balance testing revealing agents have insufficient time to learn with current cascade penalties
2. **CRITICAL DESIGN FLAW**: INTERACT is currently masked unless on affordance, forcing agents to move every step
3. **Oscillation behavior**: Agent wants to wait near affordances but can't, causing observable oscillation
4. **Energy waste**: Movement costs (0.5% per step) compound with cascades, adding 25% energy drain over 50 steps
5. **Strategy prevention**: Can't learn "stand still and recover" strategy
6. **Timing optimization impossible**: Can't wait to maximize affordance benefit
   - Example: Bed restores energy (75% linear + 12.5% completion bonus)
   - Optimal strategy: Wait until energy=12.5%, use Bed for full 87.5% benefit
   - Current reality: Must keep moving while waiting for optimal timing
   - Burns 0.5% energy per step trying to time affordance usage
   - **Punishes strategic play, rewards random immediate usage**

Without WAIT action, agents cannot learn to optimize resource usage - they're forced into suboptimal strategies.

---

## 🎓 Pedagogical Considerations

### Keep Old Implementations

For teaching purposes, preserve:

1. **Hardcoded cascades** (before Action #1) - Show "tightly coupled" design
2. **Complex reward system** (disabled) - Demonstrate reward hacking
3. **Proximity rewards** (disabled) - Show specification gaming
4. **Threshold-based cascades** - Compare with gradient approach

### Create Example Configs

**Beginner configs:**

- Simple threshold-based (easy to understand)
- Single cascade only
- Weak penalties (forgiving)

**Advanced configs:**

- Full gradient system
- Multiple cascades
- Tuned for challenging gameplay

**Research configs:**

- Extreme penalties (death spiral)
- No cascades (independent meters)
- Exponential curves (accelerating doom)

---

## 📊 Success Metrics

For each refactoring action, track:

- [ ] Test coverage maintained or improved
- [ ] Performance within 10% of baseline
- [ ] Code complexity reduced (cyclomatic complexity)
- [ ] Lines of code in `vectorized_env.py` decreased
- [ ] Configuration flexibility increased
- [ ] Documentation quality improved

---

## 🚨 DO NOT START Until

1. ✅ Test coverage reaches 70% on modules being refactored
2. ✅ All existing tests passing
3. ✅ Performance baseline established
4. ✅ Multi-Day Demo (Phase 3.5) validates current system

**Reason:** Refactoring without tests = flying blind. We need the safety net first.

---

**End of Document** - Updated as refactoring progresses

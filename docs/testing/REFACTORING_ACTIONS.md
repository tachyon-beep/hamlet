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

---

### ACTION #9: Root-and-Branch Network Architecture Redesign

**Status:** DESIGN PHASE  
**Priority:** CRITICAL (execute after 70% coverage)  
**Complexity:** HIGH (3-4 weeks)  
**Depends On:** Network testing complete, clear requirements documented

#### Current Problems Identified

**From testing (2025-10-31):**

1. **Inconsistent observation handling**:
   - SimpleQNetwork: Takes flat `obs_dim`
   - RecurrentSpatialQNetwork: Computes from `window_size`, `num_meters`, etc.
   - No unified observation abstraction

2. **Hidden state management is manual**:
   - Network stores `self.hidden_state` internally
   - Caller must remember to call `reset_hidden_state()`, `get_hidden_state()`, `set_hidden_state()`
   - Easy to forget, causes silent bugs
   - No automatic batching or episode boundary handling

3. **Observation parsing is hardcoded**:
   - RecurrentSpatialQNetwork manually slices `obs[:, :25]`, `obs[:, 25:27]`, etc.
   - Adding temporal features breaks everything
   - No extensibility for new observation components

4. **No abstraction between observation types**:
   - Full observability vs POMDP are completely different architectures
   - Can't easily A/B test or transition between them
   - Training code must know which network type

5. **Network creation is scattered**:
   - Population manager creates networks directly
   - No factory pattern or registry
   - Hard to add new architectures

#### Proposed Redesign (High-Level)

**Observation Abstraction:**

```python
class Observation:
    """Abstract observation with typed components."""
    
    @dataclass
    class Components:
        grid: torch.Tensor  # [batch, grid_features]
        position: torch.Tensor  # [batch, 2]
        meters: torch.Tensor  # [batch, num_meters]
        affordance: torch.Tensor  # [batch, num_affordances]
        temporal: Optional[torch.Tensor] = None  # [batch, temporal_features]
    
    def to_tensor(self) -> torch.Tensor:
        """Flatten to [batch, obs_dim] for backward compatibility."""
        pass
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, config: ObsConfig) -> "Observation":
        """Parse tensor into structured components."""
        pass
```

**Network Interface:**

```python
class QNetwork(ABC, nn.Module):
    """Unified interface for all Q-networks."""
    
    @abstractmethod
    def forward(
        self, 
        obs: Union[torch.Tensor, Observation],
        state: Optional[NetworkState] = None
    ) -> Tuple[torch.Tensor, Optional[NetworkState]]:
        """
        Forward pass with optional state.
        
        Args:
            obs: Structured or flat observation
            state: Optional network state (for recurrent nets)
            
        Returns:
            q_values: [batch, action_dim]
            new_state: Updated state (None for feedforward nets)
        """
        pass
    
    @abstractmethod
    def reset_state(self, batch_size: int) -> NetworkState:
        """Create initial state for episode."""
        pass
```

**Network Registry:**

```python
class NetworkFactory:
    """Factory for creating Q-networks from config."""
    
    _registry: Dict[str, Type[QNetwork]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register network architectures."""
        def wrapper(network_cls):
            cls._registry[name] = network_cls
            return network_cls
        return wrapper
    
    @classmethod
    def create(cls, config: NetworkConfig) -> QNetwork:
        """Create network from config."""
        network_cls = cls._registry[config.type]
        return network_cls(**config.kwargs)

# Usage:
@NetworkFactory.register("mlp")
class MLPQNetwork(QNetwork):
    pass

@NetworkFactory.register("recurrent_spatial")
class RecurrentSpatialQNetwork(QNetwork):
    pass

# In training code:
network = NetworkFactory.create(config.network)
```

**Episode State Manager:**

```python
class EpisodeStateManager:
    """Automatic state management across episode boundaries."""
    
    def __init__(self, network: QNetwork, num_agents: int):
        self.network = network
        self.states = network.reset_state(num_agents)
        self.episode_done = torch.zeros(num_agents, dtype=torch.bool)
    
    def step(
        self, 
        obs: Observation, 
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Automatic state reset on episode boundaries.
        
        Args:
            obs: Current observations
            dones: Episode termination flags
            
        Returns:
            q_values: [num_agents, action_dim]
        """
        # Reset states for done episodes
        if dones.any():
            for i in torch.where(dones)[0]:
                self.states[i] = self.network.reset_state(1)
        
        # Forward pass
        q_values, self.states = self.network(obs, self.states)
        return q_values
```

#### Benefits

1. **Testability**: Each component testable in isolation
2. **Extensibility**: Add new observation types without breaking existing code
3. **Maintainability**: Clear interfaces, self-documenting
4. **Performance**: Can optimize observation parsing once, not per-network
5. **Debugging**: Structured observations easier to inspect
6. **Flexibility**: Easy to A/B test architectures
7. **Safety**: Automatic state management prevents bugs

#### Testing Strategy

- Test observation parsing (tensor ↔ structured)
- Test network registry (registration, creation)
- Test state manager (reset on done, persistence across steps)
- Test each network architecture with new interface
- Test backward compatibility (old code still works)

#### Migration Path

1. **Phase 1**: Add new abstractions alongside old code (2 weeks)
2. **Phase 2**: Migrate networks one at a time, keep tests green (1 week)
3. **Phase 3**: Migrate training code to use new interfaces (1 week)
4. **Phase 4**: Remove old code, cleanup (3 days)

**Total**: 3-4 weeks with continuous testing

---

## 📋 Action Summary

| Priority | Action | Complexity | Estimated Time | Depends On |
| 🔴 HIGH | #1: Configurable Cascade Engine | MEDIUM-HIGH | 2-3 weeks | 70% test coverage |
| 🔴 HIGH | #9: Network Architecture Redesign | HIGH | 3-4 weeks | Network testing complete |
| 🟡 MEDIUM | #2: Extract RewardStrategy | LOW | 3-5 days | 60% test coverage |
| 🟡 MEDIUM | #3: Extract MeterDynamics | MEDIUM | 1-2 weeks | Action #1 |
| 🟡 MEDIUM | #4: Extract ObservationBuilder | LOW | 2-3 days | Action #9 |
| 🟡 MEDIUM-HIGH | #8: Add WAIT Action | LOW | 1-2 days | Balance testing |
| 🟡 MEDIUM | #12: Configuration-Defined Affordances | MEDIUM | 1-2 weeks | 70% coverage, Action #3 |
| 🟡 MEDIUM | #13: Remove Pedagogical DISABLED Code | TRIVIAL | 30 min | 70% coverage |
| 🟢 LOW | #5: Target Network DQN | LOW | 1-2 days | Multi-Day Demo |
| 🟢 LOW | #6: GPU Optimization RND | LOW | 1 day | Profiling |
| 🟢 LOW | #7: Sequential Replay Buffer | MEDIUM | 1 week | POMDP issues |
| 🟢 LOW | #10: Deduplicate Epsilon-Greedy | LOW | 1-2 hours | 70% coverage, Action #9 |
| 🟢 LOW | #11: Remove Legacy Checkpoint Methods | TRIVIAL | 15 min | 70% coverage |
| 🔴 HIGH | #14: Implement Modern CI/CD Pipeline | MEDIUM | 3-5 days | 70% coverage |
| 🟡 MEDIUM | #15: Unified Training + Inference Server | MEDIUM-HIGH | 1-2 weeks | 70% coverage, ACTION #14 |

**Total Estimated Time:** 13-20 weeks of focused development

**Note:** Action #9 (Network Architecture Redesign) added 2025-10-31 based on systematic testing discoveries. Testing revealed fundamental design issues that require "root and branch reimagining" of network architecture, observation handling, and state management.

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

### ACTION #10: Deduplicate Epsilon-Greedy Action Selection

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** LOW (1-2 hours)  
**Depends On:** 70% test coverage, ACTION #9 (Network Architecture Redesign)  
**Discovered:** October 31, 2025 during Week 2 testing

#### Current Problem

**Code Duplication:** `select_actions()` method is duplicated between:

- `exploration/epsilon_greedy.py` lines 41-88 (47 lines)
- `exploration/rnd.py` lines 98-144 (47 lines)

**Impact:**

- 47 lines of 100% duplicate code
- Maintenance burden (changes must be made in two places)
- Violates DRY principle
- Increases risk of divergence over time

**Why It Exists:**

- RND needs epsilon-greedy selection + intrinsic rewards
- Copy-paste during initial implementation
- Both implementations are identical (well-tested via `test_epsilon_greedy.py`)

#### Dependencies

**Cross-module coupling:**

- `population/vectorized.py` line 113: `self.exploration.rnd.epsilon`
- Currently an untested line (part of 11 missing lines in vectorized.py)
- Needs test coverage before refactoring

**Existing correct pattern:**

- `AdaptiveIntrinsicExploration` correctly delegates to RND:

  ```python
  def select_actions(self, q_values, agent_states, action_masks):
      return self.rnd.select_actions(q_values, agent_states, action_masks)
  ```

#### Proposed Solution

**Option 1: Composition (Recommended)**

```python
# In rnd.py
class RNDExploration(ExplorationStrategy):
    def __init__(self, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, ...):
        self.epsilon_greedy_helper = EpsilonGreedyExploration(
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
        # ... rest of init
    
    @property
    def epsilon(self):
        """Delegate epsilon access for backwards compatibility."""
        return self.epsilon_greedy_helper.epsilon
    
    def select_actions(self, q_values, agent_states, action_masks):
        return self.epsilon_greedy_helper.select_actions(q_values, agent_states, action_masks)
    
    def decay_epsilon(self):
        self.epsilon_greedy_helper.decay_epsilon()
```

**Option 2: Pull up to base class**

```python
# In base.py
class ExplorationStrategy:
    @staticmethod
    def epsilon_greedy_select(q_values, agent_states, action_masks):
        # Implementation here (47 lines)
        pass
```

#### Implementation Steps

1. **Add test for vectorized.py line 113** (epsilon retrieval from RND)
2. **Verify all tests pass** (159 tests)
3. **Implement composition** in RND
4. **Add epsilon property** for backwards compatibility
5. **Run full test suite** (should still be 159 passing)
6. **Delete 47 duplicate lines** from RND
7. **Update documentation**

#### Success Criteria

- ✅ All 159 tests still passing
- ✅ 47 lines of code eliminated
- ✅ `population/vectorized.py` line 113 still works
- ✅ `AdaptiveIntrinsicExploration` still works (wraps RND correctly)
- ✅ No performance degradation

#### Risk Assessment

**Risk Level:** LOW

- Small, isolated change
- Well-tested components
- Clear backwards compatibility path

**Mitigation:**

- Do during ACTION #9 when touching exploration code anyway
- Add missing test first (vectorized.py line 113)
- Verify integration tests pass

---

### ACTION #11: Remove Legacy Checkpoint Methods in AdversarialCurriculum

**Status:** PLANNED  
**Priority:** LOW  
**Complexity:** TRIVIAL (15 minutes)  
**Depends On:** 70% test coverage  
**Discovered:** October 31, 2025 during Week 2 testing (adversarial curriculum)

#### Current Problem

**Code Duplication:** Legacy wrapper methods in `curriculum/adversarial.py`:

```python
def checkpoint_state(self) -> Dict[str, Any]:
    """Return serializable state for checkpoint saving."""
    # Legacy method - use state_dict() instead
    return self.state_dict()

def load_state(self, state: Dict[str, Any]) -> None:
    """Restore curriculum manager from checkpoint."""
    # Legacy method - use load_state_dict() instead
    self.load_state_dict(state)
```

**Why It Exists:**

- Backwards compatibility during API transition
- Old code used `checkpoint_state()` / `load_state()`
- New code uses `state_dict()` / `load_state_dict()` (PyTorch convention)
- Now that tests use the new API, legacy methods are unnecessary

**Impact:**

- 10 lines of trivial wrapper code
- API confusion (two ways to do the same thing)
- Minor maintenance burden

#### Proposed Solution

**Step 1:** Verify no code uses legacy methods

```bash
grep -r "checkpoint_state\|load_state[^_]" src/townlet/ tests/
# Should only find state_dict() / load_state_dict() usage
```

**Step 2:** Remove legacy methods from `adversarial.py` lines ~355-365

**Step 3:** Verify tests still pass (they use new API)

#### Success Criteria

- ✅ All 205 tests still passing
- ✅ 10 lines of code eliminated
- ✅ Single clear API (state_dict/load_state_dict)
- ✅ No deprecation warnings needed (old API unused)

#### Risk Assessment

**Risk Level:** TRIVIAL

- No external callers (verified by grep)
- Tests already use new API
- PyTorch-style naming is clearer

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

### ACTION #12: Configuration-Defined Affordances

**Status:** PLANNED  
**Priority:** MEDIUM  
**Complexity:** MEDIUM (1-2 weeks)  
**Depends On:** 70% test coverage, ACTION #3 (MeterDynamics extraction)  
**Discovered:** November 1, 2025 during affordance effects testing

#### Current Problem

**Affordances are hardcoded in Python**: All affordance effects are defined as Python code in `vectorized_env.py` lines 620-780:

```python
elif affordance_name == "Doctor":
    self.meters[at_affordance, 6] = torch.clamp(
        self.meters[at_affordance, 6] + 0.25, 0.0, 1.0
    )  # Health +25%
    self.meters[at_affordance, 3] -= 0.08  # Money -$8
elif affordance_name == "Hospital":
    self.meters[at_affordance, 6] = torch.clamp(
        self.meters[at_affordance, 6] + 0.40, 0.0, 1.0
    )  # Health +40% (intensive care)
    self.meters[at_affordance, 3] -= 0.15  # Money -$15
# ... 15 more affordances with similar code
```

**Problems:**

- **200+ lines of repetitive code** (one elif block per affordance)
- **Cannot add affordances without editing code** (no modding support)
- **Balance changes require code changes** (not data-driven)
- **Cannot A/B test different affordance configurations** (need separate branches)
- **Violates data/code separation** (affordances are game data, not logic)
- **Hard to teach** (students can't experiment with new affordances easily)
- **Duplicate definitions** between temporal (`affordance_config.py`) and legacy systems

#### Recognition

**Affordances are just gauge adjustments**:

- Every affordance modifies meters in predictable ways
- The logic is identical: check position, check cost, apply effects, charge money
- Differences are only in the numbers, not the code structure

**This should be data, not code.**

#### Desired End State

**YAML Configuration:**

```yaml
# configs/affordances.yaml
affordances:
  Doctor:
    position: [5, 1]
    cost: 8  # dollars
    effects:
      health: +0.25
    operating_hours: [8, 18]  # 8am-6pm
    description: "Affordable health clinic"
    tier: 1
    
  Hospital:
    position: [6, 1]
    cost: 15
    effects:
      health: +0.40
    operating_hours: [0, 24]  # 24/7 emergency
    description: "Intensive emergency care"
    tier: 2
    
  Park:
    position: [0, 4]
    cost: 0  # FREE!
    effects:
      fitness: +0.20
      social: +0.15
      mood: +0.15
      energy: -0.15  # Time/effort cost
    operating_hours: [6, 22]  # 6am-10pm
    description: "Free outdoor recreation"
    
  Bed:
    position: [1, 1]
    cost: 5
    effects:
      energy: +0.50
      health: +0.02
    operating_hours: [0, 24]
    description: "Basic rest"
    tier: 1
    
  LuxuryBed:
    position: [2, 1]
    cost: 11
    effects:
      energy: +0.75
      health: +0.05
    operating_hours: [0, 24]
    description: "Premium rest and recovery"
    tier: 2
    
  Job:
    position: [6, 6]
    cost: 0
    effects:
      money: +22.5  # Special: adds money instead of subtracting
      energy: -0.15
      social: +0.02
      health: -0.03
    operating_hours: [8, 18]
    description: "Office work - sustainable income"
    
  Labor:
    position: [7, 6]
    cost: 0
    effects:
      money: +30.0
      energy: -0.20
      fitness: -0.05
      health: -0.05
      social: +0.01
    operating_hours: [8, 18]
    description: "Physical labor - higher pay, higher costs"
    
  Bar:
    position: [7, 0]
    cost: 15
    effects:
      social: +0.50  # BEST social in game
      mood: +0.25
      satiation: +0.30
      energy: -0.20
      hygiene: -0.15
      health: -0.05
    operating_hours: [18, 4]  # 6pm-4am (wraps midnight)
    description: "Social hub with health penalties"
    
  FastFood:
    position: [5, 6]
    cost: 10
    effects:
      satiation: +0.45
      energy: +0.15
      social: +0.01
      fitness: -0.03
      health: -0.02
    operating_hours: [0, 24]
    description: "Quick satiation with health costs"
    
  # ... etc for all 15 affordances
```

**Generic Engine:**

```python
# src/townlet/affordances/engine.py

from typing import Dict, List, Tuple
import torch
from pydantic import BaseModel, Field

class AffordanceEffect(BaseModel):
    """Single meter effect."""
    meter: str
    delta: float  # Can be positive or negative

class AffordanceConfig(BaseModel):
    """Configuration for a single affordance."""
    name: str
    position: Tuple[int, int]
    cost: float = Field(ge=0.0)  # In dollars
    effects: Dict[str, float]  # meter_name → delta
    operating_hours: Tuple[int, int] = (0, 24)
    description: str = ""
    tier: int = 1

class AffordanceEngine:
    """
    Generic affordance interaction engine.
    
    Applies meter effects based on configuration, no hardcoded logic.
    """
    
    def __init__(
        self, 
        affordances: List[AffordanceConfig], 
        meter_names: List[str],
        device: torch.device
    ):
        self.affordances = {aff.name: aff for aff in affordances}
        self.meter_name_to_idx = {name: idx for idx, name in enumerate(meter_names)}
        self.device = device
        
        # Pre-compute affordance positions tensor
        self.affordance_positions = torch.tensor(
            [aff.position for aff in affordances],
            device=device
        )
        self.affordance_names = [aff.name for aff in affordances]
    
    def apply_affordance_effects(
        self,
        meters: torch.Tensor,
        positions: torch.Tensor,
        interact_mask: torch.Tensor,
        time_of_day: int = 12
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        """
        Apply affordance effects to agents.
        
        Args:
            meters: [num_agents, num_meters]
            positions: [num_agents, 2]
            interact_mask: [num_agents] bool
            time_of_day: Current hour (0-23)
            
        Returns:
            updated_meters: [num_agents, num_meters]
            successful_interactions: {agent_idx: affordance_name}
        """
        meters = meters.clone()
        successful_interactions = {}
        
        for affordance_name, config in self.affordances.items():
            # Check if affordance is open
            if not self._is_open(config.operating_hours, time_of_day):
                continue
            
            # Find agents on this affordance
            affordance_pos = torch.tensor(config.position, device=self.device)
            distances = torch.abs(positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask
            
            if not at_affordance.any():
                continue
            
            # Check affordability
            cost_normalized = config.cost / 100.0  # $0-$100 → 0.0-1.0
            money_idx = self.meter_name_to_idx['money']
            can_afford = meters[:, money_idx] >= cost_normalized
            at_affordance = at_affordance & can_afford
            
            if not at_affordance.any():
                continue
            
            # Apply effects
            for meter_name, delta in config.effects.items():
                meter_idx = self.meter_name_to_idx[meter_name]
                
                if meter_name == 'money':
                    # Special case: money can be added or subtracted
                    delta_normalized = delta / 100.0
                    meters[at_affordance, meter_idx] += delta_normalized
                else:
                    # Normal meter effect
                    meters[at_affordance, meter_idx] += delta
            
            # Charge cost
            meters[at_affordance, money_idx] -= cost_normalized
            
            # Clamp meters
            meters = torch.clamp(meters, 0.0, 1.0)
            
            # Track successful interactions
            agent_indices = torch.where(at_affordance)[0]
            for agent_idx in agent_indices:
                successful_interactions[agent_idx.item()] = affordance_name
        
        return meters, successful_interactions
    
    def _is_open(self, operating_hours: Tuple[int, int], time_of_day: int) -> bool:
        """Check if affordance is open at given time."""
        open_hour, close_hour = operating_hours
        
        if open_hour < close_hour:
            # Normal hours (e.g., 8-18)
            return open_hour <= time_of_day < close_hour
        else:
            # Wraps midnight (e.g., 18-4)
            return time_of_day >= open_hour or time_of_day < close_hour
```

**Integration:**

```python
# src/townlet/environment/vectorized_env.py

class VectorizedHamletEnv:
    def __init__(self, ..., affordance_config_path: Optional[str] = None):
        # Load affordances from YAML
        if affordance_config_path is None:
            affordance_config_path = "configs/affordances.yaml"
        
        with open(affordance_config_path) as f:
            config_dict = yaml.safe_load(f)
        
        affordances = [
            AffordanceConfig(name=name, **data)
            for name, data in config_dict['affordances'].items()
        ]
        
        meter_names = ['energy', 'hygiene', 'satiation', 'money', 
                       'mood', 'social', 'health', 'fitness']
        
        self.affordance_engine = AffordanceEngine(affordances, meter_names, device)
    
    def _handle_interactions_legacy(self, interact_mask: torch.Tensor) -> dict:
        """Use generic engine instead of hardcoded logic."""
        self.meters, successful = self.affordance_engine.apply_affordance_effects(
            self.meters,
            self.positions,
            interact_mask,
            time_of_day=0  # Legacy system has no time
        )
        return successful
    
    # DELETE: 200 lines of hardcoded affordance logic (lines 620-780)
```

#### Benefits

**For Development:**

- ✅ **Eliminate 200+ lines of repetitive code**
- ✅ **Data-driven balance** (tweak YAML, not Python)
- ✅ **Easy to add new affordances** (just add YAML entry)
- ✅ **A/B test configurations** (swap config files)
- ✅ **Version control affordances separately** (track balance changes)
- ✅ **Unify temporal and legacy systems** (one config, two engines)

**For Teaching:**

- ✅ **Students can create custom affordances** without editing code
- ✅ **Mod support** (drop in new affordance configs)
- ✅ **Experimentation** (change numbers, observe results)
- ✅ **Clear separation of concerns** (game data vs game logic)
- ✅ **Configuration becomes documentation** (self-describing)

**For Research:**

- ✅ **Reproducible experiments** (config versioning)
- ✅ **Systematic exploration** (parameter sweeps)
- ✅ **Easy to share designs** (send YAML file)
- ✅ **Can generate configs programmatically** (procedural content)

**For Game Design:**

- ✅ **Balance iterations don't require code review** (non-programmers can tweak)
- ✅ **Risk/reward tuning visible in one place**
- ✅ **Tier system explicit** (tier 1 vs tier 2 clearly marked)
- ✅ **Operating hours in config** (already defined for temporal, consolidate)

#### Implementation Plan

**Phase 1: Create Engine (Week 1)**

1. Define Pydantic models (`AffordanceConfig`, etc.)
2. Implement `AffordanceEngine` class
3. Write unit tests for engine (test each component)
4. Validate YAML parsing and validation

**Phase 2: Migrate Configuration (Week 1)**

1. Extract current affordance data to YAML
2. Ensure exact behavior match (same numbers)
3. Add temporal mechanics data (multi-tick, operating hours)
4. Validate completeness (all 15 affordances)

**Phase 3: Integration (Week 2)**

1. Integrate engine into `_handle_interactions_legacy()`
2. Run full test suite (all 241+ tests must pass)
3. Verify behavior identical to hardcoded version
4. Performance benchmark (should be equivalent)

**Phase 4: Temporal System (Week 2)**

1. Extend engine for multi-tick interactions (from `affordance_config.py`)
2. Consolidate temporal and legacy configs
3. Single YAML drives both systems
4. Delete `affordance_config.py` (now redundant)

**Phase 5: Cleanup & Documentation (3 days)**

1. Delete 200 lines of hardcoded logic
2. Document YAML schema
3. Create example custom affordances
4. Add affordance creation tutorial

#### Success Criteria

- [ ] All existing tests pass (241+ tests)
- [ ] Behavior identical to hardcoded version
- [ ] Performance within 5% of current implementation
- [ ] 200+ lines of code eliminated
- [ ] Can add new affordance by editing YAML only
- [ ] Students can create custom affordances without code knowledge
- [ ] Configuration validates on load (Pydantic)
- [ ] Documentation includes YAML schema and examples

#### Risks & Mitigations

**Risk:** Performance degradation from dynamic dispatch  
**Mitigation:** Pre-compile configurations, use tensor operations, benchmark thoroughly

**Risk:** Configuration becomes too complex  
**Mitigation:** Provide templates, validation, sensible defaults

**Risk:** Breaking temporal mechanics  
**Mitigation:** Comprehensive test coverage first, careful migration

**Risk:** Students confused by YAML  
**Mitigation:** Clear documentation, examples, config generator tool

#### Pedagogical Value

**Before (Hardcoded):**

- Students see 200 lines of repetitive elif statements
- "This is what NOT to do in production code"
- Adding affordance requires editing Python
- Balance changes require code review

**After (Config-Driven):**

- Students see clean generic engine
- "This is data-driven design"
- Adding affordance is editing YAML
- Non-programmers can balance game
- **Teaching moment:** Show both versions, explain trade-offs

**Assignment Idea:**

"Design a new affordance for Hamlet. It should help with a specific survival challenge. Submit your `custom_affordances.yaml` file."

Students learn game design without needing to code!

### ACTION #13: Remove Pedagogical DISABLED Code

**Status:** PLANNED  
**Priority:** MEDIUM  
**Complexity:** TRIVIAL (30 minutes)  
**Depends On:** 70% test coverage, git history preserved  
**Discovered:** November 1, 2025 during affordance effects testing

#### Current Problem

**DISABLED code kept "for teaching"** in `vectorized_env.py`:

- Lines 1019-1147: `_calculate_shaped_rewards_COMPLEX_DISABLED()` (~129 lines)
- Lines 1158-1244: `_calculate_proximity_rewards()` and related (~87 lines)
- **Total: ~216 lines of dead code**

**Why It Exists:**

- Complex reward system caused negative accumulation bug
- Proximity rewards caused reward hacking (agents standing near affordances)
- Kept "for pedagogical value" to show students what NOT to do

**Problems:**

- ✅ Code is in git history (commit hash available)
- ✅ Anecdote is documented in AGENTS.md and ROADMAP.md
- ❌ Dead code clutters the codebase
- ❌ Confuses new contributors
- ❌ Risk of accidentally re-enabling
- ❌ Not actually used for teaching (just referenced)

#### Recognition

**Git is our teaching archive**, not the live codebase:

- Students can `git show <commit>` to see old implementations
- Documentation references the commits and explains the failures
- No need to carry dead code forward
- Live code should be clean and production-ready

#### Implementation Steps

1. Find commit hashes for COMPLEX and proximity systems
2. Update documentation with commit references (AGENTS.md, ROADMAP.md)
3. Delete dead code (lines 1019-1244 in vectorized_env.py)
4. Run full test suite (should still pass - code was never called)
5. Add comment at deletion site referencing git history

#### Expected Impact on Coverage

**Current:** `vectorized_env.py`: 84% (71 missing lines)  
**After Removal:** `vectorized_env.py`: ~98% (~1 missing line)

**This single action moves vectorized_env.py from 84% → 98% coverage!**

#### Success Criteria

- [ ] Commit hashes documented in AGENTS.md and ROADMAP.md
- [ ] 216 lines deleted from vectorized_env.py
- [ ] All tests still passing
- [ ] Coverage improves dramatically
- [ ] Comment added explaining where to find old implementations

### ACTION #14: Implement Modern CI/CD Pipeline

**Status:** PLANNED  
**Priority:** HIGH  
**Complexity:** MEDIUM (3-5 days)  
**Depends On:** 70% test coverage milestone  
**Discovered:** November 1, 2025 during documentation review

#### Current Problem

**No automated quality checks in CI:**

- No linting (code style violations accumulate)
- No type checking (mypy not enforced)
- No dead code detection (216 lines of DISABLED code went unnoticed)
- No security scanning
- No automated formatting
- Manual quality control = inconsistent quality

**Current Setup:**

- Basic pytest in CI (if any)
- Manual code review catches issues late
- No pre-commit hooks
- No automatic formatting enforcement

#### Recognition

**Modern Python projects use automated quality gates:**

- Ruff: Lightning-fast linter + formatter (replaces Black, isort, flake8, pylint)
- Mypy: Type checking catches bugs before runtime
- Vulture: Dead code detection (would have flagged 216 DISABLED lines!)
- Bandit: Security vulnerability scanning
- Pre-commit: Local checks before pushing
- GitHub Actions: Automated CI on every PR

#### Proposed Solution

**Phase 1: Local Development Tools (Day 1)**

Install and configure tools:

```bash
pip install ruff mypy vulture bandit pre-commit
```

**`pyproject.toml` configuration:**

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # function calls in argument defaults
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start lenient, tighten over time
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
strict_equality = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.vulture]
min_confidence = 80
paths = ["src/townlet"]
exclude = ["tests/", "src/hamlet/"]  # Ignore legacy code

[tool.bandit]
exclude_dirs = ["tests", "src/hamlet"]
skips = ["B101"]  # Skip assert_used (normal in tests)
```

**Phase 2: Pre-commit Hooks (Day 1-2)**

**`.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--config-file=pyproject.toml]

  - repo: local
    hooks:
      - id: vulture
        name: vulture
        entry: vulture
        language: system
        types: [python]
        args: [--min-confidence=80]

      - id: pytest-coverage
        name: pytest with coverage
        entry: bash -c 'pytest tests/test_townlet/ --cov=src/townlet --cov-report=term-missing --cov-fail-under=70'
        language: system
        pass_filenames: false
        always_run: true
```

Install hooks:

```bash
pre-commit install
pre-commit run --all-files  # Initial run
```

**Phase 3: GitHub Actions CI (Day 2-3)**

**`.github/workflows/ci.yml`:**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -e ".[dev]"
          
      - name: Ruff lint
        run: ruff check src/townlet tests/test_townlet
        
      - name: Ruff format check
        run: ruff format --check src/townlet tests/test_townlet
        
      - name: Mypy type check
        run: mypy src/townlet
        
      - name: Vulture dead code check
        run: vulture src/townlet --min-confidence=80
        
      - name: Bandit security check
        run: bandit -r src/townlet -c pyproject.toml
        
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -e ".[dev]"
          
      - name: Run tests with coverage
        run: |
          pytest tests/test_townlet/             --cov=src/townlet             --cov-report=xml             --cov-report=term-missing             --cov-fail-under=70
            
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

**Phase 4: Fix Existing Violations (Day 3-5)**

Run tools and fix issues:

```bash
# 1. Auto-fix what we can
ruff check --fix src/townlet tests/test_townlet
ruff format src/townlet tests/test_townlet

# 2. Review mypy errors
mypy src/townlet > mypy_errors.txt
# Fix type hints incrementally (don't block on 100% compliance)

# 3. Review vulture findings
vulture src/townlet --min-confidence=80 > dead_code.txt
# This WILL flag the 216 lines in ACTION #13!
# Also might find other unused imports/variables

# 4. Review bandit findings
bandit -r src/townlet -c pyproject.toml > security.txt
# Fix any high-priority security issues
```

**Phase 5: Documentation & Onboarding (Day 5)**

Update development docs:

- Add "Development Setup" section to README
- Document pre-commit workflow
- Add CI badge to README
- Create CONTRIBUTING.md with quality standards

#### Benefits

**For Development:**

- ✅ Catch bugs before runtime (mypy)
- ✅ Consistent code style (ruff format)
- ✅ Find dead code automatically (vulture)
- ✅ Security vulnerability scanning (bandit)
- ✅ Pre-commit prevents bad commits
- ✅ CI prevents bad merges

**For Teaching:**

- ✅ Students learn modern Python tooling
- ✅ Professional development practices
- ✅ Automated quality feedback
- ✅ Industry-standard workflow

**For Maintenance:**

- ✅ Code quality improves over time
- ✅ Less manual review burden
- ✅ Easier to onboard contributors
- ✅ Confidence in refactoring

#### Tool Comparison

**Ruff vs Traditional:**

- Ruff: 10-100x faster than pylint/flake8
- Replaces: Black, isort, flake8, pylint, pyupgrade
- Single tool for linting + formatting
- Written in Rust, blazingly fast

**Vulture:**

- Would have detected 216 lines of DISABLED code
- Finds unused imports, variables, functions
- Configurable confidence threshold
- Minimal false positives at 80% confidence

**Mypy:**

- Static type checking without runtime overhead
- Gradual typing (can start lenient, tighten over time)
- Catches type errors, null pointer issues
- Industry standard for Python type checking

#### Implementation Plan

**Day 1: Local Tools Setup**

1. Add dependencies to pyproject.toml
2. Configure ruff, mypy, vulture, bandit
3. Run initial scan, document violations
4. Set up pre-commit hooks

**Day 2: CI Pipeline**

1. Create .github/workflows/ci.yml
2. Configure quality and test jobs
3. Add coverage upload to Codecov
4. Test on feature branch

**Day 3-4: Fix Violations**

1. Run ruff format on all files
2. Fix ruff lint errors (auto-fix most)
3. Add type hints for mypy (prioritize)
4. Review vulture findings, remove dead code
5. Fix bandit security issues

**Day 5: Documentation**

1. Update README with CI badge
2. Add Development Setup section
3. Create CONTRIBUTING.md
4. Document pre-commit workflow

#### Success Criteria

- [ ] Ruff configured and running in CI
- [ ] Mypy type checking enabled (lenient mode initially)
- [ ] Vulture dead code detection running
- [ ] Bandit security scanning enabled
- [ ] Pre-commit hooks installed and documented
- [ ] GitHub Actions CI running on all PRs
- [ ] Coverage badge added to README
- [ ] All existing code passes quality gates
- [ ] CONTRIBUTING.md created with workflow
- [ ] <10 mypy errors remaining (fix incrementally)

#### Expected Findings

**Vulture will flag:**

- 216 lines in ACTION #13 (DISABLED reward code)
- Unused imports in legacy hamlet code
- Possibly unused helper functions
- Dead variables/constants

**Ruff will find:**

- Import ordering issues (auto-fixable)
- Line length violations (auto-fixable)
- Style inconsistencies (auto-fixable)
- Possibly unused variables

**Mypy will find:**

- Missing type hints (many)
- Type mismatches in tensor operations
- Optional/None handling issues
- Return type inconsistencies

**Strategy:** Fix critical issues (security, dead code), auto-fix style, defer some type hints to gradual improvement.

#### Pedagogical Value

**Teaching Moment:**
"Professional Python development uses automated quality tools. These aren't just for 'being pedantic' - they catch real bugs:

- Mypy catches type errors before runtime
- Vulture finds 216 lines of dead code we forgot about
- Bandit finds security vulnerabilities
- Ruff enforces consistency across contributors

This is how real teams ship quality software at scale."

#### Notes

**Why High Priority:**

- Technical debt accumulates without quality gates
- 216 lines of dead code went unnoticed (vulture would have caught it)
- Type safety prevents entire classes of bugs
- Industry-standard tooling prepares students for real work
- Small investment (3-5 days) pays dividends forever

**Why After 70% Coverage:**

- Need test safety net before fixing violations
- Coverage requirement in CI needs to be met first
- Some tools (vulture) work better with high test coverage

**Gradual Type Checking:**
Start with `disallow_untyped_defs = false`, then tighten:

1. Phase 1: Check typed functions only
2. Phase 2: Require types for new code
3. Phase 3: Add types to existing code incrementally
4. Phase 4: Enable strict mode

Don't block on 100% type coverage - improve gradually!

---

### ACTION #15: Unified Training + Inference Server

**Status:** PLANNED  
**Priority:** MEDIUM  
**Complexity:** MEDIUM-HIGH (1-2 weeks)  
**Depends On:** 70% coverage, stable training loop, ACTION #14 (CI/CD)  
**Discovered:** November 1, 2025 during demo infrastructure review

#### Current Problem

**Training, inference, and frontend are THREE separate processes:**

**Current Architecture:**

```bash
# Terminal 1: Training
python -m townlet.demo.runner configs/townlet.yaml demo.db checkpoints 10000

# Terminal 2: Inference server  
python -m townlet.demo.live_inference checkpoints 8766 0.2 10000 configs/townlet.yaml

# Terminal 3: Frontend web server
cd frontend && npm run dev
```

**Three processes to manage!**

- `runner.py`: Training process (saves checkpoints every 100 episodes)
- `live_inference.py`: WebSocket server (watches for new checkpoints, loads periodically)
- `npm run dev`: Vue.js frontend (serves HTML/JS/CSS on port 5173)

**Problems:**

- ❌ **THREE separate commands to start demo** (fiddly, error-prone)
- ❌ Inference sees stale model (100 episodes behind)
- ❌ Extra process management complexity
- ❌ Double GPU memory usage (if both use GPU)
- ❌ Checkpoint I/O adds latency
- ❌ Race conditions (inference loading while checkpoint being written)
- ❌ Can't see "hot off the press" behavior during training
- ❌ Frontend must be started separately (easy to forget)
- ❌ Port conflicts if something already on 5173 or 8766

#### Recognition

**The goal: `python run_demo.py` and you're done!**

No juggling three terminals. No "did I remember to start the frontend?" confusion.

**Modern RL systems integrate training, evaluation, and visualization:**

- Ray RLlib: Built-in rollout workers + TensorBoard integration
- SB3: VecEnv callbacks + evaluation during training
- MLflow: Unified experiment tracking with web UI
- **Single command starts everything**

**Two technical approaches:**

1. **Checkpoint-based** (safer): Serve from last completed checkpoint
2. **Live model** (faster): Serve directly from training model (read-only)

#### Proposed Solution

**Single Command: `python run_demo.py`**

```bash
# That's it! Everything starts automatically:
python run_demo.py --config configs/townlet.yaml --episodes 10000

# Opens browser to http://localhost:8766 automatically
# Training + inference + frontend all running
# Hit Ctrl+C to stop everything cleanly
```

**Unified Server Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│                    run_demo.py                               │
│              Unified Training + Inference + Web UI           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  Training Thread │  │ Inference Thread │  │ Frontend  │ │
│  │                  │  │                  │  │  Server   │ │
│  │  1. Step env     │◄─┤  1. Copy model   │  │  (Static) │ │
│  │  2. Update model │  │  2. Run episode  │  │           │ │
│  │  3. Checkpoint   │  │  3. Broadcast    │◄─┤  Port     │ │
│  │  4. Repeat       │  │  4. Wait/Repeat  │  │  8766     │ │
│  └──────────────────┘  └──────────────────┘  └───────────┘ │
│         │                      │                     │       │
│         └──────────┬───────────┴─────────────────────┘       │
│                    ▼                                         │
│           ┌─────────────────┐                                │
│           │  Shared Q-Net   │                                │
│           │  (with RWLock)  │                                │
│           └─────────────────┘                                │
│                                                              │
│  WebSocket (ws://localhost:8766/ws) ◄──► Browser            │
└──────────────────────────────────────────────────────────────┘
```

**Key Components:**

- **TrainingLoop**: Pure training logic (no I/O dependencies)
- **InferenceEngine**: Runs episodes with model snapshot
- **CheckpointManager**: Coordinates model synchronization
- **FrontendServer**: Serves static Vue.js build (embedded in Python)
- **UnifiedServer**: Orchestrates all three threads + auto-opens browser

#### Benefits

**Performance:**

- ✅ Single process (no IPC overhead)
- ✅ Shared model (no double memory)
- ✅ Optional: Serve from live model (no checkpoint I/O lag)
- ✅ Faster inference updates (every 10 episodes vs 100)

**Simplicity:**

- ✅ One command to start training + demo
- ✅ No process coordination needed
- ✅ Simpler deployment (one service)
- ✅ Unified configuration

**Features:**

- ✅ Training status API (pause/resume/metrics)
- ✅ See "hot off the press" agent behavior
- ✅ Live performance monitoring
- ✅ Easier debugging (single process)

#### Implementation Plan

**Week 1: Core Refactoring**

**Day 1-2: Extract TrainingLoop**

- Separate training logic from runner.py
- Create TrainingLoop class (pure RL logic)
- Test: Training still works identically

**Day 3-4: Extract InferenceEngine**

- Extract inference logic from live_inference.py
- Create InferenceEngine class (model snapshot + episode runner)
- Test: Inference episodes run correctly

**Day 4-5: Model Synchronization**

- Implement CheckpointManager with threading primitives
- Add model snapshot mechanism (deep copy)
- Test: Thread-safe model access

**Week 2: Integration**

**Day 1-2: Unified Server + Frontend Integration**

- Create UnifiedServer combining training + inference + frontend
- Integrate FastAPI WebSocket server (port 8766)
- **Build Vue.js frontend to static bundle** (`npm run build`)
- **Serve static frontend from Python** (FastAPI StaticFiles)
- Add training control API (pause/resume/status)
- **Auto-open browser** on startup (Python `webbrowser` module)
- Test: All three components run concurrently

**Frontend Integration Options:**

1. **Static build (RECOMMENDED)**: Build frontend once, embed in Python package
   - `frontend/dist/` → Python package resources
   - FastAPI serves static files from `/` route
   - No Node.js runtime needed for deployment!
2. **Dev mode**: Optionally proxy to `npm run dev` for development
   - Use `--dev` flag to run frontend hot-reload server
   - Production mode uses static build

**Day 3-4: Testing & Polish**

- Thread safety tests
- Performance benchmarks
- Integration tests (full training runs)
- Stress tests (multiple WebSocket clients)
- Test static frontend serving
- Test auto-browser-open functionality

**Day 5: Documentation + Packaging**

- Update README with new unified command
- Document training control API
- Add architecture diagram
- Migration guide from old two-process system
- **Create `run_demo.py` entry point**
- Update package build to include frontend static files

#### Success Criteria

- [ ] **Single command starts everything: `python run_demo.py`**
- [ ] **Browser auto-opens to <http://localhost:8766>**
- [ ] Frontend loads correctly (static files served from Python)
- [ ] Training throughput unchanged (episodes/hour)
- [ ] Inference sees model updates every 10-100 episodes (configurable)
- [ ] WebSocket clients receive step-by-step updates
- [ ] Training control API works (pause/resume/status)
- [ ] Thread-safe (no deadlocks, no race conditions)
- [ ] Memory usage acceptable (1.5x single training, not 2x)
- [ ] All tests passing (unit + integration)
- [ ] Clean shutdown (Ctrl+C stops all threads)
- [ ] **No "forgot to start frontend" errors**
- [ ] **No "forgot to start inference server" errors**
- [ ] Documentation updated with new architecture

#### Recommended Strategy

**Start with Checkpoint-Based (SAFER):**

- Training and inference fully isolated after checkpoint
- Easier to debug (clear separation)
- Proven pattern (most RL frameworks use this)
- Lower risk of training instability

**Later (if needed), migrate to Live Model:**

- Only if checkpoint I/O becomes bottleneck
- Requires careful locking (read during training updates)
- Risk: Inference could see partially updated model
- Benefit: Faster updates, no disk I/O

#### Pedagogical Value

**Teaching Moment:**
"Real-world RL systems need to train and evaluate simultaneously. This teaches:

- **Concurrent programming**: Threading, locks, async I/O
- **System architecture**: Separating concerns, shared state management
- **Performance optimization**: Minimizing synchronization overhead
- **Production patterns**: Single service vs microservices

Industry frameworks (Ray RLlib, SB3) solve this problem. Now you understand how!"

#### Notes

**Why Medium Priority:**

- Current three-process system works (not broken)
- Significant engineering effort (1-2 weeks)
- But: **MUCH better UX** - no fiddly multi-terminal juggling!

**Why After 70% Coverage:**

- Need tests before refactoring demo infrastructure
- Demo code currently untested (0% coverage)
- Want stable training loop before adding concurrency

**Why After ACTION #14 (CI/CD):**

- Thread safety bugs are subtle
- Need automated testing for concurrency issues
- CI/CD catches race conditions and deadlocks

**Frontend Build Strategy:**

- One-time setup: Build Vue.js to static files
- Python package includes `frontend/dist/` in resources
- FastAPI serves static files (no Node.js runtime needed!)
- Development mode: Optionally run `npm run dev` with `--dev` flag
- Production: Just Python, no npm/node required

**GPU Consideration:**

- Training uses GPU for forward + backward pass
- Inference only needs CPU (no gradients)
- Strategy: Keep inference model on CPU, training on GPU
- Saves GPU memory for larger batch sizes!

**User Experience Win:**

```bash
# Before (THREE commands, THREE terminals):
Terminal 1: python -m townlet.demo.runner ...
Terminal 2: python -m townlet.demo.live_inference ...  
Terminal 3: cd frontend && npm run dev

# After (ONE command, ONE terminal):
python run_demo.py --config configs/townlet.yaml --episodes 10000

# Browser auto-opens, everything just works!
```

---

---

---

## 🚨 DO NOT START Until

1. ✅ Test coverage reaches 70% on modules being refactored
2. ✅ All existing tests passing
3. ✅ Performance baseline established
4. ✅ Multi-Day Demo (Phase 3.5) validates current system

**Reason:** Refactoring without tests = flying blind. We need the safety net first.

---

**End of Document** - Updated as refactoring progresses

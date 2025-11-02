# Hamlet Architecture Documentation

**Last Updated:** November 1, 2025  
**Purpose:** Comprehensive architecture documentation to support refactoring  
**Test Coverage:** 64% (982/1525 statements, 241 tests passing)  
**Status:** Pre-refactoring baseline (target: 70% before major changes)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Major Systems](#major-systems)
3. [Data Flow](#data-flow)
4. [Key Interfaces](#key-interfaces)
5. [Dependencies](#dependencies)
6. [System Boundaries](#system-boundaries)

---

## System Overview

Hamlet is a **pedagogical Deep RL environment** implementing a survival simulation where agents learn to manage 8 interconnected meters through interactions with 15 affordances in a grid world.

### Core Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING ORCHESTRATION                        │
│                   (population/vectorized.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Environment  │  │  Curriculum  │  │ Exploration  │         │
│  │   System     │  │   System     │  │   System     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         │                  │                  │                  │
│         v                  v                  v                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           Neural Network (Q-Network)                 │       │
│  │  - SimpleQNetwork (full observability)               │       │
│  │  - RecurrentSpatialQNetwork (POMDP + LSTM)          │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           Supporting Infrastructure                  │       │
│  │  - ReplayBuffer (experience storage)                 │       │
│  │  - State DTOs (BatchedAgentState, CurriculumDecision)│       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Philosophy

- **Vectorized-first:** All operations on `[num_agents, ...]` tensors for GPU efficiency
- **Separation of concerns:** Environment, curriculum, exploration as pluggable components
- **Hot path optimization:** Step logic must be GPU-native (no Python loops)
- **Cold path flexibility:** Episode-level logic can use CPU validation
- **Pedagogical focus:** Interesting failures are teaching moments

---

## Major Systems

### 1. Environment System

**Location:** `src/townlet/environment/`  
**Core File:** `vectorized_env.py` (1247 lines)  
**Coverage:** 82%  
**Complexity:** 🔴 EXTREME (Monolithic - needs refactoring)  
**Purpose:** Simulates grid world with meter dynamics and affordance interactions

---

#### 1.1 System Overview

The Environment System is the **most complex component** in Hamlet, responsible for:

1. **Grid World Management** - Agent positions, affordance locations (15 affordances on 8×8 grid)
2. **Meter Dynamics** - 8-meter survival simulation with coupled differential equations
3. **Observation Construction** - Full observability (Level 1) vs Partial observability (Level 2 POMDP)
4. **Action Execution** - Movement (4 directions) + INTERACT action
5. **Action Masking** - Prevent invalid actions (boundaries, affordability, operating hours)
6. **Temporal Mechanics** - 24-hour cycle, multi-tick interactions (Level 2.5)
7. **Reward Calculation** - Milestone survival bonuses (sparse rewards)
8. **Terminal Conditions** - Death detection (energy OR health <= 0)

**Critical Issue:** This file does TOO MUCH. Refactoring priority is HIGH.

---

#### 1.2 Core Class: `VectorizedHamletEnv`

**Inheritance:** None (standalone class)  
**Design Pattern:** Monolithic (all logic in one class)

##### 1.2.1 Constructor (`__init__`, Lines 20-115)

```python
def __init__(
    num_agents: int,           # Parallel environments
    grid_size: int = 8,        # 8×8 grid
    device: torch.device,      # CPU or CUDA
    partial_observability: bool = False,  # Level 2 POMDP
    vision_range: int = 2,     # 5×5 window when partial
    enable_temporal_mechanics: bool = False  # Level 2.5
)
```

**State Initialization:**

- **Affordances Dictionary** (Lines 48-72): 15 affordances with fixed positions
  - Basic survival: Bed, LuxuryBed, Shower, HomeMeal, FastFood
  - Income sources: Job, Labor
  - Fitness/Social: Gym, Bar, Park
  - Mood restoration: Recreation, Therapist
  - Health restoration: Doctor, Hospital

- **Observation Dimension Calculation** (Lines 77-94):
  - Full observability: `grid_size² + 8 meters + (15 affordances + 1)`
  - Partial observability: `25 (5×5 window) + 2 (position) + 8 meters + 16 affordances`
  - Temporal mechanics: `+2` (time_of_day + interaction_progress)

- **State Tensors** (Lines 100-115):
  - `positions`: `[num_agents, 2]` - (x, y) coordinates
  - `meters`: `[num_agents, 8]` - Normalized [0, 1]
  - `dones`: `[num_agents]` - Boolean episode termination
  - `step_counts`: `[num_agents]` - Survival time tracking
  - `time_of_day`: `int` [0-23] - Global clock (temporal mechanics)
  - `interaction_progress`: `[num_agents]` - Ticks completed (temporal mechanics)
  - `last_interaction_affordance`: `List[Optional[str]]` - Track continuity
  - `last_interaction_position`: `[num_agents, 2]` - Detect movement

---

##### 1.2.2 Episode Management

**`reset()` (Lines 117-143):**

- Randomizes agent starting positions
- Initializes all meters to 100% (1.0) except money (starts at 50%)
- Resets temporal state (time=0, interaction_progress=0)
- Returns initial observations

**Key Detail:** Money normalization is `$0-$100 → [0, 1]`, meaning money cannot go negative (no debt).

---

##### 1.2.3 Observation Construction Subsystem (Lines 145-280)

**Three Observation Modes:**

**`_get_observations()` (Lines 145-154):**

- Router method: dispatches to full or partial based on config

**`_get_full_observations()` (Lines 187-231):**

- **Level 1: Full Observability**
- Grid encoding: One-hot position `[num_agents, grid_size²]`
- Current affordance encoding: `[num_agents, 16]` (15 types + "none")
- Meters: `[num_agents, 8]`
- Temporal features (if enabled): `time_of_day` [0, 1], `interaction_progress` [0, 1]
- **Total dims:** 64 (grid) + 8 (meters) + 16 (affordance) + 2 (temporal) = **90 dims** (8×8 grid)

**`_get_partial_observations()` (Lines 233-280):**

- **Level 2: Partial Observability (POMDP)**
- Local window: 5×5 centered on agent (vision_range=2)
- Affordances visible in window encoded as binary (1=affordance, 0=empty/OOB)
- Normalized position: `[x, y] / (grid_size - 1)` → [0, 1]
- Current affordance encoding: `[num_agents, 16]`
- Meters: `[num_agents, 8]`
- **Total dims:** 25 (window) + 2 (position) + 8 (meters) + 16 (affordance) = **51 dims**

**`_get_current_affordance_encoding()` (Lines 156-185):**

- One-hot encoding of affordance under agent's feet
- 16 dims: 15 affordance types + 1 "none" (not on any affordance)
- Used by both full and partial observation modes

**Design Issue:** Observation construction is tightly coupled to environment state. Should be extracted to `ObservationBuilder` class (ACTION #4).

---

##### 1.2.4 Action Masking Subsystem (Lines 282-365)

**`get_action_masks()` (Lines 282-365):**

**Purpose:** Prevent invalid actions to save exploration budget and speed learning.

**Three Masking Rules:**

1. **Boundary Masking** (Lines 292-308):

   ```python
   at_top = positions[:, 1] == 0  # Can't go UP
   at_bottom = positions[:, 1] == grid_size - 1  # Can't go DOWN
   at_left = positions[:, 0] == 0  # Can't go LEFT
   at_right = positions[:, 0] == grid_size - 1  # Can't go RIGHT
   ```

2. **Operating Hours Masking** (Lines 310-335):
   - Temporal mechanics only: Check `is_affordance_open(time_of_day, operating_hours)`
   - Example: Job closed outside 8am-6pm, Bar only open 6pm-4am

3. **Affordability Masking** (Lines 337-361):
   - Check `meters[:, 3] >= cost` (money meter)
   - Temporal mechanics: Use `cost_per_tick` from config
   - Legacy: Use `affordance_costs` dictionary (single-shot cost)

**Return:** `[num_agents, 5]` bool tensor (True=valid, False=invalid)

**Design Note:** INTERACT action is ONLY masked when physically impossible:

- Agent must be on an affordance tile
- Affordance must be open (temporal mechanics)

Affordability is enforced inside the interaction handler; if the agent is broke, the step is spent with only passive decay.

**WAIT Action:** WAIT (action index 5) is always available. Environment initialization enforces `wait_energy_cost < move_energy_cost` so WAIT remains a low-cost recovery move (only passive decay applies).

---

##### 1.2.5 Main Step Function (Lines 367-412)

**`step(actions)` (Lines 367-412):**

**Execution Order (Critical for correctness):**

1. **Execute Actions** → `_execute_actions(actions)` - Movement + INTERACT
2. **Deplete Meters** → `_deplete_meters()` - Base passive decay
3. **Cascading Effects** (3-stage differential equations):
   - `_apply_secondary_to_primary_effects()` - Satiation/Fitness/Mood → Health/Energy
   - `_apply_tertiary_to_secondary_effects()` - Hygiene/Social → Satiation/Fitness/Mood
   - `_apply_tertiary_to_primary_effects()` - Hygiene/Social → Health/Energy (weak)
4. **Check Terminal Conditions** → `_check_dones()` - Death detection
5. **Calculate Rewards** → `_calculate_shaped_rewards()` - Milestone bonuses
6. **Increment Counters** - step_counts, time_of_day
7. **Construct Observations** → `_get_observations()`

**Return:**

- `observations`: `[num_agents, obs_dim]`
- `rewards`: `[num_agents]`
- `dones`: `[num_agents]` bool
- `info`: dict with `step_counts`, `positions`, `successful_interactions`

---

##### 1.2.6 Action Execution Subsystem (Lines 414-545)

**`_execute_actions(actions)` (Lines 414-482):**

**Movement Handling:**

- Actions 0-3: UP, DOWN, LEFT, RIGHT (deltas: `[0,-1], [0,+1], [-1,0], [+1,0]`)
- Action 4: INTERACT (no movement)
- Clamps positions to grid boundaries (redundant with action masking)
- **Movement Costs** (Lines 458-474): Energy -0.5%, Hygiene -0.3%, Satiation -0.4%

**Interaction Tracking (Temporal Mechanics):**

- Detects if agent moved away from affordance → resets `interaction_progress` to 0
- Delegates INTERACT actions to `_handle_interactions()`

**Return:** `dict` mapping agent indices to affordance names (successful interactions)

---

**`_handle_interactions(interact_mask)` (Lines 484-545):**

**Multi-Tick Interaction Logic (Level 2.5 - Temporal Mechanics):**

1. **Affordance Detection** (Lines 502-507):
   - Check `distance == 0` (agent on affordance)
   - Check affordability: `meters[:, 3] >= cost_per_tick`

2. **Progress Tracking** (Lines 517-529):
   - **Continuing same affordance at same position:** `interaction_progress += 1`
   - **New affordance or moved:** Reset `interaction_progress = 1`

3. **Benefit Application** (Lines 531-537):
   - **Per-tick benefits** (75% of total, distributed): Apply from `config['benefits']['linear']`
   - **Completion bonus** (25% of total): Apply when `ticks_done == required_ticks`
   - Example: Bed requires 5 ticks for full 50% energy restoration
     - Per tick: +7.5% energy (5 × 7.5% = 37.5%)
     - Completion: +12.5% energy + 2% health (total: 50% + 2%)

4. **Cost Charging:** Subtract `cost_per_tick` from money meter per tick

5. **Progress Reset:** When job completes, reset `interaction_progress = 0`

**Design Pattern:** Accumulative rewards encourage commitment (don't abandon halfway).

---

**`_handle_interactions_legacy(interact_mask)` (Lines 547-780):**

**Single-Shot Interaction Logic (Level 1 - Legacy):**

**Purpose:** Backward compatibility when `enable_temporal_mechanics=False`

**Hardcoded Affordance Effects (Lines 621-774):**

**⚠️ CRITICAL TECHNICAL DEBT:** 200+ lines of if-elif blocks for 15 affordances.

**Example Affordance Effects:**

- **Bed** (Lines 621-629): Energy +50%, Health +2%, Cost -$5
- **LuxuryBed** (Lines 630-638): Energy +75%, Health +5%, Cost -$11 (tier 2)
- **Job** (Lines 658-667): Money +$22.50, Energy -15%, Social +2%, Health -3%
- **Labor** (Lines 668-680): Money +$30, Energy -20%, Fitness -5%, Health -5%
- **Bar** (Lines 690-704): Social +50% (BEST), Mood +25%, Health -5%, Cost -$15
- **Park** (Lines 732-745): Fitness +20%, Social +15%, Mood +15%, Energy -15% (FREE!)
- **Doctor** (Lines 746-750): Health +25%, Cost -$8 (tier 1)
- **Hospital** (Lines 751-756): Health +40%, Cost -$15 (tier 2)
- **Therapist** (Lines 757-762): Mood +40%, Cost -$15 (tier 2)

**Tier Structure:**

- **Tier 1:** Affordable (Bed, Doctor) - Basic needs
- **Tier 2:** Premium (LuxuryBed, Hospital, Therapist) - Higher effectiveness, higher cost

**ACTION #12:** Move this to YAML config for moddability.

---

##### 1.2.7 Meter Dynamics Subsystem (Lines 782-950)

**Meter Hierarchy (Coupled Differential Equations):**

```
PRIMARY (Death Conditions):
  - Health [6]: Are you alive?
  - Energy [0]: Can you move?
      ↑ strong        ↑ strong
  SECONDARY (Aggressive → Primary):
  - Satiation [2] → Health AND Energy (FUNDAMENTAL!)
  - Fitness [7] → Health (unfit → sick)
  - Mood [4] → Energy (depressed → exhausted)
      ↑ strong        ↑ strong
  TERTIARY (Quality of Life):
  - Hygiene [1] → Secondary (strong) + Primary (weak)
  - Social [5] → Secondary (strong) + Primary (weak)
      
RESOURCE:
  - Money [3]: Enables affordances
```

---

**`_deplete_meters()` (Lines 782-826):**

**Base Passive Decay (Per Step):**

```python
energy: -0.5%    # Primary
hygiene: -0.3%   # Tertiary
satiation: -0.4% # Secondary (FUNDAMENTAL)
money: 0%        # No passive depletion
mood: -0.1%      # Secondary
social: -0.6%    # Tertiary (fastest depletion!)
health: 0%       # Modulated by fitness (see below)
fitness: -0.2%   # Secondary
```

**Fitness-Modulated Health Depletion (Lines 813-826):**

- **Baseline:** 0.1% per step
- **Multiplier:** `0.5 + (2.5 × (1 - fitness))` (linear gradient)
- **Examples:**
  - 100% fitness: 0.5× multiplier = 0.05% health depletion (very healthy)
  - 50% fitness: 1.75× multiplier = 0.175% health depletion (moderate)
  - 0% fitness: 3.0× multiplier = 0.3% health depletion (get sick easily)

**Design Rationale:** Fitness is a PROTECTIVE meter - high fitness prevents health decline.

---

**`_apply_secondary_to_primary_effects()` (Lines 828-873):**

**SECONDARY → PRIMARY (Aggressive Cascades):**

**Satiation → Health AND Energy (Lines 838-854):**

- **Threshold:** 30% (below this, penalties apply)
- **Deficit Formula:** `(threshold - satiation) / threshold` → [0, 1]
- **Health Penalty:** `0.4% × deficit` (starving → sick)
- **Energy Penalty:** `0.5% × deficit` (hungry → exhausted)
- **Key Insight:** Satiation is FUNDAMENTAL - affects BOTH death conditions!

**Fitness → Health (Lines 856-858):**

- Already handled in `_deplete_meters()` via fitness-modulated multiplier
- Low fitness = 3× health depletion

**Mood → Energy (Lines 860-873):**

- **Threshold:** 30%
- **Energy Penalty:** `0.5% × deficit` (depressed → exhausted)

**Design Rationale:** FOOD FIRST, then everything else. Hunger kills fastest.

---

**`_apply_tertiary_to_secondary_effects()` (Lines 875-908):**

**TERTIARY → SECONDARY (Aggressive Cascades):**

**Hygiene → Satiation/Fitness/Mood (Lines 883-899):**

- **Threshold:** 30%
- **Satiation Penalty:** `0.2% × deficit` (dirty → loss of appetite)
- **Fitness Penalty:** `0.2% × deficit` (dirty → harder to exercise)
- **Mood Penalty:** `0.3% × deficit` (dirty → feel bad)

**Social → Mood (Lines 901-908):**

- **Threshold:** 30%
- **Mood Penalty:** `0.4% × deficit` (lonely → depressed) - Stronger than hygiene!

**Design Rationale:** Tertiary meters are quality-of-life - neglect them and secondary meters suffer.

---

**`_apply_tertiary_to_primary_effects()` (Lines 910-950):**

**TERTIARY → PRIMARY (Weak Direct Effects):**

**Hygiene → Health/Energy (Lines 918-935):**

- **Threshold:** 30%
- **Health Penalty:** `0.05% × deficit` (weak - mainly works through secondary)
- **Energy Penalty:** `0.05% × deficit` (weak)

**Social → Energy (Lines 937-950):**

- **Threshold:** 30%
- **Energy Penalty:** `0.08% × deficit` (weak)

**Design Rationale:** Tertiary meters have STRONG effects on secondary, WEAK effects on primary.

**ACTION #1:** This entire cascade system should be configuration-driven, not hardcoded.

---

##### 1.2.8 Terminal Conditions (Lines 952-978)

**`_check_dones()` (Lines 952-978):**

**Death Conditions:**

```python
health <= 0.0 OR energy <= 0.0
```

**Two Ways to Die:**

1. **Health Death:** Unfit → low health → death (slow burn)
2. **Energy Death:** Hungry/depressed → exhausted → death (faster)

**Cascade Architecture Documented in Docstring (Lines 954-976):**

- Complete description of meter hierarchy
- Key insight: "Satiation is THE foundational need"

---

##### 1.2.9 Reward Calculation Subsystem (Lines 980-1125)

**Three Reward Systems (2 disabled, 1 active):**

---

**`_calculate_shaped_rewards()` (Lines 980-1012) - ✅ ACTIVE:**

**MILESTONE SURVIVAL REWARDS (Sparse):**

**Problem Solved:** Constant per-step rewards accumulate negatively with longer survival.

**Solution:**

- **Every 10 steps:** +0.5 ("you're making progress!")
- **Every 100 steps:** +5.0 ("happy birthday!" 🎂)
- **Death:** -100.0

**Rationale:** Rewards longevity without constant accumulation. Prevents oscillation from being rewarded.

---

**`_calculate_shaped_rewards_COMPLEX_DISABLED()` (Lines 1014-1100) - ❌ DISABLED:**

**COMPLEX METER-BASED REWARDS (Dense):**

**Why Disabled (Lines 1016-1020):**

- **Problem:** Longer survival → more per-step penalties → negative total rewards
- **Example:** 200 steps with low meters = -2000 reward (backwards!)
- **Kept for reference** - demonstrates "interesting failure" for teaching

**Structure (Lines 1022-1098):**

- Tier 1: Essential PRIMARY meters (energy, hygiene) - Healthy (+0.4), Critical (-2.5)
- Money gradient: Comfortable buffer (+0.5), Critical (-2.0)
- Mood/Social: Support meters - High (+0.2/+0.15), Critical (-1.0/-1.2)
- Health: Slow burn - Healthy (+0.3), Critical (-1.0)
- Fitness: Preventive - High (+0.15), Critical (-0.8)

**Total Per Step:** Can reach -8 to +3 per step → accumulates badly over 200+ steps!

---

**`_calculate_proximity_rewards()` (Lines 1102-1125) - ❌ DISABLED:**

**PROXIMITY SHAPING REWARDS:**

**Why Disabled:**

- **Problem:** Agents learned to stand near affordances without interacting ("reward hacking")
- **Example:** Agent hovers near Bed, collects proximity rewards, never sleeps

**Logic (Lines 1115-1125):**

- Find most critical meter (lowest value below threshold)
- Calculate distance to target affordance
- Reward: `urgency × proximity × 0.5`

**Pedagogical Value:** Classic example of reward misspecification.

**ACTION #13:** Remove 216 lines of disabled code (COMPLEX + Proximity) → +10-12% coverage boost!

---

##### 1.2.10 Utility Methods (Lines 1127-1160)

**`get_affordance_positions()` (Lines 1127-1140):**

- Returns dict of affordance names → (x, y) tuples
- Used by frontend for visualization

**`randomize_affordance_positions()` (Lines 1142-1160):**

- Shuffles all affordance positions
- For generalization testing (prevent position memorization)
- Ensures no two affordances occupy same position

---

#### 1.3 Supporting File: `affordance_config.py`

**Location:** `src/townlet/environment/affordance_config.py`  
**Lines:** 303  
**Coverage:** 100% ✅  
**Purpose:** Multi-tick interaction configuration for temporal mechanics (Level 2.5)

---

**Data Structure: `AFFORDANCE_CONFIGS` (Lines 18-272):**

**Format:**

```python
{
    'AffordanceName': {
        'required_ticks': int,           # Ticks for full benefit
        'cost_per_tick': float,          # Money per tick [0, 1]
        'operating_hours': (int, int),   # (open, close) [0-23]
        'benefits': {
            'linear': {                  # 75% distributed per tick
                'meter_name': delta,     # Per-tick benefit
            },
            'completion': {              # 25% bonus on completion
                'meter_name': delta,     # Completion bonus
            }
        }
    }
}
```

**Example - Bed (Lines 20-38):**

```python
'Bed': {
    'required_ticks': 5,           # 5 INTERACTs needed
    'cost_per_tick': 0.01,         # $1 per tick ($5 total)
    'operating_hours': (0, 24),    # 24/7
    'benefits': {
        'linear': {
            'energy': +0.075,      # Per tick: (50% × 0.75) / 5 = 7.5%
        },
        'completion': {
            'energy': +0.125,      # Bonus: 50% × 0.25 = 12.5%
            'health': +0.02,       # Additional bonus
        }
    }
}
```

**Total Benefit:** 5 ticks × 7.5% + 12.5% + 2% health = 50% energy + 2% health

---

**Affordance Categories:**

**24/7 Affordances (Lines 20-125):**

- Bed, LuxuryBed (Energy tier 1 & 2)
- Shower (Hygiene)
- HomeMeal (Satiation + Health)
- Hospital (Health tier 2 - expensive)
- Gym (Fitness)
- FastFood (Quick satiation, health penalty)

**Business Hours 8am-6pm (Lines 127-209):**

- Job (Income: $22.50, Energy -15%)
- Labor (Income: $30, Energy -20%, Fitness/Health -5%)
- Doctor (Health tier 1 - affordable)
- Therapist (Mood tier 2)
- Recreation (Mood + Social, open 8am-10pm)

**Dynamic Hours (Lines 211-272):**

- CoffeeShop (Energy + Mood + Social, 8am-6pm)
- Bar (Social +50% BEST, Mood +25%, 6pm-4am - wraps midnight!)
- Park (FREE! Fitness +20%, Social +15%, Mood +15%, 6am-10pm)

---

**Helper Structures:**

**`METER_NAME_TO_IDX` (Lines 276-285):**

- Maps meter names → tensor indices
- Used by interaction handler to apply benefits

**`is_affordance_open(time_of_day, operating_hours)` (Lines 288-303):**

- Checks if affordance is open at given time
- **Handles midnight wraparound:** Bar (18, 4) means 6pm to 4am
- Logic: `time >= open OR time < close` when open > close

---

#### 1.4 Refactoring Opportunities

**Current State:** Monolithic 1247-line file doing 8 major responsibilities.

**Proposed Extractions:**

1. **ACTION #2: RewardStrategy** (3-5 days)
   - Extract reward calculation to pluggable strategy
   - Keep milestone (active), move COMPLEX+Proximity to git history
   - Enable curriculum to vary reward modes

2. **ACTION #3: MeterDynamics** (1-2 weeks)
   - Extract cascade logic to separate class
   - Make configurable (ACTION #1 prerequisite)
   - Enable students to experiment with cascade strengths

3. **ACTION #4: ObservationBuilder** (2-3 days)
   - Extract observation construction
   - Support multiple observation modes cleanly
   - Reduce coupling to environment state

4. **ACTION #12: YAML Affordances** (1-2 weeks)
   - Move legacy `_handle_interactions_legacy()` effects to config
   - Use `affordance_config.py` as template
   - Delete 200+ lines of if-elif blocks

5. **ACTION #13: Remove Dead Code** (30 minutes) 🎯 **QUICK WIN!**
   - Delete COMPLEX_DISABLED (86 lines)
   - Delete proximity_rewards (23 lines)
   - **Impact:** 82% → ~95% coverage instantly!

**Target Architecture:**

```
environment/
├── core.py              # Grid, positions (200 lines)
├── meters.py            # MeterDynamics (150 lines)
├── observations.py      # ObservationBuilder (100 lines)
├── actions.py           # Movement, masking (150 lines)
├── rewards.py           # RewardStrategy (50 lines)
├── interactions.py      # Affordance logic (100 lines)
├── affordance_config.py # Data (303 lines)
└── __init__.py          # Public API
```

**Total:** 1053 lines (vs 1247) - 15% reduction + better separation of concerns.

---

### 2. Curriculum System

**Location:** `src/townlet/curriculum/`  
**Core File:** `adversarial.py` (361 lines)  
**Coverage:** 86%  
**Complexity:** 🟡 MODERATE (State machine with performance tracking)  
**Purpose:** Auto-tuning difficulty progression based on agent performance

---

#### 2.1 System Overview

The Curriculum System implements **adversarial curriculum learning** - automatically adjusting environment difficulty based on agent performance metrics. It prevents agents from being overwhelmed by complexity early while ensuring they graduate to sparse rewards.

**Core Responsibilities:**

1. **Performance Tracking** - Survival rate, learning progress (reward improvement), policy convergence (entropy)
2. **Stage Decisions** - Advance/retreat/stay based on multi-signal analysis
3. **Per-Agent State** - Each agent has independent stage (enables mixed-stage training)
4. **Configuration Management** - 5-stage progression from shaped to sparse rewards
5. **State Persistence** - Checkpoint/restore for long training runs

**Design Pattern:** State machine with performance-gated transitions.

---

#### 2.2 Abstract Base: `CurriculumManager` (base.py, 74 lines)

**Coverage:** 77%  
**Purpose:** Define interface for pluggable curriculum strategies

##### 2.2.1 Core Interface (Lines 11-74)

```python
class CurriculumManager(ABC):
    @abstractmethod
    def get_batch_decisions(
        agent_states: BatchedAgentState,
        agent_ids: List[str]
    ) -> List[CurriculumDecision]

    @abstractmethod
    def checkpoint_state() -> Dict[str, Any]

    @abstractmethod
    def load_state(state: Dict[str, Any]) -> None
```

**Design Notes:**

- Called **once per episode** (not per step) - cold path acceptable
- Input: GPU tensors (`BatchedAgentState`)
- Output: CPU DTOs (`List[CurriculumDecision]`)
- Overhead acceptable since frequency is low

**Gap in Coverage (23%):** Abstract methods + docstrings (not executed in tests).

---

#### 2.3 Baseline Implementation: `StaticCurriculum` (static.py, 100 lines)

**Coverage:** 100% ✅  
**Purpose:** Fixed difficulty for baseline experiments and interface validation

##### 2.3.1 Constructor (Lines 14-47)

```python
def __init__(
    difficulty_level: float = 0.5,      # Fixed [0.0-1.0]
    reward_mode: str = 'shaped',        # 'shaped' or 'sparse'
    active_meters: List[str] = None,    # Default: all 6
    depletion_multiplier: float = 1.0   # Depletion rate multiplier
)
```

**Default Active Meters:** `['energy', 'hygiene', 'satiation', 'money', 'mood', 'social']`

**Note:** Health and fitness are NOT in active_meters list (they're always active, affected by cascades).

##### 2.3.2 Decision Generation (Lines 49-69)

```python
def get_batch_decisions(
    agent_states: BatchedAgentState,  # Ignored
    agent_ids: List[str]
) -> List[CurriculumDecision]
```

**Behavior:** Returns **identical decision** for all agents (no adaptation).

**Use Cases:**

- Baseline experiments (compare against adaptive curriculum)
- Interface validation (ensure curriculum system works)
- Debugging (isolate curriculum effects)

##### 2.3.3 State Persistence (Lines 71-100)

Simple serialization - just stores 4 configuration fields (difficulty, mode, meters, multiplier).

---

#### 2.4 Adversarial Implementation: `AdversarialCurriculum` (adversarial.py, 361 lines)

**Coverage:** 86%  
**Purpose:** Auto-tuning curriculum with 5-stage progression

---

##### 2.4.1 Stage Configuration (Lines 17-60)

**Data Structure: `StageConfig` (Lines 17-23):**

```python
class StageConfig(BaseModel):
    stage: int                      # 1-5
    active_meters: List[str]        # Which meters deplete
    depletion_multiplier: float     # 0.2-1.0 (% of base rate)
    reward_mode: str                # 'shaped' or 'sparse'
    description: str                # Human-readable
```

**5-Stage Progression (`STAGE_CONFIGS`, Lines 26-60):**

**Stage 1: Basic Needs (20% depletion, shaped)**

- Active: `['energy', 'hygiene']`
- Depletion: 0.2× (80% slower than normal)
- Reward: Shaped (milestone bonuses)
- **Rationale:** Learn basic movement + bed/shower loop

**Stage 2: Add Hunger (50% depletion, shaped)**

- Active: `['energy', 'hygiene', 'satiation']`
- Depletion: 0.5× (50% slower)
- Reward: Shaped
- **Rationale:** Introduce food management (HomeMeal/FastFood)

**Stage 3: Money Management (80% depletion, shaped)**

- Active: `['energy', 'hygiene', 'satiation', 'money']`
- Depletion: 0.8× (20% slower)
- Reward: Shaped
- **Rationale:** Balance income (Job/Labor) vs expenses

**Stage 4: All Meters (100% depletion, shaped)**

- Active: `['energy', 'hygiene', 'satiation', 'money', 'mood', 'social']`
- Depletion: 1.0× (full speed)
- Reward: Shaped
- **Rationale:** Master full complexity with training wheels (shaped rewards)

**Stage 5: SPARSE GRADUATION (100% depletion, sparse)**

- Active: `['energy', 'hygiene', 'satiation', 'money', 'mood', 'social']`
- Depletion: 1.0× (full speed)
- Reward: **SPARSE** (milestone only, no guidance)
- **Rationale:** Prove agent can survive without constant feedback

**Key Insight:** Only final stage is sparse - agents must EARN the right to graduate.

---

##### 2.4.2 Performance Tracker (Lines 62-106)

**Purpose:** Per-agent metric tracking for curriculum decisions.

**Class: `PerformanceTracker` (Lines 62-106):**

**State Tensors (Lines 67-76):**

```python
episode_rewards: torch.Tensor[num_agents]    # Cumulative reward this episode
episode_steps: torch.Tensor[num_agents]      # Steps survived this episode
prev_avg_reward: torch.Tensor[num_agents]    # Baseline for learning progress
agent_stages: torch.Tensor[num_agents]       # Current stage [1-5]
steps_at_stage: torch.Tensor[num_agents]     # Time at current stage
```

**Update Method (Lines 78-86):**

```python
def update_step(rewards: torch.Tensor, dones: torch.Tensor):
    # Accumulate rewards and steps
    episode_rewards += rewards
    episode_steps += 1.0
    steps_at_stage += 1.0
    
    # Reset on episode completion
    episode_rewards = where(dones, 0.0, episode_rewards)
    episode_steps = where(dones, 0.0, episode_steps)
```

**Metric Calculations:**

**Survival Rate (Lines 88-90):**

```python
def get_survival_rate(max_steps: int) -> torch.Tensor:
    return episode_steps / max_steps  # [0.0-1.0]
```

Example: 350 steps / 500 max = 0.7 survival rate (70%)

**Learning Progress (Lines 92-96):**

```python
def get_learning_progress() -> torch.Tensor:
    current_avg = episode_rewards / clamp(episode_steps, min=1.0)
    progress = current_avg - prev_avg_reward
    return progress  # Positive = improving, Negative = regressing
```

Example: Current avg reward 5.0, previous 3.5 → +1.5 learning progress (improving)

**Baseline Update (Lines 98-101):**

Called when agent advances/retreats - updates baseline for next comparison.

---

##### 2.4.3 Main Curriculum Class (Lines 108-361)

**Constructor (Lines 122-141):**

```python
def __init__(
    max_steps_per_episode: int = 500,         # Survival denominator
    survival_advance_threshold: float = 0.7,  # 70% survival to advance
    survival_retreat_threshold: float = 0.3,  # 30% survival retreats
    entropy_gate: float = 0.5,                # Policy convergence check
    min_steps_at_stage: int = 1000,           # Minimum stage duration
    device: torch.device = torch.device('cpu')
)
```

**Key Parameters:**

- **survival_advance_threshold (0.7):** Agent must survive 70%+ of episode to advance
- **survival_retreat_threshold (0.3):** Below 30% survival triggers retreat
- **entropy_gate (0.5):** Entropy < 0.5 indicates converged policy (deterministic actions)
- **min_steps_at_stage (1000):** Prevents premature advancement (must spend time at stage)

**YAML Loading (Lines 143-170):**

Factory method `from_yaml(config_path)` loads parameters from training config.

---

##### 2.4.4 Stage Advancement Logic (Lines 185-208)

**`_should_advance(agent_idx, entropy)` (Lines 185-208):**

**Multi-Signal Decision (ALL must be true):**

1. **Not at max stage:** `agent_stages[agent_idx] < 5`
2. **Minimum time spent:** `steps_at_stage[agent_idx] >= min_steps_at_stage` (1000 steps)
3. **High survival:** `survival_rate > 0.7` (surviving 70%+ of max steps)
4. **Positive learning:** `learning_progress > 0` (rewards improving)
5. **Converged policy:** `entropy < 0.5` (deterministic action selection)

**Rationale for Multi-Signal:**

- Survival alone: Could be lucky (random actions that worked once)
- Learning alone: Could be volatile (one good episode)
- Entropy alone: Could be stuck (converged to bad policy)
- **Together:** Strong evidence of mastery

**Example Advancement:**

```
Agent survives 380/500 steps = 76% survival ✓
Rewards: current_avg 6.2, prev_avg 4.8 = +1.4 progress ✓
Entropy: 0.42 < 0.5 (policy is deterministic) ✓
Steps at stage: 1250 > 1000 ✓
Stage: 2 < 5 ✓
→ ADVANCE to Stage 3!
```

---

##### 2.4.5 Stage Retreat Logic (Lines 210-230)

**`_should_retreat(agent_idx)` (Lines 210-230):**

**Retreat Conditions (OR logic - any triggers retreat):**

1. **Not at min stage:** `agent_stages[agent_idx] > 1`
2. **Minimum time spent:** `steps_at_stage[agent_idx] >= min_steps_at_stage`
3. **Low survival OR negative learning:**
   - `survival_rate < 0.3` (dying quickly)
   - `learning_progress < 0` (getting worse)

**Rationale:** If agent is struggling (low survival OR regressing), back off difficulty.

**Example Retreat:**

```
Agent survives 120/500 steps = 24% survival (< 30%) ✓
Steps at stage: 1100 > 1000 ✓
Stage: 3 > 1 ✓
→ RETREAT to Stage 2 (difficulty too high)
```

**Design Note:** Retreat takes priority over advancement (checked first in decision logic).

---

##### 2.4.6 Main Decision Generation (Lines 232-295)

**`get_batch_decisions_with_qvalues()` (Lines 232-295):**

**Purpose:** Main entry point called by `VectorizedPopulation` during training.

**Execution Flow:**

1. **Calculate Entropy (Line 241):**

   ```python
   entropies = self._calculate_action_entropy(q_values)
   ```

   Convert Q-values to action distribution entropy [0, 1]

2. **Per-Agent Decision Loop (Lines 244-289):**

   ```python
   for i, agent_id in enumerate(agent_ids):
       # Check retreat first (takes priority)
       if _should_retreat(i):
           agent_stages[i] -= 1
           steps_at_stage[i] = 0
           prev_avg_reward[i] = current_avg_i  # Update baseline
       
       # Then check advancement
       elif _should_advance(i, entropies[i]):
           agent_stages[i] += 1
           steps_at_stage[i] = 0
           prev_avg_reward[i] = current_avg_i  # Update baseline
       
       # Get current stage config
       stage = agent_stages[i].item()
       config = STAGE_CONFIGS[stage - 1]
       
       # Create decision
       difficulty_level = (stage - 1) / 4.0  # Normalize to [0.0-1.0]
       decision = CurriculumDecision(...)
       decisions.append(decision)
   ```

3. **Return Decisions (Line 295):**
   One `CurriculumDecision` per agent (may be different stages)

**Key Details:**

- **Retreat priority:** Checked before advancement (lines 247-252 before 253-258)
- **Per-agent baselines:** Only update `prev_avg_reward[i]` for advancing/retreating agent (not all agents)
- **Difficulty normalization:** Stage 1-5 → 0.0-1.0 for curriculum tracking
- **Stage counter reset:** `steps_at_stage[i] = 0` on transition

---

##### 2.4.7 Entropy Calculation (Lines 312-330)

**`_calculate_action_entropy(q_values)` (Lines 312-330):**

**Purpose:** Measure policy convergence (peaked distribution = low entropy).

**Algorithm:**

1. **Softmax Q-values to probabilities (Line 323):**

   ```python
   probs = softmax(q_values, dim=-1)  # [batch, 5 actions]
   ```

2. **Calculate entropy (Lines 326-327):**

   ```python
   log_probs = log(probs + 1e-10)  # Numerical stability
   entropy = -sum(probs * log_probs, dim=-1)  # Shannon entropy
   ```

3. **Normalize to [0, 1] (Lines 329-330):**

   ```python
   max_entropy = log(5) ≈ 1.609 (uniform distribution over 5 actions)
   normalized_entropy = entropy / log(num_actions)
   ```

**Interpretation:**

- **Entropy ≈ 1.0:** Uniform distribution (exploring - choosing actions randomly)
- **Entropy ≈ 0.5:** Moderate convergence (some preference emerging)
- **Entropy < 0.5:** Strong convergence (policy is deterministic) ✓ GATE PASSED
- **Entropy ≈ 0.0:** Deterministic (always choosing same action)

**Example:**

```
Q-values: [10.5, 2.3, 1.8, 2.1, 1.9]
Probs after softmax: [0.95, 0.01, 0.01, 0.01, 0.02]
Entropy: -0.95*log(0.95) - 4*(0.01*log(0.01)) ≈ 0.23
Normalized: 0.23 / 1.609 ≈ 0.14 (highly deterministic) ✓
```

---

##### 2.4.8 Testing Interface (Lines 297-310)

**`get_batch_decisions()` (Lines 297-310):**

**Purpose:** Simplified interface for testing (no Q-values available).

**Behavior:** Creates dummy Q-values with peaked distribution (low entropy):

```python
q_values = zeros(num_agents, 5)
q_values[:, 0] = 10.0  # Make action 0 dominant
```

This simulates a converged policy (entropy ≈ 0) for testing advancement logic.

---

##### 2.4.9 State Persistence (Lines 332-361)

**Two Interfaces (Legacy + Modern):**

**Modern: `state_dict()` / `load_state_dict()` (Lines 332-352):**

```python
def state_dict() -> Dict:
    return {
        'agent_stages': tracker.agent_stages.cpu(),
        'episode_rewards': tracker.episode_rewards.cpu(),
        'episode_steps': tracker.episode_steps.cpu(),
        'prev_avg_reward': tracker.prev_avg_reward.cpu(),
        'steps_at_stage': tracker.steps_at_stage.cpu(),
    }

def load_state_dict(state_dict: Dict):
    tracker.agent_stages = state_dict['agent_stages'].to(device)
    # ... restore all tensors
```

**Legacy: `checkpoint_state()` / `load_state()` (Lines 354-361):**

Thin wrappers around modern methods for backward compatibility.

**ACTION #11:** Remove legacy methods (15 minutes).

**Checkpointing Flow:**

1. Training calls `curriculum.state_dict()` every 100 episodes
2. Serialize to `checkpoint_ep{episode:05d}.pt` with torch.save()
3. On resume, load checkpoint and call `curriculum.load_state_dict(state)`
4. Agents resume from their individual stages (mixed-stage training preserved)

---

#### 2.5 Curriculum Decision Data Structure

**Location:** `src/townlet/training/state.py`

**Class: `CurriculumDecision` (Pydantic DTO):**

```python
class CurriculumDecision(BaseModel):
    difficulty_level: float        # [0.0-1.0] normalized stage
    active_meters: List[str]       # Which meters deplete
    depletion_multiplier: float    # Rate multiplier [0.2-1.0]
    reward_mode: str              # 'shaped' or 'sparse'
    reason: str                   # Human-readable explanation
```

**Used By:**

- Environment: Applies `depletion_multiplier` to base rates
- Reward calculation: Switches between shaped/sparse based on `reward_mode`
- Logging: `reason` field for debugging and visualization

**Note:** Environment currently does NOT use `active_meters` or `depletion_multiplier` (ACTION item).

---

#### 2.6 Integration with Training Loop

**Call Site:** `VectorizedPopulation.step_population()` (population/vectorized.py)

**Timing:** Once per episode after reset (not every step).

**Flow:**

```python
# 1. Forward pass to get Q-values
q_values = q_network(observations)

# 2. Get curriculum decisions (per-agent)
decisions = curriculum.get_batch_decisions_with_qvalues(
    agent_states, agent_ids, q_values
)

# 3. Extract parameters for environment
# (Currently not used - ACTION item to integrate)

# 4. Update performance tracker (after each step)
curriculum.tracker.update_step(rewards, dones)
```

**Known Issue:** Curriculum makes decisions but environment doesn't apply them yet. Placeholder for future integration.

---

#### 2.7 Testing Status

**Coverage: 86%** (20 lines missing)

**Tested:**

- ✅ Stage progression logic (advance/retreat)
- ✅ Entropy calculation (softmax + normalization)
- ✅ Survival rate tracking
- ✅ Learning progress calculation
- ✅ Multi-signal decision (survival + learning + entropy)
- ✅ State persistence (checkpoint/restore)
- ✅ Per-agent stage independence

**Gaps (14%):**

- ⚠️ YAML loading (`from_yaml` method)
- ⚠️ Some edge cases in bounds checking
- ⚠️ Legacy checkpoint methods (ACTION #11 - will remove)

**Test Files:**

- `tests/test_townlet/test_adversarial_curriculum.py` - Main test suite
- `tests/test_townlet/test_static_curriculum.py` - Baseline implementation

---

#### 2.8 Design Strengths

1. **Per-Agent Stages:** Mixed-stage training enables advanced agents to explore harder scenarios while beginners learn basics
2. **Multi-Signal Gating:** Prevents premature advancement (lucky streaks don't count)
3. **Retreat Mechanism:** Agents can back off if difficulty too high (adaptive safety net)
4. **Entropy-Based Convergence:** Uses policy distribution (not just rewards) to assess readiness
5. **State Machine Clarity:** 5 stages with explicit configurations (easy to understand/modify)

---

#### 2.9 Known Limitations

1. **Entropy Gate May Be Too Strict:** 0.5 threshold might prevent valid exploration in later stages
2. **Stage 5 Sparse Rewards:** Agents may struggle with sudden removal of guidance (could add Stage 6 with partial shaping)
3. **Min Steps Requirement:** 1000 steps is arbitrary (could be adaptive based on stage)
4. **No Automatic Hyperparameter Tuning:** Thresholds are manual (0.7 advance, 0.3 retreat)
5. **Environment Integration Incomplete:** Curriculum decisions computed but not applied (ACTION item)

---

#### 2.10 Future Enhancements

**Potential Improvements:**

1. **Adaptive Thresholds:** Learn optimal advancement thresholds from population statistics
2. **Gradual Sparse Transition:** Stage 4.5 with 50% shaped + 50% sparse rewards
3. **Entropy Annealing:** Relax entropy gate as stages progress (early: strict, late: flexible)
4. **Multi-Objective Stages:** Stages defined by learned skills (navigation, income, health management) not just meters
5. **Population-Level Curriculum:** Synchronize stages across population for competitive training

**ACTION Items Related to Curriculum:**

- **Integrate with Environment:** Apply `depletion_multiplier` and `active_meters` (1-2 days)
- **Remove Legacy Methods:** Delete `checkpoint_state()` / `load_state()` (15 minutes)
- **Add Stage 6:** Gradual sparse transition for smoother graduation (2-3 days)

---

### 3. Exploration System

**Location:** `src/townlet/exploration/`  
**Core Files:** `epsilon_greedy.py` (152 lines), `rnd.py` (276 lines), `adaptive_intrinsic.py` (204 lines)  
**Coverage:** 100% (epsilon_greedy, adaptive_intrinsic), 82% (rnd)  
**Complexity:** 🟢 LOW (Well-abstracted, composition-based)  
**Purpose:** Action selection strategies with novelty-seeking behavior

---

#### 3.1 System Overview

The Exploration System manages the **exploration-exploitation tradeoff** through pluggable strategies. It controls how agents select actions during training and optionally provides **intrinsic motivation** rewards to encourage novelty-seeking behavior.

**Core Responsibilities:**

1. **Action Selection** - Epsilon-greedy sampling with action masking (hot path - every step)
2. **Intrinsic Rewards** - Novelty bonuses from RND prediction error (optional)
3. **Adaptive Annealing** - Variance-based intrinsic weight decay (Phase 3 innovation)
4. **Network Training** - Update RND predictor from experience (warm path)
5. **State Persistence** - Checkpoint/restore epsilon and RND networks

**Design Pattern:** Composition over inheritance (AdaptiveIntrinsic wraps RND).

**⚠️ INCONSISTENCY DETECTED:** Current system uses **pure sparse rewards** (Phase 3.5+), but curriculum still references "shaped" mode. The exploration system is correct; curriculum documentation may be outdated.

---

#### 3.2 Abstract Base: `ExplorationStrategy` (base.py, 117 lines)

**Coverage:** 75%  
**Purpose:** Define interface for pluggable exploration strategies

##### 3.2.1 Core Interface (Lines 14-117)

```python
class ExplorationStrategy(ABC):
    @abstractmethod
    def select_actions(
        q_values: torch.Tensor,           # [batch, num_actions]
        agent_states: BatchedAgentState,  # Contains epsilons
        action_masks: torch.Tensor        # [batch, num_actions] bool
    ) -> torch.Tensor                     # [batch] actions

    @abstractmethod
    def compute_intrinsic_rewards(
        observations: torch.Tensor        # [batch, obs_dim]
    ) -> torch.Tensor                     # [batch] rewards

    @abstractmethod
    def update(batch: Dict[str, torch.Tensor]) -> None

    @abstractmethod
    def checkpoint_state() -> Dict[str, Any]

    @abstractmethod
    def load_state(state: Dict[str, Any]) -> None
```

**Hot Path Performance Notes (Lines 24-47):**

- `select_actions()` runs EVERY STEP for all agents
- Must be GPU-optimized, no validation, no CPU transfers
- Action masking is critical (respect boundary constraints)

**Intrinsic Rewards (Lines 49-70):**

- For RND: Prediction error as novelty signal
- For epsilon-greedy: Returns zeros (no intrinsic motivation)
- Hot path - runs every step

**Network Updates (Lines 72-88):**

- Called after replay buffer sampling (not hot path)
- Can be slow (backward pass acceptable)
- No-op for epsilon-greedy (no networks to train)

---

#### 3.3 Baseline: `EpsilonGreedyExploration` (epsilon_greedy.py, 152 lines)

**Coverage:** 100% ✅  
**Purpose:** Simple epsilon-greedy without intrinsic motivation

##### 3.3.1 Constructor (Lines 22-40)

```python
def __init__(
    epsilon: float = 1.0,           # Initial exploration rate (100% random)
    epsilon_decay: float = 0.995,   # Decay per episode (~1% per episode)
    epsilon_min: float = 0.01       # Minimum (1% random exploration)
)
```

**Decay Schedule:**

- Episode 1: ε = 1.0 (pure random)
- Episode 100: ε = 1.0 × 0.995^100 ≈ 0.606 (60% random)
- Episode 500: ε = 1.0 × 0.995^500 ≈ 0.082 (8% random)
- Episode 1000+: ε = 0.01 (1% random, mostly greedy)

##### 3.3.2 Action Selection with Masking (Lines 42-90)

**`select_actions()` (Lines 42-90):**

**Critical Feature:** Respects action masks to prevent invalid actions.

**Algorithm:**

1. **Apply Action Masking (Lines 62-67):**

   ```python
   if action_masks is not None:
       masked_q_values = q_values.clone()
       masked_q_values[~action_masks] = float('-inf')
   ```

   Invalid actions get -∞ Q-values → never selected by argmax

2. **Greedy Actions (Line 70):**

   ```python
   greedy_actions = argmax(masked_q_values, dim=1)
   ```

3. **Random Actions (Lines 72-84):**

   ```python
   if action_masks is not None:
       # Sample ONLY from valid actions per agent
       for i in range(batch_size):
           valid_actions = where(action_masks[i])[0]
           random_idx = randint(0, len(valid_actions), (1,))
           random_actions[i] = valid_actions[random_idx]
   else:
       random_actions = randint(0, num_actions, (batch_size,))
   ```

   **Key:** Random exploration respects masking (no boundary violations)

4. **Epsilon Selection (Lines 86-90):**

   ```python
   explore_mask = rand(batch_size) < agent_states.epsilons
   actions = where(explore_mask, random_actions, greedy_actions)
   ```

   Per-agent epsilon from `agent_states.epsilons` tensor

**Design Note:** Per-agent epsilon enables mixed-epsilon populations (early agents can explore more).

##### 3.3.3 No Intrinsic Motivation (Lines 92-103)

```python
def compute_intrinsic_rewards(observations):
    return zeros(batch_size, device=device)  # No novelty bonus
```

##### 3.3.4 Epsilon Decay (Lines 111-113)

```python
def decay_epsilon():
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Called once per episode by training loop.

##### 3.3.5 State Persistence (Lines 115-135)

Simple serialization: 3 floats (epsilon, epsilon_decay, epsilon_min).

---

#### 3.4 RND: `RNDExploration` (rnd.py, 276 lines)

**Coverage:** 82%  
**Purpose:** Random Network Distillation for novelty-based intrinsic rewards

##### 3.4.1 RND Network Architecture (Lines 11-36)

**Class: `RNDNetwork` (3-layer MLP):**

```python
Architecture:
  [obs_dim] → Linear(256) → ReLU
           → Linear(128) → ReLU
           → Linear(embed_dim) → [embeddings]

Default dimensions:
  obs_dim = 70 (for 8×8 grid full observability)
  embed_dim = 128
```

**Design Match:** Matches `SimpleQNetwork` architecture for consistency.

##### 3.4.2 Constructor (Lines 48-92)

```python
def __init__(
    obs_dim: int = 70,
    embed_dim: int = 128,
    learning_rate: float = 1e-4,
    training_batch_size: int = 128,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    device: torch.device = torch.device('cpu')
)
```

**Two Networks (Lines 76-86):**

1. **Fixed Network (Lines 76-79):**

   ```python
   fixed_network = RNDNetwork(obs_dim, embed_dim).to(device)
   for param in fixed_network.parameters():
       param.requires_grad = False  # FROZEN
   ```

   Random initialization, never trained (provides novelty target)

2. **Predictor Network (Lines 81-82):**

   ```python
   predictor_network = RNDNetwork(obs_dim, embed_dim).to(device)
   ```

   Trained to match fixed network (learns familiar states)

**Training Buffer (Lines 88-92):**

```python
obs_buffer = []  # Accumulate observations
step_counter = 0
```

Buffers observations until batch size reached (128), then trains.

##### 3.4.3 Action Selection (Lines 94-146)

**⚠️ DUPLICATION DETECTED (ACTION #10):**

`select_actions()` is **IDENTICAL** to `EpsilonGreedyExploration.select_actions()` (Lines 94-146 = Lines 42-90 in epsilon_greedy.py).

**Copy-Paste Code:**

- Same epsilon-greedy logic
- Same action masking
- Same random sampling from valid actions

**Refactoring Opportunity:** RND should COMPOSE with EpsilonGreedy, not duplicate code.

**Proposed Fix:**

```python
class RNDExploration(ExplorationStrategy):
    def __init__(...):
        # Composition instead of duplication
        self.action_selector = EpsilonGreedyExploration(...)
        self.fixed_network = ...
        self.predictor_network = ...
    
    def select_actions(self, q_values, agent_states, action_masks):
        # Delegate to epsilon-greedy
        return self.action_selector.select_actions(q_values, agent_states, action_masks)
```

##### 3.4.4 Intrinsic Reward Computation (Lines 148-165)

**`compute_intrinsic_rewards()` (Lines 148-165):**

**Algorithm (RND Core):**

```python
with torch.no_grad():
    target_features = fixed_network(observations)      # [batch, 128]
    predicted_features = predictor_network(observations)  # [batch, 128]
    
    # MSE per sample (high error = novel = high reward)
    mse_per_sample = ((target_features - predicted_features) ** 2).mean(dim=1)

return mse_per_sample  # [batch] intrinsic rewards
```

**Intuition:**

- **Familiar states:** Predictor learned to match fixed → low MSE → low intrinsic reward
- **Novel states:** Predictor hasn't seen yet → high MSE → high intrinsic reward
- **Exploration incentive:** Agent gets bonus for visiting novel states

**Example Values:**

- Just visited state: MSE ≈ 0.001 (predictor accurate)
- Never visited: MSE ≈ 5.0 (predictor confused)
- After 1000 visits: MSE ≈ 0.01 (learning curve)

##### 3.4.5 Network Training (Lines 167-212)

**`update()` (Lines 167-180):**

**Buffered Training:**

1. Add observations to buffer (Line 173-175)
2. Train when buffer reaches 128 samples (Line 178-179)
3. Call `update_predictor()` to perform gradient step

**`update_predictor()` (Lines 182-212):**

**Training Algorithm:**

```python
# 1. Stack observations into batch (Lines 187-191)
obs_batch = stack(obs_buffer[:128]).to(device)
obs_buffer = obs_buffer[128:]  # Clear buffer

# 2. Compute prediction loss (Lines 193-196)
target = fixed_network(obs_batch).detach()  # Stop gradients
predicted = predictor_network(obs_batch)
loss = mse_loss(predicted, target)

# 3. Gradient step (Lines 198-201)
optimizer.zero_grad()
loss.backward()
optimizer.step()

return loss.item()  # For logging
```

**Why Batch Training?**

- More efficient than per-sample updates (amortize overhead)
- Better gradient estimates (less noisy)
- GPU utilization (batch operations)

**Training Frequency:**

- Every 128 observations collected
- For num_agents=1: Every 128 steps
- For num_agents=16: Every 8 steps (128 / 16)

##### 3.4.6 Novelty Visualization (Lines 214-244)

**`get_novelty_map()` (Lines 214-244):**

**Purpose:** Generate heatmap of novelty values for all grid positions (for frontend visualization).

**Algorithm:**

```python
for row in range(grid_size):
    for col in range(grid_size):
        # Create observation with agent at (row, col)
        obs = zeros(1, obs_dim)
        obs[0, row * grid_size + col] = 1.0  # Grid one-hot
        obs[0, 64:70] = 0.5  # Meters placeholder
        
        # Compute novelty
        novelty = compute_intrinsic_rewards(obs)
        novelty_map[row, col] = novelty.item()

return novelty_map  # [grid_size, grid_size]
```

**Use Case:** Frontend displays heatmap showing which areas are under-explored.

##### 3.4.7 Epsilon Decay & State Persistence (Lines 246-276)

**Epsilon Decay (Lines 246-248):** Same as `EpsilonGreedyExploration`.

**State Persistence (Lines 250-276):**

Serializes:

- Both network state dicts (fixed + predictor)
- Optimizer state
- Epsilon parameters
- Observation dimensions

**Note:** Fixed network weights ARE saved (for reproducibility), even though they're never updated.

---

#### 3.5 Adaptive: `AdaptiveIntrinsicExploration` (adaptive_intrinsic.py, 204 lines)

**Coverage:** 100% ✅  
**Purpose:** RND with variance-based annealing (Phase 3 innovation)

##### 3.5.1 Key Innovation: Variance-Based Annealing

**Problem Solved:** Premature annealing bug (October 30, 2025).

**Original Bug:**

- `variance_threshold = 10.0` was too low
- Agents that "consistently failed" (low variance, low survival) triggered annealing
- Annealing should only happen when "consistently succeeding"

**Fixed (Lines 28, 156-162):**

```python
variance_threshold: float = 100.0  # Increased from 10.0
MIN_SURVIVAL_FOR_ANNEALING = 50.0  # Must be succeeding first

return (variance < variance_threshold and
        mean_survival > MIN_SURVIVAL_FOR_ANNEALING)
```

**Two-Gate System:**

1. **Low variance:** Performance is consistent (not volatile)
2. **Good performance:** Surviving 50+ steps on average

Both must be true to reduce intrinsic weight.

##### 3.5.2 Constructor (Lines 18-75)

```python
def __init__(
    obs_dim: int = 70,
    embed_dim: int = 128,
    rnd_learning_rate: float = 1e-4,
    rnd_training_batch_size: int = 128,
    initial_intrinsic_weight: float = 1.0,    # Start at 100% intrinsic
    min_intrinsic_weight: float = 0.0,        # Can anneal to 0%
    variance_threshold: float = 100.0,        # FIXED from 10.0
    survival_window: int = 100,               # Track last 100 episodes
    decay_rate: float = 0.99,                 # Exponential decay (1% per trigger)
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    device: torch.device = torch.device('cpu')
)
```

**Composition Pattern (Lines 52-62):**

```python
self.rnd = RNDExploration(
    obs_dim=obs_dim,
    embed_dim=embed_dim,
    learning_rate=rnd_learning_rate,
    training_batch_size=rnd_training_batch_size,
    epsilon_start=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    device=device,
)
```

**Design:** Contains RND instance (not inheritance) - adds annealing behavior.

**State Tracking (Lines 64-72):**

```python
current_intrinsic_weight: float = 1.0  # Starts at full weight
survival_history: List[float] = []     # Track last 100 survivals
```

##### 3.5.3 Delegated Methods (Lines 77-115)

**Action Selection (Lines 77-90):** Delegates to `self.rnd.select_actions()` (no modification).

**Intrinsic Rewards (Lines 92-105):** **WEIGHTED** by current annealing factor.

```python
def compute_intrinsic_rewards(observations):
    rnd_novelty = rnd.compute_intrinsic_rewards(observations)
    return rnd_novelty * current_intrinsic_weight  # Scale by weight
```

**Example:**

- Early training: weight=1.0 → full intrinsic rewards
- After 200 annealing triggers: weight=0.99^200 ≈ 0.134 → 13% intrinsic
- After 500 triggers: weight=0.99^500 ≈ 0.007 → 0.7% intrinsic (almost pure extrinsic)

**Network Update (Lines 107-115):** Delegates to `self.rnd.update()`.

##### 3.5.4 Annealing Logic (Lines 117-173)

**`update_on_episode_end()` (Lines 117-133):**

Called after each episode completes (by training loop).

```python
def update_on_episode_end(survival_time: float):
    # 1. Add to history
    survival_history.append(survival_time)
    
    # 2. Maintain window (last 100 episodes)
    if len(survival_history) > survival_window:
        survival_history = survival_history[-survival_window:]
    
    # 3. Check for annealing trigger
    if should_anneal():
        anneal_weight()
```

**`should_anneal()` (Lines 135-162):**

**Two-Gate Check:**

```python
def should_anneal() -> bool:
    if len(survival_history) < survival_window:
        return False  # Need 100 episodes of data
    
    recent_survivals = tensor(survival_history[-100:])
    variance = var(recent_survivals).item()
    mean_survival = mean(recent_survivals).item()
    
    MIN_SURVIVAL_FOR_ANNEALING = 50.0
    
    # BOTH gates must pass
    return (variance < 100.0 and
            mean_survival > 50.0)
```

**Example Scenarios:**

**Scenario 1: Consistently Failing (NO ANNEAL)**

- Survivals: [10, 15, 12, 8, 14, ...] (avg = 12)
- Variance: 5.0 < 100.0 ✓
- Mean: 12 < 50.0 ✗
- **Result:** Don't anneal (not succeeding yet)

**Scenario 2: Volatile Success (NO ANNEAL)**

- Survivals: [200, 50, 180, 30, 220, ...] (avg = 136)
- Variance: 5200 > 100.0 ✗
- Mean: 136 > 50.0 ✓
- **Result:** Don't anneal (inconsistent - still exploring)

**Scenario 3: Consistent Success (ANNEAL!)**

- Survivals: [180, 185, 178, 190, 182, ...] (avg = 183)
- Variance: 20.0 < 100.0 ✓
- Mean: 183 > 50.0 ✓
- **Result:** Anneal! (converged to good policy)

**`anneal_weight()` (Lines 164-167):**

```python
def anneal_weight():
    new_weight = current_intrinsic_weight * 0.99
    current_intrinsic_weight = max(new_weight, min_intrinsic_weight)
```

**Exponential Decay:** Weight drops 1% per trigger.

**Annealing Trajectory (100 triggers):**

- Trigger 0: 1.0 (100%)
- Trigger 10: 0.904 (90%)
- Trigger 50: 0.605 (60%)
- Trigger 100: 0.366 (37%)
- Trigger 200: 0.134 (13%)
- Trigger 460: 0.01 (1% - near pure extrinsic)

##### 3.5.5 Utility & Persistence (Lines 169-204)

**Get Weight (Lines 169-171):** For logging/visualization.

**Decay Epsilon (Lines 173-175):** Delegates to RND.

**State Persistence (Lines 177-204):**

Serializes:

- RND state (nested checkpoint)
- Intrinsic weight + parameters
- Full survival history (last 100 episodes)

---

#### 3.6 Dual Reward System Integration

**Location:** `src/townlet/training/replay_buffer.py`

**Storage (Lines 30-40):**

Replay buffer stores SEPARATE rewards:

```python
transitions = {
    'observations': obs,
    'actions': actions,
    'rewards_extrinsic': rewards_extrinsic,  # From environment
    'rewards_intrinsic': rewards_intrinsic,  # From RND
    'next_observations': next_obs,
    'dones': dones
}
```

**Sampling (Lines 45-60):**

Combined at sample time with current intrinsic weight:

```python
def sample(batch_size, intrinsic_weight):
    batch = random_sample(buffer, batch_size)
    
    # Combine rewards
    combined_rewards = (batch['rewards_extrinsic'] +
                       batch['rewards_intrinsic'] * intrinsic_weight)
    
    return {
        'observations': batch['observations'],
        'actions': batch['actions'],
        'rewards': combined_rewards,  # Used for Q-learning
        'next_observations': batch['next_observations'],
        'dones': batch['dones']
    }
```

**Why Separate Storage?**

- Intrinsic weight changes over time (annealing)
- Re-weighting historical experiences with new weight
- Pure extrinsic rewards preserved for final evaluation

---

#### 3.7 Training Loop Integration

**Call Sites:** `VectorizedPopulation.step_population()` (population/vectorized.py)

**Action Selection (Every Step):**

```python
# 1. Forward pass
q_values = q_network(observations)

# 2. Get action masks
action_masks = env.get_action_masks()

# 3. Select actions (hot path - GPU optimized)
actions = exploration.select_actions(q_values, agent_states, action_masks)
```

**Reward Computation (Every Step):**

```python
# 4. Step environment
observations, rewards_extrinsic, dones, info = env.step(actions)

# 5. Compute intrinsic rewards (hot path - GPU)
rewards_intrinsic = exploration.compute_intrinsic_rewards(observations)

# 6. Store in replay buffer (dual rewards)
replay_buffer.push(obs, actions, rewards_extrinsic, rewards_intrinsic, next_obs, dones)
```

**Network Training (Every 4 Steps):**

```python
# 7. Sample replay buffer (combines rewards with current weight)
batch = replay_buffer.sample(batch_size=64, intrinsic_weight=exploration.get_intrinsic_weight())

# 8. Train Q-network (DQN update)
q_network.train(batch)

# 9. Train RND predictor (if applicable)
exploration.update(batch)
```

**Episode End (Once Per Episode):**

```python
# 10. Decay epsilon
exploration.decay_epsilon()

# 11. Update annealing (adaptive only)
if isinstance(exploration, AdaptiveIntrinsicExploration):
    exploration.update_on_episode_end(survival_time=episode_steps)
```

---

#### 3.8 Testing Status

**Coverage by File:**

- ✅ `epsilon_greedy.py` - 100% (all code paths tested)
- ✅ `adaptive_intrinsic.py` - 100% (annealing logic validated)
- ⚠️ `rnd.py` - 82% (some gaps in edge cases)
- ⚠️ `base.py` - 75% (abstract methods not executed)

**Tested:**

- ✅ Action selection with masking (boundary constraints)
- ✅ Epsilon-greedy random vs greedy selection
- ✅ RND prediction error calculation
- ✅ Variance-based annealing triggers (two-gate system)
- ✅ Exponential decay of intrinsic weight
- ✅ State persistence (checkpoint/restore)
- ✅ Composition pattern (Adaptive wraps RND)

**Gaps (18% in rnd.py):**

- ⚠️ Novelty map generation (`get_novelty_map()`)
- ⚠️ Some edge cases in buffer management
- ⚠️ Optimizer state restore edge cases

**Test Files:**

- `tests/test_townlet/test_epsilon_greedy.py` - Epsilon-greedy baseline
- `tests/test_townlet/test_adaptive_intrinsic.py` - Annealing logic
- No dedicated RND tests (covered through Adaptive tests)

---

#### 3.9 Design Strengths

1. **Composition over Inheritance:** Adaptive wraps RND (clean separation)
2. **Action Masking Integration:** Respects boundary constraints in ALL strategies
3. **Dual Reward System:** Separate storage enables re-weighting over time
4. **Variance-Based Annealing:** Sophisticated gating (not just time-based)
5. **Two-Gate Safety:** Prevents premature annealing from consistent failure
6. **Pluggable Interface:** Easy to add new strategies (ICM, curiosity, etc.)

---

#### 3.10 Known Issues

1. **Code Duplication (ACTION #10):** RND duplicates epsilon-greedy action selection (100 lines)
   - **Fix:** Composition instead of copy-paste (1-2 hours)
   - **Impact:** Reduce 276 → 180 lines in rnd.py

2. **CPU Transfers in RND:** Observation buffer moves to CPU then back to GPU
   - **Fix:** Keep buffer on GPU (ACTION #6, 1 day)
   - **Impact:** 10-20% speedup in RND training

3. **Annealing Trigger Frequency:** Checks every episode (could be every N episodes)
   - **Fix:** Configurable check frequency (30 minutes)
   - **Impact:** Reduce variance computation overhead

4. **Fixed Network Saved:** RND checkpoints include frozen network (unnecessary)
   - **Fix:** Only save random seed (reproduce fixed network)
   - **Impact:** 50% smaller checkpoints

5. **No Entropy-Based Annealing:** Uses survival variance, not policy entropy
   - **Potential:** Combine with curriculum entropy gate
   - **Impact:** More principled annealing trigger

---

#### 3.11 Inconsistency Notes

**⚠️ PURE SPARSE TRAINING (Current State):**

Based on code review, the system currently uses **pure sparse rewards** (milestone-based only, no shaped rewards). This is inconsistent with:

1. **Curriculum System Documentation:** Still references "shaped" vs "sparse" modes
2. **Environment Reward Calculation:** Only `_calculate_shaped_rewards()` (milestone) is active
3. **Complex Shaped Rewards:** Disabled since Phase 2 (proximity hacking bugs)

**Actual Reward Structure (as of Phase 3.5):**

```
Extrinsic Rewards (from environment):
  - Every 10 steps: +0.5
  - Every 100 steps: +5.0
  - Death: -100.0

Intrinsic Rewards (from RND):
  - Initial weight: 1.0 (equal to extrinsic)
  - Anneals to ~0.0 over training
  - Based on novelty (prediction error)

Total Reward = Extrinsic + (Intrinsic × weight)
```

**Curriculum "shaped" Mode:** Effectively means "milestone rewards" (not meter-based shaping).

**ACTION Item:** Update curriculum documentation to reflect pure sparse reality.

---

#### 3.12 Future Enhancements

**Potential Improvements:**

1. **ICM (Intrinsic Curiosity Module):** Forward model prediction error
2. **NGU (Never Give Up):** Episodic + lifetime novelty
3. **Entropy-Regularized Exploration:** Maximize policy entropy directly
4. **Hierarchical Exploration:** Different strategies per curriculum stage
5. **Population Diversity Bonus:** Reward agents for exploring different areas

**ACTION Items:**

- **ACTION #10:** Deduplicate epsilon-greedy code in RND (1-2 hours)
- **ACTION #6:** GPU optimization for RND buffer (1 day)
- **Fix documentation:** Update curriculum "shaped" to "milestone" terminology

---

### 4. Neural Network System

**Location:** `src/townlet/agent/`  
**Core File:** `networks.py` (250 lines)  
**Coverage:** 98%  
**Complexity:** 🟡 MODERATE (RecurrentSpatialQNetwork has sophisticated architecture)  
**Purpose:** Q-value estimation with full and partial observability support

---

#### 4.1 System Overview

The Neural Network System implements **Q-value function approximation** for both full and partial observability environments. It provides two architectures optimized for different complexity levels.

**Core Responsibilities:**

1. **Q-Value Prediction** - Map observations → action values (hot path - every step)
2. **Hidden State Management** - Maintain LSTM memory for recurrent networks (POMDP)
3. **Feature Encoding** - Extract features from vision, position, meters, affordances
4. **Gradient Computation** - Backpropagation for DQN updates (warm path)
5. **State Persistence** - Checkpoint/restore network weights

**Design Pattern:** Two architectures for different observability modes (Level 1 vs Level 2).

**⚠️ KNOWN ISSUE (ACTION #9):** RecurrentSpatialQNetwork may not effectively use LSTM history. Needs "root and branch reimagining" based on testing discoveries (October 31, 2025).

---

#### 4.2 Simple Architecture: `SimpleQNetwork` (networks.py, Lines 8-30)

**Coverage:** 100% ✅  
**Purpose:** Basic MLP for full observability (Level 1)

##### 4.2.1 Architecture (Lines 8-30)

```python
class SimpleQNetwork(nn.Module):
    def __init__(obs_dim: int, action_dim: int, hidden_dim: int = 128):
        net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),      # obs_dim → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # 128 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)    # 128 → action_dim
        )
```

**Architecture Diagram:**

```
Input: [batch, obs_dim] observations
   ↓
Linear(obs_dim → 128) → ReLU
   ↓
Linear(128 → 128) → ReLU
   ↓
Linear(128 → action_dim)
   ↓
Output: [batch, action_dim] Q-values
```

**Typical Dimensions:**

- Level 1.5 (8×8 grid, full observability): `obs_dim = 90` (64 grid + 8 meters + 16 affordance + 2 temporal)
- Action dim: `5` (UP, DOWN, LEFT, RIGHT, INTERACT)
- Hidden dim: `128` (default, matches RND network)

**Total Parameters:** ~12K for obs_dim=90, action_dim=5

##### 4.2.2 Forward Pass (Lines 20-30)

```python
def forward(x: torch.Tensor) -> torch.Tensor:
    """Forward pass.
    
    Args:
        x: [batch, obs_dim] observations
    
    Returns:
        q_values: [batch, action_dim]
    """
    return self.net(x)
```

**Simple feedforward:** No hidden state, no memory, just obs → Q-values.

**Use Cases:**

- Level 1: Full observability baseline
- Level 1.5: Current active system (sparse rewards, full obs)
- Debugging: Isolate environment issues from network complexity

---

#### 4.3 Recurrent Architecture: `RecurrentSpatialQNetwork` (networks.py, Lines 33-250)

**Coverage:** 98% (1 line missing)  
**Purpose:** CNN + LSTM for partial observability (Level 2 POMDP)

##### 4.3.1 Architecture Overview (Lines 33-64)

**Docstring Claims (Lines 36-54):**

```
Architecture:
- Vision Encoder: CNN for local window → 128 features
- Position Encoder: (x, y) → 32 features
- Meter Encoder: 8 meters → 32 features
- Affordance Encoder: 15 affordance types → 32 features
- LSTM: 224 input → 256 hidden
- Q-Head: 256 → 128 → action_dim

Handles partial observations:
- Grid: [batch, window_size²] flattened local window (25 for 5×5)
- Position: [batch, 2] normalized (x, y)
- Meters: [batch, 8] normalized meter values
- Affordance: [batch, 15] one-hot affordance type (14 types + "none")
```

**⚠️ ISSUE:** Docstring says 15 affordance types, but environment uses 16 (15 types + "none").

##### 4.3.2 Constructor (Lines 66-135)

```python
def __init__(
    action_dim: int = 5,
    window_size: int = 5,        # 5×5 local vision
    num_meters: int = 8,
    hidden_dim: int = 256        # LSTM hidden dim
)
```

**Component Encoders:**

**1. Vision Encoder (Lines 77-86) - CNN:**

```python
vision_encoder = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [batch,1,5,5] → [batch,16,5,5]
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), # [batch,16,5,5] → [batch,32,5,5]
    nn.ReLU(),
    nn.Flatten(),                                 # [batch, 32*5*5] = [batch, 800]
    nn.Linear(800, 128),                          # [batch, 800] → [batch, 128]
    nn.ReLU(),
)
```

**Architecture:** 2-layer CNN preserving spatial dimensions (padding=1), then flatten + FC to compress.

**Total Parameters:** ~102K parameters in vision encoder alone (largest component).

**2. Position Encoder (Lines 88-92) - MLP:**

```python
position_encoder = nn.Sequential(
    nn.Linear(2, 32),  # (x, y) → 32 features
    nn.ReLU(),
)
```

**Purpose:** Encode agent's absolute position in grid (for spatial reasoning).

**3. Meter Encoder (Lines 94-98) - MLP:**

```python
meter_encoder = nn.Sequential(
    nn.Linear(8, 32),  # 8 meters → 32 features
    nn.ReLU(),
)
```

**Meters:** energy, hygiene, satiation, money, mood, social, health, fitness.

**4. Affordance Encoder (Lines 100-104) - MLP:**

```python
affordance_encoder = nn.Sequential(
    nn.Linear(15, 32),  # 15 affordance types → 32 features
    nn.ReLU(),
)
```

**⚠️ DIMENSION MISMATCH:** Encoder expects 15 inputs, but observation has 16 (15 affordances + "none").

**Likely Bug:** Last dimension is dropped or observation construction is inconsistent.

**5. LSTM Layer (Lines 106-113):**

```python
lstm_input_dim = 128 + 32 + 32 + 32  # 224 total
lstm = nn.LSTM(
    input_size=224,
    hidden_size=256,
    num_layers=1,
    batch_first=True
)
```

**Purpose:** Maintain temporal memory across steps (remember history).

**Hidden State:** `(h, c)` each `[1, batch, 256]` (num_layers=1).

**6. Q-Head (Lines 115-119) - MLP:**

```python
q_head = nn.Sequential(
    nn.Linear(256, 128),  # LSTM output → 128
    nn.ReLU(),
    nn.Linear(128, action_dim)  # 128 → 5 actions
)
```

**Final layer:** No activation (raw Q-values, can be negative).

**Hidden State Attribute (Lines 121-122):**

```python
self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
```

Stores current LSTM state (maintained across episode steps).

##### 4.3.3 Forward Pass (Lines 137-199)

```python
def forward(
    obs: torch.Tensor,                              # [batch, obs_dim]
    hidden: Optional[Tuple[h, c]] = None
) -> Tuple[q_values, new_hidden]:
```

**Step-by-Step:**

**1. Parse Observation Components (Lines 150-159):**

```python
batch_size = obs.shape[0]
grid_size_flat = window_size * window_size  # 25 for 5×5

# Split observation tensor
grid = obs[:, :25]                    # [batch, 25] local window
position = obs[:, 25:27]              # [batch, 2] (x, y)
meters = obs[:, 27:35]                # [batch, 8] meter values
affordance = obs[:, 35:]              # [batch, 15] affordance type
```

**⚠️ DIMENSION ISSUE:** If observation has 16 affordance dims, `affordance = obs[:, 35:]` gives 16, but encoder expects 15.

**Observation Dimension Calculation:**

```
Level 2 POMDP (from environment):
  - Grid: 25 (5×5 window)
  - Position: 2
  - Meters: 8
  - Affordance: 16 (15 types + "none")
  - Temporal: 2 (time + progress)
  Total: 53 dims (without temporal) or 55 dims (with temporal)

Network Expectation (from forward):
  - Grid: 25
  - Position: 2
  - Meters: 8
  - Affordance: 15
  Total: 50 dims
```

**⚠️ MISMATCH DETECTED:** Environment produces 53+ dims, network expects 50. This is a **critical bug**.

**2. Reshape Grid for CNN (Line 161-162):**

```python
grid_2d = grid.view(batch_size, 1, window_size, window_size)
# [batch, 25] → [batch, 1, 5, 5] for Conv2d
```

**3. Encode All Components (Lines 164-168):**

```python
vision_features = vision_encoder(grid_2d)      # [batch, 128]
position_features = position_encoder(position)  # [batch, 32]
meter_features = meter_encoder(meters)          # [batch, 32]
affordance_features = affordance_encoder(affordance)  # [batch, 32]
```

**4. Concatenate Features (Lines 170-176):**

```python
combined = torch.cat([
    vision_features,      # 128
    position_features,    # 32
    meter_features,       # 32
    affordance_features   # 32
], dim=1)  # [batch, 224]
```

**5. Prepare LSTM Input (Line 178-179):**

```python
combined = combined.unsqueeze(1)  # [batch, 1, 224]
# LSTM expects [batch, seq_len, input_dim]
```

**Sequence Length = 1:** Processing single timestep (not batched sequences).

**⚠️ LSTM ISSUE:** Training with seq_len=1 means LSTM doesn't learn from sequences, only single-step predictions. This defeats the purpose of recurrence.

**6. Hidden State Management (Lines 181-191):**

```python
# Use provided hidden state or self.hidden_state
if hidden is None:
    hidden = self.hidden_state

# If still None, initialize with zeros
if hidden is None:
    h = torch.zeros(1, batch_size, hidden_dim, device=obs.device)
    c = torch.zeros(1, batch_size, hidden_dim, device=obs.device)
    hidden = (h, c)
```

**Fallback Chain:** `hidden` param → `self.hidden_state` → zeros.

**7. LSTM Forward (Lines 193-195):**

```python
lstm_out, new_hidden = self.lstm(combined, hidden)
# lstm_out: [batch, 1, 256]
# new_hidden: Tuple[(1, batch, 256), (1, batch, 256)]

lstm_out = lstm_out.squeeze(1)  # [batch, 256]
```

**8. Q-Value Prediction (Lines 197-199):**

```python
q_values = q_head(lstm_out)  # [batch, 256] → [batch, action_dim]
return q_values, new_hidden
```

**Returns:** Q-values + updated hidden state (for next step).

##### 4.3.4 Hidden State Management Methods (Lines 201-233)

**`reset_hidden_state()` (Lines 201-216):**

```python
def reset_hidden_state(batch_size: int = 1, device = None):
    """Reset LSTM hidden state (call at episode start)."""
    if device is None:
        device = torch.device('cpu')
    
    h = torch.zeros(1, batch_size, hidden_dim, device=device)
    c = torch.zeros(1, batch_size, hidden_dim, device=device)
    self.hidden_state = (h, c)
```

**When Called:** At episode start (after `env.reset()`).

**Purpose:** Clear memory from previous episode (fresh start).

**`set_hidden_state()` (Lines 218-224):**

```python
def set_hidden_state(hidden: Tuple[h, c]):
    """Set LSTM hidden state (for episode rollouts)."""
    self.hidden_state = hidden
```

**When Called:** During episode rollouts (update after each step).

**`get_hidden_state()` (Lines 226-228):**

```python
def get_hidden_state() -> Optional[Tuple[h, c]]:
    """Get current LSTM hidden state."""
    return self.hidden_state
```

**When Called:** After forward pass to retrieve updated state.

##### 4.3.5 Hidden State Flow in Training

**Episode Rollout (Forward Pass - Maintains State):**

```python
# 1. Episode start
network.reset_hidden_state(batch_size=num_agents, device=device)

# 2. Step through episode
for step in range(max_steps):
    q_values, new_hidden = network(observations, hidden=None)
    # Uses self.hidden_state internally
    
    network.set_hidden_state(new_hidden)  # Update for next step
    
    # Select actions, step environment, etc.
```

**Training (Batch Learning - Resets State):**

```python
# Sample batch from replay buffer (random transitions)
batch = replay_buffer.sample(batch_size=64)

# Train with ZERO hidden state (treat as independent transitions)
network.reset_hidden_state(batch_size=64, device=device)
q_values, _ = network(batch['observations'], hidden=None)

# Compute loss, backward, optimize...
```

**⚠️ CRITICAL ISSUE:** Training treats transitions as independent (zero hidden state), so LSTM doesn't learn temporal dependencies. This is why ACTION #9 calls for "root and branch reimagining."

---

#### 4.4 Integration with Training Loop

**Location:** `VectorizedPopulation.step_population()` (population/vectorized.py)

**Network Selection (Constructor):**

```python
if network_type == 'simple':
    q_network = SimpleQNetwork(
        obs_dim=obs_dim,
        action_dim=5,
        hidden_dim=128
    )
elif network_type == 'recurrent':
    q_network = RecurrentSpatialQNetwork(
        action_dim=5,
        window_size=5,
        num_meters=8,
        hidden_dim=256
    )
```

**Forward Pass (Every Step - Hot Path):**

```python
# Get observations from environment
observations = env.get_observations()  # [num_agents, obs_dim]

# Forward pass through network
if network_type == 'simple':
    q_values = q_network(observations)
else:  # recurrent
    q_values, new_hidden = q_network(observations)
    q_network.set_hidden_state(new_hidden)
```

**Training (Every 4 Steps - Warm Path):**

```python
# Sample batch
batch = replay_buffer.sample(batch_size=64, intrinsic_weight=intrinsic_weight)

# Forward pass (recurrent networks use zero hidden state)
if network_type == 'simple':
    q_values = q_network(batch['observations'])
    q_values_next = q_network(batch['next_observations'])
else:  # recurrent
    q_network.reset_hidden_state(batch_size=64, device=device)
    q_values, _ = q_network(batch['observations'])
    q_values_next, _ = q_network(batch['next_observations'])

# DQN loss
q_pred = q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
q_target = batch['rewards'] + gamma * q_values_next.max(dim=1)[0] * (1 - batch['dones'])
loss = F.mse_loss(q_pred, q_target)

# Backward + optimize
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
optimizer.step()
```

**Episode Reset (Once Per Episode):**

```python
# Reset environment
observations = env.reset()

# Reset network hidden state (recurrent only)
if network_type == 'recurrent':
    q_network.reset_hidden_state(batch_size=num_agents, device=device)
```

---

#### 4.5 Testing Status

**Coverage:** 98% (1 line missing in RecurrentSpatialQNetwork)

**Tested:**

- ✅ SimpleQNetwork forward pass (shape validation)
- ✅ RecurrentSpatialQNetwork forward pass (shape validation)
- ✅ Hidden state management (reset, get, set)
- ✅ LSTM continuity across steps
- ✅ Batch processing (multiple agents)
- ⚠️ LSTM memory effectiveness (ACTION #9 - NOT validated)

**Missing Line:** Likely a defensive check or edge case in hidden state management.

**Test Files:**

- `tests/test_townlet/test_networks.py` - Shape validation, hidden state tests

**Critical Gap (ACTION #9):**

Tests validate **shape correctness** but not **memory effectiveness**.

**Needed Tests:**

1. **Sequence Memory:** Does LSTM remember information from previous steps?
2. **Temporal Credit Assignment:** Can it solve delayed reward tasks?
3. **Observation Parsing:** Are dimensions correctly split? (likely BUG found)

---

#### 4.6 Known Issues

**🔴 CRITICAL ISSUES (Require Immediate Attention):**

1. **Observation Dimension Mismatch (Lines 150-159):**
   - Environment produces 53-55 dims (25 grid + 2 pos + 8 meters + 16 affordance + 2 temporal)
   - Network expects 50 dims (25 grid + 2 pos + 8 meters + 15 affordance)
   - **Impact:** Last affordance dimension or temporal features are dropped or cause crash
   - **Fix:** Audit environment observation construction vs network parsing (2-4 hours)

2. **LSTM Not Learning Temporal Dependencies (ACTION #9):**
   - Training uses `seq_len=1` (single-step transitions)
   - Hidden state reset to zeros for batch learning
   - **Impact:** LSTM degenerates to feedforward (memory unused)
   - **Fix:** Implement sequential replay buffer (ACTION #7, 1 week) OR trajectory-based training

3. **Affordance Encoder Dimension (Line 100):**
   - Expects 15 inputs, but observation has 16 (15 types + "none")
   - Docstring (Line 54) says 15, code expects 15, environment produces 16
   - **Impact:** Dimension error or silent drop of "none" indicator
   - **Fix:** Change encoder to 16 inputs OR environment to 15 outputs (30 minutes)

**🟡 MODERATE ISSUES:**

4. **No Target Network (ACTION #5):**
   - Q-target and Q-pred use same network (unstable)
   - **Fix:** Add target network with periodic sync (1-2 days)
   - **Impact:** More stable training (less overestimation)

5. **Single-Layer LSTM:**
   - `num_layers=1` may be insufficient for complex temporal patterns
   - **Fix:** Experiment with 2-layer LSTM (1 day)
   - **Impact:** Better temporal credit assignment (maybe)

6. **CNN Preserves Dimensions:**
   - `padding=1` with `kernel_size=3` keeps 5×5 → 5×5
   - **Rationale:** Small window (5×5) needs to preserve spatial information
   - **Question:** Is this optimal? Could use strided convs to compress.

**🟢 LOW PRIORITY:**

7. **Gradient Clipping Threshold:**
   - `max_norm=10.0` is arbitrary
   - **Fix:** Tune based on gradient norms (1 day)
   - **Impact:** More stable training (marginal)

---

#### 4.7 Design Strengths

1. **Modular Encoders:** Vision, position, meters, affordances are separate (easy to modify)
2. **Batch-First LSTM:** Matches PyTorch best practices (`batch_first=True`)
3. **Hidden State Management:** Clean API (reset, get, set)
4. **CNN for Vision:** Appropriate use of convolutions for spatial data
5. **Simple Baseline:** SimpleQNetwork provides clean comparison point

---

#### 4.8 Design Weaknesses

1. **Tight Coupling to Observation Format:** Hardcoded dimension splits (fragile)
2. **No Attention Mechanism:** LSTM may struggle with long-range dependencies
3. **Single-Step Training:** LSTM trained on isolated transitions (defeats purpose)
4. **No Layer Normalization:** Could improve training stability
5. **No Residual Connections:** Could help gradient flow in deeper networks

---

#### 4.9 Refactoring Opportunities (ACTION #9)

**"Root and Branch Reimagining" (October 31, 2025 Discovery):**

**Option A: Sequential Training (Correct LSTM Usage)**

```python
# Store trajectories in replay buffer (not single transitions)
trajectories = replay_buffer.sample_trajectories(batch_size=16, seq_len=32)

# Forward through full sequences
q_network.reset_hidden_state(batch_size=16)
q_values_seq = []
for t in range(32):
    q_values, new_hidden = q_network(trajectories[:, t, :])
    q_values_seq.append(q_values)
    q_network.set_hidden_state(new_hidden)

# Compute loss over sequence
q_values_seq = torch.stack(q_values_seq, dim=1)  # [batch, seq, actions]
# ... TD loss across sequence
```

**Impact:** LSTM learns temporal patterns (2-3 weeks implementation + testing).

**Option B: Transformer Architecture**

```python
# Replace LSTM with multi-head attention
class TransformerQNetwork(nn.Module):
    def __init__(...):
        self.encoder = EncoderStack(...)  # Vision, position, etc.
        self.transformer = nn.TransformerEncoder(...)
        self.q_head = nn.Linear(...)
    
    def forward(self, obs_sequence):  # [batch, seq, obs_dim]
        features = self.encoder(obs_sequence)
        attended = self.transformer(features)  # Self-attention
        q_values = self.q_head(attended)
        return q_values
```

**Impact:** Better long-range dependencies, no hidden state management (3-4 weeks).

**Option C: Hybrid CNN-RNN-Attention**

```python
# Combine CNN (vision) + GRU (temporal) + Attention (meter importance)
class HybridQNetwork(nn.Module):
    def __init__(...):
        self.vision_encoder = CNN(...)
        self.gru = nn.GRU(...)
        self.meter_attention = nn.MultiheadAttention(...)
        self.q_head = ...
```

**Impact:** Best of all worlds (4-5 weeks, research risk).

**Recommendation:** Start with Option A (fix LSTM training correctly) before exploring alternatives.

---

#### 4.10 Future Enhancements

**Potential Improvements:**

1. **Dueling Architecture:** Separate value and advantage streams (1-2 days)
2. **Distributional RL:** Predict reward distribution (C51, QR-DQN) (1 week)
3. **Rainbow Improvements:** Noisy nets, prioritized replay, n-step returns (2 weeks)
4. **Attention Mechanisms:** Meter importance weighting (3-5 days)
5. **Model-Based RL:** World model for planning (3-4 weeks, research)

**ACTION Items:**

- **ACTION #9:** Root and branch reimagining (3-4 weeks) 🔴 HIGH PRIORITY
- **ACTION #5:** Add target network (1-2 days) 🟡 MEDIUM
- **ACTION #7:** Sequential replay buffer (1 week) 🟡 MEDIUM (enables ACTION #9)
- **Fix Observation Dimensions:** Audit environment vs network (2-4 hours) 🔴 CRITICAL

---
Meters:        [8 meters] → FC(32)
Affordance:    [15 one-hot] → FC(32)
               ↓
       Concat [224] → LSTM(256) → FC(128) → FC(action_dim)

```

**Key interfaces:**

```python
forward(obs, hidden=None) -> Tuple[q_values, new_hidden]
reset_hidden_state(batch_size, device)
get_hidden_state() -> Optional[Tuple[h, c]]
set_hidden_state(hidden)
```

---

### 5. Training Orchestration System

**Location:** `src/townlet/population/`  
**Core Files:** `base.py` (74 lines), `vectorized.py` (402 lines)  
**Coverage:** 80% (base), 92% (vectorized)  
**Complexity:** 🟡 MODERATE (coordinator, not complex logic)  
**Purpose:** Main training loop coordination and Q-network management

---

#### 5.1 System Overview

The Training Orchestration System implements the **main training loop** that coordinates all other systems. It's the "brain" that connects environment, curriculum, exploration, and networks into a cohesive training process.

**Core Responsibilities:**

1. **Q-Network Management** - Create, train, and checkpoint neural networks
2. **Action Selection** - Delegate to exploration strategy with action masking
3. **Environment Stepping** - Execute actions and collect transitions
4. **Replay Buffer** - Store and sample experiences for training
5. **DQN Updates** - Train Q-network every 4 steps (batch=64)
6. **RND Training** - Train predictor network for intrinsic rewards
7. **Curriculum Integration** - Retrieve decisions and track performance
8. **Episode Reset Handling** - Annealing, hidden state management
9. **State Management** - Maintain current observations, epsilons

**Design Pattern:** Coordinator/Orchestrator - delegates to specialized systems.

**Hot Path:** `step_population()` called every environment step (~200-500x per episode).

---

#### 5.2 Abstract Interface: `PopulationManager` (base.py, Lines 1-74)

**Coverage:** 80%  
**Purpose:** Define contract for population managers

##### 5.2.1 Abstract Methods (Lines 19-74)

```python
class PopulationManager(ABC):
    """Abstract interface for population management."""
    
    @abstractmethod
    def step_population(envs: VectorizedHamletEnv) -> BatchedAgentState:
        """Execute one training step for entire population (GPU).
        
        Coordinates:
        - Action selection via exploration strategy
        - Environment stepping (vectorized)
        - Reward calculation (extrinsic + intrinsic)
        - Replay buffer updates
        - Q-network training
        
        Args:
            envs: Vectorized environment [num_agents parallel]
        
        Returns:
            BatchedAgentState with all agent data after step
        
        Note:
            Hot path - called every step. Must be GPU-optimized.
        """
        pass
    
    @abstractmethod
    def get_checkpoint() -> PopulationCheckpoint:
        """Return Pydantic checkpoint (cold path).
        
        Aggregates:
        - Agent network weights
        - Curriculum states (per agent)
        - Exploration states (per agent)
        - Pareto frontier
        - Metrics summary
        
        Returns:
            PopulationCheckpoint (Pydantic DTO)
        """
        pass
```

**Future Abstractions:**

- `reset()` - Reset all agents
- `load_checkpoint(checkpoint)` - Restore from checkpoint
- `select_actions(...)` - Action selection interface

**Note:** Base class is minimal (only 2 abstract methods). More could be extracted.

---

#### 5.3 Vectorized Implementation: `VectorizedPopulation` (vectorized.py, Lines 1-402)

**Coverage:** 92% (32/402 lines missing)  
**Purpose:** Concrete training loop for vectorized environments

##### 5.3.1 Constructor (Lines 34-110)

```python
def __init__(
    env: VectorizedHamletEnv,
    curriculum: CurriculumManager,
    exploration: ExplorationStrategy,
    agent_ids: List[str],
    device: torch.device,
    obs_dim: int = 70,
    action_dim: int = 5,
    learning_rate: float = 0.00025,
    gamma: float = 0.99,
    replay_buffer_capacity: int = 10000,
    network_type: str = "simple",
    vision_window_size: int = 5,
)
```

**Key Initialization Steps:**

**1. Store References (Lines 68-74):**

```python
self.env = env
self.curriculum = curriculum
self.exploration = exploration
self.agent_ids = agent_ids
self.num_agents = len(agent_ids)
self.device = device
self.gamma = gamma
self.network_type = network_type
self.is_recurrent = (network_type == "recurrent")
```

**2. Create Q-Network (Lines 77-85):**

```python
if network_type == "recurrent":
    self.q_network = RecurrentSpatialQNetwork(
        action_dim=action_dim,
        window_size=vision_window_size,
        num_meters=8,
    ).to(device)
else:
    self.q_network = SimpleQNetwork(obs_dim, action_dim).to(device)
```

**Auto-Detection:** Network type from config (`'simple'` or `'recurrent'`).

**⚠️ ISSUE:** `obs_dim` parameter is passed but ignored for recurrent networks (hardcoded dimensions in `RecurrentSpatialQNetwork`).

**3. Create Optimizer (Line 87):**

```python
self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
```

**No scheduler:** Learning rate is fixed (could add decay).

**4. Create Replay Buffer (Line 90):**

```python
self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, device=device)
```

**Capacity:** 10K transitions by default (circular buffer).

**5. Initialize Training Counters (Lines 93-96):**

```python
self.total_steps = 0
self.train_frequency = 4  # Train Q-network every N steps
self.episode_step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=device)
```

**Train Frequency:** Q-network trained every 4 steps (accumulate 4 transitions before update).

**6. Initialize State (Lines 99-101):**

```python
self.current_obs: torch.Tensor = None
self.current_epsilons: torch.Tensor = None
self.current_curriculum_decisions: List = []
```

**State Management:** Maintained across calls to `step_population()`.

##### 5.3.2 Reset Method (Lines 103-125)

```python
def reset() -> None:
    """Reset all environments and state."""
    self.current_obs = self.env.reset()
    
    # Reset recurrent network hidden state (if applicable)
    if self.is_recurrent:
        self.q_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)
    
    # Get epsilon from exploration strategy (handle both direct and composed)
    if isinstance(self.exploration, AdaptiveIntrinsicExploration):
        epsilon = self.exploration.rnd.epsilon
    else:
        epsilon = self.exploration.epsilon
    
    self.current_epsilons = torch.full(
        (self.num_agents,), epsilon, device=self.device
    )
```

**Called:** Once at training start (before first episode).

**Actions:**

1. Reset environment → get initial observations
2. Reset LSTM hidden state (if recurrent network)
3. Extract epsilon from exploration strategy (handles composition)

**⚠️ ISSUE:** Epsilon extraction uses `isinstance` checks (fragile - breaks with new exploration strategies).

##### 5.3.3 Inference Action Selection (Lines 127-159)

**`select_greedy_actions()` (Lines 127-159):**

```python
def select_greedy_actions(env: VectorizedHamletEnv) -> torch.Tensor:
    """Select greedy actions with action masking for inference.
    
    This is the canonical way to select actions during inference.
    Uses the same action masking logic as training to prevent boundary violations.
    """
    with torch.no_grad():
        # Get Q-values from network
        q_output = self.q_network(self.current_obs)
        # Recurrent networks return (q_values, hidden_state)
        q_values = q_output[0] if isinstance(q_output, tuple) else q_output
        
        # Get action masks from environment
        action_masks = env.get_action_masks()
        
        # Mask invalid actions with -inf before argmax
        masked_q_values = q_values.clone()
        masked_q_values[~action_masks] = float('-inf')
        
        # Select best valid action
        actions = masked_q_values.argmax(dim=1)
    
    return actions
```

**Purpose:** Pure exploitation (no exploration) for inference server.

**Key Features:**

1. `torch.no_grad()` - No gradient computation (inference only)
2. Action masking - Prevent boundary violations
3. Handles recurrent networks (tuple unpacking)

**Used By:** `demo/live_inference.py` for visualization.

**`select_epsilon_greedy_actions()` (Lines 161-207):**

```python
def select_epsilon_greedy_actions(env: VectorizedHamletEnv, epsilon: float) -> torch.Tensor:
    """Select epsilon-greedy actions with action masking.
    
    With probability epsilon, select random valid action.
    With probability (1-epsilon), select greedy action.
    """
    with torch.no_grad():
        # Get Q-values from network
        q_output = self.q_network(self.current_obs)
        q_values = q_output[0] if isinstance(q_output, tuple) else q_output
        
        # Get action masks from environment
        action_masks = env.get_action_masks()
        
        # Mask invalid actions with -inf before argmax
        masked_q_values = q_values.clone()
        masked_q_values[~action_masks] = float('-inf')
        
        # Select best valid action (greedy)
        greedy_actions = masked_q_values.argmax(dim=1)
        
        # Epsilon-greedy exploration
        num_agents = q_values.shape[0]
        actions = torch.zeros(num_agents, dtype=torch.long, device=q_values.device)
        
        for i in range(num_agents):
            if torch.rand(1).item() < epsilon:
                # Random action from valid actions
                valid_actions = torch.where(action_masks[i])[0]
                random_idx = torch.randint(0, len(valid_actions), (1,)).item()
                actions[i] = valid_actions[random_idx]
            else:
                # Greedy action
                actions[i] = greedy_actions[i]
    
    return actions
```

**Purpose:** Epsilon-greedy exploration for training (alternative to exploration strategy).

**⚠️ CODE DUPLICATION (Related to ACTION #10):** This is a THIRD copy of epsilon-greedy logic!

- Copy 1: `exploration/epsilon_greedy.py` (100 lines)
- Copy 2: `exploration/rnd.py` (100 lines) - ACTION #10
- Copy 3: `population/vectorized.py` (47 lines) - **THIS ONE**

**Why It Exists:** Provides direct epsilon-greedy without going through exploration strategy abstraction (used by inference server).

**Should Keep?** Yes, but refactor to call shared utility function.

##### 5.3.4 Main Training Loop: `step_population()` (Lines 209-378)

**Coverage:** 92% (excellent for 170-line method)  
**Complexity:** 🟡 MODERATE (12 steps, but straightforward)

**Full Method Structure:**

```python
def step_population(envs: VectorizedHamletEnv) -> BatchedAgentState:
    """Execute one training step for entire population."""
```

**Step 1: Forward Pass (Lines 219-226):**

```python
# 1. Get Q-values from network
with torch.no_grad():
    if self.is_recurrent:
        q_values, new_hidden = self.q_network(self.current_obs)
        # Update hidden state for next step (episode rollout memory)
        self.q_network.set_hidden_state(new_hidden)
    else:
        q_values = self.q_network(self.current_obs)
```

**Key:** Maintains hidden state across episode steps (memory continuity).

**Step 2: Create Temporary State (Lines 228-238):**

```python
# 2. Create temporary agent state for curriculum decision
temp_state = BatchedAgentState(
    observations=self.current_obs,
    actions=torch.zeros(self.num_agents, dtype=torch.long, device=self.device),
    rewards=torch.zeros(self.num_agents, device=self.device),
    dones=torch.zeros(self.num_agents, dtype=torch.bool, device=self.device),
    epsilons=self.current_epsilons,
    intrinsic_rewards=torch.zeros(self.num_agents, device=self.device),
    survival_times=envs.step_counts.clone(),
    curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
    device=self.device,
)
```

**Purpose:** Curriculum needs state to make decisions (but actions haven't been selected yet, so fill with zeros).

**⚠️ AWKWARD:** Creating state before actions are selected (zeros are placeholders).

**Step 3: Get Curriculum Decisions (Lines 240-252):**

```python
# 3. Get curriculum decisions (pass Q-values if curriculum supports it)
if hasattr(self.curriculum, 'get_batch_decisions_with_qvalues'):
    # AdversarialCurriculum - pass Q-values for entropy calculation
    self.current_curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(
        temp_state,
        self.agent_ids,
        q_values,
    )
else:
    # StaticCurriculum or other curricula - no Q-values needed
    self.current_curriculum_decisions = self.curriculum.get_batch_decisions(
        temp_state,
        self.agent_ids,
    )
```

**Polymorphism:** Check for method existence (`hasattr`) to support both interfaces.

**Why Q-Values?** AdversarialCurriculum needs entropy (policy certainty) for advancement.

**Step 4: Get Action Masks (Lines 254-255):**

```python
# 4. Get action masks from environment
action_masks = envs.get_action_masks()
```

**Step 5: Select Actions (Lines 257-258):**

```python
# 5. Select actions via exploration strategy (with action masking)
actions = self.exploration.select_actions(q_values, temp_state, action_masks)
```

**Delegation:** Exploration strategy handles epsilon-greedy, RND, etc.

**Step 6: Step Environment (Lines 260-261):**

```python
# 6. Step environment
next_obs, rewards, dones, info = envs.step(actions)
```

**Step 7: Compute Intrinsic Rewards (Lines 263-266):**

```python
# 7. Compute intrinsic rewards (if RND-based exploration)
intrinsic_rewards = torch.zeros_like(rewards)
if isinstance(self.exploration, (RNDExploration, AdaptiveIntrinsicExploration)):
    intrinsic_rewards = self.exploration.compute_intrinsic_rewards(self.current_obs)
```

**Novelty Bonus:** RND computes intrinsic reward based on observation novelty.

**Step 8: Store Transition (Lines 268-275):**

```python
# 7. Store transition in replay buffer
self.replay_buffer.push(
    observations=self.current_obs,
    actions=actions,
    rewards_extrinsic=rewards,
    rewards_intrinsic=intrinsic_rewards,
    next_observations=next_obs,
    dones=dones,
)
```

**Dual Rewards:** Stores both extrinsic and intrinsic separately (combined during sampling).

**Step 9: Train RND Predictor (Lines 277-284):**

```python
# 8. Train RND predictor (if applicable)
if isinstance(self.exploration, (RNDExploration, AdaptiveIntrinsicExploration)):
    rnd = self.exploration.rnd if isinstance(self.exploration, AdaptiveIntrinsicExploration) else self.exploration
    # Accumulate observations in RND buffer
    for i in range(self.num_agents):
        rnd.obs_buffer.append(self.current_obs[i].cpu())
    # Train predictor if buffer is full
    loss = rnd.update_predictor()
```

**Mini-Batch Accumulation:** RND buffers 128 observations before training.

**Step 10: Train Q-Network (Lines 286-330):**

```python
# 9. Train Q-network from replay buffer (every train_frequency steps)
self.total_steps += 1
if self.total_steps % self.train_frequency == 0 and len(self.replay_buffer) >= 64:
    intrinsic_weight = (
        self.exploration.get_intrinsic_weight()
        if isinstance(self.exploration, AdaptiveIntrinsicExploration)
        else 1.0
    )
    batch = self.replay_buffer.sample(batch_size=64, intrinsic_weight=intrinsic_weight)
    
    # Standard DQN update (simplified, no target network for now)
    if self.is_recurrent:
        # Reset hidden states for batch training (treat transitions independently)
        batch_size = batch['observations'].shape[0]
        self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
        q_values, _ = self.q_network(batch['observations'])
        q_pred = q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            self.q_network.reset_hidden_state(batch_size=batch_size, device=self.device)
            q_next_values, _ = self.q_network(batch['next_observations'])
            q_next = q_next_values.max(1)[0]
            q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()
    else:
        q_pred = self.q_network(batch['observations']).gather(1, batch['actions'].unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            q_next = self.q_network(batch['next_observations']).max(1)[0]
            q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()
    
    loss = F.mse_loss(q_pred, q_target)
    
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
    self.optimizer.step()
    
    # Reset hidden state back to episode batch size after training
    if self.is_recurrent:
        self.q_network.reset_hidden_state(batch_size=self.num_agents, device=self.device)
```

**DQN Algorithm:**

1. Sample 64 random transitions from replay buffer
2. Compute Q-prediction: `Q(s, a)`
3. Compute Q-target: `r + γ * max_a' Q(s', a')` (Bellman equation)
4. MSE loss between prediction and target
5. Backward pass + gradient clipping (max_norm=10.0)
6. Optimizer step

**⚠️ CRITICAL ISSUE (Related to ACTION #9):** Recurrent networks reset hidden state to zeros for batch training (defeats LSTM purpose).

**No Target Network (ACTION #5):** Q-target uses same network as Q-pred (unstable - causes overestimation).

**Step 11: Update Current State (Lines 332-335):**

```python
# 10. Update current state
self.current_obs = next_obs

# Track episode steps
self.episode_step_counts += 1
```

**Step 12: Handle Episode Resets (Lines 337-358):**

```python
# 11. Handle episode resets (for adaptive intrinsic annealing)
if dones.any():
    reset_indices = torch.where(dones)[0]
    for idx in reset_indices:
        # Update adaptive intrinsic annealing
        if isinstance(self.exploration, AdaptiveIntrinsicExploration):
            survival_time = self.episode_step_counts[idx].item()
            self.exploration.update_on_episode_end(survival_time=survival_time)
        # Reset episode counter
        self.episode_step_counts[idx] = 0
    
    # Reset hidden states for agents that terminated (if using recurrent network)
    if self.is_recurrent:
        # Get current hidden state
        h, c = self.q_network.get_hidden_state()
        # Zero out hidden states for terminated agents
        h[:, reset_indices, :] = 0.0
        c[:, reset_indices, :] = 0.0
        self.q_network.set_hidden_state((h, c))
```

**Key Actions:**

1. Update adaptive intrinsic annealing (track survival time)
2. Reset episode counters for terminated agents
3. Zero out LSTM hidden state for terminated agents (fresh start next episode)

**Correctness:** Handles vectorized resets (some agents terminate while others continue).

**Step 13: Construct Return State (Lines 360-378):**

```python
# 12. Construct BatchedAgentState (use combined rewards for curriculum tracking)
total_rewards = rewards + intrinsic_rewards * (
    self.exploration.get_intrinsic_weight()
    if isinstance(self.exploration, AdaptiveIntrinsicExploration)
    else 1.0
)

state = BatchedAgentState(
    observations=next_obs,
    actions=actions,
    rewards=total_rewards,
    dones=dones,
    epsilons=self.current_epsilons,
    intrinsic_rewards=intrinsic_rewards,
    survival_times=info['step_counts'],
    curriculum_difficulties=torch.zeros(self.num_agents, device=self.device),
    device=self.device,
)

return state
```

**Combined Rewards:** Returns `extrinsic + intrinsic * weight` (used by curriculum for advancement).

##### 5.3.5 Curriculum Tracker Update (Lines 380-392)

```python
def update_curriculum_tracker(rewards: torch.Tensor, dones: torch.Tensor):
    """Update curriculum performance tracking after step.
    
    Call this after step_population if using AdversarialCurriculum.
    """
    if hasattr(self.curriculum, 'tracker') and self.curriculum.tracker is not None:
        self.curriculum.tracker.update_step(rewards, dones)
```

**Purpose:** Update curriculum's PerformanceTracker (survival rate, learning progress).

**⚠️ DESIGN ISSUE:** Why separate method? Could be integrated into `step_population()`.

**Called By:** `demo/runner.py` training loop.

##### 5.3.6 Checkpointing (Lines 394-402)

```python
def get_checkpoint() -> PopulationCheckpoint:
    """Return Pydantic checkpoint."""
    return PopulationCheckpoint(
        generation=0,
        num_agents=self.num_agents,
        agent_ids=self.agent_ids,
        curriculum_states={'global': self.curriculum.checkpoint_state()},
        exploration_states={'global': self.exploration.checkpoint_state()},
        pareto_frontier=[],
        metrics_summary={},
    )
```

**⚠️ INCOMPLETE:** Does NOT save Q-network weights!

**Missing:**

- Q-network state_dict
- Optimizer state_dict
- Replay buffer contents
- Total steps counter

**Related:** ACTION #11 (Remove Legacy Checkpoint Methods) - needs unified checkpointing.

---

#### 5.4 Integration with Other Systems

**System Dependencies:**

```text
VectorizedPopulation
    ├── Environment (VectorizedHamletEnv)
    │   └── get_action_masks(), step(), reset()
    ├── Curriculum (CurriculumManager)
    │   └── get_batch_decisions_with_qvalues()
    ├── Exploration (ExplorationStrategy)
    │   ├── select_actions()
    │   ├── compute_intrinsic_rewards()
    │   └── update_on_episode_end()
    ├── Q-Network (SimpleQNetwork or RecurrentSpatialQNetwork)
    │   ├── forward()
    │   ├── reset_hidden_state()
    │   └── set_hidden_state()
    └── Replay Buffer (ReplayBuffer)
        ├── push()
        └── sample()
```

**Flow Diagram:**

```text
step_population() called
    ↓
Q-Network forward → q_values
    ↓
Curriculum decisions ← q_values (for entropy)
    ↓
Action masks ← Environment
    ↓
Actions ← Exploration strategy (q_values, masks)
    ↓
Environment.step(actions) → (obs, rewards, dones)
    ↓
Intrinsic rewards ← RND (if enabled)
    ↓
Replay buffer.push(transition)
    ↓
[Every 4 steps] Q-Network.train(batch from replay buffer)
    ↓
[On episode end] Update annealing, reset hidden states
    ↓
Return BatchedAgentState
```

---

#### 5.5 Testing Status

**Coverage:** 92% (32/402 lines missing)

**Tested:**

- ✅ Q-network initialization (simple and recurrent)
- ✅ Forward pass shape validation
- ✅ Action masking integration
- ✅ DQN update correctness (loss computation)
- ✅ Replay buffer flow (push and sample)
- ✅ Episode reset handling (partial)
- ⚠️ Hidden state management (recurrent) - Partial coverage
- ⚠️ Multi-agent coordination - Partial coverage

**Missing Coverage (32 lines):**

- Likely edge cases in hidden state management
- RND predictor training integration
- Curriculum tracker update
- Checkpoint serialization

**Test Files:**

- `tests/test_townlet/test_population.py` - Main test suite

---

#### 5.6 Known Issues

**🔴 CRITICAL ISSUES:**

1. **LSTM Training Defeats Purpose (ACTION #9):**
   - Lines 299-308: Reset hidden state to zeros for batch training
   - Treats transitions independently (no temporal context)
   - **Impact:** LSTM degenerates to feedforward network
   - **Fix:** Implement sequential replay buffer (ACTION #7) + trajectory training

2. **No Target Network (ACTION #5):**
   - Line 314: Q-target uses same network as Q-pred
   - **Impact:** Unstable training, overestimation bias
   - **Fix:** Add target network with periodic sync (1-2 days)

3. **Incomplete Checkpointing (ACTION #11):**
   - Lines 394-402: Missing Q-network weights, optimizer state
   - **Impact:** Cannot resume training properly
   - **Fix:** Save all training state (2-4 hours)

**🟡 MODERATE ISSUES:**

4. **Code Duplication (Related to ACTION #10):**
   - Lines 161-207: Third copy of epsilon-greedy logic
   - **Impact:** Maintenance burden, inconsistency risk
   - **Fix:** Extract shared utility function (1-2 hours)

5. **Fragile Epsilon Extraction (Line 117):**
   - Uses `isinstance` checks to extract epsilon from exploration
   - **Impact:** Breaks with new exploration strategies
   - **Fix:** Add `get_epsilon()` to ExplorationStrategy interface (30 min)

6. **Awkward Temporary State (Lines 228-238):**
   - Creates state with zero actions before actions are selected
   - **Impact:** Confusing code, potential bugs if curriculum uses actions
   - **Fix:** Refactor curriculum to accept observations only (2-3 hours)

7. **Separate Curriculum Tracker Update (Lines 380-392):**
   - `update_curriculum_tracker()` must be called separately
   - **Impact:** Easy to forget, inconsistent state
   - **Fix:** Integrate into `step_population()` (30 min)

**🟢 LOW PRIORITY:**

8. **Fixed Learning Rate:**
   - No learning rate scheduler
   - **Impact:** Suboptimal convergence (maybe)
   - **Fix:** Add scheduler (1 day)

9. **Hardcoded Train Frequency:**
   - `train_frequency = 4` hardcoded (line 95)
   - **Impact:** Can't tune without code change
   - **Fix:** Make configurable (15 min)

10. **No Gradient Norm Logging:**
    - Line 323: Clips gradients but doesn't log magnitude
    - **Impact:** Can't diagnose exploding gradients
    - **Fix:** Add logging (30 min)

---

#### 5.7 Design Strengths

1. **Clean Delegation:** Each system has clear responsibility (curriculum, exploration, etc.)
2. **Vectorized:** All operations batched on GPU
3. **Action Masking:** Integrated at action selection (prevents invalid actions)
4. **Dual Rewards:** Separate extrinsic and intrinsic storage/combination
5. **Hidden State Management:** Correctly resets on episode end (per agent)
6. **Gradient Clipping:** Prevents exploding gradients (max_norm=10.0)

---

#### 5.8 Design Weaknesses

1. **God Object Tendencies:** VectorizedPopulation knows about all systems (high coupling)
2. **Type Checking (`isinstance`):** Scattered throughout (fragile, violates OOP)
3. **Missing Abstractions:** Epsilon extraction, curriculum decision retrieval
4. **Incomplete Checkpointing:** Missing critical training state
5. **No Metrics Logging:** Loss, Q-values, gradients not tracked
6. **No Prioritized Replay:** Uniform sampling (could be improved)
7. **Single-Step Transitions:** LSTM trained on isolated steps (no sequences)

---

#### 5.9 Refactoring Opportunities

**High Priority:**

1. **ACTION #5: Add Target Network (1-2 days)**

```python
# Add to __init__:
self.q_network_target = copy.deepcopy(self.q_network)
self.target_update_frequency = 1000  # Update every 1000 steps

# In step_population():
if self.total_steps % self.target_update_frequency == 0:
    self.q_network_target.load_state_dict(self.q_network.state_dict())

# In DQN update:
with torch.no_grad():
    q_next = self.q_network_target(batch['next_observations']).max(1)[0]
    q_target = batch['rewards'] + self.gamma * q_next * (~batch['dones']).float()
```

**Impact:** More stable training (prevents moving target problem).

2. **ACTION #7: Sequential Replay Buffer (1 week)**

```python
# Store trajectories instead of single transitions
class TrajectoryReplayBuffer:
    def sample_trajectories(batch_size, seq_len) -> Dict:
        # Return [batch, seq_len, obs_dim] trajectories
        pass

# In step_population():
trajectories = self.replay_buffer.sample_trajectories(batch_size=16, seq_len=32)
q_values_seq = []
for t in range(32):
    q_values, new_hidden = self.q_network(trajectories[:, t, :])
    q_values_seq.append(q_values)
# Compute TD loss over sequence
```

**Impact:** LSTM can learn temporal dependencies (fixes ACTION #9).

3. **ACTION #10: Deduplicate Epsilon-Greedy (1-2 hours)**

```python
# Create shared utility:
def select_epsilon_greedy_with_masking(
    q_values: torch.Tensor,
    action_masks: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    # ... shared logic ...
    pass

# Use in VectorizedPopulation, EpsilonGreedyExploration, RNDExploration
```

4. **ACTION #11: Complete Checkpointing (2-4 hours)**

```python
def get_checkpoint() -> PopulationCheckpoint:
    return PopulationCheckpoint(
        generation=0,
        num_agents=self.num_agents,
        agent_ids=self.agent_ids,
        q_network_state=self.q_network.state_dict(),  # ADD
        optimizer_state=self.optimizer.state_dict(),  # ADD
        total_steps=self.total_steps,  # ADD
        curriculum_states={'global': self.curriculum.checkpoint_state()},
        exploration_states={'global': self.exploration.checkpoint_state()},
        pareto_frontier=[],
        metrics_summary={},
    )
```

**Medium Priority:**

5. **Extract Interfaces for Type Checking:**

```python
# Add to ExplorationStrategy:
@abstractmethod
def get_epsilon() -> float:
    pass

# Remove isinstance checks
epsilon = self.exploration.get_epsilon()  # Clean!
```

6. **Integrate Curriculum Tracker Update:**

```python
# In step_population(), after computing rewards:
if hasattr(self.curriculum, 'tracker') and self.curriculum.tracker is not None:
    self.curriculum.tracker.update_step(rewards, dones)
```

7. **Add Metrics Logging:**

```python
# In DQN update:
metrics = {
    'loss': loss.item(),
    'q_pred_mean': q_pred.mean().item(),
    'q_target_mean': q_target.mean().item(),
    'grad_norm': torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0),
}
# Log to tensorboard or wandb
```

---

#### 5.10 Future Enhancements

**Potential Improvements:**

1. **Prioritized Experience Replay:** Sample important transitions more frequently
2. **N-Step Returns:** Use multi-step TD targets (reduces bias)
3. **Double DQN:** Use online network to select action, target network to evaluate
4. **Dueling Architecture:** Separate value and advantage estimation
5. **Noisy Networks:** Add learnable noise to network weights (exploration)
6. **Distributional RL:** Predict full return distribution (C51, QR-DQN)
7. **Multi-Agent Learning:** Independent learners → parameter sharing → communication

**ACTION Items Referenced:**

- **ACTION #5:** Add target network 🔴 HIGH
- **ACTION #7:** Sequential replay buffer 🔴 HIGH (enables #9)
- **ACTION #9:** Root and branch network reimagining 🔴 HIGH
- **ACTION #10:** Deduplicate epsilon-greedy 🟡 MEDIUM
- **ACTION #11:** Complete checkpointing 🟡 MEDIUM

---

### 6. Supporting Infrastructure

**Location:** `src/townlet/training/`  
**Core Files:** `replay_buffer.py` (117 lines), `state.py` (208 lines)  
**Coverage:** 100% (replay_buffer), 94% (state)  
**Complexity:** 🟢 LOW (simple data structures)  
**Purpose:** Data storage and transfer between hot path (training) and cold path (checkpoints)

---

#### 6.1 System Overview

The Supporting Infrastructure provides **foundational data structures** for training:

1. **ReplayBuffer** - Stores and samples transitions for off-policy learning
2. **State DTOs** - Represents agent state, curriculum decisions, checkpoints

**Design Philosophy:** Separation of concerns:

- **Hot Path (GPU):** PyTorch tensors, minimal overhead, no validation (`BatchedAgentState`)
- **Cold Path (CPU):** Pydantic models, validation, serialization (`CurriculumDecision`, `PopulationCheckpoint`)

**Why Two Paths?**

- Hot path: Called every step (~500x per episode) - performance critical
- Cold path: Called per episode or checkpoint - correctness critical

---

#### 6.2 Replay Buffer (`replay_buffer.py`, Lines 1-117)

**Coverage:** 100% ✅  
**Purpose:** Store and sample transitions for DQN training

##### 6.2.1 Architecture (Lines 7-32)

```python
class ReplayBuffer:
    """Circular buffer storing transitions with separate extrinsic/intrinsic rewards.
    
    Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
    Samples: Random mini-batches with combined rewards
    """
    
    def __init__(capacity: int = 10000, device: torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        self.position = 0  # Write pointer (circular)
        self.size = 0      # Current buffer size (0 to capacity)
        
        # Storage tensors (lazy initialization on first push)
        self.observations = None
        self.actions = None
        self.rewards_extrinsic = None
        self.rewards_intrinsic = None
        self.next_observations = None
        self.dones = None
```

**Circular Buffer:** FIFO eviction when full (oldest transitions overwritten).

**Dual Rewards:** Stores extrinsic and intrinsic separately, combines during sampling.

**Lazy Initialization:** Storage tensors created on first `push()` (dimensions unknown at construction).

##### 6.2.2 Push Method (Lines 34-83)

```python
def push(
    observations: torch.Tensor,      # [batch, obs_dim]
    actions: torch.Tensor,           # [batch]
    rewards_extrinsic: torch.Tensor, # [batch]
    rewards_intrinsic: torch.Tensor, # [batch]
    next_observations: torch.Tensor, # [batch, obs_dim]
    dones: torch.Tensor,             # [batch]
) -> None:
    """Add batch of transitions to buffer. FIFO eviction when full."""
```

**Algorithm (Lines 45-83):**

```python
batch_size = observations.shape[0]
obs_dim = observations.shape[1]

# 1. Initialize storage on first push
if self.observations is None:
    self.observations = torch.zeros(capacity, obs_dim, device=device)
    self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
    self.rewards_extrinsic = torch.zeros(capacity, device=device)
    self.rewards_intrinsic = torch.zeros(capacity, device=device)
    self.next_observations = torch.zeros(capacity, obs_dim, device=device)
    self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

# 2. Move tensors to device
observations = observations.to(device)
# ... (all tensors moved)

# 3. Circular buffer insertion (FIFO)
for i in range(batch_size):
    idx = self.position % capacity  # Wrap around at capacity
    
    self.observations[idx] = observations[i]
    self.actions[idx] = actions[i]
    self.rewards_extrinsic[idx] = rewards_extrinsic[i]
    self.rewards_intrinsic[idx] = rewards_intrinsic[i]
    self.next_observations[idx] = next_observations[i]
    self.dones[idx] = dones[i]
    
    self.position += 1
    self.size = min(self.size + 1, capacity)  # Cap at capacity
```

**Circular Logic:**

- `position` increments indefinitely (0, 1, 2, ..., capacity, capacity+1, ...)
- `idx = position % capacity` wraps to 0-capacity range
- `size` caps at capacity (represents how full buffer is)

**⚠️ PERFORMANCE NOTE:** Loop over batch_size (not vectorized). Could be optimized with slicing.

##### 6.2.3 Sample Method (Lines 85-115)

```python
def sample(batch_size: int, intrinsic_weight: float) -> Dict[str, torch.Tensor]:
    """Sample random mini-batch with combined rewards.
    
    Args:
        batch_size: Number of transitions to sample
        intrinsic_weight: Weight for intrinsic rewards (0.0-1.0, anneals over time)
    
    Returns:
        Dictionary with keys: observations, actions, rewards, next_observations, dones
        'rewards' = rewards_extrinsic + rewards_intrinsic * intrinsic_weight
    """
```

**Algorithm:**

```python
if self.size < batch_size:
    raise ValueError(f"Buffer size ({self.size}) < batch_size ({batch_size})")

# 1. Random indices
if batch_size == self.size:
    indices = torch.randperm(self.size, device=device)  # Sample all (no replacement)
else:
    indices = torch.randint(0, self.size, (batch_size,), device=device)  # With replacement

# 2. Combine rewards
combined_rewards = (
    self.rewards_extrinsic[indices] +
    self.rewards_intrinsic[indices] * intrinsic_weight
)

# 3. Return batch dictionary
return {
    'observations': self.observations[indices],
    'actions': self.actions[indices],
    'rewards': combined_rewards,
    'next_observations': self.next_observations[indices],
    'dones': self.dones[indices],
}
```

**Uniform Sampling:** All transitions have equal probability (no prioritization).

**Intrinsic Weight Annealing:**

- Early training: `intrinsic_weight = 1.0` → exploration focus
- Late training: `intrinsic_weight = 0.01` → exploitation focus
- Combined reward smoothly transitions

**⚠️ NO PRIORITIZATION (Future Enhancement):** Important transitions (high TD error) not sampled more frequently.

##### 6.2.4 Length Method (Lines 117)

```python
def __len__() -> int:
    """Return current buffer size."""
    return self.size
```

**Purpose:** Check if buffer has enough data before sampling.

---

#### 6.3 State DTOs (`state.py`, Lines 1-208)

**Coverage:** 94% (3 lines missing)  
**Purpose:** Type-safe data transfer between systems

##### 6.3.1 Cold Path: `CurriculumDecision` (Lines 13-48)

```python
class CurriculumDecision(BaseModel):
    """Cold path: Curriculum decision for environment configuration.
    
    Returned by CurriculumManager to specify environment settings.
    Validated at construction, immutable, serializable.
    """
    model_config = ConfigDict(frozen=True)  # Immutable
    
    difficulty_level: float = Field(..., ge=0.0, le=1.0)
    active_meters: List[str] = Field(..., min_length=1, max_length=6)
    depletion_multiplier: float = Field(..., gt=0.0, le=10.0)
    reward_mode: str = Field(..., pattern=r'^(shaped|sparse)$')
    reason: str = Field(..., min_length=1)
```

**Validation:**

- `difficulty_level`: 0.0 (easiest) to 1.0 (hardest)
- `active_meters`: 1-6 meters (e.g., `['energy', 'hygiene']`)
- `depletion_multiplier`: 0.1 (10x slower) to 10.0 (10x faster)
- `reward_mode`: MUST be `'shaped'` or `'sparse'` (regex pattern)
- `reason`: Human-readable explanation

**⚠️ INCONSISTENCY:** System uses `'sparse'` only, but schema allows `'shaped'` (legacy from Phase 1-2).

**Usage:** Returned by `CurriculumManager.get_batch_decisions()`.

##### 6.3.2 Cold Path: `ExplorationConfig` (Lines 51-89)

```python
class ExplorationConfig(BaseModel):
    """Cold path: Configuration for exploration strategy.
    
    Defines parameters for epsilon-greedy, RND, or adaptive intrinsic exploration.
    """
    model_config = ConfigDict(frozen=True)
    
    strategy_type: str = Field(..., pattern=r'^(epsilon_greedy|rnd|adaptive_intrinsic)$')
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    intrinsic_weight: float = Field(default=0.0, ge=0.0)
    rnd_hidden_dim: int = Field(default=256, gt=0)
    rnd_learning_rate: float = Field(default=0.0001, gt=0.0)
```

**Validation:**

- `strategy_type`: MUST be `'epsilon_greedy'`, `'rnd'`, or `'adaptive_intrinsic'`
- `epsilon`: 0.0 (greedy) to 1.0 (random)
- `epsilon_decay`: 0.995 = ~1% decay per episode
- `intrinsic_weight`: 0.0+ (no upper bound, anneals to ~0.01)
- `rnd_hidden_dim`: Hidden dimension for RND networks (256 default)
- `rnd_learning_rate`: 0.0001 default (lower than Q-network's 0.00025)

**Usage:** Loaded from YAML config, used to construct exploration strategy.

##### 6.3.3 Cold Path: `PopulationCheckpoint` (Lines 92-137)

```python
class PopulationCheckpoint(BaseModel):
    """Cold path: Serializable population state for checkpointing.
    
    Contains all state needed to restore a population training run.
    """
    model_config = ConfigDict(frozen=True)
    
    generation: int = Field(..., ge=0)
    num_agents: int = Field(..., ge=1, le=1000)
    agent_ids: List[str] = Field(...)
    curriculum_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    exploration_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pareto_frontier: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
```

**Validation:**

- `generation`: 0+ (for genetic algorithms, currently always 0)
- `num_agents`: 1-1000 agents
- `agent_ids`: List of strings (e.g., `['agent_0']`)
- `curriculum_states`: Per-agent curriculum state (e.g., `{'global': {...}}`)
- `exploration_states`: Per-agent exploration state
- `pareto_frontier`: Agent IDs on Pareto frontier (multi-objective, future)
- `metrics_summary`: Aggregated stats (e.g., `{'avg_survival': 150.5}`)

**⚠️ INCOMPLETE (Related to ACTION #11):** Does NOT include:

- Q-network weights (`state_dict`)
- Optimizer state
- Replay buffer contents
- Total training steps

**Fix Required:** Add network/optimizer state to PopulationCheckpoint schema.

##### 6.3.4 Hot Path: `BatchedAgentState` (Lines 140-208)

```python
class BatchedAgentState:
    """Hot path: Vectorized agent state for GPU training loops.
    
    All data is batched tensors (batch_size = num_agents).
    Optimized for GPU operations, minimal validation overhead.
    Use slots for memory efficiency.
    """
    __slots__ = [
        'observations', 'actions', 'rewards', 'dones',
        'epsilons', 'intrinsic_rewards', 'survival_times',
        'curriculum_difficulties', 'device'
    ]
```

**Why `__slots__`?**

- Prevents dynamic attributes (fixed memory layout)
- 20-30% memory savings vs normal classes
- Faster attribute access (no dict lookup)

**Constructor (Lines 159-186):**

```python
def __init__(
    observations: torch.Tensor,      # [batch, obs_dim]
    actions: torch.Tensor,           # [batch]
    rewards: torch.Tensor,           # [batch]
    dones: torch.Tensor,             # [batch] bool
    epsilons: torch.Tensor,          # [batch]
    intrinsic_rewards: torch.Tensor, # [batch]
    survival_times: torch.Tensor,    # [batch]
    curriculum_difficulties: torch.Tensor,  # [batch]
    device: torch.device,
):
    """Construct batched agent state.
    
    All tensors must be on the same device.
    No validation in __init__ for performance (hot path).
    """
    # Direct assignment, no validation
    self.observations = observations
    self.actions = actions
    # ... (all attributes)
```

**No Validation:** Hot path - called every step, performance critical.

**Batch Size Property (Lines 188-190):**

```python
@property
def batch_size() -> int:
    """Get batch size from observations shape."""
    return self.observations.shape[0]
```

**Device Transfer (Lines 192-207):**

```python
def to(device: torch.device) -> 'BatchedAgentState':
    """Move all tensors to specified device.
    
    Returns new BatchedAgentState (tensors are immutable after .to()).
    """
    return BatchedAgentState(
        observations=self.observations.to(device),
        actions=self.actions.to(device),
        # ... (all tensors moved)
        device=device,
    )
```

**Immutable Pattern:** Returns new instance (like PyTorch tensors).

**Telemetry Export (Lines 209-220 - not shown in file read):**

```python
def detach_cpu_summary() -> Dict[str, np.ndarray]:
    """Extract summary for telemetry (cold path).
    
    Returns dict of numpy arrays (CPU). Used for logging, checkpoints.
    """
    return {
        'rewards': self.rewards.detach().cpu().numpy(),
        'survival_times': self.survival_times.detach().cpu().numpy(),
        'epsilons': self.epsilons.detach().cpu().numpy(),
        'curriculum_difficulties': self.curriculum_difficulties.detach().cpu().numpy(),
    }
```

**Conversion:** PyTorch GPU tensors → NumPy CPU arrays (for logging, plotting).

---

#### 6.4 Design Patterns

##### 6.4.1 Hot Path vs Cold Path

**Hot Path (Every Step - Performance Critical):**

- `BatchedAgentState` - PyTorch tensors, no validation, `__slots__`
- `ReplayBuffer.push()` - Direct tensor operations
- Called ~500x per episode

**Cold Path (Per Episode - Correctness Critical):**

- `CurriculumDecision`, `ExplorationConfig`, `PopulationCheckpoint` - Pydantic models
- Validation, immutability, serialization
- Called 1x per episode or checkpoint

**Trade-off:** Hot path sacrifices safety for speed, cold path prioritizes correctness.

##### 6.4.2 Separation of Concerns

**ReplayBuffer:** Knows NOTHING about:

- Q-networks (just stores tensors)
- Curriculum (just stores transitions)
- Exploration (just combines rewards)

**Single Responsibility:** Store and sample transitions.

##### 6.4.3 Lazy Initialization

**ReplayBuffer Storage:**

```python
# Constructor: DON'T allocate yet (obs_dim unknown)
self.observations = None

# First push(): NOW allocate
if self.observations is None:
    obs_dim = observations.shape[1]  # Detected from data
    self.observations = torch.zeros(capacity, obs_dim, device=device)
```

**Benefit:** Flexible observation dimensions (supports simple and recurrent networks).

---

#### 6.5 Testing Status

**Coverage:**

- ReplayBuffer: 100% ✅ (all lines covered)
- State DTOs: 94% (3 lines missing)

**Tested:**

- ✅ ReplayBuffer push and sample
- ✅ Circular buffer wraparound
- ✅ Dual reward combination
- ✅ Pydantic validation (CurriculumDecision, ExplorationConfig)
- ✅ BatchedAgentState device transfer
- ⚠️ PopulationCheckpoint serialization (partial)

**Missing Coverage (3 lines in state.py):**

- Likely edge cases in validation or telemetry export

**Test Files:**

- `tests/test_townlet/test_replay_buffer.py` - Replay buffer tests (100% coverage)
- `tests/test_townlet/test_state.py` - State DTO tests (94% coverage)

---

#### 6.6 Known Issues

**🟡 MODERATE ISSUES:**

1. **ReplayBuffer Loop (Lines 71-83):**
   - Loops over batch_size (not vectorized)
   - **Impact:** Slower push operations (minor, only called once per step)
   - **Fix:** Use tensor slicing for batch insertion (1 hour)

```python
# Current (loop):
for i in range(batch_size):
    idx = self.position % capacity
    self.observations[idx] = observations[i]
    # ...

# Better (vectorized):
start_idx = self.position % capacity
end_idx = (self.position + batch_size) % capacity

if end_idx > start_idx:
    # No wraparound
    self.observations[start_idx:end_idx] = observations
else:
    # Wraparound (split into two slices)
    self.observations[start_idx:] = observations[:capacity - start_idx]
    self.observations[:end_idx] = observations[capacity - start_idx:]
```

2. **No Prioritized Replay (Future Enhancement):**
   - Uniform sampling (all transitions equal priority)
   - **Impact:** Slower learning (important transitions undersampled)
   - **Fix:** Implement SumTree for O(log n) prioritized sampling (1 week)

3. **PopulationCheckpoint Incomplete (ACTION #11):**
   - Missing Q-network weights, optimizer state, replay buffer
   - **Impact:** Cannot resume training properly
   - **Fix:** Extend schema to include network state (2-4 hours)

4. **reward_mode Schema Inconsistency:**
   - CurriculumDecision allows `'shaped'` or `'sparse'`
   - System only uses `'sparse'` (legacy option)
   - **Impact:** Confusing to new developers
   - **Fix:** Update schema to only allow `'sparse'` OR document `'shaped'` is legacy (15 min)

**🟢 LOW PRIORITY:**

5. **No Replay Buffer Checkpointing:**
   - Buffer contents not saved (only curriculum/exploration)
   - **Impact:** Resume training starts with empty buffer (minor)
   - **Fix:** Add `get_state_dict()` and `load_state_dict()` (2 hours)

---

#### 6.7 Design Strengths

1. **Hot/Cold Path Separation:** Performance where needed, safety where needed
2. **Type Safety:** Pydantic validation catches config errors at startup
3. **Immutable DTOs:** `frozen=True` prevents accidental mutation
4. **Lazy Initialization:** Flexible observation dimensions
5. **Dual Reward Storage:** Clean separation of extrinsic and intrinsic
6. **Device-Aware:** All tensors track device (GPU/CPU)

---

#### 6.8 Design Weaknesses

1. **Loop in ReplayBuffer:** Not fully vectorized (minor performance hit)
2. **No Prioritization:** Uniform sampling only (future enhancement)
3. **Incomplete Checkpointing:** Missing network weights
4. **No Buffer Serialization:** Can't save/load replay buffer
5. **reward_mode Inconsistency:** Schema allows unused options

---

#### 6.9 Refactoring Opportunities

**High Priority:**

1. **ACTION #11: Complete Checkpointing (2-4 hours)**

```python
class PopulationCheckpoint(BaseModel):
    # ... existing fields ...
    q_network_state: Dict[str, Any] = Field(...)  # ADD
    optimizer_state: Dict[str, Any] = Field(...)  # ADD
    total_steps: int = Field(...)  # ADD
    replay_buffer_state: Optional[Dict[str, Any]] = Field(default=None)  # OPTIONAL
```

**Medium Priority:**

2. **Vectorize ReplayBuffer Push (1 hour):**

```python
def push_vectorized(...):
    # Handle wraparound with tensor slicing (see issue #1 above)
    pass
```

3. **Add Replay Buffer Checkpointing (2 hours):**

```python
def get_state_dict() -> Dict:
    return {
        'observations': self.observations.cpu(),
        'actions': self.actions.cpu(),
        # ...
        'position': self.position,
        'size': self.size,
    }

def load_state_dict(state_dict: Dict):
    self.observations = state_dict['observations'].to(self.device)
    # ...
```

**Low Priority:**

4. **Clean Up reward_mode Schema (15 min):**

```python
# Option A: Only allow sparse
reward_mode: str = Field(..., pattern=r'^sparse$')

# Option B: Document shaped is legacy
reward_mode: str = Field(
    ...,
    pattern=r'^(shaped|sparse)$',
    description="'shaped' is legacy (unused), system uses 'sparse' only"
)
```

---

#### 6.10 Future Enhancements

**Potential Improvements:**

1. **Prioritized Experience Replay (1 week):**
   - Store TD error with each transition
   - Use SumTree for O(log n) sampling
   - Sample important transitions more frequently

2. **Hindsight Experience Replay (2 weeks):**
   - Re-label failed episodes with achieved goals
   - Improves sparse reward learning

3. **Replay Buffer Compression (1 week):**
   - Compress observations with autoencoder
   - Store latent representations (smaller memory)

4. **Multi-Step Returns (3-5 days):**
   - Store n-step TD targets in buffer
   - Reduces bias (closer to Monte Carlo)

5. **Recurrent Replay Buffer (1 week - Related to ACTION #7):**
   - Store full episodes (trajectories)
   - Sample sequences for LSTM training
   - Enables proper recurrent network training

**ACTION Items Referenced:**

- **ACTION #7:** Sequential replay buffer (1 week) 🔴 HIGH
- **ACTION #11:** Complete checkpointing (2-4 hours) 🟡 MEDIUM

---

**What it does:**

- Circular buffer for experience storage (FIFO eviction)
- Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
- Samples random mini-batches with combined rewards

**Key interfaces:**

```python
push(obs, actions, r_ext, r_int, next_obs, dones)
sample(batch_size, intrinsic_weight) -> Dict[str, torch.Tensor]
```

**Combined reward formula:**

```python
combined_reward = extrinsic + intrinsic * intrinsic_weight
```

#### State DTOs (`state.py`, 47 lines, 94% coverage)

**What it does:**

- Defines data transfer objects for hot path
- Type-safe state management

**Key classes:**

```python
@dataclass
class BatchedAgentState:
    observations: torch.Tensor
    agent_ids: torch.Tensor
    curriculum_stage: torch.Tensor
    intrinsic_weight: torch.Tensor
    epsilon: torch.Tensor
    step_count: int
    episode_count: int

@dataclass
class CurriculumDecision:
    agent_id: int
    stage: int
    action: str  # 'stay', 'advance', 'retreat'
    reason: str
```

---

### 7. Demo Infrastructure (NOT TESTED)

**Location:** `src/townlet/demo/`  
**Coverage:** 0% (404 total lines)  
**Purpose:** Live visualization and multi-day training orchestration

**⚠️ Current Architecture (3 separate processes):**

**runner.py** (326 lines) - Training orchestration

- Multi-day training with checkpointing
- Loads config from YAML
- Saves checkpoints every 100 episodes
- Logs metrics to SQLite database
- Handles graceful shutdown

**live_inference.py** (573 lines) - WebSocket server

- Loads latest checkpoint during training
- Runs inference episodes at human-watchable speed (5 steps/sec)
- Broadcasts state updates to frontend via WebSocket
- Auto-checkpoint mode: watches for new checkpoints

**database.py** (192 lines) - Metrics storage

- SQLite with WAL mode (concurrent reads)
- Stores episode metrics, affordance visits, position heatmaps
- Thread-safe for single writer + multiple readers

**🎯 Future Architecture (ACTION #15):**

- Unified `run_demo.py` - Single command for training + inference + frontend
- Eliminates three-terminal juggling act

---

## Data Flow

### Training Step Data Flow

```
1. ENVIRONMENT STATE
   ├─ Agent positions: Tensor[num_agents, 2]
   ├─ Meters: Tensor[num_agents, 8]
   ├─ Time-of-day: Tensor[num_agents]
   └─ Interaction progress: Tensor[num_agents]

2. OBSERVATION CONSTRUCTION
   └─> Environment.get_observations()
       ├─ Full mode: [grid_encoding, position, meters, time, progress]
       └─ Partial mode: [5×5 window, position, meters, affordance, time, progress]

3. Q-VALUE PREDICTION
   └─> Q-Network.forward(obs, hidden)
       └─> Q-values: Tensor[num_agents, num_actions]

4. CURRICULUM DECISION
   └─> Curriculum.get_batch_decisions_with_qvalues(states, ids, q_values)
       ├─ Calculate entropy from Q-values
       ├─ Check advancement criteria
       └─> List[CurriculumDecision]

5. ACTION MASKING
   └─> Environment.get_action_masks()
       ├─ Boundary violations (grid edges)
       ├─ Unaffordable interactions (insufficient money)
       └─ Operating hours (closed affordances)
       └─> Mask: Tensor[num_agents, num_actions]

6. ACTION SELECTION
   └─> Exploration.select_action(q_values, epsilon, mask)
       ├─ Epsilon-greedy with masking
       └─> Actions: Tensor[num_agents]

7. ENVIRONMENT STEP
   └─> Environment.step(actions)
       ├─ Process movement/interaction
       ├─ Apply meter depletion
       ├─ Apply meter cascades
       ├─ Update temporal state
       ├─ Calculate rewards
       └─> (obs, rewards, dones, info)

8. INTRINSIC REWARD (if RND enabled)
   └─> RND.compute_intrinsic_reward(obs)
       ├─ Forward through target network (frozen)
       ├─ Forward through predictor network
       ├─> Intrinsic reward = MSE(target, predicted)

9. EXPERIENCE STORAGE
   └─> ReplayBuffer.push(obs, action, r_ext, r_int, next_obs, done)

10. NETWORK TRAINING (every 4 steps)
    ├─> ReplayBuffer.sample(batch_size=64, intrinsic_weight)
    ├─> DQN update (MSE loss on Q-values)
    └─> RND predictor update (MSE loss on target features)

11. STATE UPDATE
    └─> Return BatchedAgentState
        ├─ Updated observations
        ├─ Curriculum stages
        ├─ Exploration parameters (epsilon, intrinsic_weight)
        └─ Episode/step counts
```

---

## Key Interfaces

### Core RL Loop Interface

All major systems implement consistent interfaces for the training loop:

```python
# Environment
class VectorizedHamletEnv:
    def reset() -> Observations
    def step(actions) -> Tuple[Observations, Rewards, Dones, Info]
    def get_action_masks() -> ActionMasks

# Curriculum
class CurriculumBase:
    def initialize_population(num_agents)
    def get_batch_decisions_with_qvalues(states, ids, q_values) -> Decisions
    def update_performance(rewards, dones)

# Exploration
class ExplorationBase:
    def select_action(q_values, step, mask) -> Actions
    def get_epsilon(step) -> float  # For epsilon-greedy strategies
    def compute_intrinsic_reward(obs) -> Rewards  # For intrinsic motivation

# Population (orchestrator)
class VectorizedPopulation:
    def reset()
    def step_population(envs) -> BatchedAgentState
```

### Tensor Shapes Convention

All tensors follow `[num_agents, ...]` convention:

```python
# Core state tensors
positions: [num_agents, 2]           # (x, y) grid coordinates
meters: [num_agents, 8]              # 8 meter values (0.0 to 100.0)
observations: [num_agents, obs_dim]  # Varies by observability mode
q_values: [num_agents, num_actions]  # 5 actions: Up, Down, Left, Right, Interact
actions: [num_agents]                # Selected action indices
rewards: [num_agents]                # Reward values
dones: [num_agents]                  # Boolean episode termination
action_masks: [num_agents, 5]        # Valid action mask (0=invalid, 1=valid)

# Observability modes
# Full: obs_dim = grid_size² * 16 + 2 + 8 + 2 (grid + pos + meters + time/progress)
# Partial: obs_dim = 25 * 16 + 2 + 8 + 15 + 2 (5×5 window + pos + meters + affordance + time/progress)

# Recurrent networks
hidden: Tuple[Tensor[1, num_agents, 256], Tensor[1, num_agents, 256]]  # (h, c) for LSTM
```

---

## Dependencies

### External Dependencies

```
torch>=2.0.0          # Core tensor operations, neural networks
numpy                 # Numerical operations (minimal use)
pyyaml                # Configuration loading
fastapi               # WebSocket server (demo only)
websockets            # WebSocket protocol (demo only)
sqlite3 (stdlib)      # Metrics storage (demo only)
```

### Internal Module Dependencies

```
environment/ ──┐
curriculum/  ──┼─────> population/ ─────> training/ (entry point)
exploration/ ──┤              │
agent/       ──┘              │
                              ├─────> demo/ (visualization)
                              │
training/ ────────────────────┘
```

**Dependency Rules:**

- `population/` depends on ALL other systems (orchestrator)
- `environment/`, `curriculum/`, `exploration/`, `agent/` are independent
- `training/` provides infrastructure (no business logic)
- `demo/` depends on everything (top-level orchestration)

**No Circular Dependencies:**

- Each system has clear input/output contracts
- Communication via data classes (state.py)
- No system imports population/ (except demo/)

---

## System Boundaries

### What Belongs Where

#### Environment System

✅ Grid world simulation  
✅ Meter dynamics (depletion, cascades)  
✅ Affordance interactions  
✅ Temporal mechanics  
✅ Observation construction  
✅ Reward calculation  
❌ Action selection (belongs in Exploration)  
❌ Network training (belongs in Population)  
❌ Curriculum decisions (belongs in Curriculum)

#### Curriculum System

✅ Performance tracking  
✅ Stage progression logic  
✅ Advancement/retreat decisions  
❌ Reward calculation (belongs in Environment)  
❌ Q-value prediction (belongs in Network)  
❌ Action selection (belongs in Exploration)

#### Exploration System

✅ Action selection strategies  
✅ Epsilon decay schedules  
✅ Intrinsic motivation (RND)  
✅ Novelty detection  
❌ Q-value prediction (belongs in Network)  
❌ Action masking (belongs in Environment)  
❌ Curriculum decisions (belongs in Curriculum)

#### Network System

✅ Q-value prediction  
✅ Hidden state management  
✅ Forward pass logic  
❌ Training loop (belongs in Population)  
❌ Loss calculation (belongs in Population)  
❌ Optimizer step (belongs in Population)

#### Population System (Orchestrator)

✅ Training loop coordination  
✅ DQN updates  
✅ Replay buffer management  
✅ Component initialization  
✅ Episode reset handling  
✅ Checkpoint creation  
❌ Checkpoint saving (belongs in Demo)  
❌ Metrics logging (belongs in Demo)

---

## Critical Architecture Decisions

### 1. Vectorization Strategy

**Decision:** All operations on `[num_agents, ...]` tensors  
**Rationale:** GPU efficiency, no Python loops in hot path  
**Trade-off:** More complex code vs 10-100x speedup  
**Impact:** Can't use standard gym.Env interface

### 2. Action Masking

**Decision:** Prevent invalid actions via mask, not penalties  
**Rationale:** Faster learning, no wasted exploration  
**Trade-off:** More environment complexity vs cleaner learning signal  
**Impact:** Environment must compute valid actions every step

### 3. Dual Reward System

**Decision:** Separate extrinsic and intrinsic rewards  
**Rationale:** Curriculum can vary extrinsic, exploration varies intrinsic independently  
**Trade-off:** More storage in replay buffer vs flexible composition  
**Impact:** Replay buffer stores both, combines at sample time

### 4. Curriculum Per-Agent

**Decision:** Each agent has independent stage  
**Rationale:** Mixed-stage training (advanced agents help train harder scenarios)  
**Trade-off:** More complex curriculum logic vs richer training signal  
**Impact:** Batch operations must handle heterogeneous stages

### 5. No Target Network

**Decision:** Simplified DQN without target network  
**Rationale:** Pedagogical focus (students understand single network first)  
**Trade-off:** Less stable training vs simpler architecture  
**Impact:** May need target network later (ACTION #5)

### 6. LSTM for POMDP

**Decision:** Use LSTM to maintain history in partial observability  
**Rationale:** Standard approach for POMDP in DRL  
**Trade-off:** Training complexity vs theoretical correctness  
**Impact:** Hidden state management throughout training loop  
**Known Issue:** May not actually use history effectively (ACTION #9)

### 7. Meter Cascades Hardcoded

**Decision:** Meter cascade effects are hardcoded in environment  
**Rationale:** Original implementation for quick prototyping  
**Trade-off:** Inflexible vs fast to implement  
**Impact:** Difficult to experiment with different cascade strengths (ACTION #1 will fix)

### 8. Affordance Effects Hardcoded

**Decision:** 200+ lines of elif blocks for affordance effects  
**Rationale:** Original implementation for quick prototyping  
**Trade-off:** Cannot add affordances without code changes  
**Impact:** Students can't mod/experiment (ACTION #12 will fix)

---

## Performance Characteristics

### Hot Path (Every Step)

- **Frequency:** 10-100 steps/second during training
- **Must be GPU-optimized:** All tensor operations, no Python loops
- **Files:** `vectorized_env.py`, `networks.py`, parts of `vectorized.py`
- **Bottlenecks:** Observation construction, Q-network forward pass, environment step

### Warm Path (Every 4 Steps)

- **Frequency:** Training updates
- **Should be GPU-optimized:** Mini-batch operations
- **Files:** `vectorized.py` (DQN update), `rnd.py` (predictor update)
- **Bottlenecks:** Replay buffer sampling, backward pass

### Cold Path (Per Episode)

- **Frequency:** Episodic events (resets, curriculum decisions, checkpointing)
- **Can use CPU:** Validation, state management
- **Files:** `adversarial.py`, `vectorized.py` (reset logic)
- **Bottlenecks:** Not critical for overall throughput

---

## Testing Status by System

| System | Coverage | Status | Critical Gaps |
|--------|----------|--------|---------------|
| Environment | 82% | 🟡 Good | 216 lines dead code (ACTION #13) |
| Curriculum | 86% | 🟢 Excellent | Minor gaps in edge cases |
| Exploration | 100%/82% | 🟢 Excellent | RND some gaps |
| Networks | 98% | 🟢 Excellent | 1 line missing |
| Population | 92% | 🟢 Excellent | Hidden state edge cases |
| Training Infra | 97% | 🟢 Excellent | Near complete |
| Demo | 0% | 🔴 Untested | Not critical for training |

**Overall:** 64% coverage (982/1525 statements), 241 tests passing  
**Goal:** 70% before major refactoring

---

## Known Issues & Technical Debt

### Critical Issues (Blocking Progress)

1. **LSTM May Not Use History** (ACTION #9)
   - RecurrentSpatialQNetwork has LSTM but may not leverage temporal information
   - "Root and branch reimagining" needed for network architecture
   - Discovered through systematic testing October 31, 2025

2. **216 Lines of Dead Code** (ACTION #13)
   - DISABLED reward systems kept "for teaching"
   - Actually in git history, not needed in live code
   - Drags down vectorized_env.py coverage (82% → 95% if removed)
   - 30 minute fix for huge coverage boost!

3. **Hardcoded Meter Cascades** (ACTION #1)
   - Cannot experiment with different cascade strengths
   - Students can't learn by tuning cascade parameters
   - Needs configuration-driven cascade engine

### High Priority Issues

4. **Hardcoded Affordance Effects** (ACTION #12)
   - 200+ lines of elif blocks
   - Cannot add affordances without code changes
   - Needs YAML-driven affordance system

5. **Three-Terminal Demo** (ACTION #15)
   - Must run training + inference + frontend separately
   - Fiddly, error-prone, bad UX
   - Needs unified `run_demo.py`

6. **No CI/CD Pipeline** (ACTION #14)
   - No automated linting, type checking, dead code detection
   - Vulture would have caught the 216 dead lines!
   - Needs ruff, mypy, vulture, bandit, pre-commit hooks

### Medium Priority Issues

7. **Agent Oscillation Near Affordances** (ACTION #8)
   - INTERACT action masked unless on affordance
   - Agents can't "wait" → forced to move → oscillation
   - Needs WAIT action

8. **Monolithic vectorized_env.py** (ACTIONs #2, #3, #4)
   - 433 lines doing too many things
   - Needs extraction: RewardStrategy, MeterDynamics, ObservationBuilder

### Low Priority Issues

9. **No Target Network** (ACTION #5)
   - Simplified DQN for pedagogical reasons
   - May need for stability later

10. **Duplicate Epsilon Logic** (ACTION #10)
    - RND has copy-pasted epsilon-greedy code
    - Should compose with EpsilonGreedy

11. **Legacy Checkpoint Methods** (ACTION #11)
    - Old methods in adversarial.py unused
    - 10 lines to delete

---

## Refactoring Roadmap Reference

See `docs/testing/REFACTORING_ACTIONS.md` for comprehensive refactoring plan.

**15 Actions Planned:**

- 3 HIGH priority (Network redesign, Cascade config, CI/CD)
- 7 MEDIUM priority (Extractions, Affordance config, Unified demo, Dead code removal)
- 5 LOW priority (Target network, optimizations, cleanup)

**Total Time:** 13-20 weeks  
**Prerequisite:** 70% test coverage (currently 64%, need +6%)

**Quick Win:** ACTION #13 (30 min) removes dead code → instant +10-12% coverage boost!

---

## Document Status

This document represents the **baseline architecture** before major refactoring begins.

**Next Steps:**

1. Break down each major system into subsystems and classes (detailed documentation)
2. Document class responsibilities and method signatures
3. Identify refactoring boundaries
4. Plan extraction strategies

**Maintenance:**

- Update after each major refactoring action
- Keep synchronized with actual code
- Reference from AGENTS.md for continuity

---

**Last Updated:** November 1, 2025  
**Next Update:** After 70% coverage milestone or first major refactoring action

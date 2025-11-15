Title: Energy depletion has four conflicting sources of truth

Severity: high
Status: CLOSED
Ticket Type: JANK
Subsystem: environment/meter-dynamics
Affected Version/Branch: main
Resolution: Implemented Option A - bars.yaml with three action cost fields
Closed Date: 2025-11-14

Affected Files:
- `src/townlet/config/environment.py:67-69`
- `src/townlet/environment/vectorized_env.py:162-164, 1046-1080`
- `src/townlet/substrate/grid2d.py:170` (and other substrates)
- `configs/*/training.yaml`
- `configs/*/bars.yaml`

Description:
- Energy (and other meter) depletion is defined in **four different places** with conflicting semantics.
- This violates the no-defaults principle and creates non-reproducible configs.
- Meters deplete **twice per tick**: once from `base_depletion` and once from action costs.
- Operators cannot reason about actual depletion rates without reading Python code.

Four Conflicting Definitions:

1. **Grid2D substrate defaults** (`grid2d.py:170`):
   ```python
   costs={"energy": 0.005}  # NOT USED - overridden by training.yaml
   ```

2. **training.yaml config** (all levels):
   ```yaml
   energy_move_depletion: 0.005
   energy_wait_depletion: 0.005
   energy_interact_depletion: 0.005
   ```

3. **vectorized_env.py hardcoded costs** (lines 1060-1064):
   ```python
   movement_costs[self.energy_idx] = self.move_energy_cost  # from training.yaml
   movement_costs[self.hygiene_idx] = 0.003  # HARDCODED!
   movement_costs[self.satiation_idx] = 0.004  # HARDCODED!
   ```

4. **bars.yaml base_depletion**:
   ```yaml
   base_depletion: 0.01  # Applied every tick via meter_dynamics
   ```

Reproduction:

1. Open `configs/L_N1_debug/bars.yaml`:
   ```yaml
   - name: "energy"
     base_depletion: 0.01  # 1% per step
   ```

2. Open `configs/L_N1_debug/training.yaml`:
   ```yaml
   energy_move_depletion: 0.005
   ```

3. Trace `VectorizedHamletEnv.step()`:
   ```python
   # Line 949: Base depletion
   self.meters = self.meter_dynamics.deplete_meters(self.meters, depletion_multiplier)
   # Result: energy -= 0.01

   # Lines 1046-1067: Movement cost
   if movement_mask.any():
       movement_costs[self.energy_idx] = self.move_energy_cost  # 0.005
       self.meters[movement_mask] -= movement_costs.unsqueeze(0)
   # Result: energy -= 0.005

   # TOTAL DEPLETION PER MOVEMENT TICK: 0.015 (1.5%)
   ```

4. Observe:
   - Energy depletes **1.5% per movement tick** (0.01 + 0.005)
   - Operator who reads `bars.yaml` expects 1% per tick
   - Operator who reads `training.yaml` expects combined 1.5%
   - Neither is correct without reading Python source

Expected Behavior:
- **Single source of truth** for meter depletion per tick
- `bars.yaml: base_depletion` defines depletion that occurs **every tick regardless of action**
- Action costs (movement, interaction, etc.) should ONLY exist if they're action-specific
- Currently: movement costs are hardcoded but should be in base_depletion (movement happens every tick)
- No hidden costs in Python code
- Operators can reason about depletion rates from config files alone

Semantic Clarification:
- **base_depletion**: Occurs EVERY tick regardless of action (passive decay - breathing, existing)
- **base_move_depletion**: Additional cost for movement actions (MOVE, UP, DOWN, LEFT, RIGHT, etc.)
- **base_interaction_cost**: Additional cost for INTERACT action (interacting with affordances)
- Current system conflates these: movement/interaction costs are defined in training.yaml but should be in bars.yaml

**Three Fundamental Action Types**:
1. **Existence** → Always pay `base_depletion` (happens every tick)
2. **Movement** → Pay `base_depletion + base_move_depletion` (when moving)
3. **Interaction** → Pay `base_depletion + base_interaction_cost` (when interacting with affordances)
4. **Wait** → Only pay `base_depletion` (optional, do nothing)

Actual Behavior:
- Four different definitions across three subsystems
- Double depletion (base + action costs)
- Hardcoded costs in `vectorized_env.py` that aren't in any config
- Grid2D substrate costs are defined but not used
- Impossible to understand actual depletion without code archaeology

Root Cause:
- Historical accretion of depletion mechanisms
- No clear architectural decision about where costs belong
- Mixing of "passive decay" (bars.yaml) with "action costs" (training.yaml + env.py)
- Pre-JANK-02 era: no validation of meter references allowed hidden costs

Risk:
- Config changes have unpredictable effects (which depletion source matters?)
- Training dynamics change when updating code (hardcoded costs)
- Students debugging "weird" depletion rates discover architectural mess
- Transfer learning breaks when configs have different cost structures

Proposed Solution:

**Architecture Decision: Single Source of Truth**

**Option A: `bars.yaml` with three action cost fields** (RECOMMENDED)
- Semantics:
  - `base_depletion`: Passive decay every tick (breathing, existence)
  - `base_move_depletion`: Additional cost for movement actions specifically
  - `base_interaction_cost`: Additional cost for INTERACT action specifically
  - WAIT action: only pays base_depletion
  - Movement actions: pay base_depletion + base_move_depletion
  - INTERACT action: pays base_depletion + base_interaction_cost
- Benefits:
  - Agents have meaningful choice between WAIT vs MOVE vs INTERACT
  - Clear separation of passive decay vs action-specific costs
  - All costs in bars.yaml (single source of truth)
  - Models the three fundamental things an agent does: exist, move, interact
- Changes:
  - Add `base_move_depletion` and `base_interaction_cost` fields to BarConfig
  - Delete `training.yaml` energy_*_depletion fields
  - Delete `EnvironmentConfig` energy cost fields
  - Update `vectorized_env.py` to read from bars.yaml instead of training.yaml
  - Remove hardcoded hygiene/satiation costs (lines 1061-1064)

**Option B: ActionConfig costs ONLY**
- Semantics: "Meter depletes X per action type"
- Keep action costs in ActionConfig (substrate defaults + overrides)
- Set `bars.yaml: base_depletion: 0.0` for action-driven meters
- Delete `training.yaml` energy_*_depletion fields
- Delete `vectorized_env.py` hardcoded costs

**Option C: Hybrid** (NOT RECOMMENDED - maintains confusion)
- Keep both `base_depletion` and action costs
- Document that total depletion = base + action
- Requires clear operator education

**Recommendation: Option A**

Rationale:
- Simplest mental model: "Three fundamental actions with clear costs"
- Models exactly what agents do: exist (base), move (base + movement), interact (base + interaction)
- WAIT is optional/strategic (only pays existence cost)
- Aligns with current HAMLET pedagogy (resource management over action optimization)
- Works with all substrate types (Grid2D, Continuous, Aspatial)
- No hidden costs in Python
- Single source of truth in bars.yaml

Implementation Plan (Option A - with three action cost fields):

1. **Add to `BarConfig`:** (`src/townlet/config/bar.py`)
   ```python
   base_move_depletion: float = Field(
       ge=0.0,
       description="Additional depletion per movement action (on top of base_depletion). "
                   "Total movement cost = base_depletion + base_move_depletion. "
                   "WAIT action only pays base_depletion.",
   )

   base_interaction_cost: float = Field(
       ge=0.0,
       description="Additional depletion per INTERACT action (on top of base_depletion). "
                   "Total interaction cost = base_depletion + base_interaction_cost. "
                   "Models the three fundamental actions: exist, move, interact.",
   )
   ```

2. **Remove from all `training.yaml` files:**
   ```yaml
   # DELETE:
   energy_move_depletion: 0.005
   energy_wait_depletion: 0.005
   energy_interact_depletion: 0.005
   ```

3. **Remove from `EnvironmentConfig`:** (`src/townlet/config/environment.py`)
   ```python
   # DELETE lines 67-69:
   energy_move_depletion: float = Field(...)
   energy_wait_depletion: float = Field(...)
   energy_interact_depletion: float = Field(...)
   ```

4. **Update `VectorizedHamletEnv`:** (`src/townlet/environment/vectorized_env.py`)
   ```python
   # DELETE lines 162-164:
   self.move_energy_cost = env_cfg.energy_move_depletion
   self.wait_energy_cost = env_cfg.energy_wait_depletion
   self.interact_energy_cost = env_cfg.energy_interact_depletion

   # REPLACE lines 1046-1080 with:
   # Movement costs
   if movement_mask.any():
       # Read movement costs from bars.yaml per meter
       movement_costs = torch.zeros(self.meter_count, device=self.device)
       for i, bar in enumerate(self.universe.bars):
           if hasattr(bar, 'base_move_depletion'):
               movement_costs[i] = bar.base_move_depletion

       self.meters[movement_mask] -= movement_costs.unsqueeze(0)
       self.meters = torch.clamp(self.meters, 0.0, 1.0)

   # Interaction costs
   if interaction_mask.any():
       # Read interaction costs from bars.yaml per meter
       interaction_costs = torch.zeros(self.meter_count, device=self.device)
       for i, bar in enumerate(self.universe.bars):
           if hasattr(bar, 'base_interaction_cost'):
               interaction_costs[i] = bar.base_interaction_cost

       self.meters[interaction_mask] -= interaction_costs.unsqueeze(0)
       self.meters = torch.clamp(self.meters, 0.0, 1.0)

   # DELETE WAIT-specific costs (WAIT only pays base_depletion, no extra cost)
   # DELETE all hardcoded hygiene/satiation costs
   ```

5. **Update all `bars.yaml` files:**
   ```yaml
   # BEFORE:
   - name: "energy"
     base_depletion: 0.01

   # AFTER (three fundamental action types):
   - name: "energy"
     base_depletion: 0.01           # Passive decay every tick (existence)
     base_move_depletion: 0.005     # Additional cost for movement
     base_interaction_cost: 0.005   # Additional cost for INTERACT
     description: "Energy: 1% passive per tick, +0.5% per movement, +0.5% per interaction"

   # For meters without action costs:
   - name: "money"
     base_depletion: 0.0
     base_move_depletion: 0.0         # Money doesn't deplete from movement
     base_interaction_cost: 0.0       # Money doesn't deplete from interaction
   ```

6. **Remove Grid2D substrate action costs** (already done in JANK-02 follow-up):
   ```python
   # grid2d.py:170 - already fixed to energy-only
   costs={"energy": 0.005}  # Remove entirely - now in bars.yaml
   ```

7. **Update documentation:**
   - `docs/config-schemas/bars.md`: Document base_depletion, base_move_depletion, and base_interaction_cost
   - `docs/config-schemas/training.md`: Remove energy_*_depletion fields

Migration Impact:

**Breaking Change**: YES (pre-release with zero users = free to break)

**Config Migration Required**:
- All `training.yaml` files: Remove energy_*_depletion fields
- All `bars.yaml` files: Add base_move_depletion and base_interaction_cost fields per meter
- Code: Update VectorizedHamletEnv to read from bars instead of training.yaml

**Behavioral Change**:
- Energy depletion rate stays THE SAME:
  - WAIT: 0.01 (base_depletion only)
  - MOVE: 0.015 (base_depletion 0.01 + base_move_depletion 0.005)
  - INTERACT: 0.015 (base_depletion 0.01 + base_interaction_cost 0.005)
- Semantics are now explicit: existence vs movement vs interaction
- Students see clearer cost model and meaningful WAIT vs MOVE vs INTERACT choice
- Models the three fundamental actions: exist, move, interact

**Test Updates Required**:
- Update tests that parse `training.yaml` expecting energy cost fields
- Update tests that verify action costs are applied
- Add test verifying NO action costs are applied (only base_depletion)

Tests:
- Unit: Remove energy_*_depletion from training.yaml → config loads successfully
- Unit: Verify movement does NOT apply extra energy cost (only base_depletion)
- Integration: Energy depletion rate matches bars.yaml base_depletion exactly
- Integration: All curriculum levels maintain same depletion semantics

Owner: environment

---

## ADDITIONAL NOTES

**Why JANK not BUG:**
- Requires architectural decision (which option?)
- Affects multiple subsystems
- Breaking change that needs design consensus
- Not a "fix" but a "simplification"

**Related Tickets:**
- JANK-02: Exposed this issue by making meter validation strict
- Grid2D substrate costs removed hygiene/satiation (follow-up to JANK-02)

**User Discovery Context:**
User asked: "Does Grid2D get energy costs from config or hardcode them?"
Investigation revealed four-way conflict and double depletion.

**Pedagogical Impact:**
Current confusion teaches students that RL environments are messy and hard to reason about.
Clean architecture teaches that good design makes systems transparent.

**Future Architectural Direction:**
- Non-move and non-interact commands (like WAIT) should eventually be defined in config
- Example: WAIT as a configurable noop action with its own cost structure
- This ticket focuses on the three fundamental "intrinsics": exist, move, interact
- WAIT configurability is a separate future task

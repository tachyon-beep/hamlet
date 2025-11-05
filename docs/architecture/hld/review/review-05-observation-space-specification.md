---
document_type: Component Spec / Interface Spec
status: Draft
version: 2.5
---

## SECTION 5: OBSERVATION SPACE SPECIFICATION

### 5.1 Design Principle Recap: Human Observer Test

Every element of the observation tensor must pass the human observer test:

**At inference time**:

- ✅ Can a human see/know this? → Include in observation
- ❌ Would this require telepathy/omniscience? → Exclude from observation

**At training time** (CTDE exception):

- Module C: Social Model can receive ground truth labels for supervised learning
- But at inference, Module C: Social Model only sees public cues

**Curriculum progression**:

- Level 0-3 (implemented): Pedagogical scaffolding (full observability, perfect map)
- Level 4-5 (planned): Realistic vision (partial observability, memory required)
- Level 6-7 (planned): Realistic social perception (cues only, no telepathy)
- Level 8 (planned): Realistic communication (signals without semantics)

---

### 5.2 Observation Space by Curriculum Level

**Note**: "L0-3" is shorthand for "Curriculum Level 0-3" in tables throughout this document.

| Level | Spatial | Meters | Temporal | Social | Comm | Observation Dim |
|-------|---------|--------|----------|--------|------|-----------------|
| **L0-3** | One-hot grid | 8 bars | - | - | - | grid² + 8 + (types+1) |
| **L4-5** | 5×5 window + pos | 8 bars | sin/cos time + progress | - | - | 25 + 2 + 8 + (types+1) + 3 |
| **L6** | 5×5 window + pos | 8 bars | sin/cos time + progress | Positions | - | 25 + 2 + 8 + (types+1) + 3 + 2M |
| **L7** | 5×5 window + pos | 8 bars | sin/cos time + progress | Pos + cues | - | 25 + 2 + 8 + (types+1) + 3 + (2+K)M |
| **L8** | 5×5 window + pos | 8 bars | sin/cos time + progress | Pos + cues | Family channel | 25 + 2 + 8 + (types+1) + 3 + (2+K)M + F |

Where:

- `grid`: grid_size (e.g., 5 for L0-3, 8 for L4+)
- `types`: num_affordance_types (typically 15)
- `M`: max_visible_agents (e.g., 5)
- `K`: num_cue_types (e.g., 12 for L7)
- `F`: max_family_size (typically 2-3)

---

### 5.3 Full Observability (Curriculum Level 0-3)

**Used in**: Curriculum Level 0-3 (small 5×5 grid, tutorial mode, implemented)

**Tensor structure**:

```python
observation = torch.cat([
    grid_encoding,         # [num_agents, grid_size²]  - one-hot position
    meters,               # [num_agents, 8]            - all bars
    affordance_encoding,  # [num_agents, types + 1]    - current affordance
], dim=1)

# Total dimension: grid_size² + 8 + (num_affordance_types + 1)
# Example (5×5 grid, 15 affordances): 25 + 8 + 16 = 49
```

#### Component Breakdown

**Grid Encoding** `[num_agents, grid_size²]`

```python
# One-hot encoding of agent position in flattened grid
# Example: agent at position (2, 3) in 5×5 grid
flat_index = y * grid_size + x = 3 * 5 + 2 = 17
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 1 at index 17
#                           ↑ position 17
```

**Purpose**: Agent knows exactly where it is on the grid
**Human equivalent**: "I'm at coordinates (2, 3)"

**Meters** `[num_agents, 8]`

```python
meters = [
    energy,      # index 0, range [0, 1]
    satiation,   # index 1
    mood,        # index 2
    hygiene,     # index 3
    social,      # index 4
    fitness,     # index 5
    health,      # index 6
    money,       # index 7 (normalized: 0.50 = $50)
]
```

**Purpose**: Agent has full introspection of internal state
**Human equivalent**: "I know how tired, hungry, happy I am"

**Affordance Encoding** `[num_agents, num_affordance_types + 1]`

```python
# One-hot encoding of which affordance (if any) agent is standing on
# Example: agent at Job location
affordance_encoding = [
    0,  # Bed
    0,  # Fridge
    1,  # Job ← agent is here
    0,  # Hospital
    ...,
    0,  # (num_affordance_types entries)
    0   # "none" (not on any affordance)
]

# Example: agent not on any affordance
affordance_encoding = [0, 0, 0, ..., 0, 1]  # last element = 1
```

**Purpose**: Agent knows what affordance it's standing on
**Human equivalent**: "I'm currently at the Job building"

**Scaffolding note**: In Curriculum Level 0-3, agents also receive perfect knowledge of affordance locations (via `affordances` dict), but this is NOT in the observation tensor itself. This is why SimpleQNetwork suffices — the agent can directly see where everything is without needing memory.

---

### 5.4 Partial Observability (Curriculum Level 4-5)

**Used in**: Curriculum Level 4-5 (8×8 grid, fog of war, planned)

**Tensor structure**:

```python
observation = torch.cat([
    local_grid,           # [num_agents, window²]      - 5×5 local view
    normalized_position,  # [num_agents, 2]            - where am I?
    meters,              # [num_agents, 8]            - all bars
    affordance_encoding, # [num_agents, types + 1]    - current affordance
    time_sin,            # [num_agents, 1]            - sin(2π * hour/24)
    time_cos,            # [num_agents, 1]            - cos(2π * hour/24)
    progress,            # [num_agents, 1]            - interaction ticks / 10
], dim=1)

# Total dimension: 25 + 2 + 8 + 16 + 1 + 1 + 1 = 54 (for 15 affordance types)
```

#### Component Breakdown

**Local Grid** `[num_agents, window_size²]`

```python
# 5×5 window centered on agent (vision_range = 2)
window_size = 2 * vision_range + 1 = 5

# Grid is binary: 1 = affordance visible, 0 = empty or out of bounds
# Example: agent at (4, 4) in 8×8 world
#
#   World coordinates:        Local window:
#   (2,2) (3,2) (4,2) (5,2) (6,2)    [0 0 1 0 0]  ← row 0 (world y=2)
#   (2,3) (3,3) (4,3) (5,3) (6,3)    [0 0 0 0 0]  ← row 1
#   (2,4) (3,4) (4,4) (5,4) (6,4)    [0 0 * 0 1]  ← row 2 (* = agent position)
#   (2,5) (3,5) (4,5) (5,5) (6,5)    [0 1 0 0 0]  ← row 3
#   (2,6) (3,6) (4,6) (5,6) (6,6)    [0 0 0 0 0]  ← row 4
#
# Flattened: [0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,1,0,0,0, 0,0,0,0,0]
```

**Construction algorithm**:

```python
for dy in range(-vision_range, vision_range + 1):  # -2 to +2
    for dx in range(-vision_range, vision_range + 1):
        world_x = agent_pos[0] + dx
        world_y = agent_pos[1] + dy

        # Check bounds
        if 0 <= world_x < grid_size and 0 <= world_y < grid_size:
            # Check if affordance exists at (world_x, world_y)
            has_affordance = any(
                aff_pos == [world_x, world_y]
                for aff_pos in affordances.values()
            )

            if has_affordance:
                local_idx = (dy + vision_range) * window_size + (dx + vision_range)
                local_grid[local_idx] = 1.0
```

**Purpose**: Agent sees only immediate vicinity, must explore to discover world
**Human equivalent**: "I can see 2 tiles in each direction, rest is unknown"

**Normalized Position** `[num_agents, 2]`

```python
# Absolute position scaled to [0, 1]
normalized_position = [
    x / (grid_size - 1),  # e.g., 4 / 7 = 0.571 in 8×8 grid
    y / (grid_size - 1)   # e.g., 4 / 7 = 0.571
]
```

**Purpose**: Agent knows where it is in absolute coordinates (has a compass)
**Human equivalent**: "I'm at coordinates (4, 4) in this 8×8 town"
**Why needed**: With partial observability, agent can't infer absolute position from local grid alone

**Temporal Features** `[num_agents, 3]`

```python
# Time of day (cyclical encoding)
angle = (time_of_day / 24.0) * 2 * π
time_sin = sin(angle)  # e.g., sin(2π * 14/24) for 2pm
time_cos = cos(angle)  # e.g., cos(2π * 14/24)

# Interaction progress (if doing multi-tick action)
progress = ticks_completed / 10.0  # e.g., 3/10 = 0.3 if 3 ticks into Job
```

**Why sin/cos encoding**:

```python
# Bad: linear time
time = 23  # 11pm
# Network can't know that 23 is close to 0 (midnight)

# Good: cyclical encoding
sin(2π * 23/24) ≈ -0.26
cos(2π * 23/24) ≈  0.97
sin(2π * 0/24)  =  0.00
cos(2π * 0/24)  =  1.00
# These are close in embedding space!
```

**Purpose**: Agent can plan around operating hours, understand progress through multi-tick actions
**Human equivalent**: "It's 2pm, and I'm 30% done with this 8-hour work shift"

**Architecture requirement**: LSTM or GRU to maintain spatial memory across ticks

---

### 5.5 Social Observability (Curriculum Level 6-7) — PLANNED

**Used in**: Curriculum Level 6-7 (multi-agent, competition/cooperation, planned)

**Tensor structure**:

```python
observation = torch.cat([
    # All L4-5 features
    local_grid,           # [num_agents, 25]
    normalized_position,  # [num_agents, 2]
    meters,              # [num_agents, 8]
    affordance_encoding, # [num_agents, 16]
    time_sin,            # [num_agents, 1]
    time_cos,            # [num_agents, 1]
    progress,            # [num_agents, 1]

    # NEW: Social features
    other_agent_positions,  # [num_agents, max_visible * 2]
    other_agent_cues,       # [num_agents, max_visible * num_cue_types]
], dim=1)

# Total dimension: 54 + (max_visible * 2) + (max_visible * num_cue_types)
# Example (max_visible=5, num_cue_types=12): 54 + 10 + 60 = 124
```

#### New Components for Curriculum Level 6-7

**Other Agent Positions** `[num_agents, max_visible_agents * 2]`

```python
# Positions of other agents within vision_range, relative to observer
# Padded with zeros if fewer than max_visible agents seen

# Example: agent at (4, 4) sees two others
other_positions = [
    # Agent 1 (relative position)
    1.0,   # dx = +1 (agent is at 5, 4)
    -2.0,  # dy = -2 (agent is at 4, 2)

    # Agent 2 (relative position)
    -1.0,  # dx = -1 (agent is at 3, 4)
    0.0,   # dy =  0 (agent is at 3, 4)

    # Padding (no more agents visible)
    0.0, 0.0,  # agent 3 (not present)
    0.0, 0.0,  # agent 4 (not present)
    0.0, 0.0,  # agent 5 (not present)
]
```

**Construction**:

```python
def get_visible_agents(observer_pos, all_agents, vision_range, max_visible):
    """Find other agents within vision_range of observer."""
    visible = []

    for agent in all_agents:
        if agent.id == observer.id:
            continue  # Don't observe self

        distance = manhattan_distance(observer_pos, agent.position)
        if distance <= vision_range:
            relative_pos = agent.position - observer_pos
            visible.append(relative_pos)

    # Sort by distance (closest first)
    visible.sort(key=lambda pos: abs(pos[0]) + abs(pos[1]))

    # Take top max_visible, pad if needed
    visible = visible[:max_visible]
    while len(visible) < max_visible:
        visible.append([0.0, 0.0])

    return torch.tensor(visible).flatten()
```

**Purpose**: Agent knows where other agents are (spatial awareness)
**Human equivalent**: "I see two people nearby, one to my right, one ahead of me"

**Other Agent Cues** `[num_agents, max_visible_agents * num_cue_types]`

```python
# Binary matrix: which cues each visible agent is emitting
# Each agent emits up to 3 cues (from cues.yaml, by priority)

# Example: two visible agents, 12 cue types
other_cues = [
    # Agent 1 cues (emits: looks_tired, at_job)
    1,  # looks_tired
    0,  # looks_energetic
    0,  # looks_sick
    0,  # looks_healthy
    0,  # looks_sad
    0,  # looks_happy
    0,  # looks_poor
    0,  # looks_wealthy
    0,  # looks_dirty
    1,  # at_job
    0,  # at_hospital
    0,  # at_bar

    # Agent 2 cues (emits: looks_sick, looks_poor, at_hospital)
    0,  # looks_tired
    0,  # looks_energetic
    1,  # looks_sick
    0,  # looks_healthy
    0,  # looks_sad
    0,  # looks_happy
    1,  # looks_poor
    0,  # looks_wealthy
    0,  # looks_dirty
    0,  # at_job
    1,  # at_hospital
    0,  # at_bar

    # Agents 3-5 (padding, all zeros)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 3
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 4
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # agent 5
]
```

**Cue ordering** (must be deterministic):

```python
CUE_INDICES = {
    'looks_tired': 0,
    'looks_energetic': 1,
    'looks_sick': 2,
    'looks_healthy': 3,
    'looks_sad': 4,
    'looks_happy': 5,
    'looks_poor': 6,
    'looks_wealthy': 7,
    'looks_dirty': 8,
    'at_job': 9,
    'at_hospital': 10,
    'at_bar': 11,
}
```

**Purpose**: Agent sees body language / emotional state of others
**Human equivalent**: "That person looks tired and they're at the Job. That other person looks sick and poor and they're at the Hospital."

**Curriculum Level 6 vs Level 7 difference**:

```yaml
# Curriculum Level 6: Minimal cues (just location)
cues:
  - at_job
  - at_hospital
  - at_bar
# → num_cue_types = 3

# Curriculum Level 7: Rich cues (physical + emotional + socioeconomic + location)
cues:
  - looks_tired
  - looks_energetic
  - looks_sick
  - looks_healthy
  - looks_sad
  - looks_happy
  - looks_poor
  - looks_wealthy
  - looks_dirty
  - at_job
  - at_hospital
  - at_bar
# → num_cue_types = 12
```

**Implementation requirements**:

```python
# Add to ObservationBuilder.__init__
self.max_visible_agents = config.get('max_visible_agents', 5)
self.num_cue_types = len(config.cues.cues)

# Add to build_observations (for Curriculum Level 6+)
if self.curriculum_level >= 6:
    # Get visible agents
    visible_positions = self._get_visible_agent_positions(positions)
    visible_cues = self._get_visible_agent_cues(positions)

    obs = torch.cat([obs, visible_positions, visible_cues], dim=1)
```

**Cross-reference**: See Section 6 (Cue Engine) for details on cue generation, priority handling, and integration with the observation builder.

---

### 5.6 Communication Channel (Curriculum Level 8) — PLANNED

**Used in**: Curriculum Level 8 (family communication, planned)

**Tensor structure**:

```python
observation = torch.cat([
    # All L6-7 features
    local_grid,              # [num_agents, 25]
    normalized_position,     # [num_agents, 2]
    meters,                  # [num_agents, 8]
    affordance_encoding,     # [num_agents, 16]
    time_sin,                # [num_agents, 1]
    time_cos,                # [num_agents, 1]
    progress,                # [num_agents, 1]
    other_agent_positions,   # [num_agents, max_visible * 2]
    other_agent_cues,        # [num_agents, max_visible * num_cue_types]

    # NEW: Family communication
    family_comm_channel,     # [num_agents, max_family_size]
    family_member_ids,       # [num_agents, max_family_size]  (optional)
], dim=1)

# Total dimension: 124 + max_family_size + max_family_size
# Example (max_family_size=3): 124 + 3 + 3 = 130
```

#### New Components for Curriculum Level 8

**Family Comm Channel** `[num_agents, max_family_size]`

```python
# Integer signals from family members, normalized to [0, 1]
# Each family member can broadcast one int64 value per tick

# Example: agent in 3-person family (self + 2 others)
family_comm_channel = [
    0.123,  # signal from family member 1 (encoded as 123 / 1000)
    0.456,  # signal from family member 2 (encoded as 456 / 1000)
    0.000,  # unused slot (family size < max_family_size)
]

# If agent is NOT in a family:
family_comm_channel = [0.0, 0.0, 0.0]  # all zeros
```

**Signal encoding**:

```python
# Agents can set their signal via action
action_space.add('SET_COMM_CHANNEL', param_range=[0, 999])

# On execution:
agent.current_signal = action_param  # int in [0, 999]

# In observation builder:
for member_id in agent.family_members:
    member = agents[member_id]
    normalized_signal = member.current_signal / 1000.0
    family_comm_channel.append(normalized_signal)
```

**Purpose**: Family members can coordinate via abstract signals
**Human equivalent**: "My spouse is signaling '123' to me" (meaning negotiated through learning)

**Key constraint: NO SEMANTIC BOOTSTRAPPING**

```python
# ❌ BAD: Providing signal meanings
signal_meanings = {
    0: "all_clear",
    1: "job_taken",
    2: "need_help",
}
# This defeats the research purpose!

# ✅ GOOD: Signals start meaningless
# Agents must learn correlations:
# - Parent sends 123 when at Job
# - Child observes parent at Job correlates with signal 123
# - Child learns: "signal 123 probably means Job is occupied"
# - Child's Module C: Social Model is trained via CTDE to predict this
```

**Family Member IDs** `[num_agents, max_family_size]` (optional)

```python
# Agent IDs of family members (for tracking who sent which signal)
# Normalized by population size

# Example: agent in family with members [42, 73]
family_member_ids = [
    0.42,  # member 1 is agent_42 (42 / 100 in population of 100)
    0.73,  # member 2 is agent_73
    0.00,  # unused slot
]
```

**Why this might be useful**: Agent can learn "signal from member 42 means X, signal from member 73 means Y" (personalized protocols).

**Why this might not be needed**: If all family members learn the same protocol, identity doesn't matter.

**Recommended**: Start without `family_member_ids`, add if experiments show heterogeneous protocols within families.

---

### 5.7 Complete Dimension Calculations

**Formula by curriculum level**:

```python
# L0-3: Full observability
dim_L0_L3 = (
    grid_size ** 2 +           # one-hot position
    8 +                         # meters
    (num_affordance_types + 1)  # current affordance (+ "none")
)
# Example (5×5 grid, 15 affordances): 25 + 8 + 16 = 49

# L4-5: Partial observability + temporal
window_size = 2 * vision_range + 1  # e.g., 2*2+1 = 5
dim_L4_L5 = (
    window_size ** 2 +         # local grid (5×5 = 25)
    2 +                         # normalized position
    8 +                         # meters
    (num_affordance_types + 1) +  # current affordance
    3                           # time_sin, time_cos, progress
)
# Example: 25 + 2 + 8 + 16 + 3 = 54

# Curriculum Level 6: Social (sparse cues)
dim_L6 = dim_L4_L5 + (
    max_visible_agents * 2 +              # positions
    max_visible_agents * num_cue_types_L6  # cues (3 types in Level 6)
)
# Example (max_visible=5, 3 cues): 54 + 10 + 15 = 79

# Curriculum Level 7: Social (rich cues)
dim_L7 = dim_L4_L5 + (
    max_visible_agents * 2 +              # positions
    max_visible_agents * num_cue_types_L7  # cues (12 types in Level 7)
)
# Example (max_visible=5, 12 cues): 54 + 10 + 60 = 124

# Curriculum Level 8: Family communication
dim_L8 = dim_L7 + (
    max_family_size              # comm channel
    # + max_family_size          # member IDs (optional)
)
# Example (family_size=3): 124 + 3 = 127
# Or with IDs: 124 + 3 + 3 = 130
```

**Configuration**:

```yaml
# config.yaml
observation:
  grid_size: 8
  vision_range: 2  # 5×5 window
  num_affordance_types: 15

  # Social features (L6+)
  max_visible_agents: 5
  num_cue_types: 12  # full cue set for L7

  # Family features (L8)
  max_family_size: 3  # typical: 2 parents + 1 child
  include_family_ids: false  # start without
```

---

### 5.8 Implementation Notes

#### Current State (as of code review)

**What's implemented**:

```python
# observation_builder.py contains:
- Full observability (Curriculum Level 0-3) ✓
- Partial observability (Curriculum Level 4-5) ✓
- Temporal mechanics (time_of_day, interaction_progress) ✓
```

**What needs to be added**:

```python
# For Curriculum Level 6-7:
- _get_visible_agent_positions()
- _get_visible_agent_cues()
- Integration with cue_engine (see Section 6)

# For Curriculum Level 8:
- family_comm_channel handling
- SET_COMM_CHANNEL action
- Family membership tracking
```

#### Implementation Guide: Adding Social Observability

**Step 1: Add to ObservationBuilder.**init****

```python
def __init__(self, ..., curriculum_level=0):
    # ... existing fields ...
    self.curriculum_level = curriculum_level

    if curriculum_level >= 6:
        self.max_visible_agents = config.get('max_visible_agents', 5)
        self.cue_engine = CueEngine(config.cues)
```

**Step 2: Add helper methods**

```python
def _get_visible_agent_positions(
    self,
    observer_positions: torch.Tensor,
    all_agent_positions: torch.Tensor,
) -> torch.Tensor:
    """Get relative positions of visible agents.

    Returns:
        [num_agents, max_visible_agents * 2]
    """
    batch_size = observer_positions.shape[0]
    output = torch.zeros(
        batch_size,
        self.max_visible_agents * 2,
        device=self.device
    )

    for agent_idx in range(batch_size):
        observer_pos = observer_positions[agent_idx]
        visible = []

        # Find agents within vision_range
        for other_idx in range(batch_size):
            if other_idx == agent_idx:
                continue

            other_pos = all_agent_positions[other_idx]
            distance = torch.abs(observer_pos - other_pos).sum()

            if distance <= self.vision_range:
                relative_pos = other_pos - observer_pos
                visible.append((distance, relative_pos))

        # Sort by distance, take top max_visible
        visible.sort(key=lambda x: x[0])
        visible = visible[:self.max_visible_agents]

        # Fill output tensor
        for i, (_, rel_pos) in enumerate(visible):
            output[agent_idx, i*2:(i+1)*2] = rel_pos

    return output

def _get_visible_agent_cues(
    self,
    observer_positions: torch.Tensor,
    all_agent_positions: torch.Tensor,
    all_agent_cues: list[list[str]],  # from cue_engine
) -> torch.Tensor:
    """Get cues of visible agents.

    Returns:
        [num_agents, max_visible_agents * num_cue_types]
    """
    batch_size = observer_positions.shape[0]
    output = torch.zeros(
        batch_size,
        self.max_visible_agents * self.num_cue_types,
        device=self.device
    )

    for agent_idx in range(batch_size):
        observer_pos = observer_positions[agent_idx]
        visible = []

        # Find agents within vision_range (same as positions)
        for other_idx in range(batch_size):
            if other_idx == agent_idx:
                continue

            other_pos = all_agent_positions[other_idx]
            distance = torch.abs(observer_pos - other_pos).sum()

            if distance <= self.vision_range:
                cues = all_agent_cues[other_idx]  # list of cue strings
                visible.append((distance, other_idx, cues))

        # Sort by distance, take top max_visible
        visible.sort(key=lambda x: x[0])
        visible = visible[:self.max_visible_agents]

        # Encode cues as binary vector
        for i, (_, other_idx, cues) in enumerate(visible):
            for cue_str in cues:
                cue_idx = self.cue_engine.cue_to_index[cue_str]
                flat_idx = i * self.num_cue_types + cue_idx
                output[agent_idx, flat_idx] = 1.0

    return output
```

**Step 3: Update build_observations**

```python
def build_observations(self, ...):
    # Build base observation (L0-5)
    obs = ...  # existing code

    # Add social features (Curriculum Level 6+)
    if self.curriculum_level >= 6:
        positions_social = self._get_visible_agent_positions(
            positions, positions  # observer and all agents
        )
        cues_social = self._get_visible_agent_cues(
            positions, positions, all_agent_cues
        )
        obs = torch.cat([obs, positions_social, cues_social], dim=1)

    # Add family comm (Curriculum Level 8)
    if self.curriculum_level >= 8:
        family_signals = self._get_family_comm_channel(
            agent_ids, family_data
        )
        obs = torch.cat([obs, family_signals], dim=1)

    return obs
```

#### Implementation Guide: Adding Family Communication

**Step 1: Extend action space**

```python
# In environment setup
self.action_space = [
    'UP', 'DOWN', 'LEFT', 'RIGHT',  # movement
    'INTERACT',                      # use affordance
    'WAIT',                          # do nothing
    'SET_COMM_CHANNEL',              # Curriculum Level 8 only, param: [0, 999]
]
```

**Step 2: Track family communication state**

```python
# In agent state
class AgentState:
    def __init__(self):
        self.family_id = None
        self.family_members = []  # list of agent_ids
        self.current_signal = 0   # int [0, 999]
```

**Step 3: Process SET_COMM_CHANNEL action**

```python
def step(self, actions):
    for agent_idx, action in enumerate(actions):
        if action == Action.SET_COMM_CHANNEL:
            # Get parameter (signal value)
            signal_value = action_params[agent_idx]  # int [0, 999]
            self.agents[agent_idx].current_signal = signal_value
```

**Step 4: Build family_comm_channel in observation**

```python
def _get_family_comm_channel(
    self,
    agent_ids: torch.Tensor,
    family_data: dict,
) -> torch.Tensor:
    """Get signals from family members.

    Returns:
        [num_agents, max_family_size]
    """
    output = torch.zeros(
        len(agent_ids),
        self.max_family_size,
        device=self.device
    )

    for agent_idx, agent_id in enumerate(agent_ids):
        family_id = self.agents[agent_idx].family_id

        if family_id is None:
            continue  # not in a family, leave as zeros

        family_members = self.agents[agent_idx].family_members

        for i, member_id in enumerate(family_members):
            if i >= self.max_family_size:
                break

            # Get member's current signal
            signal = self.agents[member_id].current_signal
            normalized = signal / 1000.0
            output[agent_idx, i] = normalized

    return output
```

---

### 5.9 Observation Space Summary Table

| Component | L0-3 | L4-5 | L6 | L7 | L8 | Dim | Human Equivalent |
|-----------|------|------|----|----|----|----|------------------|
| **Spatial** | | | | | | | |
| Grid (one-hot) | ✓ | - | - | - | - | grid² | "I'm at (x,y)" |
| Local window | - | ✓ | ✓ | ✓ | ✓ | 25 | "I see 5×5 around me" |
| Position (abs) | - | ✓ | ✓ | ✓ | ✓ | 2 | "My coordinates" |
| **Internal** | | | | | | | |
| Meters (8 bars) | ✓ | ✓ | ✓ | ✓ | ✓ | 8 | "How I feel" |
| Current affordance | ✓ | ✓ | ✓ | ✓ | ✓ | types+1 | "Where I'm standing" |
| **Temporal** | | | | | | | |
| Time (sin/cos) | - | ✓ | ✓ | ✓ | ✓ | 2 | "What time is it" |
| Progress | - | ✓ | ✓ | ✓ | ✓ | 1 | "% done with action" |
| **Social** | | | | | | | |
| Other positions | - | - | ✓ | ✓ | ✓ | M×2 | "Where are others" |
| Other cues | - | - | ✓ | ✓ | ✓ | M×K | "How do they look" |
| **Communication** | | | | | | | |
| Family signals | - | - | - | - | ✓ | F | "What did they say" |

**Total dimensions**:

- L0-3: 49 (5×5 grid, 15 affordances)
- L4-5: 54
- L6: 79 (5 visible, 3 cues)
- L7: 124 (5 visible, 12 cues)
- L8: 127 (family size 3)

---

### 5.10 Validation Checklist

Before claiming observation space is correctly implemented:

**Curriculum Level 0-3 (Full Observability)**:

- [ ] Agent can determine its exact position from observation
- [ ] Agent knows all 8 meter values
- [ ] Agent knows which affordance (if any) it's on
- [ ] Observation dimension = grid² + 8 + (types+1)

**Curriculum Level 4-5 (Partial + Temporal)**:

- [ ] Agent sees only 5×5 local window
- [ ] Out-of-bounds tiles are encoded as 0
- [ ] Agent knows absolute position via normalized coordinates
- [ ] Time is cyclically encoded (sin/cos)
- [ ] Interaction progress is normalized [0, 1]
- [ ] Observation dimension = 25 + 2 + 8 + (types+1) + 3

**Curriculum Level 6-7 (Social)**:

- [ ] Agent sees positions of other agents within vision_range
- [ ] Positions are relative (not absolute)
- [ ] Closest agents appear first (sorted by distance)
- [ ] Cues are binary vectors (1 = cue active)
- [ ] Max visible agents enforced (padding with zeros)
- [ ] No telepathy (only public cues, not internal bars)

**Curriculum Level 8 (Family Comm)**:

- [ ] Family members' signals are in observation
- [ ] Non-family agents receive all zeros
- [ ] Signals are normalized [0, 1]
- [ ] SET_COMM_CHANNEL action updates agent's outgoing signal
- [ ] No semantic meaning provided (agents must learn)

**General**:

- [ ] All observations are deterministic (same state → same observation)
- [ ] All values are normalized or one-hot (no raw coordinates > grid_size)
- [ ] Batch dimension is first (for vectorized environments)
- [ ] Device handling is correct (CPU/GPU)

---

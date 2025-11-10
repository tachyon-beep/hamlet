# Temporal Mechanics System

## Time-of-Day System

```mermaid
flowchart LR
    tick_counter["Tick Counter<br/>0 → 23 → 0"]
    
    subgraph "Time Encoding"
        time_sin["time_sin = sin(2π × tick / 24)<br/>Smooth cyclic encoding"]
        time_cos["time_cos = cos(2π × tick / 24)<br/>Smooth cyclic encoding"]
    end
    
    subgraph "Operating Hours Masks"
        action_masks["Action Mask Table<br/>[24 hours, num_affordances]<br/>bool tensor"]
        check_hour["Current hour: 14<br/>masks[14, :] → valid affordances"]
    end
    
    subgraph "Affordance Availability"
        job["Job (9am-5pm)<br/>hours [9, 17]"]
        hospital["Hospital (24/7)<br/>hours None or [0, 24]"]
        gym["Gym (6am-10pm)<br/>hours [6, 22]"]
    end
    
    tick_counter --> time_sin
    tick_counter --> time_cos
    tick_counter --> check_hour
    
    action_masks --> check_hour
    check_hour --> job
    check_hour --> hospital
    check_hour --> gym
    
    style tick_counter fill:#fff9c4
    style time_sin fill:#c8e6c9
    style action_masks fill:#e1f5fe
```

## Operating Hours Validation

```mermaid
flowchart TD
    affordance["Affordance Definition"]
    check_hours{Has<br/>operating_hours?}
    always_open["Always Open (24/7)<br/>No time restrictions"]
    
    parse_hours["Parse [open_hour, close_hour]"]
    validate_open{open_hour<br/>in 0-23?}
    validate_close{close_hour<br/>in 1-28?}
    
    compute_mask["Compute Availability Mask<br/>For each hour 0-23"]
    
    error_open["ERROR: Invalid open_hour<br/>Must be 0-23"]
    error_close["ERROR: Invalid close_hour<br/>Must be 1-28"]
    
    mask_ready["Mask Ready<br/>action_mask_table[hour, aff_idx]"]
    
    affordance --> check_hours
    check_hours -->|No| always_open
    check_hours -->|Yes| parse_hours
    
    parse_hours --> validate_open
    validate_open -->|No| error_open
    validate_open -->|Yes| validate_close
    
    validate_close -->|No| error_close
    validate_close -->|Yes| compute_mask
    
    compute_mask --> mask_ready
    always_open --> mask_ready
    
    style affordance fill:#fff9c4
    style compute_mask fill:#c8e6c9
    style error_open fill:#ffccbc
```

## Hour Wrapping Logic

Operating hours can span midnight:

```python
# Example: Restaurant (5pm - 2am)
operating_hours: [17, 26]  # 17:00 to 02:00 next day

# Logic:
open_hour = 17 % 24 = 17
close_hour = 26 % 24 = 2

# For hour=18: 18 >= 17 → OPEN
# For hour=1:  1 < 2 → OPEN  
# For hour=10: Not (10 >= 17 or 10 < 2) → CLOSED
```

Examples:
- `[9, 17]`: 9am-5pm (standard business hours)
- `[0, 24]` or `[0, 28]`: 24/7 (always open)
- `[22, 30]` → `[22, 6]`: 10pm-6am (overnight)
- `[6, 22]`: 6am-10pm (gym hours)

## Action Mask Table Pre-computation

Compiled during Stage 6 (Optimization):

```mermaid
flowchart TD
    affordances["Affordances List<br/>[Aff1, Aff2, ..., AffN]"]
    
    create_table["Create Mask Table<br/>Shape: [24, N]<br/>dtype: bool<br/>device: cuda"]
    
    subgraph "For each hour 0-23"
        hour_loop["hour = 0..23"]
        
        subgraph "For each affordance"
            aff_loop["aff_idx = 0..N-1"]
            check_open{"is_open(hour,<br/>open, close)?"}
            set_true["mask[hour, aff_idx] = True"]
            set_false["mask[hour, aff_idx] = False"]
        end
    end
    
    store_optimization["Store in OptimizationData<br/>action_mask_table"]
    
    affordances --> create_table
    create_table --> hour_loop
    hour_loop --> aff_loop
    aff_loop --> check_open
    check_open -->|Open| set_true
    check_open -->|Closed| set_false
    set_true --> aff_loop
    set_false --> aff_loop
    aff_loop --> store_optimization
    
    style create_table fill:#c8e6c9
    style store_optimization fill:#e1f5fe
```

## Runtime Temporal Update

```mermaid
sequenceDiagram
    participant E as Environment
    participant T as TemporalState
    participant V as VFS Registry
    participant M as ActionMasking
    
    Note over E: Every step (if temporal_mechanics enabled)
    
    E->>T: Get current tick
    T-->>E: tick (0-23)
    
    E->>T: Compute time encoding
    T->>T: time_sin = sin(2π × tick / 24)
    T->>T: time_cos = cos(2π × tick / 24)
    T-->>E: (time_sin, time_cos)
    
    E->>V: Update time_sin variable<br/>set("time_sin", time_sin, writer="engine")
    E->>V: Update time_cos variable<br/>set("time_cos", time_cos, writer="engine")
    
    E->>M: Get action masks for current hour
    M->>M: masks = action_mask_table[tick, :]
    M-->>E: valid_affordances [num_affordances]
    
    Note over E: Apply masks to Q-values<br/>before action selection
```

## Multi-Tick Interaction Timeline

```mermaid
gantt
    title Multi-Tick Interaction with Operating Hours
    dateFormat X
    axisFormat %H:00
    
    section Job (9am-5pm)
    Available: 9, 17
    Unavailable: 0, 9
    Unavailable: 17, 24
    
    section Agent Interaction
    Start Job (9:15am): milestone, 9.25, 0
    Working (Ticks 1-5): 9.25, 12.5
    Completion (12:30pm): milestone, 12.5, 0
    
    section Failure Scenario
    Start Job (4:45pm): milestone, 16.75, 0
    Working (15 min): 16.75, 17
    Job Closes (5pm): crit, milestone, 17, 0
    Failure Triggered: crit, 17, 17.25
```

## Temporal Observation Features

If `enable_temporal_mechanics: true`:

```yaml
# variables_reference.yaml additions
variables:
  - id: "time_sin"
    scope: "global"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs", "bac"]
    writable_by: ["engine"]
    default: 0.0
    
  - id: "time_cos"
    scope: "global"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs", "bac"]
    writable_by: ["engine"]
    default: 1.0
    
exposed_observations:
  - id: "obs_time_sin"
    source_variable: "time_sin"
    exposed_to: ["agent"]
    shape: []
    
  - id: "obs_time_cos"
    source_variable: "time_cos"
    exposed_to: ["agent"]
    shape: []
```

Observation dimensions:
- Without temporal: 29 dims
- With temporal: 31 dims (+2 for time_sin/time_cos)

## Temporal Mechanics Configuration

```yaml
# training.yaml
environment:
  enable_temporal_mechanics: true  # Enable time-of-day system

# affordances.yaml
affordances:
  - id: "aff_job_office"
    name: "Office Job"
    operating_hours: [9, 17]  # 9am-5pm
    
  - id: "aff_gym"
    name: "Gym"
    operating_hours: [6, 22]  # 6am-10pm
    
  - id: "aff_hospital"
    name: "Hospital"
    # No operating_hours = 24/7
    
  - id: "aff_bar"
    name: "Bar"
    operating_hours: [17, 26]  # 5pm-2am (overnight)
```

## Time-Based Reward Shaping (Future)

Potential extension:

```yaml
# affordances.yaml (future)
affordances:
  - id: "aff_job"
    modes:
      daytime:
        hours: [9, 17]
        effects:
          - meter: "money"
            amount: 0.225  # Normal pay
      overtime:
        hours: [17, 21]
        effects:
          - meter: "money"
            amount: 0.3375  # 1.5x pay
```

## Temporal State Persistence

```python
# Checkpoint includes temporal state
checkpoint["temporal_state"] = {
    "current_tick": 14,  # Current hour (0-23)
    "time_sin": 0.866,   # Encoded time
    "time_cos": 0.5,
}

# On restore:
env.temporal_state.tick = checkpoint["temporal_state"]["current_tick"]
# Time variables restored via VFS
```

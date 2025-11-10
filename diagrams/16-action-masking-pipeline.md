# Action Masking Pipeline

## Overview

Action masking prevents agents from selecting invalid actions, improving sample efficiency and training stability. The masking pipeline has multiple stages:

1. **Base Mask**: Disabled actions from configuration
2. **Boundary Constraints**: Spatial substrate edge detection
3. **Operating Hours**: Temporal affordance availability
4. **Affordance Availability**: INTERACT action validity
5. **Dead Agent Masking**: Disable all actions for dead agents

## Main Pipeline Flow

```mermaid
flowchart TD
    start[get_action_masks called<br/>Every step before action selection]
    
    base_mask["Base Mask<br/>ActionSpace.get_base_action_mask()<br/>All actions = True (default enabled)"]
    
    subgraph "Boundary Constraints (Spatial Only)"
        check_spatial{position_dim<br/>>= 2?}
        skip_spatial[Skip boundary checks<br/>Aspatial substrates]
        
        check_edges["Check agent positions<br/>at_top, at_bottom, at_left, at_right"]
        mask_movements["Mask invalid movements<br/>UP at top → False<br/>DOWN at bottom → False<br/>LEFT at left → False<br/>RIGHT at right → False"]
        
        check_3d{position_dim<br/>== 3?}
        check_z_edges["Check Z-axis positions<br/>at_floor, at_ceiling"]
        mask_z["Mask Z movements<br/>UP_Z at ceiling → False<br/>DOWN_Z at floor → False"]
    end
    
    subgraph "Operating Hours (Temporal Mechanics)"
        check_temporal{enable_temporal<br/>mechanics?}
        skip_temporal[Skip operating hours]
        
        check_all_affordances["For each affordance"]
        check_open{is_open(affordance,<br/>time_of_day)?}
        mask_closed["Mask closed affordances<br/>action_masks[:, aff_idx] = False"]
    end
    
    subgraph "INTERACT Action Validity"
        init_interact_valid["on_valid_affordance = all False"]
        loop_affordances["For each deployed affordance"]
        check_on_affordance["is_on_position(agent_pos, aff_pos)"]
        mark_valid["on_valid_affordance |= on_this_aff"]
        
        apply_interact_mask["action_masks[:, INTERACT_idx] &= on_valid_affordance"]
    end
    
    subgraph "Dead Agent Masking"
        check_dead["dead = (health <= 0) | (energy <= 0)"]
        mask_all_dead["action_masks[dead] = all False"]
    end
    
    output["Return action_masks<br/>[num_agents, action_dim] bool"]
    
    start --> base_mask
    base_mask --> check_spatial
    
    check_spatial -->|Yes| check_edges
    check_spatial -->|No| skip_spatial
    
    check_edges --> mask_movements
    mask_movements --> check_3d
    
    check_3d -->|Yes| check_z_edges
    check_3d -->|No| check_temporal
    
    check_z_edges --> mask_z
    mask_z --> check_temporal
    skip_spatial --> check_temporal
    
    check_temporal -->|Yes| check_all_affordances
    check_temporal -->|No| skip_temporal
    
    check_all_affordances --> check_open
    check_open -->|Open| init_interact_valid
    check_open -->|Closed| mask_closed
    mask_closed --> init_interact_valid
    
    skip_temporal --> init_interact_valid
    
    init_interact_valid --> loop_affordances
    loop_affordances --> check_on_affordance
    check_on_affordance --> mark_valid
    mark_valid -->|More affordances| loop_affordances
    mark_valid -->|Done| apply_interact_mask
    
    apply_interact_mask --> check_dead
    check_dead --> mask_all_dead
    mask_all_dead --> output
    
    style base_mask fill:#d1c4e9
    style mask_all_dead fill:#ffccbc
    style output fill:#c8e6c9
```

## 1. Base Mask (ActionSpace Configuration)

### Base Mask Initialization

```mermaid
sequenceDiagram
    participant E as Environment
    participant A as ActionSpace
    participant M as ActionMetadata
    
    E->>A: get_base_action_mask(num_agents, device)
    
    loop For each action
        A->>M: Check action.enabled
        
        alt action.enabled == True
            M->>A: Return True (action available)
        else action.enabled == False
            M->>A: Return False (action disabled)
        end
    end
    
    A->>A: Build mask tensor [num_agents, action_dim]
    A->>E: Return base_mask
```

### ActionSpace Structure

```mermaid
classDiagram
    class ComposedActionSpace {
        +actions: list[ActionMetadata]
        +action_labels: dict
        +interact_action_idx: int
        +wait_action_idx: int
        +get_base_action_mask(num_agents, device) torch.Tensor
    }
    
    class ActionMetadata {
        +id: int
        +name: str
        +type: str
        +enabled: bool
        +delta: list[int]
        +costs: dict[str, float]
        +effects: dict[str, float]
        +get_action_mask(num_agents, device) torch.Tensor
    }
    
    ComposedActionSpace *-- ActionMetadata : contains
    
    style ComposedActionSpace fill:#c8e6c9
    style ActionMetadata fill:#fff9c4
```

## 2. Boundary Constraints (Spatial Substrates)

### Grid2D Boundary Masking

```mermaid
graph TB
    positions["Agent Positions<br/>[num_agents, 2]<br/>(x, y) coordinates"]
    grid_size["Grid Size<br/>width × height"]
    
    subgraph "Edge Detection"
        at_top["at_top = (y == 0)"]
        at_bottom["at_bottom = (y == height - 1)"]
        at_left["at_left = (x == 0)"]
        at_right["at_right = (x == width - 1)"]
    end
    
    subgraph "Movement Masking"
        mask_up["action_masks[at_top, UP_idx] = False"]
        mask_down["action_masks[at_bottom, DOWN_idx] = False"]
        mask_left["action_masks[at_left, LEFT_idx] = False"]
        mask_right["action_masks[at_right, RIGHT_idx] = False"]
    end
    
    positions --> at_top
    positions --> at_bottom
    positions --> at_left
    positions --> at_right
    grid_size --> at_top
    grid_size --> at_bottom
    grid_size --> at_left
    grid_size --> at_right
    
    at_top --> mask_up
    at_bottom --> mask_down
    at_left --> mask_left
    at_right --> mask_right
    
    style positions fill:#d1c4e9
    style mask_up fill:#ffccbc
```

### Example: Agent at Grid Edge

```mermaid
graph TB
    subgraph "8×8 Grid (Grid2D)"
        agent_corner["Agent at (0, 0)<br/>Top-left corner"]
        agent_top["Agent at (3, 0)<br/>Top edge"]
        agent_center["Agent at (4, 4)<br/>Center"]
    end
    
    subgraph "Agent at (0,0) - Corner"
        mask_corner["Masked: UP, LEFT<br/>Valid: DOWN, RIGHT, INTERACT, WAIT"]
    end
    
    subgraph "Agent at (3,0) - Top Edge"
        mask_top["Masked: UP<br/>Valid: DOWN, LEFT, RIGHT, INTERACT, WAIT"]
    end
    
    subgraph "Agent at (4,4) - Center"
        mask_center["Masked: None<br/>Valid: UP, DOWN, LEFT, RIGHT, INTERACT, WAIT"]
    end
    
    agent_corner --> mask_corner
    agent_top --> mask_top
    agent_center --> mask_center
    
    style mask_corner fill:#ffccbc
    style mask_center fill:#c8e6c9
```

### Grid3D Z-Axis Masking

```mermaid
graph TB
    positions["Agent Positions<br/>[num_agents, 3]<br/>(x, y, z) coordinates"]
    depth["Grid Depth<br/>e.g., 10 layers"]
    
    subgraph "Z-Axis Edge Detection"
        at_floor["at_floor = (z == 0)"]
        at_ceiling["at_ceiling = (z == depth - 1)"]
    end
    
    subgraph "Z-Movement Masking"
        mask_down_z["action_masks[at_floor, DOWN_Z_idx] = False"]
        mask_up_z["action_masks[at_ceiling, UP_Z_idx] = False"]
    end
    
    positions --> at_floor
    positions --> at_ceiling
    depth --> at_floor
    depth --> at_ceiling
    
    at_floor --> mask_down_z
    at_ceiling --> mask_up_z
    
    style positions fill:#d1c4e9
    style mask_down_z fill:#ffccbc
```

### Aspatial Substrates (No Boundary Masking)

```mermaid
graph TB
    position_dim["substrate.position_dim == 0"]
    
    skip["Skip all boundary checks<br/>No spatial constraints"]
    
    note["Aspatial universes have no concept of<br/>'position' or 'boundary'<br/>Only INTERACT and WAIT actions"]
    
    position_dim --> skip
    skip --> note
    
    style skip fill:#e1f5fe
    style note fill:#fff9c4
```

## 3. Operating Hours (Temporal Mechanics)

### Affordance Operating Hours Check

```mermaid
flowchart TD
    start[For each affordance]
    
    get_hours["Get operating_hours<br/>[open_hour, close_hour]"]
    
    check_wraparound{close_hour<br/>> 24?}
    
    normal_hours["Normal Hours<br/>open <= time < close"]
    wraparound_hours["Midnight Wraparound<br/>time >= open OR time < close % 24"]
    
    check_open{Is open<br/>at time_of_day?}
    
    keep_mask["Keep action enabled<br/>action_masks[:, aff_idx] = True"]
    mask_closed["Mask action<br/>action_masks[:, aff_idx] = False"]
    
    next["Next affordance"]
    
    start --> get_hours
    get_hours --> check_wraparound
    
    check_wraparound -->|No| normal_hours
    check_wraparound -->|Yes| wraparound_hours
    
    normal_hours --> check_open
    wraparound_hours --> check_open
    
    check_open -->|Yes| keep_mask
    check_open -->|No| mask_closed
    
    keep_mask --> next
    mask_closed --> next
    next -->|More affordances| start
    
    style keep_mask fill:#c8e6c9
    style mask_closed fill:#ffccbc
```

### Operating Hours Examples

```mermaid
graph TB
    subgraph "Hospital: [0, 24] - Always Open"
        hospital_hours["open_hour=0, close_hour=24"]
        hospital_check["Always returns True<br/>0 <= time < 24 (always true)"]
    end
    
    subgraph "Job: [9, 17] - Business Hours"
        job_hours["open_hour=9, close_hour=17"]
        job_check["True if 9 <= time < 17<br/>False otherwise"]
    end
    
    subgraph "Bar: [18, 28] - Night Hours with Wraparound"
        bar_hours["open_hour=18, close_hour=28"]
        bar_wraparound["close_hour > 24 detected"]
        bar_check["True if time >= 18 OR time < 4<br/>(28 % 24 = 4)"]
    end
    
    subgraph "Gym: [6, 22] - Extended Hours"
        gym_hours["open_hour=6, close_hour=22"]
        gym_check["True if 6 <= time < 22<br/>False otherwise"]
    end
    
    hospital_hours --> hospital_check
    job_hours --> job_check
    bar_hours --> bar_wraparound
    bar_wraparound --> bar_check
    gym_hours --> gym_check
    
    style hospital_check fill:#c8e6c9
    style bar_check fill:#e1f5fe
```

### Temporal Masking Timeline

```mermaid
gantt
    title Operating Hours (0-23) for Various Affordances
    dateFormat HH
    axisFormat %H
    
    section Hospital
    Open 24/7 :active, hospital, 00, 24h
    
    section Job
    Closed :done, job_closed1, 00, 9h
    Open :active, job_open, 09, 8h
    Closed :done, job_closed2, 17, 7h
    
    section Restaurant
    Closed :done, rest_closed1, 00, 11h
    Open :active, rest_open, 11, 12h
    Closed :done, rest_closed2, 23, 1h
    
    section Bar
    Closed :done, bar_closed, 00, 18h
    Open :active, bar_open, 18, 6h
    
    section Gym
    Closed :done, gym_closed1, 00, 6h
    Open :active, gym_open, 06, 16h
    Closed :done, gym_closed2, 22, 2h
```

## 4. INTERACT Action Validity

### INTERACT Masking Logic

```mermaid
flowchart TD
    start[Check INTERACT action validity]
    
    init["on_valid_affordance = all False<br/>[num_agents] bool tensor"]
    
    loop_start["For each deployed affordance"]
    
    check_temporal{enable_temporal<br/>mechanics?}
    check_open{is_affordance_open<br/>(time_of_day)?}
    skip_closed["Skip this affordance<br/>(already masked in step 3)"]
    
    check_position["is_on_position(agent_pos, aff_pos)<br/>Returns [num_agents] bool"]
    
    update_valid["on_valid_affordance |= on_this_affordance<br/>Accumulate agents on any open affordance"]
    
    loop_end["Next affordance"]
    
    apply_mask["action_masks[:, INTERACT_idx] &= on_valid_affordance<br/>INTERACT only valid if on open affordance"]
    
    output["INTERACT mask updated"]
    
    start --> init
    init --> loop_start
    
    loop_start --> check_temporal
    check_temporal -->|Yes| check_open
    check_temporal -->|No| check_position
    
    check_open -->|Open| check_position
    check_open -->|Closed| skip_closed
    
    check_position --> update_valid
    skip_closed --> loop_end
    
    update_valid --> loop_end
    loop_end -->|More affordances| loop_start
    loop_end -->|Done| apply_mask
    
    apply_mask --> output
    
    style init fill:#d1c4e9
    style apply_mask fill:#c8e6c9
```

### Affordance Position Check

```mermaid
graph TB
    agent_positions["Agent Positions<br/>[num_agents, 2]"]
    affordance_pos["Affordance Position<br/>[2] (x, y)"]
    
    substrate["Substrate.is_on_position()"]
    
    subgraph "Grid2D: Exact Match"
        exact["distance == 0<br/>compute_distance(agent_pos, aff_pos) == 0"]
    end
    
    subgraph "Continuous: Proximity Threshold"
        proximity["distance < threshold<br/>compute_distance(...) < 0.5"]
    end
    
    subgraph "Aspatial: Always True"
        always["All agents 'everywhere'<br/>return all True"]
    end
    
    result["on_affordance<br/>[num_agents] bool"]
    
    agent_positions --> substrate
    affordance_pos --> substrate
    
    substrate --> exact
    substrate --> proximity
    substrate --> always
    
    exact --> result
    proximity --> result
    always --> result
    
    style substrate fill:#d1c4e9
    style result fill:#c8e6c9
```

## 5. Dead Agent Masking

### Dead Agent Detection

```mermaid
flowchart TD
    meters["Meters Tensor<br/>[num_agents, num_meters]"]
    
    health_idx["health_idx<br/>(from meter_name_to_index)"]
    energy_idx["energy_idx<br/>(from meter_name_to_index)"]
    
    check_health["health_dead = (meters[:, health_idx] <= 0.0)"]
    check_energy["energy_dead = (meters[:, energy_idx] <= 0.0)"]
    
    combine["dead_agents = health_dead | energy_dead<br/>[num_agents] bool"]
    
    mask_all["action_masks[dead_agents] = all False<br/>Disable ALL actions for dead agents"]
    
    note["CRITICAL: This must be LAST step<br/>Overrides all other masking"]
    
    meters --> health_idx
    meters --> energy_idx
    
    health_idx --> check_health
    energy_idx --> check_energy
    
    check_health --> combine
    check_energy --> combine
    
    combine --> mask_all
    mask_all --> note
    
    style combine fill:#ffccbc
    style note fill:#fff9c4
```

### Death Condition Examples

```mermaid
graph TB
    subgraph "Agent 0: Healthy"
        agent0_meters["energy=0.8, health=0.9"]
        agent0_check["energy > 0 AND health > 0"]
        agent0_result["alive = True<br/>All actions available"]
    end
    
    subgraph "Agent 1: Energy Depleted"
        agent1_meters["energy=0.0, health=0.5"]
        agent1_check["energy <= 0 OR health > 0"]
        agent1_result["dead = True<br/>All actions masked"]
    end
    
    subgraph "Agent 2: Health Depleted"
        agent2_meters["energy=0.6, health=0.0"]
        agent2_check["energy > 0 OR health <= 0"]
        agent2_result["dead = True<br/>All actions masked"]
    end
    
    subgraph "Agent 3: Both Depleted"
        agent3_meters["energy=0.0, health=0.0"]
        agent3_check["energy <= 0 AND health <= 0"]
        agent3_result["dead = True<br/>All actions masked"]
    end
    
    agent0_meters --> agent0_check
    agent0_check --> agent0_result
    
    agent1_meters --> agent1_check
    agent1_check --> agent1_result
    
    agent2_meters --> agent2_check
    agent2_check --> agent2_result
    
    agent3_meters --> agent3_check
    agent3_check --> agent3_result
    
    style agent0_result fill:#c8e6c9
    style agent1_result fill:#ffccbc
    style agent2_result fill:#ffccbc
    style agent3_result fill:#ffccbc
```

## Action Mask Tensor Structure

### Grid2D Action Space (8 actions)

```mermaid
graph TB
    action_masks["action_masks<br/>[num_agents, 8] bool"]
    
    subgraph "Action Indices"
        up["0: UP"]
        down["1: DOWN"]
        left["2: LEFT"]
        right["3: RIGHT"]
        up_z["4: UP_Z (N/A for 2D)"]
        down_z["5: DOWN_Z (N/A for 2D)"]
        interact["6: INTERACT"]
        wait["7: WAIT"]
    end
    
    subgraph "Example Mask (Agent at top-left on open affordance)"
        mask_example["[False, True, False, True, False, False, True, True]<br/>UP masked (at top)<br/>LEFT masked (at left)<br/>INTERACT valid (on affordance)<br/>WAIT always valid"]
    end
    
    action_masks --> up
    action_masks --> down
    action_masks --> left
    action_masks --> right
    action_masks --> up_z
    action_masks --> down_z
    action_masks --> interact
    action_masks --> wait
    
    action_masks --> mask_example
    
    style action_masks fill:#d1c4e9
    style mask_example fill:#c8e6c9
```

### Aspatial Action Space (2 actions)

```mermaid
graph TB
    action_masks["action_masks<br/>[num_agents, 2] bool"]
    
    subgraph "Action Indices"
        interact["0: INTERACT"]
        wait["1: WAIT"]
    end
    
    subgraph "Example Mask (Agent on open affordance)"
        mask_open["[True, True]<br/>Both actions valid"]
    end
    
    subgraph "Example Mask (Agent on closed affordance)"
        mask_closed["[False, True]<br/>INTERACT masked (closed)<br/>WAIT always valid"]
    end
    
    action_masks --> interact
    action_masks --> wait
    
    action_masks --> mask_open
    action_masks --> mask_closed
    
    style action_masks fill:#d1c4e9
    style mask_open fill:#c8e6c9
    style mask_closed fill:#fff9c4
```

## Integration with Action Selection

### Masked Action Selection Flow

```mermaid
sequenceDiagram
    participant E as Environment
    participant P as Population
    participant Ex as Exploration
    participant N as Network
    
    Note over E,N: Step begins, select actions
    
    E->>E: get_action_masks()
    E->>E: Return action_masks [num_agents, action_dim]
    
    E->>P: select_actions(observations)
    P->>N: forward(observations)
    N->>P: q_values [num_agents, action_dim]
    
    P->>Ex: select_actions(q_values, agent_states, action_masks)
    
    Note over Ex: Epsilon-greedy with masking
    
    alt Explore (random)
        Ex->>Ex: Sample from valid actions only<br/>action_masks.float() as probabilities
    else Exploit (greedy)
        Ex->>Ex: q_values[~action_masks] = -inf<br/>Mask invalid actions with -inf
        Ex->>Ex: argmax(q_values)
    end
    
    Ex->>P: actions [num_agents]
    P->>E: Return actions
    
    E->>E: step(actions)
```

### Masking During Epsilon-Greedy

```mermaid
flowchart TD
    q_values["Q-Values<br/>[num_agents, action_dim]"]
    action_masks["Action Masks<br/>[num_agents, action_dim] bool"]
    epsilons["Epsilons<br/>[num_agents] float"]
    
    sample_random["Sample uniform random<br/>[num_agents]"]
    
    check_explore{random < epsilon<br/>for each agent?}
    
    explore_branch["EXPLORE"]
    exploit_branch["EXPLOIT"]
    
    subgraph "Explore: Random from Valid"
        valid_actions["valid_actions = action_masks.float()"]
        normalize["probabilities = valid / sum(valid)"]
        sample["torch.multinomial(probabilities)"]
        explore_action["Random valid action"]
    end
    
    subgraph "Exploit: Greedy with Masking"
        mask_invalid["masked_q = q_values.clone()"]
        set_neginf["masked_q[~action_masks] = -inf"]
        argmax["action = argmax(masked_q)"]
        exploit_action["Best valid action"]
    end
    
    combine["Combine explore/exploit actions<br/>torch.where(explore_mask, explore_action, exploit_action)"]
    
    output["Selected Actions<br/>[num_agents]"]
    
    q_values --> exploit_branch
    action_masks --> explore_branch
    action_masks --> exploit_branch
    epsilons --> sample_random
    
    sample_random --> check_explore
    
    check_explore -->|Yes| explore_branch
    check_explore -->|No| exploit_branch
    
    explore_branch --> valid_actions
    valid_actions --> normalize
    normalize --> sample
    sample --> explore_action
    
    exploit_branch --> mask_invalid
    mask_invalid --> set_neginf
    set_neginf --> argmax
    argmax --> exploit_action
    
    explore_action --> combine
    exploit_action --> combine
    
    combine --> output
    
    style explore_action fill:#e1f5fe
    style exploit_action fill:#c8e6c9
    style output fill:#fff9c4
```

## Summary

### Masking Priority Order

```
1. Base Mask (configuration-disabled actions)
2. Boundary Constraints (spatial edge detection)
3. Operating Hours (temporal affordance availability)
4. INTERACT Validity (affordance position check)
5. Dead Agent Masking (health <= 0 OR energy <= 0) [OVERRIDES ALL]
```

### Masking by Substrate Type

| Substrate | Boundary Masking | INTERACT Masking | Dead Agent Masking |
|-----------|------------------|------------------|-------------------|
| **Grid2D** | ✅ Edge detection (UP/DOWN/LEFT/RIGHT) | ✅ Exact position match | ✅ Always applied |
| **Grid3D** | ✅ Edge + Z-axis (UP_Z/DOWN_Z) | ✅ Exact position match | ✅ Always applied |
| **GridND** | ✅ Per-dimension edge detection | ✅ Exact position match | ✅ Always applied |
| **Continuous** | ❌ No boundaries | ✅ Proximity threshold | ✅ Always applied |
| **ContinuousND** | ❌ No boundaries | ✅ Proximity threshold | ✅ Always applied |
| **Aspatial** | ❌ No position concept | ✅ Always True (agents "everywhere") | ✅ Always applied |

### Performance Considerations

- **Hot Path**: `get_action_masks()` called every step before action selection
- **GPU Tensors**: All operations vectorized (no Python loops over agents)
- **Boolean Masking**: Efficient tensor indexing with boolean masks
- **Cached Indices**: Action indices (INTERACT, WAIT, UP_Z, DOWN_Z) cached at initialization

### Key Design Principles

1. **Safety First**: Invalid actions never selected (prevents crashes, illegal moves)
2. **Sample Efficiency**: Reduces exploration space, speeds learning
3. **Pedagogical Value**: Students see why agents avoid edges, respect hours
4. **Vectorized**: Single tensor operation for all agents
5. **Substrate-Agnostic**: Masking logic adapts to substrate type (Grid2D, Aspatial, etc.)
6. **Dead Agent Override**: Dead agents can't act (final safety check)

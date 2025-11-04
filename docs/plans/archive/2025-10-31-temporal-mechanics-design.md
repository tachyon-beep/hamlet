# Temporal Mechanics & Multi-Interaction Affordances Design

**Date**: October 31, 2025
**Status**: Design Complete, Ready for Implementation
**Complexity Level**: Level 2.5 (between POMDP and Multi-Zone)

---

## Executive Summary

Add time-based mechanics and multi-interaction affordances to increase temporal planning complexity. Agents must learn:

- **When** affordances are available (operating hours)
- **How long** to commit to interactions (partial vs full completion)
- **Opportunity cost** of time ("do I finish this shower or handle urgent need?")

**Key pedagogical value**: Teaches temporal planning, commitment decisions, and opportunity cost through emergent behavior like "I'm not that dirty, quick rinse is fine" and "I've got to finish work and go to the bar but I can squeeze in half a shower."

---

## Design Decisions

### 1. Time Representation

- **24-tick day cycle**: `time_of_day = step_count % 24`
  - Tick 0 = midnight
  - Tick 6 = 6am (dawn)
  - Tick 12 = noon
  - Tick 18 = 6pm (dusk)
- **Human-readable mapping**: Ticks map to hours for intuitive schedules
- **Added to observation**: +1 float `[0.0, 1.0]` (normalized tick/24)

### 2. Multi-Interaction Mechanics

- **Linear progress with early exit** (75/25 split)
  - 75% of benefit distributed evenly across required ticks
  - 25% bonus on full completion
  - Agent can exit anytime, keeps earned benefits
- **Per-tick costs**: Money charged each tick (prevents "free sampling")
- **Progress tracking**: Environment tracks ticks completed per agent
- **Reset on movement**: Progress resets if agent moves away from affordance

### 3. Dynamic Affordances

- **Time-dependent transformations**: Same grid position → different affordance by time
  - Example: CoffeeShop (8am-6pm) → Bar (6pm-4am)
- **Enables richer temporal learning**: "Position (3,5) at tick 10 = coffee, at tick 20 = bar"

---

## Architecture

### State Representation Changes

**Current**: 64 (grid) + 8 (meters) = 72 dimensions

**New additions**:

```python
observation = [
    grid_position,           # 64-dim one-hot (existing)
    meters,                  # 8-dim floats (existing)
    time_of_day,             # 1-dim float [0.0, 1.0] (NEW)
    affordance_progress,     # 1-dim float [0.0, 1.0] (NEW - ticks_done / required_ticks)
]
# Total: 74 dimensions
```

**Rationale**: Agent learns temporal patterns ("go to Job at tick 10") and commitment state ("I'm 2/4 through Job, should I commit?") without explicit affordance memory.

### Environment State Additions

```python
class VectorizedHamletEnv:
    def __init__(self, ...):
        # Existing fields...

        # Multi-interaction tracking
        self.interaction_progress = torch.zeros(
            self.num_agents,
            dtype=torch.long,
            device=self.device
        )  # Ticks completed on current affordance

        self.last_interaction_affordance = [None] * self.num_agents
        self.last_interaction_position = torch.zeros(
            (self.num_agents, 2),
            dtype=torch.long,
            device=self.device
        )
```

### Affordance Configuration Schema

```python
affordance_configs = {
    'affordance_name': {
        'required_ticks': int,        # Number of INTERACTs for full benefit
        'cost_per_tick': float,       # Money charged per tick (normalized [0, 1])
        'operating_hours': (int, int), # (open_tick, close_tick) in [0, 23]
        'benefits': {
            'linear': {               # 75% distributed across ticks
                'meter_name': float,  # Per-tick delta
            },
            'completion': {           # 25% bonus on full completion
                'meter_name': float,  # One-time bonus
            }
        }
    }
}
```

---

## Complete Affordance Specifications

### Static Affordances (24/7 Availability)

#### Bed (Home - Always Available)

```python
'Bed': {
    'required_ticks': 5,
    'cost_per_tick': 0.01,  # $1 per tick ($5 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'energy': +0.075,  # Per tick: (50% * 0.75) / 5 ticks
        },
        'completion': {
            'energy': +0.125,  # 50% * 0.25
            'health': +0.02,   # Well-rested bonus
        }
    }
}
# Total if completed: +50% energy, +2% health, -$5
# Early exit (3 ticks): +22.5% energy, -$3
```

#### LuxuryBed (Home - Premium Rest)

```python
'LuxuryBed': {
    'required_ticks': 5,
    'cost_per_tick': 0.022,  # $2.20 per tick ($11 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'energy': +0.1125,  # Per tick: (75% * 0.75) / 5 ticks
        },
        'completion': {
            'energy': +0.1875,  # 75% * 0.25
            'health': +0.05,    # Premium rest bonus
        }
    }
}
```

#### Shower (Home - Always Available)

```python
'Shower': {
    'required_ticks': 3,
    'cost_per_tick': 0.01,  # $1 per tick ($3 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'hygiene': +0.10,  # Per tick: (40% * 0.75) / 3 ticks
        },
        'completion': {
            'hygiene': +0.10,  # 40% * 0.25 - "thorough clean" bonus
        }
    }
}
# Total if completed: +40% hygiene, -$3
# Early exit (2 ticks): +20% hygiene, -$2 (quick rinse!)
```

#### HomeMeal (Home - Always Available)

```python
'HomeMeal': {
    'required_ticks': 2,
    'cost_per_tick': 0.015,  # $1.50 per tick ($3 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'satiation': +0.16875,  # Per tick: (45% * 0.75) / 2 ticks
        },
        'completion': {
            'satiation': +0.1125,  # 45% * 0.25
            'health': +0.03,       # Nutritious meal bonus
        }
    }
}
```

#### Hospital (Emergency - 24/7)

```python
'Hospital': {
    'required_ticks': 3,
    'cost_per_tick': 0.05,  # $5 per tick ($15 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'health': +0.225,  # Per tick: (60% * 0.75) / 3 ticks
        },
        'completion': {
            'health': +0.15,  # 60% * 0.25 - full treatment bonus
        }
    }
}
```

#### Gym (24-Hour Fitness)

```python
'Gym': {
    'required_ticks': 4,
    'cost_per_tick': 0.02,  # $2 per tick ($8 total)
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'fitness': +0.1125,  # Per tick: (30% * 0.75) / 4 ticks
            'energy': -0.03,     # Exercise is tiring
        },
        'completion': {
            'fitness': +0.075,  # 30% * 0.25 - full workout bonus
            'mood': +0.05,      # Endorphin rush
        }
    }
}
```

#### FastFood (24/7 Convenience)

```python
'FastFood': {
    'required_ticks': 1,
    'cost_per_tick': 0.10,  # $10
    'operating_hours': (0, 24),
    'benefits': {
        'linear': {
            'satiation': +0.3375,  # (45% * 0.75) / 1 tick
            'energy': +0.1125,     # (15% * 0.75) / 1 tick - sugar rush
        },
        'completion': {
            'satiation': +0.1125,  # 45% * 0.25
            'energy': +0.0375,     # 15% * 0.25
            'fitness': -0.03,      # Unhealthy
            'health': -0.02,       # Long-term cost
        }
    }
}
```

### Business Hours Affordances (8am-6pm)

#### Job (Office Work - Sustainable Income)

```python
'Job': {
    'required_ticks': 4,
    'cost_per_tick': 0.0,  # Free to work
    'operating_hours': (8, 18),  # 8am-6pm
    'benefits': {
        'linear': {
            'money': +0.140625,  # Per tick: ($22.5 * 0.75) / 4 ticks
            'energy': -0.0375,   # Spread the energy cost
        },
        'completion': {
            'money': +0.05625,   # $22.5 * 0.25 - full shift bonus
            'social': +0.02,     # Coworker interaction (full day)
            'health': -0.03,     # Accumulated work stress
        }
    }
}
# Total if completed: +$22.5, -15% energy, +2% social, -3% health
# Early exit (2 ticks): +$11.25, -7.5% energy (no social, no stress penalty)
```

#### Labor (Physical Work - High Pay, High Cost)

```python
'Labor': {
    'required_ticks': 4,
    'cost_per_tick': 0.0,
    'operating_hours': (8, 18),
    'benefits': {
        'linear': {
            'money': +0.1875,   # Per tick: ($30 * 0.75) / 4 ticks
            'energy': -0.05,    # Exhausting
        },
        'completion': {
            'money': +0.075,    # $30 * 0.25
            'fitness': -0.05,   # Physical wear and tear
            'health': -0.05,    # Injury risk
            'social': +0.01,    # Minimal crew interaction
        }
    }
}
```

#### Doctor (Medical Care - Business Hours)

```python
'Doctor': {
    'required_ticks': 2,
    'cost_per_tick': 0.04,  # $4 per tick ($8 total)
    'operating_hours': (8, 18),
    'benefits': {
        'linear': {
            'health': +0.1125,  # Per tick: (30% * 0.75) / 2 ticks
        },
        'completion': {
            'health': +0.075,  # 30% * 0.25
        }
    }
}
```

#### Therapist (Mental Health - Business Hours)

```python
'Therapist': {
    'required_ticks': 3,
    'cost_per_tick': 0.05,  # $5 per tick ($15 total)
    'operating_hours': (8, 18),
    'benefits': {
        'linear': {
            'mood': +0.15,     # Per tick: (40% * 0.75) / 3 ticks
        },
        'completion': {
            'mood': +0.10,     # 40% * 0.25 - breakthrough session
            'social': +0.05,   # Therapeutic relationship
        }
    }
}
```

#### Recreation (Daytime Activity)

```python
'Recreation': {
    'required_ticks': 2,
    'cost_per_tick': 0.03,  # $3 per tick ($6 total)
    'operating_hours': (8, 22),  # 8am-10pm
    'benefits': {
        'linear': {
            'mood': +0.1125,  # Per tick: (30% * 0.75) / 2 ticks
            'social': +0.075, # Per tick: (20% * 0.75) / 2 ticks
        },
        'completion': {
            'mood': +0.075,   # 30% * 0.25
            'social': +0.05,  # 20% * 0.25
        }
    }
}
```

### Dynamic Affordances (Time-Dependent Transformations)

#### CoffeeShop (Daytime - Position Shared with Bar)

```python
'CoffeeShop': {
    'required_ticks': 1,
    'cost_per_tick': 0.02,  # $2
    'operating_hours': (8, 18),  # 8am-6pm
    'benefits': {
        'linear': {
            'energy': +0.1125,  # (15% * 0.75) / 1 tick
            'mood': +0.0375,    # (5% * 0.75) / 1 tick
            'social': +0.045,   # (6% * 0.75) / 1 tick - chitchat with barista
        },
        'completion': {
            'energy': +0.0375,  # 15% * 0.25
            'mood': +0.0125,    # 5% * 0.25
            'social': +0.015,   # 6% * 0.25
        }
    }
}
# Total: +15% energy, +5% mood, +6% social, -$2
# Social: more than Job (+2%) but less than Bar (+15%)
```

#### Bar (Evening/Night - Same Position as CoffeeShop)

```python
'Bar': {
    'required_ticks': 2,
    'cost_per_tick': 0.075,  # $7.50 per round ($15 total)
    'operating_hours': (18, 4),  # 6pm-4am (wraps midnight)
    'benefits': {
        'linear': {
            'mood': +0.075,     # Per tick: (20% * 0.75) / 2
            'social': +0.05625, # Per tick: (15% * 0.75) / 2
            'health': -0.01875, # Per tick: (-5% * 0.75) / 2
        },
        'completion': {
            'mood': +0.05,      # 20% * 0.25
            'social': +0.0375,  # 15% * 0.25
            'health': -0.0125,  # -5% * 0.25
        }
    }
}
```

#### Park (Dawn to Dusk)

```python
'Park': {
    'required_ticks': 2,
    'cost_per_tick': 0.0,  # Free
    'operating_hours': (6, 22),  # 6am-10pm
    'benefits': {
        'linear': {
            'mood': +0.0975,   # Per tick: (26% * 0.75) / 2
            'social': +0.0375, # Per tick: (10% * 0.75) / 2
        },
        'completion': {
            'mood': +0.065,    # 26% * 0.25
            'social': +0.025,  # 10% * 0.25
            'fitness': +0.02,  # Walking/outdoor activity
        }
    }
}
```

---

## Action Masking Updates

**Current masking**: Movement (blocked by boundaries), INTERACT (blocked if not on affordable affordance)

**New masking logic**:

```python
def get_action_masks(self) -> torch.Tensor:
    """
    INTERACT now checks:
    1. Am I on an affordance? (position match)
    2. Can I afford it? (money >= cost_per_tick)
    3. Is it currently open? (time_of_day in operating_hours)
    """
    action_masks = torch.ones(
        (self.num_agents, self.action_dim),
        dtype=torch.bool,
        device=self.device
    )

    # ... existing boundary checks for movement ...

    # INTERACT masking
    time_of_day = self.step_counts[0] % 24  # Global time

    for position_key, affordance_schedule in dynamic_affordances.items():
        # Get current affordance at this position (may vary by time)
        current_affordance = get_affordance_at_time(
            affordance_schedule,
            time_of_day
        )

        if current_affordance is None:
            continue  # No affordance active at this time

        # Check agents at this position
        at_position = (self.positions == parse_position(position_key)).all(dim=1)

        # Check affordability (per-tick cost)
        config = affordance_configs[current_affordance]
        cost_per_tick = config['cost_per_tick']
        can_afford = self.meters[:, 3] >= cost_per_tick

        # Valid if: on position AND can afford AND is open
        valid_interact = at_position & can_afford
        action_masks[:, 4] |= valid_interact

    return action_masks
```

**Wraparound handling** for hours like Bar (18, 4):

```python
def is_open(time_of_day: int, hours: tuple) -> bool:
    """Check if affordance is open, handling midnight wraparound."""
    open_tick, close_tick = hours

    if open_tick < close_tick:
        # Normal hours (e.g., 8-18)
        return open_tick <= time_of_day < close_tick
    else:
        # Wraparound hours (e.g., 18-4 means 6pm to 4am)
        return time_of_day >= open_tick or time_of_day < close_tick
```

---

## Interaction Progress Tracking

### Progress State Management

```python
def _handle_interactions(self, interact_mask: torch.Tensor) -> dict:
    """
    Handle INTERACT actions with multi-tick accumulation.

    Returns:
        Dictionary mapping agent indices to affordance names
    """
    successful_interactions = {}

    for agent_idx in torch.where(interact_mask)[0]:
        current_pos = self.positions[agent_idx]
        affordance_name = self._get_affordance_at_position(agent_idx)

        if affordance_name is None:
            continue

        config = affordance_configs[affordance_name]

        # Check if continuing same affordance at same position
        if (self.last_interaction_affordance[agent_idx] == affordance_name and
            torch.equal(current_pos, self.last_interaction_position[agent_idx])):
            # Continue progress
            self.interaction_progress[agent_idx] += 1
        else:
            # New affordance - reset progress
            self.interaction_progress[agent_idx] = 1
            self.last_interaction_affordance[agent_idx] = affordance_name
            self.last_interaction_position[agent_idx] = current_pos.clone()

        ticks_done = self.interaction_progress[agent_idx].item()
        required_ticks = config['required_ticks']

        # Apply per-tick benefits (75% of total, distributed)
        for meter_name, delta in config['benefits']['linear'].items():
            meter_idx = self.meter_name_to_idx[meter_name]
            self.meters[agent_idx, meter_idx] += delta

        # Charge per-tick cost
        self.meters[agent_idx, 3] -= config['cost_per_tick']  # Money

        # Completion bonus? (25% of total)
        if ticks_done == required_ticks:
            for meter_name, delta in config['benefits']['completion'].items():
                meter_idx = self.meter_name_to_idx[meter_name]
                self.meters[agent_idx, meter_idx] += delta

            # Reset progress (job complete)
            self.interaction_progress[agent_idx] = 0
            self.last_interaction_affordance[agent_idx] = None

        successful_interactions[agent_idx.item()] = affordance_name

        # Clamp meters after updates
        self.meters = torch.clamp(self.meters, 0.0, 1.0)

    return successful_interactions
```

### Progress Reset on Movement

```python
def step(self, actions: torch.Tensor):
    """
    Step environment, handling movement and progress reset.
    """
    # Store positions before movement
    old_positions = self.positions.clone()

    # ... execute movement ...

    # Reset progress for agents that moved away from their affordance
    for agent_idx in range(self.num_agents):
        if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
            # Agent moved - reset progress
            self.interaction_progress[agent_idx] = 0
            self.last_interaction_affordance[agent_idx] = None

    # ... rest of step logic ...
```

---

## UI Visualization Enhancements

### 1. Time-of-Day Gradient Bar

**Visual Design**:

- Gradient bar showing 24-tick cycle
- Day (6am-6pm): Yellow → Black gradient
- Night (6pm-6am): Black → Yellow gradient
- Current tick indicator (vertical line or arrow)

**Implementation**:

```typescript
function getTimeGradient(tick: number): string {
  if (tick >= 6 && tick < 18) {
    // Day progression: Yellow fades to black
    const progress = (tick - 6) / 12;  // 0.0 at 6am → 1.0 at 6pm
    return `linear-gradient(to right,
      hsl(48, 100%, 50%) ${(1 - progress) * 100}%,
      hsl(0, 0%, 10%) ${progress * 100}%)`;
  } else {
    // Night progression: Black fades to yellow
    const tick_normalized = tick >= 18 ? tick - 18 : tick + 6;
    const progress = tick_normalized / 12;  // 0.0 at 6pm → 1.0 at 6am
    return `linear-gradient(to right,
      hsl(0, 0%, 10%) ${(1 - progress) * 100}%,
      hsl(48, 100%, 50%) ${progress * 100}%)`;
  }
}
```

**UI Elements**:

```vue
<div class="time-bar">
  <div
    class="gradient-fill"
    :style="{ background: getTimeGradient(currentTick) }"
  />
  <div class="tick-marker" :style="{ left: `${(currentTick / 24) * 100}%` }">
    {{ formatTime(currentTick) }}
  </div>
</div>
```

### 2. Affordance Transition Animation

**Visual Design**:

- Pulsing circle during transition (tick 18: CoffeeShop → Bar)
- Icon swap with fade
- Background color shift (bright → dark for night service)

**Implementation**:

```vue
<template>
  <!-- Transition pulse effect -->
  <circle
    v-if="isTransitioning"
    :cx="affordanceX"
    :cy="affordanceY"
    r="40"
    class="transition-pulse"
    fill="none"
    stroke="white"
    stroke-width="2"
  />

  <!-- Affordance icon with time-based styling -->
  <g :class="affordanceTimeClass">
    <rect
      :x="affordanceX - 35"
      :y="affordanceY - 35"
      width="70"
      height="70"
      rx="8"
      :class="['affordance-background', affordanceTimeClass]"
    />
    <image
      :href="currentAffordanceIcon"
      :x="affordanceX - 25"
      :y="affordanceY - 25"
      width="50"
      height="50"
    />
  </g>
</template>

<style>
.transition-pulse {
  animation: pulse 1s ease-in-out;
}

@keyframes pulse {
  0%, 100% {
    opacity: 0;
    r: 40;
  }
  50% {
    opacity: 1;
    r: 50;
  }
}

.affordance-daytime {
  fill: #f59e0b;        /* Bright amber */
  background: #fef3c7;  /* Light amber bg */
}

.affordance-nighttime {
  fill: #7c3aed;        /* Deep purple */
  background: #1e1b4b;  /* Dark indigo bg */
}

.affordance-closed {
  filter: grayscale(100%);
  opacity: 0.5;
}
</style>
```

### 3. Interaction Progress Ring

**Visual Design**:

- Progress ring around agent sprite
- Color-coded: yellow (in progress), green (completed)
- Shows ticks_completed / ticks_required

**Implementation**:

```vue
<template>
  <!-- Progress ring around agent -->
  <circle
    v-if="agentProgress > 0"
    :cx="agentX"
    :cy="agentY"
    r="38"
    fill="none"
    :stroke="progressColor"
    stroke-width="3"
    :stroke-dasharray="circumference"
    :stroke-dashoffset="progressOffset"
    class="progress-ring"
    transform="rotate(-90)"
    :transform-origin="`${agentX} ${agentY}`"
  />

  <!-- Progress text -->
  <text
    v-if="agentProgress > 0"
    :x="agentX"
    :y="agentY + 50"
    text-anchor="middle"
    class="progress-text"
  >
    {{ ticksCompleted }}/{{ ticksRequired }}
  </text>
</template>

<script>
const circumference = 2 * Math.PI * 38;  // r=38

const progressOffset = computed(() => {
  const progress = ticksCompleted.value / ticksRequired.value;
  return circumference * (1 - progress);
});

const progressColor = computed(() => {
  return ticksCompleted.value === ticksRequired.value
    ? '#22c55e'  // Green (completed)
    : '#eab308'; // Yellow (in progress)
});
</script>
```

### 4. State Update Message Additions

**WebSocket payload enhancements**:

```typescript
interface StateUpdate {
  // ... existing fields ...

  time_of_day: number;  // 0-23

  interaction_progress: {
    [agent_id: string]: {
      affordance_name: string;
      ticks_completed: number;
      ticks_required: number;
      progress_ratio: number;  // 0.0-1.0 for easy rendering
    }
  };

  affordance_availability: {
    [position: string]: {  // "3,5"
      current_affordance: string | null;  // "CoffeeShop" or "Bar" or null
      is_open: boolean;
      opens_at: number | null;   // Tick when it opens (if closed)
      closes_at: number | null;  // Tick when it closes (if open)
    }
  };
}
```

---

## Pedagogical Value & Learning Progression

### Early Episodes (0-500): "Time is Confusing"

**Observable behaviors**:

- Agent goes to Job at random times (tick 3, tick 22, etc.)
- Gets stuck at closed affordances (tries to INTERACT when masked out)
- Dies at 6pm because Job closed and agent hasn't learned alternatives
- No temporal pattern recognition

**Teaching moment**: "The agent doesn't understand time yet. Watch how it learns."

### Mid Episodes (500-2000): "Learning the Clock"

**Observable behaviors**:

- Agent starts going to Job during business hours (tick 8-16)
- Completes 2-3 ticks of Job, then leaves (partial strategy)
- Discovers CoffeeShop before Job (energy boost for work)
- Still struggles with commitment (bails on Shower after 1 tick)

**Teaching moment**: "Now it knows WHEN things are available. Next: learning HOW LONG to commit."

### Late Episodes (2000-5000): "Sophisticated Scheduling"

**Observable behaviors**:

- **Morning routine**: CoffeeShop (1 tick) → Job (4 ticks, full shift)
- **Opportunistic efficiency**: Shower for 2/3 ticks when "not that dirty"
- **Evening optimization**: Bar (2 ticks, social recovery) → Bed (5 ticks, full sleep)
- **Shift splitting**: Goes to Job twice (2 ticks each) to avoid energy crash

**Teaching moment**: "This is graduate-level temporal planning. The agent has learned opportunity cost."

### Advanced Emergent Behaviors (5000+ episodes)

**Documented behaviors to watch for**:

1. **"Power napping"**: Bed for 2-3 ticks instead of full 5 when time-constrained
   - Pedagogical value: Agent learns "good enough" vs "optimal"

2. **"Pre-work prep"**: CoffeeShop → Job sequence maximizes energy during shift
   - Pedagogical value: Temporal sequencing for synergistic effects

3. **"Late-night Bar run"**: Bar at tick 22-23, then Bed at tick 0
   - Pedagogical value: Agent learns to use transition periods efficiently

4. **"Emergency triage"**: Bails on Shower (2/3 complete) to rush to Hospital
   - Pedagogical value: Dynamic re-planning based on urgent needs

5. **"Shift optimization"**: Completes exactly 4 ticks at Job, leaves immediately (no overtime)
   - Pedagogical value: Agent learns commitment threshold

### Connection to Real-World RL

**This teaches students**:

- **Temporal credit assignment**: "Which past actions led to this reward 10 ticks later?"
- **Horizon planning**: "Do I commit 5 ticks to Bed or handle urgent need?"
- **Opportunity cost**: "What am I giving up by staying here?"
- **Non-stationary environments**: "The world changes around me (time), I must adapt"

**Real-world parallels**:

- Delivery routing with time windows (packages must arrive 9am-5pm)
- Shift scheduling for workers (maximize coverage, minimize overtime)
- Resource allocation in time-sensitive contexts (hospital bed management)

---

## Implementation Risks & Mitigations

### Risk 1: Time Horizon Too Long

**Problem**: 24-tick cycle may be too long for agent to learn temporal patterns

**Mitigation**:

- Start with shorter cycle (12 ticks) for initial experiments
- Increase to 24 if learning succeeds
- Monitor Q-value variance across time steps

### Risk 2: Observation Dimensionality

**Problem**: 74 dimensions may exceed network capacity for recurrent architecture

**Mitigation**:

- Recurrent network already handles spatial patterns, temporal should be natural extension
- If needed, increase LSTM hidden size from current value
- Validate with baseline training runs

### Risk 3: Completion Bonus Too Small

**Problem**: 25% bonus may not incentivize full commitment

**Mitigation**:

- Tunable via config: `completion_bonus_ratio = 0.25` (default)
- Monitor agent behavior: if always bailing early, increase to 0.40
- Document as hyperparameter in config schema

### Risk 4: Dynamic Affordance Confusion

**Problem**: Agent may not understand position (3,5) is CoffeeShop at tick 10, Bar at tick 20

**Mitigation**:

- Observation includes time_of_day (agent can learn correlation)
- Action masking prevents invalid interactions (agent can't INTERACT with closed affordance)
- If still struggles, add affordance_type to observation (one-hot encoding)

### Risk 5: Training Time Increase

**Problem**: More complex environment may slow episode throughput

**Mitigation**:

- Current throughput: ~10,000 episodes/hour
- Expected with temporal mechanics: ~8,000 episodes/hour (20% slower)
- Still acceptable for overnight training runs
- GPU parallelization should compensate

---

## Success Metrics

**Quantitative indicators**:

1. **Temporal pattern learning**: Agent visits Job 80%+ during business hours (tick 8-18)
2. **Completion rate**: Agent completes full Job shift (4/4 ticks) 60%+ of the time
3. **Opportunistic efficiency**: Agent does partial Shower (2/3 ticks) when hygiene >60%
4. **Survival time**: Increases to >300 steps with temporal planning (vs ~200 without)

**Qualitative indicators**:

1. Agent learns "morning routine" (CoffeeShop → Job sequence)
2. Agent adapts to closed affordances (doesn't get stuck waiting for Job to open)
3. Agent shows time-dependent behavior (different strategies at tick 10 vs tick 20)
4. Agent demonstrates opportunity cost reasoning (bails on low-priority tasks for urgent needs)

**Failure modes to watch for**:

- Agent never completes full interactions (always bails at 1 tick) → bonus too small
- Agent ignores time (goes to Job at random ticks) → time not in observation properly
- Agent gets stuck at closed affordances (spams INTERACT) → action masking bug
- Training doesn't converge after 5000 episodes → complexity too high, simplify

---

## Future Extensions

### Level 3: Multi-Agent Conflict

**Once temporal mechanics work well**:

- Add resource contention: Only one agent can use Bed at a time
- Agent must learn "Is position (2,3) occupied? Should I wait or go elsewhere?"
- Teaches coordination and competition dynamics

### Level 4: Emergent Communication

**After multi-agent conflict**:

- Agents can "signal" intentions (reserve affordance for N ticks)
- Cooperative scheduling emerges
- Foundation for language grounding

### Advanced Temporal Features

- **Fatigue accumulation**: Energy depletion rate increases with consecutive work shifts
- **Circadian rhythm**: Sleep effectiveness varies by time (tick 0-6 = best sleep)
- **Rush hour pricing**: Job pays more during peak hours (tick 12-14)
- **Happy hour**: Bar costs less but gives same benefits (tick 18-20)

---

## Configuration Schema

**New config section** (`configs/townlet_temporal.yaml`):

```yaml
environment:
  grid_size: 8
  partial_observability: true
  vision_range: 2

  # NEW: Temporal mechanics
  time_system:
    enabled: true
    ticks_per_day: 24

  # NEW: Multi-interaction system
  multi_interaction:
    enabled: true
    completion_bonus_ratio: 0.25  # 75/25 split

  # NEW: Dynamic affordances
  dynamic_affordances:
    - position: [3, 5]
      daytime: CoffeeShop
      daytime_hours: [8, 18]
      nighttime: Bar
      nighttime_hours: [18, 4]

population:
  num_agents: 1
  learning_rate: 0.0001
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: recurrent

# ... rest of config ...
```

---

## Commit Plan

**Phase 1**: Environment backend (multi-interaction tracking)
**Phase 2**: Time system (24-tick cycle, action masking)
**Phase 3**: Dynamic affordances (CoffeeShop/Bar transformation)
**Phase 4**: UI visualization (gradient bar, progress rings)
**Phase 5**: Training validation (5000 episodes, monitor metrics)

**Each phase**: Commit separately with clear message documenting what changed.

---

## Related Documents

- [Milestone Rewards Design](../teachable_moments/milestone_rewards_design.md)
- [Complexity Types](../teachable_moments/complexity_types.md)
- [Trick Students Pedagogy](../teachable_moments/trick_students_pedagogy.md)

---

**Status**: Ready for worktree setup and implementation planning

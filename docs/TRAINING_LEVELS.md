# Hamlet Training Levels - Formal Specification

This document defines the progression of complexity levels in the Hamlet RL environment. Each level adds specific capabilities that increase the cognitive and planning demands on the agent.

---

## Level 1: Full Observability Baseline

**Status:** ✅ Implemented  
**Config:** `configs/level_1_full_observability.yaml`  
**Network:** SimpleQNetwork (MLP)

### Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Full (agent sees entire 8×8 grid) |
| **Memory** | Not required (stateless MLP) |
| **Temporal Mechanics** | Disabled (no time-of-day cycles) |
| **Multi-tick Interactions** | Disabled (instant affordance effects) |
| **Reward Shaping** | Sparse (milestone bonuses only, NO proximity) |
| **Curriculum** | Adversarial (5-stage progressive difficulty) |

### Teaching Value

- **Baseline for comparison**: How much does partial observability hurt?
- **Reward design**: Students learn why proximity shaping causes reward hacking
- **Curriculum learning**: Adversarial progression prevents catastrophic forgetting

### Expected Performance

- **Learning time:** 1000-2000 episodes
- **Peak survival:** 250-350 steps
- **Success criteria:** Agent learns to prioritize meters (satiation → health → energy)

---

## Level 2: Partial Observability (POMDP)

**Status:** ✅ Implemented  
**Config:** `configs/level_2_pomdp.yaml`  
**Network:** RecurrentSpatialQNetwork (CNN + LSTM)

### Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Partial (5×5 local window, agent at center) |
| **Memory** | Required (LSTM for spatial memory) |
| **Temporal Mechanics** | Disabled |
| **Multi-tick Interactions** | Disabled |
| **Reward Shaping** | Sparse |
| **Curriculum** | Adversarial |

### What Changed from Level 1

**Added:**
- Partial observability (vision_range=2 → 5×5 window)
- LSTM hidden state for memory
- Spatial reasoning requirements (build mental map)
- Target network for temporal credit assignment (ACTION #9)

**Removed:**
- Full grid visibility
- Stateless decision making

### Teaching Value

- **Working memory**: Agent must remember affordance locations
- **Exploration vs exploitation**: Balancing discovery with survival
- **Partial information**: Real-world agents rarely have perfect knowledge
- **Recurrent architectures**: When and why to use LSTM

### Expected Performance

- **Learning time:** 3000-5000 episodes (slower than Level 1)
- **Peak survival:** 150-250 steps (lower than Level 1)
- **Success criteria:** Agent explores, builds spatial memory, returns to known affordances

### Key Implementation Detail

Uses **DRQN (Deep Recurrent Q-Network)** with target network to handle temporal dependencies. Without target network, LSTM cannot learn (see ACTION #9 implementation).

---

## Level 3: Temporal Mechanics

**Status:** ✅ Implemented  
**Config:** `configs/level_3_temporal.yaml`  
**Network:** RecurrentSpatialQNetwork (CNN + LSTM)

### Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Partial (5×5 window) |
| **Memory** | Required (LSTM) |
| **Temporal Mechanics** | **Enabled (24-tick day/night cycle)** |
| **Multi-tick Interactions** | **Enabled (progressive benefits)** |
| **Reward Shaping** | Sparse |
| **Curriculum** | Adversarial |

### What Changed from Level 2

**Added:**
- 24-tick day/night cycle (time_of_day state)
- Operating hours for affordances (Job: 8am-6pm, Bar: 6pm-4am, etc.)
- Multi-tick interactions (Bed: 5 ticks for full benefit)
- Completion bonuses (75% linear progress + 25% completion reward)
- Time-based action masking (can't use closed affordances)

### Teaching Value

- **Temporal planning**: When to sleep vs work vs socialize
- **Opportunity cost**: Interrupting a 5-tick bed session loses progress
- **Scheduling constraints**: Work is only available during business hours
- **Sequence learning**: LSTM learns temporal patterns (night → Bar, day → Job)

### Expected Performance

- **Learning time:** 5000-8000 episodes (significantly harder)
- **Peak survival:** 150-250 steps (similar to Level 2 but more complex)
- **Success criteria:** Agent learns time-dependent strategies (sleep at night, work during day)

### Key Implementation Detail

Observation includes `time_of_day` (0-23) and `interaction_progress` (0-5) for current affordance. LSTM must learn which affordances are available at which times.

---

## Level 4: Multi-Zone Environment (Future)

**Status:** 🎯 Planned (Phase 6)  
**Config:** `configs/level_4_multi_zone.yaml` (not yet created)  
**Network:** Hierarchical RL architecture

### Planned Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Partial (per zone) |
| **Memory** | Hierarchical (zone-level + intra-zone) |
| **Temporal Mechanics** | Enabled |
| **Multi-tick Interactions** | Enabled |
| **Environment** | **Multiple zones (home, work, downtown)** |
| **Travel** | **Zone transitions with costs** |
| **Reward Shaping** | Sparse + hierarchical |
| **Curriculum** | Adversarial |

### Planned Changes

**Added:**
- Multiple interconnected zones (3-5 zones)
- Zone transitions (doors/gates between zones)
- Zone-specific affordances (home has Bed, work has Job, etc.)
- Travel cost (energy/time to move between zones)
- Hierarchical decision making (which zone? then which affordance?)

### Teaching Value

- **Hierarchical RL**: High-level (zone selection) + low-level (affordance selection)
- **Long-term planning**: Go home to sleep, then travel to work
- **Spatial abstraction**: Mental model of zone connectivity
- **Options framework**: Macro-actions (e.g., "go_to_work" = navigate + enter zone)

---

## Level 5: Multi-Agent Competition (Future)

**Status:** 🎯 Planned (Phase 6)  
**Config:** `configs/level_5_multi_agent.yaml` (not yet created)  
**Network:** RecurrentSpatialQNetwork + Theory of Mind module

### Planned Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Partial (see other agents in vision) |
| **Memory** | LSTM + agent tracking |
| **Temporal Mechanics** | Enabled |
| **Multi-tick Interactions** | Enabled (blocking mechanics) |
| **Multi-Agent** | **2-4 competing agents** |
| **Affordance Contention** | **Limited resources, blocking** |
| **Social Penalties** | **Proximity costs to other agents** |
| **Reward Shaping** | Sparse + social |
| **Curriculum** | Adversarial |

### Planned Changes

**Added:**
- Multiple agents (2-4) in same environment
- Affordance blocking (only 1 agent can use Bed at a time)
- Social proximity penalty (agents prefer personal space)
- Theory of mind (predict other agent behavior)
- Emergent competition/cooperation

### Teaching Value

- **Game theory**: Nash equilibria, dominant strategies
- **Multi-agent RL**: How cooperation emerges despite selfish rewards
- **Social intelligence**: Predicting and responding to others
- **Resource competition**: Territorial behavior, scheduling conflicts

---

## Level 6: Emergent Communication (Future)

**Status:** 🎯 Planned (Phase 7)  
**Config:** `configs/level_6_communication.yaml` (not yet created)  
**Network:** RecurrentSpatialQNetwork + Communication Channel

### Planned Capabilities

| Feature | Setting |
|---------|---------|
| **Observability** | Partial |
| **Memory** | LSTM |
| **Temporal Mechanics** | Enabled |
| **Multi-tick Interactions** | Enabled |
| **Multi-Agent** | 2-4 agents (family units) |
| **Affordance Contention** | Blocking |
| **Social Penalties** | Proximity costs |
| **Communication** | **Discrete symbol channel (10-20 tokens)** |
| **Reward Shaping** | Sparse + social + communication |
| **Curriculum** | Adversarial |

### Planned Changes

**Added:**
- Communication action (broadcast 1 token to nearby agents)
- Observation includes recent messages from others
- No predefined meaning (emergent protocol)
- Family units (shared reward bonus for coordination)

### Teaching Value

- **Emergent language**: Symbols ground in shared experience
- **Communication protocols**: How meaning emerges
- **Coordination games**: Family members help each other
- **Language grounding**: Symbols mean something in the world

---

## Summary Table

| Level | Observability | Memory | Temporal | Multi-Agent | Communication | Config |
|-------|--------------|--------|----------|-------------|---------------|--------|
| **L1** | Full | No (MLP) | No | Single | No | `level_1_full_observability.yaml` |
| **L2** | Partial | Yes (LSTM) | No | Single | No | `level_2_pomdp.yaml` |
| **L3** | Partial | Yes (LSTM) | **Yes** | Single | No | `level_3_temporal.yaml` |
| **L4** | Partial | Hierarchical | Yes | Single | No | `level_4_multi_zone.yaml` (future) |
| **L5** | Partial | LSTM + ToM | Yes | **Multi** | No | `level_5_multi_agent.yaml` (future) |
| **L6** | Partial | LSTM | Yes | Multi | **Yes** | `level_6_communication.yaml` (future) |

---

## Pedagogical Progression

**Week 1-2:** Level 1 (Full Observability)
- Students understand the basics: meters, affordances, curriculum
- Learn about reward shaping pitfalls (proximity hacking)
- See adversarial curriculum in action

**Week 3-4:** Level 2 (POMDP)
- Introduce partial observability and memory requirements
- Teach LSTM architectures and target networks
- Explore exploration vs exploitation tradeoffs

**Week 5-6:** Level 3 (Temporal)
- Add time-based constraints and planning
- Teach temporal credit assignment
- Multi-tick interactions and opportunity cost

**Week 7+:** Level 4-6 (Advanced)
- Hierarchical RL, multi-agent systems, emergent communication
- Graduate-level RL concepts in accessible environment

---

## Configuration Naming Convention

**Format:** `level_<number>_<primary_feature>.yaml`

**Examples:**
- `level_1_full_observability.yaml` - Emphasizes complete information
- `level_2_pomdp.yaml` - Emphasizes partial observability (POMDP)
- `level_3_temporal.yaml` - Emphasizes time-based mechanics
- `level_4_multi_zone.yaml` - Emphasizes spatial hierarchy
- `level_5_multi_agent.yaml` - Emphasizes agent interaction
- `level_6_communication.yaml` - Emphasizes emergent language

**Deprecated naming:**
- ~~`townlet_level_1_5.yaml`~~ → `level_1_full_observability.yaml`
- ~~`townlet_level_2_pomdp.yaml`~~ → `level_2_pomdp.yaml`
- ~~`townlet_level_2_5_temporal.yaml`~~ → `level_3_temporal.yaml`

---

## Testing Each Level

### Integration Tests (Fast - ~20 minutes total)

Automated tests validate each level works end-to-end:

```bash
# Run all integration tests (Level 1, 2, 3)
uv run pytest tests/test_integration/ -v

# Run specific level test
uv run pytest tests/test_integration/ -k "test_level_1" -v
uv run pytest tests/test_integration/ -k "test_level_2" -v
uv run pytest tests/test_integration/ -k "test_level_3" -v

# Quick config validation (<1 second)
uv run pytest tests/test_integration/ -k "test_all_configs_valid" -v
```

See `docs/INTEGRATION_TESTS.md` for complete documentation.

### Production Training (Full Validation)

Each level should be validated independently:

```bash
# Level 1: Full observability baseline (5K episodes)
python scripts/start_training_run.py L1_baseline configs/level_1_full_observability.yaml

# Level 2: POMDP with LSTM (10K episodes)
python scripts/start_training_run.py L2_pomdp configs/level_2_pomdp.yaml

# Level 3: Temporal mechanics (10K episodes)
python scripts/start_training_run.py L3_temporal configs/level_3_temporal.yaml
```

**Success Criteria:**
- Agent survives progressively longer episodes
- Learning curves show clear improvement
- Emergent behaviors match level objectives (L3: time-dependent strategies)

---

## Implementation Status

- ✅ **Level 1:** Fully implemented and tested
- ✅ **Level 2:** Fully implemented, target network added (ACTION #9)
- ✅ **Level 3:** Fully implemented, temporal mechanics validated
- 🎯 **Level 4:** Planned for Phase 6 (multi-zone environment)
- 🎯 **Level 5:** Planned for Phase 6 (multi-agent competition)
- 🎯 **Level 6:** Planned for Phase 7 (emergent communication)

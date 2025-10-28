# Hamlet: Multi-Agent Strategic Survival Environment
## Architecture Design Document v1.0

> **Pedagogical Mission**: Trick students into learning deep reinforcement learning, game theory, and neural architecture design by making them think they're just playing The Sims.

---

## Table of Contents

1. [Vision & Motivation](#vision--motivation)
2. [Progressive Complexity Levels](#progressive-complexity-levels)
3. [Core Architecture](#core-architecture)
4. [Partial Observability Design](#partial-observability-design)
5. [Multi-Zone Hierarchical Architecture](#multi-zone-hierarchical-architecture)
6. [Multi-Agent Game Theory](#multi-agent-game-theory)
7. [Emergent Behaviors](#emergent-behaviors)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Research Opportunities](#research-opportunities)

---

## Vision & Motivation

### The Problem with Teaching RL

Traditional RL courses start with:
- Markov Decision Processes (intimidating math)
- Bellman equations (more math)
- CartPole (boring)
- Gridworld (too simple)

**Students think**: "This is abstract, mathematical, and disconnected from real AI."

### The Hamlet Solution

Start with a relatable game:
- "Your agent needs to survive by managing energy, hygiene, and money"
- "They need to go to work, eat, sleep, and stay clean"
- "Oh, there are other agents competing for the same job..."

**Students think**: "This is just a game! I understand games!"

Then reveal:
- **Partial observability** → POMDP theory
- **Hierarchical planning** → Options framework, temporal abstraction
- **Multi-agent competition** → Game theory, Nash equilibria
- **Opponent modeling** → Theory of mind, belief states
- **Emergent strategy** → Self-play, evolutionary dynamics

**Students realize**: "Wait, I just learned graduate-level RL by playing a game."

---

## Progressive Complexity Levels

### Level 1: Single Agent, Full Observability (CURRENT)
**Status**: ✅ Implemented (baseline working)

**Environment**:
- 8×8 grid
- 4 affordances (Bed, Shower, Fridge, Job)
- 4 meters (Energy, Hygiene, Satiation, Money)
- Agent sees entire grid

**Architecture**:
- Baseline: Simple MLP (26K params)
- Advanced: Dueling DQN, Spatial CNN

**Learning Objectives**:
- Basic RL loop (observe, act, reward)
- Q-learning fundamentals
- Reward shaping
- Exploration vs exploitation

**Performance**:
- Baseline achieves +52 reward at episode 900
- Agent survives 280+ steps
- Learns basic survival cycle

**Teaching Moment**: "See? Deep learning works! Now let's make it harder..."

---

### Level 2: Partial Observability (POMDP)
**Status**: 🔨 Designed, not implemented

**Key Changes**:
- Agent sees only 5×5 local window
- Must build mental map through exploration
- Must remember where affordances are
- Can forget and get lost

**New Challenges**:
- **Exploration**: How to efficiently discover the world
- **Memory**: Where did I see that bed?
- **Planning**: Navigate to remembered locations
- **Uncertainty**: Should I go to the bed I remember, or search for closer one?

**Architecture Requirements**:
```python
class RecurrentSpatialQNetwork:
    - Vision Encoder: CNN for 5×5 window → 128 features
    - Position Encoder: (x, y) → 32 features
    - Meter Encoder: 4 meters → 32 features
    - LSTM: Maintain memory across timesteps (256 hidden)
    - Spatial Memory Module: Build internal map (20×20×64 memory grid)
    - Q-Head: Combined features → 5 action Q-values
```

**Key Components**:

1. **Recurrent Networks (LSTM/GRU)**
   - Maintains hidden state across timesteps
   - Remembers past observations
   - ~500K parameters

2. **Spatial Memory Module**
   - Explicit 20×20×64 memory grid
   - Write head: Updates memory when observing locations
   - Read head: Queries memory for decision-making
   - Attention: Focuses on relevant memories based on needs

3. **Exploration Bonuses**
   - Curiosity-driven exploration
   - Reward = 1/√(visit_count + 1)
   - Encourages visiting new locations

**Training Modifications**:
```python
class EpisodeReplayBuffer:
    """Store entire episode SEQUENCES, not just (s,a,r,s') tuples"""

    def sample_sequence(self, length=20):
        """Sample temporal sequences for LSTM training"""
        # Maintains temporal coherence for recurrent training
```

**Expected Performance**:
- 2000-3000 episodes to learn (vs 800-1000 full obs)
- Peak reward: +30 to +50 (vs +52 full obs)
- 30-40% of time spent exploring
- Realistic cognitive behavior (exploration, memory, mistakes)

**Learning Objectives**:
- POMDPs vs MDPs
- Recurrent neural networks
- Memory architectures
- Exploration strategies (epsilon-greedy, curiosity)
- Credit assignment over long sequences

**Teaching Moment**: "Your agent is now blind! How does it learn to navigate? This is how robots work in the real world."

---

### Level 3: Multi-Zone Environment (Hierarchical RL)
**Status**: 🎯 Designed, future work

**Environment**:
- **4 zones**: Home, Industrial, Commercial, Public
- **20×20 grid per zone** (80×80 total world)
- **14+ affordances** distributed across zones
- **Transportation**: Walk, Bus, Taxi between zones
- **Time mechanics**: Day/night cycle, schedules

**Zone Layout**:
```
HOME ZONE (20×20)
├─ Bed (sleep: +50 energy, +5 hygiene, -$5)
├─ Shower (+40 hygiene, -3 energy, -$3)
├─ Fridge (+45 satiation, +5 energy, -$4)
├─ TV (+20 stress relief, -5 energy, free)
└─ Closet (change clothes: +10 hygiene, free)

INDUSTRIAL ZONE (20×20)
├─ Factory (8hr shift: +$30, -15 energy, -10 hygiene)
├─ Warehouse (4hr shift: +$15, -8 energy, -5 hygiene)
└─ Office (flexible: +$20, -5 energy, -3 hygiene)

COMMERCIAL ZONE (20×20)
├─ Restaurant (+50 satiation, +20 social, -$12)
├─ Gym (+50 fitness, -10 energy, -$8)
├─ Cafe (+40 social, +15 stress relief, -$5)
├─ Shop (buy clothes: +20 hygiene, -$15)
└─ Bar (+40 social, +20 stress, -$10, -15 energy)

PUBLIC ZONE (20×20)
├─ Park (free, +30 stress relief, +15 social)
├─ Library (free, +30 knowledge, +10 stress relief)
├─ Hospital (+100 health, emergency: -$50)
├─ Bus Station (transportation hub)
└─ Community Center (+25 social, free)
```

**New Meters**:
- **Social**: Depletes with isolation, critical for mental health
- **Fitness**: Depletes over time, affects energy depletion rate
- **Stress**: Increases with work, reduces with relaxation
- **Health**: Critical meter, can cause death if zero
- **Knowledge**: Unlocks better-paying jobs

**Hierarchical Architecture**:

```python
class HierarchicalAgent:
    """Three-level decision hierarchy"""

    # LEVEL 1: Strategic (Which zone?)
    def zone_policy(self, meters, time, current_zone):
        """
        Decides: Should I stay in current zone or go elsewhere?

        Examples:
        - Energy < 20, time = 10pm → GO_HOME (bed)
        - Money < 10, time = 8am → GO_INDUSTRIAL (work)
        - Stress > 80 → GO_PUBLIC (park)

        Output: Target zone
        """
        pass

    # LEVEL 2: Tactical (How to get there?)
    def transportation_policy(self, current_zone, target_zone, money, time):
        """
        Decides: Walk, bus, or taxi?

        Trade-offs:
        - Walk: Free, slow (30 steps)
        - Bus: $3, medium (15 steps), must wait for schedule
        - Taxi: $10, instant (1 step)

        Output: Transportation method
        """
        pass

    # LEVEL 3: Operational (Navigate within zone)
    def intra_zone_policy(self, zone_map, meters, target_affordance):
        """
        Decides: Up, down, left, right, or interact?

        Uses spatial CNN with partial observability
        Maintains memory of zone layout

        Output: Movement action
        """
        pass
```

**Architecture Components**:

1. **Zone-Level Policy** (Strategic)
   ```python
   Input: Agent state (meters + time + zone)
   Layers:
   - State encoder: 8 → 128 → 64
   - Zone embeddings: 4 zones × 64 dims (learned)
   - Zone value net: 128 → 1 per zone

   Output: Q(zone | state) for each zone
   ```

2. **Transportation Policy** (Tactical)
   ```python
   Input: (from_zone, to_zone, money, time)

   Zone graph: 4×4 learned adjacency matrix
   - Encodes: Which zones connect, travel time, cost

   Path value net: zones + state → value

   Output: Transport action (walk/bus/taxi)
   ```

3. **Intra-Zone Policy** (Operational)
   ```python
   Same as Level 2 (Recurrent Spatial Q-Network)
   But now operates within single zone

   Multi-scale spatial processing:
   - 20×20 → 10×10 → 5×5 (strided convolutions)
   - Attention over spatial features
   - Global average pooling → 128 features
   ```

**Temporal Abstraction**:
- High-level actions take multiple steps
- "Go to Industrial zone" = meta-action lasting 15-30 steps
- Low-level policy executes during high-level action
- Hierarchical RL (Options framework)

**Expected Performance**:
- 5000-7000 episodes to learn
- Peak reward: +200 to +500
- 50-60% time spent in strategic planning
- Complex multi-step plans (work → gym → restaurant → home)

**Learning Objectives**:
- Hierarchical RL (Options, HAMs)
- Temporal abstraction
- Multi-scale decision making
- Graph neural networks (zone connectivity)
- Curriculum learning (start simple, add zones)

**Teaching Moment**: "Your agent now lives in a city! How does it plan a day? This is how autonomous vehicles and delivery robots work."

---

### Level 4: Multi-Agent Competition (Game Theory)
**Status**: 🎯 Designed, flagship feature

**Environment**:
- 2-10 agents in same world
- Shared resources with **limited capacity**
  - Job: Only 1 agent can work at a time
  - Fridge: Finite food (replenishes over time)
  - Bus: Seats limited, first-come-first-served
- Partial observability: Can only see agents in vision radius
- Information asymmetry: Don't know opponent meters/intentions

**The Strategic Scenario** (Motivating Example):

```
Setup:
- Agent A: Lives 3 cells from Factory
- Agent B: Lives 10 cells from Factory
- Both need money, only 1 job slot available
- Time: 6:00 AM, Factory opens at 8:00 AM

Agent A's Strategy (Information Advantage):
  6:00 - Wake up, can see path to factory
  6:30 - Sees Agent B running toward factory (in vision!)
  6:35 - "Oh no, they're racing me!"
  6:36 - Leaves immediately, shorter path = wins race

Agent B's Counter-Strategy (Information Denial):
  2:00 AM - Wakes up early (A can't see B's preparation)
  2:00-6:00 - Eats repeatedly at fridge (energy → 100)
  6:00 - Leaves for factory with max energy
  6:30 - Arrives BEFORE Agent A even wakes up
  WIN: Agent B beats A through strategic timing!

Game Theory:
- Nash Equilibrium: B leaves at 2am, A can't counter
- Information game: B exploits A's limited vision
- Temporal reasoning: 4-hour advance planning
- Resource management: Energy prep for long journey
```

**Multi-Agent Architecture**:

```python
class CompetitiveMultiAgentDQN:
    """
    Four-component architecture for strategic play
    """

    # COMPONENT 1: Self Model
    def self_model(self, observation, hidden_state):
        """
        Standard POMDP architecture
        Process my local observations, maintain my memory

        Same as Level 2 RecurrentSpatialQNetwork
        """
        pass

    # COMPONENT 2: Opponent Model
    def opponent_model(self, opponent_id, observation, belief_state):
        """
        Theory of Mind: Model other agent's state and intentions

        Maintains belief for EACH opponent:
        - Estimated position (if not visible, predict)
        - Estimated meters (infer from behavior)
        - Estimated intent (going_to_job, going_to_bed, etc.)
        - Predicted next actions

        Architecture:
        - Belief encoder: LSTM (128 → 256 hidden)
        - State predictor: 256 → 128 (forward model)
        - Intent classifier: 256 → 10 intents (softmax)
        - Action predictor: 256 → 5 actions (what will they do?)
        """
        pass

    # COMPONENT 3: Strategic Reasoner
    def strategic_reasoner(self, my_state, opponent_beliefs, resource):
        """
        Game-theoretic reasoning about competition

        Computes:
        - Win probability (will I reach resource first?)
        - Counter-strategies (how to beat opponent)
        - Information value (should I move to see them?)
        - Deception value (should I hide my intentions?)

        Returns: Best strategic action
        """
        pass

    # COMPONENT 4: Combined Q-Network
    def forward(self, obs, hidden, beliefs):
        """
        Integrate all components for final Q-values

        Pipeline:
        1. Process my observation → my_features (256)
        2. Update opponent beliefs → opponent_features (256)
        3. Strategic reasoning → strategy_features (128)
        4. Combine: [my, opp, strat] → 640 dims
        5. Final Q-network: 640 → 512 → 256 → 5 actions

        Output: Strategic Q-values considering competition
        """
        pass
```

**Belief State Representation**:

```python
class OpponentBelief:
    """What I think about opponent"""

    # Observable when in vision
    last_seen_position: Tuple[int, int]
    last_seen_time: int

    # Estimated (predicted when not visible)
    estimated_position: Tuple[int, int]  # Where I think they are now
    position_uncertainty: float  # How confident am I?

    estimated_meters: Dict[str, float]  # Inferred from behavior

    # Predicted intentions
    intent_distribution: Dict[str, float]  # {going_to_job: 0.7, ...}
    predicted_trajectory: List[Tuple[int, int]]  # Where they're heading

    # Strategic state
    threat_level: float  # Are they competing with me?
    visibility_window: List[Tuple[int, int]]  # Where they can see
    predicted_wake_time: float  # When will they start moving?
```

**Training: Self-Play Population**

```python
class SelfPlayTrainer:
    """
    Train through competitive self-play

    Population-based training:
    - Maintain population of 10-20 agents
    - Each episode: Sample 2-10 agents randomly
    - Agents compete for resources
    - Winners propagate strategies to population
    - Losers adapt or are replaced

    This creates evolutionary pressure for:
    - Strategic diversity
    - Robust strategies (work against many opponents)
    - Adaptive behavior (counter-strategies)
    """

    def train_episode(self):
        # Sample N agents from population
        active_agents = random.sample(self.population, k=num_agents)

        # Reset environment with these agents
        obs = self.env.reset(agents=active_agents)

        # Run competitive episode
        while not done:
            # Each agent decides action based on:
            # - Own partial observation
            # - Beliefs about other agents (in vision)
            # - Strategic reasoning

            actions = []
            for agent in active_agents:
                agent_obs = self._get_partial_obs(obs, agent.id)
                action, beliefs = agent.select_strategic_action(
                    agent_obs,
                    opponent_beliefs=agent.beliefs
                )
                actions.append(action)

            # Execute all actions simultaneously
            next_obs, rewards, dones, info = self.env.step(actions)

            # Assign rewards (competitive outcomes)
            for i, agent in enumerate(active_agents):
                reward = rewards[i]

                # Bonus for winning competitions
                if info[f"agent_{i}_won_job"]:
                    reward += 50.0

                # Bonus for strategic victory
                if info[f"agent_{i}_strategic_win"]:
                    reward += 25.0  # Won through clever timing

                agent.store_experience(...)

            obs = next_obs

        # Update population (winners teach losers)
        self._update_population(active_agents, performance)
```

**Competitive Reward Structure**:

```python
def calculate_competitive_reward(agent, outcome):
    """
    Reward both survival AND competitive success
    """
    reward = 0.0

    # Base survival (existing shaped rewards)
    reward += shaped_survival_reward(agent.meters)

    # Competitive outcomes
    if outcome["competed_for_resource"]:
        if outcome["won_competition"]:
            reward += 50.0  # Big bonus for winning

            # Extra bonus for strategic victory
            if outcome["strategic_victory"]:
                # e.g., preemptive 2am move, perfect timing
                reward += 25.0
        else:
            reward -= 10.0  # Penalty for losing (wasted effort)

    # Information gathering
    if outcome["observed_opponent"]:
        reward += 5.0  # Reward gaining information

    # Stealth/deception
    if outcome["moved_while_unobserved"]:
        reward += 2.0  # Reward avoiding detection

    # Prediction accuracy
    if agent.made_prediction:
        accuracy = outcome["prediction_accuracy"]
        reward += 3.0 * accuracy  # Reward accurate modeling

    # Social learning
    if outcome["learned_from_observation"]:
        reward += 1.0  # Reward inferring opponent strategy

    return reward
```

**Expected Performance**:
- 10,000-15,000 episodes for competitive equilibrium
- Diverse strategies emerge in population
- Nash equilibria discovered through self-play
- Rich strategic behaviors (see Emergent Behaviors section)

**Learning Objectives**:
- Game theory (Nash equilibria, Pareto optimality)
- Multi-agent RL (independent learners, centralized critics)
- Theory of mind (recursive reasoning)
- Self-play training
- Population-based training
- Evolutionary dynamics

**Teaching Moment**: "Your agents are now playing mind games with each other! This is how AlphaGo and OpenAI Five learned to beat humans."

---

## Core Architecture

### State Representation Evolution

**Level 1 (Full Obs)**: Flat vector
```python
state = [pos_x, pos_y, energy, hygiene, satiation, money, grid_flattened]
# Dimension: 2 + 4 + 64 = 70
```

**Level 2 (Partial Obs)**: Local window + memory
```python
state = {
    "local_grid": tensor(5, 5, 15),  # Multi-channel local vision
    "position": tensor(2),            # Global position (GPS)
    "meters": tensor(6),              # Internal state
    "memory": tensor(20, 20, 64),    # Spatial memory grid
}
```

**Level 3 (Multi-Zone)**: Hierarchical + zone-aware
```python
state = {
    "current_zone": str,                    # "home", "industrial", etc.
    "zone_map": tensor(20, 20, 15),        # Current zone (full detail)
    "adjacent_zones": Dict[str, Summary],   # Coarse summaries
    "position": tensor(2),                  # Position in current zone
    "meters": tensor(8),                    # More meters
    "time": int,                            # Time of day
    "in_activity": bool,                    # In middle of work shift?
}
```

**Level 4 (Multi-Agent)**: Add beliefs
```python
state = {
    # All from Level 3, plus:
    "opponent_beliefs": {
        opponent_id: {
            "estimated_position": tensor(2),
            "estimated_meters": tensor(8),
            "intent": tensor(10),  # Intent distribution
            "visible": bool,
        }
    },
    "contested_resources": List[str],  # Which resources are competitive
}
```

---

### Network Architecture Progression

**Level 1: Simple MLP**
```
Input (70) → [128, 128] → Output (5)
Parameters: ~26K
Use case: Baseline, full observability
Performance: +52 reward
```

**Level 2: Recurrent Spatial**
```
Vision CNN: 5×5×15 → [32, 64] → 128 features
Position MLP: 2 → [32, 32] → 32 features
Meter MLP: 6 → [64, 32] → 32 features
Combined: 192 → LSTM(256, layers=2) → Q-head → 5

Spatial Memory Module (optional):
- Memory grid: 20×20×64 learned embeddings
- Write head: Update memory from observations
- Read head: Query memory with attention
- Attention: 8 heads, 64 dims

Parameters: ~500K (with memory: ~800K)
Use case: Partial observability
Performance: +30-50 reward
```

**Level 3: Hierarchical**
```
Zone Policy:
  State → [128, 64] → Zone embeddings → Value per zone
  Parameters: ~50K

Transport Policy:
  (from, to, state) → Zone graph → Path value
  Parameters: ~30K

Intra-Zone Policy:
  Same as Level 2 Recurrent Spatial
  Parameters: ~500K

Total: ~600K parameters
Use case: Multi-zone navigation
Performance: +200-500 reward
```

**Level 4: Multi-Agent Competitive**
```
Self Model: ~500K (Level 2/3 architecture)

Opponent Models (×10):
  Each: Belief LSTM + Predictors
  Parameters: ~200K per opponent
  Total: ~2M

Strategic Reasoner:
  Combined features → Game theory module
  Parameters: ~300K

Final Q-Network:
  Integrated → Strategic Q-values
  Parameters: ~500K

Total: ~3.3M parameters
Use case: Competitive multi-agent
Performance: Diverse (depends on opponents)
```

---

### Training Hyperparameters by Level

**Level 1 (Current)**:
```yaml
learning_rate: 0.00025  # Atari DQN standard
gamma: 0.99
epsilon: 1.0 → 0.01 (decay 0.995)
batch_size: 64
replay_buffer: 10,000
episodes: 1,000
target_update: every 100 episodes
```

**Level 2 (Partial Obs)**:
```yaml
learning_rate: 0.0001  # Slower for recurrent networks
gamma: 0.99
epsilon: 1.0 → 0.05 (decay 0.997)  # Keep some exploration
batch_size: 32  # Smaller for sequence training
replay_buffer: 1,000 episodes (sequences)
sequence_length: 20  # For LSTM training
episodes: 3,000
target_update: every 200 episodes
exploration_bonus: 0.5  # Curiosity weight
```

**Level 3 (Hierarchical)**:
```yaml
# Separate LR for each hierarchy level
zone_lr: 0.0001
transport_lr: 0.0002
intra_zone_lr: 0.00005

gamma: 0.99
epsilon: 1.0 → 0.05 (decay 0.998)
batch_size: 32
replay_buffer: 2,000 episodes
episodes: 7,000
target_update: every 300 episodes

# Curriculum learning stages
curriculum:
  - zones: 1, episodes: 1000
  - zones: 2, episodes: 2000
  - zones: 4, episodes: 4000
```

**Level 4 (Multi-Agent)**:
```yaml
learning_rate: 0.00005  # Very slow, complex optimization
gamma: 0.99
epsilon: 1.0 → 0.1 (decay 0.999)  # Keep exploration high
batch_size: 16  # Small batches
replay_buffer: 500 episodes per agent

# Self-play settings
population_size: 20
agents_per_episode: 2-6 (random)
episodes: 15,000

# Opponent model training
opponent_lr: 0.0001
belief_update_rate: 0.1

# Population update
tournament_frequency: 100 episodes
replacement_rate: 0.2  # Replace worst 20%
```

---

## Partial Observability Design

### Vision Radius Design Choices

**Option A: Fixed Circular Radius**
```python
vision_radius = 2  # See 5×5 window
# Pros: Simple, fast
# Cons: Symmetric (unrealistic)
```

**Option B: Directional Vision (Cone)**
```python
vision = {
    "forward": 4 cells,  # Facing direction
    "sides": 2 cells,
    "back": 1 cell,
}
# Pros: More realistic, directional awareness matters
# Cons: Need to track agent orientation
```

**Recommendation**: Start with Option A, upgrade to B if interesting behaviors emerge.

### Memory Architecture Options

**Option 1: Pure LSTM (Implicit Memory)**
- Memory encoded in hidden state
- No explicit spatial structure
- Pros: Simple, proven
- Cons: Black box, hard to interpret

**Option 2: Explicit Spatial Memory (Recommended)**
- 20×20×64 grid of learned embeddings
- Write/read heads with attention
- Pros: Interpretable, structured
- Cons: More complex, more parameters

**Option 3: Hybrid (Best of Both)**
- LSTM for temporal memory
- Spatial grid for location memory
- Use both in decision-making

### Exploration Strategy

**Epsilon-Greedy** (baseline)
```python
if random.random() < epsilon:
    return random_action()
else:
    return argmax_Q(state)
```

**Curiosity-Driven** (recommended)
```python
intrinsic_reward = 1.0 / sqrt(visit_count[position] + 1)
total_reward = extrinsic_reward + beta * intrinsic_reward
```

**Upper Confidence Bound**
```python
Q_explore = Q_value + c * sqrt(log(N) / n_visits)
# Balance exploitation and exploration optimally
```

---

## Multi-Zone Hierarchical Architecture

### Zone Design Philosophy

**Principle**: Each zone has distinct function and atmosphere

**Home Zone**: Safety, recovery, privacy
- Affordances: Low-cost, high-restoration
- No competition (personal space)
- Fast travel to adjacent zones

**Industrial Zone**: Work, money generation
- Affordances: High reward, high cost (time/energy)
- Competitive (limited job slots)
- Far from Home (requires planning)

**Commercial Zone**: Spending, social, entertainment
- Affordances: Medium cost, social benefits
- Moderate competition
- Central location (hub)

**Public Zone**: Free resources, exploration
- Affordances: Free or low-cost
- No competition (public goods)
- Educational/social focus

### Transportation Economics

```python
class TransportOption:
    WALK = {
        "cost": 0,
        "time": 30 steps,
        "energy": -10,
        "availability": "always",
    }

    BUS = {
        "cost": 3,
        "time": 15 steps,
        "energy": -2,
        "availability": "schedule",  # Every 50 steps
        "capacity": 5,  # First-come-first-served
    }

    TAXI = {
        "cost": 10,
        "time": 5 steps,
        "energy": 0,
        "availability": "always",
        "capacity": 1,
    }
```

**Strategic implications**:
- Bus requires prediction (when will it arrive?)
- Taxi is emergency option (expensive but instant)
- Walking is default (but slow, energy cost)

### Hierarchical Action Decomposition

```
High-Level Goal: "Earn money"
  ↓
Strategic Decision: "Go to Industrial Zone"
  ↓
Tactical Decision: "Take bus (cheap, reasonable time)"
  ↓
Operational Sequence:
  1. Walk to bus stop (8 steps)
  2. Wait for bus (12 steps)
  3. Ride bus (15 steps)
  4. Walk to factory (5 steps)
  5. Interact with factory
  ↓
Total: 40 steps for complex plan
```

**Key insight**: Agent must commit to multi-step plans, can't change mind mid-bus-ride.

### Temporal Abstraction (Options Framework)

```python
class Option:
    """Semi-Markov Decision Process option"""

    def initiation_set(self, state):
        """Can this option be started in current state?"""
        pass

    def policy(self, state):
        """What actions to take while executing option"""
        pass

    def termination_condition(self, state):
        """When does option end?"""
        pass

# Example: "Go to Industrial Zone" option
class GoToIndustrialOption(Option):
    def initiation_set(self, state):
        return state.current_zone != "industrial"

    def policy(self, state):
        # Sub-policy: Navigate to zone boundary
        # Then: Take appropriate transport
        # Then: Navigate within new zone
        return self.hierarchical_policy(state)

    def termination_condition(self, state):
        return state.current_zone == "industrial"
```

---

## Multi-Agent Game Theory

### Competitive Resource Types

**Type 1: Exclusive Access** (only 1 agent at a time)
- Job: Winner works, loser must wait or find alternative
- Taxi: First to request gets it
- Critical: High stakes, binary outcomes

**Type 2: Capacity-Limited** (K agents allowed)
- Bus: 5 seats, first-come-first-served
- Restaurant: 10 tables, must wait if full
- Medium stakes, partial competition

**Type 3: Degrading Quality** (all can access, but quality decreases)
- Fridge: Food depletes, replenishes slowly
- Park: Crowded = less stress relief
- Low stakes, indirect competition

### Information States

**Perfect Information**: Both agents see each other
```
Leads to: Direct competition, reaction strategies
Example: Both visible near job, race ensues
```

**Asymmetric Information**: A sees B, but B doesn't see A
```
Leads to: Stalking, ambush strategies
Example: A follows B at distance, waits for opportunity
```

**No Information**: Neither sees the other
```
Leads to: Prediction, implicit coordination
Example: B predicts A's schedule, preempts
```

### Theory of Mind Levels

**Level 0**: Reactive (no opponent modeling)
```
"I need money, I go to job"
```

**Level 1**: First-order beliefs
```
"I think opponent is going to job"
```

**Level 2**: Second-order beliefs
```
"I think opponent thinks I'm going to job, so they won't go"
```

**Level 3**: Recursive reasoning
```
"I think opponent thinks I think they're going to job,
 so I should go early to counter their counter-strategy"
```

**Implementation**: Limit to Level 1-2 initially (Level 3+ is computationally expensive and may not emerge naturally).

### Emergent Social Behaviors (Expected)

**Implicit Coordination** (Nash Equilibrium)
```
Two agents learn to alternate job shifts without communication:
- Agent A: Takes morning shift (6am-12pm)
- Agent B: Takes afternoon shift (12pm-6pm)
- Both earn money, no conflict
- Emerges from repeated competition + punishment for conflicts
```

**Territorial Behavior**
```
Agents develop "home zones" and avoid each other's territory
- Agent A: Prefers north side of map
- Agent B: Prefers south side
- Reduces competition, increases efficiency
```

**Deceptive Feinting**
```
Agent walks toward job (visible to opponent)
Opponent races ahead to beat agent
Agent turns around, goes to bed instead (was faking)
Agent successfully deceived opponent!
```

**Information Hiding**
```
Agent wakes up very early (opponent asleep)
Moves while unobserved
Reaches resource first
Opponent confused: "How did they get there?"
```

**Reputation Formation**
```
After many episodes, agents learn opponent tendencies:
- "Agent A always goes to job at 7am" → Predictable
- "Agent B is unpredictable" → Cautious approach
- Adjusts strategy based on opponent identity
```

**Emergence of Communication** (if implemented)
```
Agents could learn to signal intentions:
- Move left-right-left = "I'm going to job"
- Move in circles = "I'm yielding this resource"
- Learned through RL, not programmed
```

---

## Emergent Behaviors

### Strategic Behaviors We Expect to See

#### 1. **Preemptive Timing** (The 2am Strategy)
```
Scenario: Agent B lives far from job, Agent A lives close
B's learned strategy:
  - Predict A's wake time (6am)
  - Calculate travel time (30 steps)
  - Wake at 2am, eat until energy=100
  - Leave at 5:30am, arrive at 6:00am
  - Beat A who hasn't even woken yet

Mathematical: B solves optimization
  argmin(wake_time) subject to:
    energy(wake_time + prep_time + travel_time) > 0
    arrival_time < opponent_wake_time + opponent_travel_time
```

#### 2. **Information Warfare** (Vision Exploitation)
```
Scenario: Agent A has vision of path to job
A's learned strategy:
  - Monitor path continuously
  - If B enters vision heading to job:
    - Calculate: Can I beat them? (distance comparison)
    - If yes: Race immediately
    - If no: Abandon, go to alternative (commercial zone)
  - If no B in vision:
    - Wait until last minute (save energy)
    - Sprint to job

Information value: Seeing opponent is worth ~20 reward
```

#### 3. **Bluffing and Deception**
```
Scenario: Agent A at low energy, sees Agent B heading to job
A's learned strategy:
  - Start walking toward job (B can see this)
  - B races ahead to beat A
  - A turns around, goes to bed (was bluffing)
  - A recovers energy, B wasted effort at job

Requirements for emergence:
  - Opponent modeling (B predicts A's intent)
  - Recursive reasoning (A models B's model of A)
  - Reward for deception (A gains by misleading B)
```

#### 4. **Adaptive Scheduling**
```
Scenario: Repeated interactions reveal patterns
Agent B learns:
  - "Agent A goes to job at 7am (80% probability)"
  - "Agent A goes to bed at 10pm"

B's counter-strategy:
  - Go to job at 6:30am (beat A's schedule)
  - If A adapts to 6am, B shifts to 5:30am
  - Arms race of earlier wake times!

Equilibrium: Both agents at 2-3am (too early = energy waste)
```

#### 5. **Zone Specialization**
```
Scenario: 4 zones, 2 agents, repeated episodes
Emergent division of labor:
  - Agent A: Specializes in Industrial zone
    - Knows factory layout perfectly
    - Optimized path, minimal wasted steps
  - Agent B: Specializes in Commercial zone
    - Frequents restaurant, gym, cafe
    - Develops different meter balance

Why: Reduces inter-agent conflict, increases efficiency
```

#### 6. **Reputation and Adaptation**
```
Scenario: Self-play training, population of agents
Agent learns opponent "personalities":
  - "This opponent is aggressive" → Avoid conflicts
  - "This opponent is predictable" → Exploit pattern
  - "This opponent is random" → Robust strategy

Implementation:
  - Opponent ID embedding in network
  - Condition policy on opponent identity
  - Different strategies per opponent
```

#### 7. **Implicit Signaling** (Proto-Communication)
```
Scenario: Agents want to coordinate without communication
Emergent signals:
  - Moving left-right-left = "I'm going to job"
  - Circling affordance = "I yield this to you"
  - Rapid movement = "Emergency, need this resource"

How it emerges:
  - Agents observe patterns in opponent behavior
  - Learn to predict intent from movement
  - Accidentally produce predictable patterns
  - Other agents exploit these patterns
  - Patterns become "language"
```

---

## Implementation Roadmap

### Phase 1: Partial Observability (Week 1-2)
**Goals**:
- ✅ Implement local vision window (5×5)
- ✅ Add LSTM for memory
- ✅ Implement exploration bonuses
- ✅ Train and validate learning

**Deliverables**:
- `RecurrentSpatialQNetwork` class
- `EpisodeReplayBuffer` for sequences
- `ExplorationBonus` module
- Training results showing exploration behavior

**Success Metrics**:
- Agent discovers all affordances in <500 episodes
- Builds usable mental map
- Peak reward: +30 to +50

---

### Phase 2: Spatial Memory Module (Week 3)
**Goals**:
- ✅ Implement explicit spatial memory grid
- ✅ Write/read heads with attention
- ✅ Visualization of learned memory

**Deliverables**:
- `SpatialMemoryModule` class
- Visualization tools (heatmaps of memory)
- Ablation study: LSTM vs Memory vs Both

**Success Metrics**:
- Visualizations show clear affordance memories
- Agent navigates to remembered locations
- Improved performance over pure LSTM

---

### Phase 3: Multi-Zone Environment (Week 4-5)
**Goals**:
- ✅ Implement 4-zone world
- ✅ Transportation system (walk/bus/taxi)
- ✅ Time mechanics (day/night cycle)
- ✅ Hierarchical policy architecture

**Deliverables**:
- `MultiZoneEnvironment` class
- `HierarchicalAgent` with 3-level policies
- Curriculum learning schedule
- Zone-specific affordances (14 total)

**Success Metrics**:
- Agent learns to navigate between zones
- Uses transportation efficiently
- Plans multi-step sequences (work → gym → home)
- Peak reward: +200 to +500

---

### Phase 4: Opponent Modeling (Week 6-7)
**Goals**:
- ✅ Implement belief state tracking
- ✅ Opponent intent prediction
- ✅ Action prediction network
- ✅ Theory of mind level 1

**Deliverables**:
- `OpponentModel` class
- Belief state visualization
- Prediction accuracy metrics
- Integration with single-agent architecture

**Success Metrics**:
- >70% accuracy predicting opponent actions
- Agent uses opponent beliefs in decisions
- Evidence of strategic behavior (waiting, ambushing)

---

### Phase 5: Multi-Agent Competition (Week 8-10)
**Goals**:
- ✅ Multi-agent environment (2-6 agents)
- ✅ Competitive resource constraints
- ✅ Self-play training loop
- ✅ Population-based training

**Deliverables**:
- `CompetitiveMultiAgentDQN` full architecture
- `SelfPlayTrainer` with population management
- Competitive reward structure
- Tournament evaluation system

**Success Metrics**:
- Diverse strategies emerge in population
- Evidence of emergent behaviors (preemption, bluffing)
- Stable Nash equilibria found
- Publication-quality results

---

### Phase 6: Advanced Features (Week 11-12)
**Goals**:
- ✅ Communication channels (if needed)
- ✅ Level 2 theory of mind (recursive)
- ✅ Adaptive meta-learning
- ✅ Transfer learning across grid sizes

**Deliverables**:
- Optional communication module
- Advanced strategic reasoner
- Meta-learning experiments
- Comprehensive benchmark suite

---

## Research Opportunities

### Publishable Research Questions

#### 1. **Partial Observability + Opponent Modeling**
**Question**: How does partial observability affect opponent modeling accuracy?

**Hypothesis**: Agents with limited vision develop more robust belief models (like humans).

**Experiments**:
- Train agents with varying vision radii (2, 3, 5, 10 cells)
- Measure opponent prediction accuracy
- Analyze belief uncertainty

**Expected Result**: Sweet spot at 3-5 cells (enough info, but needs inference)

**Publication Venue**: ICLR, NeurIPS (multi-agent track)

---

#### 2. **Emergent Communication in Competitive Settings**
**Question**: Will agents learn to communicate implicitly through movement patterns?

**Hypothesis**: Repeated interactions lead to proto-language (signaling equilibria).

**Experiments**:
- Train without explicit communication channels
- Analyze agent trajectories for patterns
- Measure "signal" consistency and "comprehension"
- Introduce communication channel, measure improvement

**Expected Result**: Implicit signals emerge for high-stakes coordination

**Publication Venue**: CoRL, AAMAS (agent communication)

---

#### 3. **Hierarchical RL in Multi-Zone POMDPs**
**Question**: Does hierarchical decomposition improve sample efficiency in structured environments?

**Hypothesis**: Temporal abstraction reduces exploration time in hierarchical spaces.

**Experiments**:
- Compare flat policy vs 2-level vs 3-level hierarchy
- Measure episodes to convergence
- Analyze reuse of low-level policies

**Expected Result**: 3-5x sample efficiency improvement

**Publication Venue**: ICML, AAAI (hierarchical RL)

---

#### 4. **Theory of Mind Development in RL Agents**
**Question**: Can RL agents learn recursive opponent modeling (level 2+ ToM)?

**Hypothesis**: Level 2 ToM emerges only in specific strategic contexts.

**Experiments**:
- Analyze opponent model depth through ablations
- Design scenarios requiring level 2 reasoning
- Measure cognitive load (parameters needed)

**Expected Result**: Level 1 common, Level 2 rare but learnable

**Publication Venue**: CogSci, NeurIPS (cognitive science track)

---

#### 5. **Transfer Learning Across Environmental Scales**
**Question**: Do spatial policies learned on 8×8 grids transfer to 20×20?

**Hypothesis**: Convolutional architectures enable zero-shot transfer.

**Experiments**:
- Train on 8×8, test on 12×12, 16×16, 20×20
- Compare CNN vs MLP transfer
- Measure fine-tuning requirements

**Expected Result**: CNN transfers well, MLP fails completely

**Publication Venue**: ICML (representation learning)

---

#### 6. **Nash Equilibria in Continuous POMDPs**
**Question**: What equilibria exist in continuous time, partial observability, multi-agent systems?

**Hypothesis**: Mixed strategy equilibria dominate pure strategies.

**Experiments**:
- Characterize learned strategies (pure vs mixed)
- Compute best-response strategies
- Test equilibrium stability

**Expected Result**: Population converges to mixed equilibrium (unpredictability)

**Publication Venue**: NeurIPS (game theory), EC (economics and computation)

---

### Educational Applications

#### Course Structure: "Deep RL Through Game Design"

**Week 1-2: The Basics (Level 1)**
- Students implement Hamlet from scratch
- Learn DQN, replay buffers, epsilon-greedy
- **Hook**: "Let's make The Sims with AI"

**Week 3-4: Memory and Uncertainty (Level 2)**
- Add partial observability
- Implement LSTM and exploration
- **Realization**: "This is actually a POMDP!"

**Week 5-6: Planning and Hierarchy (Level 3)**
- Build multi-zone environment
- Hierarchical policies
- **Realization**: "This is the Options framework!"

**Week 7-8: Competition and Strategy (Level 4)**
- Multi-agent self-play
- Opponent modeling
- **Realization**: "This is game theory + deep learning!"

**Week 9-10: Final Project**
- Students design new mechanics
- Options: Communication, new zones, social dynamics
- Present findings as research posters

**Key Insight**: By week 10, students have built a research-grade MARL system and think it was just a fun game project.

---

### Demo Scenarios

#### Demo 1: "The Morning Commute"
**Setup**:
- 2 agents, 1 job slot
- Agent A: Lives close (4 cells)
- Agent B: Lives far (12 cells)

**Narrative**:
- Show 10 episodes without opponent modeling (random races)
- Show 10 episodes with opponent modeling (B learns to leave early)
- Show final strategy: B wakes at 2am, eats, arrives first

**Audience Impact**: "OMG they learned game theory!"

---

#### Demo 2: "The Bluff"
**Setup**:
- Agent A at low energy near home
- Agent B heading to job
- Both visible to each other

**Narrative**:
- Agent A starts walking toward job (bluff)
- Agent B sees this, races ahead
- Agent A turns around, goes to bed
- Show belief state visualization: B thought A was going to job

**Audience Impact**: "The AI is lying to each other!"

---

#### Demo 3: "The Invisible Preparation"
**Setup**:
- Show Agent B's internal monologue (through visualization)
- "Agent A wakes at 6am" (predicted)
- "I need 30 steps to reach job" (calculated)
- "I must leave by 5:30am" (strategy)
- "I'll eat until 5:30am" (preparation)

**Show**:
- Agent B waking at 2am
- Eating repeatedly at fridge (energy bars filling)
- Leaving at 5:30am in darkness
- Arriving at job at 6:00am
- Agent A waking at 6:00am, confused

**Audience Impact**: "Holy shit, it planned 4 hours ahead!"

---

## Technical Specifications

### Codebase Structure

```
hamlet/
├── src/hamlet/
│   ├── environment/
│   │   ├── hamlet_env.py           # Base environment (Level 1)
│   │   ├── pomdp_env.py            # Partial obs wrapper (Level 2)
│   │   ├── multizone_env.py        # Multi-zone env (Level 3)
│   │   ├── competitive_env.py      # Multi-agent env (Level 4)
│   │   ├── affordances.py          # Resource definitions
│   │   ├── meters.py               # Agent needs system
│   │   └── zones.py                # Zone definitions
│   │
│   ├── agent/
│   │   ├── base_agent.py           # Abstract agent
│   │   ├── drl_agent.py            # DQN implementation (Level 1)
│   │   ├── recurrent_agent.py      # LSTM agent (Level 2)
│   │   ├── hierarchical_agent.py   # 3-level agent (Level 3)
│   │   └── competitive_agent.py    # Multi-agent (Level 4)
│   │
│   ├── networks/
│   │   ├── qnetwork.py             # Simple MLP
│   │   ├── spatial_qnetwork.py     # CNN-based
│   │   ├── recurrent_qnetwork.py   # LSTM-based
│   │   ├── memory_module.py        # Spatial memory
│   │   ├── opponent_model.py       # Belief networks
│   │   └── strategic_reasoner.py   # Game theory module
│   │
│   ├── training/
│   │   ├── trainer.py              # Base trainer (Level 1)
│   │   ├── recurrent_trainer.py    # Sequence trainer (Level 2)
│   │   ├── hierarchical_trainer.py # Curriculum trainer (Level 3)
│   │   └── selfplay_trainer.py     # Population trainer (Level 4)
│   │
│   └── utils/
│       ├── replay_buffer.py        # Experience storage
│       ├── episode_buffer.py       # Sequence storage
│       ├── exploration.py          # Curiosity modules
│       └── visualization.py        # Rendering tools
│
├── configs/
│   ├── level1_baseline.yaml        # Current working config
│   ├── level2_pomdp.yaml           # Partial obs config
│   ├── level3_multizone.yaml       # Hierarchical config
│   └── level4_competitive.yaml     # Multi-agent config
│
├── experiments/
│   ├── ablations/                  # Architecture comparisons
│   ├── emergence/                  # Behavior analysis
│   └── benchmarks/                 # Performance tests
│
├── docs/
│   ├── ARCHITECTURE_DESIGN.md      # This document
│   ├── TEACHING_GUIDE.md           # Educational materials
│   └── RESEARCH_AGENDA.md          # Publication plan
│
└── demos/
    ├── morning_commute.py          # 2am strategy demo
    ├── the_bluff.py                # Deception demo
    └── visualization_server.py     # Web-based visualization
```

---

### Hardware Requirements

**Level 1 (Current)**:
- CPU: Any modern CPU
- RAM: 4GB
- GPU: Optional (2x speedup)
- Training time: 3-5 minutes

**Level 2 (Partial Obs)**:
- CPU: 4+ cores recommended
- RAM: 8GB
- GPU: Recommended (RTX 2060+)
- Training time: 15-30 minutes

**Level 3 (Multi-Zone)**:
- CPU: 8+ cores
- RAM: 16GB
- GPU: Required (RTX 3070+)
- Training time: 2-4 hours

**Level 4 (Multi-Agent)**:
- CPU: 16+ cores (parallel envs)
- RAM: 32GB
- GPU: Required (RTX 3090 or A100)
- Training time: 8-24 hours

---

### Visualization Tools

**Terminal Visualization** (Current):
```python
def render_terminal(env_state, episode, step, reward):
    """
    Simple ASCII grid
    . = empty
    A = agent
    B = bed
    S = shower
    F = fridge
    J = job
    """
```

**Web Visualization** (Implemented):
```python
# FastAPI + Vue.js real-time streaming
# Shows: Grid, meters, episode stats
# WebSocket for live updates
```

**Advanced Visualization** (Future):
```python
# Belief state visualization
- Heatmap of spatial memory
- Opponent position uncertainty
- Intent distribution pie charts
- Trajectory predictions

# Strategic reasoning
- Win probability over time
- Resource contestation graph
- Theory of mind depth indicators

# Population dynamics
- Strategy diversity plot
- Nash equilibrium convergence
- Evolutionary tree of strategies
```

---

## Future Extensions: Family Dynamics & Communication

### Level 5: Family Units with Private Channels (Stretch Goal)

**Core Concept**: Agents belong to "families" with shared communication channel. What emergent behaviors arise from in-group communication?

#### Family Structure

```python
class Family:
    """
    Group of agents with shared communication channel
    """
    family_id: str
    members: List[Agent]  # 2-4 agents per family
    shared_channel: CommunicationChannel

    # Family resources (optional)
    shared_home: Zone  # Family-owned housing
    family_bank: float  # Shared money pool

    # Relationships
    member_roles: Dict[Agent, str]  # "parent", "child", "sibling"
```

#### Communication Channel Design

```python
class CommunicationChannel:
    """
    Private channel for family communication

    Key: Channel is NOT pre-programmed. Agents must learn to use it.
    """

    def send_message(self, sender: Agent, message: torch.Tensor):
        """
        Send learned representation to family members

        Args:
            message: 32-dim learned vector (NO predefined meaning)

        Broadcast to all family members in the channel
        """
        # Agents learn what to send through RL
        # No supervision on message content
        pass

    def receive_messages(self, agent: Agent) -> List[torch.Tensor]:
        """
        Receive all messages from family members

        Returns: List of 32-dim vectors from other family members
        """
        pass
```

**Crucially**: Messages are **learned representations**, not pre-defined symbols. Agents must:
1. Learn what to send (encoding)
2. Learn what received messages mean (decoding)
3. Learn when communication helps vs hurts

#### Architecture Extension

```python
class FamilyCommunicativeAgent(CompetitiveMultiAgentDQN):
    """
    Extends Level 4 agent with communication capabilities
    """

    # Communication encoder: State → Message
    def encode_message(self, my_state, intent):
        """
        Decide what to communicate to family

        Examples of emergent messages:
        - "I'm going to job" (warn others)
        - "Job is occupied" (information sharing)
        - "Help needed at (x,y)" (coordination)
        - "Danger: strong opponent nearby" (threat warning)

        But these are NOT programmed! Agent learns what's useful.
        """
        message_encoder = nn.Sequential(
            nn.Linear(256 + 64, 128),  # state + intent
            nn.ReLU(),
            nn.Linear(128, 32),  # 32-dim message
            nn.Tanh()  # Bounded message space
        )
        return message_encoder(torch.cat([my_state, intent]))

    # Communication decoder: Message → Meaning
    def decode_message(self, received_message):
        """
        Interpret message from family member

        Learns through experience what messages correlate with
        useful information.
        """
        message_decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),  # Decoded features
        )
        return message_decoder(received_message)

    # Modified decision-making
    def forward(self, obs, hidden, beliefs, family_messages):
        """
        Now considers family communication in decision

        Pipeline:
        1. Process observation (as before)
        2. Update opponent beliefs (as before)
        3. Decode family messages
        4. Strategic reasoning WITH family coordination
        5. Final Q-values considering family strategy
        """
        # Decode all family messages
        family_info = []
        for msg in family_messages:
            decoded = self.decode_message(msg)
            family_info.append(decoded)

        if family_info:
            family_aggregate = torch.mean(torch.stack(family_info), dim=0)
        else:
            family_aggregate = torch.zeros(128)

        # Combine: my_obs + opponents + strategy + family
        combined = torch.cat([
            my_features,
            opponent_features,
            strategy_features,
            family_aggregate  # NEW
        ], dim=-1)

        return q_network(combined)
```

#### Expected Emergent Behaviors

**Behavior 1: Resource Allocation**
```
Family of 3 agents discovers efficient division:
- Agent A (parent): Always goes to job (stable income)
- Agent B (child): Explores for food (fridge, restaurant)
- Agent C (child): Maintains home (bed, shower)

Communication emerges:
- A signals when shift ends → B/C coordinate pickup
- B signals food scarcity → A brings money
- C signals home crisis → family regroups

Result: 40% higher family survival rate vs independent agents
```

**Behavior 2: Threat Warnings**
```
Agent A encounters strong opponent near job
A sends warning signal to family
Family B and C avoid that zone for next N steps

Communication content (learned):
- High activation in dimension 5, 12, 27 = "danger nearby"
- Correlated with A's stress level + opponent proximity

Family learns to interpret these signals
```

**Behavior 3: Teaching and Imitation**
```
Experienced family member (agent A) discovers optimal strategy
A's behaviors visible to family B and C through communication

Communication emerges:
- A sends signal after successful job completion
- B and C learn to imitate A's approach
- "Follow me" signaling emerges naturally

Result: Faster learning for new family members (social learning)
```

**Behavior 4: Deception Across Families**
```
Family 1 vs Family 2 competition
Family 1 learns to send misleading signals

If opponent intercepts communication (possible if nearby):
- Family 1 sends "going to job" signal
- Actually going to alternative resource
- Family 2 falls for bluff, wastes effort

Arms race: Encryption-like behavior emerges?
```

**Behavior 5: Emergent Roles and Specialization**
```
Over many episodes, family develops stable roles:

"Breadwinner":
- Specializes in Industrial zone
- High stress tolerance, optimized work paths
- Signals: "income stable" / "need rest"

"Homemaker":
- Specializes in Home maintenance
- Monitors all family member meters
- Signals: "bed available" / "food ready"

"Scout":
- Explores and finds resources
- High mobility, low commitment
- Signals: "found X at (y,z)" / "zone Y is crowded"

These roles NOT programmed, emerge from optimization!
```

**Behavior 6: Multi-Generation Learning** (Very Advanced)
```
If implementing agent "death" and "birth":

Parent agents pass knowledge to offspring:
- Final messages before death = training data
- Offspring initialize with parent's knowledge
- Dynasties develop over 100+ generations

Emergent: Cultural transmission of strategies
"This is how our family survives"
```

#### Training Challenges

**Challenge 1: Communication Bootstrap Problem**
```
Early training: Messages are random noise
No agent knows what to send or how to interpret

Solution: Curriculum learning
- Stage 1: Train without communication (baseline)
- Stage 2: Add communication, reward correlation
  - Bonus: If family member uses info from message
  - Penalty: If message is ignored (waste)
- Stage 3: Full competitive environment with families
```

**Challenge 2: Language Drift**
```
Each family develops different "language"
Family A's messages incomprehensible to Family B

This is GOOD (allows family-specific strategies)
But creates research opportunity: Can we measure "language" similarity?
```

**Challenge 3: Communication Overhead**
```
Sending messages has cost (attention, processing)
Agents must learn WHEN to communicate, not just WHAT

Implementation:
- Add "send message" as explicit action (costs time step)
- Or: Continuous communication but with noise/capacity limits
```

#### Research Questions

**RQ1**: What structures emerge in learned communication?
- Analyze message space (PCA, t-SNE)
- Cluster messages by context
- Discover "proto-words" (common patterns)

**RQ2**: Does communication improve sample efficiency?
- Compare family vs independent agents
- Measure: Episodes to convergence
- Hypothesis: 2-3x faster learning with communication

**RQ3**: Can agents learn to deceive through communication?
- Intentional false signals
- Selective truth-telling
- Measure: Opponent modeling accuracy when communication exists

**RQ4**: Does "culture" emerge across generations?
- Stable family strategies over time?
- Innovation vs tradition trade-off?
- Measure: Strategy similarity across agent lifetimes

**RQ5**: Optimal family size?
- Trade-off: More members = more communication overhead
- But: More specialization possible
- Hypothesis: 3-4 members optimal

#### Implementation Roadmap

**Phase 1: Basic Communication** (Week 13-14)
- Communication channel infrastructure
- Message encoding/decoding networks
- Simple 2-agent families

**Phase 2: Emergent Language Analysis** (Week 15-16)
- Message analysis tools
- Visualization of learned "language"
- Ablation: With vs without communication

**Phase 3: Family Dynamics** (Week 17-18)
- 3-4 agent families
- Role specialization metrics
- Inter-family competition

**Phase 4: Multi-Generation** (Week 19-20)
- Agent lifecycle (birth/death)
- Knowledge transfer mechanisms
- Dynasty analysis tools

#### Potential Publications

**"Emergent Communication in Competitive Multi-Agent RL"**
- Venue: ICLR, NeurIPS (communication track)
- Focus: Language emergence without supervision
- Novel: Partial observability + competition + families

**"Social Learning Through Learned Communication"**
- Venue: AAMAS, IJCAI
- Focus: How communication accelerates learning
- Novel: Family units as learning accelerators

**"Cultural Evolution in Multi-Agent Systems"**
- Venue: ALife, ECAL
- Focus: Multi-generation strategy transmission
- Novel: Dynasties and cultural drift in RL

---

### Summary of Progression

```
Level 1: Single agent, full obs
  ↓ "What if they can't see everything?"
Level 2: Partial observability, memory
  ↓ "What if the world is bigger?"
Level 3: Multi-zone, hierarchical planning
  ↓ "What if there are other agents?"
Level 4: Multi-agent competition, theory of mind
  ↓ "What if they could talk to each other?"
Level 5: Family communication, emergent language
  ↓ "What if families compete across generations?"
Level 6: Multi-generation dynasties, cultural evolution

Final Form: Civilization simulator with emergent social structures
```

**The Dream**: Students think they're making The Sims. They accidentally invent emergent language, game theory, and cultural evolution. Mind = blown. 🤯

---

## Conclusion

Hamlet represents a **progression from toy problem to research platform**:

**For Students**:
- Engaging game-like interface
- Progressive difficulty (learn by doing)
- "Aha!" moments as complexity reveals underlying theory
- Portfolio-worthy final project

**For Researchers**:
- Rich environment for MARL research
- Unexplored: POMDPs + ToM + Hierarchical RL
- Publication opportunities across 6+ venues
- Reproducible, open-source platform

**For Demonstrators**:
- Impressive emergent behaviors (2am strategy, bluffing)
- Visual appeal (web interface, graphs)
- Narrative potential ("AI agents playing mind games")
- Scalable complexity (8×8 to 20×20 to N zones)

**Pedagogical Success Metrics**:
- Can explain Bellman equation by episode 5
- Understands POMDP by week 3
- Implements hierarchical RL by week 6
- Discovers Nash equilibrium by week 8
- **Thinks it was just a fun game project the whole time**

---

**Next Steps**:
1. Complete Level 1 checkpoint bugfix
2. Begin Level 2 implementation (partial observability)
3. Write teaching materials for educational deployment
4. Submit to NeurIPS 2026 (Multi-Agent track)

**The dream**: Students accidentally learn graduate-level RL while thinking they're just making The Sims with Python.

---

*Document Version*: 1.0
*Last Updated*: 2025-10-27
*Authors*: John & Claude
*Status*: Design Complete, Implementation In Progress
*License*: MIT (for educational use)*

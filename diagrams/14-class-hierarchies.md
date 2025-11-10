# Class Hierarchies

## Overview

This document details the class hierarchies for the major subsystems in Townlet, including:
- **Substrate Hierarchy**: Spatial abstraction layer
- **Curriculum Hierarchy**: Difficulty progression strategies
- **Exploration Hierarchy**: Action selection and intrinsic motivation
- **Network Hierarchy**: Neural network architectures
- **Replay Buffer Hierarchy**: Experience storage strategies

## 1. Substrate Hierarchy

### Abstract Base: SpatialSubstrate

```mermaid
classDiagram
    class SpatialSubstrate {
        <<abstract>>
        +position_dim: int
        +position_dtype: torch.dtype
        +action_space_size: int
        +initialize_positions(num_agents, device) torch.Tensor
        +apply_movement(positions, deltas) torch.Tensor
        +compute_distance(pos1, pos2) torch.Tensor
        +encode_observation(positions, affordances) torch.Tensor
        +get_observation_dim() int
        +normalize_positions(positions) torch.Tensor
        +get_valid_neighbors(position) list
        +is_on_position(agent_pos, target_pos) torch.Tensor
        +get_all_positions() list
        +encode_partial_observation(...) torch.Tensor
        +get_default_actions() list[ActionConfig]
    }

    class Grid2DSubstrate {
        +width: int
        +height: int
        +boundary: str
        +distance_metric: str
        +observation_encoding: str
        +topology: str
        +position_dim: int = 2
        +position_dtype = torch.long
        +action_space_size = 8
        +_encode_relative(...) torch.Tensor
        +_encode_scaled(...) torch.Tensor
        +_encode_absolute(...) torch.Tensor
        +_encode_full_grid(...) torch.Tensor
    }

    class Grid3DSubstrate {
        +width: int
        +height: int
        +depth: int
        +boundary: str
        +distance_metric: str
        +observation_encoding: str
        +topology: str
        +position_dim: int = 3
        +position_dtype = torch.long
        +action_space_size = 10
    }

    class GridNDSubstrate {
        +dimensions: list[int]
        +boundary: str
        +distance_metric: str
        +observation_encoding: str
        +topology: str
        +position_dim: int = len(dimensions)
        +position_dtype = torch.long
        +action_space_size = 2*N + 2
    }

    class ContinuousSubstrate {
        +bounds: list[tuple[float,float]]
        +position_dim: int = len(bounds)
        +position_dtype = torch.float32
        +action_space_size = 2*N + 2
        +proximity_threshold: float
    }

    class ContinuousNDSubstrate {
        +bounds: list[tuple[float,float]]
        +position_dim: int = len(bounds)
        +position_dtype = torch.float32
        +action_space_size = 2*N + 2
        +proximity_threshold: float
    }

    class AspatialSubstrate {
        +position_dim: int = 0
        +position_dtype = torch.long
        +action_space_size = 2
    }

    SpatialSubstrate <|-- Grid2DSubstrate
    SpatialSubstrate <|-- Grid3DSubstrate
    SpatialSubstrate <|-- GridNDSubstrate
    SpatialSubstrate <|-- ContinuousSubstrate
    SpatialSubstrate <|-- ContinuousNDSubstrate
    SpatialSubstrate <|-- AspatialSubstrate

    style SpatialSubstrate fill:#d1c4e9
    style Grid2DSubstrate fill:#c8e6c9
    style AspatialSubstrate fill:#e1f5fe
```

### Substrate Feature Matrix

| Substrate | position_dim | position_dtype | action_space_size | Supports POMDP | Enumerable Positions |
|-----------|--------------|----------------|-------------------|----------------|----------------------|
| Grid2D | 2 | torch.long | 8 (6 + INTERACT + WAIT) | ✅ Yes (vision_range ≤ 2) | ✅ Yes |
| Grid3D | 3 | torch.long | 10 (8 + INTERACT + WAIT) | ✅ Yes (vision_range ≤ 2) | ✅ Yes |
| GridND(N) | N | torch.long | 2*N + 2 | ❌ No (N≥4, window too large) | ✅ Yes |
| Continuous | 1-3 | torch.float32 | 2*N + 2 | ❌ No (infinite positions) | ❌ No |
| ContinuousND(N) | N | torch.float32 | 2*N + 2 | ❌ No (infinite positions) | ❌ No |
| Aspatial | 0 | torch.long | 2 (INTERACT + WAIT) | ✅ Special case | ❌ No |

### Boundary Mode Implementations

```mermaid
graph TB
    boundary_mode["Boundary Modes<br/>(Grid Substrates Only)"]
    
    clamp["clamp: Hard walls<br/>Clamp to grid edge"]
    wrap["wrap: Toroidal<br/>Pac-Man wraparound"]
    bounce["bounce: Elastic<br/>Reflect from boundary"]
    sticky["sticky: Sticky walls<br/>Stay in place when OOB"]
    
    boundary_mode --> clamp
    boundary_mode --> wrap
    boundary_mode --> bounce
    boundary_mode --> sticky
    
    style boundary_mode fill:#d1c4e9
    style clamp fill:#c8e6c9
```

### Distance Metric Implementations

```mermaid
graph TB
    distance_metric["Distance Metrics<br/>(Grid Substrates)"]
    
    manhattan["manhattan: L1 distance<br/>|x1-x2| + |y1-y2|"]
    euclidean["euclidean: L2 distance<br/>sqrt((x1-x2)² + (y1-y2)²)"]
    chebyshev["chebyshev: L∞ distance<br/>max(|x1-x2|, |y1-y2|)"]
    
    distance_metric --> manhattan
    distance_metric --> euclidean
    distance_metric --> chebyshev
    
    style distance_metric fill:#d1c4e9
    style manhattan fill:#c8e6c9
```

## 2. Curriculum Hierarchy

### Abstract Base: CurriculumManager

```mermaid
classDiagram
    class CurriculumManager {
        <<abstract>>
        +get_batch_decisions(agent_states, agent_ids) list[CurriculumDecision]
        +checkpoint_state() dict
        +load_state(state) None
    }

    class AdversarialCurriculum {
        +performance_tracker: PerformanceTracker
        +num_agents: int
        +device: torch.device
        +get_batch_decisions(...) list[CurriculumDecision]
        +checkpoint_state() dict
        +load_state(state) None
        +_decide_stage_transition(agent_idx) int
        +_get_stage_config(stage) StageConfig
    }

    class StaticCurriculum {
        +stage: int
        +active_meters: list[str]
        +depletion_multiplier: float
        +reward_mode: str
        +get_batch_decisions(...) list[CurriculumDecision]
        +checkpoint_state() dict
        +load_state(state) None
    }

    class PerformanceTracker {
        +num_agents: int
        +device: torch.device
        +episode_rewards: torch.Tensor
        +episode_steps: torch.Tensor
        +agent_stages: torch.Tensor
        +steps_at_stage: torch.Tensor
        +update_step(rewards, dones) None
        +update_episode_end(rewards, dones) None
    }

    class StageConfig {
        <<pydantic>>
        +stage: int
        +active_meters: list[str]
        +depletion_multiplier: float
        +reward_mode: str
        +description: str
    }

    CurriculumManager <|-- AdversarialCurriculum
    CurriculumManager <|-- StaticCurriculum
    AdversarialCurriculum *-- PerformanceTracker
    AdversarialCurriculum ..> StageConfig : uses

    style CurriculumManager fill:#d1c4e9
    style AdversarialCurriculum fill:#c8e6c9
    style PerformanceTracker fill:#fff9c4
```

### Adversarial Curriculum Stage Progression

```mermaid
stateDiagram-v2
    [*] --> Stage1
    Stage1 --> Stage2 : High survival + low variance
    Stage2 --> Stage3 : High survival + low variance
    Stage3 --> Stage4 : High survival + low variance
    Stage4 --> Stage5 : High survival + low variance
    Stage5 --> [*] : Graduated!

    state Stage1 {
        [*] --> Stage1Config
        state Stage1Config {
            active_meters: energy, hygiene
            depletion: 0.2x
            reward_mode: shaped
        }
    }

    state Stage2 {
        [*] --> Stage2Config
        state Stage2Config {
            active_meters: energy, hygiene, satiation
            depletion: 0.5x
            reward_mode: shaped
        }
    }

    state Stage3 {
        [*] --> Stage3Config
        state Stage3Config {
            active_meters: +money
            depletion: 0.8x
            reward_mode: shaped
        }
    }

    state Stage4 {
        [*] --> Stage4Config
        state Stage4Config {
            active_meters: all 6 meters
            depletion: 1.0x
            reward_mode: shaped
        }
    }

    state Stage5 {
        [*] --> Stage5Config
        state Stage5Config {
            active_meters: all 6 meters
            depletion: 1.0x
            reward_mode: SPARSE
        }
    }
```

## 3. Exploration Hierarchy

### Abstract Base: ExplorationStrategy

```mermaid
classDiagram
    class ExplorationStrategy {
        <<abstract>>
        +select_actions(q_values, agent_states, action_masks) torch.Tensor
        +compute_intrinsic_rewards(observations) torch.Tensor
        +update(batch) None
        +checkpoint_state() dict
        +load_state(state) None
    }

    class AdaptiveIntrinsicExploration {
        +rnd: RNDExploration
        +current_intrinsic_weight: float
        +min_intrinsic_weight: float
        +variance_threshold: float
        +survival_window: int
        +decay_rate: float
        +survival_history: list[float]
        +select_actions(...) torch.Tensor
        +compute_intrinsic_rewards(...) torch.Tensor
        +update(batch) None
        +update_survival_history(survival_time) None
        +get_current_weight() float
    }

    class RNDExploration {
        +target_network: nn.Module
        +predictor_network: nn.Module
        +optimizer: torch.optim.Adam
        +epsilon: float
        +epsilon_min: float
        +epsilon_decay: float
        +device: torch.device
        +select_actions(...) torch.Tensor
        +compute_intrinsic_rewards(...) torch.Tensor
        +update(batch) None
        +_compute_prediction_error(...) torch.Tensor
    }

    class EpsilonGreedyExploration {
        +epsilon: float
        +epsilon_min: float
        +epsilon_decay: float
        +device: torch.device
        +select_actions(...) torch.Tensor
        +compute_intrinsic_rewards(...) torch.Tensor
        +update(batch) None
    }

    class RNDNetwork {
        <<nn.Module>>
        +embed_dim: int
        +encoder: nn.Sequential
        +forward(obs) torch.Tensor
    }

    ExplorationStrategy <|-- AdaptiveIntrinsicExploration
    ExplorationStrategy <|-- RNDExploration
    ExplorationStrategy <|-- EpsilonGreedyExploration
    AdaptiveIntrinsicExploration *-- RNDExploration : composition
    RNDExploration *-- RNDNetwork : target
    RNDExploration *-- RNDNetwork : predictor

    style ExplorationStrategy fill:#d1c4e9
    style AdaptiveIntrinsicExploration fill:#c8e6c9
    style RNDNetwork fill:#fff9c4
```

### Exploration Strategy Comparison

| Strategy | Intrinsic Rewards | Epsilon Decay | Network Training | Use Case |
|----------|-------------------|---------------|------------------|----------|
| **EpsilonGreedy** | ❌ None (returns zeros) | ✅ Exponential decay | ❌ No networks | Simple exploration |
| **RND** | ✅ Prediction error (novelty) | ✅ Exponential decay | ✅ Predictor network | Novelty-driven exploration |
| **AdaptiveIntrinsic** | ✅ Weighted RND rewards | ✅ Exponential decay | ✅ RND networks | Adaptive novelty with annealing |

### RND (Random Network Distillation) Architecture

```mermaid
graph TB
    obs[Observations<br/>[batch, obs_dim]]
    
    target_net[Target Network<br/>Fixed random weights]
    predictor_net[Predictor Network<br/>Trained weights]
    
    target_embed[Target Embeddings<br/>[batch, embed_dim]]
    predictor_embed[Predictor Embeddings<br/>[batch, embed_dim]]
    
    mse[MSE Loss<br/>Prediction Error]
    intrinsic[Intrinsic Rewards<br/>[batch]]
    
    obs --> target_net
    obs --> predictor_net
    
    target_net --> target_embed
    predictor_net --> predictor_embed
    
    target_embed --> mse
    predictor_embed --> mse
    
    mse --> intrinsic
    
    style target_net fill:#ffccbc
    style predictor_net fill:#c8e6c9
    style intrinsic fill:#e1f5fe
```

## 4. Network Hierarchy

### PyTorch nn.Module Subclasses

```mermaid
classDiagram
    class nn_Module {
        <<pytorch>>
        +forward(x) torch.Tensor
        +parameters() Iterator
        +state_dict() dict
        +load_state_dict(dict) None
    }

    class SimpleQNetwork {
        +obs_dim: int
        +action_dim: int
        +hidden_dim: int
        +net: nn.Sequential
        +forward(x) torch.Tensor
    }

    class RecurrentSpatialQNetwork {
        +action_dim: int
        +window_size: int
        +position_dim: int
        +num_meters: int
        +num_affordance_types: int
        +enable_temporal_features: bool
        +hidden_dim: int
        +vision_encoder: nn.Sequential
        +position_encoder: nn.Sequential
        +meter_encoder: nn.Sequential
        +affordance_encoder: nn.Sequential
        +lstm: nn.LSTM
        +q_head: nn.Sequential
        +forward(obs, hidden) tuple
        +init_hidden(batch_size) tuple
    }

    class RNDNetwork {
        +embed_dim: int
        +encoder: nn.Sequential
        +forward(obs) torch.Tensor
    }

    nn_Module <|-- SimpleQNetwork
    nn_Module <|-- RecurrentSpatialQNetwork
    nn_Module <|-- RNDNetwork

    style nn_Module fill:#d1c4e9
    style SimpleQNetwork fill:#c8e6c9
    style RecurrentSpatialQNetwork fill:#e1f5fe
```

### SimpleQNetwork Architecture

```mermaid
graph TB
    input[Input<br/>[batch, obs_dim]]
    
    fc1[Linear: obs_dim → 256]
    ln1[LayerNorm: 256]
    relu1[ReLU]
    
    fc2[Linear: 256 → 128]
    ln2[LayerNorm: 128]
    relu2[ReLU]
    
    fc3[Linear: 128 → action_dim]
    
    output[Q-Values<br/>[batch, action_dim]]
    
    input --> fc1
    fc1 --> ln1
    ln1 --> relu1
    relu1 --> fc2
    fc2 --> ln2
    ln2 --> relu2
    relu2 --> fc3
    fc3 --> output
    
    style input fill:#d1c4e9
    style output fill:#c8e6c9
```

### RecurrentSpatialQNetwork Architecture

```mermaid
graph TB
    subgraph "Input Parsing"
        obs[Observations<br/>[batch, seq_len, obs_dim]]
        parse[Parse Components]
        
        local_window[Local Window<br/>[batch, seq, window²]]
        position[Position<br/>[batch, seq, 2]]
        meters[Meters<br/>[batch, seq, 8]]
        affordances[Affordances<br/>[batch, seq, 15]]
        temporal[Temporal<br/>[batch, seq, 4]]
    end
    
    subgraph "Encoders"
        vision_cnn[Vision Encoder<br/>CNN: window² → 128]
        position_mlp[Position Encoder<br/>MLP: 2 → 32]
        meter_mlp[Meter Encoder<br/>MLP: 8 → 32]
        affordance_mlp[Affordance Encoder<br/>MLP: 15 → 32]
    end
    
    subgraph "Recurrent Layer"
        concat[Concatenate<br/>128+32+32+32 = 224]
        lstm[LSTM<br/>224 → 256 hidden]
        hidden_state["(h, c)<br/>[1, batch, 256]"]
    end
    
    subgraph "Q-Head"
        fc1[Linear: 256 → 128]
        relu[ReLU]
        fc2[Linear: 128 → action_dim]
        q_values[Q-Values<br/>[batch, seq, action_dim]]
    end
    
    obs --> parse
    parse --> local_window
    parse --> position
    parse --> meters
    parse --> affordances
    parse --> temporal
    
    local_window --> vision_cnn
    position --> position_mlp
    meters --> meter_mlp
    affordances --> affordance_mlp
    
    vision_cnn --> concat
    position_mlp --> concat
    meter_mlp --> concat
    affordance_mlp --> concat
    
    concat --> lstm
    lstm --> hidden_state
    
    hidden_state --> fc1
    fc1 --> relu
    relu --> fc2
    fc2 --> q_values
    
    style vision_cnn fill:#c8e6c9
    style lstm fill:#e1f5fe
    style q_values fill:#fff9c4
```

### Network Architecture Comparison

| Network | Architecture | Params | Observability | Use Case |
|---------|-------------|--------|---------------|----------|
| **SimpleQNetwork** | 3-layer MLP | ~26K | Full observability | L0, L0.5, L1, L3 |
| **RecurrentSpatialQNetwork** | CNN + LSTM + MLP | ~650K | Partial observability (POMDP) | L2 |

**Key Insight**: SimpleQNetwork uses fixed 29→256→128→8 architecture for all Grid2D configs, enabling checkpoint transfer across curriculum levels!

## 5. Replay Buffer Hierarchy

### Buffer Type Selection

```mermaid
graph TB
    buffer_type[Replay Buffer Type]
    
    standard[StandardReplayBuffer<br/>Feedforward networks<br/>IID transitions]
    sequential[SequentialReplayBuffer<br/>LSTM networks<br/>Temporal sequences]
    
    buffer_type --> standard
    buffer_type --> sequential
    
    subgraph "StandardReplayBuffer"
        store_standard[Store: (s, a, r, s', done)]
        sample_standard[Sample: Random batch<br/>[batch_size]]
        iid[IID assumption<br/>No temporal order]
    end
    
    subgraph "SequentialReplayBuffer"
        store_sequential[Store: Episodes as sequences]
        sample_sequential[Sample: Random subsequences<br/>[batch_size, seq_len]]
        temporal[Preserve temporal order<br/>LSTM hidden state]
    end
    
    standard --> store_standard
    standard --> sample_standard
    standard --> iid
    
    sequential --> store_sequential
    sequential --> sample_sequential
    sequential --> temporal
    
    style buffer_type fill:#d1c4e9
    style standard fill:#c8e6c9
    style sequential fill:#e1f5fe
```

### Replay Buffer Interfaces

```mermaid
classDiagram
    class StandardReplayBuffer {
        +capacity: int
        +buffer: list
        +position: int
        +push(state, action, reward, next_state, done) None
        +sample(batch_size) dict
        +__len__() int
    }

    class SequentialReplayBuffer {
        +capacity: int
        +seq_len: int
        +episodes: list[list[Transition]]
        +current_episode: list[Transition]
        +push(state, action, reward, next_state, done) None
        +end_episode() None
        +sample(batch_size) dict
        +__len__() int
    }

    class Transition {
        <<namedtuple>>
        +state: torch.Tensor
        +action: torch.Tensor
        +reward: torch.Tensor
        +next_state: torch.Tensor
        +done: torch.Tensor
    }

    StandardReplayBuffer ..> Transition : stores
    SequentialReplayBuffer ..> Transition : stores

    style StandardReplayBuffer fill:#c8e6c9
    style SequentialReplayBuffer fill:#e1f5fe
```

## 6. Population Architecture

### VectorizedPopulation Composition

```mermaid
classDiagram
    class VectorizedPopulation {
        +q_network: nn.Module
        +target_network: nn.Module
        +optimizer: torch.optim.Adam
        +replay_buffer: ReplayBuffer
        +exploration: ExplorationStrategy
        +device: torch.device
        +num_agents: int
        +agent_ids: list[str]
        +select_actions(observations, ...) torch.Tensor
        +train_step(batch_size) float
        +update_target_network() None
        +checkpoint_state() dict
        +load_state(state) None
    }

    class SimpleQNetwork {
        +forward(x) torch.Tensor
    }

    class RecurrentSpatialQNetwork {
        +forward(obs, hidden) tuple
    }

    class StandardReplayBuffer {
        +sample(batch_size) dict
    }

    class SequentialReplayBuffer {
        +sample(batch_size) dict
    }

    class AdaptiveIntrinsicExploration {
        +select_actions(...) torch.Tensor
        +compute_intrinsic_rewards(...) torch.Tensor
    }

    VectorizedPopulation *-- SimpleQNetwork : q_network (L0/L1)
    VectorizedPopulation *-- RecurrentSpatialQNetwork : q_network (L2)
    VectorizedPopulation *-- StandardReplayBuffer : replay_buffer (feedforward)
    VectorizedPopulation *-- SequentialReplayBuffer : replay_buffer (LSTM)
    VectorizedPopulation *-- AdaptiveIntrinsicExploration : exploration

    style VectorizedPopulation fill:#d1c4e9
    style SimpleQNetwork fill:#c8e6c9
    style AdaptiveIntrinsicExploration fill:#fff9c4
```

## 7. Configuration DTOs

### Training Configuration Hierarchy

```mermaid
classDiagram
    class TrainingConfig {
        <<pydantic>>
        +max_steps_per_episode: int
        +num_agents: int
        +total_episodes: int
        +batch_size: int
        +learning_rate: float
        +gamma: float
        +target_network_update_freq: int
        +replay_buffer_capacity: int
        +checkpoint_freq: int
    }

    class EnvironmentConfig {
        <<pydantic>>
        +pomdp_enabled: bool
        +vision_range: int
        +enable_temporal_features: bool
    }

    class PopulationConfig {
        <<pydantic>>
        +network_type: str
        +hidden_dim: int
        +gradient_clip_max_norm: float
    }

    class CurriculumConfig {
        <<pydantic>>
        +type: str
        +stage: int
        +active_meters: list[str]
        +depletion_multiplier: float
        +reward_mode: str
    }

    class ExplorationConfig {
        <<pydantic>>
        +type: str
        +epsilon_start: float
        +epsilon_min: float
        +epsilon_decay: float
        +rnd_embed_dim: int
    }

    TrainingConfig ..> EnvironmentConfig : contains
    TrainingConfig ..> PopulationConfig : contains
    TrainingConfig ..> CurriculumConfig : contains
    TrainingConfig ..> ExplorationConfig : contains

    style TrainingConfig fill:#d1c4e9
```

## Summary

### Class Hierarchy Patterns

1. **Strategy Pattern**: Curriculum, Exploration, Replay Buffer
2. **Composition Over Inheritance**: AdaptiveIntrinsicExploration contains RNDExploration
3. **Abstract Base Classes**: Define interfaces for hot path (GPU) methods
4. **Pydantic DTOs**: Enforce no-defaults principle (PDR-002) for configuration
5. **PyTorch nn.Module**: All neural networks inherit from nn.Module

### Polymorphism Usage

| Abstraction | Runtime Selection | Selection Criteria |
|-------------|-------------------|-------------------|
| **SpatialSubstrate** | SubstrateFactory | substrate.yaml type field |
| **CurriculumManager** | Training config | training.yaml curriculum.type |
| **ExplorationStrategy** | Training config | training.yaml exploration.type |
| **QNetwork** | POMDP flag | environment.yaml pomdp_enabled |
| **ReplayBuffer** | Network type | "simple" → Standard, "recurrent" → Sequential |

### Inheritance vs Composition

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Inheritance** | Interface polymorphism | Grid2D inherits SpatialSubstrate |
| **Composition** | Behavior delegation | AdaptiveIntrinsic contains RND |
| **Aggregation** | Loose coupling | Population holds Exploration reference |

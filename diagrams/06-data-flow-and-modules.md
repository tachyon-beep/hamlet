# Data Flow & Module Interactions

## Complete Data Flow Pipeline

```mermaid
flowchart TD
    subgraph "Configuration Stage (Offline)"
        yaml_files["YAML Configuration Files"]
        compiler["UniverseCompiler"]
        compiled["CompiledUniverse<br/>(Immutable)"]
        cache["Cache<br/>.compiled/universe.msgpack"]
    end

    subgraph "Initialization Stage"
        load_compiled["Load/Compile Universe"]
        runtime["RuntimeUniverse<br/>(Mutable Views)"]
        create_env["Create VectorizedHamletEnv"]
        create_q["Create Q-Network"]
        create_buffer["Create Replay Buffer"]
        create_curriculum["Create Curriculum"]
        create_exploration["Create Exploration"]
        create_population["Create VectorizedPopulation"]
    end

    subgraph "Training Loop"
        reset["Reset Environment & Population"]
        obs["Observations<br/>[num_agents, obs_dim]"]
        q_values["Q-Values<br/>[num_agents, action_dim]"]
        actions["Actions<br/>[num_agents]"]
        env_transition["Environment Transition"]
        rewards["Rewards<br/>[num_agents]"]
        next_obs["Next Observations<br/>[num_agents, obs_dim]"]
        dones["Dones<br/>[num_agents]"]
        store["Store Transition"]
        train{"Train<br/>Every N Steps?"}
        sample["Sample Batch"]
        update["Update Q-Network"]
        curriculum_update["Update Curriculum"]
    end

    subgraph "Persistence Stage"
        checkpoint_save["Save Checkpoint"]
        db_log["Log to Database"]
        tb_log["Log to TensorBoard"]
    end

    yaml_files --> compiler
    compiler --> compiled
    compiled --> cache
    cache -.->|Hit| load_compiled
    yaml_files -.->|Miss| load_compiled

    load_compiled --> runtime
    runtime --> create_env
    runtime --> create_q
    runtime --> create_buffer
    runtime --> create_curriculum
    runtime --> create_exploration
    create_env --> create_population
    create_q --> create_population
    create_buffer --> create_population
    create_curriculum --> create_population
    create_exploration --> create_population

    create_population --> reset
    reset --> obs
    obs --> q_values
    q_values --> actions
    actions --> env_transition
    env_transition --> rewards
    env_transition --> next_obs
    env_transition --> dones
    rewards --> store
    next_obs --> store
    dones --> store
    store --> train

    train -->|No| obs
    train -->|Yes| sample
    sample --> update
    update --> obs

    dones --> curriculum_update
    curriculum_update --> checkpoint_save
    checkpoint_save --> db_log
    checkpoint_save --> tb_log

    style compiled fill:#fff9c4
    style obs fill:#e1f5fe
    style update fill:#ffccbc
    style checkpoint_save fill:#d1c4e9
```

## Module Dependency Graph

```mermaid
graph TB
    subgraph "Configuration Layer"
        config_hamlet[hamlet.py]
        config_bar[bar.py]
        config_cascade[cascade.py]
        config_affordance[affordance.py]
        config_training[training.py]
        config_substrate[substrate/config.py]
    end

    subgraph "Universe Layer"
        compiler[universe/compiler.py]
        compiled[universe/compiled.py]
        runtime[universe/runtime.py]
        symbol_table[universe/symbol_table.py]
        metadata[universe/dto/*]
    end

    subgraph "VFS Layer"
        vfs_schema[vfs/schema.py]
        vfs_registry[vfs/registry.py]
        vfs_obs_builder[vfs/observation_builder.py]
        vfs_adapter[universe/adapters/vfs_adapter.py]
    end

    subgraph "Substrate Layer"
        substrate_base[substrate/base.py]
        substrate_grid2d[substrate/grid2d.py]
        substrate_grid3d[substrate/grid3d.py]
        substrate_gridnd[substrate/gridnd.py]
        substrate_continuous[substrate/continuous.py]
        substrate_aspatial[substrate/aspatial.py]
        substrate_factory[substrate/factory.py]
    end

    subgraph "Environment Layer"
        env_vectorized[environment/vectorized_env.py]
        env_meter_dynamics[environment/meter_dynamics.py]
        env_cascade_engine[environment/cascade_engine.py]
        env_affordance_engine[environment/affordance_engine.py]
        env_reward_strategy[environment/reward_strategy.py]
        env_action_builder[environment/action_builder.py]
    end

    subgraph "Agent Layer"
        agent_networks[agent/networks.py]
    end

    subgraph "Training Layer"
        training_state[training/state.py]
        training_replay[training/replay_buffer.py]
        training_sequential[training/sequential_replay_buffer.py]
        training_checkpoint[training/checkpoint_utils.py]
    end

    subgraph "Population Layer"
        population_vectorized[population/vectorized.py]
        population_runtime_registry[population/runtime_registry.py]
    end

    subgraph "Curriculum Layer"
        curriculum_adversarial[curriculum/adversarial.py]
        curriculum_base[curriculum/base.py]
    end

    subgraph "Exploration Layer"
        exploration_adaptive[exploration/adaptive_intrinsic.py]
        exploration_rnd[exploration/rnd.py]
        exploration_epsilon[exploration/epsilon_greedy.py]
    end

    subgraph "Demo Layer"
        demo_runner[demo/runner.py]
        demo_database[demo/database.py]
        demo_unified_server[demo/unified_server.py]
        demo_live_inference[demo/live_inference.py]
    end

    config_hamlet --> compiler
    config_bar --> compiler
    config_cascade --> compiler
    config_affordance --> compiler
    config_training --> compiler
    config_substrate --> compiler

    compiler --> symbol_table
    compiler --> metadata
    compiler --> compiled
    compiler --> vfs_adapter
    compiled --> runtime

    vfs_schema --> vfs_registry
    vfs_schema --> vfs_obs_builder
    vfs_registry --> vfs_obs_builder
    vfs_obs_builder --> vfs_adapter
    vfs_adapter --> compiler

    config_substrate --> substrate_factory
    substrate_base --> substrate_grid2d
    substrate_base --> substrate_grid3d
    substrate_base --> substrate_gridnd
    substrate_base --> substrate_continuous
    substrate_base --> substrate_aspatial
    substrate_factory --> env_vectorized

    runtime --> env_vectorized
    substrate_factory --> env_vectorized
    vfs_registry --> env_vectorized
    vfs_obs_builder --> env_vectorized
    env_meter_dynamics --> env_vectorized
    env_cascade_engine --> env_vectorized
    env_affordance_engine --> env_vectorized
    env_reward_strategy --> env_vectorized
    env_action_builder --> env_vectorized

    agent_networks --> population_vectorized
    training_replay --> population_vectorized
    training_sequential --> population_vectorized
    training_state --> population_vectorized
    population_runtime_registry --> population_vectorized

    curriculum_base --> curriculum_adversarial
    curriculum_adversarial --> population_vectorized

    exploration_rnd --> exploration_adaptive
    exploration_epsilon --> exploration_adaptive
    exploration_adaptive --> population_vectorized

    env_vectorized --> population_vectorized
    population_vectorized --> demo_runner
    curriculum_adversarial --> demo_runner
    exploration_adaptive --> demo_runner
    demo_database --> demo_runner
    training_checkpoint --> demo_runner

    demo_runner --> demo_unified_server
    demo_runner --> demo_live_inference

    style compiler fill:#fff9c4
    style env_vectorized fill:#f3e5f5
    style population_vectorized fill:#c8e6c9
    style demo_runner fill:#d1c4e9
```

## Tensor Flow Through System

```mermaid
flowchart LR
    subgraph "Input Tensors"
        positions_in["Agent Positions<br/>[num_agents, position_dim]<br/>dtype: float32<br/>device: cuda"]
        meters_in["Meters<br/>[num_agents, meter_count]<br/>dtype: float32<br/>device: cuda"]
        actions_in["Actions<br/>[num_agents]<br/>dtype: long<br/>device: cuda"]
    end

    subgraph "Environment Processing"
        substrate_move["Substrate.move_agent()<br/>[num_agents, position_dim]"]
        meter_update["MeterDynamics.update()<br/>[num_agents, meter_count]"]
        cascade_apply["CascadeEngine.apply()<br/>[num_agents, meter_count]"]
        reward_compute["RewardStrategy.compute()<br/>[num_agents]"]
    end

    subgraph "VFS Processing"
        vfs_update["VFS Registry Update<br/>Update variable tensors"]
        vfs_read["VFS Read<br/>Concatenate observations"]
        obs_out["Observations<br/>[num_agents, obs_dim]<br/>dtype: float32<br/>device: cuda"]
    end

    subgraph "Network Processing"
        q_forward["Q-Network Forward<br/>obs → q_values"]
        q_values_out["Q-Values<br/>[num_agents, action_dim]<br/>dtype: float32<br/>device: cuda"]
        action_select["Action Selection<br/>ε-greedy"]
        actions_out["New Actions<br/>[num_agents]<br/>dtype: long<br/>device: cuda"]
    end

    subgraph "Training Processing"
        batch_sample["Sample Batch<br/>[batch_size, ...]"]
        q_train["Q-Network Training<br/>Forward + Backward"]
        grad_update["Gradient Update<br/>optimizer.step()"]
    end

    subgraph "Output Tensors"
        rewards_out["Rewards<br/>[num_agents]<br/>dtype: float32<br/>device: cuda"]
        dones_out["Dones<br/>[num_agents]<br/>dtype: bool<br/>device: cuda"]
        next_obs_out["Next Observations<br/>[num_agents, obs_dim]<br/>dtype: float32<br/>device: cuda"]
    end

    positions_in --> substrate_move
    actions_in --> substrate_move
    substrate_move --> positions_in

    meters_in --> meter_update
    meter_update --> cascade_apply
    cascade_apply --> meters_in

    meters_in --> reward_compute
    reward_compute --> rewards_out

    positions_in --> vfs_update
    meters_in --> vfs_update
    vfs_update --> vfs_read
    vfs_read --> obs_out

    obs_out --> q_forward
    q_forward --> q_values_out
    q_values_out --> action_select
    action_select --> actions_out

    obs_out --> batch_sample
    rewards_out --> batch_sample
    actions_out --> batch_sample
    batch_sample --> q_train
    q_train --> grad_update

    obs_out --> next_obs_out
    meters_in --> dones_out

    style positions_in fill:#e1f5fe
    style meters_in fill:#c8e6c9
    style obs_out fill:#fff9c4
    style q_values_out fill:#f3e5f5
    style rewards_out fill:#ffccbc
```

## Memory Layout

```mermaid
graph TB
    subgraph "GPU Memory (CUDA)"
        gpu_positions["Agent Positions<br/>4 agents × 2 dims × 4 bytes = 32B"]
        gpu_meters["Meters<br/>4 agents × 8 meters × 4 bytes = 128B"]
        gpu_observations["Observations<br/>4 agents × 29 dims × 4 bytes = 464B"]
        gpu_q_network["Q-Network Weights<br/>~26K params × 4 bytes = 104KB"]
        gpu_replay["Replay Buffer<br/>10K transitions × ~120 bytes = 1.2MB"]
        gpu_total["Total GPU Memory<br/>~2-3 GB (typical)"]
    end

    subgraph "CPU Memory"
        cpu_configs["Configuration DTOs<br/>~1 MB"]
        cpu_database["SQLite Database<br/>Growing (~100 MB per 10K episodes)"]
        cpu_checkpoints["Checkpoints<br/>~5 MB per checkpoint"]
    end

    subgraph "Disk Storage"
        disk_yaml["YAML Configs<br/>~50 KB"]
        disk_cache["Compiled Cache<br/>~500 KB"]
        disk_checkpoints["Checkpoint Files<br/>~5 MB × N checkpoints"]
        disk_database["Database File<br/>Growing with training"]
        disk_tensorboard["TensorBoard Logs<br/>~100 MB per 10K episodes"]
    end

    gpu_positions --> gpu_total
    gpu_meters --> gpu_total
    gpu_observations --> gpu_total
    gpu_q_network --> gpu_total
    gpu_replay --> gpu_total

    cpu_configs --> disk_yaml
    cpu_database --> disk_database
    cpu_checkpoints --> disk_checkpoints

    style gpu_total fill:#c8e6c9
    style cpu_database fill:#e1f5fe
    style disk_checkpoints fill:#d1c4e9
```

## Communication Patterns

```mermaid
sequenceDiagram
    participant C as Compiler
    participant R as Runtime
    participant E as Environment
    participant P as Population
    participant Q as Q-Network
    participant RB as ReplayBuffer
    participant CU as Curriculum
    participant EX as Exploration

    Note over C: Compile-Time (Once)
    C->>R: Compile configs → RuntimeUniverse

    Note over R,E: Initialization (Once per run)
    R->>E: Create Environment with metadata
    R->>P: Create Population with components

    Note over E,EX: Training Loop (Repeated)
    loop Episode
        P->>E: reset()
        E-->>P: initial_obs [num_agents, obs_dim]

        loop Step
            P->>Q: forward(obs)
            Q-->>P: q_values [num_agents, action_dim]

            P->>EX: select_action(q_values, epsilon)
            EX-->>P: actions [num_agents]

            P->>E: step(actions)
            E->>E: Update state (positions, meters)
            E->>E: Compute rewards
            E-->>P: obs', rewards, dones, info

            P->>EX: compute_intrinsic(obs)
            EX-->>P: intrinsic_rewards [num_agents]

            P->>RB: store(obs, action, reward, obs', done)

            alt Training Step
                P->>RB: sample(batch_size)
                RB-->>P: batch
                P->>Q: forward(batch.obs)
                Q-->>P: q_values
                P->>Q: backward() + optimizer.step()
            end
        end

        P->>CU: update(survival, dones)
        CU-->>P: new_stage, difficulty
        P->>EX: decay_epsilon()
    end
```

## Config → Runtime Data Transformation

```mermaid
flowchart TD
    subgraph "YAML Configs"
        yaml_bars["bars.yaml<br/>List of bar definitions"]
        yaml_cascades["cascades.yaml<br/>List of cascade definitions"]
        yaml_affordances["affordances.yaml<br/>List of affordance definitions"]
    end

    subgraph "Compilation Stage"
        parse["Parse YAMLs<br/>into DTOs"]
        validate["Validate<br/>cross-references"]
        optimize["Pre-compute<br/>tensors"]
    end

    subgraph "Compiled Artifacts"
        metadata["UniverseMetadata<br/>meter_name_to_index: dict<br/>affordance_id_to_index: dict"]
        opt_data["OptimizationData<br/>base_depletions: Tensor[meter_count]<br/>cascade_data: dict<br/>action_masks: Tensor[24, affordances]"]
    end

    subgraph "Runtime Structures"
        meters["Meters Tensor<br/>[num_agents, meter_count]<br/>Initialized to defaults"]
        affordances["Affordance Positions<br/>{id: Tensor[position_dim]}"]
        masks["Action Masks<br/>[num_agents, action_dim]<br/>From operating_hours"]
    end

    yaml_bars --> parse
    yaml_cascades --> parse
    yaml_affordances --> parse

    parse --> validate
    validate --> optimize

    optimize --> metadata
    optimize --> opt_data

    metadata --> meters
    metadata --> affordances
    opt_data --> meters
    opt_data --> masks

    style parse fill:#e1f5fe
    style metadata fill:#fff9c4
    style meters fill:#c8e6c9
```

## Error Propagation

```mermaid
flowchart TD
    source["Error Source"]

    compilation_error{"Compilation<br/>Error?"}
    runtime_error{"Runtime<br/>Error?"}
    training_error{"Training<br/>Error?"}

    subgraph "Compilation Errors"
        collect["CompilationErrorCollector"]
        format["Format with Source Map"]
        display_compile["Display Error + Hints"]
        exit_compile["Exit with Code 1"]
    end

    subgraph "Runtime Errors"
        catch_runtime["Catch Exception"]
        log_runtime["Log to Logger"]
        checkpoint_save["Save Emergency Checkpoint"]
        exit_runtime["Graceful Shutdown"]
    end

    subgraph "Training Errors"
        detect_nan["Detect NaN/Inf"]
        log_training["Log Warning"]
        clip_values["Clip/Skip Bad Batch"]
        continue_training["Continue Training"]
    end

    source --> compilation_error
    source --> runtime_error
    source --> training_error

    compilation_error -->|Yes| collect
    collect --> format
    format --> display_compile
    display_compile --> exit_compile

    runtime_error -->|Yes| catch_runtime
    catch_runtime --> log_runtime
    log_runtime --> checkpoint_save
    checkpoint_save --> exit_runtime

    training_error -->|Yes| detect_nan
    detect_nan --> log_training
    log_training --> clip_values
    clip_values --> continue_training

    style display_compile fill:#ffccbc
    style exit_runtime fill:#ffe0b2
    style continue_training fill:#c8e6c9
```

## Performance Characteristics

### Compilation Phase
- **Time**: 1-5 seconds (first run)
- **Memory**: ~50 MB
- **Cache Hit**: <100 ms
- **Output**: 500 KB - 2 MB compiled artifact

### Training Phase (per episode)
- **Time**: 0.1-2 seconds (depends on max_steps)
- **GPU Memory**: 2-3 GB
- **CPU Memory**: 200-500 MB
- **Throughput**: ~500-1000 steps/second (GPU)

### Checkpoint Operations
- **Save Time**: 100-500 ms
- **Load Time**: 200-800 ms
- **Size**: ~5 MB per checkpoint
- **Frequency**: Every 100 episodes

### Database Operations
- **Insert Time**: <1 ms per episode
- **Query Time**: 1-10 ms (recent episodes)
- **Growth Rate**: ~10 KB per episode
- **Total Size**: ~100 MB per 10K episodes

## Key Optimization Strategies

1. **Pre-computation**:
   - Base depletions → Tensor
   - Cascade data → Sorted structures
   - Action masks → 24×affordances table
   - Affordance positions → Tensor map

2. **Vectorization**:
   - All operations on `[num_agents, ...]` tensors
   - Parallel agent processing
   - GPU-native computations

3. **Caching**:
   - Compiled universe → Disk cache
   - Provenance-based invalidation
   - Lazy loading of large artifacts

4. **Batching**:
   - Experience replay batching
   - Sequential episode batching (LSTM)
   - Target network updates (every M steps)

5. **Memory Management**:
   - Circular replay buffer
   - Episode flushing (LSTM)
   - Checkpoint compression

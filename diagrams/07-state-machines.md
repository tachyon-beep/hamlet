# State Machine Diagrams

## Agent Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle: reset()

    Idle --> Moving: move action
    Idle --> Interacting: interact action
    Idle --> Idle: wait action

    Moving --> Idle: movement complete
    Moving --> Dead: meter depletes to 0

    Interacting --> InteractingMultiTick: multi-tick affordance
    Interacting --> Idle: instant affordance complete
    Interacting --> Dead: meter depletes to 0

    InteractingMultiTick --> InteractingMultiTick: progress++, per_tick effects
    InteractingMultiTick --> Idle: completion (required_ticks reached)
    InteractingMultiTick --> Idle: early exit (agent moves/changes action)
    InteractingMultiTick --> Idle: failure (availability lost)
    InteractingMultiTick --> Dead: meter depletes to 0

    Dead --> [*]: episode end

    note right of Idle
        Agent position stable
        All meters > 0
        Ready for action
    end note

    note right of Moving
        Position updating
        Energy cost applied
        Single-step transition
    end note

    note right of Interacting
        At affordance position
        Costs applied
        Effects applied
    end note

    note right of InteractingMultiTick
        Progress: ticks_completed / required_ticks
        Per-tick effects active
        Can resume if resumable=true
    end note

    note right of Dead
        Any meter ≤ 0
        Reward = 0.0
        Done flag = True
    end note
```

## Episode State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing: DemoRunner.run()

    Initializing --> CheckpointCheck: compile universe,\ncreate components

    CheckpointCheck --> LoadingCheckpoint: checkpoint exists
    CheckpointCheck --> EpisodeStart: no checkpoint

    LoadingCheckpoint --> EpisodeStart: load complete

    EpisodeStart --> Running: reset environment,\nreset population

    Running --> Running: step < max_steps\n& not all done
    Running --> EpisodeComplete: step >= max_steps\n| all done

    EpisodeComplete --> CurriculumUpdate: collect metrics

    CurriculumUpdate --> CheckpointDecision: update stages,\ndecay epsilon

    CheckpointDecision --> Saving: episode % 100 == 0
    CheckpointDecision --> NextEpisode: not checkpoint time

    Saving --> NextEpisode: checkpoint saved

    NextEpisode --> EpisodeStart: episode < max_episodes
    NextEpisode --> FinalSave: episode >= max_episodes

    FinalSave --> Cleanup: save final checkpoint

    Cleanup --> [*]: close DB,\nclose TensorBoard

    note right of Initializing
        Load configs
        Compile universe
        Create env, population,
        curriculum, exploration
    end note

    note right of Running
        Loop over steps:
        - Get observations
        - Select actions
        - Step environment
        - Store transitions
        - Train (if frequency met)
    end note

    note right of CurriculumUpdate
        Update performance trackers
        Check advance/retreat conditions
        Update difficulty multiplier
        Log transition events
    end note

    note right of Saving
        Flush all episodes
        Save Q-network state
        Save optimizer state
        Save replay buffer
        Save curriculum state
        Save affordance layout
    end note
```

## Multi-Tick Interaction State Machine

```mermaid
stateDiagram-v2
    [*] --> CheckRequirements: INTERACT action

    CheckRequirements --> CheckPosition: requirements met
    CheckRequirements --> Rejected: requirements not met

    CheckPosition --> CheckAvailability: at affordance position
    CheckPosition --> Rejected: not at position

    CheckAvailability --> CheckCapability: available\n(hours + constraints)
    CheckAvailability --> Rejected: not available

    CheckCapability --> InstantComplete: instant capability
    CheckCapability --> StartMultiTick: multi_tick capability
    CheckCapability --> Rejected: no valid capability

    StartMultiTick --> ApplyOnStart: ticks_completed = 0

    ApplyOnStart --> InProgress: on_start effects applied,\ncosts applied

    InProgress --> ApplyPerTick: each step,\nagent at same position
    InProgress --> EarlyExit: agent moves/changes action
    InProgress --> Failure: availability lost\n(hours or constraints)
    InProgress --> CheckCompletion: per_tick effects applied

    ApplyPerTick --> CheckCompletion: increment ticks_completed

    CheckCompletion --> ApplyOnCompletion: ticks_completed >= required_ticks
    CheckCompletion --> InProgress: ticks_completed < required_ticks

    ApplyOnCompletion --> Complete: on_completion effects,\ncompletion_bonus

    EarlyExit --> ApplyEarlyExit: early_exit_allowed = true
    EarlyExit --> ApplyFailure: early_exit_allowed = false

    ApplyEarlyExit --> Complete: on_early_exit effects

    Failure --> ApplyFailure: check failure conditions

    ApplyFailure --> Complete: on_failure effects

    InstantComplete --> Complete: on_start effects,\ncompletion_bonus

    Complete --> [*]: return to Idle
    Rejected --> [*]: return to Idle,\nno effects

    note right of InProgress
        State stored:
        - affordance_id
        - ticks_completed
        - resumable flag
    end note

    note right of CheckCompletion
        Progress shown in UI:
        ticks_completed / required_ticks
        e.g., "Working: 3/8"
    end note
```

## Curriculum Stage Transition State Machine

```mermaid
stateDiagram-v2
    [*] --> Stage1: initialization

    Stage1 --> Stage2: survival > 70%\n& entropy < 0.5\n& min_steps_at_stage met
    Stage1 --> Stage1: conditions not met

    Stage2 --> Stage3: survival > 70%\n& entropy < 0.5\n& min_steps_at_stage met
    Stage2 --> Stage1: survival < 30%
    Stage2 --> Stage2: neither condition

    Stage3 --> Stage4: survival > 70%\n& entropy < 0.5\n& min_steps_at_stage met
    Stage3 --> Stage2: survival < 30%
    Stage3 --> Stage3: neither condition

    Stage4 --> Stage5: survival > 70%\n& entropy < 0.5\n& min_steps_at_stage met
    Stage4 --> Stage3: survival < 30%
    Stage4 --> Stage4: neither condition

    Stage5 --> Stage5: max stage reached
    Stage5 --> Stage4: survival < 30%

    note right of Stage1
        Easiest difficulty
        depletion_multiplier = 0.1
        10x slower depletion
    end note

    note right of Stage3
        Normal difficulty
        depletion_multiplier = 1.0
        Standard depletion
    end note

    note right of Stage5
        Hardest difficulty
        depletion_multiplier = 5.0
        5x faster depletion
    end note

    note left of Stage2
        Advance Criteria:
        - Mean survival > 70% of max_steps
        - Action entropy < 0.5 (convergence)
        - At stage >= min_steps_at_stage

        Retreat Criteria:
        - Mean survival < 30% of max_steps
    end note
```

## Training Step State Machine

```mermaid
stateDiagram-v2
    [*] --> CheckFrequency: training step called

    CheckFrequency --> CheckBufferSize: step % train_frequency == 0
    CheckFrequency --> Skip: step % train_frequency != 0

    CheckBufferSize --> SampleBatch: buffer.size >= batch_size
    CheckBufferSize --> Skip: buffer.size < batch_size

    SampleBatch --> ForwardPass: batch sampled

    ForwardPass --> ComputeLoss: Q(s,a) computed

    ComputeLoss --> BackwardPass: TD loss = MSE(Q, target_Q)

    BackwardPass --> ClipGradients: loss.backward()

    ClipGradients --> OptimizerStep: max_norm = 10.0

    OptimizerStep --> CheckTargetUpdate: optimizer.step()

    CheckTargetUpdate --> UpdateTarget: training_step % target_update_frequency == 0
    CheckTargetUpdate --> LogMetrics: not update time

    UpdateTarget --> LogMetrics: target_network.load_state_dict(q_network.state_dict())

    LogMetrics --> [*]: log TD error, loss,\nmean Q-values

    Skip --> [*]: no training this step

    note right of SampleBatch
        Feedforward: random transitions
        LSTM: sequential episodes
        (length = sequence_length)
    end note

    note right of ComputeLoss
        Current: Q(s,a)
        Target: r + γ·max_a' Q_target(s',a')
        Loss: MSE(current, target)
    end note

    note right of UpdateTarget
        Soft update to stabilize learning
        Prevents moving target problem
        Updated every target_update_frequency steps
    end note
```

## Exploration Strategy State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing: create exploration strategy

    Initializing --> HighEpsilon: epsilon = epsilon_start\n(typically 1.0)

    HighEpsilon --> Selecting: select_action called

    Selecting --> RandomAction: uniform(0,1) < epsilon
    Selecting --> GreedyAction: uniform(0,1) >= epsilon

    RandomAction --> Storing: select valid random action
    GreedyAction --> Storing: argmax Q(s,a) from valid actions

    Storing --> Computing: action executed

    Computing --> CheckAnnealing: compute intrinsic reward\n(if RND enabled)

    CheckAnnealing --> Annealing: variance < threshold\n& mean_survival > 50
    CheckAnnealing --> NoAnnealing: conditions not met

    Annealing --> EpsilonDecay: intrinsic_weight *= 0.99
    NoAnnealing --> EpsilonDecay: keep intrinsic_weight

    EpsilonDecay --> LowEpsilon: epsilon *= epsilon_decay\n(e.g., 0.995)
    EpsilonDecay --> HighEpsilon: epsilon stays high

    LowEpsilon --> Selecting: epsilon approaches epsilon_min\n(e.g., 0.01)

    note right of HighEpsilon
        Early training:
        High exploration (random)
        epsilon ≈ 1.0
    end note

    note right of LowEpsilon
        Late training:
        High exploitation (greedy)
        epsilon ≈ 0.01
    end note

    note right of Computing
        Intrinsic reward:
        ||target(s) - predictor(s)||²
        Encourages novelty-seeking
    end note

    note right of Annealing
        Adaptive annealing:
        Reduce intrinsic weight when
        performance stabilizes
    end note
```

## Checkpoint Save/Load State Machine

```mermaid
stateDiagram-v2
    [*] --> CheckTrigger: checkpoint operation requested

    CheckTrigger --> SavePath: save operation
    CheckTrigger --> LoadPath: load operation

    SavePath --> FlushEpisodes: episode % 100 == 0

    FlushEpisodes --> CollectState: flush all agent episodes\nto replay buffer

    CollectState --> BuildCheckpoint: collect:\n- network weights\n- optimizer state\n- replay buffer\n- curriculum state\n- affordance layout

    BuildCheckpoint --> AddMetadata: version, episode,\ntimestamp, config_hash,\nsubstrate_metadata

    AddMetadata --> SerializeTorch: construct checkpoint dict

    SerializeTorch --> ComputeDigest: torch.save(checkpoint, path)

    ComputeDigest --> WriteDigest: SHA256(checkpoint_bytes)

    WriteDigest --> UpdateDB: .pt.sha256 file

    UpdateDB --> SaveComplete: set last_checkpoint\nin system_state

    SaveComplete --> [*]: checkpoint saved

    LoadPath --> FindLatest: scan checkpoint_dir

    FindLatest --> VerifyDigest: latest checkpoint_ep*.pt

    VerifyDigest --> LoadTorch: compare SHA256\n(warn if missing)

    LoadTorch --> ValidateVersion: safe_torch_load(path)

    ValidateVersion --> ValidateMetadata: check version\n(warn if != 3)

    ValidateMetadata --> ValidateHash: check substrate_metadata\n(position_dim, substrate_type)

    ValidateHash --> RestoreState: config_hash warning\nif mismatch

    RestoreState --> LoadComplete: restore:\n- network weights\n- optimizer state\n- curriculum state\n- affordance layout

    LoadComplete --> [*]: training resumed

    note right of FlushEpisodes
        CRITICAL: Prevents data loss
        Ensures LSTM episode sequences
        reach replay buffer before
        checkpointing
    end note

    note right of ValidateMetadata
        Breaking change check:
        Old checkpoints (v2) don't
        have substrate_metadata
        → error with migration hint
    end note
```

## Database Transaction States

```mermaid
stateDiagram-v2
    [*] --> Open: DemoDatabase.__init__

    Open --> WALMode: enable WAL mode\n(write-ahead logging)

    WALMode --> Ready: schema created

    Ready --> Writing: insert operation
    Ready --> Reading: query operation
    Ready --> Closing: close() called

    Writing --> Committing: execute INSERT

    Committing --> Ready: commit()

    Reading --> Ready: fetch results

    Closing --> Closed: conn.close()

    Closed --> [*]: database closed

    note right of WALMode
        Concurrent access:
        - Multiple readers OK
        - Single writer OK
        - Reader + writer OK
        Not safe for multiple writers
    end note

    note right of Writing
        Operations:
        - insert_episode
        - insert_affordance_visits
        - insert_position_heatmap
        - insert_recording_metadata
        - set_system_state
    end note

    note right of Reading
        Operations:
        - get_system_state
        - query recent episodes
        - query affordance transitions
        - query heatmap data
    end note
```

## LSTM Hidden State Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Uninitialized: network created

    Uninitialized --> Initialized: reset_hidden_state(batch_size)

    Initialized --> Zeroed: h = zeros[1, batch, 256]\nc = zeros[1, batch, 256]

    Zeroed --> ForwardPass: episode start

    ForwardPass --> Updated: LSTM processes observation

    Updated --> Persisted: (h, c) stored in network
    Updated --> ForwardPass: next step (same episode)
    Updated --> Zeroed: episode done (reset)
    Updated --> Checkpointed: checkpoint save

    Persisted --> Training: used for next forward pass

    Training --> Updated: hidden state evolves

    Checkpointed --> Saved: hidden_states saved to .pt

    Saved --> Restored: checkpoint load

    Restored --> ForwardPass: training resumed

    note right of Zeroed
        Reset on:
        - Episode start
        - Population reset
        Each agent has independent
        hidden state
    end note

    note right of Updated
        Hidden state carries
        temporal context across
        steps within episode

        NOT reset during episode
        (memory for POMDP)
    end note

    note right of Checkpointed
        Saved state:
        - h: [1, num_agents, 256]
        - c: [1, num_agents, 256]

        Restored on load to
        continue from exact state
    end note
```

## State Transition Summary

### Agent States
- **Idle**: Ready for action, all meters > 0
- **Moving**: Transitioning position (single step)
- **Interacting**: At affordance, applying effects
- **InteractingMultiTick**: Multi-step interaction with progress tracking
- **Dead**: Any meter ≤ 0, episode terminated

### Episode States
- **Initializing**: Loading configs, creating components
- **CheckpointCheck**: Scanning for existing checkpoint
- **LoadingCheckpoint**: Restoring from .pt file
- **Running**: Active training loop
- **EpisodeComplete**: All agents done or max_steps reached
- **CurriculumUpdate**: Adjusting difficulty
- **Saving**: Writing checkpoint to disk
- **Cleanup**: Closing resources

### Curriculum Stages
- **Stage 1-5**: Progressive difficulty levels
- **Advance**: Triggered by high survival (>70%) and low entropy (<0.5)
- **Retreat**: Triggered by low survival (<30%)
- **Stay**: Neither condition met

### Training States
- **CheckFrequency**: Determine if training this step
- **SampleBatch**: Pull from replay buffer
- **ForwardPass**: Compute Q-values
- **ComputeLoss**: TD error calculation
- **BackwardPass**: Gradient computation
- **ClipGradients**: Prevent exploding gradients
- **OptimizerStep**: Weight update
- **UpdateTarget**: Periodic target network sync

### Exploration States
- **HighEpsilon**: Early exploration phase (ε ≈ 1.0)
- **LowEpsilon**: Late exploitation phase (ε ≈ 0.01)
- **RandomAction**: Exploration (with probability ε)
- **GreedyAction**: Exploitation (with probability 1-ε)
- **Annealing**: Intrinsic weight reduction when stable

### Checkpoint States
- **SavePath**: Collect and serialize state
- **LoadPath**: Restore from disk
- **VerifyDigest**: SHA256 integrity check
- **ValidateVersion**: Compatibility check
- **ValidateMetadata**: Substrate dimension check

All state machines are deterministic and can be traced through logs for debugging.

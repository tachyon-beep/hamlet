"""Demo runner for multi-day training."""

from __future__ import annotations

import logging
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from townlet.agent.brain_config import BrainConfig, compute_brain_hash, load_brain_config
from townlet.config import HamletConfig
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.database import DemoDatabase
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.checkpoint_utils import (
    assert_checkpoint_dimensions,
    attach_universe_metadata,
    config_hash_warning,
    persist_checkpoint_digest,
    safe_torch_load,
    verify_checkpoint_digest,
)
from townlet.training.state import BatchedAgentState
from townlet.training.tensorboard_logger import TensorBoardLogger
from townlet.universe.compiler import UniverseCompiler

if TYPE_CHECKING:
    from townlet.universe.compiled import CompiledUniverse

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates multi-day demo training with checkpointing."""

    HEARTBEAT_INTERVAL = 10
    SUMMARY_INTERVAL = 50
    CHECKPOINT_INTERVAL = 100

    def __init__(
        self,
        config_dir: Path | str,
        db_path: Path | str,
        checkpoint_dir: Path | str,
        max_episodes: int | None = None,
        training_config_path: Path | str | None = None,
    ):
        """Initialize demo runner.

        Args:
            config_dir: Directory containing configuration pack
            db_path: Path to SQLite database
            checkpoint_dir: Directory for checkpoint files
            max_episodes: Maximum number of episodes to run (if None, reads from config YAML)
            training_config_path: Optional explicit path to training YAML file
        """
        self.config_dir = Path(config_dir)
        if training_config_path is None:
            self.training_config_path = self.config_dir / "training.yaml"
        else:
            self.training_config_path = Path(training_config_path)
        if not self.training_config_path.exists():
            raise FileNotFoundError(f"Training config not found: {self.training_config_path}")
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.compiled_universe: CompiledUniverse | None = None

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Pre-flight check: detect old checkpoints (Phase 4 breaking change)
        self._validate_checkpoint_compatibility()

        # Initialize database
        self.db = DemoDatabase(self.db_path)

        # Initialize TensorBoard logger
        # If checkpoint_dir is in a structured run (e.g., runs/L1_full_observability/2025-11-02_143022/checkpoints)
        # then tensorboard should be at sibling level, not inside checkpoints
        parent_dir = self.checkpoint_dir.parent
        if parent_dir.parent.parent.name == "runs" and self.checkpoint_dir.name == "checkpoints":
            # New structure: runs/LX_name/timestamp/checkpoints -> use runs/LX_name/timestamp/tensorboard
            tb_log_dir = parent_dir / "tensorboard"
        else:
            # Old structure or direct invocation: put tensorboard inside checkpoint_dir
            tb_log_dir = self.checkpoint_dir / "tensorboard"

        self.tb_logger = TensorBoardLogger(
            log_dir=tb_log_dir,
            flush_every=10,
        )

        # Load config using HamletConfig DTO (enforces no-defaults validation)
        self.hamlet_config = HamletConfig.load(self.config_dir, training_config_path=self.training_config_path)

        # Also load raw YAML for optional sections (e.g., recording)
        with open(self.training_config_path) as f:
            self.config = yaml.safe_load(f)

        # Set max_episodes: use provided value, otherwise read from config
        if max_episodes is not None:
            self.max_episodes = max_episodes
        else:
            self.max_episodes = self.hamlet_config.training.max_episodes

        self.current_episode = 0

        # Initialize components (will be created in run())
        self.env = None
        self.population = None
        self.curriculum = None
        self.exploration = None
        self.recorder = None  # Episode recorder (initialized if recording enabled)

        # TASK-005 Phase 1: Brain As Code configuration (loaded in run() if brain.yaml exists)
        self.brain_config: BrainConfig | None = None
        self.brain_hash: str | None = None

        # Shutdown flag
        self.should_shutdown = False
        # Signal handlers only work in main thread
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            signal.signal(signal.SIGINT, self._handle_shutdown)
        except ValueError:
            # Running in a worker thread (e.g., unified server)
            # Signal handling will be done by the orchestrator
            pass

    def _validate_checkpoint_compatibility(self) -> None:
        """Validate checkpoint directory doesn't contain old checkpoints.

        BREAKING CHANGE: Phase 4 changed checkpoint format.
        Old checkpoints (Version 2) will not load.

        Raises:
            ValueError: If old checkpoints detected
        """
        if not self.checkpoint_dir.exists():
            return  # No checkpoints yet (fresh start)

        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            return  # Empty directory (fresh start)

        # Check first checkpoint for substrate_metadata
        first_checkpoint_path = checkpoint_files[0]

        try:
            verify_checkpoint_digest(first_checkpoint_path, required=False)
            checkpoint = safe_torch_load(first_checkpoint_path, weights_only=False)

            # Validate checkpoint has required metadata
            if "substrate_metadata" not in checkpoint:
                raise ValueError(
                    f"Unsupported checkpoint format detected in {self.checkpoint_dir}.\n"
                    "Checkpoint missing 'substrate_metadata' field.\n"
                    "\n"
                    "Action required:\n"
                    f"  1. Delete checkpoint directory: {self.checkpoint_dir}\n"
                    "  2. Retrain model from scratch\n"
                    "\n"
                    f"Detected checkpoint: {first_checkpoint_path.name}"
                )
        except Exception as e:
            # If we can't load checkpoint, let the normal loading code handle it
            # (might be corrupted, wrong format, etc.)
            if "Unsupported checkpoint format" in str(e):
                raise  # Re-raise our validation error
            # Otherwise ignore (will fail later during actual load)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_shutdown = True

    def _cleanup(self):
        """Internal cleanup method for resources.

        Safe to call multiple times (idempotent).
        Handles partial initialization gracefully.
        """
        # Shutdown recorder if enabled
        if hasattr(self, "recorder") and self.recorder is not None:
            logger.info("Shutting down episode recorder...")
            try:
                self.recorder.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down recorder: {e}")

        # Close database connection
        if hasattr(self, "db"):
            try:
                self.db.close()
            except Exception as e:
                logger.warning(f"Error closing database: {e}")

        # Close TensorBoard logger
        if hasattr(self, "tb_logger"):
            try:
                self.tb_logger.close()
            except Exception as e:
                logger.warning(f"Error closing TensorBoard logger: {e}")

    def __enter__(self):
        """Enter context manager - return self for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup resources.

        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred

        Returns:
            False to propagate exceptions (don't suppress)
        """
        self._cleanup()
        return False  # Don't suppress exceptions

    def flush_all_agents(self):
        """Flush all agents' episodes to replay buffer before checkpoint."""
        if not self.population:
            return

        for agent_idx in range(self.population.num_agents):
            self.population.flush_episode(agent_idx)

    def save_checkpoint(self, universe: CompiledUniverse | None = None):
        """Save checkpoint at current episode."""
        # P1.1 Phase 5: Flush all agents before checkpoint
        self.flush_all_agents()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode:05d}.pt"

        # P1.1 Phase 2: Use population's complete checkpoint state
        checkpoint: dict[str, Any] = {
            "version": 3,  # Phase 5: Version 3 includes substrate metadata
            "episode": self.current_episode,
            "timestamp": time.time(),
        }

        # Phase 5: Add substrate metadata for validation
        if self.env:
            checkpoint["substrate_metadata"] = {
                "position_dim": self.env.substrate.position_dim,
                "substrate_type": type(self.env.substrate).__name__,
            }

        # Add full population state (includes q_network, optimizer, replay_buffer, etc.)
        if self.population:
            checkpoint["population_state"] = self.population.get_checkpoint_state()

        # P1.1 Phase 3: Add curriculum state (agent stages, performance trackers)
        if self.curriculum:
            checkpoint["curriculum_state"] = self.curriculum.state_dict()

        # P1.1 Phase 4: Add affordance layout (grid positions)
        if self.env:
            checkpoint["affordance_layout"] = self.env.get_affordance_positions()

        # P1.1 Phase 6: Add agent_ids for multi-agent coordination
        if self.population:
            checkpoint["agent_ids"] = self.population.agent_ids

        # Add epsilon for inference server display (use population's epsilon getter for all exploration types)
        if self.population:
            checkpoint["epsilon"] = self.population._get_current_epsilon_value()

        # Persist the training configuration for provenance
        checkpoint["training_config"] = self.config
        checkpoint["config_dir"] = str(self.config_dir)

        # TASK-005 Phase 1: Add brain_hash for checkpoint provenance
        if self.brain_hash is not None:
            checkpoint["brain_hash"] = self.brain_hash

        if universe is None:
            universe = self.compiled_universe
        if universe is not None:
            attach_universe_metadata(checkpoint, universe)

        torch.save(checkpoint, checkpoint_path)
        persist_checkpoint_digest(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update system state
        self.db.set_system_state("last_checkpoint", str(checkpoint_path))

    def load_checkpoint(self, universe: CompiledUniverse | None = None) -> int | None:
        """Load latest checkpoint if exists.

        Returns:
            Episode number of loaded checkpoint, or None if no checkpoint
        """
        # Find latest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return None

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")

        verify_checkpoint_digest(latest_checkpoint, required=False)
        checkpoint = safe_torch_load(latest_checkpoint, weights_only=False)

        if universe is None:
            universe = self.compiled_universe
        if universe is not None:
            warning = config_hash_warning(checkpoint, universe)
            if warning:
                logger.warning(warning)
            assert_checkpoint_dimensions(checkpoint, universe)

        # P1.1: Check checkpoint version
        checkpoint_version = checkpoint.get("version")
        if checkpoint_version != 3:
            raise ValueError(f"Unsupported checkpoint version: {checkpoint_version}\n" f"Expected version 3. Please retrain from scratch.")

        self.current_episode = checkpoint["episode"]

        # P1.1 Phase 2: Load full population state
        if "population_state" in checkpoint and self.population:
            self.population.load_checkpoint_state(checkpoint["population_state"])

        # P1.1 Phase 3: Load curriculum state (agent stages, performance trackers)
        if "curriculum_state" in checkpoint and self.curriculum:
            self.curriculum.load_state_dict(checkpoint["curriculum_state"])

        # P1.1 Phase 4: Load affordance layout (grid positions)
        if "affordance_layout" in checkpoint and self.env:
            self.env.set_affordance_positions(checkpoint["affordance_layout"])

        # P1.1 Phase 6: Restore agent_ids for multi-agent coordination
        if "agent_ids" in checkpoint and self.population:
            self.population.agent_ids = checkpoint["agent_ids"]

        logger.info(f"Resumed from episode {self.current_episode}")
        return self.current_episode

    def run(self):
        """Run demo training loop."""
        logger.info(f"Starting demo runner: {self.max_episodes} episodes")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

        # PDR-002: No-defaults validation done automatically by HamletConfig DTO
        # All required parameters validated at load time (lines 109-110)

        # Initialize training components
        device_str = self.hamlet_config.training.device
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")

        # Compile universe once and keep runtime view
        compiler = UniverseCompiler()
        self.compiled_universe = compiler.compile(self.config_dir)
        runtime_universe = self.compiled_universe.to_runtime()

        # Extract config parameters from DTOs (all required, validated at load time)
        num_agents = self.hamlet_config.population.num_agents
        # Get grid_size from compiled metadata (substrate.yaml is source of truth)
        grid_size = runtime_universe.metadata.grid_size
        partial_observability = self.hamlet_config.environment.partial_observability
        vision_range = self.hamlet_config.environment.vision_range
        enable_temporal_mechanics = self.hamlet_config.environment.enable_temporal_mechanics

        # Create environment from compiled universe
        self.env = VectorizedHamletEnv.from_universe(
            self.compiled_universe,
            num_agents=num_agents,
            device=device,
        )

        # Dimensions sourced from compiled metadata
        obs_dim = runtime_universe.metadata.observation_dim
        action_dim = runtime_universe.metadata.action_count

        # Create curriculum (all params required per PDR-002)
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=self.hamlet_config.curriculum.max_steps_per_episode,
            survival_advance_threshold=self.hamlet_config.curriculum.survival_advance_threshold,
            survival_retreat_threshold=self.hamlet_config.curriculum.survival_retreat_threshold,
            entropy_gate=self.hamlet_config.curriculum.entropy_gate,
            min_steps_at_stage=self.hamlet_config.curriculum.min_steps_at_stage,
            device=device,
        )

        # Create exploration (all params required per PDR-002)
        # Conditionally pass active_mask based on mask_unused_obs config
        active_mask = self.env.observation_activity.active_mask if self.hamlet_config.population.mask_unused_obs else None
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=self.hamlet_config.exploration.embed_dim,
            rnd_training_batch_size=self.hamlet_config.training.batch_size,  # Use main batch_size from config
            initial_intrinsic_weight=self.hamlet_config.exploration.initial_intrinsic_weight,
            variance_threshold=self.hamlet_config.exploration.variance_threshold,
            min_survival_fraction=self.hamlet_config.exploration.min_survival_fraction,
            max_episode_length=self.hamlet_config.curriculum.max_steps_per_episode,
            survival_window=self.hamlet_config.exploration.survival_window,
            epsilon_start=self.hamlet_config.training.epsilon_start,
            epsilon_decay=self.hamlet_config.training.epsilon_decay,
            epsilon_min=self.hamlet_config.training.epsilon_min,
            device=device,
            active_mask=active_mask,
        )

        # Get population parameters from config (all required per PDR-002)
        learning_rate = self.hamlet_config.population.learning_rate
        gamma = self.hamlet_config.population.gamma
        replay_buffer_capacity = self.hamlet_config.population.replay_buffer_capacity
        network_type = self.hamlet_config.population.network_type  # 'simple' or 'recurrent'
        vision_window_size = 2 * vision_range + 1  # 5 for vision_range=2

        # Get training hyperparameters from config (all required per PDR-002)
        train_frequency = self.hamlet_config.training.train_frequency
        target_update_frequency = self.hamlet_config.training.target_update_frequency
        batch_size = self.hamlet_config.training.batch_size
        sequence_length = self.hamlet_config.training.sequence_length
        max_grad_norm = self.hamlet_config.training.max_grad_norm
        use_double_dqn = self.hamlet_config.training.use_double_dqn

        # Create agent IDs
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # Load brain.yaml (REQUIRED for all config packs)
        brain_yaml_path = self.config_dir / "brain.yaml"
        logger.info(f"Loading brain configuration from {brain_yaml_path}")
        brain_config = load_brain_config(self.config_dir)
        brain_hash = compute_brain_hash(brain_config)
        logger.info(f"Brain config loaded: {brain_config.description}")
        logger.info(f"Brain hash: {brain_hash[:16]}... (SHA256)")

        # Store brain_config and brain_hash for checkpoint provenance
        self.brain_config = brain_config
        self.brain_hash = brain_hash

        # Create population (brain_config provides network/optimizer/Q-learning parameters)
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,  # None (managed by brain.yaml)
            gamma=gamma,  # None (managed by brain.yaml)
            replay_buffer_capacity=replay_buffer_capacity,  # None (managed by brain.yaml)
            network_type=network_type,
            vision_window_size=vision_window_size,
            tb_logger=self.tb_logger,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,  # None (managed by brain.yaml)
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_grad_norm=max_grad_norm,
            use_double_dqn=use_double_dqn,  # None (managed by brain.yaml)
            brain_config=brain_config,
            max_episodes=self.max_episodes,  # For PER beta annealing
            max_steps_per_episode=self.hamlet_config.curriculum.max_steps_per_episode,  # For PER beta annealing
        )

        self.curriculum.initialize_population(num_agents)

        # Initialize episode recorder if enabled
        recording_cfg = self.config.get("recording", {})
        if recording_cfg.get("enabled", False):
            from townlet.recording.recorder import EpisodeRecorder

            # Create recording output directory
            recording_output_dir = self.checkpoint_dir / recording_cfg.get("output_dir", "recordings")
            recording_output_dir.mkdir(parents=True, exist_ok=True)

            self.recorder = EpisodeRecorder(
                config=recording_cfg,
                output_dir=recording_output_dir,
                database=self.db,
                curriculum=self.curriculum,
            )
            logger.info(f"Episode recording enabled: {recording_output_dir}")
        else:
            logger.info("Episode recording disabled")

        # Try to resume from checkpoint
        loaded_episode = self.load_checkpoint(self.compiled_universe)
        if loaded_episode is not None:
            self.current_episode = loaded_episode + 1

        # Phase 4 - Log hyperparameters to TensorBoard
        self.hparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "network_type": network_type,
            "replay_buffer_capacity": replay_buffer_capacity,
            "grid_size": grid_size,
            "partial_observability": partial_observability,
            "vision_range": vision_range,
            "enable_temporal": enable_temporal_mechanics,
            "initial_intrinsic_weight": self.hamlet_config.exploration.initial_intrinsic_weight,
            "variance_threshold": self.hamlet_config.exploration.variance_threshold,
            "max_steps_per_episode": self.hamlet_config.curriculum.max_steps_per_episode,
        }
        # Note: final metrics will be logged at end of training
        self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics={})

        # Check for shutdown before starting training loop
        if self.should_shutdown:
            logger.info("[Training] Shutdown requested during initialization, exiting before training starts")
            return

        # Mark training started
        self.db.set_system_state("training_status", "running")
        self.db.set_system_state("start_time", str(time.time()))

        # Training loop
        try:
            while self.current_episode < self.max_episodes and not self.should_shutdown:
                episode_start = time.time()

                # Generalization test at episode 5000
                if self.current_episode == 5000:
                    logger.info("=" * 60)
                    logger.info("GENERALIZATION TEST: Randomizing affordance positions")
                    logger.info("=" * 60)

                    # Store old positions
                    old_positions = self.env.get_affordance_positions()
                    logger.info(f"Old positions: {old_positions}")

                    # Randomize
                    self.env.randomize_affordance_positions()

                    # Store new positions
                    new_positions = self.env.get_affordance_positions()
                    logger.info(f"New positions: {new_positions}")

                    # Mark in database
                    import json

                    self.db.set_system_state("affordance_randomization_episode", "5000")
                    self.db.set_system_state("old_affordance_positions", json.dumps(old_positions))
                    self.db.set_system_state("new_affordance_positions", json.dumps(new_positions))

                # Reset environment and population
                # NOTE: Only call population.reset(), which internally calls env.reset()
                # Calling both causes double reset which breaks affordance randomization
                self.population.reset()

                # Run episode
                num_agents = self.population.num_agents
                max_steps = self.hamlet_config.curriculum.max_steps_per_episode
                episode_reward = torch.zeros(num_agents, device=self.env.device)
                episode_extrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                episode_intrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                final_meters = [None for _ in range(num_agents)]
                affordance_visits = [defaultdict(int) for _ in range(num_agents)]
                custom_action_uses = [defaultdict(int) for _ in range(num_agents)]  # Track REST, MEDITATE, etc.
                # NEW: Transition tracking
                affordance_transitions = [defaultdict(lambda: defaultdict(int)) for _ in range(num_agents)]
                last_affordance = [None for _ in range(num_agents)]
                last_agent_state: BatchedAgentState | None = None

                for step in range(max_steps):
                    # Check for shutdown request every 10 steps for faster Ctrl+C response
                    if step % 10 == 0 and self.should_shutdown:
                        logger.info(f"[Training] Shutdown requested during episode {self.current_episode + 1}, step {step}/{max_steps}")
                        break

                    agent_state = self.population.step_population(self.env)
                    last_agent_state = agent_state

                    if self.curriculum.transition_events:
                        self.tb_logger.log_curriculum_transitions(
                            episode=self.current_episode,
                            events=list(self.curriculum.transition_events),
                        )
                        # Log curriculum transitions to console
                        for event in self.curriculum.transition_events:
                            logger.info(
                                f"🎓 Curriculum {event['type'].upper()}: Stage {event['from_stage']} → {event['to_stage']} "
                                f"(Survival: {event['survival_rate']:.1%}, Entropy: {event['entropy']:.3f})"
                            )
                        self.curriculum.transition_events.clear()

                    # Note: agent_state.rewards contains COMBINED rewards (extrinsic + weighted intrinsic)
                    # We need to extract pure extrinsic for separate logging
                    intrinsic_weight = self.exploration.get_intrinsic_weight() if hasattr(self.exploration, "get_intrinsic_weight") else 1.0
                    weighted_intrinsic = agent_state.intrinsic_rewards * intrinsic_weight
                    extrinsic_only = agent_state.rewards - weighted_intrinsic

                    # Accumulate rewards for episode totals
                    episode_reward += agent_state.rewards  # Combined (what agent actually receives)
                    episode_extrinsic_reward += extrinsic_only  # Pure extrinsic (for analysis)
                    episode_intrinsic_reward += agent_state.intrinsic_rewards  # Unweighted intrinsic (for analysis)

                    if "successful_interactions" in agent_state.info:
                        for agent_idx, affordance_name in agent_state.info["successful_interactions"].items():
                            if 0 <= agent_idx < num_agents:
                                # Existing: count tracking
                                affordance_visits[agent_idx][affordance_name] += 1

                                # NEW: transition tracking
                                prev = last_affordance[agent_idx]
                                if prev is not None:
                                    # Record transition: prev → current
                                    affordance_transitions[agent_idx][prev][affordance_name] += 1

                                # Update last affordance for next transition
                                last_affordance[agent_idx] = affordance_name

                    # Track custom action uses (REST, MEDITATE, etc.)
                    for idx in range(num_agents):
                        action_id = int(agent_state.actions[idx].item())
                        # Custom actions start after substrate actions
                        if action_id >= self.env.action_space.substrate_action_count:
                            action = self.env.action_space.get_action_by_id(action_id)
                            custom_action_uses[idx][action.name] += 1

                    for idx in range(num_agents):
                        if agent_state.dones[idx] and final_meters[idx] is None:
                            final_meters[idx] = self.env.meters[idx].detach().cpu()

                    # Record step if recording enabled (only agent 0 for now)
                    if self.recorder is not None:
                        # Get temporal mechanics fields if available
                        time_of_day = agent_state.info.get("time_of_day", [None] * num_agents)[0]
                        interaction_progress = agent_state.info.get("interaction_progress", [None] * num_agents)[0]

                        # Get action masks for debugging (which actions were valid)
                        action_masks = self.env.get_action_masks()[0]  # Agent 0 action masks

                        self.recorder.record_step(
                            step=step,
                            positions=self.env.positions[0],  # Agent 0 position
                            meters=self.env.meters[0],  # Agent 0 meters
                            action=(
                                agent_state.actions[0].item() if hasattr(agent_state.actions[0], "item") else int(agent_state.actions[0])
                            ),
                            reward=float(extrinsic_only[0].item()),
                            intrinsic_reward=float(agent_state.intrinsic_rewards[0].item()),
                            done=bool(agent_state.dones[0].item()),
                            q_values=agent_state.info.get("q_values", [None] * num_agents)[0],
                            epsilon=float(agent_state.epsilons[0].item()),
                            action_masks=action_masks,
                            time_of_day=time_of_day,
                            interaction_progress=interaction_progress,
                        )

                    if torch.all(agent_state.dones).item():
                        break

                if last_agent_state is None:
                    continue

                for idx in range(num_agents):
                    if final_meters[idx] is None:
                        final_meters[idx] = self.env.meters[idx].detach().cpu()

                step_counts = last_agent_state.info.get("step_counts", self.population.episode_step_counts.clone())
                step_counts = step_counts.to(self.env.device)
                curriculum_survival_tensor = step_counts.float()
                curriculum_done_tensor = torch.ones_like(last_agent_state.dones, dtype=torch.bool, device=self.env.device)
                self.population.update_curriculum_tracker(curriculum_survival_tensor, curriculum_done_tensor)

                # P1.2: Flush episode if agent survived to max_steps (recurrent networks only)
                # Without this, successful episodes never reach replay buffer → memory leak + data loss
                # CRITICAL: Loop over all agents to support multi-agent configs (not just agent 0)
                for agent_idx in range(self.population.num_agents):
                    if not last_agent_state.dones[agent_idx]:  # Agent survived to max_steps without dying
                        self.population.flush_episode(agent_idx=agent_idx)

                epsilon_value = (
                    self.exploration.rnd.epsilon if hasattr(self.exploration, "rnd") else getattr(self.exploration, "epsilon", 0.0)
                )
                intrinsic_weight_value = (
                    self.exploration.get_intrinsic_weight() if hasattr(self.exploration, "get_intrinsic_weight") else 0.0
                )

                episode_reward_cpu = episode_reward.detach().cpu()
                episode_extrinsic_cpu = episode_extrinsic_reward.detach().cpu()
                episode_intrinsic_cpu = episode_intrinsic_reward.detach().cpu()
                step_counts_cpu = step_counts.detach().cpu().long()
                stages_cpu = self.curriculum.tracker.agent_stages.detach().cpu()

                # Log metrics to database
                episode_timestamp = time.time()
                self.db.insert_episode(
                    episode_id=self.current_episode,
                    timestamp=episode_timestamp,
                    survival_time=int(step_counts_cpu[0].item()),
                    total_reward=float(episode_reward_cpu[0].item()),
                    extrinsic_reward=float(episode_extrinsic_cpu[0].item()),
                    intrinsic_reward=float(episode_intrinsic_cpu[0].item()),
                    intrinsic_weight=intrinsic_weight_value,
                    curriculum_stage=int(stages_cpu[0].item()),
                    epsilon=epsilon_value,
                )

                # NEW: Insert affordance transitions for agent 0
                if affordance_transitions[0]:
                    # Convert nested defaultdict to regular dict for database insertion
                    transitions_dict = {from_aff: dict(to_affs) for from_aff, to_affs in affordance_transitions[0].items()}
                    self.db.insert_affordance_visits(episode_id=self.current_episode, transitions=transitions_dict)

                # Finish episode recording if enabled
                if self.recorder is not None:
                    from townlet.recording.data_structures import EpisodeMetadata

                    metadata = EpisodeMetadata(
                        episode_id=self.current_episode,
                        survival_steps=int(step_counts_cpu[0].item()),
                        total_reward=float(episode_reward_cpu[0].item()),
                        extrinsic_reward=float(episode_extrinsic_cpu[0].item()),
                        intrinsic_reward=float(episode_intrinsic_cpu[0].item()),
                        curriculum_stage=int(stages_cpu[0].item()),
                        epsilon=epsilon_value,
                        intrinsic_weight=intrinsic_weight_value,
                        timestamp=episode_timestamp,
                        affordance_layout=self.env.get_affordance_positions(),
                        affordance_visits={k: v for k, v in affordance_visits[0].items()},  # Agent 0 visits
                        custom_action_uses={k: v for k, v in custom_action_uses[0].items()},  # Agent 0 custom actions
                    )
                    self.recorder.finish_episode(metadata)

                agent_metrics = []
                for idx, agent_id in enumerate(self.population.agent_ids):
                    agent_metrics.append(
                        {
                            "agent_id": agent_id,
                            "survival_time": int(step_counts_cpu[idx].item()),
                            "total_reward": float(episode_reward_cpu[idx].item()),
                            "extrinsic_reward": float(episode_extrinsic_cpu[idx].item()),
                            "intrinsic_reward": float(episode_intrinsic_cpu[idx].item()),
                            "curriculum_stage": int(stages_cpu[idx].item()),
                            "epsilon": epsilon_value,
                            "intrinsic_weight": intrinsic_weight_value,
                        }
                    )

                # Log to TensorBoard (Phase 1 - Episode metrics)
                self.tb_logger.log_multi_agent_episode(
                    episode=self.current_episode,
                    agents=agent_metrics,
                )

                # Phase 2 - Training metrics (if training occurred this episode)
                training_metrics = self.population.get_training_metrics()
                if training_metrics["training_step"] > 0:
                    self.tb_logger.log_training_step(
                        step=training_metrics["training_step"],
                        td_error=training_metrics["td_error"],
                        loss=training_metrics["loss"],
                    )

                # Phase 3 - Meter dynamics and affordance usage
                # Get actual meter names from config (supports variable meter counts)
                meter_names = self.env.bars_config.meter_names
                for idx, agent_id in enumerate(self.population.agent_ids):
                    meters_tensor = final_meters[idx]
                    if meters_tensor is not None:
                        meter_dict = {name: meters_tensor[i].item() for i, name in enumerate(meter_names)}
                        self.tb_logger.log_meters(
                            episode=self.current_episode,
                            step=int(step_counts_cpu[idx].item()),
                            meters=meter_dict,
                            agent_id=agent_id,
                        )

                for idx, visits in enumerate(affordance_visits):
                    if visits:
                        self.tb_logger.log_affordance_usage(
                            episode=self.current_episode,
                            affordance_counts=dict(visits),
                            agent_id=self.population.agent_ids[idx],
                        )

                for idx, uses in enumerate(custom_action_uses):
                    if uses:
                        self.tb_logger.log_custom_action_usage(
                            episode=self.current_episode,
                            action_counts=dict(uses),
                            agent_id=self.population.agent_ids[idx],
                        )

                # Heartbeat log every HEARTBEAT_INTERVAL episodes
                if self.current_episode % self.HEARTBEAT_INTERVAL == 0:
                    elapsed = time.time() - episode_start
                    stage_overview = "/".join(str(int(s.item())) for s in stages_cpu)

                    # Calculate reward breakdown
                    total_reward = episode_reward_cpu[0].item()
                    extrinsic_reward = episode_extrinsic_cpu[0].item()
                    intrinsic_reward = episode_intrinsic_cpu[0].item()
                    weighted_intrinsic = intrinsic_reward * intrinsic_weight_value

                    logger.info(
                        f"Episode {self.current_episode}/{self.max_episodes} | "
                        f"Survival: {int(step_counts_cpu[0].item())} steps | "
                        f"Reward: {total_reward:.2f} (Extrinsic: {extrinsic_reward:.2f}, "
                        f"Intrinsic: {intrinsic_reward:.2f}×{intrinsic_weight_value:.3f}={weighted_intrinsic:.2f}) | "
                        f"ε: {epsilon_value:.3f} | "
                        f"Stage: {stage_overview} | "
                        f"Time: {elapsed:.2f}s"
                    )

                # Detailed summary every SUMMARY_INTERVAL episodes
                if self.current_episode % self.SUMMARY_INTERVAL == 0:
                    # Get training metrics
                    training_metrics = self.population.get_training_metrics()

                    # Get final meters for agent 0
                    final_meter_values = {}
                    if final_meters[0] is not None:
                        # Get actual meter names from config (supports variable meter counts)
                        meter_names = self.env.bars_config.meter_names
                        final_meter_values = {name: final_meters[0][i].item() for i, name in enumerate(meter_names)}

                    # Get affordance usage for agent 0 with tick counts and completed interactions
                    if affordance_visits[0]:
                        summary_parts = []
                        for name, tick_count in affordance_visits[0].items():
                            # Get duration_ticks to calculate completed interactions
                            try:
                                duration = self.env.affordance_engine.get_duration_ticks(name)
                                completed = tick_count // duration
                                summary_parts.append(f"{name}: {tick_count} ({completed})")
                            except Exception:
                                # Fallback if duration lookup fails
                                summary_parts.append(f"{name}: {tick_count}")
                        affordance_summary = ", ".join(summary_parts)
                    else:
                        affordance_summary = "None"

                    logger.info("=" * 80)
                    logger.info(f"📊 SUMMARY - Episode {self.current_episode}/{self.max_episodes}")
                    logger.info("-" * 80)
                    logger.info(f"Performance:    Survival: {int(step_counts_cpu[0].item())} steps | Stage: {stage_overview}")
                    logger.info(
                        f"Rewards:        Total: {total_reward:.2f} | "
                        f"Extrinsic: {extrinsic_reward:.2f} | Intrinsic: {weighted_intrinsic:.2f}"
                    )
                    # Check if annealing would trigger (for status display)
                    annealing_active = self.exploration.should_anneal() if hasattr(self.exploration, "should_anneal") else False
                    annealing_status = "🔻 ANNEALING" if annealing_active else "✓ exploring"
                    logger.info(
                        f"Exploration:    ε: {epsilon_value:.3f} | Intrinsic Weight: {intrinsic_weight_value:.3f} | {annealing_status}"
                    )
                    if training_metrics["training_step"] > 0:
                        logger.info(
                            f"Training:       Steps: {training_metrics['training_step']} | "
                            f"Loss: {training_metrics['loss']:.4f} | TD Error: {training_metrics['td_error']:.4f}"
                        )
                    if final_meter_values:
                        logger.info(
                            f"Final Meters:   Energy: {final_meter_values.get('energy', 0):.2f} | "
                            f"Health: {final_meter_values.get('health', 0):.2f} | "
                            f"Money: ${final_meter_values.get('money', 0) * 100:.1f}"
                        )
                    logger.info(f"Affordances:    {affordance_summary}")

                    # Log custom action usage (REST, MEDITATE, etc.)
                    if custom_action_uses[0]:
                        custom_summary_parts = [f"{name}: {count}" for name, count in custom_action_uses[0].items()]
                        custom_summary = ", ".join(custom_summary_parts)
                        logger.info(f"Custom Actions: {custom_summary}")

                    # Log total interaction ticks for debugging
                    total_interaction_ticks = sum(affordance_visits[0].values()) if affordance_visits[0] else 0
                    total_custom_actions = sum(custom_action_uses[0].values()) if custom_action_uses[0] else 0
                    total_interactions = total_interaction_ticks + total_custom_actions
                    logger.info(
                        f"Interactions:   {total_interaction_ticks} affordance ticks + {total_custom_actions} custom actions = "
                        f"{total_interactions}/{int(step_counts_cpu[0].item())} steps "
                        f"({100 * total_interactions / max(1, int(step_counts_cpu[0].item())):.1f}%)"
                    )
                    logger.info("=" * 80)

                # Checkpoint every CHECKPOINT_INTERVAL episodes
                if self.current_episode % self.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(self.compiled_universe)

                # Decay epsilon for next episode and sync telemetry
                self.exploration.decay_epsilon()
                self.population.sync_exploration_metrics()

                self.current_episode += 1

        finally:
            # Save final checkpoint
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint(self.compiled_universe)

            # Phase 4 - Log final metrics with hyperparameters
            if self.population is not None:
                final_metrics = {
                    "final_episode": self.current_episode,
                    "total_training_steps": self.population.total_steps,
                }
                # Re-log hyperparameters with final metrics for comparison
                if hasattr(self, "hparams"):
                    self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics=final_metrics)

            self.db.set_system_state("training_status", "completed")

            # Use extracted cleanup method
            self._cleanup()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # Get paths from environment or args
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "configs/test"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "demo_state.db"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints"
    # If max_episodes provided via CLI, use it; otherwise pass None to read from config
    max_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else None

    config_path = Path(config_arg)
    if config_path.is_dir():
        config_dir = config_path
        training_config = config_dir / "training.yaml"
    else:
        config_dir = config_path.parent
        training_config = config_path

    runner = DemoRunner(
        config_dir=config_dir,
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        max_episodes=max_episodes,  # None = read from config
        training_config_path=training_config,
    )
    runner.run()

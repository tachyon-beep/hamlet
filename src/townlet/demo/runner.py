"""Demo runner for multi-day training."""

import logging
import signal
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.database import DemoDatabase
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.state import BatchedAgentState
from townlet.training.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class DemoRunner:
    """Orchestrates multi-day demo training with checkpointing."""

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

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        # Load config
        with open(self.training_config_path) as f:
            self.config = yaml.safe_load(f)

        # Set max_episodes: use provided value, otherwise read from config, otherwise default to 10000
        if max_episodes is not None:
            self.max_episodes = max_episodes
        else:
            training_cfg = self.config.get("training", {})
            self.max_episodes = training_cfg.get("max_episodes", 10000)

        self.current_episode = 0

        # Initialize components (will be created in run())
        self.env = None
        self.population = None
        self.curriculum = None
        self.exploration = None
        self.recorder = None  # Episode recorder (initialized if recording enabled)

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

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_shutdown = True

    def flush_all_agents(self):
        """Flush all agents' episodes to replay buffer before checkpoint."""
        if not self.population:
            return

        for agent_idx in range(self.population.num_agents):
            self.population.flush_episode(agent_idx)

    def save_checkpoint(self):
        """Save checkpoint at current episode."""
        # P1.1 Phase 5: Flush all agents before checkpoint
        self.flush_all_agents()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode:05d}.pt"

        # P1.1 Phase 2: Use population's complete checkpoint state
        checkpoint = {
            "version": 2,  # P1.1: Checkpoint format version for future migration
            "episode": self.current_episode,
            "timestamp": time.time(),
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

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update system state
        self.db.set_system_state("last_checkpoint", str(checkpoint_path))

    def load_checkpoint(self) -> int | None:
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

        checkpoint = torch.load(latest_checkpoint, weights_only=False)

        # P1.1: Check checkpoint version
        checkpoint_version = checkpoint.get("version", 1)  # Default to v1 for legacy checkpoints
        if checkpoint_version != 2:
            logger.warning(f"Loading v{checkpoint_version} checkpoint (current version is 2)")
            # Future: Add migration logic here if needed

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

        # Initialize training components
        device_str = self.config.get("training", {}).get("device", "cuda")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Extract config parameters
        curriculum_cfg = self.config.get("curriculum", {})
        exploration_cfg = self.config.get("exploration", {})
        population_cfg = self.config.get("population", {})
        environment_cfg = self.config.get("environment", {})

        # Get environment parameters from config
        num_agents = population_cfg.get("num_agents", 1)
        grid_size = environment_cfg.get("grid_size", 8)
        partial_observability = environment_cfg.get("partial_observability", False)
        vision_range = environment_cfg.get("vision_range", 2)
        enabled_affordances = environment_cfg.get("enabled_affordances", None)  # None = all affordances
        move_energy_cost = environment_cfg.get(
            "energy_move_depletion",
            environment_cfg.get("move_energy_cost", 0.005),
        )
        wait_energy_cost = environment_cfg.get(
            "energy_wait_depletion",
            environment_cfg.get("wait_energy_cost", 0.001),
        )
        interact_energy_cost = environment_cfg.get(
            "energy_interact_depletion",
            environment_cfg.get("interact_energy_cost", 0.0),
        )

        # Create environment FIRST (need it to auto-detect dimensions)
        self.env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=grid_size,
            device=device,
            partial_observability=partial_observability,
            vision_range=vision_range,
            enabled_affordances=enabled_affordances,
            move_energy_cost=move_energy_cost,
            wait_energy_cost=wait_energy_cost,
            interact_energy_cost=interact_energy_cost,
            config_pack_path=self.config_dir,
        )

        # Auto-detect dimensions from environment (avoids hardcoded config values)
        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        # Create curriculum
        self.curriculum = AdversarialCurriculum(
            max_steps_per_episode=curriculum_cfg.get("max_steps_per_episode", 500),
            survival_advance_threshold=curriculum_cfg.get("survival_advance_threshold", 0.7),
            survival_retreat_threshold=curriculum_cfg.get("survival_retreat_threshold", 0.3),
            entropy_gate=curriculum_cfg.get("entropy_gate", 0.5),
            min_steps_at_stage=curriculum_cfg.get("min_steps_at_stage", 1000),
            device=device,
        )

        # Get training parameters from config
        training_cfg = self.config.get("training", {})

        # Create exploration (use auto-detected obs_dim)
        self.exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=exploration_cfg.get("embed_dim", 128),
            initial_intrinsic_weight=exploration_cfg.get("initial_intrinsic_weight", 1.0),
            variance_threshold=exploration_cfg.get("variance_threshold", 100.0),  # Increased from 10.0
            survival_window=exploration_cfg.get("survival_window", 100),
            epsilon_start=training_cfg.get("epsilon_start", 1.0),
            epsilon_decay=training_cfg.get("epsilon_decay", 0.995),
            epsilon_min=training_cfg.get("epsilon_min", 0.01),
            device=device,
        )

        # Get population parameters from config
        learning_rate = population_cfg.get("learning_rate", 0.00025)
        gamma = population_cfg.get("gamma", 0.99)
        replay_buffer_capacity = population_cfg.get("replay_buffer_capacity", 10000)
        network_type = population_cfg.get("network_type", "simple")  # 'simple' or 'recurrent'
        vision_window_size = 2 * vision_range + 1  # 5 for vision_range=2

        # Get training hyperparameters from config
        train_frequency = training_cfg.get("train_frequency", 4)
        target_update_frequency = training_cfg.get("target_update_frequency", 100)
        batch_size = training_cfg.get("batch_size", None)  # None = auto-select based on network type
        sequence_length = training_cfg.get("sequence_length", 8)
        max_grad_norm = training_cfg.get("max_grad_norm", 10.0)

        # Create agent IDs
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # Create population with correct API
        self.population = VectorizedPopulation(
            env=self.env,
            curriculum=self.curriculum,
            exploration=self.exploration,
            agent_ids=agent_ids,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            replay_buffer_capacity=replay_buffer_capacity,
            network_type=network_type,
            vision_window_size=vision_window_size,
            tb_logger=self.tb_logger,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_grad_norm=max_grad_norm,
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
        loaded_episode = self.load_checkpoint()
        if loaded_episode is not None:
            self.current_episode = loaded_episode + 1

        # Phase 4 - Log hyperparameters to TensorBoard
        self.hparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "network_type": network_type,
            "replay_buffer_capacity": replay_buffer_capacity,
            "grid_size": environment_cfg.get("grid_size", 8),
            "partial_observability": environment_cfg.get("partial_observability", False),
            "vision_range": vision_range,
            "enable_temporal": environment_cfg.get("enable_temporal_mechanics", False),
            "initial_intrinsic_weight": exploration_cfg.get("initial_intrinsic_weight", 1.0),
            "variance_threshold": exploration_cfg.get("variance_threshold", 100.0),
            "max_steps_per_episode": curriculum_cfg.get("max_steps_per_episode", 500),
        }
        # Note: final metrics will be logged at end of training
        self.tb_logger.log_hyperparameters(hparams=self.hparams, metrics={})

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
                max_steps = curriculum_cfg.get("max_steps_per_episode", 500)
                episode_reward = torch.zeros(num_agents, device=self.env.device)
                episode_extrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                episode_intrinsic_reward = torch.zeros(num_agents, device=self.env.device)
                final_meters = [None for _ in range(num_agents)]
                affordance_visits = [defaultdict(int) for _ in range(num_agents)]
                # NEW: Transition tracking
                affordance_transitions = [defaultdict(lambda: defaultdict(int)) for _ in range(num_agents)]
                last_affordance = [None for _ in range(num_agents)]
                last_agent_state: BatchedAgentState | None = None

                for step in range(max_steps):
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
                            action=agent_state.actions[0].item() if hasattr(agent_state.actions[0], "item") else int(agent_state.actions[0]),
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
                    transitions_dict = {
                        from_aff: dict(to_affs)
                        for from_aff, to_affs in affordance_transitions[0].items()
                    }
                    self.db.insert_affordance_visits(
                        episode_id=self.current_episode,
                        transitions=transitions_dict
                    )

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
                meter_names = [
                    "energy",
                    "hygiene",
                    "satiation",
                    "money",
                    "mood",
                    "social",
                    "health",
                    "fitness",
                ]
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

                # Heartbeat log every 10 episodes
                if self.current_episode % 10 == 0:
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

                # Detailed summary every 50 episodes
                if self.current_episode % 50 == 0:
                    # Get training metrics
                    training_metrics = self.population.get_training_metrics()

                    # Get final meters for agent 0
                    final_meter_values = {}
                    if final_meters[0] is not None:
                        meter_names = ["energy", "hygiene", "satiation", "money", "mood", "social", "health", "fitness"]
                        final_meter_values = {name: final_meters[0][i].item() for i, name in enumerate(meter_names)}

                    # Get affordance usage for agent 0
                    affordance_summary = ", ".join(f"{name}: {count}" for name, count in affordance_visits[0].items()) if affordance_visits[0] else "None"

                    logger.info("=" * 80)
                    logger.info(f"📊 SUMMARY - Episode {self.current_episode}/{self.max_episodes}")
                    logger.info("-" * 80)
                    logger.info(f"Performance:    Survival: {int(step_counts_cpu[0].item())} steps | Stage: {stage_overview}")
                    logger.info(f"Rewards:        Total: {total_reward:.2f} | Extrinsic: {extrinsic_reward:.2f} | Intrinsic: {weighted_intrinsic:.2f}")
                    logger.info(f"Exploration:    ε: {epsilon_value:.3f} | Intrinsic Weight: {intrinsic_weight_value:.3f}")
                    if training_metrics["training_step"] > 0:
                        logger.info(f"Training:       Steps: {training_metrics['training_step']} | Loss: {training_metrics['loss']:.4f} | TD Error: {training_metrics['td_error']:.4f}")
                    if final_meter_values:
                        logger.info(f"Final Meters:   Energy: {final_meter_values.get('energy', 0):.2f} | Health: {final_meter_values.get('health', 0):.2f} | Money: ${final_meter_values.get('money', 0)*100:.1f}")
                    logger.info(f"Affordances:    {affordance_summary}")
                    logger.info("=" * 80)

                # Checkpoint every 100 episodes
                if self.current_episode % 100 == 0:
                    self.save_checkpoint()

                # Decay epsilon for next episode and sync telemetry
                self.exploration.decay_epsilon()
                self.population.sync_exploration_metrics()

                self.current_episode += 1

        finally:
            # Save final checkpoint
            logger.info("Training complete, saving final checkpoint...")
            self.save_checkpoint()

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

            # Shutdown recorder if enabled
            if self.recorder is not None:
                logger.info("Shutting down episode recorder...")
                self.recorder.shutdown()

            self.db.close()

            # Close TensorBoard logger
            self.tb_logger.close()


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

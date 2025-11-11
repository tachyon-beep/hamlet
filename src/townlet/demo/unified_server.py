"""
Unified Server Orchestrator

Coordinates training, inference, and frontend in a single process.

Architecture:
- Training runs in background thread
- Inference server runs in separate thread (FastAPI/uvicorn)
- Frontend runs as subprocess (npm run dev)

Author: Hamlet Project
Date: November 2, 2025
"""

import logging
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

# Import DemoRunner at module level to avoid Python 3.13 atexit threading issues
# (importing TensorBoard/TensorFlow in background threads triggers atexit registration errors)
from townlet.demo.runner import DemoRunner

logger = logging.getLogger(__name__)


class UnifiedServer:
    """
    Orchestrates training, inference, and frontend for unified demo experience.

    Components:
    1. Training Thread: Runs DemoRunner in background
    2. Inference Thread: Runs LiveInferenceServer (FastAPI WebSocket)
    3. Frontend Subprocess: Runs Vue dev server (npm run dev)

    All components coordinate for graceful shutdown on SIGINT/SIGTERM.
    """

    def __init__(
        self,
        config_dir: str,
        total_episodes: int,
        checkpoint_dir: str | None = None,
        inference_port: int = 8766,
        training_config_path: str | None = None,
    ):
        """
        Initialize unified server.

        Args:
            config_dir: Directory containing configuration pack (training.yaml, affordances.yaml, etc.)
            total_episodes: Total number of episodes to train
            checkpoint_dir: Directory for checkpoints (auto-generated if None)
            inference_port: Port for inference WebSocket server
        """
        self.config_dir = Path(config_dir)
        if training_config_path:
            self.training_config_path = Path(training_config_path)
        else:
            # Default: assume training.yaml inside config_dir
            self.training_config_path = self.config_dir / "training.yaml"
        if not self.training_config_path.exists():
            raise FileNotFoundError(f"Training config not found: {self.training_config_path}")
        self.total_episodes = total_episodes
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.inference_port = inference_port
        self._config_cache: dict | None = None

        # Component handles (initialized in start())
        self.training_thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None

        # Shutdown coordination
        self.shutdown_requested = False
        self.training_completed_normally = False  # Track normal vs error completion
        self._shutdown_lock = threading.Lock()

        logger.debug(
            f"UnifiedServer initialized with config_dir={self.config_dir}, "
            f"training_config={self.training_config_path}, "
            f"episodes={total_episodes}, port={inference_port}"
        )

    def _persist_config_snapshot(self, run_root: Path) -> None:
        """
        Copy the active config pack into the run directory for provenance.

        Args:
            run_root: Base directory for the current run (parent of checkpoints)
        """
        snapshot_dir = run_root / "config_snapshot"
        try:
            if snapshot_dir.exists():
                logger.debug("Config snapshot already exists at %s", snapshot_dir)
                return

            logger.info("Saving config snapshot to %s", snapshot_dir)
            shutil.copytree(self.config_dir, snapshot_dir)

            # Record the training config path used (for legacy single-file configs)
            if self.training_config_path and self.training_config_path.exists():
                training_copy = snapshot_dir / "training.yaml"
                if not training_copy.exists():
                    shutil.copy2(self.training_config_path, training_copy)
        except Exception as exc:
            logger.warning("Failed to persist config snapshot: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))

    def start(self) -> None:
        """
        Start all components and block until shutdown.

        This is the main entry point called from run_demo.py.
        It starts training, inference, and frontend, then blocks
        until a shutdown signal is received.
        """
        logger.info("Starting unified server components...")

        try:
            # Determine checkpoint directory upfront (before any threads start)
            if self.checkpoint_dir is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                base_dir = self._determine_run_directory(timestamp)
                self.checkpoint_dir = base_dir / "checkpoints"
                logger.info(f"Auto-generated checkpoint dir: {self.checkpoint_dir}")

            # Ensure checkpoint directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Persist a snapshot of the configuration for provenance
            self._persist_config_snapshot(self.checkpoint_dir.parent)

            # Add file logging to run directory
            self._setup_file_logging(self.checkpoint_dir.parent)

            # Phase 2: Start training thread
            logger.info("[Training] Starting background training...")
            self.training_thread = threading.Thread(
                target=self._run_training,
                name="TrainingThread",
                daemon=False,  # Explicit join required
            )
            self.training_thread.start()
            logger.info("[Training] Thread started")

            # Phase 3: Start inference server in background
            logger.info("[Inference] Starting WebSocket server...")
            self.inference_thread = threading.Thread(
                target=self._run_inference,
                name="InferenceThread",
                daemon=True,  # Daemon so it dies with main process
            )
            self.inference_thread.start()
            logger.info(f"[Inference] Server starting on port {self.inference_port}")

            # Display helpful commands
            tensorboard_dir = self.checkpoint_dir.parent / "tensorboard"
            logger.info("=" * 60)
            logger.info("✅ Training + Inference servers operational")
            logger.info("=" * 60)
            logger.info("")
            logger.info("📊 To view live visualization (optional):")
            logger.info("   Terminal 2:")
            logger.info("   $ cd frontend && npm run dev -- --host 0.0.0.0")
            logger.info("   Then open: http://localhost:5173")
            logger.info("")
            logger.info("📈 To view training metrics (optional):")
            logger.info("   Terminal 3:")
            logger.info(f"   $ tensorboard --logdir {tensorboard_dir} --bind_all")
            logger.info("   Then open: http://localhost:6006")
            logger.info("")
            logger.info(f"💾 Checkpoints: {self.checkpoint_dir}")
            logger.info(f"🔌 Inference port: {self.inference_port}")
            logger.info("")
            logger.info("Press Ctrl+C to stop gracefully")
            logger.info("=" * 60)

            # Block until shutdown requested
            while not self.shutdown_requested:
                time.sleep(1)

                # Check if training thread died unexpectedly (error condition)
                if self.training_thread and not self.training_thread.is_alive():
                    if not self.training_completed_normally:
                        logger.warning("[Training] Thread died unexpectedly!")
                        self.shutdown_requested = True

                # Check if inference thread died unexpectedly
                if self.inference_thread and not self.inference_thread.is_alive():
                    logger.warning("[Inference] Thread died unexpectedly!")
                    self.shutdown_requested = True

        except KeyboardInterrupt:
            # Shouldn't reach here (signal handler calls stop())
            # But handle just in case
            logger.info("KeyboardInterrupt received in start()")
            self.stop()

        logger.info("UnifiedServer.start() exiting")

    def _load_config(self) -> dict:
        """Load and cache the YAML configuration."""
        if self._config_cache is None:
            with open(self.training_config_path) as f:
                data = yaml.safe_load(f) or {}
            self._config_cache = data
        return self._config_cache

    def _determine_run_directory(self, timestamp: str) -> Path:
        """
        Determine the base run directory for auto-generated checkpoints.

        Prefers explicit output_subdir specified in config; falls back to
        legacy name inference from config file path.
        """
        config = self._load_config()
        run_metadata = config.get("run_metadata") or {}
        output_subdir = run_metadata.get("output_subdir")

        if output_subdir:
            level_name = self._sanitize_folder_name(str(output_subdir))
            if not level_name:
                logger.warning("Config run_metadata.output_subdir is empty after sanitisation; falling back to legacy level detection.")
                level_name = self._infer_level_name()
        else:
            level_name = self._infer_level_name()

        return Path("runs") / level_name / timestamp

    def _infer_level_name(self) -> str:
        """Legacy fallback: infer run folder name from config stem."""
        if self.config_dir:
            config_stem = self.config_dir.name.lower()
        else:
            config_stem = self.training_config_path.stem.lower()
        if "level_1" in config_stem or "full_observability" in config_stem:
            return "L1_full_observability"
        if "level_2" in config_stem or "partial_observability" in config_stem or "pomdp" in config_stem:
            return "L2_partial_observability"
        if "level_3" in config_stem or "temporal" in config_stem:
            return "L3_temporal_mechanics"
        if "level_4" in config_stem or "multi_agent" in config_stem:
            return "L4_multi_agent"
        if "level_0" in config_stem or "minimal" in config_stem:
            return "L0_0_minimal"
        return "training"

    @staticmethod
    def _sanitize_folder_name(value: str) -> str:
        """
        Make a filesystem-safe folder name.

        Keeps alphanumerics, dash, underscore, and dot, replacing others with underscores.
        """
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
        return sanitized

    def _setup_file_logging(self, run_dir: Path) -> None:
        """
        Add file handler to save logs to run directory.

        Args:
            run_dir: Base directory for the current run (e.g., runs/L0_0_minimal/2025-11-03_123456)
        """
        log_file = run_dir / "training.log"

        # Create file handler with same format as console
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Use same format as console logging
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        # Add to root logger (captures all module logs)
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        # Store handler for cleanup on shutdown
        self._file_handler = file_handler

        logger.info(f"📝 Logging to file: {log_file}")

    def stop(self) -> None:
        """
        Gracefully stop all components.

        Shutdown sequence:
        1. Set shutdown flag
        2. Stop training thread (finish current episode)
        3. Stop inference server
        4. Stop frontend subprocess
        5. Wait for all to exit cleanly
        """
        with self._shutdown_lock:
            if self.shutdown_requested:
                logger.debug("Shutdown already in progress")
                return

            self.shutdown_requested = True

        logger.info("Initiating graceful shutdown...")

        # Stop inference server
        if self.inference_thread and self.inference_thread.is_alive():
            logger.info("[Inference] Stopping WebSocket server...")

            # Shut down uvicorn server gracefully
            if hasattr(self, "_inference_uvicorn_server"):
                self._inference_uvicorn_server.should_exit = True

            self.inference_thread.join(timeout=10)
            if self.inference_thread.is_alive():
                logger.warning("[Inference] Thread did not stop within timeout!")
            else:
                logger.info("[Inference] Thread stopped successfully")

        # Stop training thread last (may need to finish episode)
        if self.training_thread and self.training_thread.is_alive():
            logger.info("[Training] Stopping training thread...")
            if hasattr(self, "runner") and self.runner:
                self.runner.should_shutdown = True
            logger.info("[Training] Waiting for current episode to finish...")
            self.training_thread.join(timeout=30)
            if self.training_thread.is_alive():
                logger.warning("[Training] Thread did not stop within timeout!")
            else:
                logger.info("[Training] Thread stopped successfully")

        logger.info("All components stopped successfully")

        # Close file handler
        if hasattr(self, "_file_handler") and self._file_handler:
            self._file_handler.close()
            logging.getLogger().removeHandler(self._file_handler)

    def _run_training(self) -> None:
        """
        Training thread entry point.

        Creates DemoRunner and runs training loop.
        Assumes self.checkpoint_dir has been set by start() method.
        """
        try:
            logger.info("[Training] Initializing DemoRunner...")

            # Type narrowing: checkpoint_dir is guaranteed to be set by start()
            assert self.checkpoint_dir is not None, "checkpoint_dir must be set by start() before calling _run_training()"

            # Create database path (sibling to checkpoints)
            db_path = self.checkpoint_dir.parent / "metrics.db"

            # Create DemoRunner
            self.runner = DemoRunner(
                config_dir=str(self.config_dir),
                training_config_path=str(self.training_config_path),
                db_path=str(db_path),
                checkpoint_dir=str(self.checkpoint_dir),
                max_episodes=self.total_episodes,
            )

            logger.info("[Training] Starting training loop...")
            self.runner.run()
            logger.info("[Training] Training loop completed normally")

            # Mark as normal completion
            self.training_completed_normally = True
            self.shutdown_requested = True  # Trigger shutdown

        except Exception as e:
            logger.error(f"[Training] Error in training thread: {e}")
            logger.exception("Full traceback:")
            # Set shutdown flag so main thread knows we died (with error)
            self.shutdown_requested = True

    def _run_inference(self) -> None:
        """
        Inference server thread entry point.

        Runs LiveInferenceServer with uvicorn.
        """
        import asyncio

        import uvicorn

        from townlet.demo.live_inference import LiveInferenceServer

        try:
            logger.info("[Inference] Initializing LiveInferenceServer...")

            # Type narrowing: checkpoint_dir is guaranteed to be set by start()
            assert self.checkpoint_dir is not None, "checkpoint_dir must be set by start() before calling _run_inference()"

            # Use the same checkpoint directory as training
            self.inference_server = LiveInferenceServer(
                checkpoint_dir=self.checkpoint_dir,
                port=self.inference_port,
                step_delay=0.2,  # 5 steps/sec
                total_episodes=self.total_episodes,
                config_dir=self.config_dir,
                training_config_path=self.training_config_path,
            )

            logger.info(f"[Inference] Starting uvicorn on 0.0.0.0:{self.inference_port}...")

            # Create uvicorn server with proper async support
            config = uvicorn.Config(
                app=self.inference_server.app,
                host="0.0.0.0",
                port=self.inference_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)

            # Run in asyncio event loop (allows clean shutdown)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Store server for shutdown signal
            self._inference_uvicorn_server = server

            try:
                loop.run_until_complete(server.serve())
            finally:
                loop.close()

            logger.info("[Inference] Server stopped normally")

        except Exception as e:
            logger.error(f"[Inference] Error in inference thread: {e}")
            logger.exception("Full traceback:")
            self.shutdown_requested = True

    def _start_frontend(self) -> None:
        """
        Start frontend subprocess.

        Runs 'npm run dev' in frontend directory.
        """
        try:
            import subprocess

            # Find frontend directory (project root / frontend)
            frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"

            if not frontend_dir.exists():
                logger.warning(f"[Frontend] Directory not found: {frontend_dir}")
                logger.warning("[Frontend] Skipping frontend startup")
                return

            # Check if npm is available
            try:
                subprocess.run(["npm", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("[Frontend] npm not found in PATH")
                logger.warning("[Frontend] Skipping frontend startup")
                return

            logger.info(f"[Frontend] Starting npm run dev in {frontend_dir}...")

            # Start frontend process
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line-buffered
            )

            # Wait for server to be ready (look for "Local:" in output)
            ready = False
            timeout_counter = 0.0  # Use float to match time.sleep increment
            max_timeout = 30  # 30 seconds
            detected_port: int | None = None  # Extract port from npm output

            while not ready and timeout_counter < max_timeout:
                if self.frontend_process.poll() is not None:
                    logger.error("[Frontend] Process died during startup!")
                    return

                # Read output line by line (non-blocking-ish)
                # Type check: stdout is guaranteed to be IO[str] by Popen text=True
                if self.frontend_process.stdout is None:
                    logger.error("[Frontend] stdout is None, cannot read output")
                    return

                line = self.frontend_process.stdout.readline()
                if line:
                    logger.debug(f"[Frontend] {line.strip()}")
                    # Extract port from output like "Local:   http://localhost:5173/"
                    if "Local:" in line and "localhost:" in line:
                        import re

                        port_match = re.search(r"localhost:(\d+)", line)
                        if port_match:
                            detected_port = int(port_match.group(1))
                            ready = True
                            break

                time.sleep(0.1)
                timeout_counter += 0.1

            if ready and detected_port:
                logger.info(f"[Frontend] Dev server ready at http://localhost:{detected_port}")
                # Note: Browser auto-open removed - user can manually navigate to URL
                # Future: Add --open-browser CLI flag if needed
            else:
                logger.warning("[Frontend] Server did not become ready within timeout")

        except Exception as e:
            logger.error(f"[Frontend] Error starting subprocess: {e}")
            logger.exception("Full traceback:")

    def _stop_frontend(self) -> None:
        """
        Stop frontend subprocess gracefully.

        Sends SIGTERM and waits for clean exit.
        """
        if not self.frontend_process:
            return

        try:
            import subprocess

            logger.info("[Frontend] Sending SIGTERM to subprocess...")
            self.frontend_process.terminate()

            try:
                self.frontend_process.wait(timeout=5)
                logger.info("[Frontend] Subprocess terminated cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("[Frontend] Subprocess did not stop, sending SIGKILL...")
                self.frontend_process.kill()
                self.frontend_process.wait()
                logger.info("[Frontend] Subprocess killed")
        except Exception as e:
            logger.error(f"[Frontend] Error stopping subprocess: {e}")

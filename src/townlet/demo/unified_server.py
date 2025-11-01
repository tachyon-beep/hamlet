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
import sys
import threading
import time
from pathlib import Path
from typing import Optional

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
        config_path: str,
        total_episodes: int,
        checkpoint_dir: Optional[str] = None,
        inference_port: int = 8766,
    ):
        """
        Initialize unified server.

        Args:
            config_path: Path to training configuration YAML
            total_episodes: Total number of episodes to train
            checkpoint_dir: Directory for checkpoints (auto-generated if None)
            inference_port: Port for inference WebSocket server
        """
        self.config_path = Path(config_path)
        self.total_episodes = total_episodes
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.inference_port = inference_port

        # Component handles (initialized in start())
        self.training_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        # Shutdown coordination
        self.shutdown_requested = False
        self.training_completed_normally = False  # Track normal vs error completion
        self._shutdown_lock = threading.Lock()

        logger.debug(
            f"UnifiedServer initialized with config={config_path}, "
            f"episodes={total_episodes}, port={inference_port}"
        )

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
                from datetime import datetime

                # Infer level name from config
                config_stem = self.config_path.stem
                if "level_1" in config_stem or "full_observability" in config_stem:
                    level_name = "L1_full_observability"
                elif (
                    "level_2" in config_stem
                    or "partial_observability" in config_stem
                    or "pomdp" in config_stem
                ):
                    level_name = "L2_partial_observability"
                elif "level_3" in config_stem or "temporal" in config_stem:
                    level_name = "L3_temporal_mechanics"
                elif "level_4" in config_stem or "multi_agent" in config_stem:
                    level_name = "L4_multi_agent"
                else:
                    level_name = "training"

                # Generate timestamp directory
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                base_dir = Path("runs") / level_name / timestamp
                self.checkpoint_dir = base_dir / "checkpoints"
                logger.info(f"Auto-generated checkpoint dir: {self.checkpoint_dir}")

            # Ensure checkpoint directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

            logger.info("─" * 60)
            logger.info("Training + Inference servers operational.")
            logger.info(f"Open frontend separately: cd frontend && npm run dev")
            logger.info(f"Frontend will connect to: http://localhost:{self.inference_port}")
            logger.info("Press Ctrl+C to stop.")
            logger.info("─" * 60)

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

    def _run_training(self) -> None:
        """
        Training thread entry point.

        Creates DemoRunner and runs training loop.
        Assumes self.checkpoint_dir has been set by start() method.
        """
        try:
            logger.info("[Training] Initializing DemoRunner...")

            # Import here to avoid circular dependencies
            from townlet.demo.runner import DemoRunner

            # Create database path (sibling to checkpoints)
            db_path = self.checkpoint_dir.parent / "metrics.db"

            # Create DemoRunner
            self.runner = DemoRunner(
                config_path=str(self.config_path),
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

            # Use the same checkpoint directory as training
            self.inference_server = LiveInferenceServer(
                checkpoint_dir=self.checkpoint_dir,
                port=self.inference_port,
                step_delay=0.2,  # 5 steps/sec
                total_episodes=self.total_episodes,
                config_path=self.config_path,
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
            import webbrowser

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
            timeout_counter = 0
            max_timeout = 30  # 30 seconds

            while not ready and timeout_counter < max_timeout:
                if self.frontend_process.poll() is not None:
                    logger.error("[Frontend] Process died during startup!")
                    return

                # Read output line by line (non-blocking-ish)
                line = self.frontend_process.stdout.readline()
                if line:
                    logger.debug(f"[Frontend] {line.strip()}")
                    if "Local:" in line or f":{self.frontend_port}" in line:
                        ready = True
                        break

                time.sleep(0.1)
                timeout_counter += 0.1

            if ready:
                logger.info(f"[Frontend] Dev server ready at http://localhost:{self.frontend_port}")

                # Open browser if requested
                if self.open_browser:
                    time.sleep(1)  # Give server a moment to fully stabilize
                    logger.info(
                        f"[Frontend] Opening browser to http://localhost:{self.frontend_port}"
                    )
                    webbrowser.open(f"http://localhost:{self.frontend_port}")
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

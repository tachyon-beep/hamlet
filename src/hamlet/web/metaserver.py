"""
Unified Metaserver for Hamlet Visualization.

Handles both inference and training modes on a single port (8765):
- /ws endpoint: Inference mode (multiple concurrent sessions)
- /ws/training endpoint: Training mode (singleton broadcast)

Architecture:
- Inference sessions are independent per-client
- Training broadcasts to all connected clients
- AsyncIO for non-blocking concurrent operations
"""

import asyncio
import json
from typing import Dict, Set
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing managers
from hamlet.web.websocket import WebSocketManager
from hamlet.web.training_server import TrainingBroadcaster
from hamlet.training.config import MetricsConfig
from hamlet.training.metrics_manager import MetricsManager


app = FastAPI(
    title="Hamlet Metaserver",
    description="Unified server for inference and training visualization",
    version="2.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers
inference_manager = WebSocketManager()  # Handles /ws endpoint
training_broadcaster = TrainingBroadcaster()  # Handles /ws/training endpoint


def _create_metrics_manager(db_path: Path) -> MetricsManager | None:
    if not db_path.exists():
        return None

    config = MetricsConfig(
        tensorboard=False,
        tensorboard_dir="/tmp/unused",
        database=True,
        database_path=str(db_path),
        replay_storage=False,
        live_broadcast=False,
    )
    return MetricsManager(config, experiment_name="metaserver_api")


@app.on_event("startup")
async def startup_event():
    """Initialize metaserver on startup."""
    print("=" * 60)
    print("HAMLET METASERVER STARTED")
    print("=" * 60)
    print("Inference endpoint:  ws://localhost:8765/ws")
    print("Training endpoint:   ws://localhost:8765/ws/training")
    print()
    print("Both modes available on single server!")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    await inference_manager.shutdown()
    print("Metaserver shutdown complete")


@app.get("/")
async def root():
    """Health check and server status."""
    return {
        "status": "running",
        "service": "Hamlet Metaserver",
        "version": "2.0.0",
        "endpoints": {
            "inference": "/ws",
            "training": "/ws/training"
        },
        "active_sessions": {
            "inference": len(inference_manager.active_connections),
            "training": len(training_broadcaster.active_connections),
        },
        "training_active": training_broadcaster.is_training,
    }


@app.get("/api/models")
async def list_models():
    """List available trained models."""
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}

    models = [
        str(f.name)
        for f in models_dir.glob("*.pt")
    ]

    return {
        "models": models,
        "count": len(models)
    }


@app.get("/api/failures")
async def list_failures(
    agent: str | None = None,
    reason: str | None = None,
    min_episode: int | None = None,
    max_episode: int | None = None,
    limit: int | None = 20,
    db_path: str = "metrics.db",
):
    manager = _create_metrics_manager(Path(db_path))
    if manager is None:
        return {"failures": []}

    try:
        failures = manager.query_failure_events(
            agent_id=agent,
            reason=reason,
            min_episode=min_episode,
            max_episode=max_episode,
            limit=limit,
        )
        return {"failures": failures}
    finally:
        manager.close()


@app.get("/api/failure_summary")
async def failure_summary(
    agent: str | None = None,
    reason: str | None = None,
    min_episode: int | None = None,
    max_episode: int | None = None,
    top: int | None = 10,
    db_path: str = "metrics.db",
):
    manager = _create_metrics_manager(Path(db_path))
    if manager is None:
        return {"summary": []}

    try:
        summary = manager.get_failure_summary(
            agent_id=agent,
            reason=reason,
            min_episode=min_episode,
            max_episode=max_episode,
            top_n=top,
        )
        return {"summary": summary}
    finally:
        manager.close()


@app.websocket("/ws")
async def inference_endpoint(websocket: WebSocket):
    """
    Inference mode WebSocket endpoint.

    Each connection gets its own independent session with:
    - Dedicated environment instance
    - Selected trained agent
    - Isolated state

    Multiple users can watch different agents simultaneously.
    """
    await inference_manager.connect(websocket)

    try:
        # Keep connection alive and handle incoming commands
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle control commands
            if message.get("type") == "control":
                command = message.get("command")

                if command == "play":
                    inference_manager.play()
                elif command == "pause":
                    inference_manager.pause()
                elif command == "step":
                    inference_manager.step()
                elif command == "reset":
                    inference_manager.reset()
                elif command == "set_speed":
                    speed = message.get("speed", 1.0)
                    inference_manager.set_speed(speed)
                elif command == "load_model":
                    model = message.get("model")
                    await inference_manager.load_model(model)

    except WebSocketDisconnect:
        await inference_manager.disconnect(websocket)
    except Exception as e:
        print(f"Inference WebSocket error: {e}")
        await inference_manager.disconnect(websocket)


@app.websocket("/ws/training")
async def training_endpoint(websocket: WebSocket):
    """
    Training mode WebSocket endpoint.

    Singleton broadcast mode:
    - One training session running at a time
    - All connected clients receive same updates
    - Additional clients can "tune in" to ongoing training

    Commands:
    - start_training: Begin new training session (if none running)
    - pause: Pause ongoing training
    - resume: Resume paused training
    - status: Get current training status
    """
    await training_broadcaster.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            command = message.get("command")

            if command == "start_training":
                # Start training in background
                num_episodes = message.get("num_episodes", 100)
                batch_size = message.get("batch_size", 32)
                buffer_capacity = message.get("buffer_capacity", 10000)
                show_every = message.get("show_every", 5)
                step_delay = message.get("step_delay", 0.2)

                asyncio.create_task(training_broadcaster.start_training(
                    num_episodes=num_episodes,
                    batch_size=batch_size,
                    buffer_capacity=buffer_capacity,
                    show_every=show_every,
                    step_delay=step_delay,
                ))

            elif command == "pause":
                training_broadcaster.pause_event.clear()
                await websocket.send_text(json.dumps({"type": "paused"}))

            elif command == "resume":
                training_broadcaster.pause_event.set()
                await websocket.send_text(json.dumps({"type": "resumed"}))

            elif command == "status":
                await websocket.send_text(json.dumps({
                    "type": "training_status",
                    "is_training": training_broadcaster.is_training,
                    "current_episode": training_broadcaster.current_episode,
                    "total_episodes": training_broadcaster.total_episodes,
                }))

    except WebSocketDisconnect:
        await training_broadcaster.disconnect(websocket)
    except Exception as e:
        print(f"Training WebSocket error: {e}")
        await training_broadcaster.disconnect(websocket)


def main():
    """Entry point for metaserver."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info"
    )


if __name__ == "__main__":
    main()

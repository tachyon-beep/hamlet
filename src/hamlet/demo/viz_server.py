"""Visualization server for multi-day demo."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from hamlet.demo.database import DemoDatabase

logger = logging.getLogger(__name__)


class VizServer:
    """Streams demo state from SQLite to browsers via WebSocket."""

    def __init__(
        self,
        db_path: Path | str,
        frontend_dir: Path | str,
        port: int = 8765,
    ):
        """Initialize visualization server.

        Args:
            db_path: Path to demo SQLite database
            frontend_dir: Path to built frontend assets
            port: WebSocket port
        """
        self.db_path = Path(db_path)
        self.frontend_dir = Path(frontend_dir)
        self.port = port

        self.db = DemoDatabase(db_path)
        self.clients = set()
        self.broadcast_task = None

        # Create FastAPI app
        self.app = FastAPI(title="Hamlet Demo Visualization")

        # Register routes and lifecycle events
        self.app.get("/")(self.serve_index)
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.on_event("startup")(self.startup)
        self.app.on_event("shutdown")(self.shutdown)

        # Mount static files
        if self.frontend_dir.exists():
            self.app.mount("/assets", StaticFiles(directory=self.frontend_dir / "assets"), name="assets")

    async def startup(self):
        """Start background broadcast task."""
        self.broadcast_task = asyncio.create_task(self.broadcast_loop())
        logger.info("Started broadcast task")

    async def shutdown(self):
        """Stop background broadcast task."""
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped broadcast task")

    async def serve_index(self):
        """Serve frontend index.html."""
        index_path = self.frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"error": "Frontend not built. Run: cd frontend && npm run build"}

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for streaming updates."""
        await websocket.accept()
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        try:
            # Keep connection alive, wait for client disconnect
            while True:
                await websocket.receive_text()
        except Exception as e:
            logger.info(f"Client disconnected: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client removed. Total clients: {len(self.clients)}")

    async def broadcast_loop(self):
        """Background task that broadcasts updates to all clients."""
        logger.info("Broadcast loop started")
        while True:
            try:
                await self.broadcast_update()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1)

    async def broadcast_update(self):
        """Query database and broadcast to all clients."""
        if not self.clients:
            return  # No clients, skip broadcast

        try:
            # Get latest episode
            episodes = self.db.get_latest_episodes(limit=1)
            if not episodes:
                return

            latest = episodes[0]

            # Build update message
            update = {
                'type': 'state_update',
                'episode': latest['episode_id'],
                'timestamp': latest['timestamp'],
                'metrics': {
                    'survival_time': latest['survival_time'],
                    'total_reward': latest['total_reward'],
                    'extrinsic_reward': latest['extrinsic_reward'],
                    'intrinsic_reward': latest['intrinsic_reward'],
                    'intrinsic_weight': latest['intrinsic_weight'],
                    'curriculum_stage': latest['curriculum_stage'],
                    'epsilon': latest['epsilon'],
                },
                # TODO: Add position_heatmap and affordance_graph
            }

            # Broadcast to all clients
            dead_clients = set()
            for client in self.clients:
                try:
                    await client.send_json(update)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    dead_clients.add(client)

            # Remove dead clients
            self.clients -= dead_clients

        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")


def run_server(
    db_path: str = "demo_state.db",
    frontend_dir: str = "frontend/dist",
    port: int = 8765,
):
    """Run visualization server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    # Initialize server
    server = VizServer(db_path, frontend_dir, port)

    logger.info(f"Starting visualization server on port {port}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Frontend: {frontend_dir}")

    # Run FastAPI
    uvicorn.run(server.app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    import sys
    run_server(
        db_path=sys.argv[1] if len(sys.argv) > 1 else "demo_state.db",
        frontend_dir=sys.argv[2] if len(sys.argv) > 2 else "frontend/dist",
        port=int(sys.argv[3]) if len(sys.argv) > 3 else 8765,
    )

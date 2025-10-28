"""
FastAPI web server for Hamlet visualization.

Serves static files and provides WebSocket streaming for real-time visualization.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from pathlib import Path
from typing import List
import asyncio

from hamlet.web.websocket import WebSocketManager


app = FastAPI(
    title="Hamlet Visualization",
    description="Real-time DRL agent visualization",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager (singleton)
ws_manager = WebSocketManager()


@app.on_event("startup")
async def startup_event():
    """Initialize simulation runner on startup."""
    print("Starting Hamlet visualization server...")
    print("WebSocket endpoint: ws://localhost:8765/ws")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    await ws_manager.shutdown()
    print("Server shutdown complete")


# Static file serving (for production frontend)
frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        """Serve the Vue frontend."""
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse(
            content="<h1>Hamlet Visualization</h1><p>Frontend not built. Run: cd frontend && npm run build</p>"
        )
else:
    @app.get("/")
    async def root():
        """Development placeholder."""
        return HTMLResponse(
            content="""
            <h1>Hamlet Visualization Server</h1>
            <p>Server is running on port 8765</p>
            <p>Frontend not found. To set up:</p>
            <ol>
                <li>cd frontend</li>
                <li>npm install</li>
                <li>npm run dev (for development)</li>
                <li>npm run build (for production)</li>
            </ol>
            <p>WebSocket endpoint: <code>ws://localhost:8765/ws</code></p>
            """
        )


# REST API endpoints

@app.get("/api/status")
async def get_status():
    """Get current server and simulation status."""
    return {
        "status": "running",
        "connected_clients": ws_manager.get_connection_count(),
        "simulation_running": ws_manager.is_running(),
    }


@app.get("/api/models")
async def list_models():
    """List available trained agent models."""
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}

    models = [f.name for f in models_dir.glob("*.pt")]
    return {"models": models}


# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time simulation streaming.

    Protocol:
        - Server → Client: JSON messages with state updates
        - Client → Server: JSON commands (play/pause/step/reset/speed/load_model)
    """
    await ws_manager.connect(websocket)
    try:
        # Keep connection alive and handle incoming commands
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle control commands
            if message.get("type") == "control":
                command = message.get("command")

                if command == "play":
                    ws_manager.play()
                elif command == "pause":
                    ws_manager.pause()
                elif command == "step":
                    ws_manager.step()
                elif command == "reset":
                    ws_manager.reset()
                elif command == "set_speed":
                    speed = message.get("speed", 1.0)
                    ws_manager.set_speed(speed)
                elif command == "load_model":
                    model = message.get("model")
                    await ws_manager.load_model(model)

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


def main():
    """Run the web server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,  # WebSocket-friendly port
        log_level="info"
    )


if __name__ == "__main__":
    main()

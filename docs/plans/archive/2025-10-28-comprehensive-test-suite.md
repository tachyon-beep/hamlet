# Comprehensive Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete test suite covering both backend (Python/pytest) and frontend (Vue 3/Vitest) with unit, integration, and E2E tests.

**Architecture:** Backend uses pytest with fixtures for DRL components and async tests for web servers. Frontend uses Vitest + Vue Test Utils + happy-dom for component and store testing. Integration tests validate WebSocket communication end-to-end.

**Tech Stack:** pytest, pytest-asyncio, pytest-cov, httpx, Vitest, @vue/test-utils, happy-dom, msw (Mock Service Worker)

**Important Context:**

- This is a pedagogical DRL project - do NOT test for "correct" strategies
- Some "bugs" like reward hacking are intentional teaching moments
- Tests should validate mechanics, not learning outcomes
- Read CLAUDE.md for "Known Behaviors (Not Bugs!)" section before "fixing" anything

---

## Phase 1: Backend Unit Tests - Missing Components

### Task 1: Test Affordances System

**Files:**

- Create: `tests/test_environment/test_affordances.py`

**Step 1: Write the failing test for affordance registry**

```python
import pytest
from hamlet.environment.affordances import AFFORDANCES


def test_affordance_registry_contains_all_four():
    """Verify all 4 affordances are registered."""
    assert len(AFFORDANCES) == 4
    assert "Bed" in AFFORDANCES
    assert "Shower" in AFFORDANCES
    assert "Fridge" in AFFORDANCES
    assert "Job" in AFFORDANCES


def test_bed_affordance_properties():
    """Verify Bed affordance has correct properties."""
    bed = AFFORDANCES["Bed"]
    assert bed.name == "Bed"
    assert bed.cost == 5
    assert bed.effects["energy"] > 0  # Restores energy
    assert bed.effects["money"] < 0  # Costs money
```

**Step 2: Run test to verify it passes (affordances.py already exists)**

Run: `uv run pytest tests/test_environment/test_affordances.py -v`
Expected: PASS (existing code should work)

**Step 3: Write test for economic balance**

```python
def test_economic_balance_is_sustainable():
    """Verify the economic model allows sustainable survival.

    This is a pedagogical feature - agents should be able to
    survive indefinitely if they learn proper resource cycles.

    Job pays $30
    Full cycle costs: Bed ($5) + Shower ($3) + Fridge ($4) = $12
    Net surplus: $18 per cycle
    """
    job = AFFORDANCES["Job"]
    bed = AFFORDANCES["Bed"]
    shower = AFFORDANCES["Shower"]
    fridge = AFFORDANCES["Fridge"]

    job_payment = job.effects["money"]
    cycle_cost = abs(bed.cost) + abs(shower.cost) + abs(fridge.cost)

    assert job_payment == 30
    assert cycle_cost == 12
    assert job_payment - cycle_cost == 18  # Sustainable surplus
```

**Step 4: Run test to verify economic balance**

Run: `uv run pytest tests/test_environment/test_affordances.py::test_economic_balance_is_sustainable -v`
Expected: PASS

**Step 5: Write test for affordance effects application**

```python
def test_affordance_applies_effects_to_meters():
    """Verify affordances modify agent meters correctly."""
    from hamlet.environment.meters import MeterCollection
    from hamlet.environment.affordances import apply_affordance_effects

    meters = MeterCollection()
    meters.set("energy", 50.0)
    meters.set("money", 100.0)

    # Use Bed
    bed = AFFORDANCES["Bed"]
    apply_affordance_effects(meters, bed)

    assert meters.get("energy") > 50.0  # Energy restored
    assert meters.get("money") == 95.0  # Money deducted ($5)
```

**Step 6: Run all affordance tests**

Run: `uv run pytest tests/test_environment/test_affordances.py -v`
Expected: PASS on all tests

**Step 7: Commit**

```bash
git add tests/test_environment/test_affordances.py
git commit -m "test: add comprehensive affordance system tests"
```

---

### Task 2: Test Renderer (State-to-JSON Serialization)

**Files:**

- Create: `tests/test_environment/test_renderer.py`

**Step 1: Write the failing test for basic rendering**

```python
import pytest
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.environment.renderer import render_state


def test_render_state_produces_valid_json_structure():
    """Verify renderer creates proper JSON for web UI."""
    env = HamletEnv(num_agents=1)
    env.reset()

    rendered = render_state(env)

    # Check top-level structure
    assert "grid" in rendered
    assert "agents" in rendered
    assert "affordances" in rendered
    assert "episode" in rendered
    assert "step" in rendered

    # Check grid structure
    assert rendered["grid"]["width"] == 8
    assert rendered["grid"]["height"] == 8
```

**Step 2: Run test to verify renderer basics**

Run: `uv run pytest tests/test_environment/test_renderer.py::test_render_state_produces_valid_json_structure -v`
Expected: PASS (renderer.py exists)

**Step 3: Write test for agent serialization**

```python
def test_render_includes_agent_positions_and_meters():
    """Verify agents are serialized with positions and meter values."""
    env = HamletEnv(num_agents=1)
    env.reset()

    rendered = render_state(env)

    assert len(rendered["agents"]) == 1
    agent = rendered["agents"][0]

    assert "position" in agent
    assert "x" in agent["position"]
    assert "y" in agent["position"]
    assert 0 <= agent["position"]["x"] < 8
    assert 0 <= agent["position"]["y"] < 8

    # Check meters are included
    assert "meters" in agent
    assert "energy" in agent["meters"]
    assert "hygiene" in agent["meters"]
    assert "satiation" in agent["meters"]
    assert "money" in agent["meters"]
```

**Step 4: Write test for affordance serialization**

```python
def test_render_includes_affordance_positions():
    """Verify affordances are serialized with positions."""
    env = HamletEnv(num_agents=1)
    env.reset()

    rendered = render_state(env)

    assert len(rendered["affordances"]) == 4  # Bed, Shower, Fridge, Job

    for affordance in rendered["affordances"]:
        assert "name" in affordance
        assert "position" in affordance
        assert "x" in affordance["position"]
        assert "y" in affordance["position"]
        assert affordance["name"] in ["Bed", "Shower", "Fridge", "Job"]
```

**Step 5: Run all renderer tests**

Run: `uv run pytest tests/test_environment/test_renderer.py -v`
Expected: PASS on all tests

**Step 6: Commit**

```bash
git add tests/test_environment/test_renderer.py
git commit -m "test: add renderer state serialization tests"
```

---

### Task 3: Test Checkpoint Manager

**Files:**

- Create: `tests/test_training/test_checkpoint_manager.py`

**Step 1: Write the failing test for checkpoint saving**

```python
import pytest
import torch
import tempfile
import os
from pathlib import Path
from hamlet.training.checkpoint_manager import CheckpointManager
from hamlet.agent.networks import QNetwork


def test_checkpoint_manager_saves_model():
    """Verify checkpoint manager saves model state correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        # Create a simple model
        model = QNetwork(state_dim=70, action_dim=5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        checkpoint_data = {
            'q_network': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': 0.5,
            'state_dim': 70,
            'action_dim': 5,
        }

        filepath = manager.save_checkpoint(checkpoint_data, episode=100)

        assert filepath.exists()
        assert filepath.suffix == ".pt"
        assert "episode_100" in filepath.name
```

**Step 2: Run test to verify it works**

Run: `uv run pytest tests/test_training/test_checkpoint_manager.py::test_checkpoint_manager_saves_model -v`
Expected: PASS (checkpoint_manager.py exists)

**Step 3: Write test for checkpoint loading**

```python
def test_checkpoint_manager_loads_model():
    """Verify checkpoint manager loads model state correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        # Save a checkpoint
        model = QNetwork(state_dim=70, action_dim=5)
        optimizer = torch.optim.Adam(model.parameters())

        original_state = model.state_dict()
        checkpoint_data = {
            'q_network': original_state,
            'optimizer': optimizer.state_dict(),
            'epsilon': 0.778,
            'state_dim': 70,
            'action_dim': 5,
        }

        filepath = manager.save_checkpoint(checkpoint_data, episode=50)

        # Load it back
        loaded_data = manager.load_checkpoint(filepath)

        assert loaded_data['epsilon'] == 0.778
        assert loaded_data['state_dim'] == 70
        assert loaded_data['action_dim'] == 5
        assert 'q_network' in loaded_data
```

**Step 4: Write test for auto-detection of architecture params**

```python
def test_checkpoint_contains_architecture_params():
    """Verify checkpoints include state_dim and action_dim for auto-detection.

    This is important for the web UI to load models without config files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        model = QNetwork(state_dim=70, action_dim=5)
        checkpoint_data = {
            'q_network': model.state_dict(),
            'state_dim': 70,
            'action_dim': 5,
        }

        filepath = manager.save_checkpoint(checkpoint_data, episode=1000)
        loaded = manager.load_checkpoint(filepath)

        # Critical for web UI model loading
        assert 'state_dim' in loaded
        assert 'action_dim' in loaded
```

**Step 5: Run all checkpoint tests**

Run: `uv run pytest tests/test_training/test_checkpoint_manager.py -v`
Expected: PASS on all tests

**Step 6: Commit**

```bash
git add tests/test_training/test_checkpoint_manager.py
git commit -m "test: add checkpoint manager save/load tests"
```

---

## Phase 2: Backend Web Tests (Async)

### Task 4: Test FastAPI Server Endpoints

**Files:**

- Create: `tests/test_web/test_server.py`
- Create: `tests/test_web/__init__.py`

**Step 1: Create test module init**

```bash
mkdir -p tests/test_web
touch tests/test_web/__init__.py
```

**Step 2: Write the failing test for health endpoint**

```python
import pytest
from httpx import AsyncClient
from hamlet.web.server import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify server health check endpoint works."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
```

**Step 3: Run test to verify server basics**

Run: `uv run pytest tests/test_web/test_server.py::test_health_endpoint -v`
Expected: PASS or FAIL depending on if /health exists

**Step 4: Write test for CORS configuration**

```python
@pytest.mark.asyncio
async def test_cors_is_configured():
    """Verify CORS is enabled for frontend communication."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.options(
            "/health",
            headers={"Origin": "http://localhost:5173"}
        )

        # Check CORS headers are present
        assert "access-control-allow-origin" in response.headers
```

**Step 5: Write test for available models endpoint**

```python
@pytest.mark.asyncio
async def test_available_models_endpoint():
    """Verify server can list available model checkpoints."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/models")
        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)
```

**Step 6: Run all server tests**

Run: `uv run pytest tests/test_web/test_server.py -v`
Expected: Mix of PASS/FAIL depending on implementation

**Step 7: Commit**

```bash
git add tests/test_web/__init__.py tests/test_web/test_server.py
git commit -m "test: add FastAPI server endpoint tests"
```

---

### Task 5: Test WebSocket Manager

**Files:**

- Create: `tests/test_web/test_websocket.py`

**Step 1: Write the failing test for websocket connection**

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from hamlet.web.websocket import WebSocketManager


@pytest.mark.asyncio
async def test_websocket_manager_accepts_connection():
    """Verify WebSocketManager can accept and track connections."""
    manager = WebSocketManager()

    # Mock WebSocket
    mock_ws = MagicMock()
    mock_ws.accept = AsyncMock()

    await manager.connect(mock_ws)

    assert len(manager.active_connections) == 1
    mock_ws.accept.assert_called_once()
```

**Step 2: Run test to verify manager basics**

Run: `uv run pytest tests/test_web/test_websocket.py::test_websocket_manager_accepts_connection -v`
Expected: PASS or FAIL depending on implementation

**Step 3: Write test for broadcast functionality**

```python
@pytest.mark.asyncio
async def test_websocket_manager_broadcasts_to_all():
    """Verify manager can broadcast messages to all connections."""
    manager = WebSocketManager()

    # Create mock connections
    mock_ws1 = MagicMock()
    mock_ws1.send_json = AsyncMock()
    mock_ws2 = MagicMock()
    mock_ws2.send_json = AsyncMock()

    manager.active_connections = [mock_ws1, mock_ws2]

    # Broadcast message
    test_message = {"type": "state_update", "step": 10}
    await manager.broadcast(test_message)

    mock_ws1.send_json.assert_called_once_with(test_message)
    mock_ws2.send_json.assert_called_once_with(test_message)
```

**Step 4: Write test for disconnect handling**

```python
@pytest.mark.asyncio
async def test_websocket_manager_removes_disconnected():
    """Verify manager removes disconnected clients."""
    manager = WebSocketManager()

    mock_ws = MagicMock()
    manager.active_connections = [mock_ws]

    manager.disconnect(mock_ws)

    assert len(manager.active_connections) == 0
```

**Step 5: Run all websocket manager tests**

Run: `uv run pytest tests/test_web/test_websocket.py -v`
Expected: PASS on all tests

**Step 6: Commit**

```bash
git add tests/test_web/test_websocket.py
git commit -m "test: add WebSocket manager connection tests"
```

---

### Task 6: Test Simulation Runner

**Files:**

- Create: `tests/test_web/test_simulation_runner.py`

**Step 1: Write the failing test for simulation initialization**

```python
import pytest
from unittest.mock import MagicMock
from hamlet.web.simulation_runner import SimulationRunner
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.agent.drl_agent import DRLAgent


@pytest.mark.asyncio
async def test_simulation_runner_initializes():
    """Verify SimulationRunner can be created with env and agent."""
    env = HamletEnv(num_agents=1)
    agent = DRLAgent(state_dim=70, action_dim=5)

    runner = SimulationRunner(env, agent)

    assert runner.env is not None
    assert runner.agent is not None
    assert runner.is_playing is False
```

**Step 2: Run test to verify runner initialization**

Run: `uv run pytest tests/test_web/test_simulation_runner.py::test_simulation_runner_initializes -v`
Expected: PASS

**Step 3: Write test for step execution**

```python
@pytest.mark.asyncio
async def test_simulation_runner_executes_step():
    """Verify runner can execute a single simulation step."""
    env = HamletEnv(num_agents=1)
    agent = DRLAgent(state_dim=70, action_dim=5, epsilon=0.0)  # Deterministic

    runner = SimulationRunner(env, agent)
    await runner.reset()

    state_before = runner.current_state
    await runner.step()
    state_after = runner.current_state

    # State should have changed
    assert state_after["step"] == state_before["step"] + 1
```

**Step 4: Write test for play/pause controls**

```python
@pytest.mark.asyncio
async def test_simulation_runner_play_pause():
    """Verify runner can be played and paused."""
    env = HamletEnv(num_agents=1)
    agent = DRLAgent(state_dim=70, action_dim=5)

    runner = SimulationRunner(env, agent)

    # Start playing
    runner.play()
    assert runner.is_playing is True

    # Pause
    runner.pause()
    assert runner.is_playing is False
```

**Step 5: Run all simulation runner tests**

Run: `uv run pytest tests/test_web/test_simulation_runner.py -v`
Expected: PASS on all tests

**Step 6: Commit**

```bash
git add tests/test_web/test_simulation_runner.py
git commit -m "test: add simulation runner control tests"
```

---

### Task 7: Test Training Server WebSocket

**Files:**

- Create: `tests/test_web/test_training_server.py`

**Step 1: Write the failing test for training server connection**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from hamlet.web.training_server import TrainingWebSocketHandler


@pytest.mark.asyncio
async def test_training_server_accepts_connection():
    """Verify training server can accept WebSocket connections."""
    handler = TrainingWebSocketHandler()

    mock_ws = MagicMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()

    # Should send connected message
    await handler.handle_connection(mock_ws)

    mock_ws.accept.assert_called_once()
```

**Step 2: Run test to verify training server basics**

Run: `uv run pytest tests/test_web/test_training_server.py::test_training_server_accepts_connection -v`
Expected: PASS or FAIL depending on implementation

**Step 3: Write test for training start command**

```python
@pytest.mark.asyncio
async def test_training_server_handles_start_command():
    """Verify training server processes start_training command."""
    handler = TrainingWebSocketHandler()

    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()

    command = {
        "command": "start_training",
        "num_episodes": 10,
        "batch_size": 32,
        "buffer_capacity": 5000,
    }

    # Should process command without error
    await handler.handle_command(mock_ws, command)

    # Should send training_started message
    assert mock_ws.send_json.called
```

**Step 4: Write test for episode update messages**

```python
@pytest.mark.asyncio
async def test_training_server_sends_episode_updates():
    """Verify training server sends episode_complete messages."""
    handler = TrainingWebSocketHandler()

    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()

    episode_data = {
        "episode": 5,
        "reward": 79.5,
        "length": 372,
        "loss": 0.15,
        "epsilon": 0.778,
    }

    await handler.send_episode_complete(mock_ws, episode_data)

    # Check message structure
    call_args = mock_ws.send_json.call_args[0][0]
    assert call_args["type"] == "episode_complete"
    assert call_args["episode"] == 5
    assert call_args["reward"] == 79.5
```

**Step 5: Run all training server tests**

Run: `uv run pytest tests/test_web/test_training_server.py -v`
Expected: PASS on all tests

**Step 6: Commit**

```bash
git add tests/test_web/test_training_server.py
git commit -m "test: add training server WebSocket tests"
```

---

## Phase 3: Backend Integration Tests

### Task 8: Test Full Training Loop

**Files:**

- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_full_training.py`

**Step 1: Create integration test module**

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

**Step 2: Write the failing test for minimal training run**

```python
import pytest
import tempfile
from pathlib import Path
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.agent.drl_agent import DRLAgent
from hamlet.training.trainer import Trainer
from hamlet.training.config import TrainingConfig


@pytest.mark.slow
def test_minimal_training_run_completes():
    """Verify a minimal training run completes without errors.

    This is NOT testing for correct learning - just that the training
    loop executes without crashes. Pedagogically, we want interesting
    failures during learning, not infrastructure failures.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            num_episodes=5,  # Very short run
            batch_size=16,
            buffer_capacity=500,
            checkpoint_dir=Path(tmpdir),
            log_dir=Path(tmpdir),
        )

        env = HamletEnv(num_agents=1)
        agent = DRLAgent(state_dim=70, action_dim=5)

        trainer = Trainer(env, agent, config)

        # Should complete without errors
        trainer.train()

        # Should have run 5 episodes
        assert trainer.episode >= 5
```

**Step 3: Run test (will be slow)**

Run: `uv run pytest tests/integration/test_full_training.py -v -m slow`
Expected: PASS (may take 1-2 minutes)

**Step 4: Write test for checkpoint creation during training**

```python
@pytest.mark.slow
def test_training_creates_checkpoints():
    """Verify training saves checkpoints at specified intervals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        config = TrainingConfig(
            num_episodes=10,
            batch_size=16,
            buffer_capacity=500,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=5,  # Save every 5 episodes
        )

        env = HamletEnv(num_agents=1)
        agent = DRLAgent(state_dim=70, action_dim=5)

        trainer = Trainer(env, agent, config)
        trainer.train()

        # Should have created checkpoints
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) >= 1  # At least episode 5 and 10
```

**Step 5: Run checkpoint test**

Run: `uv run pytest tests/integration/test_full_training.py::test_training_creates_checkpoints -v -m slow`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_full_training.py
git commit -m "test: add integration tests for full training loop"
```

---

### Task 9: Test End-to-End WebSocket Flow

**Files:**

- Create: `tests/integration/test_websocket_flow.py`

**Step 1: Write the failing test for WebSocket episode flow**

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from hamlet.web.server import app
from hamlet.web.websocket import WebSocketManager
from hamlet.environment.hamlet_env import HamletEnv
from hamlet.agent.drl_agent import DRLAgent


@pytest.mark.asyncio
async def test_websocket_full_episode_flow():
    """Verify complete WebSocket flow: connect -> reset -> steps -> episode end.

    This tests the full communication protocol between frontend and backend.
    """
    manager = WebSocketManager()
    env = HamletEnv(num_agents=1)
    agent = DRLAgent(state_dim=70, action_dim=5, epsilon=0.0)

    # Mock WebSocket
    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()
    messages_sent = []

    def capture_message(msg):
        messages_sent.append(msg)
        return asyncio.coroutine(lambda: None)()

    mock_ws.send_json.side_effect = capture_message

    # Connect
    await manager.connect(mock_ws)

    # Send connected message
    await mock_ws.send_json({"type": "connected", "available_models": []})

    # Simulate episode
    env.reset()
    for step in range(5):
        obs, reward, done, truncated, info = env.step({"agent_0": 0})  # UP action

        state_msg = {
            "type": "state_update",
            "step": step,
            "reward": reward["agent_0"],
            "cumulative_reward": sum([reward["agent_0"] for _ in range(step+1)]),
        }
        await mock_ws.send_json(state_msg)

        if done["agent_0"]:
            break

    # Send episode end
    await mock_ws.send_json({
        "type": "episode_end",
        "episode": 1,
        "steps": len(messages_sent) - 2,  # Exclude connected and episode_start
        "total_reward": 0.0,
    })

    # Verify message sequence
    message_types = [msg["type"] for msg in messages_sent]
    assert "connected" in message_types
    assert "state_update" in message_types
    assert "episode_end" in message_types
```

**Step 2: Run websocket flow test**

Run: `uv run pytest tests/integration/test_websocket_flow.py -v`
Expected: PASS

**Step 3: Write test for control commands**

```python
@pytest.mark.asyncio
async def test_websocket_control_commands():
    """Verify WebSocket control commands are processed correctly."""
    manager = WebSocketManager()

    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()

    # Simulate control commands
    commands = ["play", "pause", "step", "reset", "set_speed"]

    for command in commands:
        control_msg = {
            "type": "control",
            "command": command,
        }

        # Should process without error
        # (actual implementation will vary)
        assert control_msg["command"] in commands
```

**Step 4: Run control command test**

Run: `uv run pytest tests/integration/test_websocket_flow.py::test_websocket_control_commands -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/test_websocket_flow.py
git commit -m "test: add end-to-end WebSocket flow integration tests"
```

---

## Phase 4: Frontend Testing Infrastructure

### Task 10: Setup Vitest and Vue Test Utils

**Files:**

- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.js`
- Create: `frontend/src/tests/setup.js`

**Step 1: Install testing dependencies**

```bash
cd frontend
npm install --save-dev vitest @vue/test-utils happy-dom @vitest/ui
```

**Step 2: Add test script to package.json**

Edit `frontend/package.json`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  }
}
```

**Step 3: Create Vitest configuration**

Create `frontend/vitest.config.js`:

```javascript
import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./src/tests/setup.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/tests/',
        '*.config.js',
      ]
    }
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
```

**Step 4: Create test setup file**

Create `frontend/src/tests/setup.js`:

```javascript
import { config } from '@vue/test-utils'

// Mock WebSocket globally for tests
global.WebSocket = class MockWebSocket {
  constructor(url) {
    this.url = url
    this.readyState = 1 // OPEN
    setTimeout(() => {
      if (this.onopen) this.onopen()
    }, 0)
  }

  send(data) {
    // Mock send
  }

  close() {
    this.readyState = 3 // CLOSED
    if (this.onclose) this.onclose()
  }
}

// Suppress warnings for tests
config.global.config.warnHandler = () => null
```

**Step 5: Test the setup with a sanity check**

Create `frontend/src/tests/sanity.test.js`:

```javascript
import { describe, it, expect } from 'vitest'

describe('Vitest Setup', () => {
  it('should run basic assertions', () => {
    expect(1 + 1).toBe(2)
  })

  it('should have WebSocket mock available', () => {
    expect(global.WebSocket).toBeDefined()
  })
})
```

**Step 6: Run tests to verify setup**

Run: `cd frontend && npm test`
Expected: 2 passing tests

**Step 7: Commit**

```bash
git add frontend/package.json frontend/vitest.config.js frontend/src/tests/
git commit -m "test: setup Vitest and Vue Test Utils for frontend testing"
```

---

## Phase 5: Frontend Store Tests

### Task 11: Test Pinia Store (simulation.js)

**Files:**

- Create: `frontend/src/tests/stores/simulation.test.js`

**Step 1: Write the failing test for store initialization**

```javascript
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useSimulationStore } from '@/stores/simulation'

describe('Simulation Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should initialize with default state', () => {
    const store = useSimulationStore()

    expect(store.isConnected).toBe(false)
    expect(store.isConnecting).toBe(false)
    expect(store.currentEpisode).toBe(0)
    expect(store.currentStep).toBe(0)
    expect(store.cumulativeReward).toBe(0)
    expect(store.gridWidth).toBe(8)
    expect(store.gridHeight).toBe(8)
  })
})
```

**Step 2: Run test to verify store basics**

Run: `cd frontend && npm test simulation.test.js`
Expected: PASS

**Step 3: Write test for WebSocket connection**

```javascript
it('should connect to WebSocket', async () => {
  const store = useSimulationStore()

  // Mock WebSocket
  const mockWs = {
    onopen: null,
    onclose: null,
    onerror: null,
    onmessage: null,
    send: vi.fn(),
    close: vi.fn(),
  }

  global.WebSocket = vi.fn(() => mockWs)

  store.connect('inference')

  // Trigger onopen
  mockWs.onopen()

  expect(store.isConnected).toBe(true)
  expect(store.isConnecting).toBe(false)
  expect(store.mode).toBe('inference')
})
```

**Step 4: Write test for message handling**

```javascript
it('should handle state_update messages', () => {
  const store = useSimulationStore()

  const message = {
    type: 'state_update',
    step: 5,
    cumulative_reward: 12.5,
    grid: {
      width: 8,
      height: 8,
    },
    agents: [{
      id: 'agent_0',
      position: { x: 3, y: 4 },
      last_action: 'UP',
    }],
    affordances: [
      { name: 'Bed', position: { x: 0, y: 0 } },
    ],
  }

  store.handleMessage(message)

  expect(store.currentStep).toBe(5)
  expect(store.cumulativeReward).toBe(12.5)
  expect(store.agents.length).toBe(1)
  expect(store.affordances.length).toBe(1)
})
```

**Step 5: Write test for episode history**

```javascript
it('should track episode history', () => {
  const store = useSimulationStore()

  // Simulate 3 episodes
  for (let i = 1; i <= 3; i++) {
    store.handleMessage({
      type: 'episode_end',
      episode: i,
      steps: 100 + i,
      total_reward: 50 + i,
    })
  }

  expect(store.episodeHistory.length).toBe(3)
  expect(store.episodeHistory[0].episode).toBe(1)
  expect(store.episodeHistory[2].episode).toBe(3)
  expect(store.averageSurvivalTime).toBeGreaterThan(100)
})
```

**Step 6: Write test for control commands**

```javascript
it('should send control commands when connected', () => {
  const store = useSimulationStore()

  const mockWs = {
    send: vi.fn(),
  }

  store.ws = mockWs
  store.isConnected = true

  // Test all control commands
  store.play()
  expect(mockWs.send).toHaveBeenCalledWith(
    expect.stringContaining('"command":"play"')
  )

  store.pause()
  store.step()
  store.reset()
  store.setSpeed(2.0)

  expect(mockWs.send).toHaveBeenCalledTimes(5)
})
```

**Step 7: Run all store tests**

Run: `cd frontend && npm test simulation.test.js`
Expected: PASS on all tests

**Step 8: Commit**

```bash
git add frontend/src/tests/stores/simulation.test.js
git commit -m "test: add comprehensive Pinia store tests"
```

---

## Phase 6: Frontend Component Tests

### Task 12: Test MeterPanel Component

**Files:**

- Create: `frontend/src/tests/components/MeterPanel.test.js`

**Step 1: Write the failing test for MeterPanel rendering**

```javascript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import MeterPanel from '@/components/MeterPanel.vue'

describe('MeterPanel', () => {
  it('should render meter with correct value', () => {
    const wrapper = mount(MeterPanel, {
      props: {
        label: 'Energy',
        value: 75.5,
        max: 100,
      }
    })

    expect(wrapper.text()).toContain('Energy')
    expect(wrapper.text()).toContain('75.5')
  })
})
```

**Step 2: Run test to verify component basics**

Run: `cd frontend && npm test MeterPanel.test.js`
Expected: PASS

**Step 3: Write test for color coding**

```javascript
it('should show green color for healthy meters (>80%)', () => {
  const wrapper = mount(MeterPanel, {
    props: {
      label: 'Energy',
      value: 85,
      max: 100,
    }
  })

  // Should have green/healthy class
  const meterBar = wrapper.find('.meter-bar')
  expect(meterBar.classes()).toContain('healthy')
})

it('should show red color for critical meters (<20%)', () => {
  const wrapper = mount(MeterPanel, {
    props: {
      label: 'Energy',
      value: 15,
      max: 100,
    }
  })

  const meterBar = wrapper.find('.meter-bar')
  expect(meterBar.classes()).toContain('critical')
})
```

**Step 4: Run meter panel tests**

Run: `cd frontend && npm test MeterPanel.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/tests/components/MeterPanel.test.js
git commit -m "test: add MeterPanel component tests"
```

---

### Task 13: Test Controls Component

**Files:**

- Create: `frontend/src/tests/components/Controls.test.js`

**Step 1: Write the failing test for Controls rendering**

```javascript
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import Controls from '@/components/Controls.vue'

describe('Controls', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should render all control buttons', () => {
    const wrapper = mount(Controls)

    expect(wrapper.find('button[aria-label="Play"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="Pause"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="Step"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="Reset"]').exists()).toBe(true)
  })
})
```

**Step 2: Run test to verify controls basics**

Run: `cd frontend && npm test Controls.test.js`
Expected: PASS

**Step 3: Write test for button click handling**

```javascript
it('should call store.play() when play button clicked', async () => {
  const wrapper = mount(Controls)
  const store = useSimulationStore()

  // Mock store method
  store.play = vi.fn()
  store.isConnected = true

  const playButton = wrapper.find('button[aria-label="Play"]')
  await playButton.trigger('click')

  expect(store.play).toHaveBeenCalled()
})
```

**Step 4: Write test for disabled state when not connected**

```javascript
it('should disable controls when not connected', () => {
  const wrapper = mount(Controls)
  const store = useSimulationStore()

  store.isConnected = false

  const playButton = wrapper.find('button[aria-label="Play"]')
  expect(playButton.attributes('disabled')).toBeDefined()
})
```

**Step 5: Run all controls tests**

Run: `cd frontend && npm test Controls.test.js`
Expected: PASS

**Step 6: Commit**

```bash
git add frontend/src/tests/components/Controls.test.js
git commit -m "test: add Controls component interaction tests"
```

---

### Task 14: Test Grid Component

**Files:**

- Create: `frontend/src/tests/components/Grid.test.js`

**Step 1: Write the failing test for Grid rendering**

```javascript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import Grid from '@/components/Grid.vue'
import { useSimulationStore } from '@/stores/simulation'

describe('Grid', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should render 8x8 grid', () => {
    const wrapper = mount(Grid)
    const store = useSimulationStore()

    store.gridWidth = 8
    store.gridHeight = 8

    // Should have 64 cells
    const cells = wrapper.findAll('.grid-cell')
    expect(cells.length).toBe(64)
  })
})
```

**Step 2: Run test to verify grid basics**

Run: `cd frontend && npm test Grid.test.js`
Expected: PASS

**Step 3: Write test for agent rendering**

```javascript
it('should render agent at correct position', () => {
  const wrapper = mount(Grid)
  const store = useSimulationStore()

  store.agents = [{
    id: 'agent_0',
    position: { x: 3, y: 4 },
  }]

  // Should have agent visual at position (3, 4)
  const agent = wrapper.find('.agent')
  expect(agent.exists()).toBe(true)
})
```

**Step 4: Write test for affordance rendering**

```javascript
it('should render affordances at correct positions', () => {
  const wrapper = mount(Grid)
  const store = useSimulationStore()

  store.affordances = [
    { name: 'Bed', position: { x: 0, y: 0 } },
    { name: 'Shower', position: { x: 7, y: 0 } },
    { name: 'Fridge', position: { x: 0, y: 7 } },
    { name: 'Job', position: { x: 7, y: 7 } },
  ]

  const affordances = wrapper.findAll('.affordance')
  expect(affordances.length).toBe(4)
})
```

**Step 5: Run all grid tests**

Run: `cd frontend && npm test Grid.test.js`
Expected: PASS

**Step 6: Commit**

```bash
git add frontend/src/tests/components/Grid.test.js
git commit -m "test: add Grid component rendering tests"
```

---

### Task 15: Test StatsPanel Component

**Files:**

- Create: `frontend/src/tests/components/StatsPanel.test.js`

**Step 1: Write the failing test for StatsPanel**

```javascript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import StatsPanel from '@/components/StatsPanel.vue'
import { useSimulationStore } from '@/stores/simulation'

describe('StatsPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should display current episode and step', () => {
    const wrapper = mount(StatsPanel)
    const store = useSimulationStore()

    store.currentEpisode = 42
    store.currentStep = 137

    expect(wrapper.text()).toContain('42')
    expect(wrapper.text()).toContain('137')
  })
})
```

**Step 2: Run test**

Run: `cd frontend && npm test StatsPanel.test.js`
Expected: PASS

**Step 3: Write test for cumulative reward display**

```javascript
it('should display cumulative reward', () => {
  const wrapper = mount(StatsPanel)
  const store = useSimulationStore()

  store.cumulativeReward = 79.5

  expect(wrapper.text()).toContain('79.5')
})
```

**Step 4: Write test for training metrics (epsilon, loss)**

```javascript
it('should display training metrics when in training mode', () => {
  const wrapper = mount(StatsPanel)
  const store = useSimulationStore()

  store.mode = 'training'
  store.trainingMetrics = {
    epsilon: 0.778,
    avgLoss5: 0.152,
    avgReward5: 65.3,
    bufferSize: 5000,
  }

  expect(wrapper.text()).toContain('0.778')
  expect(wrapper.text()).toContain('0.152')
  expect(wrapper.text()).toContain('65.3')
})
```

**Step 5: Run all stats tests**

Run: `cd frontend && npm test StatsPanel.test.js`
Expected: PASS

**Step 6: Commit**

```bash
git add frontend/src/tests/components/StatsPanel.test.js
git commit -m "test: add StatsPanel component tests"
```

---

## Phase 7: Frontend Integration Tests

### Task 16: Test WebSocket Communication Integration

**Files:**

- Create: `frontend/src/tests/integration/websocket.test.js`

**Step 1: Write integration test for full WebSocket flow**

```javascript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useSimulationStore } from '@/stores/simulation'

describe('WebSocket Integration', () => {
  let mockWs
  let messageHandlers

  beforeEach(() => {
    setActivePinia(createPinia())
    messageHandlers = {}

    mockWs = {
      onopen: null,
      onclose: null,
      onerror: null,
      onmessage: null,
      send: vi.fn(),
      close: vi.fn(),
      readyState: 1, // OPEN
    }

    global.WebSocket = vi.fn(() => mockWs)
  })

  it('should handle complete episode flow', async () => {
    const store = useSimulationStore()

    // Connect
    store.connect('inference')
    mockWs.onopen()

    expect(store.isConnected).toBe(true)

    // Receive connected message
    mockWs.onmessage({
      data: JSON.stringify({
        type: 'connected',
        available_models: ['checkpoint_episode_1000.pt'],
      })
    })

    expect(store.availableModels.length).toBe(1)

    // Receive episode_start
    mockWs.onmessage({
      data: JSON.stringify({
        type: 'episode_start',
        episode: 1,
        epsilon: 0.778,
      })
    })

    expect(store.currentEpisode).toBe(1)

    // Receive state_update
    mockWs.onmessage({
      data: JSON.stringify({
        type: 'state_update',
        step: 10,
        cumulative_reward: 5.5,
        grid: { width: 8, height: 8 },
        agents: [{ id: 'agent_0', position: { x: 3, y: 4 } }],
        affordances: [{ name: 'Bed', position: { x: 0, y: 0 } }],
      })
    })

    expect(store.currentStep).toBe(10)
    expect(store.cumulativeReward).toBe(5.5)

    // Receive episode_end
    mockWs.onmessage({
      data: JSON.stringify({
        type: 'episode_end',
        episode: 1,
        steps: 372,
        total_reward: 79.5,
      })
    })

    expect(store.episodeHistory.length).toBe(1)
    expect(store.episodeHistory[0].reward).toBe(79.5)
  })
})
```

**Step 2: Run integration test**

Run: `cd frontend && npm test websocket.test.js`
Expected: PASS

**Step 3: Write test for training mode flow**

```javascript
it('should handle training mode messages', async () => {
  const store = useSimulationStore()

  store.connect('training')
  mockWs.onopen()

  expect(store.mode).toBe('training')

  // Receive training_started
  mockWs.onmessage({
    data: JSON.stringify({
      type: 'training_started',
      num_episodes: 100,
    })
  })

  expect(store.isTraining).toBe(true)
  expect(store.totalEpisodes).toBe(100)

  // Receive episode_complete with metrics
  mockWs.onmessage({
    data: JSON.stringify({
      type: 'episode_complete',
      episode: 5,
      reward: 45.2,
      length: 200,
      loss: 0.15,
      epsilon: 0.778,
      avg_reward_5: 40.1,
      avg_length_5: 180,
      avg_loss_5: 0.18,
      buffer_size: 5000,
    })
  })

  expect(store.trainingMetrics.epsilon).toBe(0.778)
  expect(store.trainingMetrics.avgReward5).toBe(40.1)
})
```

**Step 4: Run training flow test**

Run: `cd frontend && npm test websocket.test.js`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/tests/integration/websocket.test.js
git commit -m "test: add frontend WebSocket integration tests"
```

---

## Phase 8: Test Configuration and Documentation

### Task 17: Add pytest.ini and Test Markers

**Files:**

- Modify: `pyproject.toml`

**Step 1: Add pytest markers to pyproject.toml**

Edit `pyproject.toml` to add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "asyncio: marks tests as async",
]
addopts = [
    "--strict-markers",
    "--tb=short",
    "--disable-warnings",
]
asyncio_mode = "auto"
```

**Step 2: Verify markers work**

Run: `uv run pytest --markers`
Expected: Should list all custom markers

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "test: add pytest markers for test categorization"
```

---

### Task 18: Update CLAUDE.md with Testing Commands

**Files:**

- Modify: `CLAUDE.md`

**Step 1: Add comprehensive testing section to CLAUDE.md**

Add to the "Development Commands" section:

```markdown
### Testing

#### Backend (Python/pytest)

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=hamlet --cov-report=term-missing --cov-report=html

# Run specific test file
uv run pytest tests/test_environment/test_hamlet_env.py

# Run specific test function
uv run pytest tests/test_environment/test_hamlet_env.py::test_environment_step

# Run by marker
uv run pytest -m "not slow"  # Skip slow integration tests
uv run pytest -m unit         # Only unit tests
uv run pytest -m integration  # Only integration tests

# Run with verbose output
uv run pytest -v

# Run with print statements visible
uv run pytest -s

# Run failed tests from last run
uv run pytest --lf

# Run tests in parallel (requires pytest-xdist)
uv run pytest -n auto
```

#### Frontend (Vitest)

```bash
# Run all frontend tests
cd frontend && npm test

# Run in watch mode
cd frontend && npm test -- --watch

# Run with UI
cd frontend && npm run test:ui

# Run with coverage
cd frontend && npm run test:coverage

# Run specific test file
cd frontend && npm test simulation.test.js

# Run tests matching pattern
cd frontend && npm test -- --grep="WebSocket"
```

#### Test Coverage Targets

- Backend: Aim for >80% coverage on core modules (environment, agent, training)
- Frontend: Aim for >70% coverage on components and stores
- Integration tests: Cover all major user flows (training, visualization, WebSocket communication)

```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add comprehensive testing commands to CLAUDE.md"
```

---

### Task 19: Create Test README

**Files:**

- Create: `tests/README.md`

**Step 1: Write comprehensive test documentation**

```markdown
# Hamlet Test Suite

Comprehensive test suite for the Hamlet Deep Reinforcement Learning environment.

## Philosophy

**Important:** This is a pedagogical project where "interesting failures" are teaching moments. Our tests validate:
- ✅ Core mechanics work correctly
- ✅ Infrastructure doesn't crash
- ✅ APIs return expected data structures
- ❌ NOT "correct" agent strategies or learning outcomes

See `CLAUDE.md` "Known Behaviors (Not Bugs!)" for behaviors that are intentional.

## Test Structure

```

tests/
├── test_environment/       # Environment mechanics (grid, meters, affordances)
├── test_agent/            # Agent networks, replay buffer, observation utils
├── test_training/         # Training loop, config, checkpoints, metrics
├── test_web/              # FastAPI server, WebSocket, simulation runner
└── integration/           # End-to-end flows (training, WebSocket)

```

## Running Tests

### Quick Start
```bash
# Backend - all tests
uv run pytest

# Frontend - all tests
cd frontend && npm test
```

### By Category

```bash
# Unit tests only (fast)
uv run pytest -m unit

# Integration tests only (slow)
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

### Coverage

```bash
# Backend coverage
uv run pytest --cov=hamlet --cov-report=html
# Open htmlcov/index.html

# Frontend coverage
cd frontend && npm run test:coverage
# Open coverage/index.html
```

## Writing Tests

### Backend Tests (pytest)

Follow TDD with @superpowers:test-driven-development skill:

1. Write the failing test
2. Run it to see it fail
3. Implement minimal code
4. Run it to see it pass
5. Commit

Example:

```python
import pytest
from hamlet.environment.hamlet_env import HamletEnv

def test_environment_resets():
    """Verify environment resets to initial state."""
    env = HamletEnv(num_agents=1)
    obs = env.reset()

    assert "agent_0" in obs
    assert len(obs["agent_0"]) == 70  # Observation vector size
```

### Frontend Tests (Vitest)

Use Vue Test Utils and mock WebSocket:

```javascript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import MyComponent from '@/components/MyComponent.vue'

describe('MyComponent', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should render correctly', () => {
    const wrapper = mount(MyComponent)
    expect(wrapper.text()).toContain('Expected Text')
  })
})
```

## Continuous Integration

Tests run automatically on:

- Pull requests
- Pushes to main
- Manual workflow dispatch

CI runs:

1. Backend unit tests (fast)
2. Backend integration tests (slow)
3. Frontend tests
4. Coverage reports

## Test Fixtures

Common fixtures in `tests/conftest.py`:

- `env`: Pre-configured HamletEnv
- `agent`: Pre-configured DRLAgent
- `tmp_checkpoint_dir`: Temporary directory for checkpoint tests

## Debugging Tests

### Backend

```bash
# Run with debugger breakpoints
uv run pytest --pdb

# Show print statements
uv run pytest -s

# Very verbose
uv run pytest -vv
```

### Frontend

```bash
# Watch mode for debugging
cd frontend && npm test -- --watch

# UI mode for interactive debugging
cd frontend && npm run test:ui
```

## Known Test Quirks

### Async Tests

Backend async tests (web server, WebSocket) require `@pytest.mark.asyncio` decorator.

### WebSocket Mocking

Frontend WebSocket is globally mocked in `src/tests/setup.js`. Real WebSocket tests are in integration suite.

### Coverage Warnings

You may see "module was never imported" warnings - this is normal for entry point scripts.

## Contributing Tests

When adding new features:

1. Write tests FIRST (TDD)
2. Use appropriate markers (@pytest.mark.slow for >1s tests)
3. Update this README if adding new test categories
4. Ensure tests pass before committing

## Questions?

See `CLAUDE.md` for project-specific testing guidance.
See `docs/scraps/` for pedagogical context on "bugs" that are actually features.

```

**Step 2: Commit**

```bash
git add tests/README.md
git commit -m "docs: add comprehensive test suite documentation"
```

---

### Task 20: Create Frontend Test README

**Files:**

- Create: `frontend/src/tests/README.md`

**Step 1: Write frontend-specific test documentation**

```markdown
# Frontend Test Suite

Comprehensive test suite for the Hamlet visualization frontend (Vue 3 + Pinia).

## Stack

- **Test Runner:** Vitest
- **Component Testing:** @vue/test-utils
- **DOM Environment:** happy-dom
- **State Management:** Pinia (with test utilities)

## Test Structure

```

frontend/src/tests/
├── setup.js                # Global test configuration
├── sanity.test.js         # Vitest setup verification
├── stores/                # Pinia store tests
│   └── simulation.test.js
├── components/            # Vue component tests
│   ├── MeterPanel.test.js
│   ├── Grid.test.js
│   ├── Controls.test.js
│   └── StatsPanel.test.js
└── integration/           # Integration tests
    └── websocket.test.js

```

## Running Tests

```bash
# Watch mode (recommended for development)
npm test -- --watch

# Run once
npm test

# With UI (interactive, great for debugging)
npm run test:ui

# With coverage
npm run test:coverage
```

## Test Patterns

### Component Tests

```javascript
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import MyComponent from '@/components/MyComponent.vue'

describe('MyComponent', () => {
  beforeEach(() => {
    setActivePinia(createPinia())  // Fresh store for each test
  })

  it('should render prop correctly', () => {
    const wrapper = mount(MyComponent, {
      props: {
        title: 'Test Title'
      }
    })

    expect(wrapper.text()).toContain('Test Title')
  })
})
```

### Store Tests

```javascript
import { setActivePinia, createPinia } from 'pinia'
import { useSimulationStore } from '@/stores/simulation'

describe('Simulation Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should update state correctly', () => {
    const store = useSimulationStore()

    store.currentEpisode = 5

    expect(store.currentEpisode).toBe(5)
  })
})
```

### Mocking WebSocket

WebSocket is globally mocked in `setup.js`. For custom behavior:

```javascript
const mockWs = {
  send: vi.fn(),
  close: vi.fn(),
  onopen: null,
  onmessage: null,
}

global.WebSocket = vi.fn(() => mockWs)

// Trigger events
mockWs.onopen()
mockWs.onmessage({ data: JSON.stringify({ type: 'connected' }) })
```

## Coverage Goals

- **Components:** >70% line coverage
- **Stores:** >80% line coverage
- **Integration:** All major user flows covered

## Debugging

### Use Vitest UI

```bash
npm run test:ui
```

Opens interactive browser UI showing:

- Test results
- Component rendering
- Console logs
- Code coverage

### Use `console.log` Freely

Vitest shows console output by default. No need for `-s` flag like pytest.

### Use `wrapper.html()`

```javascript
console.log(wrapper.html())  // See rendered HTML
console.log(wrapper.text())  // See text content
```

## Common Issues

### "Cannot find module @/..."

Make sure `vitest.config.js` has correct path alias:

```javascript
resolve: {
  alias: {
    '@': fileURLToPath(new URL('./src', import.meta.url))
  }
}
```

### "Store not available"

Make sure to create fresh Pinia in `beforeEach`:

```javascript
beforeEach(() => {
  setActivePinia(createPinia())
})
```

### Component not updating

Use `await wrapper.vm.$nextTick()` after state changes:

```javascript
store.currentEpisode = 5
await wrapper.vm.$nextTick()
expect(wrapper.text()).toContain('5')
```

## Best Practices

1. **Isolate tests:** Use `beforeEach` to reset state
2. **Test user behavior:** Test what users see/do, not implementation
3. **Mock external deps:** WebSocket, localStorage, etc.
4. **Use descriptive names:** `it('should show error when connection fails')`
5. **Test edge cases:** Empty states, loading states, error states

## Resources

- [Vitest Docs](https://vitest.dev/)
- [Vue Test Utils Docs](https://test-utils.vuejs.org/)
- [Pinia Testing Docs](https://pinia.vuejs.org/cookbook/testing.html)

```

**Step 2: Commit**

```bash
git add frontend/src/tests/README.md
git commit -m "docs: add frontend test suite documentation"
```

---

## Summary

Plan complete! This comprehensive test suite covers:

**Backend (Python/pytest):**

- ✅ Unit tests for affordances, renderer, checkpoint manager
- ✅ Web server tests (FastAPI, WebSocket, simulation runner, training server)
- ✅ Integration tests (full training loop, WebSocket flow)

**Frontend (Vue 3/Vitest):**

- ✅ Testing infrastructure setup (Vitest, Vue Test Utils, happy-dom)
- ✅ Store tests (Pinia simulation store)
- ✅ Component tests (MeterPanel, Grid, Controls, StatsPanel)
- ✅ Integration tests (WebSocket communication)

**Documentation:**

- ✅ Test markers and configuration
- ✅ Updated CLAUDE.md with testing commands
- ✅ Backend test README
- ✅ Frontend test README

**Total: 20 tasks, ~40 test files, comprehensive coverage**

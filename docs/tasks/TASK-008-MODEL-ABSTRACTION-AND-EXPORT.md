# TASK-008: Model Abstraction and Export

**Status**: Planned
**Priority**: MEDIUM (enables external model usage)
**Estimated Effort**: 36 hours (includes ONNX export)
**Dependencies**: None (fallback format implemented for BRAIN_AS_CODE)
**Enables**: External model usage, transfer learning, model serving
**Created**: 2025-11-05
**Completed**: TBD

**Keywords**: model-abstraction, inference, checkpoint-format, onnx-export, brain-as-code, external-usage, transfer-learning, portability
**Subsystems**: inference (new), agent (networks), population (refactor), demo (LiveInferenceServer)
**Architecture Impact**: Major (new inference subsystem, refactored checkpoint format)
**Breaking Changes**: No (backward compatible with v2 training checkpoints)

---

## AI-Friendly Summary (Skim This First!)

**What**: Extract standalone HamletModel class for inference, decouple from training infrastructure, add lightweight inference checkpoints with ONNX export

**Why**: Students need to extract trained models for external projects, transfer learning, and research without dragging entire training stack

**Scope**: HamletModel class, inference checkpoint format (v1), ONNX export, backward compatibility; does NOT include model serving API or batch inference tools

**Quick Assessment**:

- **Current Limitation**: Can't load model without VectorizedPopulation, curriculum, exploration (training dependencies)
- **After Implementation**: One-line model loading, portable checkpoints (~10MB vs ~50MB), ONNX export for other frameworks
- **Unblocks**: Student projects, transfer learning, ensemble methods, ablation studies
- **Impact Radius**: 8 files (4 new, 4 modified)

**Decision Point**: If you're not working on inference/model export, STOP READING HERE.

---

## Problem Statement

### Current Constraint

**Model weights are tightly coupled to training infrastructure:**

```python
# Current: Can't load model without full training stack
from townlet.population.vectorized import VectorizedPopulation
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.environment.vectorized_env import VectorizedHamletEnv

# Must recreate entire training environment just to get model
env = VectorizedHamletEnv(num_agents=1, grid_size=8, ...)  # ~500 lines of config
curriculum = AdversarialCurriculum(...)  # Don't even use for inference!
exploration = AdaptiveIntrinsicExploration(...)  # Just need epsilon value!
population = VectorizedPopulation(env, curriculum, exploration, ...)  # Finally get q_network

# Load checkpoint
checkpoint = torch.load("checkpoint_ep10000.pt")
population.q_network.load_state_dict(checkpoint["population_state"]["q_network"])

# Now can run inference
q_values = population.q_network(obs)  # But still need hidden state management...
```

**Problems:**

1. **Tight Coupling**: Need 4 classes (Env, Curriculum, Exploration, Population) just to load model
2. **Complex Checkpoints**: Mix training state (optimizer, replay buffer) with model weights (~50MB, 80% waste)
3. **No Model Class**: Networks are just `nn.Module`, no inference API
4. **Hidden State Chaos**: LSTM hidden state managed in 3 places (network, population, episode containers)
5. **No Portability**: Can't use models outside HAMLET codebase

**Example showing complexity** (src/townlet/demo/live_inference.py:230):

```python
# LiveInferenceServer must recreate full training stack
self.env = VectorizedHamletEnv(...)  # Full environment (don't need for pure inference)
self.curriculum = AdversarialCurriculum(...)  # Don't use curriculum for inference!
self.exploration = AdaptiveIntrinsicExploration(...)  # Just need epsilon!
self.population = VectorizedPopulation(env, curriculum, exploration, ...)  # Just to get q_network

# Just want:
# self.model = HamletModel.load_from_checkpoint("model.pt")
# result = self.model.predict(obs, epsilon=0.1)
```

### Why This Is Technical Debt, Not Design

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: More expressive

- ✅ Enables: Use models in external projects (student research)
- ✅ Enables: Transfer learning (load pretrained, fine-tune)
- ✅ Enables: Ensemble methods (combine multiple models)
- ✅ Enables: Model serving (REST API, if needed later)
- ✅ Enables: ONNX export (use in other frameworks)
- ❌ Does NOT: Break training (VectorizedPopulation still works)

**Conclusion**: Technical debt from missing abstraction layer

### Impact of Current Constraint

**Cannot Create**:

- **Student Projects**: "Load my trained model in custom environment"
- **Transfer Learning**: "Fine-tune pretrained model on new task"
- **Ensemble Methods**: "Combine 5 models trained with different seeds"
- **Model Comparison**: "Compare SimpleQNetwork vs RecurrentSpatialQNetwork"
- **External Deployment**: "Use HAMLET model in production system"

**Pedagogical Cost**:

- Students can't extract models for their research projects
- High barrier to experimentation (must understand training infrastructure)
- Can't A/B test different architectures easily

**Research Cost**:

- Ablation studies require modifying training code
- Transfer learning experiments blocked
- Ensemble methods blocked

**From Analysis**: High-leverage change (36 hours → enables entire class of use cases)

---

## Solution Overview

### Design Principle

**Core Philosophy**: "Models are first-class citizens, independent of training infrastructure"

**Key Insight**: Separate model (inference) from training (learning). Model should be pure function: `predict(obs, hidden) → (actions, q_values, new_hidden)`.

### Architecture Changes

**1. New Subsystem**: `src/townlet/inference/`

- `model.py`: HamletModel class (standalone inference)
- `checkpoint.py`: Inference checkpoint format (v1)
- `onnx_export.py`: ONNX export utilities

**2. agent/networks.py**: Networks remain pure `nn.Module` (no changes)

**3. population/vectorized.py**: Refactor to use HamletModel internally (optional, Phase 7)

**4. demo/live_inference.py**: Refactor to use HamletModel (drop VectorizedPopulation dependency)

**5. demo/runner.py**: Save inference checkpoints alongside training checkpoints

### Compatibility Strategy

**Backward Compatibility**:

- HamletModel can load from v2 training checkpoints (existing format)
- Training checkpoints still saved (v2 format unchanged)
- New inference checkpoints (v1 format) saved alongside training checkpoints

**Migration Path**:

- Old code works (load v2 training checkpoints)
- New code preferred (load v1 inference checkpoints, faster)
- Migration script: convert v2 → v1 (one-time operation)

**Versioning**:

- v2 training checkpoints: `checkpoint_ep*.pt` (existing)
- v1 inference checkpoints: `model_ep*.pt` (new)
- Format version in checkpoint dict: `"format_version": "inference_v1"`

---

## Detailed Design

### Phase 1: HamletModel Core (8 hours)

**Objective**: Create standalone HamletModel class with functional API

**Changes**:

- File: `src/townlet/inference/__init__.py` (NEW)
  ```python
  from .model import HamletModel
  __all__ = ["HamletModel"]
  ```

- File: `src/townlet/inference/model.py` (NEW)
  ```python
  """Standalone model for HAMLET inference."""

  import torch
  import torch.nn as nn
  from pathlib import Path
  from torch import Tensor
  import logging

  from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

  logger = logging.getLogger(__name__)

  class HamletModel:
      """Self-contained inference model."""

      def __init__(self, network: nn.Module):
          """Initialize with compiled network (from BRAIN_AS_CODE)."""
          self.network = network
          self.is_recurrent = isinstance(network, RecurrentSpatialQNetwork)
          self.device = next(network.parameters()).device

      def predict(
          self,
          obs: torch.Tensor,
          hidden: tuple[Tensor, Tensor] | None = None,
          epsilon: float = 0.0,
      ) -> dict:
          """Pure inference function.

          Args:
              obs: [batch, obs_dim] observations
              hidden: Optional LSTM hidden state (h, c) for recurrent models
              epsilon: Exploration rate (0.0 = greedy)

          Returns:
              {
                  "actions": [batch] selected actions,
                  "q_values": [batch, action_dim] Q-values,
                  "hidden": (h, c) new hidden state (recurrent only),
              }
          """
          with torch.no_grad():
              if self.is_recurrent:
                  q_values, new_hidden = self.network(obs, hidden)
              else:
                  q_values = self.network(obs)
                  new_hidden = None

              # Action selection
              if epsilon > 0:
                  # Epsilon-greedy
                  batch_size = obs.shape[0]
                  random_mask = torch.rand(batch_size, device=self.device) < epsilon
                  random_actions = torch.randint(0, q_values.shape[1], (batch_size,), device=self.device)
                  greedy_actions = torch.argmax(q_values, dim=1)
                  actions = torch.where(random_mask, random_actions, greedy_actions)
              else:
                  # Greedy
                  actions = torch.argmax(q_values, dim=1)

              return {
                  "actions": actions,
                  "q_values": q_values,
                  "hidden": new_hidden,
              }

      @classmethod
      def load_from_checkpoint(cls, path: Path, device: str | torch.device = "cpu") -> "HamletModel":
          """Load from inference checkpoint (v1 format)."""
          # Implementation in Phase 3
          ...

      @classmethod
      def load_from_training_checkpoint(cls, path: Path, device: str | torch.device = "cpu") -> "HamletModel":
          """Load from training checkpoint (v2 format, backward compat)."""
          # Implementation in Phase 3
          ...
  ```

**Tests**:

- [ ] Test: Load SimpleQNetwork, predict with batch of observations
- [ ] Test: Load RecurrentSpatialQNetwork, predict with hidden state
- [ ] Test: Epsilon-greedy action selection
- [ ] Test: Greedy action selection (epsilon=0.0)
- [ ] Test: Batch prediction (multiple observations)

**Success Criteria**: HamletModel.predict() works for both network types

---

### Phase 2: Inference Checkpoint Format (4 hours)

**Objective**: Define inference_v1 checkpoint format with BRAIN_AS_CODE integration

**Changes**:

- File: `src/townlet/inference/checkpoint.py` (NEW)
  ```python
  """Inference checkpoint format (v1)."""

  import torch
  import torch.nn as nn
  from pathlib import Path
  import time
  import hashlib
  import json
  import logging

  logger = logging.getLogger(__name__)

  INFERENCE_CHECKPOINT_VERSION = "inference_v1"

  def save_inference_checkpoint(
      model: HamletModel,
      path: Path,
      metadata: dict,
      brain_spec: dict | None = None,
  ):
      """Save inference checkpoint.

      Args:
          model: HamletModel to save
          path: Output path
          metadata: Training metadata (episode, epsilon, etc.)
          brain_spec: brain.yaml content (from BRAIN_AS_CODE) or None for fallback
      """
      checkpoint = {
          "format_version": INFERENCE_CHECKPOINT_VERSION,

          # Model weights
          "state_dict": model.network.state_dict(),

          # Metadata
          "metadata": metadata,
          "created_at": time.time(),
      }

      # BRAIN_AS_CODE integration (optional, preferred)
      if brain_spec is not None:
          checkpoint["brain_spec"] = brain_spec
          checkpoint["brain_spec_hash"] = _compute_hash(brain_spec)
      else:
          # Fallback: Embed minimal network params (no BRAIN_AS_CODE)
          checkpoint["network_params"] = _extract_network_params(model.network)

      torch.save(checkpoint, path)

  def load_inference_checkpoint(path: Path, device: str | torch.device) -> tuple[dict | None, dict]:
      """Load inference checkpoint.

      Returns:
          (brain_spec_or_None, checkpoint_dict)
          - brain_spec is None if checkpoint uses fallback format
      """
      checkpoint = torch.load(path, weights_only=False, map_location=device)

      # Validate format
      if checkpoint.get("format_version") != INFERENCE_CHECKPOINT_VERSION:
          raise ValueError(f"Unsupported checkpoint version: {checkpoint.get('format_version')}")

      # Extract brain spec (if available)
      brain_spec = checkpoint.get("brain_spec")
      if brain_spec is not None:
          # Verify brain spec hash (optional, warns if mismatch)
          expected_hash = checkpoint.get("brain_spec_hash")
          actual_hash = _compute_hash(brain_spec)
          if expected_hash != actual_hash:
              logger.warning(f"Brain spec hash mismatch (expected {expected_hash}, got {actual_hash})")

      return brain_spec, checkpoint

  def _extract_network_params(network: nn.Module) -> dict:
      """Extract minimal network params for fallback checkpoint format.

      Returns:
          {
              "network_type": "simple" | "recurrent",
              "obs_dim": int,
              "action_dim": int,
              "window_size": int (recurrent only),
          }
      """
      from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

      if isinstance(network, SimpleQNetwork):
          # Infer from state_dict
          state_dict = network.state_dict()
          obs_dim = state_dict["net.0.weight"].shape[1]
          action_dim = state_dict["net.6.weight"].shape[0]
          return {
              "network_type": "simple",
              "obs_dim": obs_dim,
              "action_dim": action_dim,
          }
      elif isinstance(network, RecurrentSpatialQNetwork):
          # Infer from state_dict
          state_dict = network.state_dict()
          action_dim = state_dict["q_head.2.weight"].shape[0]
          window_size = int(state_dict["vision_encoder.0.weight"].shape[2])  # Conv2d kernel size
          return {
              "network_type": "recurrent",
              "action_dim": action_dim,
              "window_size": window_size,
          }
      else:
          raise ValueError(f"Unknown network type: {type(network)}")

  def _compute_hash(obj: dict) -> str:
      """Compute SHA-256 hash of dictionary."""
      import hashlib
      import json
      content = json.dumps(obj, sort_keys=True)
      return f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
  ```

**Format (with BRAIN_AS_CODE - preferred)**:

```python
{
    "format_version": "inference_v1",

    # BRAIN_AS_CODE integration (preferred)
    "brain_spec": {
        "network": {
            "type": "recurrent_spatial",
            "vision_encoder": {"type": "cnn", "filters": [16, 32], ...},
            "position_encoder": {"type": "mlp", "hidden_dim": 32},
            "meter_encoder": {"type": "mlp", "hidden_dim": 32},
            "affordance_encoder": {"type": "mlp", "hidden_dim": 32},
            "lstm": {"hidden_dim": 256},
            "q_head": {"type": "mlp", "hidden_dims": [128]},
        }
    },
    "brain_spec_hash": "sha256:abc123...",

    # Model weights (~10MB for RecurrentSpatialQNetwork)
    "state_dict": {...},

    # Metadata
    "metadata": {
        "training_episode": 10000,
        "training_config_path": "configs/L2_partial_observability",
        "epsilon": 0.05,
        "curriculum_stage": 5,
    },
    "created_at": 1699123456.78,
}
```

**Format (fallback without BRAIN_AS_CODE)**:

```python
{
    "format_version": "inference_v1",

    # Fallback: Minimal network params (when BRAIN_AS_CODE unavailable)
    "network_params": {
        "network_type": "recurrent",
        "action_dim": 5,
        "window_size": 5,
    },

    # Model weights (~10MB for RecurrentSpatialQNetwork)
    "state_dict": {...},

    # Metadata
    "metadata": {
        "training_episode": 10000,
        "training_config_path": "configs/L2_partial_observability",
        "epsilon": 0.05,
        "curriculum_stage": 5,
    },
    "created_at": 1699123456.78,
}
```

**Tests**:

- [ ] Test: Save and load inference checkpoint (round-trip)
- [ ] Test: Brain spec hash validation
- [ ] Test: Checkpoint ~5-10x smaller than training checkpoint

**Success Criteria**: Inference checkpoints are self-contained and portable

---

### Phase 3: Load from Checkpoints (6 hours)

**Objective**: HamletModel can load from both v1 (inference) and v2 (training) checkpoints

**Changes**:

- File: `src/townlet/inference/model.py`
  - Implement `load_from_checkpoint()` (v1 inference format)
  - Implement `load_from_training_checkpoint()` (v2 training format, backward compat)
  - Handle both BRAIN_AS_CODE and fallback formats:
    ```python
    @classmethod
    def load_from_checkpoint(cls, path: Path, device: str | torch.device = "cpu") -> "HamletModel":
        """Load from inference checkpoint (v1)."""
        from townlet.inference.checkpoint import load_inference_checkpoint

        brain_spec, checkpoint = load_inference_checkpoint(path, device)

        # Try BRAIN_AS_CODE first (preferred)
        if brain_spec is not None:
            try:
                from townlet.compiler import compile_brain
                network = compile_brain(brain_spec)
            except ImportError:
                # BRAIN_AS_CODE not available, fall through to fallback
                logger.warning("BRAIN_AS_CODE compiler not available, using fallback")
                brain_spec = None

        # Fallback: Build network from minimal params
        if brain_spec is None:
            network_params = checkpoint["network_params"]
            network = _build_network_from_params(network_params)

        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        network.to(device)

        return cls(network=network)

    @classmethod
    def load_from_training_checkpoint(cls, path: Path, device: str | torch.device = "cpu") -> "HamletModel":
        """Load from training checkpoint (v2, backward compat)."""
        checkpoint = torch.load(path, weights_only=False, map_location=device)

        # Extract training config
        training_config = checkpoint["training_config"]
        pop_config = training_config["population"]
        env_config = training_config["environment"]

        # Infer network params from config
        network_type = pop_config["network_type"]

        if network_type == "simple":
            from townlet.agent.networks import SimpleQNetwork
            # Infer obs_dim and action_dim from checkpoint state_dict
            state_dict = checkpoint["population_state"]["q_network"]
            obs_dim = state_dict["net.0.weight"].shape[1]
            action_dim = _infer_action_dim(state_dict, network_type="simple")
            network = SimpleQNetwork(obs_dim=obs_dim, action_dim=action_dim)
        elif network_type == "recurrent":
            from townlet.agent.networks import RecurrentSpatialQNetwork
            vision_range = env_config.get("vision_range", 2)
            window_size = 2 * vision_range + 1
            state_dict = checkpoint["population_state"]["q_network"]
            action_dim = _infer_action_dim(state_dict, network_type="recurrent")
            network = RecurrentSpatialQNetwork(
                action_dim=action_dim,
                window_size=window_size,
            )

        network.load_state_dict(checkpoint["population_state"]["q_network"])
        network.eval()
        network.to(device)

        return cls(network=network)

    # Helper functions (module-level, not class methods)

    def _build_network_from_params(params: dict) -> nn.Module:
        """Build network from minimal params (fallback format).

        Args:
            params: {
                "network_type": "simple" | "recurrent",
                "obs_dim": int (simple only),
                "action_dim": int,
                "window_size": int (recurrent only),
            }

        Returns:
            Initialized network (weights to be loaded separately)
        """
        from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

        network_type = params["network_type"]
        action_dim = params["action_dim"]

        if network_type == "simple":
            obs_dim = params["obs_dim"]
            return SimpleQNetwork(obs_dim=obs_dim, action_dim=action_dim)
        elif network_type == "recurrent":
            window_size = params["window_size"]
            return RecurrentSpatialQNetwork(action_dim=action_dim, window_size=window_size)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def _infer_action_dim(state_dict: dict, network_type: str) -> int:
        """Infer action dimension from checkpoint state_dict.

        Args:
            state_dict: Network state dict
            network_type: "simple" | "recurrent"

        Returns:
            action_dim (e.g., 5 for HAMLET)
        """
        if network_type == "simple":
            # SimpleQNetwork: net.6 is final layer (Linear(128, action_dim))
            return state_dict["net.6.weight"].shape[0]
        elif network_type == "recurrent":
            # RecurrentSpatialQNetwork: q_head.2 is final layer
            return state_dict["q_head.2.weight"].shape[0]
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    ```

**Tests**:

- [ ] Test: Load v1 inference checkpoint (both network types)
- [ ] Test: Load v2 training checkpoint (both network types)
- [ ] Test: Weights loaded correctly (verify Q-values match)
- [ ] Test: Network in eval mode

**Success Criteria**: Can load models from both checkpoint formats

---

### Phase 4: Save Inference Checkpoints (2 hours)

**Objective**: DemoRunner saves inference checkpoints alongside training checkpoints

**Changes**:

- File: `src/townlet/demo/runner.py`
  - Add helper method to load brain spec (with fallback):
    ```python
    def _load_brain_spec(self) -> dict | None:
        """Load brain.yaml spec from config directory.

        Returns:
            brain_spec dict if BRAIN_AS_CODE available, None for fallback
        """
        brain_path = self.config_dir / "brain.yaml"

        # Try to load brain.yaml if it exists
        if brain_path.exists():
            try:
                import yaml
                with open(brain_path) as f:
                    brain_spec = yaml.safe_load(f)
                logger.info(f"Loaded brain spec from {brain_path}")
                return brain_spec
            except Exception as e:
                logger.warning(f"Failed to load brain.yaml: {e}, using fallback")
                return None
        else:
            # BRAIN_AS_CODE not yet implemented, use fallback
            logger.debug("brain.yaml not found, using fallback checkpoint format")
            return None
    ```

  - Add helper methods to access current training state:
    ```python
    def _get_current_epsilon(self) -> float:
        """Get current exploration epsilon from exploration strategy."""
        return self.exploration.epsilon

    def _get_current_stage(self) -> int:
        """Get current curriculum stage."""
        return self.curriculum.current_stage
    ```

  - In `save_checkpoint()`: After saving training checkpoint, save inference checkpoint
    ```python
    def save_checkpoint(self):
        episode = self.current_episode

        # Existing: Save training checkpoint (v2 format)
        training_checkpoint = {...}  # Full state
        training_path = self.checkpoint_dir / f"checkpoint_ep{episode:05d}.pt"
        torch.save(training_checkpoint, training_path)
        logger.info(f"Saved training checkpoint: {training_path.name}")

        # NEW: Save inference checkpoint (v1 format)
        from townlet.inference.checkpoint import save_inference_checkpoint
        from townlet.inference import HamletModel

        # Wrap network in HamletModel
        model = HamletModel(network=self.population.q_network)

        # Try to load brain spec (preferred, but optional)
        brain_spec = self._load_brain_spec()

        # Metadata
        metadata = {
            "training_episode": episode,
            "training_config_path": str(self.config_dir),
            "epsilon": self._get_current_epsilon(),  # Extract from self.exploration.epsilon
            "curriculum_stage": self._get_current_stage(),  # Extract from self.curriculum.current_stage
        }

        # Save inference checkpoint (with or without brain_spec)
        inference_path = self.checkpoint_dir / f"model_ep{episode:05d}.pt"
        save_inference_checkpoint(model, inference_path, metadata, brain_spec)
        logger.info(f"Saved inference checkpoint: {inference_path.name} "
                   f"(format: {'BRAIN_AS_CODE' if brain_spec else 'fallback'})")
    ```

**Tests**:

- [ ] Test: Both checkpoints saved (checkpoint_ep*.pt and model_ep*.pt)
- [ ] Test: Inference checkpoint ~5-10x smaller
- [ ] Test: Can load from inference checkpoint and predict

**Success Criteria**: Training saves both checkpoint formats

---

### Phase 5: Refactor LiveInferenceServer (6 hours)

**Objective**: LiveInferenceServer uses HamletModel, drops VectorizedPopulation dependency

**Changes**:

- File: `src/townlet/demo/live_inference.py`
  - Remove: VectorizedPopulation, AdversarialCurriculum, AdaptiveIntrinsicExploration creation
  - Add: HamletModel loading
  - Simplify: Use model.predict() instead of population methods

  **Before**:
  ```python
  class LiveInferenceServer:
      def __init__(self, ...):
          # Create full training stack
          self.env = VectorizedHamletEnv(...)
          self.curriculum = AdversarialCurriculum(...)
          self.exploration = AdaptiveIntrinsicExploration(...)
          self.population = VectorizedPopulation(env, curriculum, exploration, ...)

      async def _run_episode(self):
          actions = self.population.select_epsilon_greedy_actions(self.env, epsilon)
  ```

  **After**:
  ```python
  class LiveInferenceServer:
      def __init__(self, ...):
          # Just need environment (for state transitions) and model
          self.env = VectorizedHamletEnv(...)
          self.model: HamletModel | None = None

      async def _load_checkpoint(self):
          # Load inference checkpoint (or training checkpoint for backward compat)
          if inference_checkpoint_exists:
              self.model = HamletModel.load_from_checkpoint(latest_inference_checkpoint)
          else:
              self.model = HamletModel.load_from_training_checkpoint(latest_training_checkpoint)

      async def _run_episode(self):
          result = self.model.predict(obs, hidden, epsilon)
          actions = result["actions"]
          q_values = result["q_values"]
          hidden = result["hidden"]
  ```

**Tests**:

- [ ] Integration test: Load model and run episode
- [ ] Test: Works with both v1 and v2 checkpoints
- [ ] Test: LSTM hidden state managed correctly
- [ ] Test: Backward compatible (old checkpoints work)

**Success Criteria**: LiveInferenceServer simplified, no VectorizedPopulation dependency

---

### Phase 6: ONNX Export (4 hours)

**Objective**: Export HamletModel to ONNX format for use in other frameworks

**Changes**:

- File: `src/townlet/inference/onnx_export.py` (NEW)
  ```python
  """ONNX export utilities for HamletModel."""

  import torch
  import torch.nn as nn
  import torch.onnx
  import numpy as np
  import logging
  from pathlib import Path

  from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

  logger = logging.getLogger(__name__)

  def _infer_obs_dim(network: nn.Module) -> int:
      """Infer observation dimension from network state_dict.

      Args:
          network: SimpleQNetwork or RecurrentSpatialQNetwork

      Returns:
          obs_dim for creating example inputs
      """
      from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork

      if isinstance(network, SimpleQNetwork):
          # SimpleQNetwork: net.0 is first layer (Linear(obs_dim, 256))
          return network.state_dict()["net.0.weight"].shape[1]
      elif isinstance(network, RecurrentSpatialQNetwork):
          # RecurrentSpatialQNetwork: observation is decomposed
          # Local grid: window_size * window_size (e.g., 5×5 = 25)
          # Position: 2
          # Meters: 8
          # Affordance at position: 15
          # Temporal extras: 4
          # Total: 25 + 2 + 8 + 15 + 4 = 54 for window_size=5
          state_dict = network.state_dict()
          window_size = int(state_dict["vision_encoder.0.weight"].shape[2])  # Conv2d kernel size
          obs_dim = window_size * window_size + 2 + 8 + 15 + 4
          return obs_dim
      else:
          raise ValueError(f"Unknown network type: {type(network)}")

  def export_to_onnx(
      model: HamletModel,
      output_path: Path,
      batch_size: int = 1,
      opset_version: int = 17,
  ):
      """Export HamletModel to ONNX format.

      Args:
          model: HamletModel to export
          output_path: Output .onnx file path
          batch_size: Batch size for example input
          opset_version: ONNX opset version (17 = ONNX 1.12+)
      """
      model.network.eval()

      # Create example input
      if model.is_recurrent:
          # Recurrent: obs + hidden state
          obs_dim = _infer_obs_dim(model.network)
          example_obs = torch.randn(batch_size, obs_dim, device=model.device)

          # LSTM hidden state: (h, c), each [1, batch, hidden_dim]
          hidden_dim = model.network.hidden_dim
          h = torch.zeros(1, batch_size, hidden_dim, device=model.device)
          c = torch.zeros(1, batch_size, hidden_dim, device=model.device)
          example_hidden = (h, c)

          # Export with dynamic batch size
          torch.onnx.export(
              model.network,
              (example_obs, example_hidden),
              output_path,
              input_names=["obs", "hidden_h", "hidden_c"],
              output_names=["q_values", "new_hidden_h", "new_hidden_c"],
              dynamic_axes={
                  "obs": {0: "batch_size"},
                  "hidden_h": {1: "batch_size"},
                  "hidden_c": {1: "batch_size"},
                  "q_values": {0: "batch_size"},
                  "new_hidden_h": {1: "batch_size"},
                  "new_hidden_c": {1: "batch_size"},
              },
              opset_version=opset_version,
          )
      else:
          # Feedforward: obs only
          obs_dim = _infer_obs_dim(model.network)
          example_obs = torch.randn(batch_size, obs_dim, device=model.device)

          # Export with dynamic batch size
          torch.onnx.export(
              model.network,
              example_obs,
              output_path,
              input_names=["obs"],
              output_names=["q_values"],
              dynamic_axes={
                  "obs": {0: "batch_size"},
                  "q_values": {0: "batch_size"},
              },
              opset_version=opset_version,
          )

      logger.info(f"Exported model to ONNX: {output_path}")

  def verify_onnx_export(onnx_path: Path, model: HamletModel):
      """Verify ONNX export matches PyTorch model."""
      import onnx
      import onnxruntime as ort

      # Load ONNX model
      onnx_model = onnx.load(onnx_path)
      onnx.checker.check_model(onnx_model)

      # Create ONNX runtime session
      session = ort.InferenceSession(onnx_path)

      # Create test input
      obs_dim = _infer_obs_dim(model.network)
      test_obs = torch.randn(1, obs_dim, device="cpu")

      # PyTorch forward
      with torch.no_grad():
          if model.is_recurrent:
              pytorch_output, _ = model.network(test_obs, None)
          else:
              pytorch_output = model.network(test_obs)

      # ONNX forward
      if model.is_recurrent:
          hidden_dim = model.network.hidden_dim
          h = np.zeros((1, 1, hidden_dim), dtype=np.float32)
          c = np.zeros((1, 1, hidden_dim), dtype=np.float32)
          onnx_inputs = {
              "obs": test_obs.cpu().numpy(),
              "hidden_h": h,
              "hidden_c": c,
          }
          onnx_outputs = session.run(None, onnx_inputs)
          onnx_q_values = onnx_outputs[0]
      else:
          onnx_inputs = {"obs": test_obs.cpu().numpy()}
          onnx_outputs = session.run(None, onnx_inputs)
          onnx_q_values = onnx_outputs[0]

      # Compare outputs
      pytorch_q_values = pytorch_output.cpu().numpy()
      max_diff = np.abs(pytorch_q_values - onnx_q_values).max()

      if max_diff < 1e-5:
          logger.info(f"✅ ONNX export verified (max diff: {max_diff:.2e})")
          return True
      else:
          logger.warning(f"⚠️ ONNX export mismatch (max diff: {max_diff:.2e})")
          return False
  ```

- File: `src/townlet/inference/model.py`
  - Add method:
    ```python
    def export_to_onnx(self, output_path: Path, **kwargs):
        """Export model to ONNX format."""
        from townlet.inference.onnx_export import export_to_onnx
        export_to_onnx(self, output_path, **kwargs)
    ```

**Command-line tool**:

```python
# scripts/export_onnx.py
import argparse
from pathlib import Path
from townlet.inference import HamletModel

def main():
    parser = argparse.ArgumentParser(description="Export HAMLET model to ONNX")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path (.pt)")
    parser.add_argument("output", type=Path, help="Output path (.onnx)")
    parser.add_argument("--verify", action="store_true", help="Verify export")
    args = parser.parse_args()

    # Load model
    model = HamletModel.load_from_checkpoint(args.checkpoint)

    # Export
    model.export_to_onnx(args.output)

    # Verify
    if args.verify:
        from townlet.inference.onnx_export import verify_onnx_export
        verify_onnx_export(args.output, model)

if __name__ == "__main__":
    main()
```

**Usage**:

```bash
# Export inference checkpoint to ONNX
python scripts/export_onnx.py \
    runs/L2_partial_observability/2025-11-05_123456/checkpoints/model_ep10000.pt \
    hamlet_model.onnx \
    --verify
```

**Tests**:

- [ ] Test: Export SimpleQNetwork to ONNX
- [ ] Test: Export RecurrentSpatialQNetwork to ONNX
- [ ] Test: Verify ONNX output matches PyTorch
- [ ] Test: ONNX model loads in onnxruntime
- [ ] Test: Dynamic batch size works

**Success Criteria**: ONNX export works for both network types, outputs match PyTorch

---

### Phase 7: Tests & Documentation (6 hours)

**Objective**: Comprehensive testing and user documentation

**Testing**:

- [ ] Unit tests: HamletModel.predict() (100% coverage)
- [ ] Unit tests: Checkpoint save/load (round-trip)
- [ ] Unit tests: ONNX export/verify
- [ ] Integration tests: Load and predict (both checkpoint formats)
- [ ] Integration tests: LiveInferenceServer with HamletModel
- [ ] Regression tests: Existing training tests pass

**Documentation**:

- File: `docs/manual/INFERENCE_USAGE.md` (NEW)
  ```markdown
  # Inference Usage Guide

  ## Load Trained Model

  ```python
  from townlet.inference import HamletModel

  # Load inference checkpoint (v1, faster)
  model = HamletModel.load_from_checkpoint("runs/.../model_ep10000.pt")

  # Or load training checkpoint (v2, backward compat)
  model = HamletModel.load_from_training_checkpoint("runs/.../checkpoint_ep10000.pt")
  ```

  ## Run Inference

  ```python
  # Greedy policy
  result = model.predict(obs, epsilon=0.0)
  action = result["actions"][0]

  # Epsilon-greedy (exploration)
  result = model.predict(obs, epsilon=0.1)
  ```

  ## Recurrent Models (LSTM)

  ```python
  # Reset hidden state at episode start
  hidden = None

  for step in range(500):
      result = model.predict(obs, hidden, epsilon=0.0)
      action = result["actions"][0]
      hidden = result["hidden"]  # Carry forward

      obs, reward, done, info = env.step(action)
      if done:
          hidden = None  # Reset for next episode
  ```

  ## Export to ONNX

  ```bash
  python scripts/export_onnx.py model_ep10000.pt hamlet.onnx --verify
  ```
  ```

- File: `docs/manual/MODEL_EXPORT.md` (NEW)
  - Example: Use model in custom environment
  - Example: Transfer learning
  - Example: Ensemble methods

- Update: `CLAUDE.md`
  - Add inference architecture section
  - Document HamletModel API
  - Note ONNX export capabilities

**Success Criteria**: Documentation complete, all tests pass

---

## Testing Strategy

### Test Coverage Requirements

**Unit Tests**:

- HamletModel: 100% coverage (predict, load, save)
- Checkpoint utilities: 100% coverage
- ONNX export: 100% coverage

**Integration Tests**:

- [ ] Load SimpleQNetwork from v1 checkpoint, predict
- [ ] Load RecurrentSpatialQNetwork from v1 checkpoint, predict with hidden state
- [ ] Load from v2 training checkpoint (backward compat)
- [ ] LiveInferenceServer with HamletModel
- [ ] Round-trip: Save inference checkpoint, load, predict (match original)
- [ ] ONNX export: PyTorch and ONNX outputs match

**Property-Based Tests**:

- [ ] Property: Q-values deterministic for same obs (epsilon=0.0)
- [ ] Property: Hidden state evolution consistent
- [ ] Property: ONNX and PyTorch outputs within epsilon

### Regression Testing

**Critical Paths**:

- [ ] Existing training runs still work (VectorizedPopulation unchanged)
- [ ] Old checkpoints load correctly (backward compatibility)
- [ ] LiveInferenceServer works with both checkpoint formats

**Performance Testing**:

- [ ] Inference checkpoint loading ~5x faster than training checkpoint
- [ ] Model.predict() latency <10ms per batch (GPU)
- [ ] ONNX inference latency comparable to PyTorch

---

## Migration Guide

### For Existing Code

**Before** (load from training checkpoint):

```python
# Old way: Recreate training stack
from townlet.population.vectorized import VectorizedPopulation

env = VectorizedHamletEnv(...)
curriculum = AdversarialCurriculum(...)
exploration = AdaptiveIntrinsicExploration(...)
population = VectorizedPopulation(env, curriculum, exploration, ...)

checkpoint = torch.load("checkpoint_ep10000.pt")
population.q_network.load_state_dict(checkpoint["population_state"]["q_network"])

q_values = population.q_network(obs)
```

**After** (use HamletModel):

```python
# New way: One-line loading
from townlet.inference import HamletModel

model = HamletModel.load_from_training_checkpoint("checkpoint_ep10000.pt")

result = model.predict(obs, epsilon=0.0)
q_values = result["q_values"]
```

**Migration Script**:

```bash
# Convert existing checkpoints to inference format
python scripts/migrate_checkpoints.py runs/L2_partial_observability/2025-11-05_123456/checkpoints/
```

---

## Examples

### Example 1: Load and Use Model

```python
from townlet.inference import HamletModel
import torch

# Load trained model
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Create observation
obs = torch.randn(1, 54)  # POMDP observation

# Greedy inference
result = model.predict(obs, epsilon=0.0)
print(f"Action: {result['actions'][0]}")
print(f"Q-values: {result['q_values'][0]}")
```

### Example 2: Transfer Learning

```python
from townlet.inference import HamletModel
import torch

# Load pretrained model
base_model = HamletModel.load_from_checkpoint("pretrained_model.pt")

# Freeze early layers
for param in base_model.network.vision_encoder.parameters():
    param.requires_grad = False

# Fine-tune on new task
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, base_model.network.parameters()),
    lr=0.001
)

# Training loop...
```

### Example 3: Export to ONNX

```bash
# Export to ONNX
python scripts/export_onnx.py model_ep10000.pt hamlet.onnx --verify

# Use in ONNX runtime (Python)
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("hamlet.onnx")
obs = np.random.randn(1, 54).astype(np.float32)

# For feedforward
outputs = session.run(None, {"obs": obs})
q_values = outputs[0]

# For recurrent
h = np.zeros((1, 1, 256), dtype=np.float32)
c = np.zeros((1, 1, 256), dtype=np.float32)
outputs = session.run(None, {"obs": obs, "hidden_h": h, "hidden_c": c})
q_values, new_h, new_c = outputs
```

---

## Acceptance Criteria

### Must Have (Blocking)

- [ ] HamletModel class implemented and tested
- [ ] Can load from v1 inference checkpoints
- [ ] Can load from v2 training checkpoints (backward compat)
- [ ] Inference checkpoints 5-10x smaller (~10MB vs ~50MB)
- [ ] predict() is pure function (no side effects)
- [ ] ONNX export works for both network types
- [ ] ONNX outputs match PyTorch (within 1e-5)
- [ ] LiveInferenceServer refactored to use HamletModel
- [ ] All tests pass (unit + integration)
- [ ] Documentation complete (INFERENCE_USAGE.md, MODEL_EXPORT.md)

### Should Have (Important)

- [ ] Migration script (v2 → v1 checkpoints)
- [ ] Command-line ONNX export tool
- [ ] ONNX verification tool
- [ ] Examples: transfer learning, ensemble

### Could Have (Future)

- [ ] Batch inference utilities (separate task)
- [ ] Model comparison tools (separate task)
- [ ] Curriculum evaluation tools (separate task)
- [ ] TorchScript export (in addition to ONNX)

---

## Risk Assessment

### Technical Risks

**Risk 1: BRAIN_AS_CODE integration incomplete**

- **Severity**: LOW (was HIGH, now MITIGATED)
- **Mitigation**: ✅ Implemented fallback checkpoint format that doesn't require BRAIN_AS_CODE
- **Details**: Checkpoint format supports both brain_spec (preferred) and network_params (fallback)
- **Fallback**: System automatically uses minimal network_params if brain.yaml unavailable

**Risk 2: ONNX export issues with LSTM**

- **Severity**: MEDIUM
- **Mitigation**: LSTM is standard PyTorch op, well-supported in ONNX
- **Contingency**: Document ONNX limitations, support feedforward export only initially

**Risk 3: Hidden state API complexity**

- **Severity**: MEDIUM
- **Mitigation**: Provide StatefulModel wrapper if needed
- **Contingency**: Document best practices, provide examples

**Risk 4: Backward compatibility with old checkpoints**

- **Severity**: MEDIUM
- **Mitigation**: Extensive testing of v2 checkpoint loading
- **Contingency**: Provide migration tool, document manual migration

### Blocking Dependencies

- ✅ **NONE**: Originally depended on TASK-004 (BRAIN_AS_CODE), now mitigated with fallback format
  - If TASK-004 complete: Uses brain_spec (preferred, self-describing checkpoints)
  - If TASK-004 incomplete: Uses network_params (fallback, minimal but functional)
  - Both formats coexist, system auto-detects which to use

### Impact Radius

**Files Modified**: 4
- `src/townlet/demo/runner.py` (add inference checkpoint save)
- `src/townlet/demo/live_inference.py` (refactor to use HamletModel)
- `CLAUDE.md` (document inference API)
- `docs/manual/UNIFIED_SERVER_USAGE.md` (update)

**Files Created**: 4
- `src/townlet/inference/__init__.py`
- `src/townlet/inference/model.py`
- `src/townlet/inference/checkpoint.py`
- `src/townlet/inference/onnx_export.py`
- `docs/manual/INFERENCE_USAGE.md`
- `docs/manual/MODEL_EXPORT.md`
- `scripts/export_onnx.py`
- `scripts/migrate_checkpoints.py`

**Tests Added**: ~20 tests

**Breaking Changes**: None (fully backward compatible)

**Blast Radius**: Medium (new subsystem, refactored LiveInferenceServer, but training unchanged)

---

## Effort Breakdown

### Detailed Estimates

**Phase 1**: 8 hours
- Create inference/ directory: 0.5h
- HamletModel class skeleton: 1h
- predict() implementation: 2h
- Epsilon-greedy action selection: 1h
- Tests: 3.5h

**Phase 2**: 4 hours
- Define checkpoint format: 1h
- save_inference_checkpoint(): 1h
- load_inference_checkpoint(): 1h
- Tests: 1h

**Phase 3**: 6 hours
- load_from_checkpoint() (v1): 2h
- load_from_training_checkpoint() (v2): 3h
- Tests: 1h

**Phase 4**: 2 hours
- Modify DemoRunner.save_checkpoint(): 1h
- Tests: 1h

**Phase 5**: 6 hours
- Refactor LiveInferenceServer: 4h
- Tests: 2h

**Phase 6**: 4 hours
- ONNX export implementation: 2h
- ONNX verification: 1h
- Tests + CLI tool: 1h

**Phase 7**: 6 hours
- Documentation: 3h
- Migration script: 1h
- Integration tests: 2h

**Total**: 36 hours

**Confidence**: MEDIUM (BRAIN_AS_CODE integration is unknown, ONNX export might have edge cases)

### Assumptions

- TASK-004 (BRAIN_AS_CODE) complete before this task
- compile_brain() API available
- ONNX opset 17+ supported
- Standard PyTorch layers export cleanly

---

## Future Work (Explicitly Out of Scope)

### Not Included in This Task

1. **Batch Inference Tools**
   - **Why Deferred**: Core model abstraction is sufficient for MVP
   - **Follow-up Task**: Separate evaluation task (6-8h)

2. **Model Comparison Tools**
   - **Why Deferred**: Students can implement themselves
   - **Follow-up Task**: Separate evaluation task (4-6h)

3. **Model Serving API**
   - **Why Deferred**: Not pedagogical priority
   - **Follow-up Task**: If production use case emerges (40-60h)

4. **TorchScript Export**
   - **Why Deferred**: ONNX covers most use cases
   - **Follow-up Task**: If requested by students (2-3h)

### Enables Future Tasks

- **Evaluation tools**: Batch inference, model comparison
- **Research**: Transfer learning, ensemble methods, ablation studies
- **Production**: Model serving API (if needed)

---

## References

### Related Documentation

- **Research**: `docs/research/RESEARCH-INFERENCE-ARCHITECTURE.md`
- **Research**: `docs/research/RESEARCH-EXTENDED-USE-CASES.md`
- **Synthesis**: `docs/research/RESEARCH-INFERENCE-ARCHITECTURE-SYNTHESIS.md`
- **Architecture**: `docs/architecture/BRAIN_AS_CODE.md`

### Related Tasks

- **Prerequisite**: TASK-004 (BRAIN_AS_CODE) for checkpoint format
- **Parallel Work**: TASK-007 (Live Viz) - can be done in same PR
- **Enables**: Future evaluation and research tasks

### Code References

- `src/townlet/agent/networks.py:11` - SimpleQNetwork
- `src/townlet/agent/networks.py:38` - RecurrentSpatialQNetwork
- `src/townlet/population/vectorized.py:107` - Q-network creation
- `src/townlet/demo/live_inference.py:230` - Current model loading (to be refactored)
- `src/townlet/demo/runner.py:400` - Checkpoint saving (to be extended)

---

## Notes for Implementer

### Before Starting

- [ ] Read `docs/research/RESEARCH-INFERENCE-ARCHITECTURE.md` for design rationale
- [ ] Coordinate with TASK-004 (BRAIN_AS_CODE) for checkpoint format
- [ ] Review both network types (SimpleQNetwork, RecurrentSpatialQNetwork)
- [ ] Can be implemented in same PR as TASK-007 (both touch LiveInferenceServer)

### During Implementation

- [ ] Keep predict() as pure function (no side effects)
- [ ] Test ONNX export early (Phase 6) to catch issues
- [ ] Use functional API for hidden state (don't store in network)
- [ ] Maintain backward compatibility (load v2 checkpoints)

### Before Marking Complete

- [ ] All acceptance criteria met
- [ ] ONNX export verified (outputs match PyTorch)
- [ ] Backward compatibility tested (load old checkpoints)
- [ ] Documentation complete and tested
- [ ] LiveInferenceServer simplified (no VectorizedPopulation)
- [ ] Both checkpoint formats save correctly

---

**END OF TASK SPECIFICATION**

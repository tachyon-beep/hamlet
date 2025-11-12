# DAC Phase 3: Runtime Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement complete runtime reward computation engine with 9 extrinsic strategies, 11 shaping bonuses, and VectorizedPopulation integration.

**Architecture:** DACEngine compiles declarative DAC specs into GPU-native computation graphs. Each strategy/bonus is a closure compiled at initialization. Modifiers apply context-sensitive multipliers. Composition formula: `total_reward = extrinsic + (intrinsic * effective_weight) + shaping`.

**Tech Stack:** PyTorch (GPU tensors), Pydantic (DTOs), VFS registry (variable access), TDD (test-driven development)

**Status:**
- ✅ Phase 1 (DTO Layer): COMPLETE - 23 tests passing
- ✅ Phase 2 (Compiler Integration): COMPLETE - 8 tests passing
- ✅ Phase 3A (Infrastructure): COMPLETE - DACEngine skeleton + modifiers (5 tests passing)
- ⏳ Phase 3B (Extrinsic Strategies): THIS PLAN
- ⏳ Phase 3C (Shaping Bonuses): THIS PLAN
- ⏳ Phase 3D (Composition & Integration): THIS PLAN

---

## Phase 3B: Extrinsic Reward Strategies (6-8 hours)

Each strategy is a discrete 30-45 minute task with full TDD cycle.

### Task 3B.1: Multiplicative Strategy

**Files:**
- Modify: `src/townlet/environment/dac_engine.py` (replace `_compile_extrinsic` placeholder)
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

Add to test file:

```python
class TestExtrinsicStrategies:
    """Test extrinsic reward strategy compilation."""

    def test_multiplicative_strategy(self):
        """Multiplicative strategy: reward = base * bar1 * bar2 * ..."""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=2.0,
                bars=["energy"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Create meter values: [0.5, 0.8, 1.0, 0.0]
        meters = torch.tensor([[0.5], [0.8], [1.0], [0.0]], device=device)
        dones = torch.tensor([False, False, False, True], device=device)

        # Calculate extrinsic rewards
        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: base * energy (2.0 * [0.5, 0.8, 1.0, 0.0])
        # Dead agents get 0.0
        expected = torch.tensor([1.0, 1.6, 2.0, 0.0], device=device)
        assert torch.allclose(extrinsic, expected)

    def test_multiplicative_multiple_bars(self):
        """Multiplicative with multiple bars: reward = base * energy * health"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # meters: [energy, health]
        meters = torch.tensor(
            [[0.5, 0.8], [1.0, 1.0], [0.2, 0.5], [0.0, 1.0]], device=device
        )
        dones = torch.tensor([False, False, False, False], device=device)

        extrinsic = engine.extrinsic_fn(meters, dones)

        # Expected: 1.0 * energy * health
        expected = torch.tensor([0.4, 1.0, 0.1, 0.0], device=device)
        assert torch.allclose(extrinsic, expected, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestExtrinsicStrategies::test_multiplicative_strategy -v`

Expected: FAIL (returns zeros from placeholder)

**Step 3: Write minimal implementation**

Modify `src/townlet/environment/dac_engine.py` - replace `_compile_extrinsic()`:

```python
def _compile_extrinsic(self) -> Callable:
    """Compile extrinsic strategy into computation function.

    Returns:
        Function that computes extrinsic rewards [num_agents]
    """
    strategy = self.dac_config.extrinsic

    if strategy.type == "multiplicative":
        # reward = base * bar1 * bar2 * ...
        base = strategy.base if strategy.base is not None else 1.0
        bar_ids = strategy.bars

        def compute_multiplicative(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
            """Multiplicative: reward = base * product(bars)"""
            # Start with base
            reward = torch.full((self.num_agents,), base, device=self.device, dtype=torch.float32)

            # Multiply by each bar
            for bar_id in bar_ids:
                bar_idx = self._get_bar_index(bar_id)
                reward = reward * meters[:, bar_idx]

            # Dead agents get 0.0
            reward = torch.where(dones, torch.zeros_like(reward), reward)

            return reward

        return compute_multiplicative

    # Fallback for unimplemented strategies
    def placeholder(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.num_agents, device=self.device)

    return placeholder
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestExtrinsicStrategies -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): implement multiplicative extrinsic strategy

- Add TestExtrinsicStrategies test class
- Implement multiplicative strategy: reward = base * bar1 * bar2 * ...
- Support single and multiple bar multiplication
- Dead agents receive 0.0 reward
- Add 2 comprehensive tests

Part of TASK-004C Phase 3B (1/9 strategies)"
```

---

### Task 3B.2: Constant Base with Shaped Bonus Strategy

**Files:**
- Modify: `src/townlet/environment/dac_engine.py`
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

```python
def test_constant_base_with_shaped_bonus(self):
    """Constant base + shaped bonus: reward = base + sum(bonuses)"""
    device = torch.device("cpu")
    num_agents = 4

    vfs_registry = VariableRegistry(
        variables=[
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                default=1.0,
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                lifetime="episode",
            )
        ],
        num_agents=num_agents,
        device=device,
    )

    dac_config = DriveAsCodeConfig(
        version="1.0",
        modifiers={},
        extrinsic=ExtrinsicStrategyConfig(
            type="constant_base_with_shaped_bonus",
            base_reward=1.0,
            bar_bonuses=[
                BarBonusConfig(bar="energy", center=0.5, scale=0.5),
            ],
        ),
        intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
    )

    engine = DACEngine(dac_config, vfs_registry, device, num_agents)

    # Energy values: [0.0, 0.5, 1.0, 0.25]
    meters = torch.tensor([[0.0], [0.5], [1.0], [0.25]], device=device)
    dones = torch.tensor([False, False, False, False], device=device)

    extrinsic = engine.extrinsic_fn(meters, dones)

    # Expected: base_reward + scale * (energy - center)
    # [1.0 + 0.5 * (0.0 - 0.5), 1.0 + 0.5 * (0.5 - 0.5), 1.0 + 0.5 * (1.0 - 0.5), 1.0 + 0.5 * (0.25 - 0.5)]
    # = [0.75, 1.0, 1.25, 0.875]
    expected = torch.tensor([0.75, 1.0, 1.25, 0.875], device=device)
    assert torch.allclose(extrinsic, expected)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestExtrinsicStrategies::test_constant_base_with_shaped_bonus -v`

Expected: FAIL (strategy not implemented)

**Step 3: Write minimal implementation**

Add to `_compile_extrinsic()` after multiplicative:

```python
elif strategy.type == "constant_base_with_shaped_bonus":
    # reward = base + sum(bar_bonuses) + sum(variable_bonuses)
    base_reward = strategy.base_reward if strategy.base_reward is not None else 1.0
    bar_bonuses = strategy.bar_bonuses
    variable_bonuses = strategy.variable_bonuses

    def compute_constant_base(meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Constant base + shaped bonuses"""
        # Start with base reward
        reward = torch.full((self.num_agents,), base_reward, device=self.device, dtype=torch.float32)

        # Add bar bonuses
        for bonus_config in bar_bonuses:
            bar_idx = self._get_bar_index(bonus_config.bar)
            bar_value = meters[:, bar_idx]
            bonus = bonus_config.scale * (bar_value - bonus_config.center)
            reward = reward + bonus

        # Add variable bonuses (from VFS)
        for bonus_config in variable_bonuses:
            var_value = self.vfs_registry.get(bonus_config.variable, reader=self.vfs_reader)
            bonus = bonus_config.weight * var_value
            reward = reward + bonus

        # Dead agents get 0.0
        reward = torch.where(dones, torch.zeros_like(reward), reward)

        return reward

    return compute_constant_base
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestExtrinsicStrategies -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): implement constant_base_with_shaped_bonus strategy

- Implement reward = base + sum(bar_bonuses) + sum(variable_bonuses)
- Support bar-based bonuses (deviation from center)
- Support VFS variable bonuses
- Add test for L0.5 bug fix use case
- Dead agents receive 0.0 reward

Part of TASK-004C Phase 3B (2/9 strategies)"
```

---

### Task 3B.3-3B.9: Remaining Extrinsic Strategies

**Note:** Tasks 3B.3 through 3B.9 follow the same TDD pattern. Each implements one strategy type:

- **3B.3: additive_unweighted** - `reward = base + sum(bars)`
- **3B.4: weighted_sum** - `reward = sum(weight_i * source_i)` with sources
- **3B.5: polynomial** - `reward = sum(weight_i * source_i^exponent_i)`
- **3B.6: threshold_based** - Bonuses when bars cross thresholds
- **3B.7: aggregation** - `reward = base + op(bars)` where op ∈ {min, max, mean, product}
- **3B.8: vfs_variable** - `reward = vfs[variable] * weight` (escape hatch)
- **3B.9: hybrid** - `reward = sum(weight_i * strategy_i)` (composable)

Each follows the same 5-step pattern:
1. Write failing test
2. Run test (verify fail)
3. Add `elif strategy.type == "..."` case to `_compile_extrinsic()`
4. Run test (verify pass)
5. Commit

**For brevity, detailed implementations are omitted but follow the established pattern.** Each strategy is 30-45 minutes including tests and commit.

**Combined Commit for 3B.3-3B.9** (if implementing in batch):

```bash
git commit -m "feat(dac): implement remaining 7 extrinsic strategies

- additive_unweighted: reward = base + sum(bars)
- weighted_sum: reward = sum(weight_i * source_i)
- polynomial: reward = sum(weight_i * source_i^exponent_i)
- threshold_based: bonuses when crossing thresholds
- aggregation: reward = base + {min,max,mean,product}(bars)
- vfs_variable: reward = vfs[variable] * weight
- hybrid: reward = sum(weighted sub-strategies)

Add 14+ comprehensive tests covering all strategies.
All strategies support dead agent handling (return 0.0).

Part of TASK-004C Phase 3B (3-9/9 strategies COMPLETE)"
```

---

## Phase 3C: Shaping Bonuses (4-6 hours)

Shaping bonuses add behavioral incentives on top of extrinsic rewards. Each is a discrete 20-30 minute task.

### Task 3C.1: Approach Reward Bonus

**Files:**
- Modify: `src/townlet/environment/dac_engine.py` (replace `_compile_shaping` placeholder)
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

Add new test class:

```python
class TestShapingBonuses:
    """Test shaping bonus compilation."""

    def test_approach_reward_bonus(self):
        """Approach reward: bonus for moving toward target affordance"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=1.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.0),
            shaping=[
                ApproachRewardConfig(
                    type="approach_reward",
                    target_affordance="Bed",
                    trigger=TriggerCondition(condition="always"),
                    bonus=0.1,
                    decay_with_distance=True,
                ),
            ],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Verify shaping function compiled
        assert len(engine.shaping_fns) == 1

        # Test requires positions/distances (kwargs)
        # For now, just verify compilation
        assert callable(engine.shaping_fns[0])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestShapingBonuses::test_approach_reward_bonus -v`

Expected: FAIL (shaping_fns is empty list)

**Step 3: Write minimal implementation**

Replace `_compile_shaping()` in `src/townlet/environment/dac_engine.py`:

```python
def _compile_shaping(self) -> list[Callable]:
    """Compile shaping bonuses into computation functions.

    Returns:
        List of shaping bonus functions
    """
    compiled = []

    for shaping_config in self.dac_config.shaping:
        if shaping_config.type == "approach_reward":
            # Closure captures config
            cfg = shaping_config

            def compute_approach_reward(**kwargs) -> torch.Tensor:
                """Approach reward: bonus for proximity to target affordance"""
                # Extract positions and affordance positions from kwargs
                agent_positions = kwargs.get("agent_positions")  # [num_agents, ndim]
                affordance_positions = kwargs.get("affordance_positions")  # dict[str, Tensor]

                if agent_positions is None or affordance_positions is None:
                    # No position data, return zeros
                    return torch.zeros(self.num_agents, device=self.device)

                target_pos = affordance_positions.get(cfg.target_affordance)
                if target_pos is None:
                    return torch.zeros(self.num_agents, device=self.device)

                # Compute distances
                distances = torch.norm(agent_positions - target_pos, dim=1)

                # Apply bonus with optional distance decay
                if cfg.decay_with_distance:
                    # Inverse distance decay: bonus / (1 + distance)
                    bonus = cfg.bonus / (1.0 + distances)
                else:
                    # Constant bonus regardless of distance
                    bonus = torch.full_like(distances, cfg.bonus)

                return bonus

            compiled.append(compute_approach_reward)

        elif shaping_config.type == "completion_bonus":
            # TODO: Implement in next task
            pass

        elif shaping_config.type == "vfs_variable":
            # Closure captures config
            cfg = shaping_config

            def compute_vfs_shaping(**kwargs) -> torch.Tensor:
                """VFS variable shaping (escape hatch)"""
                var_value = self.vfs_registry.get(cfg.variable, reader=self.vfs_reader)
                return cfg.weight * var_value

            compiled.append(compute_vfs_shaping)

    return compiled
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestShapingBonuses -v`

Expected: 1 passed

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): implement approach_reward and vfs_variable shaping bonuses

- Implement _compile_shaping() skeleton
- Add approach_reward: bonus for proximity to target affordance
- Support distance decay (inverse distance)
- Add vfs_variable shaping (escape hatch for custom logic)
- Add test for approach reward compilation

Part of TASK-004C Phase 3C (1/11 bonuses)"
```

---

### Task 3C.2-3C.11: Remaining Shaping Bonuses

**Note:** Tasks 3C.2 through 3C.11 follow the same TDD pattern. Each implements one bonus type:

- **3C.2: completion_bonus** - Reward for finishing affordance interactions
- **3C.3: efficiency_bonus** - Reward for achieving goals with minimal resource use
- **3C.4: state_achievement** - Bonus for reaching target bar/variable values
- **3C.5: streak_bonus** - Bonus for consecutive similar actions
- **3C.6: diversity_bonus** - Bonus for exploring different affordances
- **3C.7: timing_bonus** - Temporal rewards (time-of-day dependent)
- **3C.8: economic_efficiency** - Reward for money management
- **3C.9: balance_bonus** - Reward for keeping bars balanced
- **3C.10: crisis_avoidance** - Penalty for letting bars drop too low

Each follows the 5-step TDD pattern. **For brevity, detailed implementations are omitted.**

**Combined Commit for 3C.2-3C.10** (if implementing in batch):

```bash
git commit -m "feat(dac): implement remaining 9 shaping bonuses

- completion_bonus: reward finishing interactions
- efficiency_bonus: reward minimal resource use
- state_achievement: bonus for target values
- streak_bonus: reward consecutive actions
- diversity_bonus: reward exploration variety
- timing_bonus: temporal incentives
- economic_efficiency: money management
- balance_bonus: reward bar balance
- crisis_avoidance: penalty for low bars

Add 18+ comprehensive tests covering all bonuses.
All bonuses return [num_agents] shaped rewards.

Part of TASK-004C Phase 3C (2-11/11 bonuses COMPLETE)"
```

---

## Phase 3D: Composition & Integration (2-3 hours)

### Task 3D.1: Complete calculate_rewards() Implementation

**Files:**
- Modify: `src/townlet/environment/dac_engine.py` (replace `calculate_rewards` placeholder)
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

```python
class TestRewardComposition:
    """Test complete reward calculation with composition."""

    def test_calculate_rewards_full_pipeline(self):
        """Full pipeline: extrinsic + (intrinsic * modifiers) + shaping"""
        device = torch.device("cpu")
        num_agents = 4

        vfs_registry = VariableRegistry(
            variables=[
                VariableDef(
                    id="energy",
                    scope="agent",
                    type="scalar",
                    default=1.0,
                    readable_by=["agent", "engine"],
                    writable_by=["engine"],
                    lifetime="episode",
                )
            ],
            num_agents=num_agents,
            device=device,
        )

        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "energy_crisis": ModifierConfig(
                    bar="energy",
                    ranges=[
                        RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                        RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=2.0, bars=["energy"]),
            intrinsic=IntrinsicStrategyConfig(
                strategy="none",
                base_weight=0.5,
                apply_modifiers=["energy_crisis"],
            ),
            shaping=[],
        )

        engine = DACEngine(dac_config, vfs_registry, device, num_agents)

        # Meters: energy values [0.2, 0.5, 0.8, 1.0]
        meters = torch.tensor([[0.2], [0.5], [0.8], [1.0]], device=device)
        dones = torch.tensor([False, False, False, False], device=device)
        step_counts = torch.tensor([10, 10, 10, 10], device=device)
        intrinsic_raw = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

        total_rewards, intrinsic_weights, components = engine.calculate_rewards(
            step_counts=step_counts,
            dones=dones,
            meters=meters,
            intrinsic_raw=intrinsic_raw,
        )

        # Expected extrinsic: 2.0 * energy = [0.4, 1.0, 1.6, 2.0]
        assert torch.allclose(components["extrinsic"], torch.tensor([0.4, 1.0, 1.6, 2.0], device=device))

        # Expected intrinsic weights: base_weight * energy_crisis modifier
        # energy < 0.3 → 0.0, energy >= 0.3 → 1.0
        # effective_weight = 0.5 * [0.0, 1.0, 1.0, 1.0] = [0.0, 0.5, 0.5, 0.5]
        expected_weights = torch.tensor([0.0, 0.5, 0.5, 0.5], device=device)
        assert torch.allclose(intrinsic_weights, expected_weights)

        # Expected intrinsic: intrinsic_raw * effective_weight = 1.0 * [0.0, 0.5, 0.5, 0.5]
        assert torch.allclose(components["intrinsic"], torch.tensor([0.0, 0.5, 0.5, 0.5], device=device))

        # Total: extrinsic + intrinsic + shaping (shaping=0)
        expected_total = torch.tensor([0.4, 1.5, 2.1, 2.5], device=device)
        assert torch.allclose(total_rewards, expected_total)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestRewardComposition::test_calculate_rewards_full_pipeline -v`

Expected: FAIL (returns zeros from placeholder)

**Step 3: Write minimal implementation**

Replace `calculate_rewards()` in `src/townlet/environment/dac_engine.py`:

```python
def calculate_rewards(
    self,
    step_counts: torch.Tensor,
    dones: torch.Tensor,
    meters: torch.Tensor,
    intrinsic_raw: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate total rewards with DAC composition.

    Formula:
        total_reward = extrinsic + (intrinsic * effective_intrinsic_weight) + shaping

    Where:
        effective_intrinsic_weight = base_weight * modifier1 * modifier2 * ...

    Args:
        step_counts: [num_agents] current step count
        dones: [num_agents] agent death flags
        meters: [num_agents, meter_count] normalized meter values
        intrinsic_raw: [num_agents] raw intrinsic curiosity values
        **kwargs: Additional context (positions, affordance states, etc.)

    Returns:
        total_rewards: [num_agents] final rewards
        intrinsic_weights: [num_agents] effective intrinsic weights
        components: dict of reward components (extrinsic, intrinsic, shaping)
    """
    # 1. Compute extrinsic rewards
    extrinsic = self.extrinsic_fn(meters, dones)

    # 2. Compute effective intrinsic weight with modifiers
    base_weight = self.dac_config.intrinsic.base_weight
    effective_weight = torch.full((self.num_agents,), base_weight, device=self.device, dtype=torch.float32)

    # Apply modifiers to intrinsic weight
    for mod_name in self.dac_config.intrinsic.apply_modifiers:
        if mod_name in self.modifiers:
            modifier_fn = self.modifiers[mod_name]
            multiplier = modifier_fn(meters)
            effective_weight = effective_weight * multiplier

    # 3. Compute intrinsic rewards
    intrinsic = intrinsic_raw * effective_weight

    # 4. Compute shaping bonuses
    shaping_total = torch.zeros(self.num_agents, device=self.device)
    for shaping_fn in self.shaping_fns:
        shaping_bonus = shaping_fn(**kwargs)
        shaping_total = shaping_total + shaping_bonus

    # 5. Apply composition rules
    composition = self.dac_config.composition

    # Compose total reward
    total_rewards = extrinsic + intrinsic + shaping_total

    # Apply clipping if configured
    if composition.clip_min is not None or composition.clip_max is not None:
        clip_min = composition.clip_min if composition.clip_min is not None else -float('inf')
        clip_max = composition.clip_max if composition.clip_max is not None else float('inf')
        total_rewards = torch.clamp(total_rewards, min=clip_min, max=clip_max)

    # 6. Package components for logging
    components = {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "shaping": shaping_total,
    }

    return total_rewards, effective_weight, components
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestRewardComposition -v`

Expected: 1 passed

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): implement complete reward composition pipeline

- Implement calculate_rewards() with full composition formula
- Compute extrinsic rewards via compiled strategy
- Apply modifiers to intrinsic weight (crisis suppression)
- Compute intrinsic rewards with effective weight
- Sum shaping bonuses
- Support composition rules (clipping, normalization)
- Return components for logging
- Add comprehensive integration test

Part of TASK-004C Phase 3D (1/3 tasks)

Formula: total = extrinsic + (intrinsic * Π(modifiers)) + Σ(shaping)"
```

---

### Task 3D.2: Add Component Logging Support

**Files:**
- Modify: `src/townlet/environment/dac_engine.py`
- Test: `tests/test_townlet/unit/environment/test_dac_engine.py`

**Step 1: Write the failing test**

```python
def test_component_logging_enabled(self):
    """Component logging includes detailed breakdown"""
    device = torch.device("cpu")
    num_agents = 4

    vfs_registry = VariableRegistry(
        variables=[
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                default=1.0,
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                lifetime="episode",
            )
        ],
        num_agents=num_agents,
        device=device,
    )

    # Enable component logging
    dac_config = DriveAsCodeConfig(
        version="1.0",
        modifiers={},
        extrinsic=ExtrinsicStrategyConfig(type="multiplicative", base=2.0, bars=["energy"]),
        intrinsic=IntrinsicStrategyConfig(strategy="none", base_weight=0.5),
        composition=CompositionConfig(log_components=True, log_modifiers=False),
    )

    engine = DACEngine(dac_config, vfs_registry, device, num_agents)

    meters = torch.tensor([[0.5], [0.8], [1.0], [0.2]], device=device)
    dones = torch.tensor([False, False, False, False], device=device)
    step_counts = torch.tensor([10, 10, 10, 10], device=device)
    intrinsic_raw = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

    _, _, components = engine.calculate_rewards(
        step_counts=step_counts,
        dones=dones,
        meters=meters,
        intrinsic_raw=intrinsic_raw,
    )

    # Should have detailed components
    assert "extrinsic" in components
    assert "intrinsic" in components
    assert "shaping" in components
    assert "total" in components
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestRewardComposition::test_component_logging_enabled -v`

Expected: FAIL ('total' key not in components)

**Step 3: Write minimal implementation**

Modify `calculate_rewards()` to add logging support:

```python
# At end of calculate_rewards(), before return:

# 6. Package components for logging
components = {
    "extrinsic": extrinsic,
    "intrinsic": intrinsic,
    "shaping": shaping_total,
}

# Add detailed logging if enabled
if self.log_components:
    components["total"] = total_rewards
    components["intrinsic_weight"] = effective_weight

if self.log_modifiers:
    # Log individual modifier values
    for mod_name in self.dac_config.intrinsic.apply_modifiers:
        if mod_name in self.modifiers:
            modifier_fn = self.modifiers[mod_name]
            components[f"modifier_{mod_name}"] = modifier_fn(meters)

return total_rewards, effective_weight, components
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/unit/environment/test_dac_engine.py::TestRewardComposition -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add src/townlet/environment/dac_engine.py tests/test_townlet/unit/environment/test_dac_engine.py
git commit -m "feat(dac): add component logging support

- Add 'total' and 'intrinsic_weight' to components when log_components=True
- Add individual modifier values when log_modifiers=True
- Support detailed reward debugging and analysis
- Add test for component logging

Part of TASK-004C Phase 3D (2/3 tasks)"
```

---

### Task 3D.3: Integrate with VectorizedPopulation

**Files:**
- Modify: `src/townlet/population/vectorized.py`
- Test: `tests/test_townlet/integration/test_dac_integration.py` (new)

**Step 1: Write the failing integration test**

Create new integration test file:

```python
"""Integration tests for DAC with VectorizedPopulation."""

import pytest
import torch
from pathlib import Path

from townlet.universe.compiler import UniverseCompiler


class TestDACVectorizedPopulationIntegration:
    """Test DAC integration with training pipeline."""

    def test_dac_integration_with_minimal_config(self, tmp_path):
        """DAC engine integrated with VectorizedPopulation"""
        # Copy L0_0_minimal and add drive_as_code.yaml
        import shutil
        source = Path("configs/L0_0_minimal")
        dest = tmp_path / "test_config"
        shutil.copytree(source, dest)

        # Create minimal DAC config
        import yaml
        dac_config = {
            "drive_as_code": {
                "version": "1.0",
                "modifiers": {},
                "extrinsic": {
                    "type": "multiplicative",
                    "base": 1.0,
                    "bars": ["energy"],
                },
                "intrinsic": {
                    "strategy": "none",
                    "base_weight": 0.0,
                },
            }
        }
        (dest / "drive_as_code.yaml").write_text(yaml.dump(dac_config))

        # Compile universe
        compiler = UniverseCompiler()
        compiled = compiler.compile(dest, use_cache=False)

        # Verify DAC config loaded
        assert compiled.dac_config is not None
        assert compiled.drive_hash is not None

        # Create environment (will use DAC for rewards)
        env = compiled.create_environment(num_agents=4, device="cpu")

        # Run one step
        actions = torch.zeros(4, dtype=torch.long)
        obs, rewards, dones, truncated, info = env.step(actions)

        # Verify rewards computed via DAC
        assert rewards.shape == (4,)
        assert torch.all(torch.isfinite(rewards))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_townlet/integration/test_dac_integration.py::TestDACVectorizedPopulationIntegration::test_dac_integration_with_minimal_config -v`

Expected: FAIL (VectorizedPopulation doesn't use DAC yet)

**Step 3: Write minimal implementation**

Modify `src/townlet/population/vectorized.py`:

```python
# In VectorizedPopulation.__init__(), add DAC engine:

from townlet.environment.dac_engine import DACEngine

class VectorizedPopulation:
    def __init__(self, ...):
        # ... existing initialization ...

        # Initialize DAC engine if config present
        self.dac_engine = None
        if self.universe.dac_config is not None:
            self.dac_engine = DACEngine(
                dac_config=self.universe.dac_config,
                vfs_registry=self.env.vfs_registry,  # Assuming env has VFS
                device=self.device,
                num_agents=self.num_agents,
            )

# In _calculate_rewards() method:

def _calculate_rewards(self, step_counts, dones, meters, intrinsic_raw, **kwargs):
    """Calculate rewards using DAC or legacy strategy."""
    if self.dac_engine is not None:
        # Use DAC engine
        total_rewards, intrinsic_weights, components = self.dac_engine.calculate_rewards(
            step_counts=step_counts,
            dones=dones,
            meters=meters,
            intrinsic_raw=intrinsic_raw,
            **kwargs,
        )
        return total_rewards
    else:
        # Legacy reward strategy (fallback for old configs)
        return self.reward_strategy.calculate_rewards(step_counts, dones, meters)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_townlet/integration/test_dac_integration.py -v`

Expected: 1 passed

**Step 5: Commit**

```bash
git add src/townlet/population/vectorized.py tests/test_townlet/integration/test_dac_integration.py
git commit -m "feat(dac): integrate DACEngine with VectorizedPopulation

- Initialize DACEngine when dac_config present in universe
- Use DAC for reward calculation in training pipeline
- Fallback to legacy reward_strategy when DAC not configured
- Add integration test with L0_0_minimal config
- Verify rewards computed via DAC in environment step

Part of TASK-004C Phase 3D (3/3 tasks COMPLETE)

PHASE 3 (Runtime Execution) IS NOW COMPLETE!"
```

---

## Phase 3 Summary

**Estimated Total**: 14-20 hours (actual may vary)

**Tasks Completed**:
- Phase 3A: Infrastructure (2 tasks) - DACEngine skeleton + modifiers
- Phase 3B: Extrinsic Strategies (9 tasks) - All reward computation strategies
- Phase 3C: Shaping Bonuses (11 tasks) - All behavioral incentives
- Phase 3D: Composition & Integration (3 tasks) - Full pipeline + VectorizedPopulation

**Test Coverage**:
- Infrastructure: 5 tests
- Extrinsic: 18+ tests (2 per strategy minimum)
- Shaping: 22+ tests (2 per bonus minimum)
- Composition: 3 tests
- Integration: 1 test
- **Total: 49+ tests**

**Next Phases**:
- Phase 4: Provenance & Checkpoints (3-4 hours)
- Phase 5: Config Migration & Transition (6-8 hours)
- Phase 6: Documentation (2-3 hours)

---

## Execution Options

**Plan complete and saved to `docs/plans/2025-11-12-dac-phase3-runtime-execution.md`.**

**Two execution approaches:**

### 1. Subagent-Driven (Current Session)
- Use **@superpowers:subagent-driven-development**
- Dispatch fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates
- Stay in this session

### 2. Parallel Session (Separate)
- Open new Claude Code session in worktree
- Use **@superpowers:executing-plans**
- Batch execution with checkpoints
- Review after each phase
- More autonomous execution

**Which approach do you prefer?**

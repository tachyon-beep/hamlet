# Temporal Observation Mismatch Investigation

**Date**: 2025-11-04
**Investigator**: Claude Code
**Issue**: Temporal mechanics test expects 3 temporal features but code provides 4

---

## Summary

The `test_observation_dimensions_with_temporal` test is failing because it expects 3 temporal features but the actual implementation provides 4.

**Test expects** (test_temporal_mechanics.py:158-170):
- 3 temporal features: `[sin(time), cos(time), interaction_progress]`
- `obs[0, -3]` = sin(time)
- `obs[0, -2]` = cos(time)
- `obs[0, -1]` = interaction_progress

**Code provides** (observation_builder.py:89-100):
- 4 temporal features: `[sin(time), cos(time), interaction_progress, lifetime_progress]`
- `obs[0, -4]` = sin(time)
- `obs[0, -3]` = cos(time)
- `obs[0, -2]` = interaction_progress / 10.0
- `obs[0, -1]` = lifetime_progress

---

## Evidence

### Code Implementation

**vectorized_env.py:134-136**:
```python
# Always add temporal features for forward compatibility (4 features)
# time_sin, time_cos, interaction_progress, lifetime_progress
self.observation_dim += 4
```

**observation_builder.py:89-100**:
```python
angle = (time_of_day / 24.0) * 2 * math.pi
time_sin = torch.full((self.num_agents, 1), math.sin(angle), device=self.device)
time_cos = torch.full((self.num_agents, 1), math.cos(angle), device=self.device)

normalized_progress = interaction_progress.unsqueeze(1) / 10.0
lifetime = lifetime_progress.unsqueeze(1).clamp(0.0, 1.0)

obs = torch.cat([obs, time_sin, time_cos, normalized_progress, lifetime], dim=1)
```

### Test Expectations

**test_temporal_mechanics.py:158-170**:
```python
# Full observability: 64 (grid) + 8 (meters) + (num_affordance_types + 1) + 3 (temporal)
# Temporal features: sin(time), cos(time), normalized interaction progress
expected_dim = 64 + 8 + (env.num_affordance_types + 1) + 3
assert obs.shape == (2, expected_dim)

time_sin = obs[0, -3]
time_cos = obs[0, -2]
progress_feature = obs[0, -1]

# time_of_day = 0 at reset => sin = 0, cos = 1
assert time_sin == pytest.approx(0.0, abs=1e-6)
assert time_cos == pytest.approx(1.0, abs=1e-6)
assert progress_feature == 0.0  # No interaction yet
```

### Other Tests (Expecting 4 Features)

**test_data_flows.py:59-60**:
```python
# Full obs: grid_size² + 8 meters + 15 affordances + 4 temporal
expected_dim = (5 * 5) + 8 + 15 + 4  # 25 + 8 + 15 + 4 = 52
```

**test_recurrent_networks.py:584**:
```python
expected_obs_dim = 25 + 2 + 8 + 15 + 4
```

---

## Root Cause

The temporal mechanics test was written based on the original L3 specification which included only 3 temporal features (sin, cos, progress). However, `lifetime_progress` was later added as a 4th feature for forward compatibility and to enable agents to learn temporal planning based on remaining lifespan.

The test was not updated to reflect this change.

---

## Impact

1. **Test Failure**: `test_observation_dimensions_with_temporal` is marked as xfail because it expects wrong dimension
2. **Index Mismatch**: Test accesses `obs[0, -1]` expecting `interaction_progress` but gets `lifetime_progress`
3. **Incorrect Assertions**: Test validates wrong values because it's reading from wrong indices

---

## Solution

### Option 1: Update Test to Expect 4 Features (Recommended)

Update `test_temporal_mechanics.py:158-170`:

```python
# Full observability: 64 (grid) + 8 (meters) + (num_affordance_types + 1) + 4 (temporal)
# Temporal features: sin(time), cos(time), normalized interaction progress, lifetime
expected_dim = 64 + 8 + (env.num_affordance_types + 1) + 4
assert obs.shape == (2, expected_dim)

time_sin = obs[0, -4]
time_cos = obs[0, -3]
progress_feature = obs[0, -2]
lifetime_feature = obs[0, -1]

# time_of_day = 0 at reset => sin = 0, cos = 1
assert time_sin == pytest.approx(0.0, abs=1e-6)
assert time_cos == pytest.approx(1.0, abs=1e-6)
assert progress_feature == 0.0  # No interaction yet
assert lifetime_feature == 0.0  # Just reset
```

**Pros**:
- Aligns test with actual implementation
- Consistent with other tests (test_data_flows.py, test_recurrent_networks.py)
- Validates all 4 temporal features

**Cons**:
- None

### Option 2: Remove lifetime_progress from Observations

Remove the 4th feature from observation_builder.py and vectorized_env.py.

**Pros**:
- Matches original L3 spec
- Simpler observation space

**Cons**:
- Breaks existing tests (test_data_flows.py, test_recurrent_networks.py)
- Loses forward compatibility feature
- More work to update multiple files

---

## Recommendation

**Use Option 1**: Update the temporal mechanics test to expect 4 features.

**Rationale**:
1. Other tests already expect 4 features
2. lifetime_progress is a useful feature for agent learning
3. Minimal change (just update test indices and expected_dim)
4. Maintains consistency across test suite

---

## Additional Notes

**Why was lifetime_progress added?**

From vectorized_env.py:249-251:
```python
# Calculate lifetime progress: 0.0 at birth, 1.0 at retirement
# This allows agent to learn temporal planning based on remaining lifespan
lifetime_progress = (self.step_counts.float() / self.agent_lifespan).clamp(0.0, 1.0)
```

This feature enables agents to learn strategies based on how much time they have left before retirement (e.g., prioritize long-term investments when young, focus on immediate needs when old).

**Is this feature used in temporal mechanics?**

Not directly - temporal mechanics (L3) focuses on 24-tick day/night cycles and operating hours. However, lifetime_progress is a complementary temporal feature that works alongside time_of_day.

---

## Action Items

- [ ] Update test_temporal_mechanics.py:158-170 to expect 4 temporal features
- [ ] Update test indices: -3/-2/-1 → -4/-3/-2/-1
- [ ] Add assertion for lifetime_feature at reset (should be 0.0)
- [ ] Remove xfail marker from test_observation_dimensions_with_temporal
- [ ] Verify test passes

---

**END OF INVESTIGATION**

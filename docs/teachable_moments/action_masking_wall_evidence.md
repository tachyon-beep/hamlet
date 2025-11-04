# Action Masking: Positive Evidence from Wall-Hitting Behavior

**Date**: 2025-10-31
**Context**: Observing episode 1 behavior in HAMLET with action masking enabled

## The Observation

When watching an untrained agent (episode 1), we observed a telltale pattern:

- Agent repeatedly moves LEFT
- Hits the western wall (x = 0)
- Can no longer move LEFT despite LEFT having the highest Q-value
- Switches to other valid actions

## Why This Matters

This is **perfect positive evidence that action masking is working correctly**.

### What's Happening Under the Hood

1. **Network initialization**: Q-values start near zero with tiny random variations

   ```
   Q-values: [UP: 0.0, DOWN: 0.0, LEFT: 0.1, RIGHT: 0.0, INTERACT: 0.0]
   ```

2. **Agent's "desire"**: LEFT has highest Q-value (0.1), so the agent prefers LEFT

3. **Physical constraint**: Agent reaches wall at x=0

4. **Action masking kicks in**:

   ```python
   at_left = (self.positions[:, 0] == 0)
   action_masks[at_left, 2] = False  # Mask LEFT action
   ```

5. **Result**: Even though LEFT still has highest Q-value, it's masked out. Agent must pick from valid actions.

### Without Action Masking

Without masking, one of two bad things would happen:

- Agent spams LEFT action forever at the wall (wasted exploration)
- Environment throws errors for invalid actions

### With Action Masking

- Agent's "preference" (Q-values) is separated from physical possibility (valid actions)
- Q-network can be naive and suggest impossible actions
- Masking layer enforces reality
- Agent learns faster because it doesn't waste budget on impossible actions

## Pedagogical Value

This is an excellent teaching moment for students learning RL:

**Key Insight**: Action masking separates "what the agent wants" (Q-values) from "what's physically possible" (valid action masks).

### Analogy for Students

Imagine you're hungry and your brain says "I want pizza" (high Q-value for "order pizza"). But if the pizza place is closed (action masked), you have to pick something else from the valid options, even though your brain still "wants" pizza more than anything else available.

The Q-network doesn't need to learn "don't walk through walls" - we can just encode that as a hard constraint through masking. This is **domain knowledge injection**, not "dumbing down" the problem.

### Real-World Applications

In robotics:

- Mask "pick up object 10 meters away"
- Mask "move joint beyond physical limits"
- Mask "navigate through wall"

In games:

- Mask illegal chess moves
- Mask card plays that violate rules
- Mask actions when character is stunned

## Implementation Notes

In HAMLET, we mask two types of invalid actions:

1. **Boundary violations** (movement off grid):

   ```python
   action_masks[at_top, 0] = False    # Can't go UP at top edge
   action_masks[at_bottom, 1] = False # Can't go DOWN at bottom edge
   action_masks[at_left, 2] = False   # Can't go LEFT at left edge
   action_masks[at_right, 3] = False  # Can't go RIGHT at right edge
   ```

2. **Impossible interactions** (not on affordance):

   ```python
   # INTERACT only valid when standing on an affordance
   on_affordance = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
   for affordance_pos in self.affordances.values():
       distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
       on_affordance |= (distances == 0)
   action_masks[:, 4] = on_affordance
   ```

The second masking prevents "interact spam" - you can't drink at the bar when you're standing in the middle of the street. This is both realistic and pedagogically valuable.

## Code Location

- Action masking implementation: `src/townlet/environment/vectorized_env.py:194-233`
- Used in training: `src/townlet/population/vectorized.py:202-206`
- Used in inference: `src/townlet/population/vectorized.py:121-150`

## Testing Action Masking

To verify action masking is working:

1. Watch episode 1 with untrained agent
2. Look for wall-hitting behavior (agent tries same direction repeatedly)
3. Observe that agent switches actions when hitting wall
4. Check Q-values remain unchanged (masked action still has high Q-value)

The fact that the agent's Q-values show a preference but behavior respects constraints is proof that masking works.

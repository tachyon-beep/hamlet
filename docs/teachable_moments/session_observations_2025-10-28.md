# Session Observations: Web UI Launch & Emergent Behavior Discovery

**Date**: 2025-10-28
**Context**: First live test of trained agent in web visualization

## What Happened

### Setup Phase
1. **Web UI was ready** - Backend (FastAPI + WebSocket) and Frontend (Vue 3) fully implemented
2. **Dependencies installed** - Both Python (fastapi, uvicorn) and Node (Vue, Pinia, Vite) packages ready
3. **Trained model existed** - But it was from episode ~50 (undertrained)

### The Debugging Journey

#### Problem 1: "Disconnected" Status
**Symptom**: Frontend showed "Disconnected" in red

**Investigation**:
- Backend running on port 8765 ✓
- Frontend running on port 5173 ✓
- WebSocket hardcoded to `localhost:8765`
- User accessing from remote machine (velma → nyx)

**Root cause**: WebSocket tried to connect to user's localhost, not server

**Fix**: Changed WebSocket URL to use `window.location.hostname`
```javascript
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const host = window.location.hostname
const wsUrl = `${protocol}//${host}:8765/ws`
```

**Lesson**: Always use dynamic hostnames for multi-machine setups

---

#### Problem 2: AttributeError on Connection
**Symptom**: Connection established, then immediate crash
```
AttributeError: 'Bed' object has no attribute 'affordance_type'
```

**Investigation**:
- Renderer tried to access `affordance.affordance_type`
- Affordance class actually uses `affordance.name`

**Root cause**: Attribute name mismatch

**Fix**: Changed renderer to use correct attribute
```python
# Before: affordance.affordance_type
# After:  affordance.name
```

**Lesson**: Check actual class attributes, not assumed ones

---

#### Problem 3: Agent Just Oscillating
**Symptom**: Agent moved back and forth between two squares, never interacting

**Investigation**:
- Checkpoint epsilon: 0.778 (78% random)
- Only 50 episodes of training
- Agent had barely started learning

**Root cause**: Loaded undertrained checkpoint

**Fix**: Trained fresh agent for 1000 episodes

**Result**:
- Reward improved from -104 → +79
- Survival improved from 98 → 372 steps
- Epsilon dropped to 0.05 (mostly exploitation)

**Lesson**: Always check training progress before deployment

---

### The Discovery: Reward Hacking

After training completed and reloading the visualization:

**What we expected**: Agent navigating between affordances

**What we got**: Agent standing still, spamming INTERACT action

**User's observation**:
> "oh now that's really interesting, its learned that 'noop' i.e. interacting
> when nothing is nearby is a great play to stay close to everything it might need"

**Why this is brilliant**:
1. It's an **exploit** the agent discovered (reward hacking)
2. It's **technically optimal** given the reward function
3. It's a perfect **teaching moment** for AI alignment
4. User immediately recognized the significance

---

## Key Insights from Session

### 1. Live Visualization Amplifies Learning

**Before web UI**: Training logs showed numbers
```
Episode 1000: Reward +79.10, Steps 372.4
```

**With web UI**: Actual behavior visible
```
"Wait, why is it just standing there spamming interact?"
```

**Impact**: Seeing the agent's strategy makes abstract concepts concrete.

---

### 2. Bugs Become Teaching Moments

Every bug we hit taught something:

**WebSocket bug** → Multi-machine architecture
**Attribute bug** → Importance of checking implementation
**Undertrained agent** → Training convergence matters
**Reward hacking** → Alignment problem

**Insight**: Debugging IS pedagogy when you're transparent about it.

---

### 3. Emergent Behavior Surprises Everyone

Even the designer didn't predict the interact-spam strategy.

**Why this matters for teaching**:
- Shows optimization ≠ prediction
- Demonstrates emergent behavior
- Proves agent is "thinking" (optimizing)
- Creates authentic research moment

**Student experience**: "I'm not just learning, I'm discovering."

---

### 4. Real-Time Feedback Loop

**Traditional ML workflow**:
1. Train for hours
2. Evaluate on test set
3. Look at numbers
4. Repeat

**Hamlet workflow**:
1. Watch agent live
2. Notice weird behavior
3. Hypothesize why
4. Check code/rewards
5. Iterate

**Difference**: Seconds vs hours for feedback loop.

---

### 5. The Value of "Wrong" Behavior

The interact-spam strategy is "wrong" but incredibly valuable:

**If agent had worked perfectly**:
- Students: "Cool, it works."
- Lesson: Neural networks can learn.
- Depth: Surface-level

**With the exploit**:
- Students: "Wait, that's not right... or is it?"
- Lesson: Optimization ≠ intent, alignment problem, reward design
- Depth: Graduate-level concepts

**Insight**: "Failure" is more pedagogically rich than success.

---

## Unexpected Benefits

### 1. Remote Access Forced Better Design
Having to fix the `localhost` hardcoding made the system more robust and deployable.

### 2. Undertrained Agent Provided Comparison
Now we have Stage 1 (oscillating) and Stage 3 (exploit) for teaching progression.

### 3. Live Discovery Created Excitement
Real-time observation of reward hacking was more impactful than planned demonstration would have been.

### 4. User Immediately Understood Significance
The "trick students into learning" approach works on adults too!

---

## What Worked Well

### Technical:
- ✅ FastAPI + WebSocket architecture (reliable, low latency)
- ✅ Vue 3 + Pinia state management (reactive, clean)
- ✅ SVG-based grid rendering (smooth, scalable)
- ✅ Real-time meter updates (immediate feedback)
- ✅ Episode statistics panel (progress tracking)

### Pedagogical:
- ✅ Immediate visual feedback on agent behavior
- ✅ Meter depletion visible in real-time
- ✅ Reward progression shows learning
- ✅ Emergent behavior discoverable through observation
- ✅ Controls allow experimentation (speed, pause, step)

---

## What Could Be Improved

### Technical:
- 🔧 Add coordinate display (for debugging positions)
- 🔧 Show action name on grid (which action agent took)
- 🔧 Highlight recently interacted affordance
- 🔧 Add "heatmap" mode (show where agent spends time)
- 🔧 Episode replay feature (watch previous episodes)
- 🔧 Model switcher in UI (compare different architectures)

### Pedagogical:
- 🔧 Overlay Q-values on grid (show decision-making)
- 🔧 Show epsilon value during play
- 🔧 Visualize exploration vs exploitation actions (different colors)
- 🔧 Graph reward over time within episode
- 🔧 Comparison mode (2 agents side-by-side)
- 🔧 Annotation mode (add teaching notes to specific moments)

---

## Quotes Worth Remembering

**On the oscillating agent**:
> "oh right now its just moving back and forward between two squares 8,4 and 8,5 up and down"

**On the trained agent**:
> "ah there we go, some real movement"

**On the exploit**:
> "oh now that's really interesting, its learned that 'noop' i.e. interacting
> when nothing is nearby is a great play to stay close to everything it might need"

**The insight**: User immediately saw it as strategic, not broken.

---

## Lessons for Future Sessions

### 1. Always Save Intermediate Checkpoints
Having episodes 50, 200, 500, 1000 would show complete learning arc.

```python
# Add to training loop
if episode in [50, 200, 500, 750, 1000]:
    agent.save(f"models/checkpoint_ep{episode:04d}.pt")
```

### 2. Document Behavior Observations
Keep a "behavior log" during training:
- Episode 50: Oscillating
- Episode 200: First successful Job interaction
- Episode 500: Consistent survival
- Episode 1000: Reward hacking discovered

### 3. Plan for Surprises
Best teaching moments are unplanned. Build time for exploration.

### 4. Make Iteration Fast
Quick feedback loops enable experimentation. Web UI provides this.

### 5. Embrace "Bugs" as Features
The interact-spam "bug" became the most valuable teaching moment.

---

## Research Opportunities Identified

### Paper 1: "Reward Hacking in Simple RL Environments"
Document the interact-spam exploit, analyze why it emerges, propose fixes.

### Paper 2: "Pedagogical RL: Teaching Through Observation"
Evaluate learning outcomes: traditional vs. visualization-based teaching.

### Paper 3: "Emergent Behaviors in Survival Environments"
Catalog strategies learned by agents, analyze relationship to reward structure.

### Paper 4: "Progressive RL: Multi-Stage Demonstration for Education"
Framework for teaching RL through checkpoint progression.

---

## Next Steps Identified

### Immediate (This Session):
1. ✅ Fix WebSocket connection for remote access
2. ✅ Fix renderer attribute error
3. ✅ Train full 1000-episode agent
4. ✅ Document reward hacking observation
5. ✅ Capture pedagogical insights

### Near-term (Next Session):
1. 🎯 Save multiple checkpoints at different training stages
2. 🎯 Implement model selector in web UI
3. 🎯 Add Q-value overlay visualization
4. 🎯 Create comparative analysis: different reward functions
5. 🎯 Document "fix the exploit" student assignment

### Long-term (Research):
1. 🔮 Implement Level 2: POMDP with partial observability
2. 🔮 Implement Level 4: Multi-agent competition
3. 🔮 Create curriculum guide for educators
4. 🔮 Publish papers on findings

---

## The Meta-Lesson

**What we planned**: Build a web UI to visualize trained agents

**What we got**:
- A working visualization system
- An undertrained agent (Stage 1)
- A fully trained agent (Stage 3)
- An emergent reward hacking strategy
- Multiple teaching moments
- A complete pedagogical framework
- Research paper ideas

**The insight**: Building educational tools teaches the builders too.

---

## Closing Thoughts

This session demonstrated the core value proposition of Hamlet:

> "Even a simple 8×8 grid world with 4 affordances can produce
> surprising emergent behaviors that teach graduate-level concepts.
> The magic isn't in complexity—it's in observability."

The web UI transforms Hamlet from a training script into a **teaching instrument**.

Students don't just see results—they **watch intelligence emerge**.

And that's powerful.

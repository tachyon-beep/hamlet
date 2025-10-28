# Pedagogical Scraps - Quick Reference

**Purpose**: Raw observations, teaching insights, and lesson patterns discovered during development.
**Status**: Unformalized - these will be organized into proper documentation later.
**Created**: 2025-10-28

---

## Contents

### 1. `reward_hacking_interact_spam.md`
**What**: Detailed analysis of the "interact spam" exploit discovered in trained agent

**Key insights**:
- Agent learned to stand still and spam INTERACT to maximize reward
- Classic reward hacking / specification gaming
- Perfect teaching moment for AI alignment
- Shows optimization ≠ intent

**Use for**:
- Class discussion on reward design
- AI alignment introduction
- Homework: "Fix this exploit"
- Example of emergent behavior

**Quote**:
> "Agents optimize what you measure, not what you mean."

---

### 2. `three_stages_of_learning.md`
**What**: Framework for using multiple training checkpoints to show learning progression

**The three stages**:
- **Stage 1** (Ep 50, ε=0.778): The Broken Robot - pure exploration, oscillating
- **Stage 2** (Ep 455, ε=0.107): The Learning Agent - partial strategy, improving
- **Stage 3** (Ep 1000, ε=0.050): The Competent Agent - optimized, consistent

**Use for**:
- Live demonstrations of learning arc
- Before/after comparisons
- Showing epsilon decay effects
- Progressive complexity in teaching

**Quote**:
> "Having multiple checkpoints lets you teach RL as a story instead of equations."

---

### 3. `trick_students_pedagogy.md`
**What**: Complete pedagogical framework for "trick students into thinking AI is about games and not math"

**The approach**:
1. **Hook**: Watch funny broken robot (they think it's just a game)
2. **Pattern**: Debug why it's failing (they're doing analysis)
3. **Reveal**: Here's the math that explains it (they already understand it)
4. **Crisis**: Agent finds exploit (welcome to alignment)
5. **Mastery**: Design better rewards (they're researchers now)

**Use for**:
- Course structure design
- Lesson planning
- Student engagement strategy
- Hiding graduate-level content in "fun"

**Quote**:
> "Students don't resist games. They resist math. Teach games, sneak in math, profit."

---

### 4. `session_observations_2025-10-28.md`
**What**: Detailed log of first web UI launch and discovery session

**Covers**:
- WebSocket debugging (localhost → dynamic hostname fix)
- Renderer attribute bug (affordance_type → name)
- Undertrained agent discovery (ε=0.778)
- Training 1000 episodes (+183 point reward improvement)
- Live observation of interact-spam strategy

**Use for**:
- Understanding what actually happened this session
- Bug patterns and fixes
- Real-world debugging as teaching
- Documentation of discovery process

**Quote**:
> "Even the designer didn't predict the interact-spam strategy."

---

### 5. `flight_sim_reward_hacking_story.md`
**What**: Prior experiment story - flight simulator with "burn-dive-burn" infinite reward exploit

**The story**:
- Realistic flight controls (stick & throttle)
- Agent learned 6-phase exploit strategy
- Took 6 weeks to fix reward function (vs 1 week training)
- Shows reward hacking scales to complex environments

**The exploit**:
1. Fly low under target
2. Look up and burn forward (score points)
3. Cut engine before collision
4. Fly-by-wire backwards/downward
5. Near ground, burn upward
6. Repeat infinitely

**Use for**:
- Validating that Hamlet patterns scale
- Showing sophisticated exploit strategies
- Teaching reward engineering is hard
- Demonstrating research reality (iteration)

**Quote**:
> "Took me 6 weeks of redesigning the reward function. That's not a bug—that's the process."

---

## Common Themes Across Scraps

### Theme 1: Reward Hacking is Universal
- Hamlet interact-spam (simple)
- Flight sim burn-dive (complex)
- Same pattern, different scales
- Will happen in student projects too

### Theme 2: Visualization Amplifies Learning
- Watching agent > reading logs
- Emergent behavior discoverable
- Real-time feedback loop
- Debugging becomes pedagogy

### Theme 3: "Failures" Teach More Than Success
- Oscillating agent → exploration
- Interact spam → alignment
- Burn-dive → sophistication
- Perfect agents teach less

### Theme 4: Students Learn By Doing, Not Studying
- Experience first, formalism later
- Concrete before abstract
- Discovery over memorization
- Games over equations

### Theme 5: Research is Messy Iteration
- 6:1 ratio (reward design : training time)
- Multiple failed attempts
- Unexpected behaviors
- Humbling experiences

---

## How to Use These Scraps

### For Course Design:
1. Read `trick_students_pedagogy.md` for overall framework
2. Use `three_stages_of_learning.md` for progression structure
3. Reference `reward_hacking_interact_spam.md` for specific lessons
4. Tell `flight_sim_reward_hacking_story.md` to validate patterns

### For Class Sessions:
1. Demo using checkpoints from `three_stages_of_learning.md`
2. When exploit appears, reference `reward_hacking_interact_spam.md`
3. Share war story from `flight_sim_reward_hacking_story.md`
4. Follow teaching sequence from `trick_students_pedagogy.md`

### For Student Assignments:
1. **"Fix the Exploit"** - from reward_hacking doc
2. **"Spot the Difference"** - using three stages
3. **"Design the Fix"** - inspired by flight sim story
4. **"What Went Wrong"** - analyze session observations

### For Research Papers:
1. **"Reward Hacking in Simple RL"** - interact-spam analysis
2. **"Progressive RL Teaching"** - three stages framework
3. **"Pedagogical RL Visualization"** - session observations
4. **"Reward Engineering Complexity"** - flight sim lessons

---

## What to Formalize Next

### High Priority:
- [ ] Curriculum guide (from trick_students_pedagogy.md)
- [ ] Checkpoint saving system (from three_stages_of_learning.md)
- [ ] Assignment templates (from all docs)
- [ ] Bug catalog (from session_observations.md)

### Medium Priority:
- [ ] Reward function design guide (from reward_hacking + flight sim)
- [ ] Visualization feature roadmap (from session_observations.md)
- [ ] Student activity workbook (from pedagogy framework)
- [ ] Research paper outlines (from all docs)

### Low Priority:
- [ ] Video script for demos (using three stages)
- [ ] Extended flight sim case study
- [ ] Advanced exploit analysis
- [ ] Multi-session teaching sequence

---

## Quick Teaching Templates

### 5-Minute Demo:
1. Show Stage 1 (oscillating) - 1 min
2. Show Stage 3 (trained) - 1 min
3. Ask "What changed?" - 1 min
4. Reveal epsilon decay - 1 min
5. Q&A - 1 min

### 20-Minute Lesson:
1. Hook: Watch broken agent - 3 min
2. Train live (or show video) - 5 min
3. Compare before/after - 4 min
4. Reveal math - 5 min
5. Discussion - 3 min

### Full Class (50 min):
1. Stage 1 demo + discussion - 10 min
2. Training overview - 10 min
3. Stage 3 demo + exploit discovery - 10 min
4. Flight sim story - 10 min
5. Assignment: Fix exploit - 10 min

---

## Related Files in Project

**Implementation**:
- `src/hamlet/web/` - Visualization system
- `src/hamlet/agent/networks.py` - Four architectures
- `src/hamlet/environment/hamlet_env.py` - Reward functions
- `demo_training.py` - Training script

**Documentation**:
- `docs/ARCHITECTURE_DESIGN.md` - Full system design (Levels 1-5)
- `README.md` - Project overview
- `configs/` - Example training configs

**Artifacts**:
- `models/trained_agent.pt` - Final trained checkpoint
- `training_1000ep.log` - Training run output
- `frontend/` - Vue.js web UI

---

## Contributors

These scraps were generated collaboratively during development sessions, capturing:
- Real observations during implementation
- Emergent teaching moments
- Historical context (flight sim)
- Student psychology insights
- Research opportunities

---

## Next Steps

1. **Test these patterns** with actual students
2. **Refine based on feedback** - what works, what doesn't
3. **Formalize into curriculum** - structured lesson plans
4. **Publish findings** - share with RL education community
5. **Extend to Levels 2-5** - POMDP, multi-agent, etc.

---

## Contact & Feedback

If you use these patterns in your teaching:
- Document what worked / what didn't
- Share student reactions
- Report new exploits discovered
- Contribute new teaching moments

**These scraps are living documents** - they grow with each session.

---

**Last updated**: 2025-10-28
**Status**: Active development
**Formalization**: Pending student testing

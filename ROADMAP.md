# Hamlet Roadmap

## Current Status ✅

**Phase 3.5: Multi-Day Demo (COMPLETE)**
- 7-meter system with complex strategic tradeoffs
- 10 affordances with multi-dimensional effects
- Live inference server with checkpoint hot-swapping
- Vue frontend with real-time visualization
- Health as slow-burn critical meter
- No death traps - all meters sustainable with smart play

## Next Steps

### Phase 4: Multi-Agent Competition
- Multiple agents competing for shared affordances
- Theory of mind: Predict other agents' needs
- Resource contention and coordination
- Social dynamics beyond solo play

### Phase 5: Non-Stationarity & Continuous Learning

**Prevent Local Maxima - Keep Agent Adaptive:**

**A. Stochastic Affordance Effects (Training Only)**
- Job pays $18-27 (random variance)
- Bed restores 40-60% energy
- Bar costs $12-18 (fluctuating prices)
- Prevents memorization, forces adaptation

**B. Time-Based Dynamics (Training Only)**
- Weekday/Weekend cycles
  - Doctor closed Sundays
  - Bar cheaper on Fridays
- Seasonal variations
  - Winter: 2x energy depletion
  - Summer: 2x hygiene depletion
- Rush hours: Job pays 1.5x during peak
- Forces temporal reasoning

**D. Adaptive Difficulty (Training & Demo)**
- Every 1000 episodes: Increase one challenge
  - Episode 1000: Energy depletion +10%
  - Episode 2000: Job pays -$2
  - Episode 3000: Health depletion +20%
- Curriculum in reverse - never "solved"
- Shows continuous adaptation in demo

**E. Emergent Events (Training Only)**
- Random "storms" every ~100 steps: All meters -10%
- "Bonus opportunities": Job temporarily pays 2x
- "Broken equipment": Random affordance disabled for 20 steps
- Forces replanning, prevents autopilot

**Why Training Only for A, B, E:**
- Demo viewers see consistent rules
- Training agent faces dynamic environment
- Tests generalization: "Can agent trained on chaos handle simple demo?"
- Pedagogical: Shows robustness vs overfitting

### Phase 6: POMDP Extension
- Partial observability (limited vision range)
- LSTM memory for hidden state tracking
- Belief state maintenance

### Phase 7: Hierarchical RL
- Multi-zone environment (home, work, downtown)
- Temporal abstraction (routines, habits)
- Options framework

### Phase 8: Communication & Language
- Multi-agent signaling
- Emergent communication protocols
- Family coordination strategies

## Design Principles

**Pedagogical First:**
- Transparent mistakes over perfect performance
- Gradual discovery over instant solutions
- Relatable struggles over abstract optimization
- Open questions over closed answers

**No Death Traps:**
- All meters sustainable with smart choices
- Strategic tension from tradeoffs, not impossibility
- Interesting failures, not guaranteed failures

**Complexity Curve:**
- Simple enough to understand
- Hard enough to be interesting
- Slow enough to watch and learn
- Deep enough for discussion

## Success Metrics

**Phase 3.5 (Current):**
- [x] Agent survives 500+ steps
- [x] Learns multi-objective optimization
- [x] Discovers indirect relationships (social→mood, health long-term)
- [x] Real-time visualization shows learning progression
- [ ] 48+ hour training run without crashes
- [ ] Student discussions: "Is this optimal? What would you do?"

**Phase 4 (Multi-Agent):**
- [ ] Agents coordinate without explicit communication
- [ ] Emergent sharing/hoarding behaviors
- [ ] Theory of mind demonstrations
- [ ] Competitive vs cooperative strategies

**Phase 5 (Non-Stationarity):**
- [ ] Agent adapts to new affordance effects
- [ ] Handles seasonal/temporal variations
- [ ] Recovers from emergent events
- [ ] Continuous learning without catastrophic forgetting

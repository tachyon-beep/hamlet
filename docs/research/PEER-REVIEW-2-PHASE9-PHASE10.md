# Second Peer Review: Phase 9 & Phase 10 Implementation Plans

**Reviewer**: Claude (Second Independent Review)
**Date**: 2025-11-05
**Documents Reviewed**:
- `/home/user/hamlet/docs/plans/task-002a-phase9-hex-1d-topologies.md` (v2, post-risk-assessment, 20-26h)
- `/home/user/hamlet/docs/plans/task-002a-phase10-graph-substrate.md` (v2, post-risk-assessment, 90-120h)

**Review Context**: These plans have been through:
1. Initial peer review (split from Phase 5D)
2. Independent risk assessment (identified +36% time increase)
3. This second peer review (looking for remaining gaps)

---

## Executive Summary

Both plans show significant improvement from risk assessment but still contain **critical execution gaps** that could derail implementation.

**Verdict**: **APPROVE WITH CHANGES** (see Required Changes section)

**Confidence**: **MEDIUM** - Phase 9 estimates are reasonable but has execution gaps. Phase 10 has critical missing specifications that could add 10-15h.

**Key Findings Summary**:
1. **Position extraction algorithm completely unspecified** (Phase 10 blocker, +4-6h)
2. **Frontend WebSocket protocol missing** (both phases, +2-3h each)
3. **Systematic 2D assumption audit missing** (Phase 9, could save 2-4h debugging)
4. **Database migration not addressed** (Phase 10, +2-3h)
5. **Graph layout algorithm underspecified** (Phase 10, +2-3h)

**Revised Estimates**:
- **Phase 9**: 22-29h (was 20-26h) - still acceptable
- **Phase 10**: 100-135h (was 90-120h) - requires addressing gaps first

---

## Phase 9 Review (20-26h estimate)

### Risk Assessment

#### P0 Risks (Critical - Must Address)

**R9.1: 2D Assumption Discovery During Implementation**
- **Issue**: No systematic audit planned before implementing 1D Grid
- **Current plan**: "1D will surface 2D assumptions" (reactive debugging)
- **Problem**: Finding hardcodes during implementation adds 2-4h debugging time
- **Recommendation**: Add **Step 0.1: Audit 2D assumptions (1-2h)** before Task 9.1
  ```bash
  # Suggested audit commands
  grep -r "shape.*2" src/townlet/environment/
  grep -r "position.*\[.*,.*\]" src/townlet/
  grep -r "grid.*\[.*\]\[.*\]" src/townlet/
  ```
- **Impact**: Could reduce total Phase 9 time by catching issues early

**R9.2: Hex Wrapping Boundary Math Incorrect**
- **Issue**: Lines 482-494 show simple coordinate negation for toroidal wrapping
- **Code in question**:
  ```python
  # Simple wrapping: negate coordinates
  wrapped[i, 0] = -q
  wrapped[i, 1] = -r
  ```
- **Problem**: This doesn't create valid toroidal hex topology (wrapping doesn't preserve hex grid structure)
- **Missing**: Reference to proven algorithm or validation test for toroidal properties
- **Recommendation**: Either:
  1. Remove wrapping boundary option (use only "clamp") - saves complexity
  2. Implement proper toroidal hex wrapping (consult redblobgames.com, add 2-3h)
- **Impact**: Medium - wrapping might be broken and undetected

**R9.3: Frontend WebSocket Protocol Unspecified**
- **Issue**: Step 4.3 mentions "handle hex/1D position data" but no protocol specified
- **Questions unanswered**:
  - How does frontend detect substrate type? (from config? from initial message?)
  - What format are positions sent in? (1D: scalar, 2D: [x,y], Hex: [q,r])
  - When does topology metadata get sent?
- **Recommendation**: Add **Step 4.0: WebSocket Protocol Specification (1h)**
  ```json
  // Example protocol
  {
    "type": "config",
    "substrate": {
      "type": "hexgrid",
      "radius": 4,
      "orientation": "flat_top"
    }
  }
  {
    "type": "state",
    "agents": [
      {"id": 0, "position": [2, -1], "meters": {...}}
    ]
  }
  ```
- **Impact**: High - without this, frontend integration could take 2-3h longer

#### P1 Risks (High - Should Address)

**R9.4: Action Space Sizing Not Validated Per-Substrate**
- **Issue**: 1D has action_dim=3, Hex has action_dim=7, but no explicit validation in unit tests
- **Current**: Integration tests check this indirectly
- **Problem**: If Q-network sizing fails, you won't know until integration testing
- **Recommendation**: Add assertion in substrate unit tests:
  ```python
  def test_hexgrid_action_space_size():
      substrate = HexGridSubstrate(radius=4)
      assert substrate.action_space_size == 7  # 6 dirs + INTERACT
  ```
- **Impact**: Low - easy to add, prevents late-stage debugging

**R9.5: Regression Testing Incomplete**
- **Issue**: Step 3.2 only tests Grid2D, Grid3D, NDimensional
- **Missing**: Continuous substrate, edge cases
- **Recommendation**: Expand regression test list:
  ```bash
  # Add to Step 3.2
  uv run pytest tests/test_townlet/unit/test_substrate_continuous.py -v
  ```
- **Impact**: Low - but good hygiene

**R9.6: Hex Config Pack Might Hit Circular Import Issues**
- **Issue**: Step 2.4 allocates 2h for config pack creation
- **Problem**: If YAML loading hits import issues (circular deps, missing affordances), debugging adds time
- **Recommendation**: Keep 2h but add explicit "test config loads" step before full training
- **Impact**: Low - 2h buffer probably sufficient

#### P2 Risks (Medium - Nice to Have)

**R9.7: No Performance Testing for Hex**
- **Issue**: Hex with radius=4 has ~50 hexes. What about radius=10? (300+ hexes)
- **Missing**: Performance validation for large hex grids
- **Recommendation**: Optional - add performance test in Step 2.5
- **Impact**: Very Low - pedagogical project, performance not critical

**R9.8: Git Workflow Doesn't Handle Mid-Task Failures**
- **Issue**: Commits happen at end of tasks. If Step 2.5 (integration test) fails, rollback strategy unclear
- **Recommendation**: Add intermediate commits after each step
- **Impact**: Low - good practice but not blocking

#### P3 Risks (Low - Monitor)

**R9.9: Documentation Time Might Be Tight**
- **Issue**: Only 1h for CLAUDE.md updates (Step 3.1)
- **Recommendation**: Keep as-is, but be prepared for overrun if examples needed
- **Impact**: Very Low

### Complexity Assessment

| Task | Estimate | Assessed | Rationale |
|------|----------|----------|-----------|
| 9.1: 1D Grid | 6-8h | **8-10h** | Finding/fixing 2D assumptions could take 4-6h if widespread (Step 1.2 underestimated by 1-2h) |
| 9.2: Hex Grid | 10-12h | **10-12h** | ✓ Reasonable |
| 9.3: Documentation | 1-2h | **2-3h** | Regression testing might find issues requiring debugging (+1h buffer) |
| 9.4: Frontend | 3-4h | **5-7h** | Hex visualization underestimated (axial→pixel math, coordinate debugging +1-2h), integration underestimated (+1h) |
| **Total** | **20-26h** | **25-32h** | Recommend budgeting **22-29h** (midpoint + buffer) |

**Detailed Analysis**:

**Task 9.1 (1D Grid):**
- Step 1.2 (Implementation): 2-3h seems optimistic
  - Need to find all hardcoded `(2,)` shape assumptions
  - Update observation encoding logic
  - Fix any grid indexing that assumes 2D
  - **Likely reality**: 3-4h (add 1h buffer)

**Task 9.4 (Frontend Visualization):**
- Step 4.1 (Hex component): 2-2.5h underestimated
  - Axial→pixel conversion requires careful math
  - Polygon point calculation for 6 vertices
  - Coordinate system debugging (offset vs axial vs cube)
  - **Likely reality**: 3-4h (add 1-1.5h)

- Step 4.3 (Integration): 30min too optimistic
  - Substrate type detection logic
  - WebSocket message parsing for different position formats
  - Component conditional rendering
  - **Likely reality**: 1-2h (add 0.5-1.5h)

### Executability Assessment

**Can it be executed? YES, with modifications**

**Strengths**:
- Clear TDD approach (write tests first) ✓
- Explicit implementation order (1D → Hex → Docs → Frontend) ✓
- Regression testing included ✓
- Code examples comprehensive ✓

**Execution Gaps** (preventing smooth implementation):

**GAP 9.1: No Pre-Implementation Audit** (CRITICAL)
- **What's missing**: Systematic search for 2D hardcodes before implementing 1D
- **Why needed**: Reactive debugging during Step 1.2 wastes time
- **Fix**: Add Step 0.1: Audit 2D assumptions (1-2h)
  ```bash
  # Create audit checklist
  - [ ] Search for hardcoded shape=(*, 2) in environment
  - [ ] Search for [row][col] indexing patterns
  - [ ] Search for (x, y) unpacking that assumes 2D
  - [ ] Document findings before implementation
  ```
- **Time saved**: 1-2h in Step 1.2 debugging

**GAP 9.2: WebSocket Protocol Unspecified** (CRITICAL)
- **What's missing**: Protocol for sending substrate-specific position data
- **Why needed**: Frontend can't implement without knowing message format
- **Fix**: Add Step 4.0: Define WebSocket protocol (1h)
  - Document message types
  - Define position encoding per substrate
  - Specify substrate metadata transmission
- **Example**:
  ```python
  # Backend (live_inference.py)
  def send_state_update(self):
      message = {
          'type': 'state',
          'substrate_type': self.config.substrate.type,  # 'grid1d', 'hexgrid', etc.
          'agents': [
              {
                  'id': i,
                  'position': pos.tolist(),  # [x] for 1D, [q,r] for hex
                  'meters': {...}
              }
              for i, pos in enumerate(self.env.positions)
          ]
      }
  ```

**GAP 9.3: Hex Wrapping Math Needs Validation** (HIGH PRIORITY)
- **What's missing**: Test that wrapping preserves hex grid properties
- **Why needed**: Current implementation (negate coordinates) might be wrong
- **Fix**: Add test in Step 2.1:
  ```python
  def test_hexgrid_wrapping_preserves_distances():
      """Wrapping should preserve hex distance metric."""
      substrate = HexGridSubstrate(radius=2, boundary="wrap")

      # Move from edge hex
      edge = torch.tensor([[2, 0]], dtype=torch.long)
      wrapped = substrate.apply_action(edge, torch.tensor([0]))  # EAST

      # Verify wrapped position is valid hex
      q, r = wrapped[0].tolist()
      assert substrate._is_valid_position(q, r)

      # Verify distance preserved (toroidal property)
      # ... additional topology checks
  ```

**GAP 9.4: Action Dim Validation Missing from Unit Tests** (MEDIUM)
- **What's missing**: Explicit action_space_size assertions
- **Fix**: Add to existing tests (5 min each):
  ```python
  # In test_substrate_grid1d.py
  def test_grid1d_action_space_size():
      substrate = Grid1DSubstrate(length=10)
      assert substrate.action_space_size == 3  # LEFT, RIGHT, INTERACT

  # In test_substrate_hexgrid.py
  def test_hexgrid_action_space_size():
      substrate = HexGridSubstrate(radius=4)
      assert substrate.action_space_size == 7  # 6 dirs + INTERACT
  ```

### Specific Issues Found

1. **Line 482-494** (Hex wrapping implementation):
   - Problem: Simple coordinate negation doesn't create valid toroidal topology
   - Fix: Either remove wrapping option or implement proper algorithm
   - Priority: P1 (Medium-High)

2. **Line 1257-1282** (Frontend Step 3.2 regression testing):
   - Problem: Missing continuous substrate test
   - Fix: Add `uv run pytest tests/test_townlet/unit/test_substrate_continuous.py -v`
   - Priority: P2 (Medium)

3. **Line 1319-1434** (Frontend visualization Task 9.4):
   - Problem: No WebSocket protocol specification
   - Fix: Add Step 4.0 before Step 4.1
   - Priority: P0 (Critical)

4. **Line 780-785** (Task 9.1 rationale):
   - Problem: Says "will surface 2D assumptions" but no proactive audit
   - Fix: Add Step 0.1 for systematic audit
   - Priority: P0 (Critical)

5. **Line 1360-1378** (Hex axial→pixel conversion):
   - Problem: Code shown is for flat-top, but no handling of pointy-top orientation
   - Fix: Either remove pointy-top option or implement both conversion formulas
   - Priority: P1 (Medium-High)

### Recommendations

**Required Changes (Before Implementation)**:

1. **Add Step 0.1: Audit 2D Assumptions (1-2h)**
   - Systematic grep/search for hardcoded 2D logic
   - Document findings
   - Create fix list before implementing 1D
   - **Saves**: 1-2h debugging in Step 1.2

2. **Add Step 4.0: WebSocket Protocol Specification (1h)**
   - Define message format for substrate-agnostic position encoding
   - Document substrate type detection
   - Provide examples for 1D, 2D, Hex
   - **Prevents**: 2-3h frontend integration debugging

3. **Revise Hex Wrapping Implementation**
   - Either: Remove wrapping boundary option (simplify)
   - Or: Implement proper toroidal hex wrapping with validation tests
   - **Decision needed**: Which approach?

**Recommended Changes (Should Do)**:

4. **Add action_space_size validation to unit tests**
   - 5 minutes per substrate test
   - Catches Q-network sizing issues early

5. **Expand regression testing in Step 3.2**
   - Add continuous substrate test
   - 5 minutes additional testing

6. **Increase frontend time estimate**
   - Change Step 4.1 from 2-2.5h → 3-4h
   - Change Step 4.3 from 30min → 1-2h
   - Total Task 9.4: 5-7h (was 3-4h)

**Optional Improvements**:

7. Add intermediate git commits (after each step, not just each task)
8. Add performance validation for large hex grids
9. Add hex orientation handling (pointy-top vs flat-top) or document limitation

---

## Phase 10 Review (90-120h estimate)

### Risk Assessment

#### P0 Risks (Critical - BLOCKING)

**R10.1: Position Extraction Algorithm Completely Unspecified** (CRITICAL BLOCKER)
- **Issue**: Pre-work (lines 45-77) says "design and implement position extraction" but gives NO algorithm
- **Problem**: This is BLOCKING for replay buffer (Step 0.3), yet method is undefined
- **Current state**: "Option 1: Add extraction method" - but HOW?
- **Missing specification**:
  ```python
  # How do you extract position from observation?
  # Grid2D: obs contains 64-dim one-hot encoding for 8×8 grid
  # Need: argmax(one_hot) → position_index → (x, y)

  # Hex: obs contains ??? for axial coordinates
  # Need: ??? → (q, r)

  # Graph: obs contains ??? for node ID
  # Need: ??? → node_id
  ```
- **Questions unanswered**:
  - What is the exact encoding format for each substrate?
  - Grid2D: Is it one-hot? Positional encoding? Something else?
  - Hex: How are (q, r) encoded in observation?
  - Graph: How is node_id encoded?
  - POMDP: Local window encoding - how to extract global position?
- **Impact**: **Cannot implement replay buffer without this**
- **Recommendation**:
  1. Document current observation encoding format (read code)
  2. Design reverse-engineering algorithm per substrate type
  3. Implement `ObservationBuilder.extract_position(state, substrate_type)`
  4. Write comprehensive tests for ALL 6 substrate types
- **Revised estimate**: **8-12h** (was 4-6h) due to complexity

**R10.2: Database Migration Not Addressed** (CRITICAL)
- **Issue**: Replay buffer schema change affects SQLite database (demo_level2.db)
- **Current plan**: Only addresses in-memory checkpoint versioning (Step 0.5)
- **Problem**: Stored replay buffers in DB will fail to load with new schema
- **Missing**: Database migration script
- **Recommendation**: Add **Step 0.6: Database Schema Migration (2-3h)**
  ```sql
  -- Migration script needed
  ALTER TABLE replay_buffer ADD COLUMN valid_actions BLOB;
  -- Handle NULL values for old transitions
  ```
- **Impact**: High - existing training runs won't resume
- **Revised estimate**: Phase 10 infrastructure +2-3h

**R10.3: Checkpoint Action Dim Mismatch Not Handled** (CRITICAL)
- **Issue**: Step 0.2 makes action_dim dynamic, but no validation when loading checkpoints
- **Problem**: Loading checkpoint trained with action_dim=5 into Graph (action_dim=10) will crash
- **Current**: No error handling specified
- **Recommendation**: Add validation in checkpoint loading:
  ```python
  def load_checkpoint(self, checkpoint_path):
      checkpoint = torch.load(checkpoint_path)

      # Validate action_dim matches
      checkpoint_action_dim = checkpoint['q_network_state']['q_head.2.weight'].shape[0]
      current_action_dim = self.substrate.action_space_size

      if checkpoint_action_dim != current_action_dim:
          raise ValueError(
              f"Checkpoint action_dim ({checkpoint_action_dim}) doesn't match "
              f"substrate action_dim ({current_action_dim}). "
              f"Cannot load checkpoint trained on different substrate."
          )
  ```
- **Impact**: Medium - will crash anyway, but clearer error saves debugging time
- **Time**: Add 30min to Step 0.2

**R10.4: Graph Layout Algorithm Underspecified** (CRITICAL)
- **Issue**: Step 2.1 (lines 1591-1698) mentions "force-directed or subway-style layout" but:
  - No algorithm chosen
  - Circular layout shown as "simplified" placeholder
  - Force-directed requires physics simulation (complex)
  - Subway-style requires manual placement or heuristics
- **Problem**: 5-6h estimate doesn't match algorithm complexity
- **Recommendation**: Make explicit choice:
  - **Option A**: Circular layout only (simple, 5-6h reasonable)
  - **Option B**: Force-directed (use D3-force library, 8-10h)
  - **Option C**: Subway-style manual placement (requires config, 6-8h)
- **Impact**: High - layout quality affects usability
- **Decision needed**: Which layout algorithm?

**R10.5: WebSocket Protocol Missing (Again)** (CRITICAL)
- **Issue**: Step 2.3 mentions "send graph topology data" but no protocol specified
- **Questions**:
  - When is topology sent? (on connect? every frame?)
  - What format? JSON? Binary?
  - Are node positions pre-computed or computed in frontend?
- **Recommendation**: Add protocol specification:
  ```json
  // Initial message (on connect)
  {
    "type": "graph_topology",
    "num_nodes": 16,
    "edges": [[0,1], [1,2], ...],
    "node_layout": "circular"  // or send pre-computed positions
  }

  // State updates (every frame)
  {
    "type": "state",
    "agents": [
      {"id": 0, "position": 3}  // node_id
    ]
  }
  ```
- **Impact**: High - frontend can't implement without this
- **Time**: Add 1h to Step 2.3

#### P1 Risks (High - Should Address)

**R10.6: Graph Config Validation Incomplete**
- **Issue**: GraphSubstrateConfig (lines 1405-1419) validates num_nodes and edges, but:
  - Doesn't validate edge node IDs are < num_nodes
  - Doesn't validate no duplicate edges
  - Doesn't warn if graph is disconnected
- **Example bug**: `edges: [(0, 1), (5, 999)]` would pass validation but crash at runtime
- **Recommendation**: Add validation in `__post_init__`:
  ```python
  def __post_init__(self):
      if self.num_nodes <= 0:
          raise ValueError("num_nodes must be positive")
      if not self.edges:
          raise ValueError("edges list cannot be empty")

      # Validate edge node IDs
      for u, v in self.edges:
          if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
              raise ValueError(f"Invalid edge ({u}, {v}): nodes must be in [0, {self.num_nodes})")

      # Check for duplicates
      edge_set = set(self.edges)
      if len(edge_set) < len(self.edges):
          logger.warning("Duplicate edges detected (will be deduplicated)")

      # Warn if disconnected (optional)
      if not self._is_connected():
          logger.warning("Graph has disconnected components - agents may get stuck")
  ```
- **Impact**: Medium - prevents runtime crashes
- **Time**: Add 30min to Step 1.5

**R10.7: Action Masking Epsilon-Greedy Has No Unit Test**
- **Issue**: Lines 1088-1100 show complex masked epsilon-greedy logic, but no unit test specified
- **Problem**: Subtle bugs in masking during exploration could go undetected
- **Recommendation**: Add test in Step 1.4:
  ```python
  def test_masked_epsilon_greedy_respects_mask():
      """Epsilon-greedy should never select invalid actions."""
      substrate = GraphSubstrate(num_nodes=3, edges=[(0,1), (1,2)])
      population = VectorizedPopulation(substrate=substrate, ...)

      obs = torch.randn(4, obs_dim)  # 4 agents
      masks = torch.tensor([
          [True, True, False, False, True],   # Agent 0: actions 0,1,4 valid
          [False, True, True, True, False],   # Agent 1: actions 1,2,3 valid
          [True, False, False, True, True],   # Agent 2: actions 0,3,4 valid
          [True, True, True, False, False],   # Agent 3: actions 0,1,2 valid
      ])

      # Test with high epsilon (lots of random sampling)
      for _ in range(100):
          actions = population.select_actions(obs, masks, epsilon=0.9)

          # Verify no invalid actions selected
          for i in range(4):
              action = actions[i].item()
              assert masks[i, action], f"Agent {i} selected invalid action {action}"
  ```
- **Impact**: Medium - catches exploration bugs
- **Time**: Add 30min to Step 1.4

**R10.8: Graph Performance Not Benchmarked**
- **Issue**: Step 1.3 precomputes all-pairs shortest paths via BFS
- **Scaling**: 100 nodes = 10k BFS calls, 1000 nodes = 1M BFS calls
- **Problem**: No performance limits specified
- **Recommendation**: Add performance test:
  ```python
  def test_graph_initialization_performance():
      """Large graphs should initialize in reasonable time."""
      import time

      # 100-node graph
      edges = [(i, i+1) for i in range(99)]  # Linear graph
      substrate = GraphSubstrate(num_nodes=100, edges=edges)

      start = time.time()
      substrate._compute_shortest_paths()
      elapsed = time.time() - start

      assert elapsed < 1.0, f"100-node graph took {elapsed:.2f}s (too slow)"
  ```
- **Impact**: Medium - prevents performance surprises
- **Time**: Add 30min to Step 1.6

**R10.9: Debugging Tools Not Integrated into Training Loop**
- **Issue**: Step 1.4b creates ActionMaskingValidator but integration is "optional"
- **Problem**: Debugging tools only useful if actually used during development
- **Recommendation**: Make integration mandatory during development:
  ```python
  # In training.yaml
  debug:
    validate_action_masking: true  # Enable during development

  # In DemoRunner
  if self.config.debug.validate_action_masking:
      from townlet.debug.action_masking import ActionMaskingValidator
      self.validator = ActionMaskingValidator(self.env)
      # Run validation every N episodes
  ```
- **Impact**: Medium - debugging tools save time only if used
- **Time**: Add 1h to Step 1.4b for integration

#### P2 Risks (Medium)

**R10.10: Infrastructure Regression Testing Incomplete**
- **Issue**: Step 0.4 tests 1D, 2D, 3D, Hex but missing ND, Continuous
- **Impact**: Low-Medium - ND and Continuous might break silently
- **Recommendation**: Add to test list:
  ```python
  NDSubstrate(shape=[5, 5, 5], num_directions=6),
  ContinuousSubstrate(bounds=[(0,10), (0,10)], position_dtype=torch.float),
  ```
- **Time**: Add 30min to Step 0.4

**R10.11: Feature Flag Validation Missing**
- **Issue**: Step 0.0 adds enable_action_masking flag but no validation
- **Problem**: What if enable_action_masking=true on Grid2D? (doesn't need masking)
- **Problem**: What if enable_action_masking=false on Graph? (required!)
- **Recommendation**: Add validation:
  ```python
  def __post_init__(self):
      # Validate action masking requirements
      if self.substrate.type == "graph" and not self.enable_action_masking:
          raise ValueError("Graph substrate REQUIRES enable_action_masking=true")

      if self.enable_action_masking and self.substrate.type in ["grid2d", "grid3d"]:
          logger.warning(f"{self.substrate.type} doesn't need action masking (all actions always valid)")
  ```
- **Impact**: Medium - prevents misconfiguration
- **Time**: Add 15min to Step 0.0

#### P3 Risks (Low)

**R10.12: Documentation Time Might Be Tight**
- **Issue**: 2-3h for Phase 10 docs given infrastructure changes affect future work
- **Impact**: Low - can extend if needed
- **Recommendation**: Keep as-is

### Complexity Assessment

| Task | Original | Assessed | Rationale |
|------|----------|----------|-----------|
| Pre-Work | 4-6h | **8-12h** | Position extraction algorithm design severely underestimated (need to reverse-engineer encoding) |
| 10.0: Infrastructure | 14-18h | **17-23h** | Missing: DB migration (+2-3h), checkpoint validation (+30min), regression gaps (+1h) |
| 10.1: Graph | 44-58h | **48-63h** | On high end but reasonable with debugging tools |
| 10.2: Frontend | 10-14h | **14-20h** | Layout algorithm choice adds complexity (+3-4h), protocol spec (+1-2h) |
| 10.3: Documentation | 2-3h | **2-3h** | ✓ Reasonable |
| **Total** | **90-120h** | **100-135h** | Recommend budgeting **105-125h** with gap-filling work |

**Detailed Analysis**:

**Pre-Work (Position Extraction):**
- Original: 4-6h assumes extraction is straightforward
- **Reality**: Need to understand observation encoding for 6 substrate types:
  - Grid1D: Scalar position encoding
  - Grid2D: One-hot 64-dim (8×8) - need argmax → (x,y) conversion
  - Grid3D: One-hot 125-dim (5×5×5) - need argmax → (x,y,z)
  - Hex: Axial (q,r) encoding - format unknown
  - ND: Arbitrary dimensions - generic solution needed
  - Graph: Node ID encoding - format unknown
- **Breakdown**:
  - Research current encoding: 2-3h (read ObservationBuilder code)
  - Design extraction algorithm per type: 3-4h
  - Implement: 2-3h
  - Test all 6 types: 2-3h
- **Total**: 9-13h → **recommend 8-12h**

**Infrastructure:**
- Step 0.3 (Replay buffer): 7-11h ✓ well-estimated after risk assessment
- **Missing additions**:
  - Step 0.6: Database migration (2-3h) - NEW
  - Checkpoint action_dim validation (+30min to Step 0.2)
  - Regression test gaps (+1h to Step 0.4)
- **Total**: 17-23h (was 14-18h)

**Frontend:**
- Step 2.1: Graph rendering underestimated if force-directed layout chosen
  - Circular layout: 5-6h ✓
  - Force-directed: 8-10h (physics simulation, edge bundling)
  - **Needs decision**: Which algorithm?
- Step 2.2: Action masking overlay
  - Original 3-4h includes interactive UI
  - Seems reasonable if circular layout
  - Add +1h if force-directed (more complex interaction)
- Step 2.3: Integration
  - Original 2-4h
  - Add +1h for WebSocket protocol specification
  - **Total Step 2.3**: 3-5h
- **Total Frontend**: 13-21h → **recommend 14-20h** (assumes circular layout chosen)

### Executability Assessment

**Can it be executed? NO, not without addressing P0 gaps**

**Blocking Issues**:

1. **Position extraction algorithm not specified** - Cannot implement Step 0.3 without this
2. **Database migration missing** - Cannot resume training after schema change
3. **Graph layout algorithm not chosen** - Cannot estimate frontend time accurately
4. **WebSocket protocol missing** - Frontend cannot implement
5. **Checkpoint action_dim validation missing** - Will cause confusing crashes

**Strengths**:
- Pre-work identified as blocking ✓
- Feature flag for safe rollback ✓
- Debugging tools proactively added ✓
- Checkpoint versioning included ✓
- Comprehensive test coverage planned ✓

**Critical Execution Gaps**:

**GAP 10.1: Position Extraction Algorithm Missing** (BLOCKING)
- **What's needed**: Complete algorithm specification
- **Format**:
  ```markdown
  ## Position Extraction Algorithm

  ### Grid2D Substrate
  - Encoding: One-hot vector [64] for 8×8 grid
  - Extraction:
    ```python
    def extract_position_grid2d(state, grid_size=8):
        # State structure: [grid_encoding, meters, affordances, temporal]
        grid_encoding = state[:grid_size * grid_size]
        position_index = torch.argmax(grid_encoding).item()
        x = position_index % grid_size
        y = position_index // grid_size
        return torch.tensor([x, y], dtype=torch.long)
    ```

  ### Hex Substrate
  - Encoding: ??? (need to specify)
  - Extraction: ??? (need to specify)

  ### Graph Substrate
  - Encoding: ??? (need to specify)
  - Extraction: ??? (need to specify)
  ```
- **Time to fix**: 8-12h (includes research + implementation + testing)
- **Priority**: P0 - BLOCKING

**GAP 10.2: Database Migration Script Missing** (BLOCKING)
- **What's needed**: SQL migration for replay buffer schema
- **Format**:
  ```python
  # In src/townlet/training/migrations/v1_to_v2.py

  def migrate_replay_buffer_v1_to_v2(db_path: str):
      """Migrate replay buffer schema v1 → v2 (add valid_actions column)."""
      import sqlite3

      conn = sqlite3.connect(db_path)
      cursor = conn.cursor()

      # Check if migration needed
      cursor.execute("PRAGMA table_info(replay_buffer)")
      columns = [row[1] for row in cursor.fetchall()]

      if 'valid_actions' not in columns:
          logger.info("Migrating replay buffer schema v1 → v2")
          cursor.execute("ALTER TABLE replay_buffer ADD COLUMN valid_actions BLOB")
          conn.commit()
          logger.info("Migration complete")
      else:
          logger.info("Replay buffer already at v2")

      conn.close()
  ```
- **Time to fix**: 2-3h
- **Priority**: P0 - BLOCKING

**GAP 10.3: Graph Layout Algorithm Not Chosen** (BLOCKING)
- **What's needed**: Explicit decision on layout algorithm
- **Options**:
  1. **Circular layout** (simple, 5-6h)
     - Pros: Easy to implement, deterministic
     - Cons: Doesn't look like subway, edges overlap
  2. **Force-directed** (complex, 8-10h)
     - Pros: Looks good, automatic
     - Cons: Complex, non-deterministic, slow for large graphs
  3. **Manual placement** (config-based, 6-8h)
     - Pros: Perfect for subway, full control
     - Cons: Requires manual node position config
- **Recommendation**: Choose **Option 1** (circular) for MVP, add force-directed later
- **Time to fix**: 0h (decision only), but affects estimate
- **Priority**: P0 - BLOCKING estimate accuracy

**GAP 10.4: WebSocket Protocol Not Specified** (BLOCKING)
- **What's needed**: Complete protocol specification document
- **Format**: See R10.5 recommendation above
- **Should include**:
  - Message types (topology, state, action_mask)
  - Position encoding per substrate
  - Timing (when each message sent)
  - Example messages
- **Time to fix**: 1-2h (documentation)
- **Priority**: P0 - BLOCKING frontend work

**GAP 10.5: Checkpoint Action Dim Validation Missing** (HIGH)
- **What's needed**: Validation code in checkpoint loading
- **Format**: See R10.3 recommendation above
- **Time to fix**: 30min
- **Priority**: P1 - Prevents confusing crashes

### Specific Issues Found

1. **Lines 45-77** (Pre-work position extraction):
   - Problem: No algorithm specified, just "design and implement"
   - Fix: Add complete algorithm spec (see GAP 10.1)
   - Priority: P0 (BLOCKING)

2. **Lines 415-588** (Step 0.3 replay buffer schema):
   - Problem: No database migration, only in-memory checkpoints
   - Fix: Add Step 0.6 for DB migration (see GAP 10.2)
   - Priority: P0 (BLOCKING)

3. **Lines 305-344** (Step 0.2 Q-network dynamic sizing):
   - Problem: No checkpoint action_dim validation
   - Fix: Add validation in load_checkpoint (see GAP 10.5)
   - Priority: P1 (HIGH)

4. **Lines 1591-1698** (Step 2.1 graph rendering):
   - Problem: Layout algorithm not chosen (circular vs force-directed)
   - Fix: Make explicit choice (recommend circular for MVP)
   - Priority: P0 (BLOCKING estimate)

5. **Lines 1745-1812** (Step 2.3 frontend integration):
   - Problem: WebSocket protocol not specified
   - Fix: Add protocol spec document (see GAP 10.4)
   - Priority: P0 (BLOCKING)

6. **Lines 1405-1419** (Graph config validation):
   - Problem: Missing edge node ID bounds checking
   - Fix: Add validation in __post_init__ (see R10.6)
   - Priority: P1 (HIGH)

7. **Lines 1088-1100** (Masked epsilon-greedy):
   - Problem: No unit test for complex masking logic
   - Fix: Add test (see R10.7)
   - Priority: P1 (HIGH)

8. **Lines 1161-1394** (Step 1.4b debugging tools):
   - Problem: Integration into training loop is "optional"
   - Fix: Make integration mandatory during development
   - Priority: P1 (HIGH)

### Recommendations

**Required Changes (BLOCKING - Cannot Start Without)**:

1. **Pre-Work: Specify Position Extraction Algorithm (8-12h)**
   - Research current observation encoding format
   - Design extraction algorithm for each substrate type:
     - Grid1D: scalar → position
     - Grid2D: one-hot → (x, y)
     - Grid3D: one-hot → (x, y, z)
     - Hex: ??? → (q, r)
     - ND: one-hot → position vector
     - Graph: ??? → node_id
   - Implement `ObservationBuilder.extract_position(state, substrate_type)`
   - Test all 6 substrate types
   - **Deliverable**: Complete algorithm spec document + implementation + tests
   - **Blocks**: Step 0.3 (replay buffer) cannot proceed without this

2. **Add Step 0.6: Database Schema Migration (2-3h)**
   - Write SQL migration script for replay_buffer table
   - Add valid_actions BLOB column
   - Handle NULL values for old transitions
   - Test migration on existing databases
   - **Deliverable**: Migration script + tests
   - **Blocks**: Cannot resume training after schema change

3. **Choose Graph Layout Algorithm (Decision + Time Adjust)**
   - **Recommend**: Circular layout for MVP (5-6h)
   - **Future**: Add force-directed as enhancement
   - Adjust Step 2.1 estimate accordingly
   - **Deliverable**: Decision documented in plan
   - **Blocks**: Cannot estimate frontend accurately

4. **Add WebSocket Protocol Specification (1-2h)**
   - Document message types (topology, state, action_mask)
   - Define position encoding per substrate
   - Provide examples
   - **Deliverable**: Protocol spec document
   - **Blocks**: Frontend cannot implement

**Required Changes (HIGH PRIORITY - Should Do Before Implementation)**:

5. **Add Checkpoint Action Dim Validation (30min)**
   - Add validation in checkpoint loading
   - Clear error message on mismatch
   - Add to Step 0.2
   - **Prevents**: Confusing crashes when loading incompatible checkpoints

6. **Add Graph Config Validation (30min)**
   - Validate edge node IDs in bounds
   - Check for duplicates
   - Warn if disconnected
   - Add to Step 1.5
   - **Prevents**: Runtime crashes from invalid configs

7. **Add Unit Test for Masked Epsilon-Greedy (30min)**
   - Test that invalid actions never selected
   - Run 100 iterations with high epsilon
   - Add to Step 1.4
   - **Prevents**: Subtle masking bugs

8. **Integrate Debugging Tools into Training Loop (1h)**
   - Make ActionMaskingValidator integration non-optional during dev
   - Add validation every N episodes
   - Add to Step 1.4b
   - **Prevents**: Wasted time debugging without tools

**Recommended Changes (Should Do)**:

9. Expand infrastructure regression tests (add ND, Continuous) - 30min
10. Add feature flag validation (prevent misconfiguration) - 15min
11. Add graph performance benchmark (100-node limit) - 30min
12. Add intermediate git commits (after each step) - 0h (process change)

---

## Cross-Phase Analysis

### Issue CPA.1: Action Mapping Dependency Unclear

**Problem**: Both phases mention action mapping:
- **Phase 9** Task 9.1 (line 780): "1D will surface hardcoded 2D assumptions" in action mapping
- **Phase 10** Step 0.1 (line 183): "Refactor action mapping" to delegate to substrates

**Question**: Are these the same changes?

**Analysis**:
- Phase 9 implementation might partially fix action mapping for 1D
- Phase 10 needs complete refactor for Graph (variable action spaces)
- **Risk**: Phase 9 fixes might conflict with Phase 10 refactor

**Recommendation**:
- **Phase 9**: Keep action mapping changes minimal (just fix 1D-specific issues)
- **Phase 10**: Do complete refactor (delegate to substrates)
- **Document**: Phase 10 Step 0.1 should note "builds on Phase 9 fixes"
- **Alternative**: Move ALL action mapping refactor to Phase 10 Pre-Work (before Graph)
  - Phase 9 works around action mapping (hardcode 1D/Hex separately)
  - Phase 10 cleans up properly
  - **Cleaner separation** but delays Phase 9 simplification benefits

**Impact**: Low-Medium - manageable with clear scoping

### Issue CPA.2: Frontend Integration Between Phases

**Problem**: Both phases modify frontend:
- **Phase 9** Task 9.4: Add Hex/1D visualization components
- **Phase 10** Task 10.2: Add Graph visualization + action masking overlay

**Risk**: Changes might conflict if not coordinated

**Analysis**:
- Both modify `frontend/src/App.vue` (substrate type switching)
- Both modify `frontend/src/websocket.js` (position handling)
- **Potential conflict**: Substrate detection logic

**Recommendation**:
- **Phase 9**: Add substrate type detection framework (extensible)
  ```javascript
  // App.vue - Phase 9
  computed: {
    substrateType() {
      // Detect from config or WebSocket
      return this.config.substrate.type;  // 'grid1d', 'hexgrid', etc.
    }
  }

  // Use v-if switching
  <Grid1DVisualization v-if="substrateType === 'grid1d'" />
  <HexGridVisualization v-if="substrateType === 'hexgrid'" />
  <GridVisualization v-else />  // Fallback for 2D/3D/ND
  ```

- **Phase 10**: Extend framework (don't replace)
  ```javascript
  // App.vue - Phase 10 (adds to existing)
  <GraphVisualization v-if="substrateType === 'graph'" />
  <ActionMaskingOverlay v-if="config.training.enable_action_masking" />
  ```

- **Git strategy**: Phase 9 creates extensible pattern, Phase 10 follows it

**Impact**: Low - easily managed with clear interface

### Issue CPA.3: Checkpoint Compatibility Phase 9 → Phase 10

**Problem**: Phase 9 doesn't mention checkpoint versioning, Phase 10 adds it

**Questions**:
- Do Phase 9 changes affect observation_dim? (1D vs 2D vs Hex have different dims)
- Can Phase 9 checkpoints load into Phase 10 code?
- What about replay buffers?

**Analysis**:
- **Observation dim changes**: YES
  - 1D: position_dim=1 → different obs_dim
  - Hex: position_dim=2 but different grid encoding
  - **Each substrate has different obs_dim**
- **Checkpoints**: Substrate-specific anyway (can't load Grid2D checkpoint into Hex)
- **Replay buffers**: Phase 10 adds valid_actions field

**Recommendation**:
- **Phase 9**: Checkpoints are substrate-specific (document this)
  - L0_1D_line checkpoints only compatible with 1D substrate
  - L1_hex_strategy checkpoints only compatible with Hex substrate
  - No cross-substrate loading needed

- **Phase 10**: Replay buffer schema change is breaking
  - Old checkpoints (Phase 9 or earlier) won't have valid_actions
  - Migration (Step 0.5) handles this: adds valid_actions=None for old transitions
  - **This is already addressed in Phase 10 plan** ✓

**Impact**: Very Low - already handled, just needs documentation

### Issue CPA.4: Infrastructure Changes Affect Both Phases

**Problem**: Phase 10 infrastructure changes (dynamic action_dim, replay buffer schema) affect Phase 9 substrates retroactively

**Questions**:
- Should Phase 9 be implemented before or after Phase 10 infrastructure?
- What if infrastructure breaks Phase 9 substrates?

**Analysis**:
- **Current plan**: Phase 9 → Phase 10 (sequential)
- **Infrastructure refactor** (Phase 10 Task 0) affects ALL substrates
- **Risk**: Phase 10 infrastructure changes might require Phase 9 rework

**Recommendation**:
- **Keep sequential order** (Phase 9 → Phase 10) but:
  - Phase 10 Step 0.4 (infrastructure testing) MUST test Phase 9 substrates (1D, Hex)
  - Explicit regression requirement: "Verify 1D and Hex still work after infrastructure changes"
  - Budget 1-2h for Phase 9 fixes if infrastructure breaks something

- **Alternative approach** (NOT recommended):
  - Do Phase 10 Task 0 (infrastructure) first
  - Then do Phase 9 (Hex + 1D)
  - Then do Phase 10 Task 1-3 (Graph + frontend)
  - **Downside**: Delays Phase 9 benefits, violates dependency order

**Impact**: Low - Step 0.4 already includes regression testing

### Issue CPA.5: Documentation Coherence

**Problem**: Two separate documentation updates (Phase 9 Step 3.1, Phase 10 Step 3.1)

**Risk**: Inconsistent terminology, duplicate examples, missing cross-references

**Recommendation**:
- **Phase 9**: Document Hex + 1D substrates, mention "Graph coming in Phase 10"
- **Phase 10**: Document Graph + infrastructure, add cross-references to Phase 9 substrates
- **After both complete**: Unified substrate comparison table
  ```markdown
  | Substrate | Dimensions | Action Space | Use Cases |
  |-----------|------------|--------------|-----------|
  | Grid1D    | 1          | 3            | Conveyor belts, sequences |
  | Grid2D    | 2          | 5            | Standard RL environments |
  | Grid3D    | 3          | 7            | Voxel worlds |
  | Hex       | 2 (axial)  | 7            | Strategy games |
  | Graph     | Variable   | Variable     | Networks, transit |
  | ND        | N          | 2N+1         | High-dimensional spaces |
  ```

**Impact**: Very Low - documentation cleanup after implementation

### Summary of Cross-Phase Issues

| Issue | Severity | Resolution | Time Impact |
|-------|----------|------------|-------------|
| CPA.1: Action mapping dependency | Medium | Document scope clearly, Phase 10 builds on Phase 9 | 0h (doc only) |
| CPA.2: Frontend integration | Low | Phase 9 creates extensible pattern, Phase 10 extends | 0h (already planned) |
| CPA.3: Checkpoint compatibility | Very Low | Already handled by versioning, just document | 0h (doc only) |
| CPA.4: Infrastructure affects both | Low | Phase 10 Step 0.4 tests Phase 9 substrates | +1-2h buffer |
| CPA.5: Documentation coherence | Very Low | Add cross-references, unified table after both | 0h (post-impl cleanup) |

**Overall Cross-Phase Risk**: **LOW** - Issues are manageable with clear documentation

---

## Revised Estimates (If Gaps Addressed)

### Phase 9: 22-29h (from 20-26h)

**Changes**:
- Add Step 0.1: Audit 2D assumptions (+1-2h)
- Add Step 4.0: WebSocket protocol spec (+1h)
- Increase Step 1.2 (1D implementation) by +1h (debugging 2D assumptions)
- Increase Step 4.1 (Hex visualization) by +1h (coordinate math)
- Increase Step 4.3 (Frontend integration) by +1h (protocol work)
- Increase Step 3.2 (Regression testing) by +1h (may find issues)

**Breakdown**:
- **Minimum**: 22h (if 2D assumptions minimal, regression clean, hex math straightforward)
- **Expected**: 25h (some 2D fixes, minor regression issues)
- **Maximum**: 29h (widespread 2D assumptions, hex wrapping issues, frontend debugging)

**Recommendation**: Budget **24-26h** (midpoint with small buffer)

### Phase 10: 105-125h (from 90-120h)

**Changes**:
- Increase Pre-Work to 8-12h (from 4-6h) - position extraction underestimated
- Add Step 0.6: Database migration (+2-3h)
- Add checkpoint validation (+30min to Step 0.2)
- Add WebSocket protocol spec (+1-2h to Step 2.3)
- Increase frontend estimate (+4-6h total) if force-directed layout chosen
- Add regression test expansion (+30min to Step 0.4)

**Breakdown (assuming circular layout for frontend)**:
- **Minimum**: 100h (position extraction goes smoothly, no major issues)
- **Expected**: 110h (some complexity in extraction, minor debugging)
- **Maximum**: 125h (position extraction complex, graph performance issues, frontend bugs)

**Breakdown (if force-directed layout chosen)**:
- **Minimum**: 105h
- **Expected**: 115h
- **Maximum**: 135h

**Recommendation**:
- **Circular layout**: Budget **105-120h**
- **Force-directed layout**: Budget **110-130h**
- **Suggest**: Choose circular layout for MVP, document force-directed as future enhancement

### Combined Phases: 127-155h (from 110-146h)

**Best case**: 122h (everything goes smoothly)
**Expected case**: 135h (normal debugging)
**Worst case**: 154h (multiple complications)

**Recommendation**: Budget **130-145h** total for both phases

---

## Conclusion

### Phase 9 Assessment

**Verdict**: **APPROVE WITH MINOR CHANGES**

**Readiness**: Can start after addressing:
1. Add Step 0.1: Audit 2D assumptions (1-2h)
2. Add Step 4.0: WebSocket protocol spec (1h)
3. Clarify hex wrapping implementation or remove wrapping option
4. Add action_space_size validation to unit tests

**Confidence**: **MEDIUM-HIGH** (70%)
- Estimates are reasonable with small buffer
- Execution path is clear
- Risks are manageable
- Main uncertainty: extent of 2D hardcoding

**Risk Level**: **MEDIUM**
- No blocking unknowns
- Can be executed with modifications
- Frontend work might overrun slightly

### Phase 10 Assessment

**Verdict**: **APPROVE WITH REQUIRED CHANGES**

**Readiness**: **CANNOT start until blocking gaps addressed**:
1. Position extraction algorithm specified (8-12h)
2. Database migration added (2-3h)
3. Graph layout algorithm chosen (decision)
4. WebSocket protocol specified (1-2h)
5. Checkpoint action_dim validation added (30min)

**Confidence**: **MEDIUM** (60%)
- Many unknowns still present
- Pre-work severely underestimated
- Frontend complexity depends on layout choice
- Main uncertainty: position extraction complexity, graph performance

**Risk Level**: **MEDIUM-HIGH**
- Multiple P0 blocking gaps
- Complex action masking integration
- Breaking infrastructure changes
- Frontend visualization challenging

### Go/No-Go Recommendation

**Phase 9**: **GO** (after minor changes)
- Address 2D audit gap
- Specify WebSocket protocol
- Adjust time estimate to 24-26h
- **Expected outcome**: Successfully implement Hex + 1D in ~25h

**Phase 10**: **NO-GO** (until P0 gaps addressed)
- Must complete gap-filling work first (~12-18h)
- Then can proceed with implementation
- **Recommended sequence**:
  1. Gap-filling work (12-18h):
     - Position extraction algorithm (8-12h)
     - Database migration design (2-3h)
     - WebSocket protocol spec (1-2h)
     - Layout algorithm decision (1h)
  2. Implementation (90-120h original + adjustments)
  3. Total: 102-138h

**Overall Recommendation**:
1. **Implement Phase 9 first** (24-26h) with minor modifications
2. **Complete Phase 10 gap-filling work** (12-18h) during or after Phase 9
3. **Begin Phase 10 implementation** (100-125h) after gaps addressed
4. **Total timeline**: 136-169h for both phases

This sequential approach de-risks Phase 10 and allows learning from Phase 9 implementation.

---

## Appendix: Required Pre-Implementation Work

### For Phase 9 (2-3h total)

**Document 1: 2D Assumption Audit Checklist**
```markdown
# Phase 9 Pre-Work: 2D Assumption Audit

## Locations to Check:
- [ ] src/townlet/environment/vectorized_env.py - action mapping
- [ ] src/townlet/environment/observation.py - grid encoding
- [ ] src/townlet/population/vectorized.py - position handling
- [ ] src/townlet/substrate/base.py - interface assumptions

## Known Hardcodes to Fix:
1. [location] - [description] - [fix approach]
2. ...

## Estimated Fix Time: [X-Y hours]
```

**Document 2: WebSocket Protocol Specification**
```markdown
# Frontend WebSocket Protocol (Phase 9 + 10)

## Message Types

### 1. Configuration (sent on connect)
{
  "type": "config",
  "substrate": {
    "type": "grid1d" | "grid2d" | "hexgrid" | "graph",
    ...substrate-specific fields
  }
}

### 2. State Update (sent every frame)
{
  "type": "state",
  "agents": [
    {
      "id": 0,
      "position": <varies by substrate>,
      "meters": {...}
    }
  ]
}

## Position Encoding by Substrate
- grid1d: number (scalar)
- grid2d: [x, y]
- hexgrid: [q, r]
- graph: number (node_id)
```

### For Phase 10 (12-18h total)

**Document 3: Position Extraction Algorithm Specification**
```markdown
# Position Extraction from Observation State

## Background
Observation state is flattened vector containing:
- Grid encoding (varies by substrate)
- Meters (8 values)
- Affordance encoding (15 values)
- Temporal extras (4 values)

## Challenge
Need to extract position component to call substrate.get_valid_actions(position)

## Solution per Substrate

### Grid2D
- Encoding: One-hot [64] for 8×8 grid
- Extraction:
  [detailed algorithm]

### Hex
- Encoding: [TO BE DETERMINED - requires code research]
- Extraction:
  [detailed algorithm]

[... etc for all substrates]

## Implementation
- [ ] ObservationBuilder.extract_position(state, substrate_type)
- [ ] Unit tests for all 6 substrate types
- [ ] Round-trip validation (encode → extract → matches original)
```

**Document 4: Database Migration Script**
```python
# src/townlet/training/migrations/v1_to_v2.py

def migrate_replay_buffer_schema(db_path: str):
    """Add valid_actions column to replay buffer."""
    # [complete implementation]

def test_migration():
    """Test migration on sample database."""
    # [complete test]
```

**Document 5: Graph Layout Algorithm Decision**
```markdown
# Graph Visualization Layout Algorithm

## Options Evaluated:
1. Circular layout - [pros/cons]
2. Force-directed - [pros/cons]
3. Manual placement - [pros/cons]

## Decision: [CHOICE]

## Implementation Approach:
[detailed approach]

## Time Estimate: [X-Y hours]
```

**Document 6: Complete WebSocket Protocol Extension**
```markdown
# WebSocket Protocol Extension for Graph Substrate

## Additional Message Types

### Graph Topology (sent on connect for graph substrate)
{
  "type": "graph_topology",
  "num_nodes": 16,
  "edges": [[0,1], [1,2], ...],
  "layout": "circular"
}

[... complete protocol]
```

---

**End of Peer Review**

**Next Steps**:
1. Review these findings with implementation team
2. Address P0 blocking gaps for Phase 10
3. Make minor modifications to Phase 9
4. Adjust time budgets accordingly
5. Proceed with implementation in sequence: Phase 9 → Phase 10 gap-filling → Phase 10 implementation

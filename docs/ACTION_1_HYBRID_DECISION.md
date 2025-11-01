# ACTION #1: Hybrid Approach Decision

**Date**: November 1, 2025  
**Status**: APPROVED  
**Timeline**: 2-3 weeks (Weeks 1-3 of 9-week plan)

---

## Executive Decision

**Use SOFTWARE_DEFINED_WORLD.md as structural template, implement with our validated mathematics.**

### What This Means

✅ **SDW Structure** (the framework):
- 4-file YAML organization: `bars.yaml`, `cascades.yaml`, `affordances.yaml`, `cues.yaml`
- Clean separation of concerns (ontology, physics, actions, communication)
- Module B ready (World Model can learn from config)

✅ **Our Math** (the proven content):
- Gradient penalties (smooth, proportional consequences)
- 30% thresholds (validated with 275 passing tests)
- Current cascade strengths (tested over thousands of episodes)

### Why This Is The Right Call

1. **Zero Risk**: Keep all proven math → zero behavioral change → all 275 tests pass
2. **SDW Compliance**: Use approved structure → Module B prerequisite satisfied
3. **Best of Both**: Clean architecture + validated gameplay
4. **Teaching Value**: Can add SDW's multiplier approach later for experiments

---

## The Two Designs Compared

### SDW Spec (SOFTWARE_DEFINED_WORLD.md)
```yaml
# Multiplier approach (binary threshold)
cascades:
  - source: satiation
    target: health
    threshold: 0.2  # 20%
    modify_depletion_rate: 1.5  # multiply by 1.5x when below threshold
```

### Our Validated Math (275 tests)
```yaml
# Gradient penalty approach (smooth proportional)
cascades:
  - source: satiation
    target: health
    threshold: 0.3  # 30%
    strength: 0.004  # penalty = strength * deficit
```

### Key Differences

| Aspect | SDW | Our Math | Hybrid Decision |
|--------|-----|----------|-----------------|
| **File Structure** | 4 files | Flexible | ✅ Use SDW (4 files) |
| **Cascade Math** | Multipliers | Gradient penalties | ✅ Use ours (gradient) |
| **Threshold** | 20% | 30% | ✅ Use ours (30%) |
| **Gameplay** | Binary cliffs | Smooth slopes | ✅ Keep smooth |
| **Validation** | Theoretical | 275 tests | ✅ Keep tested |

---

## What We're Building

### File 1: `configs/bars.yaml`
```yaml
# Meter definitions & base depletion rates
# This is the "what exists" ontology

bars:
  - name: "energy"
    index: 0
    tier: "pivotal"  # SDW terminology
    base_depletion: 0.005  # Our tested value
    
  - name: "health"
    index: 6
    tier: "pivotal"
    base_depletion: 0.001  # Our tested value
    
  # ... 6 more meters

terminal_conditions:
  - meter: "health"
    operator: "<="
    value: 0.0
  - meter: "energy"
    operator: "<="
    value: 0.0
```

### File 2: `configs/cascades.yaml`
```yaml
# Threshold-based cascade effects
# This is the "how they interact" physics

modulations:
  - name: "fitness_health_modulation"
    source: "fitness"
    target: "health"
    base_multiplier: 0.5  # Our tested value
    range: 2.5  # Our tested value

cascades:
  - name: "satiation_to_health"
    source: "satiation"
    target: "health"
    threshold: 0.3  # Our tested value (not SDW's 0.2)
    strength: 0.004  # Our tested value (gradient, not multiplier)
  
  # ... 9 more cascades
```

### File 3: `configs/affordances.yaml`
```yaml
# Already exists! No changes needed for ACTION #1
# This is ACTION #12 territory
```

### File 4: `configs/cues.yaml`
```yaml
# Social tells for Module C (future)
# Not needed for ACTION #1
# Will be important for Level 4+ multi-agent
```

---

## Implementation Strategy

### Week 1: Core Implementation (Days 1-5)

**Days 1-2: Schema & Validation**
- Create `bars.yaml` and `cascades.yaml`
- Create Pydantic models for type safety
- Write config loader with validation
- Test: Config loads without errors

**Days 3-5: CascadeEngine**
- Create `CascadeEngine` class
- Implement gradient penalty logic
- Replace hardcoded meter_dynamics.py logic
- Test: All 275 tests still pass (zero behavioral change)

### Week 2: Extension & Alternatives (Days 6-10)

**Days 6-7: Add SDW Features**
- Optional: Add multiplier support alongside gradient
- Create `configs/cascades/sdw_multiplier.yaml`
- Test: Both approaches work

**Days 8-10: Alternative Configs**
- `configs/cascades/default.yaml` - current behavior
- `configs/cascades/weak_cascades.yaml` - 50% strength (pedagogy)
- `configs/cascades/strong_cascades.yaml` - 150% strength (challenge)
- `configs/cascades/sdw_official.yaml` - 20% thresholds + multipliers
- `configs/cascades/level_3_preview.yaml` - 13 meters (future)

### Week 3: Testing & Documentation (Days 11-15)

**Days 11-12: Comprehensive Testing**
- Characterization tests (config == hardcoded)
- Performance benchmarks (<5% overhead)
- Edge case validation

**Days 13-15: Documentation**
- Student guide: "Experimenting with Cascade Physics"
- Update AGENTS.md with new architecture
- Create CASCADE_CONFIG_GUIDE.md

---

## Success Criteria

### Must Have (Week 1)
✅ Config-driven system produces **identical behavior** to hardcoded  
✅ All 275 tests pass with zero regressions  
✅ SDW-compliant 4-file structure  
✅ Type-safe config loading with validation  

### Should Have (Week 2)
✅ 5 alternative configs for pedagogical experiments  
✅ Support for both gradient and multiplier approaches  
✅ Performance <5% overhead vs hardcoded  

### Nice to Have (Week 3)
✅ Student-friendly documentation  
✅ Config validation with helpful error messages  
✅ Examples showing how to add new meters  

---

## Benefits of Hybrid Approach

1. **Zero Risk**: Keep 275 passing tests → no behavior changes → safe refactor
2. **SDW Compliance**: Module B prerequisite satisfied → moonshot ready
3. **Pedagogical Richness**: Students can experiment with both math models
4. **Future Proof**: Clean structure ready for Level 3+ complexity
5. **Teaching Moments**: Comparing gradient vs multiplier shows design trade-offs

---

## Documentation Trail

- **ACTION_1_DESIGN.md**: Full design specification (570+ lines)
- **ACTION_1_EVALUATION.md**: SDW comparison analysis (450+ lines)
- **SOFTWARE_DEFINED_WORLD.md**: Original approved spec (290 lines)
- **CLEANUP_ROADMAP.md**: Overall project tracking
- **This Document**: Quick reference for implementation

---

## Next Steps

1. ✅ Design approved (this document)
2. ⏳ Begin Week 1 Day 1: Create `bars.yaml`
3. ⏳ Begin Week 1 Day 1: Create `cascades.yaml`
4. ⏳ Begin Week 1 Day 2: Create Pydantic models
5. ⏳ Continue through implementation plan...

**Ready to start implementation! 🚀**

# Woven Mind Assessment

*Investigated: 2026-01-16*
*By: Claude (cognitive memory session)*

## Overview

Woven Mind is a dual-process cognitive architecture implementing System 1 (fast) and System 2 (slow) thinking with automatic mode switching based on surprise detection.

## Architecture

```
WovenMind (facade)
    ├── Loom (mode switching)
    │   ├── SurpriseDetector (prediction error monitoring)
    │   └── ModeController (FAST/SLOW selection)
    ├── Hive (System 1 - Hebbian pattern matching)
    ├── Cortex (System 2 - deliberate abstraction)
    └── ConsolidationEngine ("sleep" - pattern transfer)
```

## Component Status

| Component | Status | Evidence |
|-----------|--------|----------|
| WovenMind instantiation | **Works** | Creates successfully, returns valid stats |
| Training (Hive) | **Works** | `mind.train(text)` populates Hive nodes |
| Surprise detection | **Works** | Novel tokens → surprise=1.0, triggers SLOW |
| Mode switching | **Works** | Transitions recorded, FAST→SLOW on surprise |
| Consolidation pattern tracking | **Fragile** | Activations empty on repeated processing |
| Consolidation pattern transfer | **Untested** | Never triggered in experiments |

## Verified Behavior

### Surprise Detection Works
```python
mind = WovenMind()
mind.train("the quick brown fox")
result = mind.process(["completely", "novel", "tokens"])
# Result: mode=SLOW, surprise=1.0, source=cortex
```

### Mode Switching Works
```python
transitions = mind.get_transition_history()
# Shows FAST → SLOW transition with trigger=SURPRISE
```

### Consolidation Has Issues
```python
# Same pattern processed 3 times:
# Iteration 1: activations={'networks', 'process', 'neural'}
# Iteration 2: activations=set()  # Empty!
# Iteration 3: activations=set()  # Empty!
# Pattern frequency stays at 1, never triggers transfer
```

## Key Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `surprise_threshold` | 0.3 | Above this → SLOW mode |
| `confidence_threshold` | 0.6 | Below this → SLOW mode |
| `transfer_threshold` | 3 | Pattern must be seen N times for consolidation |
| `decay_factor` | 0.9 | How much to decay unused connections |

## Integration Opportunities

### High Value (Works Now)
1. **Surprise detection for drift warning**
   - Map surprise levels to cognitive states
   - surprise < 0.2 → FOCUSED
   - 0.2 < surprise < 0.3 → DRIFTING
   - surprise > 0.3 → trigger recover()

2. **Mode switching for recovery trigger**
   - TransitionTrigger.SURPRISE → auto-recover()
   - TransitionTrigger.CONFIDENCE_LOW → checkpoint

### Medium Value (Needs Work)
3. **Consolidation scheduler**
   - The pattern is good (periodic background process)
   - Implementation has bugs with pattern tracking

### Low Value (Incomplete)
4. **Pattern transfer Hive→Cortex**
   - Never successfully triggered in tests
   - May need debugging or redesign

## Recommendations

1. **Use the concepts, not necessarily the code**
   - Surprise detection concept is sound
   - Could implement simpler version in CognitiveMemory

2. **Loom is the most valuable component**
   - SurpriseDetector algorithm is solid
   - ModeController logic is clean
   - Could import directly or adapt

3. **Skip Consolidation for now**
   - Pattern tracking has edge cases
   - Our CEL compaction serves similar purpose

4. **Open Questions Answered**
   - Q1 (detect daydreaming): Use surprise levels as early warning
   - Q2 (periodic health check): Adapt scheduler pattern
   - Q3 (concurrent intents): Not addressed by Woven Mind

## Files Read

- `cortical/reasoning/woven_mind.py` - Facade (405 lines)
- `cortical/reasoning/loom.py` - Mode switching (1121 lines)
- `cortical/reasoning/consolidation.py` - Sleep cycle (635 lines)
- `cortical/reasoning/loom_hive.py` - Not read
- `cortical/reasoning/loom_cortex.py` - Not read

## Next Steps (Decision Point)

Options:
1. Integrate Loom's SurpriseDetector into CognitiveMemory
2. Implement simpler surprise-based drift detection ourselves
3. Explore other reasoning tools (PRISM, QAPV, etc.)
4. Something else entirely

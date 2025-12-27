# EPIC: Cognitive NLU/NLG - Knowledge Base

**Epic ID:** EPIC-cognitive-nlu-nlg
**Created:** 2025-12-27
**Status:** Active
**Handoff:** H-20251227-154444-f2e718d0

---

## Vision

Build systems that exhibit **meta-learning** - the ability to improve their own learning process through experience. This goes beyond pattern matching to genuine understanding and generation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LEARNING TO LEARN AND DO                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LEVEL 0: Pattern Matching (current)                                 │
│  └── Recognize patterns seen before                                  │
│                                                                       │
│  LEVEL 1: Pattern Generalization (Sprint 1)                          │
│  └── Apply patterns to novel situations with confidence              │
│                                                                       │
│  LEVEL 2: Knowledge Transfer (Sprint 2)                              │
│  └── Transfer learning across domains via analogy                    │
│                                                                       │
│  LEVEL 3: Self-Assessment (Sprint 3)                                 │
│  └── Verify understanding through generation                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria

| Criteria | Metric | How to Verify |
|----------|--------|---------------|
| Explanation capability | PLN can answer "why?" queries | Demo script with inference trace |
| Confidence tracking | All outputs include uncertainty | WovenMind.process() returns confidence |
| Analogical transfer | Cross-domain inference works | Test cases show domain transfer |
| Self-assessment | System identifies knowledge gaps | Metacognitive queries work |
| Generative verification | Generate → verify loop works | Demo shows understanding verification |
| Test coverage | ≥95% for new code | `coverage report --include="cortical/*"` |
| Working demo | Each sprint has runnable demo | `scripts/cognitive_integration_demo.py` |

---

## Sprint Overview

| Sprint | Focus | Deliverable | Independence |
|--------|-------|-------------|--------------|
| **Sprint 1** | Explanation & Confidence | PLN explain(), WovenMind confidence | Standalone - enhances existing inference |
| **Sprint 2** | Analogical Transfer | Domain mapper, metacognitive monitor | Standalone - adds transfer learning |
| **Sprint 3** | Generative Understanding | Verify-through-generation loop | Standalone - adds self-assessment |

Each sprint produces a **working demo** that can be used independently.

---

## Part I: Sub-Agent Guardrails

### Prime Directives

Every agent working on this epic MUST follow these rules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRIME DIRECTIVES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. READ BEFORE WRITE                                                │
│     Never modify code you haven't read first.                        │
│     Use the Read tool on target files before editing.                │
│                                                                       │
│  2. TDD: TESTS FIRST                                                 │
│     Write failing tests BEFORE implementation.                       │
│     RED → GREEN → REFACTOR cycle is mandatory.                       │
│                                                                       │
│  3. TESTS IN CORRECT LOCATION                                        │
│     Unit tests → tests/unit/test_<module>.py                         │
│     Integration → tests/integration/test_<feature>_integration.py    │
│     Performance → tests/performance/test_<feature>_perf.py           │
│     NEVER put benchmarks in unit tests.                              │
│                                                                       │
│  4. CHECK GoT BEFORE STARTING                                        │
│     Run: python scripts/got_utils.py task show T-XXXXX               │
│     Check for blockers and dependencies.                             │
│                                                                       │
│  5. VERIFY BEFORE REPORTING SUCCESS                                  │
│     Run the verification command in your task.                       │
│     Check git status to confirm changes persisted.                   │
│                                                                       │
│  6. DOCUMENT DECISIONS                                               │
│     Log significant choices to GoT.                                  │
│     python scripts/got_utils.py decision log "..." --rationale       │
│                                                                       │
│  7. ESCALATE UNCERTAINTY                                             │
│     If unsure, stop and report back.                                 │
│     Better to ask than to guess wrong.                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Test Location Rules

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEST LOCATION REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Test Type           │ Location                      │ Markers       │
│  ────────────────────┼───────────────────────────────┼───────────────│
│  Unit tests          │ tests/unit/test_*.py          │ (none)        │
│  Integration tests   │ tests/integration/test_*.py   │ (none)        │
│  Performance tests   │ tests/performance/test_*.py   │ @pytest.mark.slow │
│  Behavioral tests    │ tests/behavioral/test_*.py    │ (none)        │
│  Regression tests    │ tests/regression/test_*.py    │ (none)        │
│                                                                       │
│  ❌ ANTI-PATTERNS:                                                   │
│  - Putting time.sleep() tests in unit/ (use @pytest.mark.slow)      │
│  - Putting benchmarks in unit/ (use performance/)                   │
│  - Putting multi-component tests in unit/ (use integration/)        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### File Modification Rules

| Category | Files | Rules |
|----------|-------|-------|
| **ALLOWED** | Files in your task scope | Modify freely |
| **ALLOWED** | `tests/unit/test_<your_module>.py` | Create/modify for your feature |
| **ALLOWED** | `cortical/reasoning/__init__.py` | Add exports only |
| **REQUIRES PERMISSION** | `cortical/reasoning/prism_*.py` | Existing PRISM modules |
| **REQUIRES PERMISSION** | `cortical/reasoning/woven_mind.py` | Core WovenMind |
| **REQUIRES PERMISSION** | `CLAUDE.md` | Project documentation |
| **FORBIDDEN** | `.got/*` | Use CLI only, never direct edits |
| **FORBIDDEN** | `scripts/got_utils.py` | Infrastructure |

---

## Part II: Reference Documentation

### Must-Read Before Starting

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `docs/definition-of-done.md` | What "done" means | Before completing any task |
| `docs/testing-strategy.md` | Test organization | Before writing tests |
| `docs/automated-testing-techniques.md` | TDD workflow, mocking | Before writing tests |
| `docs/woven-mind-user-guide.md` | WovenMind API | Before modifying WovenMind |
| `samples/memories/2025-12-27-handoff-cognitive-nlu-nlg-exploration.md` | Vision and ideas | Before starting any sprint |

### Key Existing Code

| Module | Purpose | Lines | Stability |
|--------|---------|-------|-----------|
| `cortical/reasoning/prism_pln.py` | Probabilistic logic | ~719 | Stable - extend, don't rewrite |
| `cortical/reasoning/woven_mind.py` | Dual-process facade | ~404 | Stable - extend carefully |
| `cortical/reasoning/loom.py` | Mode switching | ~1,115 | Stable - may need extension |
| `cortical/query/analogy.py` | Analogy completion | ~200 | Foundation for Sprint 2 |
| `cortical/reasoning/consolidation.py` | Memory transfer | ~634 | Reference for patterns |

---

## Part III: Task Template

Each task in this epic follows this structure:

```markdown
## Task: [T-YYYYMMDD-HHMMSS-XXXXXXXX]

### Goal (WHY)
[One sentence describing what problem this solves]

### Interface Contract (WHAT)
```python
# Function signatures with types
def new_function(param: Type) -> ReturnType:
    """Brief description."""
    pass
```

### Integration Points (WHERE)
- Extends: `module.Class` or `module.function`
- Used by: [What will call this]
- Depends on: [What this calls]

### Test Requirements (HOW TO VERIFY)
**Location:** `tests/unit/test_<module>.py`

```python
# Specific test cases that must pass
def test_basic_functionality():
    """Must pass for task to be complete."""
    pass

def test_edge_case_empty_input():
    """Handle empty input gracefully."""
    pass
```

**Verification command:**
```bash
python -m pytest tests/unit/test_<module>.py -v
python -m coverage run -m pytest tests/ && python -m coverage report --include="cortical/*"
```

### Anti-Patterns (WHAT NOT TO DO)
- ❌ Don't [specific mistake to avoid]
- ❌ Don't [another common mistake]
- ❌ Don't [third thing to avoid]

### Freedom Zone (IMPLEMENTOR'S CHOICE)
- ✅ Algorithm details are up to you
- ✅ Internal helper functions as needed
- ✅ Error message wording
- ✅ Logging verbosity

### Exploration Notes (IF EXPLORATION TASK)
Before implementing, investigate:
1. [Question to answer]
2. [Thing to research]
3. [Pattern to find]

Report findings before proceeding to implementation.
```

---

## Part IV: Recovery Procedures

### If Confused About Context

```bash
# 1. Check current sprint and task status
python scripts/got_utils.py sprint status
python scripts/got_utils.py task show T-XXXXX

# 2. Read the handoff
cat samples/memories/2025-12-27-handoff-cognitive-nlu-nlg-exploration.md

# 3. Check this knowledge base
cat docs/epic-cognitive-nlu-nlg-knowledge-base.md

# 4. Run the demo to see current state
python scripts/cognitive_integration_demo.py --load ./cognitive_state --verbose
```

### If Tests Fail

```bash
# 1. Check what's failing
python -m pytest tests/ -v --tb=short 2>&1 | head -50

# 2. Check if it's your change or pre-existing
git stash && python -m pytest tests/ -q && git stash pop

# 3. If pre-existing, document and continue
# If your change, fix before proceeding
```

### If Blocked by Another Task

```bash
# 1. Document the blocker
python scripts/got_utils.py task update T-XXXXX --blocked-by T-YYYYY

# 2. Check if blocker is in progress
python scripts/got_utils.py task show T-YYYYY

# 3. Work on non-blocked tasks or escalate
```

---

## Part V: Demo Requirements

Each sprint must produce a runnable demo. The demo should:

1. **Be self-contained** - Run with a single command
2. **Show the new capability** - Not just "it doesn't crash"
3. **Include verification** - Assert expected behavior
4. **Be documented** - Comments explain what's happening

### Demo Template

```python
#!/usr/bin/env python
"""
Sprint N Demo: [Capability Name]

This demo shows:
1. [First capability]
2. [Second capability]
3. [Third capability]

Run with: python scripts/sprint_N_demo.py
"""

def main():
    print("=" * 60)
    print("SPRINT N DEMO: [CAPABILITY NAME]")
    print("=" * 60)

    # Setup
    print("\n1. Setting up...")
    # ... setup code ...

    # Demonstrate capability 1
    print("\n2. Demonstrating [capability 1]...")
    # ... demo code ...
    assert result.expected_property, "Capability 1 verification failed"
    print("   ✓ [Capability 1] working")

    # Demonstrate capability 2
    print("\n3. Demonstrating [capability 2]...")
    # ... demo code ...
    assert result.expected_property, "Capability 2 verification failed"
    print("   ✓ [Capability 2] working")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE - All capabilities verified")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Part VI: GoT Commands Quick Reference

```bash
# Task management
python scripts/got_utils.py task show T-XXXXX          # View task details
python scripts/got_utils.py task start T-XXXXX         # Mark as in_progress
python scripts/got_utils.py task complete T-XXXXX      # Mark as complete
python scripts/got_utils.py task update T-XXXXX --notes "..."  # Add notes

# Decision logging
python scripts/got_utils.py decision log "Decision" --rationale "Why"

# Sprint status
python scripts/got_utils.py sprint status              # Current sprint
python scripts/got_utils.py dashboard                  # Full overview

# Handoff management
python scripts/got_utils.py handoff list               # All handoffs
python scripts/got_utils.py handoff accept H-XXX --agent agent-name

# Queries
python scripts/got_utils.py query "what blocks T-XXXXX"
python scripts/got_utils.py query "what depends on T-XXXXX"
```

---

## Appendix: Existing Test Patterns

### Unit Test Example (PLN)

```python
# tests/unit/test_prism_pln.py
class TestTruthValue:
    def test_revision_combines_evidence(self):
        tv1 = TruthValue(strength=0.8, confidence=0.5)
        tv2 = TruthValue(strength=0.6, confidence=0.3)
        revised = tv1.revise(tv2)
        assert 0.6 < revised.strength < 0.8
        assert revised.confidence > max(tv1.confidence, tv2.confidence)
```

### Integration Test Example (WovenMind)

```python
# tests/integration/test_woven_mind_integration.py
class TestWovenMindIntegration:
    def test_full_processing_cycle(self):
        mind = WovenMind()
        mind.train("neural networks process data")
        result = mind.process(["neural", "networks"])
        assert result.mode in (ThinkingMode.FAST, ThinkingMode.SLOW)
        assert len(result.activations) > 0
```

### Performance Test Example

```python
# tests/performance/test_inference_perf.py
import time
import pytest

@pytest.mark.slow
class TestInferencePerformance:
    def test_inference_under_100ms(self, loaded_pln):
        start = time.time()
        for _ in range(100):
            loaded_pln.query("test_query")
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.1, f"Inference too slow: {elapsed:.3f}s"
```

---

*This knowledge base should be consulted before starting any task in EPIC-cognitive-nlu-nlg.*

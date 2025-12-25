# Task Knowledge Base: Woven Mind + PRISM Marriage

## Sub-Agent Working Guidelines

---

> *"Know the rules well, so you can break them effectively."*
> — Dalai Lama (but we prefer you follow them)

---

## Part I: Sub-Agent Guardrails

### 1.1 The Prime Directives

Every sub-agent working on this project MUST follow these rules:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SUB-AGENT PRIME DIRECTIVES                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. READ BEFORE WRITE                                                   │
│     Never modify code you haven't read first.                          │
│     Use the Read tool on target files before editing.                  │
│                                                                         │
│  2. TEST BEFORE COMMIT                                                  │
│     All changes must pass existing tests.                              │
│     New code must have tests written FIRST (TDD).                      │
│     Run: python -m pytest tests/ -q                                    │
│                                                                         │
│  3. STAY IN YOUR LANE                                                   │
│     Only modify files explicitly listed in your task.                  │
│     If you need to touch other files, ASK first.                       │
│                                                                         │
│  4. DOCUMENT YOUR DECISIONS                                             │
│     Log significant decisions to GoT.                                  │
│     Use: python scripts/got_utils.py decision log "..." --rationale    │
│                                                                         │
│  5. VERIFY BEFORE REPORTING SUCCESS                                     │
│     Run the verification command specified in your task.               │
│     Check git status to confirm changes persisted.                     │
│                                                                         │
│  6. ESCALATE UNCERTAINTY                                                │
│     If unsure, stop and report back.                                   │
│     Better to ask than to guess wrong.                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Modification Rules

```python
# ALLOWED: Files explicitly in your task scope
✅ cortical/reasoning/loom.py           # If assigned to Loom tasks
✅ tests/unit/test_loom.py              # Tests for your feature
✅ cortical/reasoning/__init__.py       # Adding exports only

# REQUIRES PERMISSION: Shared infrastructure
⚠️ cortical/reasoning/prism_*.py        # Existing PRISM modules
⚠️ cortical/reasoning/thought_graph.py  # Core data structures
⚠️ CLAUDE.md                            # Project documentation

# FORBIDDEN: Never touch these
❌ .git/*                                # Git internals
❌ .got/*                                # GoT state (use CLI only)
❌ scripts/got_utils.py                  # Infrastructure
❌ Other agents' task files              # Parallel work
```

### 1.3 Communication Protocol

**When starting a task:**
```bash
# 1. Mark task as in_progress
python scripts/got_utils.py task start T-XXXXX

# 2. Read the task details
python scripts/got_utils.py task show T-XXXXX

# 3. Read all referenced files before starting
```

**During work:**
```bash
# Log significant decisions
python scripts/got_utils.py decision log "Chose X over Y" \
  --rationale "Because Z provides better performance"

# If blocked, create a blocker
python scripts/got_utils.py task update T-XXXXX --blocked-by T-YYYYY
```

**When completing:**
```bash
# 1. Run verification
python -m pytest tests/unit/test_your_feature.py -v

# 2. Check coverage
python -m coverage run -m pytest tests/ && python -m coverage report

# 3. Mark complete
python scripts/got_utils.py task complete T-XXXXX

# 4. Verify git status
git status
```

### 1.4 Error Recovery Protocol

If something goes wrong:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ERROR RECOVERY TREE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Test Failure?                                                          │
│  ├── Is it YOUR test? → Fix it                                         │
│  └── Is it EXISTING test? → STOP, report, don't break others           │
│                                                                         │
│  Import Error?                                                          │
│  ├── Missing dependency? → Check cortical/reasoning/__init__.py        │
│  └── Circular import? → Refactor, use TYPE_CHECKING guard              │
│                                                                         │
│  Git Conflict?                                                          │
│  ├── Your file only? → Resolve locally                                 │
│  └── Shared file? → STOP, report, coordinate with main agent           │
│                                                                         │
│  Unsure how to proceed?                                                 │
│  └── ALWAYS: Stop, document what you know, report back                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part II: Task Knowledge Sheets

### Task T1.1: Design Loom Interface Contract

**ID:** T-20251225-230408-c8fe382b

#### Context
The Loom is the integration layer between Hebbian Hive (fast, System 1) and Cultured Cortex (slow, System 2). Before implementing, we need a clear interface contract that both systems will respect.

#### What You Need to Know

**Read these files first:**
```
docs/woven-mind-architecture.md          # The Loom concept
docs/research-prism-woven-mind-comparison.md  # How systems relate
cortical/reasoning/prism_attention.py    # UnifiedAttention (lines 1-50)
cortical/reasoning/prism_got.py          # SynapticMemoryGraph (lines 1-100)
cortical/reasoning/prism_slm.py          # PRISMLanguageModel (lines 1-80)
```

**Key concepts:**
- `ThinkingMode`: Enum with FAST (Hive) and SLOW (Cortex) values
- `SurpriseSignal`: Data class carrying surprise magnitude and context
- `ModeController`: Decides which system handles a request

**Design constraints:**
1. The Loom must be **stateless** for the routing decision (no memory of previous requests)
2. State lives in Hive and Cortex, not in Loom
3. Interface must support **async** for future parallelization
4. Must integrate with existing `UnifiedAttention` class

#### Deliverables
```
cortical/reasoning/loom.py:
  - ThinkingMode enum
  - LoomConfig dataclass (thresholds, parameters)
  - LoomInterface protocol (abstract base)
  - SurpriseSignal dataclass
```

#### Acceptance Criteria
- [ ] Interface is documented with docstrings
- [ ] Type hints on all public methods
- [ ] No implementation yet (interface only)
- [ ] Protocol can be implemented by mock for testing

#### Verification Command
```bash
python -c "from cortical.reasoning.loom import ThinkingMode, LoomConfig, SurpriseSignal; print('Interface OK')"
```

#### Guardrails
- DO NOT implement the full Loom yet - interface only
- DO NOT modify existing PRISM files
- DO add to `cortical/reasoning/__init__.py` exports

---

### Task T1.2: Implement SurpriseDetector Class

**ID:** T-20251225-230414-3170674f

#### Context
Surprise is the signal that switches from fast to slow thinking. When predictions don't match reality, we're surprised. High surprise → engage Cortex. Low surprise → stay in Hive.

#### What You Need to Know

**Read these files first:**
```
cortical/reasoning/loom.py               # Your interface from T1.1
cortical/reasoning/prism_got.py          # PredictionResult (lines 320-340)
cortical/reasoning/prism_got.py          # SynapticEdge.prediction_accuracy
docs/woven-mind-architecture.md          # Surprise concept (search "surprise")
```

**Algorithm outline:**
```python
class SurpriseDetector:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.baseline_surprise: float = 0.0
        self.history: deque = deque(maxlen=100)

    def compute_surprise(
        self,
        predicted: Dict[str, float],  # node_id → probability
        actual: Set[str]               # activated node IDs
    ) -> SurpriseSignal:
        """
        Surprise = how much actual differs from predicted.

        Uses KL divergence or simpler:
        - For each predicted node: |predicted_prob - (1 if in actual else 0)|
        - Average over all predictions
        - Normalize by baseline
        """
        ...

    def update_baseline(self, surprise: float) -> None:
        """Adapt baseline based on running average."""
        self.history.append(surprise)
        self.baseline_surprise = sum(self.history) / len(self.history)

    def should_engage_slow(self, signal: SurpriseSignal) -> bool:
        """True if surprise exceeds threshold."""
        return signal.magnitude > self.threshold
```

**Design considerations:**
1. **Baseline adaptation**: Surprise is relative. If everything is surprising, nothing is.
2. **History window**: Don't adapt too fast (100 samples) or too slow
3. **Threshold tuning**: 0.3 is a starting point, should be configurable

#### Deliverables
```
cortical/reasoning/loom.py:
  - SurpriseDetector class (full implementation)

tests/unit/test_loom.py:
  - test_surprise_zero_when_perfect_prediction
  - test_surprise_high_when_all_wrong
  - test_baseline_adapts
  - test_threshold_triggers_slow_mode
```

#### Acceptance Criteria
- [ ] SurpriseDetector fully implemented
- [ ] At least 4 unit tests
- [ ] Tests pass with ≥ 95% coverage on new code
- [ ] Docstrings on all public methods

#### Verification Command
```bash
python -m pytest tests/unit/test_loom.py -v -k "surprise"
python -m coverage run -m pytest tests/unit/test_loom.py && python -m coverage report --include="cortical/reasoning/loom.py"
```

#### Guardrails
- DO use simple surprise metric first (MSE), optimize later if needed
- DO NOT add dependencies outside stdlib
- DO handle edge cases: empty predictions, empty actual, all match

---

### Task T1.3: Implement ModeController Class

**ID:** T-20251225-230420-6ebf03e9

#### Context
The ModeController is the traffic cop of the Loom. It receives requests, checks surprise, and routes to the appropriate system (Hive or Cortex).

#### What You Need to Know

**Read these files first:**
```
cortical/reasoning/loom.py               # SurpriseDetector from T1.2
cortical/reasoning/prism_attention.py    # UnifiedAttention (full file)
cortical/reasoning/prism_got.py          # IncrementalReasoner.predict_next()
cortical/reasoning/prism_slm.py          # PRISMLanguageModel.generate()
```

**Architecture:**
```
                    ┌─────────────────┐
                    │ ModeController  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ SurpriseDetector│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        Low Surprise    Med Surprise   High Surprise
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │  HIVE   │    │ HYBRID  │    │ CORTEX  │
        │ (fast)  │    │ (both)  │    │ (slow)  │
        └─────────┘    └─────────┘    └─────────┘
```

**Key methods:**
```python
class ModeController:
    def __init__(
        self,
        surprise_detector: SurpriseDetector,
        hive: Optional["PRISMLanguageModel"] = None,
        cortex: Optional["IncrementalReasoner"] = None,
    ):
        self.detector = surprise_detector
        self.hive = hive
        self.cortex = cortex
        self.current_mode = ThinkingMode.FAST
        self._mode_history: List[Tuple[datetime, ThinkingMode]] = []

    def route(
        self,
        input_nodes: Set[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ModeDecision:
        """
        Decide which system should handle this input.

        Returns ModeDecision with:
        - mode: ThinkingMode
        - confidence: float (how sure are we?)
        - reason: str (for debugging/logging)
        """
        ...

    def execute(
        self,
        input_nodes: Set[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Full pipeline: route → execute in appropriate system → return.
        """
        decision = self.route(input_nodes, context)
        if decision.mode == ThinkingMode.FAST:
            return self._execute_hive(input_nodes, context)
        else:
            return self._execute_cortex(input_nodes, context)
```

**Design considerations:**
1. **Hybrid mode**: When surprise is medium (0.2-0.4), run BOTH systems
2. **Mode stickiness**: Don't flip-flop; add hysteresis
3. **Fallback**: If one system fails, try the other

#### Deliverables
```
cortical/reasoning/loom.py:
  - ModeDecision dataclass
  - ProcessingResult dataclass
  - ModeController class

tests/unit/test_loom.py:
  - test_low_surprise_routes_to_hive
  - test_high_surprise_routes_to_cortex
  - test_mode_history_tracked
  - test_hybrid_mode_for_medium_surprise
```

#### Acceptance Criteria
- [ ] ModeController routes correctly based on surprise
- [ ] Mode transitions are logged
- [ ] Hybrid mode works
- [ ] At least 4 unit tests pass

#### Verification Command
```bash
python -m pytest tests/unit/test_loom.py -v -k "mode"
```

#### Guardrails
- DO allow Hive and Cortex to be None (for testing in isolation)
- DO NOT block on missing systems - gracefully degrade
- DO log all mode transitions for observability

---

### Task T1.4: Wire Loom to PRISM-Attention

**ID:** T-20251225-230426-2c0bff92

#### Context
PRISM already has `UnifiedAttention` that coordinates across systems. The Loom should integrate with it, not replace it.

#### What You Need to Know

**Read these files first:**
```
cortical/reasoning/prism_attention.py    # Full file, especially UnifiedAttention
cortical/reasoning/loom.py               # ModeController from T1.3
```

**Integration approach:**
```python
# In prism_attention.py, UnifiedAttention should use Loom

class UnifiedAttention:
    def __init__(
        self,
        graph: "PRISMGraph",
        slm: Optional["PRISMLanguageModel"] = None,
        pln: Optional["PLNReasoner"] = None,
        loom: Optional["ModeController"] = None,  # NEW
    ):
        self.loom = loom or ModeController(SurpriseDetector())
        ...

    def attend(self, query: str) -> AttentionResult:
        # Get mode decision from Loom
        mode = self.loom.route(self._query_to_nodes(query))

        if mode.mode == ThinkingMode.FAST:
            # Use only synaptic attention
            return self._fast_attend(query)
        else:
            # Use full PLN + GoT reasoning
            return self._slow_attend(query)
```

**Important:** This modifies an existing PRISM file. Extra care required.

#### Deliverables
```
cortical/reasoning/prism_attention.py:
  - Add loom parameter to UnifiedAttention.__init__
  - Add mode-aware routing in attend()
  - Preserve backward compatibility (loom is optional)
```

#### Acceptance Criteria
- [ ] UnifiedAttention accepts optional Loom
- [ ] Existing tests still pass (backward compat)
- [ ] Mode affects attention behavior
- [ ] New integration test verifies routing

#### Verification Command
```bash
python -m pytest tests/unit/test_prism_attention.py -v
python -m pytest tests/unit/test_loom.py -v -k "integration"
```

#### Guardrails
- DO preserve all existing behavior when loom=None
- DO NOT change method signatures of existing public methods
- DO add new method _fast_attend() and _slow_attend() rather than modifying attend()
- DO run full test suite before committing: `python -m pytest tests/ -q`

---

### Task T1.5: Add Observable Mode Transitions

**ID:** T-20251225-230432-204115ad

#### Context
For debugging and monitoring, we need to see when the system switches modes. This is observability infrastructure.

#### What You Need to Know

**Read these files first:**
```
cortical/observability.py                # Existing metrics infrastructure
cortical/reasoning/loom.py               # ModeController from T1.3
```

**Observability requirements:**
1. **Logging**: Mode transitions should log at INFO level
2. **Metrics**: Track mode counts, transition frequency
3. **Events**: Emit events that can be subscribed to
4. **History**: Keep bounded history of recent transitions

**Integration with existing observability:**
```python
from cortical.observability import record_metric, timed

class ModeController:
    @timed("loom_route")
    def route(self, input_nodes: Set[str], ...) -> ModeDecision:
        decision = self._compute_decision(input_nodes)

        # Record metrics
        record_metric(f"loom_mode_{decision.mode.name.lower()}", 1)

        # Log transition
        if decision.mode != self.current_mode:
            self._log_transition(self.current_mode, decision.mode)
            record_metric("loom_mode_transitions", 1)

        return decision
```

#### Deliverables
```
cortical/reasoning/loom.py:
  - ModeTransitionEvent dataclass
  - Observable mode history in ModeController
  - Integration with cortical.observability

tests/unit/test_loom.py:
  - test_mode_transition_logged
  - test_metrics_recorded
  - test_history_bounded
```

#### Acceptance Criteria
- [ ] Mode transitions are logged
- [ ] Metrics are recorded via observability module
- [ ] History is bounded (configurable max size)
- [ ] Events can be retrieved for debugging

#### Verification Command
```bash
python -m pytest tests/unit/test_loom.py -v -k "observ"
python -c "
from cortical.reasoning.loom import ModeController, SurpriseDetector, ThinkingMode
mc = ModeController(SurpriseDetector())
# Simulate transitions
mc.current_mode = ThinkingMode.SLOW
mc._log_transition(ThinkingMode.FAST, ThinkingMode.SLOW)
print(f'History: {len(mc._mode_history)} entries')
print('Observability OK')
"
```

#### Guardrails
- DO use existing observability.py infrastructure
- DO NOT add external logging dependencies
- DO keep history bounded to prevent memory leaks
- DO use structured logging (not just print statements)

---

### Task T1.6: Write Unit Tests for Loom

**ID:** T-20251225-230438-256a73b4

#### Context
This task ensures comprehensive test coverage for all Loom components. Tests should be written to TDD standards - they define the expected behavior.

#### What You Need to Know

**Read these files first:**
```
tests/unit/test_prism_got.py             # Example PRISM unit tests
tests/unit/test_prism_slm.py             # More examples
cortical/reasoning/loom.py               # Implementation from T1.1-T1.5
```

**Test organization:**
```
tests/unit/test_loom.py
├── TestThinkingMode           # Enum tests
├── TestLoomConfig             # Configuration tests
├── TestSurpriseSignal         # Data class tests
├── TestSurpriseDetector       # Core surprise logic
│   ├── test_perfect_prediction_zero_surprise
│   ├── test_complete_miss_high_surprise
│   ├── test_partial_match_medium_surprise
│   ├── test_baseline_adaptation
│   ├── test_threshold_boundary
│   └── test_empty_inputs_handled
├── TestModeController         # Routing logic
│   ├── test_fast_mode_default
│   ├── test_slow_mode_on_high_surprise
│   ├── test_hybrid_mode_medium_surprise
│   ├── test_mode_history_tracked
│   ├── test_mode_stickiness
│   └── test_graceful_degradation
└── TestLoomIntegration        # End-to-end
    ├── test_full_pipeline_fast_path
    ├── test_full_pipeline_slow_path
    └── test_attention_integration
```

**Test patterns to follow:**
```python
import pytest
from unittest.mock import Mock, patch
from cortical.reasoning.loom import (
    SurpriseDetector,
    ModeController,
    ThinkingMode,
    SurpriseSignal,
)


class TestSurpriseDetector:
    """Tests for the SurpriseDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector with default settings."""
        return SurpriseDetector(threshold=0.3)

    def test_perfect_prediction_zero_surprise(self, detector):
        """When predictions perfectly match actual, surprise is zero."""
        predicted = {"node_a": 1.0, "node_b": 0.0}
        actual = {"node_a"}

        signal = detector.compute_surprise(predicted, actual)

        assert signal.magnitude == pytest.approx(0.0, abs=0.01)

    def test_complete_miss_high_surprise(self, detector):
        """When predictions completely miss, surprise is high."""
        predicted = {"node_a": 1.0}
        actual = {"node_b"}  # Completely different

        signal = detector.compute_surprise(predicted, actual)

        assert signal.magnitude > 0.5
```

#### Deliverables
```
tests/unit/test_loom.py:
  - Full test file with all classes above
  - ≥ 95% coverage on cortical/reasoning/loom.py
  - All tests pass
```

#### Acceptance Criteria
- [ ] Test file exists with proper structure
- [ ] All test classes implemented
- [ ] Coverage ≥ 95% on loom.py
- [ ] All tests pass
- [ ] Edge cases covered (empty inputs, None values, etc.)

#### Verification Command
```bash
python -m pytest tests/unit/test_loom.py -v
python -m coverage run -m pytest tests/unit/test_loom.py
python -m coverage report --include="cortical/reasoning/loom.py" --fail-under=95
```

#### Guardrails
- DO use pytest fixtures, not unittest.TestCase
- DO use mocks for Hive and Cortex (don't require full systems)
- DO test edge cases explicitly
- DO NOT skip tests - fix failing tests instead
- DO follow existing test naming conventions

---

## Part III: Sprint 2-6 Task Summaries

### Sprint 2: Hebbian Hive Enhancement

| Task ID | Title | Key Files | Main Challenge |
|---------|-------|-----------|----------------|
| T2.1 | Add lateral_inhibition() | prism_slm.py | Algorithm design |
| T2.2 | Implement k_winners_take_all() | prism_slm.py | Efficiency |
| T2.3 | Add HomeostasisRegulator | New: homeostasis.py | Novel code |
| T2.4 | Implement excitability adjustment | homeostasis.py | Integration |
| T2.5 | Add spreading_activation() | prism_slm.py | Graph traversal |
| T2.6 | Create HiveNode wrapper | prism_slm.py | Data structure |
| T2.7 | Write tests | test_prism_slm.py | Coverage |
| T2.8 | Benchmark sparsity | scripts/benchmark_*.py | Performance |

**Key guardrails for Sprint 2:**
- Lateral inhibition must produce 5-10% active nodes (sparse)
- Homeostasis must converge within 100 iterations
- Spreading activation must not explode (use decay)

---

### Sprint 3: Cortex Abstraction

| Task ID | Title | Key Files | Main Challenge |
|---------|-------|-----------|----------------|
| T3.1 | Design Abstraction class | New: abstraction.py | Interface design |
| T3.2 | Implement PatternDetector | abstraction.py | Algorithm |
| T3.3 | Add abstraction_candidates() | abstraction.py | Heuristics |
| T3.4 | Implement form_abstraction() | abstraction.py | Graph manipulation |
| T3.5 | Wire to PRISM-PLN | prism_pln.py | Integration |
| T3.6 | Add GoalStack class | New: goal_stack.py | State management |
| T3.7 | Implement goal progress | goal_stack.py | Tracking |
| T3.8-9 | Write tests | test_*.py | Coverage |

**Key guardrails for Sprint 3:**
- Abstractions must have ≥3 component patterns to form
- Goal progress must be monotonic (no regression)
- PLN truth values must propagate to abstractions

---

### Sprint 4: The Loom Weaves

| Task ID | Title | Key Files | Main Challenge |
|---------|-------|-----------|----------------|
| T4.1 | Connect Loom to Hive | loom.py, prism_slm.py | Integration |
| T4.2 | Connect Loom to Cortex | loom.py, prism_got.py | Integration |
| T4.3 | Implement surprise_from_predictions | loom.py | Algorithm |
| T4.4 | Add attention_routing | prism_attention.py | Coordination |
| T4.5 | Create WovenMind facade | New: woven_mind.py | API design |
| T4.6 | Implement process() | woven_mind.py | Orchestration |
| T4.7-8 | Tests and demo | test_*.py, examples/ | Validation |

**Key guardrails for Sprint 4:**
- WovenMind must be usable with Hive OR Cortex alone
- Mode switching must be observable in demo
- No breaking changes to existing PRISM APIs

---

### Sprint 5: Consolidation

| Task ID | Title | Key Files | Main Challenge |
|---------|-------|-----------|----------------|
| T5.1 | Design ConsolidationEngine | New: consolidation.py | Architecture |
| T5.2 | Implement pattern_transfer | consolidation.py | Algorithm |
| T5.3 | Implement abstraction_mining | consolidation.py | Novel algorithm |
| T5.4 | Add decay_cycle | consolidation.py | Timing |
| T5.5 | Create scheduler | consolidation.py | Background work |
| T5.6 | Wire to WovenMind | woven_mind.py | Integration |
| T5.7-8 | Tests and benchmark | test_*.py | Validation |

**Key guardrails for Sprint 5:**
- Consolidation must be interruptible (not blocking)
- Pattern transfer must preserve prediction accuracy
- Decay must not destroy high-value connections

---

### Sprint 6: Integration & Polish

| Task ID | Title | Key Files | Main Challenge |
|---------|-------|-----------|----------------|
| T6.1 | E2E integration tests | test_integration_*.py | Scope |
| T6.2 | Performance benchmarks | scripts/benchmark_*.py | Measurement |
| T6.3 | User documentation | docs/*.md | Clarity |
| T6.4 | Tutorial notebook | examples/*.ipynb | Pedagogy |
| T6.5 | Update CLAUDE.md | CLAUDE.md | Completeness |
| T6.6 | Comprehensive demo | examples/woven_demo.py | Showmanship |
| T6.7 | Verify coverage | - | Quality gate |
| T6.8 | Code review | - | Polish |
| T6.9 | Release notes | docs/release-*.md | Communication |

**Key guardrails for Sprint 6:**
- No new features - polish and documentation only
- All public APIs must be documented
- Demo must work without modification on fresh checkout

---

## Part IV: Quick Reference

### File Creation Checklist

When creating a new file:

```bash
# 1. Create the file with proper header
cat > cortical/reasoning/new_file.py << 'EOF'
"""
Brief description.

Detailed description of what this module does.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
# ... imports
EOF

# 2. Add to __init__.py exports
# Edit cortical/reasoning/__init__.py

# 3. Create test file
cat > tests/unit/test_new_file.py << 'EOF'
"""Tests for new_file module."""

import pytest
from cortical.reasoning.new_file import ...


class TestNewClass:
    """Tests for NewClass."""

    def test_basic(self):
        pass
EOF

# 4. Run tests to verify
python -m pytest tests/unit/test_new_file.py -v
```

### Common Commands

```bash
# Task management
python scripts/got_utils.py task list --status pending
python scripts/got_utils.py task start T-XXXXX
python scripts/got_utils.py task complete T-XXXXX

# Testing
python -m pytest tests/unit/test_loom.py -v
python -m pytest tests/ -q  # Full suite, quiet
python -m pytest tests/ -x  # Stop on first failure

# Coverage
python -m coverage run -m pytest tests/
python -m coverage report --include="cortical/reasoning/*"
python -m coverage html  # Generate HTML report

# Git
git status
git diff cortical/reasoning/loom.py
git add -p  # Interactive staging
```

### Emergency Procedures

**If tests are failing and you can't fix them:**
```bash
# 1. Stash your changes
git stash

# 2. Verify tests pass without your changes
python -m pytest tests/ -q

# 3. Pop your changes back
git stash pop

# 4. Report the issue with details
python scripts/got_utils.py decision log "Blocked: tests fail with message X" \
  --rationale "Attempted fixes Y and Z, need help"
```

**If you've modified the wrong file:**
```bash
# 1. Check what you changed
git diff

# 2. Restore specific file
git checkout -- path/to/wrong/file.py

# 3. Verify
git status
```

---

*This knowledge base is a living document. Update it as you learn.*

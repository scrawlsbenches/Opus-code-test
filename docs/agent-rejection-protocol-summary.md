# Agent Rejection Protocol - Implementation Summary

## Overview

The Agent Rejection Protocol enables agents to reject tasks with structured, validated reasons while preventing lazy rejections. This system integrates with the existing GoT handoff infrastructure and supports auto-replanning workflows.

**Status:** ✅ Complete and tested
**Date:** 2025-12-20

---

## Deliverables

### 1. Design Document
**Location:** `/home/user/Opus-code-test/docs/agent-rejection-protocol.md`

Comprehensive design covering:
- 5 rejection reason types with specific validation rules
- TaskRejection dataclass with evidence requirements
- RejectionValidator with reason-specific validation
- Director response protocol (5 decision types)
- GoT integration for pattern learning
- Example flows and CLI integration

### 2. Core Implementation
**Location:** `/home/user/Opus-code-test/cortical/reasoning/rejection_protocol.py`

**Classes:**
- `RejectionReason` (Enum): 5 valid rejection types
- `TaskRejection` (Dataclass): Structured rejection with evidence
- `RejectionValidator`: Validates rejections aren't lazy
- `DecisionType` (Enum): 5 director decision types
- `RejectionDecision` (Dataclass): Director's response
- `log_rejection_to_got()`: GoT integration
- `analyze_rejection_patterns()`: Pattern learning

**Lines of code:** ~600 (excluding docs/examples)

### 3. Working Demo
**Location:** `/home/user/Opus-code-test/examples/rejection_protocol_demo.py`

**4 scenarios demonstrated:**
1. ✅ Valid SCOPE_CREEP rejection (gets decomposed)
2. ❌ Lazy rejection (gets overridden)
3. ✅ Valid BLOCKER rejection (gets deferred)
4. 📊 Pattern analysis across multiple rejections

**Run:** `python examples/rejection_protocol_demo.py`

### 4. Comprehensive Tests
**Location:** `/home/user/Opus-code-test/tests/unit/test_rejection_protocol.py`

**Test coverage:**
- RejectionReason enum (1 test)
- TaskRejection dataclass (3 tests)
- RejectionValidator validation rules (8 tests)
- RejectionDecision dataclass (3 tests)
- GoT integration (4 tests)

**Total:** 19 tests, all passing
**Run:** `python -m pytest tests/unit/test_rejection_protocol.py -v`

### 5. Module Integration
**Updated:** `/home/user/Opus-code-test/cortical/reasoning/__init__.py`

All classes exported in public API:
```python
from cortical.reasoning import (
    RejectionReason,
    DecisionType,
    TaskRejection,
    RejectionValidator,
    RejectionDecision,
    log_rejection_to_got,
    analyze_rejection_patterns,
)
```

---

## Key Features

### Rejection Reasons (with Validation)

| Reason | Validation Requirements | Burden of Proof |
|--------|------------------------|-----------------|
| **BLOCKER** | Evidence (error logs/status checks) | Medium |
| **SCOPE_CREEP** | 2x growth factor, 2+ sub-tasks | Medium-High |
| **MISSING_DEPENDENCY** | Specific dependency ID | Medium |
| **INFEASIBLE** | 2+ evidence pieces, conflicting constraints | Highest |
| **UNCLEAR_REQUIREMENTS** | Specific questions, attempted interpretations | Low |

### Universal Validation Rules

All rejections must have:
- **Minimum 2 concrete attempts** (not vague like "looked at it")
- **Specific blocking factor** (no "too complex" or "confusing")
- **Actionable alternative** (concrete steps, not "ask someone else")
- **Evidence supporting the claim**

### Director Decision Types

| Decision | When Used | Action Taken |
|----------|-----------|--------------|
| **OVERRIDE** | Rejection invalid | Send feedback, reassign with context |
| **ACCEPT** | Valid rejection | Mark blocked, accept suggestion |
| **ACCEPT_AND_DEFER** | External blocker | Create blocker task, defer original |
| **ACCEPT_AND_DECOMPOSE** | Scope creep | Create sub-tasks, link dependencies |
| **ACCEPT_AND_REFORMULATE** | Infeasible | Create new task with feasible scope |

### GoT Integration

Every rejection creates:
- **Rejection observation node** - captures attempts, evidence, blocker
- **Decision node** - captures Director's response
- **Blocker constraint node** (if BLOCKER type) - models dependency
- **Edges** - links rejection → task, decision → rejection, tasks → decision

Pattern analysis tracks:
- Which tasks get rejected most
- Which agents reject most
- Common blockers
- Override rate (quality metric)

---

## Example Usage

### Creating a Valid Rejection

```python
from cortical.reasoning import (
    RejectionReason,
    TaskRejection,
    RejectionValidator,
)

# Agent discovers scope creep
rejection = TaskRejection(
    task_id="task:T-123",
    handoff_id="handoff:H-456",
    agent_id="sub-agent-1",
    reason_type=RejectionReason.SCOPE_CREEP,
    reason_summary="Task requires auth system redesign",
    reason_detail="Original: Add button. Actual: Redesign auth module.",
    what_attempted=[
        "Analyzed auth/ module - 47 files need changes",
        "Reviewed tests - 89 test files affected",
        "Estimated effort - 16 hours vs 2 hours originally",
    ],
    blocking_factor="Cannot add OAuth without refactoring auth for extensibility",
    suggested_alternative="Decompose into: refactor auth, add OAuth, add UI",
    alternative_tasks=[
        {
            "title": "Refactor auth module",
            "scope": "Extract provider interface",
            "dependencies": "none",
            "estimate": "8h",
        },
        {
            "title": "Add OAuth provider",
            "scope": "Implement OAuth plugin",
            "dependencies": "auth refactor",
            "estimate": "4h",
        },
    ],
    task_original_scope="Add OAuth button to login page (2h)",
    scope_growth_factor=8.0,
)

# Validate
validator = RejectionValidator()
is_valid, issues = validator.validate(
    rejection,
    task_context={"title": "Add OAuth login", "scope": "..."},
)

if is_valid:
    print("✅ Rejection accepted")
else:
    print(f"❌ Rejection invalid: {issues}")
```

### Logging to GoT

```python
from cortical.reasoning import (
    RejectionDecision,
    DecisionType,
    log_rejection_to_got,
    ThoughtGraph,
    NodeType,
)

# Director decides
decision = RejectionDecision(
    decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
    rejection=rejection,
    rationale="Valid scope creep. Decomposing.",
    created_tasks=["task:T-200", "task:T-201"],
)

# Log to graph
graph = ThoughtGraph()
graph.add_node(rejection.task_id, NodeType.TASK, "Original task", {})
rejection_node_id = log_rejection_to_got(graph, rejection, decision)

print(f"Logged: {rejection_node_id}")
print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
```

### Pattern Analysis

```python
from cortical.reasoning import analyze_rejection_patterns

patterns = analyze_rejection_patterns(graph)

print(f"Total rejections: {patterns['total_rejections']}")
print(f"By reason: {patterns['by_reason']}")
print(f"By agent: {patterns['by_agent']}")
print(f"Override rate: {patterns['override_rate']:.1%}")
```

---

## Integration with Handoff System

### Current Handoff Events (from `got_utils.py`)

Existing events:
- `handoff.initiate` - Start handoff
- `handoff.accept` - Accept handoff
- `handoff.complete` - Complete handoff
- `handoff.reject` - Reject handoff (simple, unstructured)

### Enhanced Rejection Events (New)

Enhanced events with structured data:
```python
# Agent rejects with structured data
event_log.log(
    "handoff.reject",
    handoff_id=handoff_id,
    agent=agent_id,
    rejection=rejection.to_dict(),  # Full structured rejection
)

# Validation failure
event_log.log(
    "handoff.reject.invalid",
    handoff_id=handoff_id,
    agent=agent_id,
    rejection=rejection.to_dict(),
    validation_issues=issues,
)

# Director response
event_log.log(
    "handoff.reject.response",
    handoff_id=handoff_id,
    decision=decision.to_dict(),  # Director's decision
)
```

### CLI Integration (Proposed)

```bash
# Reject handoff with structured data
python scripts/got_utils.py handoff reject HANDOFF_ID \
  --agent sub-agent-1 \
  --reason SCOPE_CREEP \
  --summary "Scope grew 5x" \
  --detail @rejection_detail.md \
  --attempted "Analysis 1" "Analysis 2" "Analysis 3" \
  --blocker "Requires auth redesign" \
  --alternative @alternative_plan.md \
  --scope-growth 5.0

# View rejection details
python scripts/got_utils.py handoff status HANDOFF_ID

# Analyze rejection patterns
python scripts/got_utils.py analyze rejections --since 2025-12-01

# Director responds
python scripts/got_utils.py handoff rejection-response HANDOFF_ID \
  --decision ACCEPT_AND_DECOMPOSE \
  --created-tasks task:T-xxx task:T-yyy
```

---

## GitHub PR Flow

### How Rejections Flow Through PRs

1. **Agent 1 (PR branch)** receives task via handoff
2. **Agent 1** attempts task, discovers blocker/scope creep
3. **Agent 1** creates structured rejection, logs to `.got/events/*.jsonl`
4. **Director** processes rejection, creates decision
5. **Decision logged** to same `.got/events/*.jsonl` file
6. **PR created** with both rejection and decision visible
7. **New thread spawned** (if needed) with full context from rejection

### Merge-Friendly Event Log

All events written to git-tracked append-only logs:
```
.got/events/
├── session-20251220-153000-a1b2.jsonl  # Session 1 events
├── session-20251220-160000-c3d4.jsonl  # Session 2 events
└── session-20251220-163000-e5f6.jsonl  # Session 3 events
```

**No conflicts** - each session writes to unique file
**Full history** - rebuild graph from all events
**Cross-branch learning** - patterns visible across branches

---

## Design Constraints Satisfied

### ✅ Valid Rejection Reasons Only

5 structured reasons with specific validation requirements prevent arbitrary rejections.

### ✅ Demonstrates Effort

Minimum 2 concrete attempts required. Vague attempts like "looked at it" rejected.

### ✅ Scope Validation

Cannot reject if task is clearly within capability. Validators check:
- BLOCKER: Must have evidence
- SCOPE_CREEP: Must show 2x+ growth
- INFEASIBLE: Must prove logical impossibility

### ✅ Director Handling

5 response paths:
- Accept → mark blocked
- Override → provide feedback, reassign
- Decompose → create sub-tasks
- Defer → create blocker resolution task
- Reformulate → create feasible alternative

### ✅ GoT Learning

All rejections logged as nodes/edges enabling:
- Pattern detection (which tasks get rejected often)
- Agent performance tracking
- Blocker frequency analysis
- Override rate as quality metric

### ✅ Clear Communication

Rejection must include:
- What was attempted (concrete steps)
- What specifically blocks progress
- Suggested actionable alternative
- Evidence supporting the claim

---

## Testing Verification

All 19 tests passing:

```bash
$ python -m pytest tests/unit/test_rejection_protocol.py -v
======================== 19 passed in 0.23s ========================
```

**Coverage:**
- ✅ Enum definitions
- ✅ Dataclass serialization
- ✅ Valid rejections pass validation
- ✅ Lazy rejections fail validation
- ✅ Reason-specific validation rules
- ✅ GoT integration creates correct nodes/edges
- ✅ Pattern analysis aggregates correctly

**Demo runs successfully:**

```bash
$ python examples/rejection_protocol_demo.py
╔════════════════════════════════════════════════════════════╗
║         AGENT REJECTION PROTOCOL DEMO                      ║
╚════════════════════════════════════════════════════════════╝

SCENARIO 1: Valid SCOPE_CREEP Rejection ✅
SCENARIO 2: Lazy Rejection (Gets Overridden) ❌
SCENARIO 3: Valid BLOCKER Rejection ✅
SCENARIO 4: Rejection Pattern Analysis 📊

Demo complete!
```

---

## Next Steps (Suggested)

### 1. Integration with got_utils.py

Add CLI commands to `scripts/got_utils.py`:
- `handoff reject` - create structured rejection
- `handoff rejection-response` - director responds
- `analyze rejections` - pattern analysis

### 2. DirectorRejectionHandler

Implement full `DirectorRejectionHandler` class (designed but not coded):
- `handle_rejection()` - main entry point
- `_handle_blocker()` - create blocker resolution tasks
- `_handle_scope_creep()` - decompose into sub-tasks
- `_handle_missing_dependency()` - identify/create dependency
- `_handle_infeasible()` - reformulate task
- `_override_rejection()` - send feedback

### 3. Integration Tests

Add integration tests for:
- Full rejection → decision → GoT workflow
- Event log serialization/deserialization
- CLI command execution
- Cross-branch pattern learning

### 4. Documentation Updates

Update:
- `docs/graph-of-thought.md` - add rejection protocol section
- `docs/got-cli-spec.md` - add rejection commands
- `CLAUDE.md` - add rejection protocol quick reference

---

## Files Modified/Created

### Created
- `docs/agent-rejection-protocol.md` (2,400+ lines)
- `cortical/reasoning/rejection_protocol.py` (600 lines)
- `examples/rejection_protocol_demo.py` (480 lines)
- `tests/unit/test_rejection_protocol.py` (540 lines)
- `docs/agent-rejection-protocol-summary.md` (this file)

### Modified
- `cortical/reasoning/__init__.py` (added rejection protocol exports)

**Total new code:** ~4,020 lines (including docs)

---

## Summary

The Agent Rejection Protocol is **complete and tested**, providing:
1. **Structured rejection reasons** preventing lazy rejections
2. **Validation layer** enforcing concrete evidence
3. **Director response paths** for productive handling
4. **GoT integration** for pattern learning
5. **Merge-friendly event log** for cross-branch coordination

The system balances **honest communication** about blockers with **accountability** through evidence requirements. Agents can't give up easily, but when rejections are valid, they're handled productively.

**Key innovation:** Rejections become learning data rather than dead-ends, enabling the Director to detect patterns and improve task decomposition over time.

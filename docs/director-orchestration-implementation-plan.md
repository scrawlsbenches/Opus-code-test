# Director Orchestration Enhancement - Implementation Plan

**Task ID:** T-20251214-174530-6aa8-012
**Branch:** claude/director-orchestration-improvements-H42Ys
**Created:** 2025-12-15
**Coverage Baseline:** 91% (3767 tests)

---

## Executive Summary

This plan implements execution tracking, verification automation, metrics collection, and task system integration for the director orchestration system.

---

## Implementation Phases

### Phase 1: Foundation (Core Classes)

**Goal:** Create the core orchestration utilities module with plan and execution tracking.

**Files to Create:**
- `scripts/orchestration_utils.py` - Core orchestration classes
- `tests/unit/test_orchestration_utils.py` - Unit tests

**Classes to Implement:**

```python
class OrchestrationPlan:
    """Represents a decomposed, batched execution plan."""
    - plan_id: str (OP-YYYYMMDD-HHMMSS-XXXX format)
    - title: str
    - goal: dict with summary and success_criteria
    - batches: List[Batch]
    - save() / load() methods

class Batch:
    """A group of agents to execute together."""
    - batch_id: str (B1, B2, etc.)
    - name: str
    - batch_type: str (parallel | sequential)
    - agents: List[Agent]
    - depends_on: List[str]
    - status: str

class Agent:
    """Single agent task within a batch."""
    - agent_id: str
    - task_type: str (research | implement | test | verify)
    - description: str
    - scope: dict (files_read, files_write, constraints)
    - status: str
    - result: Optional[AgentResult]

class ExecutionTracker:
    """Tracks live execution state."""
    - start_batch()
    - record_agent_result()
    - complete_batch()
    - record_replan()
```

**Acceptance Criteria:**
- [ ] All classes implemented with type hints
- [ ] Save/load with atomic writes (temp → rename pattern)
- [ ] Merge-friendly IDs using timestamp+session pattern
- [ ] Unit tests with >90% coverage of new code
- [ ] Tests pass in <5 seconds

---

### Phase 2: Verification Automation

**Goal:** Create automated batch verification script.

**Files to Create:**
- `scripts/verify_batch.py` - Verification script with CLI

**Functions to Implement:**

```python
def verify_tests(quick: bool = False) -> VerificationResult:
    """Run tests and return pass/fail with details."""

def verify_no_conflicts(modified_files: Dict[str, List[str]]) -> VerificationResult:
    """Check no files were modified by multiple agents."""

def verify_git_status() -> VerificationResult:
    """Check git status for problems."""

def run_verification(quick: bool = False) -> BatchVerificationReport:
    """Run all verification checks and return comprehensive report."""
```

**CLI Interface:**
```bash
python scripts/verify_batch.py                    # Full verification
python scripts/verify_batch.py --quick            # Smoke tests only
python scripts/verify_batch.py --check conflicts  # Only check file conflicts
python scripts/verify_batch.py --json             # JSON output for parsing
```

**Acceptance Criteria:**
- [ ] Runs smoke tests by default (--quick) or full tests
- [ ] Detects file conflicts from agent results
- [ ] Reports git status issues
- [ ] Returns structured results for programmatic use
- [ ] Unit tests for each verification function

---

### Phase 3: Metrics & Task Integration

**Goal:** Add metrics collection and integrate with task system.

**Files to Modify:**
- `scripts/orchestration_utils.py` - Add OrchestrationMetrics class
- `scripts/task_utils.py` - Add orchestration integration functions

**New Classes/Functions:**

```python
class OrchestrationMetrics:
    """Collects and aggregates orchestration metrics."""
    - record(event_type, **kwargs) - Append to JSONL
    - get_summary() - Aggregate statistics
    - get_failure_patterns() - Analyze common failures

# In task_utils.py:
def create_orchestration_tasks(plan, session) -> List[Task]:
    """Auto-create tasks for each batch in an orchestration plan."""

def link_plan_to_task(plan_id, task_id) -> None:
    """Link an orchestration plan to an existing task."""
```

**Storage:**
- `.claude/orchestration/metrics.jsonl` - Append-only metrics log

**Acceptance Criteria:**
- [ ] Metrics recorded in append-only JSONL format
- [ ] Summary statistics aggregated correctly
- [ ] Tasks auto-created with proper dependencies
- [ ] Unit tests for metrics and task integration

---

### Phase 4: Documentation & Polish

**Goal:** Update documentation and ensure everything works end-to-end.

**Files to Update:**
- `.claude/commands/director.md` - Add tracking commands reference
- `docs/parallel-agent-orchestration.md` - Add metrics section
- `CLAUDE.md` - Add quick reference entries

**End-to-End Testing:**
- Create sample orchestration plan
- Execute verification script
- Verify metrics recorded
- Verify task linking works

**Acceptance Criteria:**
- [ ] Director command updated with new CLI references
- [ ] Documentation includes usage examples
- [ ] End-to-end flow verified
- [ ] All tests pass
- [ ] Coverage maintained at >90%

---

## Directory Structure

```
.claude/orchestration/
├── plans/                          # Orchestration plan files
│   └── {timestamp}_{session}.json  # Individual plan instances
├── executions/                     # Execution state tracking
│   └── {plan_id}_execution.json    # Linked to specific plan
└── metrics.jsonl                   # Append-only metrics log

scripts/
├── orchestration_utils.py          # NEW: Core orchestration utilities
└── verify_batch.py                 # NEW: Batch verification script

tests/unit/
└── test_orchestration_utils.py     # NEW: Unit tests
```

---

## Risk Mitigation & Backup Plans

### Risk 1: Schema Changes Break Existing Data
**Mitigation:** Version field in all schemas, migration support if needed
**Backup:** Schemas are new, no existing data to break

### Risk 2: Tests Fail After Implementation
**Mitigation:** Run tests after each phase, fix before proceeding
**Backup:** Revert to last working commit, investigate separately

### Risk 3: Coverage Drops Below 90%
**Mitigation:** Write unit tests alongside implementation
**Backup:** Add focused tests for uncovered lines before finalizing

### Risk 4: Sub-agent Produces Incorrect Code
**Mitigation:** Review all sub-agent output before committing
**Backup:** Fix issues directly, document patterns for future

### Risk 5: Git Conflicts on Push
**Mitigation:** Pull before each commit, use merge-friendly patterns
**Backup:** Rebase on latest, resolve conflicts manually

---

## Execution Order

1. **Phase 1 First** - Foundation must exist before other phases
2. **Phase 2 Second** - Verification can be developed independently
3. **Phase 3 Third** - Requires Phase 1 classes to exist
4. **Phase 4 Last** - Documentation and polish after features work

**Parallel Opportunities:**
- Phase 2 can be developed in parallel with Phase 3 after Phase 1 completes
- Unit tests can be written alongside implementation

---

## Definition of Done Checklist

### Code Complete
- [ ] All classes implemented with type hints
- [ ] Unit tests written and passing
- [ ] Full test suite passing (3767+ tests)
- [ ] Coverage maintained at >90%

### Documentation Complete
- [ ] All public functions have Google-style docstrings
- [ ] CLAUDE.md quick reference updated
- [ ] Director command docs updated

### Verification Complete
- [ ] End-to-end flow tested
- [ ] Edge cases tested (empty plan, failed agents)
- [ ] Limitations documented

### Issue Tracking Complete
- [ ] Any discovered issues added to tasks
- [ ] Task linked to orchestration plan

### Truly Done
- [ ] All changes committed with descriptive messages
- [ ] Changes pushed to branch
- [ ] Ready for review

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Test Coverage | >90% |
| New Tests | 20+ |
| New Lines of Code | ~800-1000 |
| Execution Time | <5s for unit tests |
| Documentation | Complete |

---

*Plan created: 2025-12-15*
*Parent Task: T-20251214-174530-6aa8-012*
